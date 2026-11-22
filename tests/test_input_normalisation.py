"""Every network input is of order one, in every state space, link and score space the code offers.

The inputs are captured at the first layer of each MLP while the networks are evaluated along the true latent
paths of the training observations, i.e. on the states the posterior paths settle on once the model is trained.
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from ndim.config import (
    DriftConfig,
    ExperimentConfig,
    load_config,
    off_scale_inputs,
    typical_input_sizes,
)
from ndim.data import load_or_generate_dataset
from ndim.model import build_model
from ndim.models.common import MLP
from ndim.models.drift import MLPDrift
from ndim.train import train

#: the experiments of the repository
SHIPPED = ["intensity/cir_mlp", "state_space/cir_burn_in"]
#: settings offered on top of the default model; each one changes the coordinates some network works in
VARIANTS = [
    ["model.state_space=intensity"],
    ["model.link=softplus"],
    ["model.posterior.score_space=intensity"],
    ["model.diffusion.kind=mlp"],
    ["model.state_space=intensity", "model.diffusion.kind=mlp"],
    ["model.posterior.kind=remaining_count"],
    ["model.posterior.phi_uses_state=false"],             # the faster form: phi runs once, inside prepare, on the events only
]
#: mean number of events of an observation in the paper's data (Sec. 4.1)
MEAN_COUNT = 145.0


def _first_layer_inputs(cfg: ExperimentConfig) -> Dict[str, torch.Tensor]:
    """``{network: [rows, n_inputs]}`` for every MLP of the model; for the Deep Sets ``phi``, future events only."""
    ds, _ = load_or_generate_dataset(cfg)
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    net = model.score_net
    paths = ds.train_paths                                                   # true intensity, [n_train, N + 1]
    state = model.link.inverse(paths.clamp_min(1e-6)) if model.real_states else paths
    captured: Dict[str, List[torch.Tensor]] = {}
    gate = {"value": None}

    def hook(name: str):
        def fn(module, inputs):
            x = inputs[0].detach()
            if name == "score_net.phi" and gate["value"] is not None:
                x = x[gate["value"] > 0]
            captured.setdefault(name, []).append(x.reshape(-1, x.shape[-1]))
        return fn

    for name, module in model.named_modules():
        if isinstance(module, MLP):
            module.net[0].register_forward_pre_hook(hook(name))
    B = paths.shape[0]
    with torch.no_grad():
        for frac in (1.0, 0.5):
            n_prime = cfg.horizon_steps(frac)
            t_grid = torch.arange(n_prime, dtype=torch.float32) * cfg.dt
            gate["value"] = None
            if getattr(net, "phi_uses_state", True) is False:
                # phi without the state runs once, inside prepare, on every (observation, grid time, event): keep the future events
                gate["value"] = net.event_features(ds.train.events, t_grid, n_prime * cfg.dt)[1]
            ctx = net.prepare(ds.train.events, t_grid, n_prime * cfg.dt)
            for k in range(n_prime):
                t = torch.full((B,), k * cfg.dt)
                gate["value"] = None
                model.drift(state[:, k], t)
                model.diffusion(state[:, k], t)
                inp = net.row_inputs(ctx, k, 1)
                gate["value"] = inp.get("gate")
                net(state[:, k], inp)
    return {name: torch.cat(chunks) for name, chunks in captured.items()}


def _assert_order_one(inputs: Dict[str, torch.Tensor], expect: List[str]) -> None:
    assert sorted(inputs) == sorted(expect)
    for name, x in inputs.items():
        # rho: state, t / T, T' / T, remaining count; what follows is the pooled embedding, an output of phi
        n_inputs = 4 if name == "score_net.rho" else x.shape[1]
        for j in range(n_inputs):
            size = x[:, j].abs().double()
            median, high = float(size.median()), float(torch.quantile(size, 0.95))
            assert 0.05 < median < 3.0 and high < 5.0, f"{name} input {j}: median {median:.3g}, 95% quantile {high:.3g}"


@pytest.mark.parametrize("name", SHIPPED)
def test_inputs_of_the_shipped_experiments_are_of_order_one(name: str, base_overrides: List[str]) -> None:
    cfg = load_config(name, profile="tiny", overrides=base_overrides)
    networks = ["score_net.phi", "score_net.rho"] + (["drift.net"] if cfg.model.drift.kind == "mlp" else [])
    _assert_order_one(_first_layer_inputs(cfg), networks)


@pytest.mark.parametrize("overrides", VARIANTS, ids=lambda o: ",".join(s.split(".", 1)[1] for s in o))
def test_inputs_are_of_order_one_in_every_coordinate_system(overrides: List[str], make_cfg) -> None:
    """No setting has to be accompanied by a scale: the state scale follows the coordinates of each network."""
    cfg = make_cfg(*overrides)
    networks = ["drift.net", "score_net.rho"]
    networks += ["score_net.phi"] if cfg.model.posterior.kind == "deepset" else []
    networks += ["diffusion.net"] if cfg.model.diffusion.kind == "mlp" else []
    _assert_order_one(_first_layer_inputs(cfg), networks)


def test_a_model_built_without_a_config_file_gets_its_scales() -> None:
    cfg = ExperimentConfig()
    assert cfg.model.drift.z_scale is None and cfg.model.posterior.z_scale is None      # nothing chosen yet
    model = build_model(cfg)
    assert model.drift.z_scale == pytest.approx(4.382, abs=1e-3)                        # log 80 for X = log Z
    assert cfg.model.posterior.z_scale == model.score_net.z_scale == model.drift.z_scale
    with pytest.raises(ValueError, match="not set"):                                    # a sub-config built by hand
        MLPDrift(DriftConfig(), T=4.0)


@pytest.mark.parametrize("name", SHIPPED)
def test_scales_of_the_shipped_experiments_fit_the_data(name: str) -> None:
    cfg = load_config(name)
    sizes = typical_input_sizes(cfg, MEAN_COUNT)
    assert all(0.3 < size < 2.0 for size in sizes.values()), sizes
    assert off_scale_inputs(cfg, sizes) == []


def test_a_scale_that_does_not_fit_the_data_is_reported(make_cfg) -> None:
    # the intensity is about 36 events per unit time: divided by 4 the state input is about 9
    cfg = make_cfg("model.state_space=intensity", "model.intensity_scale=4")
    sizes = typical_input_sizes(cfg, MEAN_COUNT)
    assert sizes["drift state / z_scale"] == pytest.approx(MEAN_COUNT / 4.0 / 4.0)
    assert off_scale_inputs(cfg, sizes) == ["drift state / z_scale", "posterior state / z_scale"]
    # too small is reported as well: the gap divided by T
    cfg = make_cfg("model.posterior.gap_scale=4.0")
    assert off_scale_inputs(cfg, typical_input_sizes(cfg, MEAN_COUNT)) == ["gap / gap_scale"]
    # a latent state near zero (X = log Z at one event per unit time) is not a scaling problem
    cfg = make_cfg("model.intensity_scale=1.0", "model.posterior.count_scale=4.0")
    sizes = typical_input_sizes(cfg, 4.0)
    assert sizes["drift state / z_scale"] == pytest.approx(0.0, abs=1e-9)
    assert off_scale_inputs(cfg, sizes) == []


def test_training_logs_the_input_sizes_and_warns_when_a_scale_is_off(trained_smoke_run, make_cfg) -> None:
    log = (trained_smoke_run.output_dir / "train.log").read_text()
    assert "network inputs at the mean event rate of the training observations" in log
    assert "far from one" not in log
    res = train(make_cfg("name=smoke_pytest_off_scale", "train.epochs=1", "model.state_space=intensity",
                         "model.intensity_scale=4"))
    log = (res.output_dir / "train.log").read_text()
    assert "WARNING" in log and "posterior state / z_scale is" in log and "far from one" in log
