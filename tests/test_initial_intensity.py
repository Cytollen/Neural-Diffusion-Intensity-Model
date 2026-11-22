"""The initial intensity ``Z_0`` of the model (``model.z0``): the truth's for synthetic data, a number, or an estimate
from the counts of the first part of the window for observations of an unspecified model (``data.source: file``)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ndim.config import MCMCConfig, load_config, model_z0, z0_mode
from ndim.data import estimate_initial_intensity, load_or_generate_dataset, make_truth_diffusion, make_truth_drift, resolve_z0
from ndim.elbo import compute_objective
from ndim.mcmc import sample_posterior_mcmc_batch
from ndim.metrics import expected_path_deviation
from ndim.model import NDIM, build_model
from ndim.sde import brownian_increments
from ndim.train import train

CPU = torch.device("cpu")
INF = float("inf")


def test_estimate_is_the_mean_event_rate_over_the_first_part_of_the_window() -> None:
    events = torch.tensor([[0.1, 0.4, 0.5, 0.6, 2.0], [0.2, 3.0, INF, INF, INF]])
    assert estimate_initial_intensity(events, T=4.0) == pytest.approx((3 + 1) / 2 / 0.5)          # the first 1/8: (0, 0.5]
    assert estimate_initial_intensity(events, T=4.0, window=0.25) == pytest.approx((4 + 1) / 2 / 1.0)
    assert estimate_initial_intensity(events, T=4.0, window=1.0) == pytest.approx((5 + 2) / 2 / 4.0)
    assert estimate_initial_intensity(torch.full((3, 2), INF), T=4.0) == 0.0                      # no events at all
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            estimate_initial_intensity(events, T=4.0, window=bad)
    with pytest.raises(ValueError):
        estimate_initial_intensity(events[:0], T=4.0)


@pytest.mark.parametrize("name", ["base", "intensity/cir_mlp"])
def test_synthetic_examples_start_at_the_truths_initial_intensity(name: str) -> None:
    cfg = load_config(name)
    assert cfg.model.z0 == "auto" and cfg.data.source == "synthetic" and z0_mode(cfg) == "truth"
    assert model_z0(cfg) == cfg.data.truth.z0 == 5.0
    assert cfg.model.z0_window == 0.125                                           # the first eighth of [0, T]


def test_a_number_sets_the_initial_intensity_of_the_model_only(make_cfg, smoke_dataset) -> None:
    cfg = make_cfg("model.state_space=intensity", "model.z0=7.5")
    assert z0_mode(cfg) == "value" and model_z0(cfg) == 7.5 and cfg.data.truth.z0 == 5.0
    model = build_model(cfg)
    assert model.initial_state == 7.5
    events = smoke_dataset.train.events[:2]
    W = brownian_increments(2 * 2, 25, cfg.dt, CPU, torch.Generator().manual_seed(0))
    out = compute_objective(model, events, 25, 2, W, cfg)
    assert torch.all(out.sim.Z[:, 0] == 7.5)                                      # the posterior paths start there
    _, paths = expected_path_deviation(model.drift, make_truth_drift(cfg.data.truth), model.diffusion, cfg, 4, CPU,
                                       torch.Generator().manual_seed(0), return_paths=True)
    assert torch.all(paths["learned"][:, 0] == 7.5) and torch.all(paths["true"][:, 0] == 5.0)   # the truth keeps its own
    latent = build_model(make_cfg("model.z0=7.5"))                                # default model: X = log Z
    assert latent.initial_state == pytest.approx(math.log(7.5))
    # MCMC under the true prior starts at the truth's value unless told otherwise
    truth = cfg.data.truth
    for z0, expected in ((None, 5.0), (model_z0(cfg), 7.5)):
        mc, _ = sample_posterior_mcmc_batch(make_truth_drift(truth), make_truth_diffusion(truth), events, cfg, 25,
                                            MCMCConfig(n_chains=3, n_steps=2), CPU, torch.Generator().manual_seed(0), z0=z0)
        assert torch.all(mc[:, :, 0] == expected)


@pytest.mark.parametrize("bad", ["model.z0=nearest", "model.z0=-1.0", "model.z0=true", "model.z0_window=0.0", "model.z0_window=1.5",
                                  "data.source=web", "data.source=file"])
def test_validation(make_cfg, bad: str) -> None:
    with pytest.raises(ValueError):
        make_cfg(bad)


def test_latent_state_needs_a_positive_initial_intensity(make_cfg) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        make_cfg("model.z0=0.0")                                                  # X = log Z has no value at Z = 0
    assert model_z0(make_cfg("model.state_space=intensity", "model.z0=0.0")) == 0.0


def test_estimate_is_resolved_by_training_and_saved_with_the_run(make_cfg, smoke_dataset) -> None:
    cfg = make_cfg("name=smoke_pytest_z0_estimate", "model.z0=estimate", "train.epochs=1", "train.eval_paths=1")
    assert z0_mode(cfg) == "estimate"
    with pytest.raises(ValueError, match="resolve_z0"):
        model_z0(cfg)                                                             # not known before the data are seen
    expected = estimate_initial_intensity(smoke_dataset.train.events, cfg.data.T, 0.125)
    assert expected > 0
    res = train(cfg)
    assert cfg.model.z0 == pytest.approx(expected) and model_z0(cfg) == pytest.approx(expected)
    saved = yaml.safe_load((res.output_dir / "config.yaml").read_text())
    assert saved["model"]["z0"] == pytest.approx(expected) and saved["model"]["z0_window"] == 0.125
    loaded, _ = NDIM.load_checkpoint(res.output_dir / "ckpt_last.pt")             # the checkpoint carries the number
    assert loaded.initial_state == pytest.approx(math.log(expected)) and "initial intensity Z_0" in (res.output_dir / "train.log").read_text()
    wider = make_cfg("model.z0=estimate", "model.z0_window=0.5")
    assert resolve_z0(wider, smoke_dataset.train) == pytest.approx(estimate_initial_intensity(smoke_dataset.train.events, cfg.data.T, 0.5))
    assert resolve_z0(make_cfg(), smoke_dataset.train) == 5.0                      # synthetic data: the truth's value, nothing estimated


def _event_file(path: Path, dataset, **extra) -> Path:
    train, test = dataset.train.events.numpy().copy(), dataset.test.events.numpy().copy()
    train[~np.isfinite(train)] = np.nan                                            # nan padding is accepted as well as inf
    np.savez(path, train_events=train, test_events=test, **extra)
    return path


def test_observations_of_an_unspecified_model_from_a_file(make_cfg, smoke_dataset, tmp_root: Path) -> None:
    from ndim.evaluate import evaluate_run

    path = _event_file(tmp_root / "observed_events.npz", smoke_dataset, T=np.array(4.0))
    file_args = ["data.source=file", f"data.path={json.dumps(str(path))}"]
    cfg = make_cfg("name=smoke_pytest_file_data", *file_args, "train.epochs=1", "train.eval_paths=1", "train.max_abs_control=30",
                   "eval.n_initial=200")
    assert z0_mode(cfg) == "estimate" and cfg.burn_steps == 6                     # no truth: a burn-in, a quarter of the window
    ds, source = load_or_generate_dataset(cfg)
    assert source == path and not ds.has_truth and ds.train_paths is None and ds.test_paths is None
    assert torch.equal(ds.train.counts, smoke_dataset.train.counts) and torch.equal(ds.test.events, smoke_dataset.test.events[:, : ds.test.K_max])
    with pytest.raises(ValueError):
        ds.save(tmp_root / "not_cached.npz")

    res = train(cfg)
    expected = estimate_initial_intensity(smoke_dataset.train.events, 4.0, 0.125)
    assert cfg.model.x_start == pytest.approx(math.log(expected)) and not res.dataset.has_truth      # the paths start at the data's rate
    saved = yaml.safe_load((res.output_dir / "config.yaml").read_text())
    assert saved["model"]["burn_in"] == "auto" and saved["model"]["x_start"] == pytest.approx(math.log(expected))
    assert "burn-in: the diffusion of the model starts at t = -0.96" in (res.output_dir / "train.log").read_text()
    assert math.isfinite(res.history[-1]["test_elbo"]) and (res.output_dir / "drift_snapshot.png").stat().st_size > 0

    metrics = evaluate_run(res.output_dir, ckpt="last")
    assert metrics["errors"] == [], metrics["errors"]
    assert metrics["truth_available"] is False
    assert metrics["stages_skipped_without_truth"] == ["l2", "wasserstein"]
    assert set(metrics["stages"]) == {"curves", "drift", "samples", "initial", "posterior"}
    learned = metrics["learned_model"]                                            # the learned prior against the observations
    assert len(learned["rate_observed"]) == len(learned["rate_learned_prior"]) == len(learned["bin_edges"]) - 1 == 25
    assert learned["bin_edges"][0] == 0.0 and learned["bin_edges"][-1] == pytest.approx(4.0)
    mean_count = float(smoke_dataset.train.counts.float().mean())
    assert sum(r * 0.16 for r in learned["rate_observed"]) == pytest.approx(mean_count, rel=1e-6)      # the bins hold every event
    assert learned["events_per_observation"]["observed_mean"] == pytest.approx(mean_count, rel=1e-6)
    assert learned["events_per_observation"]["learned_mean"] > 0 and all(v > 0 for v in learned["count_sd_learned_model"])
    assert "drift_error_on_occupied_states" not in metrics and "parameters" not in metrics           # nothing is compared with a truth
    assert metrics["posterior_reference"]["kind"] == "mcmc_learned_prior" and metrics["posterior_reference"]["uses_learned_parameters"]
    assert metrics["samples"]["true_path_mean_final"] is None and metrics["samples"]["count_mean_generated"] > 0
    initial = metrics["initial"]                                                  # the learned distribution of Z_0, on its own
    assert initial["burn_in"] == pytest.approx(0.96) and initial["x_start"] == pytest.approx(math.log(expected))
    assert initial["learned"]["std"] > 0 and initial["data_event_rate_at_start"] == pytest.approx(expected) and "true" not in initial
    case = metrics["posterior"]["cases"]["most_events_full"]                      # against MCMC under the learned prior
    assert case["w2_vi_vs_reference"] >= 0 and case["w2_prior_vs_reference"] >= 0 and "true_z0" not in case
    assert case["w2_between_two_reference_runs"] > 0                              # the scale of the sampling error
    assert case["z0_vi"]["std"] > 0 and case["z0_reference"]["std"] > 0
    for name in ("learned_model.png", "generated_vs_true_counts.png", "initial_distribution.png", "posterior_most_events_full.png",
                 "posterior_fewest_events_partial.png"):
        assert (res.output_dir / "eval" / name).stat().st_size > 0, name

    # without the burn-in: a fixed Z_0, estimated from the counts
    fixed = make_cfg("name=smoke_pytest_file_data_fixed", *file_args, "model.burn_in=0", "train.epochs=1", "train.eval_paths=1")
    assert fixed.burn_steps == 0
    res_fixed = train(fixed)
    assert model_z0(fixed) == pytest.approx(expected)
    m = evaluate_run(res_fixed.output_dir, ckpt="last", which=["initial", "posterior"])
    assert m["errors"] == [] and m["stages"]["initial"]["status"] == "skipped"
    assert m["posterior"]["cases"]["most_events_full"]["z0_reference"]["std"] == pytest.approx(0.0, abs=1e-5)   # every path starts at Z_0

    with pytest.raises(ValueError, match="truth"):
        make_cfg(*file_args, "model.z0=truth")                                    # there is no truth to take it from
    assert model_z0(make_cfg(*file_args, "model.z0=3.0")) == 3.0                  # a number is still allowed


def test_event_file_errors(make_cfg, smoke_dataset, tmp_root: Path) -> None:
    def cfg_for(path: Path, *extra: str):
        return make_cfg("data.source=file", f"data.path={json.dumps(str(path))}", *extra)

    with pytest.raises(FileNotFoundError):
        load_or_generate_dataset(cfg_for(tmp_root / "no_such_file.npz"))
    only_train = tmp_root / "only_train.npz"
    np.savez(only_train, train_events=smoke_dataset.train.events.numpy())
    with pytest.raises(ValueError, match="test_events"):
        load_or_generate_dataset(cfg_for(only_train))
    good = _event_file(tmp_root / "events_T4.npz", smoke_dataset, T=np.array(4.0))
    with pytest.raises(ValueError, match="recorded"):
        load_or_generate_dataset(cfg_for(good, "data.T=2.0"))                     # the file says T = 4
    no_T = _event_file(tmp_root / "events_no_T.npz", smoke_dataset)
    with pytest.raises(ValueError, match="outside"):
        load_or_generate_dataset(cfg_for(no_T, "data.T=2.0"))                     # events after the end of the window
