"""The warm start for a fixed diffusion coefficient that an untrained model cannot start with (``ndim.warm_start``).

With the bound on the CIR diffusion of a latent state (``model.diffusion.max_step_std``, on by default) an untrained
model can start at a large coefficient, so the cases that need the warm start switch the bound off (``UNBOUNDED``)."""
from __future__ import annotations

import json

import pytest
import torch

from ndim import warm_start as ws
from ndim.train import train

CPU = torch.device("cpu")
UNBOUNDED = "model.diffusion.max_step_std=null"


def _rows(history):
    return [{k: v for k, v in h.items() if k != "time_s"} for h in history]


def test_nothing_is_done_when_the_model_can_start(make_cfg, smoke_dataset) -> None:
    """``train.warm_start: auto`` probes the untrained model and leaves a run that can start exactly as it is."""
    plain = train(make_cfg("name=warm_start_off"), dataset=smoke_dataset)
    cfg = make_cfg("name=warm_start_not_needed", "train.warm_start=auto")
    assert ws.finite_share(ws._model_at(cfg, 1.0, CPU), smoke_dataset, cfg, CPU) >= ws.MIN_FINITE
    probed = train(cfg, dataset=smoke_dataset)
    assert "warm_start" not in probed.extras and not (probed.output_dir / "warm_start.json").exists()
    assert _rows(probed.history) == _rows(plain.history)                   # the probe does not touch the random state of the run
    with pytest.raises(ValueError, match="warm_start"):
        make_cfg("train.warm_start=always")


def test_a_coefficient_too_large_to_start_with_is_reached_in_stages(make_cfg, smoke_dataset) -> None:
    cfg = make_cfg("name=warm_start_needed", "model.diffusion.eta_init=2.0", UNBOUNDED, "train.warm_start=auto")
    share = ws.finite_share(ws._model_at(cfg, 2.0, CPU), smoke_dataset, cfg, CPU)
    assert share < ws.MIN_FINITE                                            # the untrained model runs off at this coefficient
    bounded = make_cfg("model.diffusion.eta_init=2.0", "train.warm_start=auto")
    assert ws.finite_share(ws._model_at(bounded, 2.0, CPU), smoke_dataset, bounded, CPU) == 1.0   # with the bound it starts as it is
    assert ws.finite_share(ws._model_at(cfg, 2.0, CPU), smoke_dataset, cfg, CPU) == share      # the probe is deterministic
    result = train(cfg, dataset=smoke_dataset)
    warm = result.extras["warm_start"]
    etas = [s["eta"] for s in warm["stages"]]
    assert warm["target"] == 2.0 and etas[0] < 2.0 and etas[-1] == pytest.approx(2.0) and etas == sorted(etas)
    assert warm["stages"][0]["epochs"] == ws.START_EPOCHS and all(s["epochs"] == ws.STEP_EPOCHS for s in warm["stages"][1:])
    assert all(b <= a * ws.FACTORS[0] * (1 + 1e-9) for a, b in zip(etas, etas[1:]))            # no raise larger than the largest factor
    assert "state" not in warm and warm["epochs"] == sum(s["epochs"] for s in warm["stages"])
    # the run proper: its own epochs, at the coefficient of the config, from the networks of the last stage
    assert [h["epoch"] for h in result.history] == [1, 2] and float(result.model.diffusion.eta) == pytest.approx(2.0)
    saved = json.loads((result.output_dir / "warm_start.json").read_text())
    assert [s["eta"] for s in saved["stages"]] == etas and saved["target"] == 2.0
    summary = json.loads((result.output_dir / "train_summary.json").read_text())
    assert summary["epochs"] == 2 and summary["warm_start_epochs"] == warm["epochs"]
    stage_dirs = sorted(p.name for p in (result.output_dir / "warm_start").iterdir())
    assert len(stage_dirs) == len(etas) and all((result.output_dir / "warm_start" / d / "ckpt_last.pt").exists() for d in stage_dirs)
    log = (result.output_dir / "train.log").read_text()
    assert "warm start: the untrained model cannot start" in log and "epoch    2/2" in log     # the log of the run holds both
    # a later run in the same folder that needs no warm start does not keep the record of this one
    again = train(make_cfg("name=warm_start_needed", "train.warm_start=auto"), dataset=smoke_dataset)
    assert "warm_start" not in again.extras and not (again.output_dir / "warm_start.json").exists()


def test_networks_are_carried_over_without_the_diffusion(make_cfg) -> None:
    cfg = make_cfg("model.diffusion.eta_init=0.5")
    source = ws._model_at(cfg, 0.5, CPU)
    with torch.no_grad():
        for p in source.drift.parameters():
            p.add_(1.0)
    target = ws._model_at(cfg, 2.0, CPU, state=source.state_dict())
    assert float(target.diffusion.eta) == pytest.approx(2.0)               # its own coefficient
    for a, b in zip(source.drift.parameters(), target.drift.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(RuntimeError, match="do not fit"):
        ws.load_networks(target, {"drift.unknown": torch.zeros(1)})


def test_a_trained_diffusion_gets_no_warm_start(make_cfg, smoke_dataset, caplog) -> None:
    cfg = make_cfg("model.diffusion={kind: cir, eta_init: 2.0, trainable: true}", "train.warm_start=auto")
    import logging

    assert ws.warm_start(cfg, smoke_dataset, cfg.output_dir, logging.getLogger("ndim.test"), CPU) is None


def test_a_stage_that_breaks_down_ends_the_warm_start(make_cfg, smoke_dataset, monkeypatch) -> None:
    import importlib

    module = importlib.import_module("ndim.train")
    real = module.compute_objective

    def objective(*args, **kwargs):
        out = real(*args, **kwargs)
        out.loss = out.loss * float("nan")
        return out

    monkeypatch.setattr(module, "compute_objective", objective)
    cfg = make_cfg("name=warm_start_breaks", "model.diffusion.eta_init=2.0", UNBOUNDED, "train.warm_start=auto")
    with pytest.raises(ws.WarmStartError, match="warm start: training broke down.*another seed"):
        train(cfg, dataset=smoke_dataset)
