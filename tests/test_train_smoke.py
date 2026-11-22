"""End-to-end smoke test of ``ndim.train.train`` on ``configs/smoke.yaml`` (2 epochs, seconds)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from ndim.model import NDIM
from ndim.train import TrainResult, train

EXPECTED_FILES = [
    "config.yaml", "history.json", "history.csv", "training_curves.png", "drift_snapshot.png",
    "ckpt_best.pt", "ckpt_last.pt", "train.log", "train_summary.json",
]


def test_train_smoke_outputs(trained_smoke_run: TrainResult) -> None:
    res = trained_smoke_run
    out = Path(res.output_dir)
    assert out.name == "smoke_pytest_run"
    for f in EXPECTED_FILES:
        assert (out / f).exists(), f"missing {f}"
        assert (out / f).stat().st_size > 0, f"empty {f}"
    assert isinstance(res.model, NDIM)
    assert res.dataset is not None and len(res.dataset.train) == 16
    assert not res.interrupted


def test_train_smoke_history(trained_smoke_run: TrainResult) -> None:
    res = trained_smoke_run
    hist = res.history
    assert len(hist) == 2                                    # smoke.yaml: epochs = 2
    assert [h["epoch"] for h in hist] == [1, 2]
    for h in hist:
        # eval_every = 1 -> every epoch carries test metrics (fixed noise) and train-subset metrics
        for k in ("test_elbo", "test_loglik", "test_kl", "eval_train_elbo", "eval_train_loglik", "eval_train_kl"):
            assert k in h and math.isfinite(h[k]), k
        assert h["test_elbo"] == pytest.approx(h["test_loglik"] - h["test_kl"], rel=1e-4)
        for k in ("train_elbo_allhorizons", "train_loglik_allhorizons", "train_kl_allhorizons", "time_s", "kl_weight"):
            assert k in h and math.isfinite(h[k]), k
        assert "gradnorm_prior" in h and "gradnorm_posterior" in h
    # every mini-batch is trained at the full horizon, so the running train ELBO is defined
    assert math.isfinite(hist[0]["train_elbo"])
    # 16 observations in mini-batches of 8, each trained at the five horizons of base.yaml: 10 updates per epoch
    assert all(h["n_updates"] + h["n_skipped"] == 2 * 5 for h in hist)
    assert res.best_epoch in (1, 2)
    assert res.best_test_elbo == pytest.approx(max(h["test_elbo"] for h in hist))
    # history files agree with the in-memory history
    hj = json.loads((res.output_dir / "history.json").read_text())
    assert len(hj) == 2 and hj[-1]["epoch"] == 2
    with open(res.output_dir / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and set(rows[0]) >= {"epoch", "test_elbo", "train_elbo"}
    summary = json.loads((res.output_dir / "train_summary.json").read_text())
    assert summary["epochs"] == 2 and summary["best_epoch"] == res.best_epoch


def test_checkpoints_reload(trained_smoke_run: TrainResult) -> None:
    res = trained_smoke_run
    model, extra = NDIM.load_checkpoint(res.output_dir / "ckpt_best.pt")
    assert set(extra) >= {"epoch", "history", "best_epoch", "best_test_elbo"}
    assert extra["epoch"] == res.best_epoch
    assert model.cfg.name == "smoke_pytest_run"
    model_last, extra_last = NDIM.load_checkpoint(res.output_dir / "ckpt_last.pt")
    assert extra_last["epoch"] == 2
    # the last checkpoint holds the final in-memory parameters
    for (n1, p1), (n2, p2) in zip(res.model.named_parameters(), model_last.named_parameters()):
        assert n1 == n2 and torch.allclose(p1.detach().cpu(), p2.detach())
    assert model_last.named_scalars() == {}                 # MLP drift, fixed diffusion: no scalar parameters
    assert not (res.output_dir / "param_history.png").exists()


def test_every_mini_batch_is_trained_at_every_horizon(make_cfg, monkeypatch) -> None:
    """One optimizer step per mini-batch and per entry of ``train.horizons``, in the order of the list."""
    import importlib

    train_module = importlib.import_module("ndim.train")   # the module: the name ndim.train is the function it exports
    cfg = make_cfg("name=smoke_pytest_horizons", "train.epochs=1", "train.horizons=[1.0, 0.5, 1.0]", "train.eval_paths=1")
    seen = []
    objective = train_module.compute_objective

    def recording_objective(model, events, n_prime, *args, **kwargs):
        seen.append((int(events.shape[0]), int(n_prime)))
        return objective(model, events, n_prime, *args, **kwargs)

    monkeypatch.setattr(train_module, "compute_objective", recording_objective)
    res = train(cfg)
    full, half = cfg.horizon_steps(1.0), cfg.horizon_steps(0.5)
    # 16 observations in mini-batches of 8: two mini-batches, each at the three horizons in turn
    assert seen == [(8, full), (8, half), (8, full)] * 2
    assert res.history[0]["n_updates"] + res.history[0]["n_skipped"] == 6
    assert math.isfinite(res.history[0]["train_elbo"]) and math.isfinite(res.history[0]["train_elbo_allhorizons"])


def test_paper_parametric_run_reports_the_drift_parameters(make_cfg, tmp_path) -> None:
    """The paper's parametric model (state = intensity, drift ``kappa (theta - z)``): the two drift parameters are
    logged every epoch, plotted, and reported by the evaluation next to the truth."""
    from ndim import evaluate

    cfg = make_cfg("name=smoke_pytest_paper_parametric", "model.state_space=intensity", "model.drift.kind=cir",
                   "model.posterior.z_scale=80")
    res = train(cfg)
    assert type(res.model.diffusion).__name__ == "CIRSqrtDiffusion"          # the known diffusion sqrt(Z) ...
    assert not any(p.requires_grad for p in res.model.diffusion.parameters())  # ... is not trained
    assert set(res.model.named_scalars()) == {"drift.kappa", "drift.theta"}
    assert len(res.history) == 2
    for h in res.history:
        assert math.isfinite(h["drift.kappa"]) and math.isfinite(h["drift.theta"]) and math.isfinite(h["test_elbo"])
    assert (res.output_dir / "param_history.png").stat().st_size > 0
    metrics = evaluate.evaluate_run(res.output_dir, out_dir=tmp_path, which=["curves", "drift"], device="cpu",
                                    overrides=["eval.drift_3d_html=false"])
    assert not metrics["errors"]
    assert (tmp_path / "param_history.png").stat().st_size > 0
    params = metrics["parameters"]
    assert set(params["learned"]) == {"drift.kappa", "drift.theta"}
    assert params["truth"]["kappa"] == 0.3 and params["truth"]["theta"] == 80.0


def test_train_one_epoch_sensitivity_mode(make_cfg) -> None:
    """The forward-sensitivity gradient mode trains end to end (1 epoch, remaining-count posterior)."""
    cfg = make_cfg("name=smoke_pytest_sens", "train.epochs=1", "train.grad_mode=sensitivity",
                   "model.posterior.kind=remaining_count", "train.eval_paths=1", "train.plot_every=1")
    res = train(cfg)
    assert len(res.history) == 1
    assert math.isfinite(res.history[0]["test_elbo"])
    assert res.history[0]["n_skipped"] == 0
    assert (res.output_dir / "ckpt_last.pt").exists()


def test_train_resume(trained_smoke_run: TrainResult, make_cfg) -> None:
    """Resuming from ``ckpt_last.pt`` with more epochs continues the history."""
    cfg = make_cfg("name=smoke_pytest_resume", "train.epochs=3")
    res = train(cfg, dataset=trained_smoke_run.dataset, resume_from=trained_smoke_run.output_dir / "ckpt_last.pt")
    assert [h["epoch"] for h in res.history] == [1, 2, 3]
    assert res.history[:2] == trained_smoke_run.history[:2]


def test_resume_keeps_the_best_checkpoint_of_the_run_it_continues(make_cfg, smoke_dataset) -> None:
    """A run continued in its own folder replaces ``ckpt_best.pt`` only by a state with a better test ELBO."""
    import json

    from ndim.model import NDIM

    first = train(make_cfg("name=smoke_pytest_best"), dataset=smoke_dataset)
    out = first.output_dir
    _, before = NDIM.load_checkpoint(out / "ckpt_best.pt")
    assert before["epoch"] == first.best_epoch and before["best_test_elbo"] == first.best_test_elbo
    res = train(make_cfg("name=smoke_pytest_best", "train.epochs=4"), dataset=smoke_dataset, resume_from=out / "ckpt_last.pt")
    evaluated = {h["epoch"]: h["test_elbo"] for h in res.history if "test_elbo" in h}
    best_epoch = max(evaluated, key=evaluated.get)
    assert sorted(evaluated) == [1, 2, 3, 4]
    assert res.best_epoch == best_epoch and res.best_test_elbo == evaluated[best_epoch]       # over all epochs, not the new ones only
    _, after = NDIM.load_checkpoint(out / "ckpt_best.pt")
    assert after["epoch"] == best_epoch and after["best_test_elbo"] == evaluated[best_epoch]
    summary = json.loads((out / "train_summary.json").read_text())
    assert summary["epochs"] == 4 and summary["epochs_this_run"] == 2 and summary["best_epoch"] == best_epoch
    # nothing to train: the best checkpoint is not replaced by the last state
    same = train(make_cfg("name=smoke_pytest_best", "train.epochs=4"), dataset=smoke_dataset, resume_from=out / "ckpt_last.pt")
    assert same.best_epoch == best_epoch and NDIM.load_checkpoint(out / "ckpt_best.pt")[1]["epoch"] == best_epoch
    # continued in another folder, there is no earlier best to keep: the best of the new epochs
    moved = train(make_cfg("name=smoke_pytest_best_moved", "train.epochs=4"), dataset=smoke_dataset, resume_from=out / "ckpt_last.pt")
    assert moved.best_epoch in (3, 4) and NDIM.load_checkpoint(moved.output_dir / "ckpt_best.pt")[1]["epoch"] == moved.best_epoch


def test_a_run_whose_losses_are_all_non_finite_is_stopped(make_cfg, smoke_dataset, monkeypatch, base_overrides, capsys) -> None:
    """No update can bring back a model whose paths all run off: training stops after two such epochs and says so."""
    import importlib
    import json

    module = importlib.import_module("ndim.train")
    real = module.compute_objective
    calls = {"n": 0, "nan_until": 10 ** 9}

    def objective(*args, **kwargs):
        out = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] <= calls["nan_until"]:
            out.loss = out.loss * float("nan")
        return out

    monkeypatch.setattr(module, "compute_objective", objective)
    result = train(make_cfg("name=smoke_pytest_dead", "train.epochs=6"), dataset=smoke_dataset)
    assert result.broke_down_at == 1 and len(result.history) == module.DEAD_EPOCHS == 2            # stopped, 4 epochs short
    assert all(h["n_updates"] == 0 and h["n_skipped"] == 10 for h in result.history)
    summary = json.loads((result.output_dir / "train_summary.json").read_text())
    assert summary["broke_down_at"] == 1 and summary["epochs"] == 2
    log = (result.output_dir / "train.log").read_text()
    assert "training has broken down" in log and "finer grid" in log
    assert (result.output_dir / "ckpt_best.pt").exists() and (result.output_dir / "ckpt_last.pt").exists()
    # one epoch without a finite update, then training goes on: no stop
    calls.update(n=0, nan_until=10)
    result = train(make_cfg("name=smoke_pytest_one_dead_epoch", "train.epochs=3"), dataset=smoke_dataset)
    assert result.broke_down_at is None and [h["n_updates"] for h in result.history] == [0, 10, 10]
    assert "broke_down_at" not in json.loads((result.output_dir / "train_summary.json").read_text())
    # the script says so with its exit code, and does not evaluate a run in which no state had a finite test ELBO
    from scripts import train as train_script

    config = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    calls.update(n=0, nan_until=10 ** 9)
    capsys.readouterr()
    assert train_script.main(["--config", config, "--set", *base_overrides, "name=smoke_pytest_dead_script", "train.epochs=6"]) == 1
    assert "BROKE DOWN at epoch 1" in capsys.readouterr().out
    real_evaluation = module.evaluate_elbo
    monkeypatch.setattr(module, "evaluate_elbo", lambda *a, **k: {key: float("nan") for key in real_evaluation(*a, **k)})
    calls.update(n=0, nan_until=10 ** 9)
    code = train_script.main(["--config", config, "--evaluate", "--set", *base_overrides, "name=smoke_pytest_dead_unevaluated", "train.epochs=6"])
    assert code == 1 and "nothing is evaluated" in capsys.readouterr().out
    assert not (result.output_dir.parent / "smoke_pytest_dead_unevaluated" / "eval").exists()
