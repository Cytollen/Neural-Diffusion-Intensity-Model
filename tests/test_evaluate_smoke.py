"""Smoke test of the evaluation suite ``ndim.evaluate.evaluate_run`` on the shared
2-epoch smoke training run (``conftest.trained_smoke_run``): all stages with a 20-step MCMC, every expected
figure exists, ``metrics.json`` has the required content and no errors, stage isolation works.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ndim import evaluate
from ndim.train import TrainResult

ALL_STAGES = ["curves", "drift", "samples", "posterior", "wasserstein", "l2"]

EXPECTED_FIGURES = [
    "training_curves.png",
    "drift_comparison.png",
    "drift_heatmap.png",
    "drift_3d.html",
    "prior_paths_comparison.png",
    "generated_vs_true_counts.png",
    "generated_vs_true_counts_paths.png",
    "posterior_fewest_events_full.png",
    "posterior_most_events_full.png",
    "posterior_fewest_events_partial.png",
    "posterior_most_events_partial.png",
    "wasserstein.png",
]


@pytest.fixture(scope="module")
def eval_result(trained_smoke_run: TrainResult, tmp_path_factory: pytest.TempPathFactory):
    out_dir = tmp_path_factory.mktemp("eval")
    result = evaluate.evaluate_run(
        Path(trained_smoke_run.output_dir),
        out_dir=out_dir,
        overrides=["eval.mcmc.n_steps=20", "eval.mcmc.n_chains=4"],
        ckpt="best",
        which=ALL_STAGES,
        device="cpu",
    )
    return result, out_dir


def test_evaluate_run_writes_metrics_without_errors(eval_result) -> None:
    result, out_dir = eval_result
    assert isinstance(result, dict)
    mpath = out_dir / "metrics.json"
    assert mpath.exists()
    metrics = json.loads(mpath.read_text())
    errors = metrics.get("errors", {}) or {}
    assert not errors, f"evaluation stages failed: {errors}"
    assert (out_dir / "eval.log").exists() and "evaluation done" in (out_dir / "eval.log").read_text()
    assert set(metrics["stages"]) == set(ALL_STAGES)
    assert all(v["status"] == "ok" for v in metrics["stages"].values())
    # the returned dict is the JSON content
    assert result["stages"] == metrics["stages"]


def test_evaluate_run_figures_exist(eval_result) -> None:
    _, out_dir = eval_result
    missing = [f for f in EXPECTED_FIGURES if not (out_dir / f).exists()]
    assert not missing, f"missing figures: {missing}"
    for f in EXPECTED_FIGURES:
        assert (out_dir / f).stat().st_size > 0, f


def test_evaluate_run_metrics_content(eval_result) -> None:
    _, out_dir = eval_result
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # final test ELBO on the full test set
    te = metrics["test_elbo"]
    assert set(("elbo", "loglik", "kl")) <= set(te) and te["n_obs"] == metrics["dataset"]["n_test"]
    assert abs(te["elbo"] - (te["loglik"] - te["kl"])) < 1e-3
    # expected path deviation between the learned and the true prior
    dev = metrics["expected_path_deviation"]
    assert dev["l2"] >= 0 and dev["l1"] >= 0 and dev["rmse"] >= 0
    # no comparison of computation time with MCMC
    assert "predictive" not in metrics and "predictive" not in evaluate.STAGES
    # Wasserstein summary: train VI / test VI / test prior baseline
    ws = metrics["wasserstein"]
    assert set(("train VI", "test VI", "test prior baseline")) <= set(ws["summary"])
    assert len(ws["train"]["w2_vi"]) == min(metrics["config"]["eval"]["wasserstein_n_obs"], metrics["dataset"]["n_train"])
    # learned scalars vs truth: none here (MLP drift, fixed diffusion)
    params = metrics["parameters"]
    assert params["learned"] == {}
    assert params["truth"]["kappa"] == 0.3 and params["truth"]["theta"] == 80.0
    assert not (out_dir / "param_history.png").exists()
    # posterior cases + MCMC diagnostics
    cases = metrics["posterior"]["cases"]
    assert set(cases) == {"fewest_events_full", "fewest_events_partial", "most_events_full", "most_events_partial"}
    for c in cases.values():
        assert c["reference"]["kind"] == "analytic"
        assert c["reference"]["exact_finite_dimensional"] is True
        assert "mcmc" not in c
        assert "w2_vi_vs_reference" in c
    assert cases["fewest_events_full"]["n_events_total"] <= cases["most_events_full"]["n_events_total"]
    assert cases["fewest_events_partial"]["n_events_observed"] <= cases["fewest_events_full"]["n_events_observed"]
    assert metrics["posterior_reference"]["kind"] == "analytic"
    assert metrics["posterior_reference"]["uses_learned_parameters"] is False
    assert metrics["mcmc"]["comparison_figures"] is False and "timing_table" not in metrics["mcmc"]
    assert not (out_dir / "mcmc_traces.png").exists()
    # timings, resolved config, checkpoint info
    assert set(metrics["timings"]) >= set(ALL_STAGES) | {"total"}
    assert metrics["config"]["eval"]["mcmc"]["n_steps"] == 20            # override applied
    assert metrics["config"]["train"]["epochs"] == 2                     # training section untouched
    assert metrics["checkpoint"]["path"].endswith("ckpt_best.pt")

    # every numeric leaf is finite (NaN / inf are written as null)
    def _walk(o):
        if isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
        elif isinstance(o, float):
            assert math.isfinite(o), "non-finite value in metrics.json"

    _walk(metrics)


def test_evaluate_run_subset_of_stages(trained_smoke_run: TrainResult, tmp_path: Path) -> None:
    """``which`` restricts the stages; ``out_dir`` defaults to ``run_dir/eval``; non-eval overrides are ignored."""
    result = evaluate.evaluate_run(Path(trained_smoke_run.output_dir), which=["curves", "drift"], device="cpu",
                                   overrides=["train.epochs=999", "eval.drift_3d_html=false"])
    assert isinstance(result, dict)
    default_out = Path(trained_smoke_run.output_dir) / "eval"
    assert (default_out / "metrics.json").exists()
    assert (default_out / "training_curves.png").exists()
    assert (default_out / "drift_comparison.png").exists()
    assert not (default_out / "drift_3d.html").exists()                  # eval override applied
    assert not (default_out / "wasserstein.png").exists()                # stage not requested
    metrics = json.loads((default_out / "metrics.json").read_text())
    assert not (metrics.get("errors") or {})
    assert {"curves", "drift"} <= set(metrics["stages"])
    assert metrics["config"]["train"]["epochs"] == 2                     # training override ignored
    with pytest.raises(ValueError):
        evaluate.evaluate_run(Path(trained_smoke_run.output_dir), which=["nope"], device="cpu")


def test_rerun_merges_into_existing_report(trained_smoke_run: TrainResult, tmp_path: Path) -> None:
    """Re-running a subset of stages keeps the other stages' results and replaces the re-run ones."""
    out_dir = tmp_path / "eval_merge"
    evaluate.evaluate_run(Path(trained_smoke_run.output_dir), out_dir=out_dir, which=["l2", "curves"], device="cpu")
    first = json.loads((out_dir / "metrics.json").read_text())
    assert "expected_path_deviation" in first and "test_elbo" in first
    evaluate.evaluate_run(Path(trained_smoke_run.output_dir), out_dir=out_dir, which=["drift"], device="cpu",
                          overrides=["eval.drift_3d_html=false"])
    second = json.loads((out_dir / "metrics.json").read_text())
    assert second["expected_path_deviation"] == first["expected_path_deviation"]      # kept
    assert set(second["stages"]) == {"l2", "curves", "drift"}
    assert second["stages_requested"] == ["drift"] and second["last_run"]["stages"] == ["drift"]
    assert "parameters" in second and "total" in second["timings"] and set(second["timings"]) >= {"l2", "curves", "drift"}


def test_failing_stage_is_isolated(trained_smoke_run: TrainResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage that raises is recorded in ``metrics['errors']`` and the other stages still run."""
    import ndim.metrics as metrics_mod

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(metrics_mod, "expected_path_deviation", boom)
    out_dir = tmp_path / "eval_fail"
    result = evaluate.evaluate_run(Path(trained_smoke_run.output_dir), out_dir=out_dir, which=["l2", "curves"], device="cpu")
    errs = result["errors"]
    assert len(errs) == 1 and errs[0]["stage"] == "l2" and "synthetic failure" in errs[0]["error"] and "Traceback" in errs[0]["traceback"]
    assert result["stages"]["l2"]["status"] == "failed" and result["stages"]["curves"]["status"] == "ok"
    assert (out_dir / "training_curves.png").exists()
    assert "synthetic failure" in (out_dir / "eval.log").read_text()
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert saved["errors"][0]["stage"] == "l2"


def test_load_eval_config_restricts_sections(trained_smoke_run: TrainResult) -> None:
    run_dir = Path(trained_smoke_run.output_dir)
    cfg = evaluate.load_eval_config(run_dir, overrides=["eval.n_generated=3", "train.device=cpu", "model.drift.hidden=[3]", "data.n_train=1"])
    assert cfg.eval.n_generated == 3 and cfg.train.device == "cpu"
    assert cfg.model.drift.hidden == [8] and cfg.data.n_train == 16      # architecture / data untouched
    cfg2 = evaluate.load_eval_config(run_dir, profile="laptop")
    assert cfg2.eval.mcmc.n_chains == 32 and cfg2.train.batch_size == 8   # profile touches eval only (not train.batch_size)
    with pytest.raises(FileNotFoundError):
        evaluate.load_eval_config(run_dir / "does_not_exist")


def test_analytic_comparison_never_calls_mcmc_and_removes_stale_trace(trained_smoke_run, tmp_path, monkeypatch):
    import ndim.metrics as metric_functions
    def forbidden(*args, **kwargs):
        raise AssertionError("MCMC must not run for the analytic CIR reference")
    monkeypatch.setattr(metric_functions, "sample_posterior_mcmc_batch", forbidden)
    output = tmp_path / "analytic"
    output.mkdir()
    (output / "mcmc_traces.png").write_bytes(b"stale old evaluation")
    result = evaluate.evaluate_run(trained_smoke_run.output_dir, out_dir=output,
                                   which=["posterior", "wasserstein"], overrides=["eval.n_paths_posterior=2"])
    assert not result["errors"]
    assert result["posterior_reference"]["kind"] == "analytic"
    assert not (output / "mcmc_traces.png").exists()
    for entry in result["posterior"]["cases"].values():
        assert entry["reference"]["exact_finite_dimensional"]
        assert entry["reference"]["truth"]["eta"] == 1.


def test_mcmc_reference_for_the_comparison_figures(trained_smoke_run, tmp_path):
    result = evaluate.evaluate_run(
        trained_smoke_run.output_dir, out_dir=tmp_path, which=["posterior"],
        overrides=["eval.posterior_reference=mcmc",
                   "eval.mcmc.n_steps=3", "eval.n_paths_posterior=2"])
    assert not result["errors"]
    assert result["posterior_reference"]["kind"] == "mcmc"
    assert (tmp_path / "mcmc_traces.png").exists()
    for entry in result["posterior"]["cases"].values():
        assert entry["mcmc"]["n_steps"] == 3
        assert "w2_vi_vs_mcmc" in entry


def test_reference_switch_invalidates_stale_stage_results(tmp_path):
    from ndim.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.eval.posterior_reference = "analytic"
    metrics = {"posterior_reference": {"kind": "mcmc"},
               "posterior": {"stale": True}, "samples": {"kept": True},
               "wasserstein": {"stale": True},
               "stages": {"curves": {"status": "ok"}, "posterior": {"status": "ok"}, "samples": {"status": "ok"}},
               "timings": {"posterior": 1., "curves": 1., "samples": 1.}, "errors": []}
    assert evaluate._configure_posterior_reference(metrics, cfg, tmp_path) == "analytic"
    assert "posterior" not in metrics and "wasserstein" not in metrics
    assert "samples" in metrics                                     # a stage that uses no reference is still valid
    assert set(metrics["stages"]) == {"curves", "samples"}
    assert set(metrics["timings"]) == {"curves", "samples"}
