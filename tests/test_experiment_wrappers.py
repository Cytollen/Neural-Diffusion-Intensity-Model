"""The example wrappers in ``experiments/``: which stages each one runs, with which config and flags, and an end-to-end
``--smoke`` pass of each."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import _runner  # noqa: E402  (after the sys.path insertion)

#: the two synthetic examples: each generates its data, then trains and evaluates
CONFIGS = {
    "intensity_cir_mlp": "configs/intensity/cir_mlp.yaml",
    "state_space_cir_burn_in": "configs/state_space/cir_burn_in.yaml",
}


def _stages(name: str, **kwargs):
    return importlib.import_module(name).stages(_runner.Experiment(**kwargs))


def test_the_wrappers_are_the_examples_of_the_repository() -> None:
    wrappers = sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.py") if not p.stem.startswith("_"))
    assert wrappers == sorted(CONFIGS)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_wrapper_stages_and_configs(name: str) -> None:
    stages = _stages(name)
    assert [s.key for s in stages] == ["data", "train"]
    assert stages[0].command == ["scripts/generate_data.py", "--config", CONFIGS[name]]
    assert stages[1].command == ["scripts/train.py", "--config", CONFIGS[name], "--evaluate"]     # the whole evaluation
    for stage in stages:
        assert (ROOT / stage.command[0]).is_file() and (ROOT / CONFIGS[name]).is_file()
        assert "--profile" not in stage.command and "--set" not in stage.command   # the configs as written


def test_profile_overrides_and_smoke_mode() -> None:
    config = CONFIGS["intensity_cir_mlp"]
    stages = {s.key: s.command for s in _stages("intensity_cir_mlp", profile="laptop", overrides=["train.epochs=3", "eval.n_generated=4"])}
    assert stages["train"] == ["scripts/train.py", "--config", config, "--profile", "laptop", "--evaluate",
                               "--set", "train.epochs=3", "eval.n_generated=4"]
    assert stages["data"] == ["scripts/generate_data.py", "--config", config, "--profile", "laptop", "--set", "train.epochs=3", "eval.n_generated=4"]
    smoke = {s.key: s.command for s in _stages("intensity_cir_mlp", smoke=True, profile="gpu")}
    for command in smoke.values():
        assert command[command.index("--profile") + 1] == "tiny"     # --smoke replaces the profile
    assert smoke["train"][-3:] == ["--set", "output_root=outputs/smoke", *_runner.SMOKE_OVERRIDES]
    assert _runner.SMOKE_OVERRIDES == ["train.max_abs_control=30"]   # keeps the paths of a 20-step run finite


def test_burn_in_example_starts_its_smoke_pass_where_the_config_says() -> None:
    """Full run and smoke pass both start the burn-in at X = 0 (intensity 1): the bounds on the diffusion and on the
    control keep the paths of the 20 smoke updates finite from there."""
    for smoke in (False, True):
        commands = [a for s in _stages("state_space_cir_burn_in", smoke=smoke) for a in s.command]
        assert not any("x_start" in a for a in commands)
    smoke = {s.key: s.command for s in _stages("state_space_cir_burn_in", smoke=True, overrides=["model.x_start=1.0"])}
    sets = smoke["train"][smoke["train"].index("--set") + 1:]
    assert sets == ["output_root=outputs/smoke", *_runner.SMOKE_OVERRIDES, "model.x_start=1.0"]      # the user's overrides come last


def test_dry_run_and_stage_selection(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("--dry-run must not start a stage")

    monkeypatch.setattr(_runner.subprocess, "call", forbidden)
    module = importlib.import_module("intensity_cir_mlp")
    assert _runner.run(module.__doc__, module.stages, ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[1/2] data" in out and "[2/2] train" in out
    assert _runner.run(module.__doc__, module.stages, ["--dry-run", "--only", "train"]) == 0
    out = capsys.readouterr().out
    assert "[1/1] train" in out and "scripts/generate_data.py" not in out
    assert _runner.run(module.__doc__, module.stages, ["--dry-run", "--skip", "train"]) == 0
    assert "[1/1] data" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown stage"):
        _runner.run(module.__doc__, module.stages, ["--dry-run", "--only", "nope"])


def test_failed_stages_are_counted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    calls = []

    def fake_call(command, cwd=None):
        calls.append((command, cwd))
        return 3 if "scripts/train.py" in command else 0

    monkeypatch.setattr(_runner.subprocess, "call", fake_call)
    module = importlib.import_module("intensity_cir_mlp")
    assert _runner.run(module.__doc__, module.stages, []) == 1       # exit code = number of failed stages
    assert [c[0][1] for c in calls] == ["scripts/generate_data.py", "scripts/train.py"]   # the data stage still ran first
    assert all(c[0][0] == sys.executable and c[1] == str(ROOT) for c in calls)            # run from the repository root
    assert "train: FAILED (exit 3)" in capsys.readouterr().out


@pytest.mark.parametrize("name", list(CONFIGS))
def test_wrapper_smoke_pass(name: str, tmp_root: Path) -> None:
    """``experiments/<example>.py --smoke`` end to end (two short subprocesses), written to the scratch space."""
    module = importlib.import_module(name)
    out_root = tmp_root / f"wrapper_outputs_{name}"
    # JSON-quoted so that a path with spaces or colons survives the YAML parsing of --set values
    overrides = [f"output_root={json.dumps(str(out_root))}", f"data.cache_dir={json.dumps(str(tmp_root / 'wrapper_data'))}"]
    assert _runner.run(module.__doc__, module.stages, ["--smoke", "--set", *overrides]) == 0
    run_dir = out_root / name
    burn_in = name == "state_space_cir_burn_in"
    expected = ["config.yaml", "ckpt_best.pt", "eval/metrics.json", "eval/drift_comparison.png", "eval/generated_vs_true_counts.png",
                "eval/posterior_fewest_events_full.png", "eval/posterior_most_events_partial.png", "eval/wasserstein.png"]
    for item in expected + (["eval/initial_distribution.png", "eval/mcmc_traces.png"] if burn_in else []):
        assert (run_dir / item).stat().st_size > 0, item
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    assert not metrics["errors"]
    assert set(metrics["stages"]) == {"curves", "drift", "l2", "samples", "initial", "posterior", "wasserstein"}
    assert not (run_dir / "eval" / "predictive_timing.png").exists()              # no comparison of computation time
    model = metrics["config"]["model"]
    if burn_in:
        assert model["state_space"] == "real" and model["burn_in"] == 1.0 and model["x_start"] == 0.0    # the start of the config
        assert model["diffusion"]["max_step_std"] == 1.0 and metrics["config"]["train"]["max_abs_control"] == 30.0
        assert metrics["posterior_reference"]["kind"] == "mcmc" and metrics["mcmc"]["config"]["kind"] == "pcn"
        assert metrics["stages"]["initial"]["status"] == "ok" and metrics["initial"]["true"]["mean"] == pytest.approx(40.0, abs=8.0)
    else:
        assert model["state_space"] == "intensity" and model["drift"]["kind"] == "mlp"
        assert metrics["posterior_reference"]["kind"] == "analytic"               # the exact CIR posterior
        assert metrics["stages"]["initial"]["status"] == "skipped"               # a known initial intensity: nothing to compare
