"""Observations of one's own: the file formats, the automatic split and scales, ``scripts/fit.py`` (one command from a
file of event times to a trained and evaluated model) and the preparation of the US bank call-centre data."""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ndim import plots
from ndim.data import EventData, load_or_generate_dataset, read_event_file, split_observations, suggest_scales


CPU = torch.device("cpu")


def _rows(dataset):
    return dataset.train.as_lists() + dataset.test.as_lists()


def _same(a, b) -> bool:
    """The same event times in every observation (padding left aside)."""
    def times(x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.sort(x[np.isfinite(x)])

    return len(a) == len(b) and all(times(x).shape == times(y).shape and np.allclose(times(x), times(y), atol=1e-5) for x, y in zip(a, b))


# --------------------------------------------------------------------------------------
# File formats
# --------------------------------------------------------------------------------------
def test_every_format_holds_the_same_observations(smoke_dataset, tmp_root: Path) -> None:
    rows = _rows(smoke_dataset)
    rows[3] = rows[3][:0]                                                # an observation without events
    k_max = max(len(r) for r in rows)
    padded = np.full((len(rows), k_max), np.nan)
    for i, r in enumerate(rows):
        padded[i, : len(r)] = r
    folder = tmp_root / "formats"
    folder.mkdir(exist_ok=True)
    np.save(folder / "events.npy", padded)
    np.savez(folder / "events.npz", events=padded, T=np.array(4.0))
    (folder / "events.json").write_text(json.dumps([[float(t) for t in r] for r in rows]))
    (folder / "with_T.json").write_text(json.dumps({"events": [[float(t) for t in r] for r in rows], "T": 4.0}))
    # one observation per line; comments and blank lines are skipped, a lone separator is an observation without events
    (folder / "lines.csv").write_text("# event times\n\n" + "\n".join(",".join(f"{t:.7f}" for t in r) if len(r) else "," for r in rows) + "\n\n")
    (folder / "blanks.txt").write_text("\n".join(" ".join(f"{t:.7f}" for t in r) if len(r) else ";" for r in rows) + "\n")
    # a csv saved by a spreadsheet program: a byte-order mark first, Windows line ends, padded with empty cells
    (folder / "spreadsheet.csv").write_bytes(b"\xef\xbb\xbf" + "\r\n".join(
        (",".join(f"{t:.7f}" for t in r) + "," * (k_max - len(r))) if len(r) else "," * (k_max - 1) for r in rows).encode() + b"\r\n")
    for name in ("events.npy", "events.npz", "events.json", "with_T.json", "lines.csv", "blanks.txt", "spreadsheet.csv"):
        content = read_event_file(folder / name)
        assert content.train is None and content.test is None and _same(content.events, rows), name
        assert content.T == (4.0 if name in ("events.npz", "with_T.json") else None), name
    # one event per line with a header; an observation without events cannot be written this way
    long_rows = [r for r in rows if len(r)]
    (folder / "long.csv").write_text("observation,time\n" + "".join(f"day{k:02d},{t:.7f}\n" for k, r in enumerate(long_rows) for t in r))
    (folder / "long_other_names.tsv").write_text("t\tweight\tid\n" + "".join(f"{t:.7f}\t1\t{k}\n" for k, r in enumerate(long_rows) for t in r))
    assert _same(read_event_file(folder / "long.csv").events, long_rows)
    assert _same(read_event_file(folder / "long_other_names.tsv").events, long_rows)        # the columns are found by name
    split = folder / "split.npz"
    np.savez(split, train_events=padded[:16], test_events=padded[16:])
    content = read_event_file(split)
    assert content.events is None and _same(content.train, rows[:16]) and _same(content.test, rows[16:])


def test_unreadable_files(tmp_root: Path) -> None:
    folder = tmp_root / "unreadable"
    folder.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError):
        read_event_file(folder / "missing.csv")
    (folder / "events.xlsx").write_text("0.5")
    with pytest.raises(ValueError, match="unknown file type"):
        read_event_file(folder / "events.xlsx")
    (folder / "words.csv").write_text("0.5,1.5\n0.2,later\n")
    with pytest.raises(ValueError, match="line 2"):
        read_event_file(folder / "words.csv")
    (folder / "one_column.csv").write_text("time\n0.5\n")
    with pytest.raises(ValueError, match="two columns"):
        read_event_file(folder / "one_column.csv")
    (folder / "empty.csv").write_text("# nothing\n")
    with pytest.raises(ValueError, match="no observations"):
        read_event_file(folder / "empty.csv")
    np.savez(folder / "other_keys.npz", times=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="events"):
        read_event_file(folder / "other_keys.npz")


def test_observations_without_a_split_are_split_at_random(make_cfg, smoke_dataset, tmp_root: Path) -> None:
    rows = _rows(smoke_dataset)
    path = tmp_root / "all_observations.json"
    path.write_text(json.dumps([[float(t) for t in r] for r in rows]))
    args = ["data.source=file", f"data.path={json.dumps(str(path))}"]
    ds, source = load_or_generate_dataset(make_cfg(*args))
    assert source == path and len(ds.test) == 6 and len(ds.train) == 18 and not ds.has_truth          # a quarter of 24 is held out
    train_idx, test_idx = split_observations(24, 0.25, 0)
    assert sorted(train_idx + test_idx) == list(range(24)) and train_idx == sorted(train_idx) and test_idx == sorted(test_idx)
    assert _same(ds.train.as_lists(), [rows[i] for i in train_idx]) and _same(ds.test.as_lists(), [rows[i] for i in test_idx])
    again, _ = load_or_generate_dataset(make_cfg(*args))                                              # the same split every time
    assert torch.equal(again.test.counts, ds.test.counts)
    other, _ = load_or_generate_dataset(make_cfg(*args, "data.seed=1"))                               # another seed, another split
    assert split_observations(24, 0.25, 1)[1] != test_idx and len(other.test) == 6
    half, _ = load_or_generate_dataset(make_cfg(*args, "data.test_fraction=0.5"))
    assert len(half.test) == 12
    assert split_observations(2, 0.01, 0)[1] != [] and len(split_observations(2, 0.99, 0)[0]) == 1    # never an empty part
    with pytest.raises(ValueError, match="two observations"):
        split_observations(1, 0.25, 0)
    for bad in ("data.test_fraction=0.0", "data.test_fraction=1.0"):
        with pytest.raises(ValueError, match="test_fraction"):
            make_cfg(*args, bad)
    with pytest.raises(ValueError, match="outside"):
        load_or_generate_dataset(make_cfg(*args, "data.T=2.0"))


def test_scales_are_taken_from_the_training_observations() -> None:
    data = EventData.from_lists([np.linspace(0.1, 23.9, k) for k in (250, 300, 317)], 24.0)
    assert suggest_scales(data) == {"model.intensity_scale": 12.0, "model.posterior.count_scale": 289.0}      # 289 events in 24 hours
    sparse = EventData.from_lists([np.array([0.5]), np.array([])], 10.0)                                       # less than one event each
    assert suggest_scales(sparse) == {"model.intensity_scale": 0.1, "model.posterior.count_scale": 1.0}


# --------------------------------------------------------------------------------------
# scripts/fit.py
# --------------------------------------------------------------------------------------
def test_fit_trains_and_evaluates_from_a_file_of_event_times(smoke_dataset, tmp_root: Path, capsys: pytest.CaptureFixture,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import fit

    rows = _rows(smoke_dataset)
    path = tmp_root / "my_events.csv"
    path.write_text("\n".join(",".join(f"{t:.6f}" for t in r) for r in rows) + "\n")
    out = tmp_root / "fit_outputs"
    common = [str(path), "--T", "4", "--profile", "tiny", "--set", f"output_root={json.dumps(str(out))}", "eval.n_initial=200"]
    assert fit.main([*common, "--dry-run"]) == 0                          # reads the data and stops
    printed = capsys.readouterr().out
    assert "24 realisations on [0, 4]: 18 for training, 6 for testing" in printed and "burn-in of 0.96" in printed
    assert "eta * sqrt(Z) of the intensity with eta = " in printed and "estimated from the training observations" in printed
    assert "diffusion coefficient: eta = " in printed                      # the estimate, with its interval or the floor
    assert not (out / "my_events").exists()

    assert fit.main(common) == 0
    run = out / "my_events"                                               # named after the file
    import yaml

    saved = yaml.safe_load((run / "config.yaml").read_text())
    counts = torch.tensor([float(len(rows[i])) for i in split_observations(24, 0.25, 0)[0]])
    assert saved["data"]["source"] == "file" and saved["data"]["T"] == 4.0 and Path(saved["data"]["path"]).resolve() == path.resolve()
    assert saved["model"]["intensity_scale"] == pytest.approx(float(counts.mean()) / 4.0, rel=5e-3)           # set from the data
    assert saved["model"]["posterior"]["count_scale"] == pytest.approx(float(counts.mean()), rel=5e-3)
    # the default diffusion: the CIR form with a coefficient estimated from the training observations, not trained
    assert saved["model"]["diffusion"]["kind"] == "cir" and saved["model"]["diffusion"]["trainable"] is False
    estimate = json.loads((run / "eta_estimate.json").read_text())
    assert saved["model"]["diffusion"]["eta_init"] == pytest.approx(estimate["eta"], rel=1e-3) and estimate["n_observations"] == 18
    assert estimate["eta"] >= estimate["floor"] > 0 and (run / "eta_estimate.png").stat().st_size > 0
    assert saved["train"]["warm_start"] == "auto"
    assert saved["model"]["burn_in"] == "auto" and isinstance(saved["model"]["x_start"], float)
    metrics = json.loads((run / "eval" / "metrics.json").read_text())
    assert metrics["errors"] == [] and set(metrics["stages"]) == {"curves", "drift", "samples", "initial", "posterior"}
    for name in ("learned_model.png", "training_curves.png", "generated_vs_true_counts.png", "initial_distribution.png",
                 "posterior_most_events_full.png"):
        assert (run / "eval" / name).stat().st_size > 0, name
    summary = (run / "RESULTS.md").read_text()
    likelihood = metrics["test_likelihood"]                                # the model against two models without a latent path
    assert likelihood["n_obs"] == 6 and likelihood["n_particles"] == 2000 and len(likelihood["per_observation"]["marginal_loglik"]) == 6
    assert likelihood["elbo_gap"] == pytest.approx(likelihood["marginal_loglik"] - likelihood["elbo"])
    assert likelihood["marginal_minus_poisson"]["mean"] == pytest.approx(likelihood["marginal_loglik"] - likelihood["poisson"])
    for text in ("# Results: my_events", "18 for training, 6 held out", "Burn-in: the diffusion starts at t = -0.96",
                 "marginal log-likelihood (particle filter, 2000 particles)", "reference: Poisson process with the event rate",
                 "eval/learned_model.png", "Learned posterior against MCMC under the learned prior",
                 "the CIR diffusion eta sqrt(Z) of the intensity, with eta = ", "eta_estimate.png"):
        assert text in summary, text
    assert ("estimated from the training observations before training" in summary) != ("the floor" in summary)

    # evaluating the run again writes the summary again, with the remarks of the first time
    from scripts import evaluate

    (run / "fit_notes.json").write_text(json.dumps(["A remark made when the run was set up."]))
    (run / "RESULTS.md").unlink()
    assert evaluate.main(["--run", str(run), "--which", "samples", "--set", "eval.n_generated=6"]) == 0
    again = (run / "RESULTS.md").read_text()
    assert "A remark made when the run was set up." in again and "eval/learned_model.png" in again
    assert "between two MCMC runs" in again
    # a run that is still training has no train_summary.json yet: the summary names the best epoch evaluated so far
    from ndim.report import write_results

    (run / "train_summary.json").rename(run / "train_summary.kept")
    assert "best test ELBO" in write_results(run).read_text()
    (run / "train_summary.kept").rename(run / "train_summary.json")
    assert set(metrics["learned_model"]["euler_gain"]) >= {"min", "share_below_minus_one", "share_below_minus_one_burn_in", "share_negative"}
    # a run that broke down, and a drift too stiff for the grid, are the first things the summary says
    kept, kept_metrics = (run / "train_summary.json").read_text(), (run / "eval" / "metrics.json").read_text()
    (run / "train_summary.json").write_text(json.dumps({**json.loads(kept), "broke_down_at": 2}))
    stiff = json.loads(kept_metrics)
    stiff["learned_model"]["euler_gain"].update(min=-2.2, share_below_minus_one=0.2, share_below_minus_one_burn_in=0.9, share_below_minus_one_window=0.02)
    (run / "eval" / "metrics.json").write_text(json.dumps(stiff))
    text = write_results(run).read_text()
    assert "- Training broke down at epoch 2" in text and "too stiff for the Euler grid on 20.0% of the steps" in text and "mostly on the burn-in" in text
    (run / "train_summary.json").write_text(kept)
    (run / "eval" / "metrics.json").write_text(kept_metrics)
    text = write_results(run).read_text()
    assert "broke down" not in text
    # the two bounds of the state-space parametrization: named in what was run, and remarked on only when they act
    assert "capped at 1 / sqrt(dt), which changes it only below the intensity" in text and "posterior control bounded at 30" in text
    bounds = metrics["test_elbo"]["bounds"]
    assert bounds["control_bound"] == 30.0 and bounds["control_bound_share"] is not None and bounds["diffusion_bound_share"] is not None
    assert metrics["learned_model"]["diffusion_bound_share"] is not None
    acting = json.loads(kept_metrics)
    acting["test_elbo"]["bounds"].update(control_bound_share=0.25, diffusion_bound_share=0.0)
    acting["learned_model"]["diffusion_bound_share"] = 0.1
    (run / "eval" / "metrics.json").write_text(json.dumps(acting))
    text = write_results(run).read_text()
    assert "The bound on the diffusion holds on 10.0% of the steps" in text and "The bound on the posterior control (30) holds on 25.0%" in text
    quiet = json.loads(kept_metrics)
    quiet["test_elbo"]["bounds"].update(control_bound_share=0.0, diffusion_bound_share=0.0)
    quiet["learned_model"]["diffusion_bound_share"] = 0.0
    (run / "eval" / "metrics.json").write_text(json.dumps(quiet))
    assert "The bound on" not in write_results(run).read_text()
    (run / "eval" / "metrics.json").write_text(kept_metrics)
    write_results(run)
    synthetic = tmp_root / "outputs" / "smoke_pytest_run"                  # a run on synthetic data gets no such summary
    if (synthetic / "config.yaml").exists() and (synthetic / "history.json").exists():
        assert write_results(synthetic) is None

    # the unit diffusion, a name of one's own, no evaluation
    assert fit.main([*common, "--diffusion", "unit", "--name", "unit_run", "--epochs", "1", "--no-evaluate"]) == 0
    saved = yaml.safe_load((out / "unit_run" / "config.yaml").read_text())
    assert saved["model"]["diffusion"]["kind"] == "constant" and saved["model"]["diffusion"]["eta_init"] == 1.0
    assert saved["model"]["diffusion"]["trainable"] is False and saved["train"]["epochs"] == 1
    assert (out / "unit_run" / "RESULTS.md").exists() and not (out / "unit_run" / "eval").exists()
    assert not (out / "unit_run" / "eta_estimate.json").exists()

    # the coefficient set by hand
    assert fit.main([*common, "--eta", "0.5", "--name", "eta_by_hand", "--epochs", "1", "--no-evaluate"]) == 0
    saved = yaml.safe_load((out / "eta_by_hand" / "config.yaml").read_text())
    assert saved["model"]["diffusion"] ["kind"] == "cir" and saved["model"]["diffusion"]["eta_init"] == 0.5
    assert not (out / "eta_by_hand" / "eta_estimate.json").exists()
    assert "eta = 0.5, as given and not trained" in (out / "eta_by_hand" / "RESULTS.md").read_text()

    # a warm start that breaks down ends the command with its message and exit code 1, not with a traceback
    import importlib

    from ndim.warm_start import WarmStartError

    def broken(*args, **kwargs):
        raise WarmStartError("warm start: training broke down at the diffusion coefficient 0.2 (configured: 0.5)")

    monkeypatch.setattr(importlib.import_module("ndim.train"), "train", broken)
    capsys.readouterr()
    assert fit.main([*common, "--eta", "0.5", "--name", "warm_start_fails", "--epochs", "1", "--no-evaluate"]) == 1
    assert "BROKE DOWN: warm start: training broke down" in capsys.readouterr().err


def test_fit_reports_what_is_missing(smoke_dataset, tmp_root: Path) -> None:
    from scripts import fit

    CLIError = fit.CLIError
    path = tmp_root / "events_for_errors.csv"
    path.write_text("0.5,1.5,3.5\n0.2,2.0\n1.0,3.9\n")
    with pytest.raises(CLIError, match="--T"):
        fit.main([str(path), "--dry-run"])                                # a text file does not say how long the window is
    with pytest.raises(CLIError, match="outside"):
        fit.main([str(path), "--T", "3", "--dry-run"])                    # an event after the end of the window
    with pytest.raises(CLIError, match="not found"):
        fit.main([str(tmp_root / "nowhere.csv"), "--T", "4", "--dry-run"])
    stored = tmp_root / "events_with_T.json"
    stored.write_text(json.dumps({"events": [[0.5, 1.5], [0.2], [1.0, 3.9]], "T": 4.0}))
    assert fit.main([str(stored), "--dry-run"]) == 0                      # T is in the file
    for kind in fit.DIFFUSIONS:
        assert fit.main([str(stored), "--dry-run", "--diffusion", kind]) == 0
    assert sorted(fit.DIFFUSIONS) == ["cir", "cir-learned", "network", "unit"]
    assert fit.main([str(stored), "--dry-run", "--eta", "0.3"]) == 0
    with pytest.raises(CLIError, match="--eta"):
        fit.main([str(stored), "--dry-run", "--eta", "0.3", "--diffusion", "network"])
    with pytest.raises(CLIError, match="--eta"):
        fit.main([str(stored), "--dry-run", "--eta", "-1"])
    with pytest.raises(CLIError, match="disagrees"):
        fit.main([str(stored), "--T", "5", "--dry-run"])


def test_batch_size_follows_the_memory_of_the_machine(make_cfg) -> None:
    from scripts import fit

    cfg = make_cfg("train.batch_size=16", "train.m_paths=10", "data.n_steps=120", "data.T=24.0", "model.burn_in=6.0", "model.x_start=0.0")
    assert cfg.burn_steps == 30
    assert fit.training_memory_gb(cfg, 444, 16) == pytest.approx(4.27, abs=0.05)          # the US bank example
    assert fit.choose_batch_size(cfg, 444, 64.0) == (16, None)                             # enough memory: as configured
    batch, why = fit.choose_batch_size(cfg, 444, 8.0)
    assert batch == 8 and "batch size 8 in place of 16" in why and "444 events" in why
    assert fit.choose_batch_size(cfg, 444, 1.0)[0] == 2                                    # never below the smallest size
    assert fit.choose_batch_size(cfg, 444, None) == (16, None)                             # memory unknown: as configured
    fast = make_cfg("train.batch_size=16", "model.posterior.phi_uses_state=false")
    assert fit.choose_batch_size(fast, 444, 1.0) == (16, None)                             # the estimate is for the default posterior only
    cfg.train.device = "cuda"
    assert fit.choose_batch_size(cfg, 444, 8.0) == (16, None)                              # a GPU has its own memory


# --------------------------------------------------------------------------------------
# The results page for observations without a truth
# --------------------------------------------------------------------------------------
def test_learned_model_figure(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(-1.0, 4.0, 51)
    paths = np.exp(1.0 + 0.3 * np.cumsum(rng.standard_normal((200, 51)), axis=1) * np.sqrt(0.1))
    edges = np.linspace(0.0, 4.0, 9)
    observed = {"rate": np.full(8, 3.0), "count_sd": np.full(8, 1.4), "totals": rng.poisson(12.0, 40).astype(float)}
    model = {"rate": np.full(8, 2.8), "count_sd": np.full(8, 1.3), "totals": rng.poisson(11.0, 200).astype(float)}

    def drift(z: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return 0.5 * (3.0 - z)

    def diffusion(z: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return 0.3 * z

    out = plots.plot_learned_model(t, torch.as_tensor(paths), edges, observed, model, drift, diffusion, 4.0, path=tmp_path / "learned_model.png",
                                   title="the learned model")
    assert out == tmp_path / "learned_model.png" and out.stat().st_size > 0
    # without a burn-in the paths start at 0
    assert plots.plot_learned_model(t[10:], paths[:, 10:], edges, observed, model, drift, diffusion, 4.0) is None


# --------------------------------------------------------------------------------------
# scripts/prepare_citibike.py
# --------------------------------------------------------------------------------------
def test_citibike_preparation(tmp_root: Path, capsys: pytest.CaptureFixture) -> None:
    import zipfile

    from scripts import prepare_citibike

    CLIError = prepare_citibike.CLIError
    assert prepare_citibike.file_names([2025])[0] == "JC-202505-citibike-tripdata.csv.zip" and len(prepare_citibike.file_names([2025, 2026])) == 10
    assert prepare_citibike.federal_holidays(2025) == {"2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01"}
    assert prepare_citibike.federal_holidays(2026) == {"2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07"}       # 4 July 2026 is a Saturday

    # monthly files as Citi Bike publishes them: a zip with a csv whose column started_at holds the local start time
    rng = np.random.default_rng(0)
    source = tmp_root / "citibike_source"
    source.mkdir(exist_ok=True)
    per_day = 400
    for name in prepare_citibike.file_names([2025]):
        month = int(name[7:9])
        rows = ["ride_id,rideable_type,started_at,ended_at"]
        for day in range(1, 29):
            for sec in np.sort(rng.uniform(0.0, 86400.0, per_day)):
                h, m, sec_of_minute = int(sec // 3600), int(sec % 3600 // 60), sec % 60
                rows.append(f"r,classic_bike,2025-{month:02d}-{day:02d} {h:02d}:{m:02d}:{sec_of_minute:06.3f},later")
        rows.append("r,classic_bike,not a time,later")                                                              # skipped
        with zipfile.ZipFile(source / name, "w") as zf:
            zf.writestr(name[:-4], "\n".join(rows) + "\n")
    days = prepare_citibike.read_start_times([source / name for name in prepare_citibike.file_names([2025])])
    assert len(days) == 5 * 28 and all(len(v) == per_day and np.all(np.diff(v) >= 0) for v in days.values())
    arrays = prepare_citibike.prepare(days, keep=0.25, seed=1)
    dates = [str(d) for d in arrays["dates"]]
    import datetime

    assert all(datetime.date.fromisoformat(d).weekday() < 5 for d in dates) and dates == sorted(dates)
    assert not set(dates) & prepare_citibike.federal_holidays(2025) and "2025-05-27" in dates and "2025-05-24" not in dates
    events = arrays["events"]
    assert float(arrays["T"]) == 24.0 and np.nanmin(events) > 0.0 and np.nanmax(events) <= 24.0 and list(arrays["n_trips"]) == [per_day] * len(dates)
    kept = np.isfinite(events).sum(axis=1)
    assert abs(kept.mean() - 0.25 * per_day) < 5.0 and np.array_equal(np.nan_to_num(prepare_citibike.prepare(days, 0.25, 1)["events"]), np.nan_to_num(events))
    with pytest.raises(CLIError):
        prepare_citibike.prepare(days, keep=0.0)

    # the command: the files are fetched into --raw when they are missing, and the result is a file for scripts/fit.py
    raw, out = tmp_root / "citibike_raw", tmp_root / "citibike_test.npz"
    with pytest.raises(CLIError, match="--download"):
        prepare_citibike.main(["--raw", str(raw), "--years", "2025", "--out", str(out)])
    assert prepare_citibike.main(["--raw", str(raw), "--years", "2025", "--out", str(out), "--download", "--base-url", source.as_uri(),
                                  "--keep", "0.25", "--seed", "1"]) == 0
    printed = capsys.readouterr().out
    assert f"{len(dates)} weekdays from {dates[0]} to {dates[-1]}" in printed and "coefficient of the diffusion" in printed and "--eta" in printed
    assert printed.rstrip().splitlines()[-1] == f"fit: python scripts/fit.py {out}"                        # the plain command: it estimates eta itself
    assert prepare_citibike.main(["--raw", str(raw), "--years", "2025", "--out", str(out), "--keep", "0.25", "--seed", "1", "--no-eta"]) == 0
    printed = capsys.readouterr().out
    assert "coefficient of the diffusion" not in printed and printed.rstrip().splitlines()[-1] == f"fit: python scripts/fit.py {out}"
    assert sorted(p.name for p in raw.iterdir()) == sorted(prepare_citibike.file_names([2025]))
    content = read_event_file(out)
    assert content.T == 24.0 and _same(content.events, [e[np.isfinite(e)] for e in events])
    assert out.with_name("citibike_test_eta.png").stat().st_size > 0
    with pytest.raises(CLIError, match="could not download"):
        prepare_citibike.main(["--raw", str(tmp_root / "citibike_none"), "--years", "2024", "--download", "--base-url", source.as_uri()])


# --------------------------------------------------------------------------------------
# scripts/prepare_usbank.py
# --------------------------------------------------------------------------------------
def test_usbank_preparation(tmp_root: Path, capsys: pytest.CaptureFixture) -> None:
    from scripts import prepare_usbank

    CLIError = prepare_usbank.CLIError

    # calls per minute: the k-th of c calls of minute j arrives at j + k/c minutes
    times = prepare_usbank.arrival_times(np.array([2.0, 0.0, 1.0, 4.0]))
    assert np.allclose(times * 60.0, [0.5, 1.0, 3.0, 3.25, 3.5, 3.75, 4.0])
    assert prepare_usbank.arrival_times(np.zeros(5)).size == 0

    # a raw file: two complete Fridays, one Friday with a missing minute, one Thursday
    rng = np.random.default_rng(0)
    lines = []
    for date, complete in (("2001-03-30", True), ("2001-04-06", True), ("2001-04-13", False), ("2001-04-05", True)):
        for minute in range(1440):
            if not complete and minute == 700:
                continue
            lines.append(f"{int(rng.integers(150, 250)):7d} {date} {float(minute)}")
    raw = tmp_root / "calls_by_minute.txt"
    raw.write_text("\n".join(lines) + "\n")
    arrays = prepare_usbank.prepare(raw, thinning=0.001, seed=123)
    events = arrays["events"]
    assert list(arrays["dates"]) == ["2001-03-30", "2001-04-06"] and float(arrays["T"]) == 24.0          # complete Fridays, in order
    assert events.shape[0] == 2 and np.nanmin(events) > 0.0 and np.nanmax(events) <= 24.0
    kept = np.isfinite(events).sum(axis=1)
    assert np.all(np.abs(kept - 0.001 * arrays["calls_per_day"]) < 5 * np.sqrt(0.001 * arrays["calls_per_day"]))   # about one call in a thousand
    again = prepare_usbank.prepare(raw, thinning=0.001, seed=123)
    assert np.array_equal(np.nan_to_num(again["events"]), np.nan_to_num(events))                          # the seed fixes the thinning
    assert prepare_usbank.prepare(raw, day_of_week=3)["dates"].tolist() == ["2001-04-05"]                 # the Thursday
    morning = prepare_usbank.prepare(raw, start_minute=480, end_minute=720)
    assert float(morning["T"]) == 4.0 and np.nanmax(morning["events"]) <= 4.0 and morning["events"].shape[0] == 2
    for bad in ({"thinning": 0.0}, {"start_minute": 100, "end_minute": 50}, {"day_of_week": 6}):
        with pytest.raises(CLIError):
            prepare_usbank.prepare(raw, **bad)

    # the command writes a file that the model reads as it is
    out = tmp_root / "usbank_test.npz"
    assert prepare_usbank.main(["--csv", str(raw), "--out", str(out)]) == 0
    assert "2 complete fridays" in capsys.readouterr().out
    content = read_event_file(out)
    assert content.T == 24.0 and len(content.events) == 2 and content.train is None
    assert prepare_usbank.prepare(raw, with_counts=True)["calls_per_minute"].shape == (2, 1440)

    # with enough days the script also estimates the coefficient of the diffusion from all calls
    lines = []
    for week in range(8):
        date = (datetime.date(2001, 3, 30) + datetime.timedelta(days=7 * week)).isoformat()
        level = rng.gamma(20.0, 1.0 / 20.0)                               # days differ in level
        lines += [f"{int(rng.poisson(200.0 * level)):7d} {date} {float(minute)}" for minute in range(1440)]
    raw8 = tmp_root / "calls_by_minute_8_days.txt"
    raw8.write_text("\n".join(lines) + "\n")
    out8 = tmp_root / "usbank_test_8_days.npz"
    assert prepare_usbank.main(["--csv", str(raw8), "--out", str(out8)]) == 0
    printed = capsys.readouterr().out
    assert "from all calls of the 6 training days" in printed and "sqrt(0.001) x" in printed
    assert f"fit: python scripts/fit.py {out8} --steps 240 --epochs 60 --eta " in printed
    assert out8.with_name("usbank_test_8_days_eta.png").exists()
    assert prepare_usbank.main(["--csv", str(raw8), "--out", str(out8), "--no-eta"]) == 0
    assert "training days" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# Likelihood of observations without a truth: particle filter and reference models
# --------------------------------------------------------------------------------------
def test_events_per_step_follows_the_likelihood_convention() -> None:
    from ndim.metrics import events_per_step

    events = torch.tensor([[0.05, 0.1, 0.19, 0.4, 1.0, float("inf")], [0.99, float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]])
    counts = events_per_step(events, 0.1, 10)
    assert counts.shape == (2, 10) and counts.sum(1).tolist() == [5.0, 1.0]
    assert counts[0].tolist() == [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]      # 0.1 opens step 1; an event at T is in the last step
    assert counts[1, 9] == 1.0


def test_particle_filter_gives_the_marginal_likelihood(make_cfg, smoke_dataset) -> None:
    from ndim.likelihood import poisson_loglik
    from ndim.metrics import events_per_step, marginal_loglik_particle_filter
    from ndim.model import build_model
    from ndim.sde import brownian_increments

    events = smoke_dataset.test.events[:4]
    cfg = make_cfg("model.burn_in=0.96", "model.x_start=3.0", "model.drift.time_input=true",
                   "model.diffusion={kind: constant, eta_init: 0.3, trainable: true}")
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    N, dt, nb = cfg.data.n_steps, cfg.dt, model.n_burn
    logp, ess = marginal_loglik_particle_filter(model, events, 4000, torch.Generator().manual_seed(0))
    again, _ = marginal_loglik_particle_filter(model, events, 4000, torch.Generator().manual_seed(1))
    assert logp.shape == (4,) and torch.isfinite(logp).all() and float(ess.min()) >= 1.0
    assert float((logp - again).abs().max()) < 0.5                                   # two seeds agree
    # against importance sampling of whole prior paths, which is reliable for a model this smooth
    W = brownian_increments(40000, nb + N, dt, CPU, torch.Generator().manual_seed(2))
    paths = model.simulate_prior(W)
    for b in range(4):
        ll = poisson_loglik(paths, events[b:b + 1], paths.shape[0], dt, N, cfg.likelihood).double()
        direct = float(torch.logsumexp(ll, 0) - math.log(ll.numel()))
        assert float(logp[b]) == pytest.approx(direct, abs=0.6), b
    # without noise the prior is one path, and the estimate is the Poisson log-likelihood of that path
    with torch.no_grad():
        model.diffusion.log_eta.fill_(-30.0)
    exact_path = model.simulate_prior(torch.zeros(1, nb + N)).double()[0]
    counts = events_per_step(events, dt, N)
    exact = (counts * torch.log(exact_path[:-1])[None, :]).sum(1) - exact_path[:-1].sum() * dt
    quiet, _ = marginal_loglik_particle_filter(model, events, 20, torch.Generator().manual_seed(0))
    assert torch.allclose(quiet, exact, atol=1e-3)


def test_reference_models() -> None:
    from ndim.metrics import events_per_step, reference_logliks

    g = torch.Generator().manual_seed(0)
    n_steps, dt = 20, 0.5

    def draw(n: int, busy: torch.Tensor) -> torch.Tensor:
        """Observations with rate 6 on the first half of [0, 10] and 2 on the second, times a factor per observation."""
        rows = []
        for f in busy:
            times = torch.cat([torch.rand(int(torch.poisson(30.0 * f, generator=g)), generator=g) * 5.0,
                               5.0 + torch.rand(int(torch.poisson(10.0 * f, generator=g)), generator=g) * 5.0])
            rows.append(times.clamp_min(1e-6))
        out = torch.full((n, max(len(r) for r in rows)), float("inf"))
        for i, r in enumerate(rows):
            out[i, : len(r)] = r
        return out

    # no variation between observations: the factor is not needed, and the fitted shape is at the top of its range
    plain = reference_logliks(draw(200, torch.ones(200)), draw(100, torch.ones(100)), dt, n_steps)
    assert plain["rate"].shape == (n_steps,) and plain["rate"][:10].mean() == pytest.approx(6.0, rel=0.1) and plain["rate"][10:].mean() == pytest.approx(2.0, rel=0.15)
    assert plain["gamma_factor_cv"] < 0.06 and abs(float((plain["poisson_gamma"] - plain["poisson"]).mean())) < 0.3
    # a factor with a coefficient of variation of 0.5: it is recovered, and the model with the factor wins clearly
    factor = lambda n: torch.distributions.Gamma(4.0, 4.0).sample((n,))             # noqa: E731
    torch.manual_seed(0)
    varied = reference_logliks(draw(300, factor(300)), draw(150, factor(150)), dt, n_steps)
    assert varied["gamma_factor_cv"] == pytest.approx(0.5, abs=0.08)
    assert float((varied["poisson_gamma"] - varied["poisson"]).mean()) > 2.0
    counts = events_per_step(draw(5, torch.ones(5)), dt, n_steps)
    assert counts.shape == (5, n_steps)

