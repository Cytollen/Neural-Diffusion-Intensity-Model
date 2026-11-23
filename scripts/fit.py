#!/usr/bin/env python3
"""Fit the model to your own observations: n independent realisations of a point process on [0, T].

One command reads the event times, sets what depends on the data, trains, evaluates and writes a summary:

  python scripts/fit.py my_events.csv --T 24

DATA is the path of the file with the observations. The file can be anywhere; a relative path is read from the folder
the command is run in. It holds the event times of every observation, measured from the start of its window, so that
they lie in (0, T]:

  .csv / .txt   one observation per line, its event times separated by commas or blanks; or, with a header line,
                one event per line with the two columns observation,time
  .json         a list with one list of event times per observation
  .npy          a 2-D array, one row per observation, padded with nan
  .npz          the array events (as the .npy); or train_events and test_events when you split the observations
                yourself; T may be stored too

Set from the data (every one of them can be overridden with --set):

  data.T                        --T, or the T stored in the file
  the test observations         a quarter of the observations, picked at random (--test-fraction, data.seed)
  model.intensity_scale         the mean event rate of the training observations
  model.posterior.count_scale   their mean number of events
  model.diffusion.eta_init      the coefficient eta of the diffusion, estimated from the training observations (--eta)
  train.batch_size              16, or less when 16 would not fit comfortably into the memory of this machine

Nothing about the intensity has to be specified. The model is configs/observations.yaml: the latent state is
X = log Z (the state-space parametrization), the drift b(x, t) is a network, and a burn-in of T/4 before the window
lets the model learn the distribution of the initial intensity. The diffusion of the intensity has the CIR form
eta * sqrt(Z): eta is estimated from the training observations before training and then kept fixed (--diffusion cir,
the default; --eta sets it by hand). The other forms: the same with eta trained (--diffusion cir-learned), a network
sigma(x, t) of the latent state (--diffusion network), or the constant 1 for X in the time unit of the data
(--diffusion unit).

Results go to outputs/<name>/ in the folder the command is run in (--name; default: the name of the file):

  RESULTS.md        what was run, the main numbers, and the figures to look at
  eval/             the figures and metrics.json
  ckpt_best.pt      the trained model (state with the best test ELBO); train.log, history.csv, config.yaml
  eta_estimate.png  the estimate of eta; warm_start/: the first stages of training when a warm start was needed

Examples
  python scripts/fit.py my_events.csv --T 24
  python scripts/fit.py my_events.csv --T 24 --dry-run
  python scripts/fit.py data/citibike_jc_weekdays.npz
  python scripts/fit.py my_events.json --T 10 --epochs 50 --profile laptop --diffusion network
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

try:
    from _cli import CLIError, REPO_ROOT, collect_overrides, common_parent, eval_only_overrides, make_parser, resolve_profile, run_main
except ImportError:  # launched as ``python -m scripts.fit``
    from scripts._cli import (CLIError, REPO_ROOT, collect_overrides, common_parent, eval_only_overrides, make_parser,  # type: ignore
                              resolve_profile, run_main)

CONFIG = "configs/observations.yaml"
#: ``--diffusion``: the diffusion of the latent state X = log Z
DIFFUSIONS = {
    # eta * sqrt(Z) for the intensity; eta is estimated before training, or set with --eta, and kept fixed
    "cir": "{kind: cir, eta_init: auto, trainable: false}",
    # the same form with eta trained, from a small start
    "cir-learned": "{kind: cir, eta_init: 0.1, trainable: true}",
    "network": "{kind: mlp, eta_init: 0.2, trainable: true, time_input: true}",       # a network sigma(x, t)
    "unit": "{kind: constant, eta_init: 1.0, trainable: false}",                      # the constant 1, not trained
}
DIFFUSION_TEXT = {"cir-learned": "eta * sqrt(Z) of the intensity, eta learned", "network": "a network sigma(x, t) of the latent state",
                  "unit": "the constant 1 for the latent state, not trained"}
BATCH_SIZES = (16, 8, 4, 2)
MEMORY_SHARE = 0.45            # share of the machine's memory that one update may take


def _physical_memory_gb() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (AttributeError, OSError, ValueError):
        return None


def training_memory_gb(cfg: Any, k_max: int, batch_size: int) -> float:
    """Rough peak memory of training in GB for the default posterior, whose event network is evaluated for every
    path, event and Euler step: about 260 bytes each (float32, with the graph kept for backpropagation), on top of
    1.5 GB. Measured on the US bank example (444 events, 150 steps, 10 paths: 4.2 GB at batch 16, 2.8 GB at 8)."""
    steps = int(cfg.burn_steps) + int(cfg.data.n_steps)
    return 1.5 + 2.6e-7 * int(batch_size) * int(cfg.train.m_paths) * int(k_max) * steps


def choose_batch_size(cfg: Any, k_max: int, memory_gb: Optional[float]) -> Tuple[int, Optional[str]]:
    """The batch size of the config, or the largest smaller one whose training fits into ``MEMORY_SHARE`` of the
    machine's memory; with it, a sentence saying why (``None`` when nothing was changed)."""
    configured = int(cfg.train.batch_size)
    post = cfg.model.posterior
    on_gpu = str(cfg.train.device).startswith("cuda")  # GPU memory is not checked
    if memory_gb is None or on_gpu or post.factory or post.kind != "deepset" or not post.phi_uses_state:
        return configured, None
    budget = MEMORY_SHARE * memory_gb
    for b in [configured] + [b for b in BATCH_SIZES if b < configured]:
        if training_memory_gb(cfg, k_max, b) <= budget or b == BATCH_SIZES[-1]:
            if b == configured:
                return configured, None
            return b, (f"batch size {b} in place of {configured}: with up to {k_max} events per observation, {configured} would take "
                       f"about {training_memory_gb(cfg, k_max, configured):.1f} GB of the {memory_gb:.0f} GB of this machine "
                       f"(set it yourself with --set train.batch_size=...)")
    return configured, None


def _config_path(path: Path) -> str:
    """The path as it is stored in the config: relative to the working directory when the file lies below it."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser(__doc__, parents=[common_parent(config=False)])
    parser.add_argument("data", metavar="DATA", help="file with the event times of the observations (.csv, .txt, .json, .npy, .npz), "
                                                     "anywhere on this machine; a relative path is read from the folder the command is run in")
    parser.add_argument("--T", type=float, default=None, metavar="T", help="length of the observation window (needed unless the file stores T)")
    parser.add_argument("--name", default=None, help="name of the run: results go to outputs/<name>/ (default: the name of the file)")
    parser.add_argument("--steps", type=int, default=None, metavar="N", help="Euler steps on [0, T] (default: 100, the value of the config)")
    parser.add_argument("--diffusion", choices=sorted(DIFFUSIONS), default="cir",
                        help="diffusion: cir, eta * sqrt(Z) of the intensity with eta estimated from the training observations and "
                             "kept fixed (default); cir-learned, the same with eta trained; network, sigma(x, t) of the latent state "
                             "X = log Z; unit, the constant 1 for X")
    parser.add_argument("--eta", type=float, default=None, metavar="ETA",
                        help="with --diffusion cir: the coefficient eta, in place of its estimate from the observations")
    parser.add_argument("--epochs", type=int, default=None, help="training epochs (default: the value of the config, 100)")
    parser.add_argument("--test-fraction", type=float, default=None, metavar="F",
                        help="share of the observations held out for testing when the file does not split them (default 0.25)")
    parser.add_argument("--config", default=CONFIG, metavar="PATH", help="the model and training settings")
    parser.add_argument("--no-evaluate", action="store_true", help="train only")
    parser.add_argument("--dry-run", action="store_true", help="read the data, print the settings and stop")
    args = parser.parse_args(argv)

    from ndim.config import load_config
    from ndim.data import load_or_generate_dataset, read_event_file, suggest_scales

    path = Path(args.data)
    if not path.exists():
        raise CLIError(f"file not found: {path} (a relative path is read from the folder the command is run in, {Path.cwd()})")
    try:
        content = read_event_file(path)
    except (FileNotFoundError, ValueError) as exc:
        raise CLIError(str(exc)) from exc
    if args.T is not None and content.T is not None and abs(args.T - content.T) > 1e-9:
        raise CLIError(f"--T {args.T:g} disagrees with the T = {content.T:g} stored in {path}")
    T = args.T if args.T is not None else content.T
    if T is None:
        raise CLIError(f"the length of the observation window is needed: pass --T (every event time of {path.name} must lie in (0, T])")
    if T <= 0:
        raise CLIError("--T must be positive")
    if args.eta is not None and (args.diffusion != "cir" or not args.eta > 0):
        raise CLIError("--eta is the positive coefficient of the fixed CIR diffusion: it goes with --diffusion cir")
    name = args.name or re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "observations"
    config = args.config if Path(args.config).exists() else str(REPO_ROOT / args.config)
    profile = resolve_profile(args.profile)
    user = collect_overrides(args)

    fixed = ["data.source=file", f"data.path={json.dumps(_config_path(path))}", f"data.T={float(T)!r}", f"name={json.dumps(name)}",
             f"model.diffusion={DIFFUSIONS[args.diffusion]}"]
    if args.eta is not None:
        fixed.append(f"model.diffusion.eta_init={float(args.eta)!r}")
    if args.steps is not None:
        fixed.append(f"data.n_steps={int(args.steps)}")
    if args.epochs is not None:
        fixed.append(f"train.epochs={int(args.epochs)}")
    if args.test_fraction is not None:
        fixed.append(f"data.test_fraction={float(args.test_fraction)!r}")
    try:
        first = load_config(config, profile=profile, overrides=fixed + user)
        dataset, _ = load_or_generate_dataset(first)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise CLIError(str(exc)) from exc
    train_data, test_data = dataset.train, dataset.test
    k_max = max(train_data.K_max, test_data.K_max)

    # what depends on the data; the user's own --set come last and win
    from_data = [f"{key}={value!r}" for key, value in suggest_scales(train_data).items()]
    notes: List[str] = []
    cfg = load_config(config, profile=profile, overrides=fixed + from_data + user)
    if not any(o.split("=", 1)[0].strip() == "train.batch_size" for o in user):
        batch, why = choose_batch_size(cfg, k_max, _physical_memory_gb())
        if why:
            from_data.append(f"train.batch_size={batch}")
            notes.append(why[0].upper() + why[1:] + ".")
            cfg = load_config(config, profile=profile, overrides=fixed + from_data + user)
            memory = _physical_memory_gb()
            if memory is not None and training_memory_gb(cfg, k_max, batch) > MEMORY_SHARE * memory:
                notes.append(f"Even at batch size {batch} training is estimated at {training_memory_gb(cfg, k_max, batch):.1f} GB. Keep fewer "
                             "events per observation (thin them) or use --set model.posterior.phi_uses_state=false (docs/OWN_DATA.md, Cost).")

    # the coefficient of the CIR diffusion, unless it was given
    from ndim.cir_eta import resolve_eta

    if not args.dry_run:
        for stale in ("eta_estimate.json", "eta_estimate.png"):            # of an earlier run of this name
            (cfg.output_dir / stale).unlink(missing_ok=True)
    estimate = resolve_eta(cfg, train_data, None if args.dry_run else cfg.output_dir)
    diff = cfg.model.diffusion
    if diff.kind == "cir" and not diff.trainable:
        diffusion_text = f"eta * sqrt(Z) of the intensity with eta = {diff.eta_init:g}, " + (
            "estimated from the training observations" if estimate is not None else "as given") + " and not trained"
    else:
        diffusion_text = DIFFUSION_TEXT.get(args.diffusion, args.diffusion)

    counts = train_data.counts.double()
    n_batches = -(-len(train_data) // int(cfg.train.batch_size))
    burn = cfg.burn_steps * cfg.dt
    print(f"observations: {path} | {len(train_data) + len(test_data)} realisations on [0, {T:g}]: {len(train_data)} for training, {len(test_data)} for testing")
    print(f"events per observation: mean {counts.mean():.1f}, from {int(counts.min())} to {int(counts.max())} (training observations)")
    print(f"model: X = log Z, drift network b(x, t), diffusion: {diffusion_text} | Euler steps: {cfg.data.n_steps} (dt = {cfg.dt:g})"
          + (f" | burn-in of {burn:g} before the window" if cfg.burn_steps else " | no burn-in"))
    if estimate is not None:
        print(f"diffusion coefficient: {estimate.summary()}")
    print(f"scales from the data: intensity {cfg.model.intensity_scale:g} per unit time, {cfg.model.posterior.count_scale:g} events per observation")
    print(f"training: {cfg.train.epochs} epochs, batch size {cfg.train.batch_size}, {n_batches * len(cfg.train.horizons)} updates per epoch | device {cfg.train.device}")
    for note in notes:
        print(f"note: {note}")
    print(f"results: {cfg.output_dir.resolve()}")
    sys.stdout.flush()
    if args.dry_run:
        return 0

    from ndim.train import train
    from ndim.utils import timer
    from ndim.warm_start import WarmStartError

    try:
        with timer() as t:
            result = train(cfg)
    except WarmStartError as e:
        print(f"BROKE DOWN: {e}", file=sys.stderr)
        return 1
    cfg = result.model.cfg  # the config as trained, with x_start resolved
    dead = bool(result.broke_down_at) and not math.isfinite(result.best_test_elbo)       # broke down before any state was worth keeping
    print(f"trained {len(result.history)} epochs in {t['seconds'] / 60:.1f} min | best test ELBO {result.best_test_elbo:.3f} at epoch {result.best_epoch}"
          + (" | INTERRUPTED" if result.interrupted else "")
          + (f" | BROKE DOWN at epoch {result.broke_down_at}: "
             + ("no state had a finite test ELBO, nothing is evaluated" if dead else f"the state of epoch {result.best_epoch} is evaluated")
             if result.broke_down_at else ""))
    sys.stdout.flush()
    metrics = None
    if not args.no_evaluate and not dead:
        from ndim.evaluate import evaluate_run

        metrics = evaluate_run(result.output_dir, profile=profile, overrides=eval_only_overrides(collect_overrides(args, include_device=False)) or None,
                               ckpt="best", device=args.device)
    from ndim.report import FIGURES, write_results

    summary = write_results(result.output_dir, notes)
    print(f"\nsummary: {summary.resolve()}")
    text = summary.read_text()
    if "## Remarks" in text:
        for line in text.split("## Remarks", 1)[1].split("\n## ", 1)[0].strip().splitlines():
            if line.startswith("- "):
                print(f"  remark: {line[2:]}")
    if metrics is not None:
        eval_dir = Path(result.output_dir) / "eval"
        for fig, text in FIGURES[:4]:
            if (eval_dir / fig).exists():
                print(f"  {eval_dir / fig}: {text.split(':')[0].split(' (')[0]}")
        errors = metrics.get("errors") or []
        if errors:
            print(f"  {len(errors)} evaluation stage(s) failed: " + ", ".join(str(e.get('stage')) for e in errors))
            return 1
    return 1 if result.broke_down_at else 0


if __name__ == "__main__":
    run_main(main)
