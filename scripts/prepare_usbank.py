#!/usr/bin/env python3
"""Prepare the US bank call-centre data: one observation of the arrival process per day.

The raw file lists the number of calls that arrived in every minute, as three columns separated by blanks and
without a header: ``calls date minute`` (date as YYYY-MM-DD, minute 0 to 1439 of the day). It is not distributed
with this repository: pass its location with --csv.

Every complete day of the chosen weekday becomes one observation on [0, T], with T in hours:

  1. the calls of a minute are placed evenly inside that minute (the k-th of c calls at k/c of the minute);
  2. the arrivals are thinned independently with probability --thinning. A thinned Cox process is a Cox process
     whose intensity is multiplied by that probability; 0.001 keeps about 290 of the 290,000 calls of a day.

The result is an .npz with ``events`` (one row per day in order of date, padded with nan), ``T``, the dates and the
settings. The days are split into training and test observations when the file is loaded (data.test_fraction,
data.seed), so the file can be passed to scripts/fit.py as it is.

The script also estimates the coefficient eta of the diffusion eta * sqrt(Z) of the intensity from ALL calls of the
training days, where it is far better determined than from the thinned observations, and converts it to the thinned
observations (factor sqrt(thinning)): the value to pass to scripts/fit.py with --eta.

Examples
  python scripts/prepare_usbank.py --csv /path/to/usbank_calls_byminute.csv
  python scripts/fit.py data/usbank_friday.npz --eta 0.322
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from _cli import CLIError, REPO_ROOT, make_parser, run_main
except ImportError:  # launched as ``python -m scripts.prepare_usbank``
    from scripts._cli import CLIError, REPO_ROOT, make_parser, run_main  # type: ignore

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MINUTES_PER_DAY = 1440
TEST_FRACTION, SPLIT_SEED = 0.25, 0          # the split that scripts/fit.py makes by default


def load_day_counts(csv_path: Path, day_of_week: int, start_minute: int, end_minute: int) -> Tuple[List[str], np.ndarray]:
    """Calls per minute of every complete day of one weekday: the dates and a ``[n_days, n_minutes]`` array."""
    per_day: Dict[str, np.ndarray] = {}
    with open(csv_path) as f:
        for k, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                calls, date, minute = float(parts[0]), parts[1], int(float(parts[2]))
            except (IndexError, ValueError) as exc:
                raise CLIError(f"{csv_path}, line {k}: expected three columns 'calls date minute', got '{line.strip()}'") from exc
            if calls < 0:
                raise CLIError(f"{csv_path}, line {k}: negative number of calls")
            if 0 <= minute < MINUTES_PER_DAY:
                per_day.setdefault(date, np.full(MINUTES_PER_DAY, np.nan))[minute] = calls
    dates, rows = [], []
    for date in sorted(per_day):
        try:
            weekday = datetime.date.fromisoformat(date).weekday()
        except ValueError as exc:
            raise CLIError(f"{csv_path}: '{date}' is not a date of the form YYYY-MM-DD") from exc
        window = per_day[date][start_minute:end_minute]
        if weekday == day_of_week and not np.isnan(window).any():          # days with a missing minute are left out
            dates.append(date)
            rows.append(window)
    if not rows:
        raise CLIError(f"{csv_path} has no complete {WEEKDAYS[day_of_week]} between minutes {start_minute} and {end_minute}")
    return dates, np.stack(rows)


def arrival_times(counts: np.ndarray, minutes_per_unit: float = 60.0) -> np.ndarray:
    """Arrival times of one day from its calls per minute: the k-th of ``c`` calls of minute ``j`` at ``j + k / c``
    minutes, k = 1, ..., c, so that every time lies in (0, T]. Returned in units of ``minutes_per_unit`` (hours)."""
    c = np.rint(counts).astype(np.int64)
    minute = np.repeat(np.arange(c.size), c)
    first = np.cumsum(c) - c                                             # index of the first call of every minute
    k = np.arange(int(c.sum())) - np.repeat(first, c) + 1
    return (minute + k / np.repeat(c, c)) / float(minutes_per_unit)


def prepare(csv_path: Path, day_of_week: int = 4, start_minute: int = 0, end_minute: int = MINUTES_PER_DAY,
            thinning: float = 0.001, seed: int = 123, with_counts: bool = False) -> Dict[str, np.ndarray]:
    """The thinned arrival times of every complete day, as the arrays that are saved (see the module docstring).
    ``with_counts`` adds ``calls_per_minute``, the calls of every minute before thinning (``[n_days, n_minutes]``)."""
    if not 0 <= start_minute < end_minute <= MINUTES_PER_DAY:
        raise CLIError("the minutes must satisfy 0 <= start-minute < end-minute <= 1440")
    if not 0.0 < thinning <= 1.0:
        raise CLIError("--thinning is a probability in (0, 1]")
    dates, counts = load_day_counts(csv_path, day_of_week, start_minute, end_minute)
    # the days are thinned in a random order and stored in order of date
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(dates))
    kept: List[Optional[np.ndarray]] = [None] * len(dates)
    for i in order:
        times = arrival_times(counts[i])
        kept[i] = times[rng.random(times.shape[0]) < thinning]
    k_max = max(max(len(e) for e in kept), 1)
    events = np.full((len(dates), k_max), np.nan)
    for i, e in enumerate(kept):
        events[i, : len(e)] = e
    arrays = {"events": events, "T": np.array((end_minute - start_minute) / 60.0), "dates": np.array(dates),
              "calls_per_day": counts.sum(axis=1), "thinning": np.array(float(thinning)), "seed": np.array(int(seed)),
              "day_of_week": np.array(int(day_of_week)), "start_minute": np.array(int(start_minute))}
    if with_counts:
        arrays["calls_per_minute"] = counts
    return arrays


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser(__doc__)
    parser.add_argument("--csv", required=True, metavar="PATH", help="the raw file: calls per minute, columns 'calls date minute'")
    parser.add_argument("--out", default=None, metavar="PATH", help="output .npz (default: data/usbank_<weekday>.npz)")
    parser.add_argument("--day-of-week", type=int, default=4, metavar="D", help="weekday of the observations, Monday = 0")
    parser.add_argument("--start-minute", type=int, default=0, help="first minute of the day that is kept")
    parser.add_argument("--end-minute", type=int, default=MINUTES_PER_DAY, help="end of the window, in minutes of the day")
    parser.add_argument("--thinning", type=float, default=0.001, metavar="P", help="probability with which an arrival is kept")
    parser.add_argument("--seed", type=int, default=123, help="seed of the thinning")
    parser.add_argument("--no-eta", action="store_true", help="do not estimate the coefficient of the diffusion from all calls")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise CLIError(f"--csv not found: {csv_path}")
    if not 0 <= args.day_of_week <= 6:
        raise CLIError("--day-of-week is 0 (Monday) to 6 (Sunday)")
    arrays = prepare(csv_path, args.day_of_week, args.start_minute, args.end_minute, args.thinning, args.seed, with_counts=True)
    calls_per_minute = arrays.pop("calls_per_minute")
    out = Path(args.out) if args.out else REPO_ROOT / "data" / f"usbank_{WEEKDAYS[args.day_of_week]}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)

    events, T, dates = arrays["events"], float(arrays["T"]), arrays["dates"]
    n_events = np.isfinite(events).sum(axis=1)
    hours = np.arange(0.0, T + 1e-9, max(T / 8.0, 1e-9))
    rate = [float(((events > a) & (events <= b)).sum() / len(dates) / (b - a)) for a, b in zip(hours[:-1], hours[1:])]
    print(f"{len(dates)} complete {WEEKDAYS[args.day_of_week]}s from {dates[0]} to {dates[-1]}; window of {T:g} hours")
    print(f"calls per day: mean {arrays['calls_per_day'].mean():,.0f} (from {arrays['calls_per_day'].min():,.0f} to {arrays['calls_per_day'].max():,.0f})")
    print(f"kept with probability {args.thinning:g} (seed {args.seed}): {n_events.mean():.1f} events per day on average "
          f"(from {n_events.min()} to {n_events.max()})")
    print("mean events per hour over eight equal parts of the window: " + ", ".join(f"{r:.1f}" for r in rate))
    print(f"wrote {out}")
    if args.no_eta or len(dates) < 4:
        return 0

    from ndim.cir_eta import estimate_from_counts
    from ndim.data import split_observations
    from ndim.plots import plot_eta_estimate

    train_idx, _ = split_observations(len(dates), TEST_FRACTION, SPLIT_SEED)
    estimate = estimate_from_counts(calls_per_minute[train_idx], T)        # the calls per minute are counts on a fine grid
    scale = float(np.sqrt(args.thinning))
    figure = out.with_name(out.stem + "_eta.png")
    plot_eta_estimate(estimate, figure, unit="hours", title=f"U.S. bank, all calls of the {len(train_idx)} training days: coefficient of the diffusion")
    print(f"coefficient of the diffusion eta * sqrt(Z), from all calls of the {len(train_idx)} training days (time in hours): {estimate.summary()}")
    print(f"for the observations kept with probability {args.thinning:g}: sqrt({args.thinning:g}) x {estimate.eta:.4g} = {scale * estimate.eta:.3f} "
          f"(95% interval {scale * estimate.interval[0]:.3f} to {scale * estimate.interval[1]:.3f}); figure: {figure}")
    print(f"fit: python scripts/fit.py {out} --eta {scale * estimate.eta:.3f}")
    return 0


if __name__ == "__main__":
    run_main(main)
