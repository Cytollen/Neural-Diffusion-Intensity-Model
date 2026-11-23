#!/usr/bin/env python3
"""Prepare the Citi Bike example: the trip starts of every weekday in Jersey City as one observation of a point process.

Data: the monthly trip files ``JC-YYYYMM-citibike-tripdata.csv.zip`` that Citi Bike publishes
(https://citibikenyc.com/system-data, files at https://s3.amazonaws.com/tripdata/index.html). They are used under
the Citi Bike data licence (https://ride.citibikenyc.com/data-sharing-policy), which does not allow passing them on
as a dataset: this repository holds neither the files nor the observations made from them. The script reads them
from --raw, or fetches them first with --download (ten files, about 37 MB, for the default years).

An event is the start of a trip (column ``started_at``, local time). An observation is the set of trip starts of one
day on [0, 24], in hours since midnight. Kept are the weekdays of May to September that are not U.S. federal
holidays, and of their trips each one with probability --keep: a thinned Cox process is a Cox process whose
intensity is multiplied by that probability, and 0.08 leaves about 285 of the 3,500 trips of a day.

The result is an .npz with ``events`` (one row per day in order of date, padded with nan), ``T``, the dates, the
number of trips of every day before thinning and the settings: a file for scripts/fit.py.

The script also estimates the coefficient eta of the diffusion eta * sqrt(Z) of the intensity from ALL trips of the
training days, where it is far better determined than from the thinned observations, and converts it to the thinned
observations (factor sqrt(keep)). scripts/fit.py makes its own estimate from the thinned observations; --eta passes
this one instead.

Examples
  python scripts/prepare_citibike.py --download
  python scripts/prepare_citibike.py --raw /path/to/the/zip/files
  python scripts/fit.py data/citibike_jc_weekdays.npz
"""
from __future__ import annotations

import csv
import datetime
import io
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

try:
    from _cli import CLIError, REPO_ROOT, make_parser, run_main
except ImportError:  # launched as ``python -m scripts.prepare_citibike``
    from scripts._cli import CLIError, REPO_ROOT, make_parser, run_main  # type: ignore

BASE_URL = "https://s3.amazonaws.com/tripdata/"
MONTHS = (5, 6, 7, 8, 9)
YEARS = (2025, 2026)
TEST_FRACTION, SPLIT_SEED = 0.25, 0          # the split that scripts/fit.py makes by default


def file_names(years: Sequence[int]) -> List[str]:
    return [f"JC-{year}{month:02d}-citibike-tripdata.csv.zip" for year in years for month in MONTHS]


def federal_holidays(year: int) -> Set[str]:
    """The U.S. federal holidays between May and September, on the days they are observed (ISO dates)."""
    def observed(day: datetime.date) -> datetime.date:
        return day - datetime.timedelta(days=1) if day.weekday() == 5 else day + datetime.timedelta(days=1) if day.weekday() == 6 else day

    memorial = datetime.date(year, 5, 31)
    memorial -= datetime.timedelta(days=memorial.weekday())                # the last Monday of May
    labor = datetime.date(year, 9, 1)
    labor += datetime.timedelta(days=(7 - labor.weekday()) % 7)            # the first Monday of September
    return {d.isoformat() for d in (memorial, observed(datetime.date(year, 6, 19)), observed(datetime.date(year, 7, 4)), labor)}


def download(names: Sequence[str], raw_dir: Path, base_url: str = BASE_URL) -> None:
    """Fetch the files that are not in ``raw_dir`` yet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = raw_dir / name
        if target.exists():
            continue
        url = base_url.rstrip("/") + "/" + name
        print(f"downloading {url}", flush=True)
        part = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(url) as response, open(part, "wb") as fh:
                shutil.copyfileobj(response, fh)
        except (urllib.error.URLError, OSError) as exc:
            part.unlink(missing_ok=True)
            raise CLIError(f"could not download {url}: {exc}") from exc
        part.rename(target)


def read_start_times(archives: Sequence[Path]) -> Dict[str, np.ndarray]:
    """Start times of all trips in the zip files, by date: seconds since midnight, sorted. The files are read in
    place; nothing is unpacked."""
    days: Dict[str, List[float]] = {}
    for archive in archives:
        n_file = 0
        try:
            zf = zipfile.ZipFile(archive)
        except (zipfile.BadZipFile, OSError) as exc:
            raise CLIError(f"{archive} is not a readable zip file: {exc}") from exc
        with zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv") and not m.startswith("__MACOSX")]
            for member in members:
                with zf.open(member) as fh:
                    reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""))
                    header = next(reader, [])
                    if "started_at" not in header:
                        raise CLIError(f"{archive}: {member} has no column 'started_at'")
                    col = header.index("started_at")
                    for row in reader:
                        try:
                            stamp = row[col]                                # "2025-05-20 15:35:37.858"
                            day, clock = stamp[:10], stamp[11:]
                            datetime.date.fromisoformat(day)
                            h, m, sec = clock.split(":")
                            seconds = int(h) * 3600 + int(m) * 60 + float(sec)
                            if not 0.0 <= seconds < 86400.0:
                                raise ValueError(stamp)
                        except (ValueError, IndexError):
                            continue                                        # a row without a readable start time
                        days.setdefault(day, []).append(round(seconds, 3))
                        n_file += 1
        print(f"{archive.name}: {n_file:,} trips", flush=True)
    return {day: np.sort(np.asarray(days[day], dtype=np.float64)) for day in sorted(days)}


def prepare(days: Dict[str, np.ndarray], keep: float = 0.08, seed: int = 123) -> Dict[str, np.ndarray]:
    """One observation per weekday of May to September that is not a holiday: every trip kept with probability
    ``keep``. The arrays that are saved (see the module docstring)."""
    if not 0.0 < keep <= 1.0:
        raise CLIError("--keep is a probability in (0, 1]")
    holidays = set().union(*(federal_holidays(year) for year in {int(day[:4]) for day in days})) if days else set()
    chosen = [day for day in sorted(days)
              if (d := datetime.date.fromisoformat(day)).month in MONTHS and d.weekday() < 5 and day not in holidays]
    if not chosen:
        raise CLIError("the files hold no weekday between May and September")
    rng = np.random.default_rng(int(seed))
    rows = []
    for day in chosen:
        seconds = days[day]
        hours = np.sort(seconds[rng.random(seconds.shape[0]) < keep]) / 3600.0
        hours[hours <= 0.0] = 1e-6                                          # a trip at midnight sharp belongs to (0, T]
        rows.append(hours)
    events = np.full((len(rows), max(max(len(r) for r in rows), 1)), np.nan)
    for k, r in enumerate(rows):
        events[k, : len(r)] = r
    return {"events": events, "T": np.array(24.0), "dates": np.array(chosen), "n_trips": np.array([len(days[day]) for day in chosen]),
            "keep": np.array(float(keep)), "seed": np.array(int(seed))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser(__doc__)
    parser.add_argument("--raw", default=None, metavar="DIR", help="folder with the monthly zip files (default: data/citibike_raw)")
    parser.add_argument("--download", action="store_true", help="fetch the files that are missing in --raw from Citi Bike first")
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS), metavar="YEAR", help="years whose May to September are used")
    parser.add_argument("--out", default=None, metavar="PATH", help="output .npz (default: data/citibike_jc_weekdays.npz)")
    parser.add_argument("--keep", type=float, default=0.08, metavar="P", help="probability with which a trip is kept")
    parser.add_argument("--seed", type=int, default=123, help="seed of the thinning")
    parser.add_argument("--base-url", default=BASE_URL, help="where --download fetches the files from")
    parser.add_argument("--no-eta", action="store_true", help="do not estimate the coefficient of the diffusion from all trips")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw) if args.raw else REPO_ROOT / "data" / "citibike_raw"
    names = file_names(args.years)
    if args.download:
        download(names, raw_dir, args.base_url)
    missing = [name for name in names if not (raw_dir / name).exists()]
    if missing:
        raise CLIError(f"{len(missing)} of the {len(names)} monthly files are not in {raw_dir} ({', '.join(missing[:3])}"
                       f"{', ...' if len(missing) > 3 else ''}): get them from {BASE_URL}index.html, or run with --download")
    days = read_start_times([raw_dir / name for name in names])
    arrays = prepare(days, args.keep, args.seed)
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "citibike_jc_weekdays.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)

    dates, n_trips = arrays["dates"], arrays["n_trips"]
    counts = np.isfinite(arrays["events"]).sum(axis=1)
    print(f"{len(dates)} weekdays from {dates[0]} to {dates[-1]}; trips per day: {n_trips.mean():,.0f} on average "
          f"(from {n_trips.min():,} to {n_trips.max():,})")
    print(f"kept with probability {args.keep:g} (seed {args.seed}): {counts.mean():.1f} events per day on average (from {counts.min()} to "
          f"{counts.max()}), standard deviation {counts.std(ddof=1):.1f} (a Poisson process with this mean: {np.sqrt(counts.mean()):.1f})")
    print(f"wrote {out}")
    fit_command = f"fit: python scripts/fit.py {out}"
    if args.no_eta or len(dates) < 4:
        print(fit_command)
        return 0

    from ndim.cir_eta import estimate_cir_eta
    from ndim.data import split_observations
    from ndim.plots import plot_eta_estimate

    train_idx, _ = split_observations(len(dates), TEST_FRACTION, SPLIT_SEED)
    estimate = estimate_cir_eta([days[str(dates[i])] / 3600.0 for i in train_idx], 24.0)
    scale = float(np.sqrt(args.keep))
    figure = out.with_name(out.stem + "_eta.png")
    plot_eta_estimate(estimate, figure, unit="hours", title=f"Citi Bike, all trips of the {len(train_idx)} training days: coefficient of the diffusion")
    print(f"coefficient of the diffusion eta * sqrt(Z), from all trips of the {len(train_idx)} training days (time in hours): {estimate.summary()}")
    print(f"for the observations kept with probability {args.keep:g}: sqrt({args.keep:g}) x {estimate.eta:.4g} = {scale * estimate.eta:.3f} "
          f"(95% interval {scale * estimate.interval[0]:.3f} to {scale * estimate.interval[1]:.3f}); figure: {figure}")
    print(f"scripts/fit.py makes its own estimate from the thinned observations; --eta {scale * estimate.eta:.3f} passes this one instead")
    print(fit_command)
    return 0


if __name__ == "__main__":
    run_main(main)
