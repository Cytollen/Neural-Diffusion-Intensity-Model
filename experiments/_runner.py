"""Shared plumbing of the experiment wrappers in this folder.

A wrapper runs one example end to end by calling the generic scripts of ``scripts/`` in order: the dataset, then
training followed by the evaluation of the trained model. It adds nothing to the method. Every stage is a single
command, printed before it runs, that can also be launched on its own from the repository root.

Flags shared by all wrappers::

    --profile {laptop,gpu,PATH}   hardware profile merged on top of the config (default: the config as written)
    --smoke                       tiny end-to-end pass: sizes of configs/tiny.yaml, results under outputs/smoke/,
                                  posterior control bounded (see SMOKE_OVERRIDES)
    --only STAGE [STAGE ...]      run only these stages
    --skip STAGE [STAGE ...]      run everything except these stages
    --set KEY=VALUE [...]         config overrides passed to every stage (same syntax as the scripts)
    --dry-run                     print the commands without running them

A failing stage is reported and the wrapper continues with the next one; the exit code is the number of
failed stages.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: settings of the ``--smoke`` pass (seconds per stage instead of minutes or hours)
SMOKE_PROFILE = "tiny"
SMOKE_OUTPUT_ROOT = "outputs/smoke"
# the smoke pass is too short for training to settle: bounding the control keeps its paths finite
SMOKE_OVERRIDES = ["train.max_abs_control=30"]


@dataclass
class Stage:
    """One step of an experiment: a script of ``scripts/`` with its arguments."""
    key: str                  # short name used by --only / --skip
    title: str                # what the stage produces
    command: List[str]        # script path relative to the repository root, then its arguments


class Experiment:
    """Builds the stages of an experiment from its configs and the wrapper's command-line flags."""

    def __init__(self, profile: Optional[str] = None, smoke: bool = False, overrides: Sequence[str] = ()) -> None:
        self.smoke = bool(smoke)
        self.profile = SMOKE_PROFILE if self.smoke else profile
        self.overrides = ([f"output_root={SMOKE_OUTPUT_ROOT}", *SMOKE_OVERRIDES] if self.smoke else []) + list(overrides)

    # ------------------------------------------------------------------ shared argument groups
    def _profile_args(self) -> List[str]:
        return ["--profile", self.profile] if self.profile else []

    def _set_args(self) -> List[str]:
        return ["--set", *self.overrides] if self.overrides else []

    # ------------------------------------------------------------------ stages
    def data(self, config: str, key: str = "data", title: str = "generate (or reuse) the synthetic dataset") -> Stage:
        return Stage(key, title, ["scripts/generate_data.py", "--config", config, *self._profile_args(), *self._set_args()])

    def train(self, config: str, key: str = "train", title: str = "train, then evaluate the trained model") -> Stage:
        return Stage(key, title, ["scripts/train.py", "--config", config, *self._profile_args(), "--evaluate", *self._set_args()])


def _select(stages: Sequence[Stage], only: Sequence[str], skip: Sequence[str]) -> List[Stage]:
    keys = [s.key for s in stages]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate stage keys: {keys}")
    unknown = [k for k in [*only, *skip] if k not in keys]
    if unknown:
        raise SystemExit(f"error: unknown stage(s) {unknown}; this experiment has {keys}")
    return [s for s in stages if (not only or s.key in only) and s.key not in skip]


def run(doc: Optional[str], build: Callable[[Experiment], List[Stage]], argv: Optional[Sequence[str]] = None) -> int:
    """Parse the common flags, build the stages with ``build`` and run them in order."""
    text = (doc or "").strip()
    description, _, epilog = text.partition("\n\n")
    parser = argparse.ArgumentParser(description=description, epilog=epilog or None, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None, metavar="{laptop,gpu,PATH}",
                        help="hardware profile merged on top of the config (default: the config as written)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"tiny end-to-end pass: sizes of configs/{SMOKE_PROFILE}.yaml, results under {SMOKE_OUTPUT_ROOT}/")
    parser.add_argument("--only", nargs="+", default=[], metavar="STAGE", help="run only these stages")
    parser.add_argument("--skip", nargs="+", default=[], metavar="STAGE", help="run everything except these stages")
    parser.add_argument("--set", dest="overrides", action="append", nargs="+", default=[], metavar="KEY=VALUE",
                        help="config overrides passed to every stage, e.g. --set train.epochs=20")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args(argv)

    overrides = [item for group in args.overrides for item in group]
    experiment = Experiment(profile=args.profile, smoke=args.smoke, overrides=overrides)
    stages = _select(build(experiment), args.only, args.skip)

    failed: List[str] = []
    t_start = time.time()
    for i, stage in enumerate(stages, 1):
        command = [sys.executable, *stage.command]
        print(f"\n[{i}/{len(stages)}] {stage.key}: {stage.title}\n    {shlex.join(['python', *stage.command])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        code = subprocess.call(command, cwd=str(REPO_ROOT))
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"[{i}/{len(stages)}] {stage.key}: {status} in {time.time() - t0:.0f}s", flush=True)
        if code != 0:
            failed.append(stage.key)
    if not args.dry_run:
        print(f"\nfinished {len(stages) - len(failed)}/{len(stages)} stages in {(time.time() - t_start) / 60:.1f} min"
              + (f"; failed: {', '.join(failed)}" if failed else ""), flush=True)
    return len(failed)
