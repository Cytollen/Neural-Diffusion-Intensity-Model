"""Shared command-line plumbing for the ``scripts/`` entry points.

Importing this module puts the repository root on ``sys.path`` (so ``import ndim`` works no matter
where a script is launched from) and provides:

* :func:`common_parent` -- an ``argparse`` parent parser with the common flags
  ``--config PATH``, ``--profile {laptop,gpu,PATH}``, ``--set KEY=VALUE ...`` (repeatable) and ``--device``;
* :func:`resolve_profile` -- ``'laptop'`` / ``'gpu'`` -> ``configs/laptop.yaml`` / ``configs/gpu.yaml``;
* :func:`load_cfg_from_args` -- build an :class:`ndim.config.ExperimentConfig` from the parsed flags;
* :func:`print_metrics_summary`, a compact view of an evaluation, and :func:`run_main`.

Usage in a script::

    from _cli import common_parent, load_cfg_from_args, make_parser, run_main

    def main(argv=None) -> int:
        parser = make_parser(__doc__, parents=[common_parent()])
        ...
        return 0

    if __name__ == "__main__":
        run_main(main)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"
PROFILE_ALIASES: Dict[str, Path] = {"laptop": CONFIG_DIR / "laptop.yaml", "gpu": CONFIG_DIR / "gpu.yaml"}
EVAL_STAGES: Tuple[str, ...] = ("curves", "drift", "l2", "samples", "initial", "posterior", "wasserstein")
DEVICE_CHOICES: Tuple[str, ...] = ("cpu", "mps", "cuda", "auto")


class CLIError(Exception):
    """User-facing error. :func:`run_main` prints it as ``error: ...`` (no traceback) and exits with code 2."""


class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults and keep the docstring's line breaks in ``--help``."""


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------
def common_parent(
    config: bool = True,
    config_required: bool = True,
    profile: bool = True,
    overrides: bool = True,
    device: bool = True,
    device_help: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Parent parser holding the flags shared by every script."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("configuration")
    if config:
        g.add_argument(
            "--config", required=config_required, default=None, metavar="PATH",
            help="experiment YAML, e.g. configs/intensity/cir_mlp.yaml ('extends' chains are honoured; bare names resolve in configs/)",
        )
    if profile:
        g.add_argument(
            "--profile", default=None, metavar="{laptop,gpu,PATH}",
            help="hardware profile merged on top of the config: 'laptop' (configs/laptop.yaml), 'gpu' (configs/gpu.yaml) or a YAML path",
        )
    if overrides:
        g.add_argument(
            "--set", dest="overrides", action="append", nargs="+", default=[], metavar="KEY=VALUE",
            help="override config entries with dotted keys and YAML values, e.g. --set train.epochs=5 name=myrun "
                 "(repeatable; applied after the profile)",
        )
    if device:
        g.add_argument(
            "--device", default=None, metavar="DEV",
            help=device_help or "torch device: cpu, mps, cuda or auto (sets train.device; default: value in the config)",
        )
    return p


def make_parser(doc: Optional[str], parents: Sequence[argparse.ArgumentParser] = (), prog: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser whose description is the first paragraph of ``doc`` and whose epilog is the rest."""
    text = (doc or "").strip()
    desc, _, epilog = text.partition("\n\n")
    return argparse.ArgumentParser(prog=prog, description=desc, epilog=epilog or None, parents=list(parents), formatter_class=_Formatter)


# --------------------------------------------------------------------------------------
# Profiles / overrides / configs
# --------------------------------------------------------------------------------------
def resolve_profile(profile: Optional[str]) -> Optional[str]:
    """``None``/``''``/``'none'`` -> ``None``; ``'laptop'``/``'gpu'`` -> the YAML under ``configs/``; else a path."""
    if profile is None:
        return None
    key = str(profile).strip()
    if key == "" or key.lower() == "none":
        return None
    if key.lower() in PROFILE_ALIASES:
        return str(PROFILE_ALIASES[key.lower()])
    p = Path(key)
    if p.exists():
        return str(p)
    if (CONFIG_DIR / key).exists():
        return str(CONFIG_DIR / key)
    if (CONFIG_DIR / f"{key}.yaml").exists():
        return str(CONFIG_DIR / f"{key}.yaml")
    raise CLIError(f"unknown profile '{profile}': use 'laptop', 'gpu' or the path of a YAML file")


def flatten_overrides(sets: Optional[Iterable[Any]]) -> List[str]:
    """``--set a=1 b=2 --set c=3`` is parsed as ``[["a=1", "b=2"], ["c=3"]]``; flatten and validate."""
    out: List[str] = []
    for item in sets or []:
        parts = item if isinstance(item, (list, tuple)) else [item]
        for s in parts:
            s = str(s).strip()
            if not s:
                continue
            if "=" not in s:
                raise CLIError(f"--set expects KEY=VALUE pairs, got '{s}'")
            out.append(s)
    return out


def collect_overrides(args: argparse.Namespace, include_device: bool = True, extra: Optional[Sequence[str]] = None) -> List[str]:
    """All ``--set`` overrides (+ ``train.device=<--device>`` when requested and given, + ``extra``)."""
    out = flatten_overrides(getattr(args, "overrides", None))
    dev = getattr(args, "device", None)
    if include_device and dev:
        out.append(f"train.device={dev}")
    out.extend(extra or [])
    return out


def load_cfg_from_args(args: argparse.Namespace, extra_overrides: Optional[Sequence[str]] = None, include_device: bool = True):
    """``load_config(--config, profile=--profile, overrides=--set [+ device] [+ extra])`` with friendly errors."""
    from ndim.config import load_config

    config = getattr(args, "config", None)
    if not config:
        raise CLIError("--config is required")
    overrides = collect_overrides(args, include_device=include_device, extra=extra_overrides)
    try:
        return load_config(config, profile=resolve_profile(getattr(args, "profile", None)), overrides=overrides or None)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        raise CLIError(f"could not load config '{config}': {e}") from e


def eval_only_overrides(overrides: Sequence[str]) -> List[str]:
    """Keep the overrides that make sense for ``evaluate_run`` (``eval.*``, ``train.device``, ``train.num_threads``)."""
    keep = ("eval.", "train.device", "train.num_threads")
    return [o for o in overrides if o.split("=", 1)[0].strip().startswith(keep)]


def checkpoint_path(run_dir: Path, ckpt: str) -> Path:
    p = Path(run_dir) / f"ckpt_{ckpt}.pt"
    if not p.exists():
        raise CLIError(f"checkpoint {p} not found (available: {sorted(q.name for q in Path(run_dir).glob('ckpt_*.pt'))})")
    return p


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------
def fmt_value(v: Any, spec: str = "{:.4g}") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return spec.format(v)
    return str(v)


def flatten_scalars(d: Dict[str, Any], prefix: str = "", skip: Sequence[str] = ("config",), max_depth: int = 3) -> Dict[str, Any]:
    """Flatten nested dicts of scalars into dotted keys (lists and non-scalars are dropped)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k in skip:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            if max_depth > 0:
                out.update(flatten_scalars(v, key + ".", skip=(), max_depth=max_depth - 1))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def print_metrics_summary(metrics: Dict[str, Any], out_dir: Optional[Path] = None, max_lines: int = 60) -> None:
    """Print a compact view of an ``evaluate_run`` result: scalar entries, then recorded errors."""
    if not isinstance(metrics, dict):
        print(f"metrics: {metrics!r}")
        return
    flat = flatten_scalars(metrics)
    if out_dir is not None:
        print(f"evaluation outputs: {Path(out_dir).resolve()}")
    for i, (k, v) in enumerate(flat.items()):
        if i >= max_lines:
            print(f"  ... ({len(flat) - max_lines} more entries in metrics.json)")
            break
        print(f"  {k}: {fmt_value(v)}")
    errors = metrics.get("errors")
    if errors:
        print(f"WARNING: {len(errors)} evaluation stage(s) failed:")
        for e in (errors if isinstance(errors, (list, tuple)) else [errors]):
            if isinstance(e, dict):
                print(f"  - {e.get('stage', '?')}: {e.get('error', e)}")
            else:
                print(f"  - {e}")


def run_main(main: Callable[[Optional[Sequence[str]]], int]) -> None:
    """Run ``main(None)`` and exit with its return code (2 on :class:`CLIError`, 130 on Ctrl-C)."""
    try:
        code = main(None)
    except CLIError as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        code = 130
    sys.exit(int(code or 0))
