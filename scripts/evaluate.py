#!/usr/bin/env python3
"""Evaluate a finished training run: compare the trained model with the truth, from a checkpoint.

Runs ``ndim.evaluate.evaluate_run`` on ``<run>/config.yaml`` + ``<run>/ckpt_{best,last}.pt`` and writes into
``<run>/eval/`` (or --out): training curves, learned against true drift as slices / heat-map / 3-D html (paper
Fig. 2), cumulative counts of generated against true observations (Fig. 3), the learned against the true
distribution of the initial intensity (with a burn-in), prior / reference / learned posterior bands for the test
observations with the fewest and most events, on the full and on half of the window, with the intensity path that
generated the observation (Fig. 4a/b), Wasserstein distances between the learned and the reference posterior, the
expected path deviation, and ``metrics.json`` with everything (plus ``eval.log``).

For a run on observations read from a file (no truth) the stages compare the model with the observations instead
(``learned_model.png``, MCMC under the learned prior) and ``<run>/RESULTS.md`` is written again.

Stages (--which): curves drift l2 samples initial posterior wasserstein (default: all). A stage that fails is
logged, recorded under ``metrics.json["errors"]`` and does not abort the others; the exit code is then 1.
A profile / --set only touch the ``eval`` section and ``train.device`` / ``train.num_threads`` -- the model is
fixed by the checkpoint. Useful knobs: ``--set eval.mcmc.n_steps=1000 eval.n_paths_posterior=32``.

Examples
  python scripts/evaluate.py --run outputs/intensity_cir_mlp --profile laptop
  python scripts/evaluate.py --run outputs/state_space_cir_burn_in --ckpt last --which curves drift initial
  python scripts/evaluate.py --run outputs/smoke --set eval.mcmc.n_steps=20 --out outputs/smoke/eval_quick
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from _cli import (CLIError, EVAL_STAGES, checkpoint_path, collect_overrides, common_parent, eval_only_overrides,
                      make_parser, print_metrics_summary, resolve_profile, run_main)
except ImportError:  # launched as ``python -m scripts.evaluate``
    from scripts._cli import (CLIError, EVAL_STAGES, checkpoint_path, collect_overrides, common_parent,  # type: ignore
                              eval_only_overrides, make_parser, print_metrics_summary, resolve_profile, run_main)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser(__doc__, parents=[common_parent(config=False)])
    parser.add_argument("--run", required=True, metavar="DIR", help="training output directory (contains config.yaml and ckpt_*.pt)")
    parser.add_argument("--ckpt", choices=("best", "last"), default="best", help="which checkpoint to evaluate")
    parser.add_argument("--which", nargs="+", choices=EVAL_STAGES, default=None, metavar="STAGE",
                        help=f"stages to run (default: all). Choices: {', '.join(EVAL_STAGES)}")
    parser.add_argument("--out", metavar="DIR", default=None, help="output directory (default: <run>/eval)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if not (run_dir / "config.yaml").exists():
        raise CLIError(f"{run_dir / 'config.yaml'} not found: --run must be a finished scripts/train.py output directory")
    ckpt = checkpoint_path(run_dir, args.ckpt)
    out_dir = Path(args.out) if args.out else run_dir / "eval"

    all_overrides = collect_overrides(args, include_device=False)
    overrides = eval_only_overrides(all_overrides)
    dropped = [o for o in all_overrides if o not in overrides]
    if dropped:
        print(f"warning: ignoring overrides that do not belong to the eval section (the model is fixed by the checkpoint): {dropped}",
              file=sys.stderr)

    from ndim.evaluate import evaluate_run

    print(f"run: {run_dir.resolve()} | checkpoint: {ckpt.name} | stages: {' '.join(args.which) if args.which else 'all'} | "
          f"profile: {args.profile or '-'} | device: {args.device or 'config'}")
    print(f"output_dir: {out_dir.resolve()}")
    sys.stdout.flush()

    metrics = evaluate_run(
        run_dir,
        out_dir=out_dir,
        profile=resolve_profile(args.profile),
        overrides=overrides or None,
        ckpt=args.ckpt,
        which=args.which,
        device=args.device,
    )
    print_metrics_summary(metrics, out_dir)
    if out_dir.resolve() == (run_dir / "eval").resolve():
        from ndim.report import write_results

        summary = write_results(run_dir)            # observations from a file: RESULTS.md follows the evaluation
        if summary is not None:
            print(f"summary: {summary.resolve()}")
    errors = metrics.get("errors") if isinstance(metrics, dict) else None
    return 1 if errors else 0


if __name__ == "__main__":
    run_main(main)
