#!/usr/bin/env python3
"""Train a Neural Diffusion Intensity Model from a YAML config (paper Sec. 3, Algorithm in App. F) and optionally evaluate it.

The run writes into ``<output_root>/<name>/``: config.yaml, train.log, history.{json,csv},
training_curves.png (train vs test ELBO / log-lik / KL), drift_snapshot.png, param_history.png,
ckpt_best.pt (best test ELBO) and ckpt_last.pt. With --evaluate the evaluation suite
(ndim.evaluate.evaluate_run) runs on the finished run and writes into ``<output_root>/<name>/eval/``.

Examples
  python scripts/train.py --config configs/intensity/cir_mlp.yaml --profile laptop
  python scripts/train.py --config configs/state_space/cir_burn_in.yaml --profile gpu --evaluate
  python scripts/train.py --config configs/smoke.yaml --set name=smoke_cli train.epochs=3
  python scripts/train.py --config configs/intensity/cir_mlp.yaml --resume outputs/intensity_cir_mlp/ckpt_last.pt --set train.epochs=150
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from _cli import (CLIError, EVAL_STAGES, collect_overrides, common_parent, eval_only_overrides,
                      load_cfg_from_args, make_parser, print_metrics_summary, resolve_profile, run_main)
except ImportError:  # launched as ``python -m scripts.train``
    from scripts._cli import (CLIError, EVAL_STAGES, collect_overrides, common_parent, eval_only_overrides,  # type: ignore
                              load_cfg_from_args, make_parser, print_metrics_summary, resolve_profile, run_main)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser(__doc__, parents=[common_parent()])
    parser.add_argument("--resume", metavar="PATH", default=None,
                        help="checkpoint to resume from (e.g. outputs/<name>/ckpt_last.pt); training continues from its epoch "
                             "up to train.epochs (raise it with --set train.epochs=...)")
    parser.add_argument("--evaluate", action="store_true",
                        help="run ndim.evaluate.evaluate_run on the output directory when training finishes")
    parser.add_argument("--eval-which", nargs="+", choices=EVAL_STAGES, default=None, metavar="STAGE",
                        help=f"evaluation stages for --evaluate (default: all). Choices: {', '.join(EVAL_STAGES)}")
    parser.add_argument("--eval-ckpt", choices=("best", "last"), default="best", help="checkpoint used by --evaluate")
    args = parser.parse_args(argv)

    cfg = load_cfg_from_args(args)
    resume = Path(args.resume) if args.resume else None
    if resume is not None and not resume.exists():
        raise CLIError(f"--resume checkpoint not found: {resume}")

    from ndim.train import train
    from ndim.utils import timer
    from ndim.warm_start import WarmStartError

    print(f"config: {args.config} | profile: {args.profile or '-'} | name: {cfg.name} | device: {cfg.train.device} "
          f"| drift: {cfg.model.drift.kind} | posterior: {cfg.model.posterior.kind} | eta trainable: {cfg.model.diffusion.trainable}")
    print(f"output_dir: {cfg.output_dir.resolve()}")
    sys.stdout.flush()

    try:
        with timer() as t:
            res = train(cfg, resume_from=resume)
    except WarmStartError as e:
        print(f"BROKE DOWN: {e}", file=sys.stderr)
        return 1

    scalars = res.model.named_scalars()
    line = (f"done: {len(res.history)} epochs{' (resumed)' if resume is not None else ''} in {t['seconds']:.1f}s | best test ELBO "
            f"{res.best_test_elbo:.3f} at epoch {res.best_epoch}")
    if res.history and "test_loglik" in res.history[-1]:
        last = res.history[-1]
        line += f" | last test loglik {last['test_loglik']:.3f} kl {last['test_kl']:.3f}"
    if scalars:
        line += " | " + " ".join(f"{k}={v:.4g}" for k, v in scalars.items())
    if res.interrupted:
        line += " | INTERRUPTED"
    dead = bool(res.broke_down_at) and not math.isfinite(res.best_test_elbo)        # broke down before any state was worth keeping
    if res.broke_down_at:
        line += f" | BROKE DOWN at epoch {res.broke_down_at} (see the log)" + (": no state had a finite test ELBO, nothing is evaluated" if dead else "")
    print(f"output_dir: {res.output_dir.resolve()}")
    print(line)
    sys.stdout.flush()

    if args.evaluate and not dead:
        from ndim.evaluate import evaluate_run

        overrides = eval_only_overrides(collect_overrides(args, include_device=False))
        metrics = evaluate_run(
            res.output_dir,
            profile=resolve_profile(args.profile),
            overrides=overrides or None,
            ckpt=args.eval_ckpt,
            which=args.eval_which,
            device=args.device,
        )
        print_metrics_summary(metrics, res.output_dir / "eval")
        from ndim.report import write_results

        summary = write_results(res.output_dir)     # observations from a file: a summary of the run and its figures
        if summary is not None:
            print(f"summary: {summary.resolve()}")
    return 1 if res.broke_down_at else 0


if __name__ == "__main__":
    run_main(main)
