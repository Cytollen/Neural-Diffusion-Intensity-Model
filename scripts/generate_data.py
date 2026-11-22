#!/usr/bin/env python3
"""Generate (or load from the cache) the synthetic Cox-process dataset of a config and print its statistics.

The data-generating process is the paper's CIR intensity ``dZ = kappa (theta - Z) dt - c t dt + eta sqrt(Z) dB``
(``data.truth``) observed through a Poisson process with intensity ``Z_t`` on ``[0, T]`` (paper Sec. 4.1,
App. G Alg. 2: Euler paths on the grid + exact thinning). The dataset is cached as
``<data.cache_dir>/cir_..._ntr<n_train>_nte<n_test>_seed<seed>.npz`` and shared by every run with the same
``data`` section; training re-uses it automatically.

Prints per split: number of observations, event-count mean / std / min / max, ``K_max`` (padded width) and
the range of the latent intensity paths, and saves ``<data.cache_dir>/<name>_summary.png`` with the mean +/- 1 std
band of the cumulative counts ``N(t)`` of the training (blue) vs test (red) observations (the true-vs-true version
of paper Fig. 3, a quick sanity check that both splits come from the same process).

Examples
  python scripts/generate_data.py --config configs/intensity/cir_mlp.yaml
  python scripts/generate_data.py --config configs/state_space/cir_burn_in.yaml --force
  python scripts/generate_data.py --config configs/smoke.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

try:
    from _cli import common_parent, load_cfg_from_args, make_parser, run_main
except ImportError:  # launched as ``python -m scripts.generate_data``
    from scripts._cli import common_parent, load_cfg_from_args, make_parser, run_main  # type: ignore


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser(__doc__, parents=[common_parent(device=False)])
    parser.add_argument("--force", action="store_true", help="regenerate the dataset even if the cached .npz exists")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help="where to save the summary figure (default: <data.cache_dir>/<name>_summary.png)")
    parser.add_argument("--no-plot", action="store_true", help="skip the summary figure")
    args = parser.parse_args(argv)
    cfg = load_cfg_from_args(args, include_device=False)

    from ndim.data import dataset_filename, describe_truth_z0, estimate_initial_intensity, load_or_generate_dataset
    from ndim.plots import plot_cumulative_counts
    from ndim.utils import timer

    synthetic = cfg.data.source == "synthetic"
    cache_path = Path(cfg.data.cache_dir) / dataset_filename(cfg.data)
    existed = cache_path.exists()
    with timer() as t:
        ds, path = load_or_generate_dataset(cfg, force=args.force)
    action = "read from the file" if not synthetic else "regenerated" if (args.force or not existed) else "loaded from cache"
    tr = cfg.data.truth
    print(f"config: {args.config} | name: {cfg.name}")
    if synthetic:
        print(f"truth: dZ = {tr.kappa:g} ({tr.theta:g} - Z) dt"
              + f" + {tr.eta:g} sqrt(Z) dB, {describe_truth_z0(tr)}, T = {cfg.data.T:g}, n_steps = {cfg.data.n_steps} (dt = {cfg.dt:g}), "
              f"seed = {cfg.data.seed}, substeps = {cfg.data.sim_substeps}")
    else:
        print(f"observations of an unspecified model, T = {cfg.data.T:g}, n_steps = {cfg.data.n_steps} (dt = {cfg.dt:g})")
    print(f"dataset: {path.resolve()} ({action} in {t['seconds']:.2f}s)")
    for split, ed, paths in (("train", ds.train, ds.train_paths), ("test", ds.test, ds.test_paths)):
        counts = ed.counts.numpy()
        if counts.size == 0:
            print(f"  {split:5s}: n=0")
            continue
        line = (f"  {split:5s}: n={len(ed):4d} | events per observation: mean {counts.mean():7.2f}  std {counts.std():6.2f}  "
                f"min {int(counts.min()):4d}  max {int(counts.max()):4d} | K_max={ed.K_max}")
        if paths is not None:
            p = paths.numpy()
            line += f" | latent intensity: mean {p.mean():.2f}  max {p.max():.2f}  final-time mean {p[:, -1].mean():.2f}"
        print(line)
    if ds.has_truth:
        expected = float(ds.train_paths.sum(dim=1).mean() * cfg.dt) if len(ds.train) else float("nan")
        print(f"  E[N(T)] from the training paths (sum_k Z_k dt): {expected:.2f} vs observed mean {ds.train.counts.float().mean():.2f}")
    window = float(cfg.model.z0_window)
    print(f"  mean event rate of the training observations over the first {window:g} of [0, T]: "
          f"{estimate_initial_intensity(ds.train.events, cfg.data.T, window):.2f}"
          + (f" (the truth: {describe_truth_z0(tr)})" if synthetic else " (the estimate of the initial intensity, model.z0: estimate)"))

    if args.no_plot:
        return 0
    out = Path(args.out) if args.out else Path(cfg.data.cache_dir) / f"{cfg.name}_summary.png"
    train_lists, test_lists = ds.train.as_lists(), ds.test.as_lists()
    title = f"{cfg.name}: cumulative counts, train (n={len(ds.train)}) vs test (n={len(ds.test)})"
    plot_cumulative_counts(train_lists, test_lists, float(cfg.data.T), path=out, title=title,
                           true_label="train samples", generated_label="test samples")
    print(f"summary figure: {out.resolve()}")
    return 0


if __name__ == "__main__":
    run_main(main)
