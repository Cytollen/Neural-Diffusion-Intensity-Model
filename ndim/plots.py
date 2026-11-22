"""Matplotlib figures. Colour conventions follow the paper:

* true prior: black with gray bands; true data samples: blue, learned prior samples: red,
* true posterior reference: blue, learned (VI) posterior: red, prior: black with gray bands,
* learned drift: purple, true drift: goldenrod.

Training-time plots (``plot_training_curves``, ``plot_drift_snapshot``, ``plot_parameter_history``)
live here; the evaluation figures (cumulative counts, posterior comparison, Wasserstein) are
added by ``ndim.evaluate`` helpers further below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from .plot_style import SAVE_DPI, figure_style  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

Tensor = torch.Tensor

COLORS = {
    "true_prior": "blue",
    "learned_prior": "red",
    "prior": "black",
    "mcmc": "blue",
    "vi": "red",
    "learned_drift": "#7b2cbf",
    "true_drift": "#e6a700",
    "train": "blue",
    "test": "red",
    "events": "blue",
    "true_path": "green",
}


def _finish(fig: plt.Figure, path: Optional[Path]) -> Optional[Path]:
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return path
    return None


def _robust_ylim(values: Sequence[float], low_q: float = 0.05, high_q: float = 1.0, pad: float = 0.1) -> Optional[Tuple[float, float]]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size < 3:
        return None
    lo, hi = np.quantile(v, low_q), np.quantile(v, high_q)
    if hi <= lo:
        return None
    span = hi - lo
    return lo - pad * span, hi + pad * span


# --------------------------------------------------------------------------------------
# Training diagnostics
# --------------------------------------------------------------------------------------
@figure_style
def plot_training_curves(history: List[Dict[str, float]], path: Optional[Path] = None, title: str = "") -> Optional[Path]:
    """Train vs test ELBO / log-likelihood / KL per epoch.

    ``history`` rows have keys ``epoch``, ``train_elbo``, ``train_loglik``, ``train_kl`` (running averages
    over the epoch's full-horizon batches) and, on evaluation epochs, ``eval_train_elbo``, ``test_elbo``, ...
    (fixed-noise estimates on a fixed train subset and on the test set).
    """
    ep = np.array([h["epoch"] for h in history], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, label in zip(axes, ["elbo", "loglik", "kl"], ["ELBO", "log-likelihood", "KL = 1/2 int u^2 dt"]):
        tr = np.array([h.get(f"train_{key}", np.nan) for h in history], dtype=float)
        ax.plot(ep, tr, color=COLORS["train"], alpha=0.35, lw=1.2, label="train (running, full horizon)")
        ev_ep = [h["epoch"] for h in history if f"test_{key}" in h]
        ev_tr = [h.get(f"eval_train_{key}", np.nan) for h in history if f"test_{key}" in h]
        ev_te = [h[f"test_{key}"] for h in history if f"test_{key}" in h]
        if ev_ep:
            ax.plot(ev_ep, ev_tr, "o-", color=COLORS["train"], ms=4, lw=1.6, label="train subset (fixed noise)")
            ax.plot(ev_ep, ev_te, "s-", color=COLORS["test"], ms=4, lw=1.6, label="test (fixed noise)")
        allv = list(tr[np.isfinite(tr)]) + list(ev_tr) + list(ev_te)
        yl = _robust_ylim(allv, low_q=0.05 if key != "kl" else 0.0, high_q=1.0 if key != "kl" else 0.95)
        if yl is not None:
            ax.set_ylim(*yl)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    if title:
        fig.suptitle(title)
    return _finish(fig, path)


@figure_style
def plot_parameter_history(
    history: List[Dict[str, float]],
    keys: Sequence[str],
    truth: Optional[Dict[str, float]] = None,
    path: Optional[Path] = None,
    title: str = "",
) -> Optional[Path]:
    """Trajectories of scalar parameters (e.g. ``drift.kappa``, ``drift.theta``, ``diffusion.eta``) vs epoch."""
    keys = [k for k in keys if any(k in h for h in history)]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ep = [h["epoch"] for h in history if k in h]
        v = [h[k] for h in history if k in h]
        ax.plot(ep, v, color=COLORS["learned_drift"], lw=1.8, label="learned")
        if truth and k.split(".")[-1] in truth:
            ax.axhline(truth[k.split(".")[-1]], color=COLORS["true_drift"], ls="--", lw=1.6, label="truth")
        ax.set_xlabel("epoch")
        ax.set_ylabel(k)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    if title:
        fig.suptitle(title)
    return _finish(fig, path)


@torch.no_grad()
@figure_style
def plot_drift_snapshot(
    learned_drift: Callable[[Tensor, Tensor], Tensor],
    true_drift: Optional[Callable[[Tensor, Tensor], Tensor]],
    z_max: float,
    T: float,
    times: Sequence[float] = (0.0, 0.5, 1.0),
    occupancy: Optional[Tensor] = None,
    path: Optional[Path] = None,
    title: str = "",
    device: Optional[torch.device] = None,
    z_min: float = 0.0,
) -> Optional[Path]:
    """Learned vs true drift ``b(z, t)`` as 1-D slices in ``z`` for a few times ``t = frac * T``.

    ``occupancy`` (optional, ``[n_paths, n_grid]`` true latent paths) draws a histogram of visited states
    behind the curves: the drift is only identifiable where the data-generating diffusion spends time
    (paper Sec. 4.1 / Prop. 3.3).
    """
    device = device or torch.device("cpu")
    z = torch.linspace(float(z_min), float(z_max), 400, device=device)
    fig, ax = plt.subplots(figsize=(12, 6))
    if occupancy is not None:
        ax2 = ax.twinx()
        ax2.hist(occupancy.flatten().cpu().numpy(), bins=60, range=(0, float(z_max)), color="0.85", density=True)
        ax2.set_yticks([])
        ax2.set_ylabel("occupation density (gray)", color="0.5")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    styles = ["-", "--", ":", "-."]
    for i, frac in enumerate(times):
        t = torch.full_like(z, float(frac) * float(T))
        bl = learned_drift(z, t).cpu().numpy()
        ax.plot(z.cpu().numpy(), bl, color=COLORS["learned_drift"], ls=styles[i % len(styles)], lw=2, label=f"learned, t={frac * T:g}")
        if true_drift is not None:
            bt = true_drift(z, t).cpu().numpy()
            ax.plot(z.cpu().numpy(), bt, color=COLORS["true_drift"], ls=styles[i % len(styles)], lw=1.6, label=f"true, t={frac * T:g}")
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.set_xlabel("z")
    ax.set_ylabel("drift b(z, t)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    if title:
        ax.set_title(title)
    return _finish(fig, path)


# ======================================================================================
# Evaluation figures.
# All functions accept numpy or torch inputs, take ``path: Optional[Path]`` and return it via
# ``_finish`` (``None`` when no path is given, in which case the figure is closed unsaved).
# ======================================================================================
import logging as _logging  # noqa: E402

from typing import Any, Union  # noqa: E402

_eval_logger = _logging.getLogger("ndim.plots")

ArrayLike = Union[np.ndarray, Tensor, Sequence[float]]


def _np(x: Any, dtype: Any = np.float64) -> np.ndarray:
    """``torch.Tensor`` / list / array -> float numpy array (detached, on CPU)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _finish_or_close(fig: plt.Figure, path: Optional[Path]) -> Optional[Path]:
    """``_finish`` plus closing the figure when it is not saved (keeps long evaluation runs leak-free)."""
    out = _finish(fig, path)
    if path is None:
        plt.close(fig)
    return out


def _wrap(title: str, width: int = 80) -> str:
    """Wrap a long title onto several lines so it is never clipped by the figure border."""
    import textwrap

    return "\n".join(textwrap.fill(part, width=width) for part in str(title).split("\n"))


def _cumulative_counts(events_list: Sequence[ArrayLike], t_grid: np.ndarray) -> np.ndarray:
    """``N_b(t) = #{i : tau_{b,i} <= t}`` on ``t_grid`` -> ``[n_obs, len(t_grid)]`` (non-finite padding ignored)."""
    out = np.zeros((len(events_list), t_grid.shape[0]), dtype=np.float64)
    for b, ev in enumerate(events_list):
        e = _np(ev).reshape(-1)
        e = np.sort(e[np.isfinite(e)])
        out[b] = np.searchsorted(e, t_grid, side="right")
    return out


def _band(ax: plt.Axes, t: np.ndarray, paths: np.ndarray, color: str, label: str, show_individual: bool,
          max_individual: int, lw: float = 2.0, alpha_band: float = 0.2, zorder: int = 2,
          count: bool = False, band_color: Optional[str] = None) -> None:
    """Mean / standard-deviation legend pairs and faint optional sample paths."""
    n = paths.shape[0]
    mean = paths.mean(axis=0)
    std = paths.std(axis=0) if n > 1 else np.zeros_like(mean)
    drawstyle = "steps-post" if count else "default"
    if show_individual:
        for i in range(min(n, int(max_individual))):
            ax.plot(t, paths[i], color=color, lw=0.8, alpha=0.15 if count else 0.3,
                    drawstyle=drawstyle, zorder=1)
    ax.plot(t, mean, color=color, lw=lw, alpha=0.8 if count else 1.0,
            drawstyle=drawstyle, label=f"{label} Mean" + (f" (n={n})" if count else ""), zorder=zorder + 1)
    lower = np.maximum(mean - std, 0) if count else mean - std
    ax.fill_between(t, lower, mean + std, color=band_color or color, alpha=alpha_band,
                    label=f"{label} ± Std", zorder=2)


@figure_style
def plot_prior_paths_comparison(
    t_grid: ArrayLike,
    true_paths: ArrayLike,
    learned_paths: ArrayLike,
    path: Optional[Path] = None,
    title: str = "",
    *,
    true_scheme: str = "Intensity Euler (full truncation)",
    learned_scheme: str = "Intensity Euler (full truncation)",
    n_pairs: int = 3,
) -> Optional[Path]:
    """Prior intensity means/SDs and paired paths sharing Brownian increments.

    Rows must already be paired by the simulator. This function never draws
    paths or resamples the pairs, so the displayed paths are those used by
    the common-noise path-deviation metric.
    """
    t, truth, learned = _np(t_grid), _np(true_paths), _np(learned_paths)
    if (t.ndim != 1 or truth.ndim != 2 or learned.shape != truth.shape
            or truth.shape[0] < 1 or truth.shape[1] != len(t)):
        raise ValueError("Prior paths must have matching nonempty [path,time] shapes")
    if not all(np.isfinite(a).all() for a in (t, truth, learned)):
        raise ValueError("Prior paths and time grid must be finite")
    if int(n_pairs) < 1:
        raise ValueError("n_pairs must be positive")
    count = min(int(n_pairs), truth.shape[0])
    fig = plt.figure(figsize=(12, max(4., count * 1.9)))
    grid = fig.add_gridspec(count, 2, width_ratios=(1.2, 1.))
    ax = fig.add_subplot(grid[:, 0])
    _band(ax, t, truth, COLORS["true_prior"], "True prior", False, 0)
    _band(ax, t, learned, COLORS["learned_prior"], "Learned prior", False, 0)
    ax.set(title=f"Mean and standard deviation ({truth.shape[0]} paths)", xlabel="Time", ylabel="Intensity Z")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    for index in range(count):
        paired = fig.add_subplot(grid[index, 1])
        paired.plot(t, truth[index], color=COLORS["true_prior"], label="True prior", lw=1.5)
        paired.plot(t, learned[index], color=COLORS["learned_prior"], label="Learned prior", lw=1.5, ls="--")
        paired.set(title=f"Pair {index + 1}: identical Brownian increments", ylabel="Intensity Z")
        paired.grid(alpha=.3)
        if index == 0:
            paired.legend(fontsize=8, ncol=2)
        if index == count - 1:
            paired.set_xlabel("Time")
        else:
            paired.tick_params(labelbottom=False)
    caption = f"True: {true_scheme}. Learned: {learned_scheme}. Bands show SD, not mean confidence intervals."
    fig.suptitle(_wrap(title or "True and learned prior SDE paths", 100), y=1.01)
    fig.text(.5, -.015, _wrap(caption, 125), ha="center", va="top", fontsize=9)
    return _finish_or_close(fig, path)


# --------------------------------------------------------------------------------------
# Fig. 3: cumulative counts of true vs generated observations
# --------------------------------------------------------------------------------------
@figure_style
def plot_cumulative_counts(
    true_events: Sequence[ArrayLike],
    generated_events: Sequence[ArrayLike],
    T: float,
    n_grid: int = 1000,
    show_individual: bool = False,
    max_individual: int = 64,
    path: Optional[Path] = None,
    title: str = "",
    true_label: str = "true data samples",
    generated_label: str = "learned prior samples",
) -> Optional[Path]:
    """Paper Fig. 3: mean +/- 1 std bands of the cumulative counting processes ``N(t)`` of true (blue) and
    generated (red) observations; ``show_individual`` adds up to ``max_individual`` thin individual curves per set.

    A well-fitted generative model (learned prior ``b_theta`` + thinning) reproduces both the mean growth
    (``E int_0^t Z_s ds``) and the spread of the true counting processes.
    """
    t = np.linspace(0.0, float(T), int(n_grid))
    Nt = _cumulative_counts(true_events, t)
    Ng = _cumulative_counts(generated_events, t)
    fig, ax = plt.subplots(figsize=(12, 4))
    if Nt.shape[0] > 0:
        _band(ax, t, Nt, COLORS["true_prior"], true_label, show_individual, max_individual, count=True)
    if Ng.shape[0] > 0:
        _band(ax, t, Ng, COLORS["learned_prior"], generated_label, show_individual, max_individual, count=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel 1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.suptitle(_wrap(title, 140) if title else "Cumulative Count Comparison")
    return _finish_or_close(fig, path)


# --------------------------------------------------------------------------------------
# Burn-in: distribution of the initial intensity
# --------------------------------------------------------------------------------------
@figure_style
def plot_initial_distribution(
    true_z0: Optional[ArrayLike],
    learned_z0: ArrayLike,
    t_burn: ArrayLike,
    burn_paths: ArrayLike,
    path: Optional[Path] = None,
    title: str = "",
    max_individual: int = 40,
    n_bins: int = 60,
    true_density: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    true_modes: Optional[Sequence[Tuple[float, float]]] = None,
    data_rate: Optional[float] = None,
) -> Optional[Path]:
    """Distribution of the initial intensity ``Z_0``: the truth's (blue) against the one the learned prior produces.

    Left: normalised histograms of draws of ``Z_0`` from the truth and of the learned prior at ``t = 0``, on common
    bins. Right: the learned prior on the burn-in ``[-burn_in, 0]`` (``burn_paths [n, len(t_burn)]``): mean +/- 1 std
    band and a few paths from the start value to ``t = 0``, with the truth's ``Z_0`` (mean +/- std) marked at ``t = 0``.
    ``true_density = (x, density)`` adds the exact density of the truth to the left panel; ``true_modes`` (a list of
    ``(mean, std)``, for a mixture) marks every mode at ``t = 0`` in place of the overall mean. Without a truth
    (``true_z0=None``: observations of an unspecified model) only the learned distribution is drawn, with
    ``data_rate``, the mean event rate of the observations at the start of the window, marked for orientation.
    """
    learned = _np(learned_z0).reshape(-1)
    true = _np(true_z0).reshape(-1) if true_z0 is not None else None
    t, P = _np(t_burn).reshape(-1), _np(burn_paths)
    both = np.concatenate([true, learned]) if true is not None else learned
    lo, hi = np.quantile(both, [0.002, 0.998])
    if hi <= lo:
        lo, hi = float(both.min()) - 0.5, float(both.max()) + 0.5
    bins = np.linspace(lo, hi, int(n_bins) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if true is not None:
        ax.hist(true, bins=bins, density=True, color=COLORS["true_prior"], alpha=0.45,
                label=f"true $Z_0$ (mean {true.mean():.2f}, sd {true.std():.2f})")
    ax.hist(learned, bins=bins, density=True, color=COLORS["learned_prior"], alpha=0.45,
            label=f"learned prior at t = 0 (mean {learned.mean():.2f}, sd {learned.std():.2f})")
    if true_density is not None:
        ax.plot(_np(true_density[0]).reshape(-1), _np(true_density[1]).reshape(-1), color=COLORS["true_prior"], lw=1.8,
                label="true density (exact)")
    if data_rate is not None:
        ax.axvline(float(data_rate), color=COLORS["true_prior"], lw=1.6, ls="--",
                   label=f"mean event rate at the start of the window ({float(data_rate):.2f})")
    ax.set_xlabel("Initial intensity $Z_0$")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    _band(ax, t, P, COLORS["learned_prior"], "learned prior", True, max_individual)
    if true is None:
        if data_rate is not None:
            ax.plot([0.0], [float(data_rate)], "o", color=COLORS["true_prior"], zorder=10, label="mean event rate at the start of the window")
    elif true_modes:
        for k, (mode_mean, mode_std) in enumerate(true_modes):
            ax.errorbar([0.0], [float(mode_mean)], yerr=[float(mode_std)], fmt="o", color=COLORS["true_prior"], capsize=4, zorder=10,
                        label="true modes of $Z_0$ (mean ± sd)" if k == 0 else None)
    else:
        ax.errorbar([0.0], [true.mean()], yerr=[true.std()], fmt="o", color=COLORS["true_prior"], capsize=4, zorder=10,
                    label="true $Z_0$ (mean ± sd)")
    # a learned prior may travel far on the burn-in: a logarithmic axis keeps the arrival at t = 0 readable
    positive = P[P > 0]
    if positive.size and float(P.max()) > 20.0 * max(float(np.quantile(P[:, -1], 0.99)), 1e-12):
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(float(positive.min()) * 0.7, float(np.quantile(P[:, -1], 0.01)) * 0.05))
    ax.set_xlabel("Time (burn-in, before the observation window)")
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    if title:
        fig.suptitle(_wrap(title, 140))
    return _finish_or_close(fig, path)


# --------------------------------------------------------------------------------------
# Fig. 4: prior / MCMC posterior / VI posterior for one observation
# --------------------------------------------------------------------------------------
def _dash_coinciding(drawn: Sequence[Tuple[str, plt.Line2D, np.ndarray, np.ndarray]], tol: float) -> str:
    """Keep mean curves that lie on top of each other visible. ``drawn`` holds ``(label, mean line, mean, std)`` of the
    path sets of a figure, in drawing order. Sets whose band is thinner than ``tol`` (about the width of a line) and
    whose mean lies within ``tol`` of the mean of another such set hide one another, and only the one drawn last would
    be seen: their mean lines are changed to alternating dashes. Returns a sentence that names them, for the legend, or
    ``""`` when no sets coincide. A model without spread gives this case: its prior and its posteriors are one curve.
    """
    thin = [i for i, d in enumerate(drawn) if float(np.max(d[3])) < tol]
    same = [i for i in thin if any(float(np.max(np.abs(drawn[i][2] - drawn[j][2]))) < tol for j in thin if j != i)]
    if len(same) < 2:
        return ""
    dash = 3.0                                                # length of a dash, in line widths
    for rank, i in enumerate(same):
        drawn[i][1].set_linestyle((rank * dash, (dash, (len(same) - 1) * dash)))
        drawn[i][1].set_dash_capstyle("butt")
    names = [drawn[i][0] for i in same]
    return f"{', '.join(names[:-1])} and {names[-1]} coincide (bands thinner than a line): drawn as alternating dashes"


@figure_style
def plot_posterior_comparison(
    t_grid: ArrayLike,
    prior_paths: Optional[ArrayLike],
    mcmc_paths: Optional[ArrayLike],
    vi_paths: Optional[ArrayLike],
    events: ArrayLike,
    t_bar: Optional[float],
    path: Optional[Path] = None,
    title: str = "",
    show_individual: bool = False,
    max_individual: int = 32,
    reference_label: str = "MCMC posterior",
    true_path: Optional[ArrayLike] = None,
    prior_label: str = "true prior",
) -> Optional[Path]:
    """Paper Fig. 4: mean +/- 1 std bands of the prior (gray), the truth posterior reference (blue) and the amortised VI
    posterior (red) intensity paths ``Z_t`` for one observation, with blue event dots along the bottom.

    ``t_bar`` is the conditioning horizon ``T'``: when it is ``< T`` (Fig. 4b, partial observation) a dashed
    vertical line marks it, events after ``T'`` are drawn in light gray (they were *not* seen by either
    posterior) and both posteriors follow the prior dynamics after ``T'``. Any of the three path sets may
    be ``None`` (e.g. the comparison reference is disabled). ``show_individual`` adds thin individual paths.
    ``true_path`` (``[len(t_grid)]``) is the true latent intensity of this observation, known for synthetic data:
    it is drawn as a dashed line. Neither posterior sees it. ``prior_label`` names the gray band: the true prior, or the
    learned one for observations of an unspecified model. Sets that coincide to within the width of a line (the prior
    and the posteriors of a model without spread) are drawn as alternating dashes and named in the legend, see
    ``_dash_coinciding``; otherwise the set drawn last would cover the others.
    """
    t = _np(t_grid).reshape(-1)
    T = float(t[-1])
    ev = _np(events).reshape(-1)
    ev = ev[np.isfinite(ev)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sets = [
        (prior_paths, COLORS["prior"], prior_label, 2),
        (vi_paths, COLORS["vi"], "learned posterior", 4),
        (mcmc_paths, COLORS["mcmc"], reference_label, 6),
    ]
    y_min, y_max = np.inf, 0.0
    drawn = []                                                # (label, mean line, mean, std) of every set that is drawn
    for paths, color, label, z in sets:
        if paths is None:
            continue
        P = _np(paths)
        if P.ndim != 2 or P.shape[1] != t.shape[0]:
            raise ValueError(f"{label} paths must be [n, len(t_grid)={t.shape[0]}], got {P.shape}")
        _band(ax, t, P, color, label, show_individual, max_individual, zorder=z,
              band_color="gray" if label == prior_label else color)
        m, s = P.mean(axis=0), (P.std(axis=0) if P.shape[0] > 1 else 0.0)
        drawn.append((label, ax.lines[-1], m, np.broadcast_to(np.asarray(s, dtype=float), m.shape)))
        y_min = min(y_min, float(np.min(m - s)))
        y_max = max(y_max, float(np.max(m + s)))
    if true_path is not None:
        tp = _np(true_path).reshape(-1)
        if tp.shape[0] != t.shape[0]:
            raise ValueError(f"true_path must have len(t_grid)={t.shape[0]} values, got {tp.shape[0]}")
        ax.plot(t, tp, color=COLORS["true_path"], lw=1.6, ls="--", zorder=8, label="true latent path of this observation")
        y_min, y_max = min(y_min, float(np.min(tp))), max(y_max, float(np.max(tp)))
    y_max = y_max if y_max > 0 else 1.0
    # events as dots below the paths; the held-out events of a partial observation stay, in gray
    y_min = y_min if np.isfinite(y_min) else 0.0
    span = max(y_max - y_min, 1.0)
    coincide = _dash_coinciding(drawn, tol=0.004 * span)      # half the width of a mean line, in units of the y-axis
    event_y = y_min - 0.045 * span
    if ev.size:
        seen = ev <= (float(t_bar) if t_bar is not None else np.inf)
        if np.any(seen):
            ax.scatter(ev[seen], np.full(seen.sum(), event_y), color=COLORS["events"], s=18,
                       label="Events 1", zorder=10)
        if np.any(~seen):
            ax.scatter(ev[~seen], np.full((~seen).sum(), event_y), color="0.7", s=18,
                       label="Events after T′ (unobserved)", zorder=10)
    if t_bar is not None and float(t_bar) < T:
        ax.axvline(float(t_bar), color="0.25", ls="--", lw=1.2, label=f"T' = {float(t_bar):g} (conditioning horizon)")
        ax.axvspan(float(t_bar), T, color="0.93", zorder=0)
    ax.margins(x=0.05)
    ax.set_ylim(y_min - 0.09 * span, y_max + 0.06 * span)
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.3)
    # the legend goes below the axes, where it covers no path
    if coincide:  # handles long enough to show the dashes
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, handlelength=3.2,
                  title=_wrap(coincide, 150), title_fontsize=9)
    else:
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    if title:
        ax.set_title(_wrap(title, 140), fontsize=12)
    return _finish_or_close(fig, path)


# --------------------------------------------------------------------------------------
# Fig. 2 / Fig. 11: learned vs true drift
# --------------------------------------------------------------------------------------
def _drift_on_grid(fn: Callable[[Tensor, Tensor], Tensor], z: Tensor, t: Tensor) -> np.ndarray:
    """Evaluate ``fn`` on the mesh ``z x t`` -> ``[len(z), len(t)]`` numpy."""
    zz, tt = torch.meshgrid(z, t, indexing="ij")
    out = fn(zz.reshape(-1), tt.reshape(-1))
    return out.reshape(zz.shape).detach().cpu().numpy().astype(np.float64)


def _occupation_density(occupancy_paths: ArrayLike, T: float, z_max: float, n_z: int = 40, n_t: Optional[int] = None):
    """2-D occupation histogram of ``(t_k, Z_k)`` over paths ``[n, G]`` -> ``(t_centers, z_centers, density [n_z, n_t])``."""
    P = _np(occupancy_paths)
    n, G = P.shape
    tg = np.linspace(0.0, float(T), G)
    n_t = int(n_t if n_t is not None else max(4, min(G - 1, 25)))
    H, ze, te = np.histogram2d(P.ravel(), np.tile(tg, n), bins=[int(n_z), n_t], range=[[0.0, float(z_max)], [0.0, float(T)]], density=True)
    try:  # light smoothing so the contours of a small sample are not jagged
        from scipy.ndimage import gaussian_filter

        H = gaussian_filter(H, sigma=0.9, mode="nearest")
    except Exception:  # pragma: no cover - scipy is a hard dependency, but do not fail a figure over it
        pass
    return 0.5 * (te[1:] + te[:-1]), 0.5 * (ze[1:] + ze[:-1]), H


def _mass_levels(H: np.ndarray, fractions: Sequence[float] = (0.99, 0.9, 0.5)) -> List[Tuple[float, float]]:
    """``[(level, fraction), ...]`` sorted by increasing density level: the super-level set ``{H >= level}``
    encloses ``fraction`` of the total mass of ``H`` (so the lowest level is the outermost contour)."""
    flat = np.sort(H.ravel())[::-1]
    if flat.size == 0 or not np.isfinite(flat[0]) or flat[0] <= 0:
        return []
    cum = np.cumsum(flat) / flat.sum()
    out: Dict[float, float] = {}
    for f in sorted(float(x) for x in fractions):            # smaller fraction -> higher level
        k = int(np.searchsorted(cum, f))
        lvl = float(flat[min(k, flat.size - 1)])
        if lvl > 0 and lvl not in out:
            out[lvl] = f
    return sorted(out.items())


@torch.no_grad()
@figure_style
def plot_drift_comparison(
    learned_drift: Callable[[Tensor, Tensor], Tensor],
    true_drift: Optional[Callable[[Tensor, Tensor], Tensor]],
    z_max: float,
    T: float,
    times: Sequence[float] = (0.0, 0.5, 1.0),
    occupancy: Optional[ArrayLike] = None,
    path: Optional[Path] = None,
    title: str = "",
    device: Optional[torch.device] = None,
    z_min: float = 0.0,
    quantity: str = "drift",
) -> Optional[Path]:
    """Paper Fig. 2 (1-D slices): learned (purple) vs true (goldenrod) drift ``b(z, t)`` as functions of ``z``,
    one panel per time ``t = frac * T`` in ``times``.

    ``occupancy`` (``[n_paths, N+1]`` true latent paths) draws, behind each panel, the histogram of the states
    visited in the time window around that ``t`` (windows partition ``[0, T]`` between consecutive slice times):
    the drift is only identifiable where the data-generating diffusion actually spends time (paper Sec. 4.1,
    Prop. 3.3). ``plot_drift_snapshot`` is the single-panel variant used during training.
    """
    symbol = "σ" if quantity == "diffusion" else "b"
    device = device or torch.device("cpu")
    times = [float(f) for f in times]
    z = torch.linspace(float(z_min), float(z_max), 400, device=device)
    zn = z.cpu().numpy()
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.3), squeeze=False, sharey=True)
    occ = _np(occupancy) if occupancy is not None else None
    if occ is not None:
        tg = np.linspace(0.0, float(T), occ.shape[1])
        half = 0.5 * float(T) / max(n - 1, 1)
    learned_vals, true_vals = [], []
    for i, frac in enumerate(times):
        ax = axes[0, i]
        t = torch.full_like(z, frac * float(T))
        bl = learned_drift(z, t).detach().cpu().numpy().astype(np.float64)
        learned_vals.append(bl)
        if occ is not None:
            ax2 = ax.twinx()
            mask = np.abs(tg - frac * float(T)) <= half + 1e-9
            vals = occ[:, mask].ravel()
            ax2.hist(vals, bins=50, range=(0.0, float(z_max)), color="0.86", density=True, zorder=0)
            ax2.set_yticks([])
            if i == n - 1:
                ax2.set_ylabel(f"occupation density (gray, |t - t_i| <= {half:g})", color="0.5", fontsize=9)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        ax.plot(zn, bl, color=COLORS["learned_drift"], lw=2.2, label=f"learned {symbol}(z, t)")
        if true_drift is not None:
            bt = true_drift(z, t).detach().cpu().numpy().astype(np.float64)
            true_vals.append(bt)
            ax.plot(zn, bt, color=COLORS["true_drift"], lw=1.8, ls="--", label=f"true {symbol}(z, t)")
        ax.axhline(0.0, color="0.3", lw=0.6)
        ax.set_title(f"t = {frac * float(T):g}" + ("  (= T)" if abs(frac - 1.0) < 1e-9 else ""), fontsize=12)
        ax.set_xlabel("z")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(f"{quantity} {symbol}(z, t)")
            ax.legend(fontsize=9, loc="upper right")
    # the full range of the coefficients, so that a poor fit stays visible
    ref = np.concatenate(true_vals + learned_vals)
    lo, hi = float(np.min(ref)), float(np.max(ref))
    pad = 0.08 * max(hi - lo, 1e-6)
    axes[0, 0].set_ylim(lo - pad, hi + pad)
    if title:
        fig.suptitle(_wrap(title, 110))
    return _finish_or_close(fig, path)


@torch.no_grad()
@figure_style
def plot_learned_model(
    t_grid: ArrayLike,
    prior_paths: ArrayLike,
    bin_edges: ArrayLike,
    observed: Dict[str, ArrayLike],
    model: Dict[str, ArrayLike],
    intensity_drift: Callable[[Tensor, Tensor], Tensor],
    intensity_diffusion: Callable[[Tensor, Tensor], Tensor],
    T: float,
    path: Optional[Path] = None,
    title: str = "",
    device: Optional[torch.device] = None,
    n_sample_paths: int = 10,
    n_z: int = 110,
    n_t: int = 90,
) -> Optional[Path]:
    """The learned model of observations that have no known truth, on one page.

    Top row, the learned prior against the observations: the intensity over time against the event rate of the data;
    the spread between observations of the number of events per time bin; a few intensity paths of the learned prior.
    Bottom row, what was learned: drift and diffusion of the intensity over the states the learned prior visits
    (between its 0.5% and 99.5% quantiles at every time, and at least 30% either side of its mean), and the number
    of events of an observation.

    ``t_grid [G]`` / ``prior_paths [n, G]``: intensity paths of the learned prior, from the start of the burn-in
    (negative times) to ``T``. ``bin_edges [n_bins + 1]`` on ``[0, T]``. ``observed`` and ``model`` hold ``rate``,
    ``count_sd`` (per bin) and ``totals`` (events per observation).
    """
    device = device or torch.device("cpu")
    t, P = _np(t_grid), _np(prior_paths)
    edges = _np(bin_edges)
    centres, widths = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    window = t >= -1e-9
    mean, sd = P.mean(axis=0), P.std(axis=0)
    red, blue = COLORS["learned_prior"], COLORS["true_prior"]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))

    def mark_burn_in(ax: plt.Axes) -> None:
        if t[0] < -1e-9:
            ax.axvspan(t[0], 0.0, color="0.92", zorder=0, label="burn-in (no observations)")
            ax.axvline(0.0, color="0.5", lw=0.8)

    # ---- intensity over time
    ax = axes[0, 0]
    mark_burn_in(ax)
    ax.fill_between(t, np.maximum(mean - sd, 0.0), mean + sd, color=red, alpha=0.2, label="learned prior ± Std")
    ax.plot(t, mean, color=red, lw=2.0, label="learned prior Mean")
    ax.step(edges, np.r_[_np(observed["rate"]), _np(observed["rate"])[-1]], where="post", color=blue, lw=1.6, label="event rate of the observations")
    # scaled to the observation window; the burn-in may leave the axis
    ax.set_ylim(0.0, 1.12 * max(float((mean + sd)[window].max()), float(_np(observed["rate"]).max())))
    ax.set_title("Intensity over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- spread between observations, bin by bin
    ax = axes[0, 1]
    ax.plot(centres, _np(observed["count_sd"]), color=blue, lw=1.8, marker="o", ms=3.5, label="observations")
    ax.plot(centres, _np(model["count_sd"]), color=red, lw=1.8, marker="s", ms=3.0, label="learned model")
    ax.plot(centres, np.sqrt(np.maximum(_np(observed["rate"]) * widths, 0.0)), color="0.4", lw=1.2, ls=":",
            label="Poisson with the observed rate (no variation of the intensity)")
    ax.set_title("Std between observations of the events per time bin")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Std of the number of events in a bin of {widths.mean():.3g}")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ---- sample paths
    ax = axes[0, 2]
    mark_burn_in(ax)
    for i in range(min(int(n_sample_paths), P.shape[0])):
        ax.plot(t, P[i], lw=0.9, alpha=0.75)
    ax.plot(t, mean, color="black", lw=1.6, ls="--", label="learned prior Mean")
    ax.set_title(f"{min(int(n_sample_paths), P.shape[0])} intensity paths of the learned prior")
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    ax.set_ylim(0.0, float(np.quantile(P[:, window], 0.999)) * 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ---- drift and diffusion of the intensity over the visited states
    Pw, tw = P[:, window], t[window]
    lo, hi = np.quantile(Pw, 0.005, axis=0), np.quantile(Pw, 0.995, axis=0)
    # at least 30% either side of the mean, so that a prior without spread still shows its coefficients
    lo, hi = np.minimum(lo, 0.7 * mean[window]), np.maximum(hi, 1.3 * mean[window])
    z_lo, z_hi = max(float(lo.min()) * 0.9, 1e-6), float(hi.max()) * 1.05
    z = torch.linspace(z_lo, z_hi, int(n_z), device=device)
    tt = torch.linspace(0.0, float(T), int(n_t), device=device)
    B, S = _drift_on_grid(intensity_drift, z, tt), _drift_on_grid(intensity_diffusion, z, tt)            # [n_z, n_t]
    zz = z.cpu().numpy()[:, None]
    visited = (zz >= np.interp(tt.cpu().numpy(), tw, lo)[None, :]) & (zz <= np.interp(tt.cpu().numpy(), tw, hi)[None, :])
    B, S = np.where(visited, B, np.nan), np.where(visited, S, np.nan)
    extent = [0.0, float(T), z_lo, z_hi]
    vmax = float(np.nanquantile(np.abs(B), 0.98)) if np.isfinite(B).any() else 1.0
    im = axes[1, 0].imshow(B, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-(vmax or 1.0), vmax=vmax or 1.0, interpolation="nearest")
    fig.colorbar(im, ax=axes[1, 0], label="drift of the intensity (red: upwards)")
    axes[1, 0].set_title("Learned drift of the intensity")
    im = axes[1, 1].imshow(S, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=0.0,
                           vmax=float(np.nanquantile(S, 0.98)) if np.isfinite(S).any() else 1.0, interpolation="nearest")
    fig.colorbar(im, ax=axes[1, 1], label="diffusion of the intensity")
    axes[1, 1].set_title("Diffusion of the intensity")
    for ax in axes[1, :2]:
        ax.plot(tw, mean[window], color="black", lw=1.4, ls="--", label="learned prior Mean")
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        ax.set_facecolor("0.96")
        ax.legend(fontsize=8, loc="upper right")

    # ---- number of events of an observation
    ax = axes[1, 2]
    obs_tot, mod_tot = _np(observed["totals"]), _np(model["totals"])
    lo_c, hi_c = float(min(obs_tot.min(), np.quantile(mod_tot, 0.001))), float(max(obs_tot.max(), np.quantile(mod_tot, 0.999)))
    bins = np.linspace(lo_c, hi_c + 1e-9, 26)
    ax.hist(mod_tot, bins=bins, density=True, color=red, alpha=0.35, label=f"learned model: mean {mod_tot.mean():.1f}, Std {mod_tot.std():.1f}")
    ax.hist(obs_tot, bins=bins, density=True, histtype="step", color=blue, lw=1.8, label=f"observations: mean {obs_tot.mean():.1f}, Std {obs_tot.std():.1f}")
    ax.set_title("Number of events of an observation")
    ax.set_xlabel("Events on [0, T]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    if title:
        fig.suptitle(_wrap(title, 150))
    return _finish_or_close(fig, path)


@torch.no_grad()
@figure_style
def plot_drift_heatmap(
    learned_drift: Callable[[Tensor, Tensor], Tensor],
    true_drift: Optional[Callable[[Tensor, Tensor], Tensor]],
    z_max: float,
    T: float,
    occupancy_paths: Optional[ArrayLike],
    path: Optional[Path] = None,
    title: str = "",
    device: Optional[torch.device] = None,
    n_z: int = 120,
    n_t: int = 60,
    z_min: float = 0.0,
) -> Optional[Path]:
    """Paper Fig. 2 (2-D view): heat-maps of the learned drift ``b_theta(z, t)`` (left, diverging colour map
    centred at 0) and of the absolute error ``|b_theta - b_true|`` (right) over a ``(t, z)`` grid.

    The occupation density of ``occupancy_paths`` (``[n, N+1]`` true latent paths) is overlaid as contour lines
    enclosing 50 / 90 / 99 % of the visited ``(t, Z_t)`` mass, so the reader can see that the drift is only
    learned (and only needs to be) where the data live. Without ``true_drift`` the right panel shows the
    true-drift-free alternative ``|b_theta|``.
    """
    device = device or torch.device("cpu")
    z = torch.linspace(float(z_min), float(z_max), int(n_z), device=device)
    t = torch.linspace(0.0, float(T), int(n_t), device=device)
    B_l = _drift_on_grid(learned_drift, z, t)                         # [n_z, n_t]
    err = np.abs(B_l - _drift_on_grid(true_drift, z, t)) if true_drift is not None else np.abs(B_l)
    extent = [0.0, float(T), float(z_min), float(z_max)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    vmax = float(np.nanmax(np.abs(B_l))) if np.isfinite(B_l).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im0 = axes[0].imshow(B_l, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], label="learned drift b(z, t)")
    axes[0].set_title("learned drift b_theta(z, t)", fontsize=12)
    im1 = axes[1].imshow(err, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], label="|learned - true|" if true_drift is not None else "|learned|")
    axes[1].set_title("absolute error |b_theta - b_true|(z, t)" if true_drift is not None else "|b_theta(z, t)|", fontsize=12)
    if occupancy_paths is not None:
        tc, zc, H = _occupation_density(occupancy_paths, T, z_max, n_z=40)
        level_fracs = _mass_levels(H)
        if level_fracs:
            levels = [l for l, _ in level_fracs]
            fmt = {l: f"{f:.0%}" for l, f in level_fracs}
            for ax, col in zip(axes, ("0.15", "white")):
                cs = ax.contour(tc, zc, H, levels=levels, colors=col, linewidths=0.9, alpha=0.9)
                ax.clabel(cs, fmt=fmt, fontsize=6.5, inline=True)
            axes[0].plot([], [], color="0.15", lw=0.9, label="occupation of true paths (50/90/99% mass)")
            axes[0].legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_ylabel("z")
    if title:
        fig.suptitle(_wrap(title, 100))
    return _finish_or_close(fig, path)


@torch.no_grad()
def drift_3d_figure(
    learned_drift: Callable[[Tensor, Tensor], Tensor],
    true_drift: Optional[Callable[[Tensor, Tensor], Tensor]],
    z_max: float,
    T: float,
    device: Optional[torch.device] = None,
    n_z: int = 100,
    n_t: int = 100,
    title: str = "Learned Neural Network Drift",
    z_min: float = 0.0,
):
    """Paper Fig. 2 as a plotly figure: the learned drift (purple) and the true drift (orange) over ``(Z, t)``.

    Drawn as in the paper: ``Z`` on the x axis, ``t`` on the y axis, two solid-coloured surfaces of opacity 0.7 on
    a 100 x 100 grid, Plotly's default theme.
    """
    import plotly.graph_objects as go  # type: ignore

    device = device or torch.device("cpu")
    z = torch.linspace(float(z_min), float(z_max), int(n_z), device=device)
    t = torch.linspace(0.0, float(T), int(n_t), device=device)
    Z, Tm = np.meshgrid(z.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64))   # [n_t, n_z]

    def surface(drift: Callable[[Tensor, Tensor], Tensor], colour: str, name: str):
        # a constant surfacecolor with a one-colour scale gives a solid surface
        return go.Surface(x=Z, y=Tm, z=_drift_on_grid(drift, z, t).T, surfacecolor=np.full_like(Z, 1.0),
                          colorscale=[[0, colour], [1, colour]], cmin=0, cmax=1, showscale=False, opacity=0.7, name=name)

    fig = go.Figure()
    fig.add_trace(surface(learned_drift, "purple", "Neural Network Drift"))
    if true_drift is not None:
        fig.add_trace(surface(true_drift, "orange", "True Drift"))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Z", yaxis_title="t", zaxis_title="Drift b(Z, t)"),
        legend=dict(title="Legend"),
    )
    return fig


@torch.no_grad()
@figure_style
def plot_drift_3d_html(
    learned_drift: Callable[[Tensor, Tensor], Tensor],
    true_drift: Optional[Callable[[Tensor, Tensor], Tensor]],
    z_max: float,
    T: float,
    path: Path,
    device: Optional[torch.device] = None,
    n_z: int = 100,
    n_t: int = 100,
    title: str = "Learned Neural Network Drift",
    z_min: float = 0.0,
) -> Optional[Path]:
    """Paper Fig. 2 (3-D surfaces, see ``drift_3d_figure``), written as a self-contained interactive HTML file.
    ``plotly`` is imported lazily; when it is missing the function logs a warning and returns ``None`` (no file
    is written).
    """
    try:
        fig = drift_3d_figure(learned_drift, true_drift, z_max, T, device=device, n_z=n_z, n_t=n_t, title=title, z_min=z_min)
    except ImportError as e:  # pragma: no cover - depends on the environment
        _eval_logger.warning("plotly not available (%s); skipping %s", e, path)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


# --------------------------------------------------------------------------------------
# Fig. 5 and Wasserstein summaries
# --------------------------------------------------------------------------------------
def _w2_color(label: str) -> str:
    key = label.lower()
    if "baseline" in key or "prior" in key:
        return COLORS["prior"]
    if "test" in key:
        return COLORS["test"]
    return COLORS["train"]


@figure_style
def plot_wasserstein_bars(
    results: Dict[str, float],
    path: Optional[Path] = None,
    title: str = "",
    errors: Optional[Dict[str, float]] = None,
    ylabel: str = "2-Wasserstein distance to MCMC posterior",
) -> Optional[Path]:
    """Bar chart of Wasserstein summaries, e.g. ``{"train VI": .., "test VI": .., "test prior baseline": ..}``
    (paper Fig. 5 at a single ``n``): train (blue) vs test (red) VI posterior and the no-correction prior
    baseline (gray). ``errors`` (same keys) adds error bars (e.g. std over observations); values are annotated.
    """
    labels = list(results.keys())
    vals = np.array([float(results[k]) if results[k] is not None else np.nan for k in labels], dtype=float)
    errs = np.array([float(errors.get(k, 0.0)) if errors else 0.0 for k in labels], dtype=float)
    errs = np.where(np.isfinite(errs), errs, 0.0)
    fig, ax = plt.subplots(figsize=(1.6 * max(len(labels), 3) + 2.0, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, np.nan_to_num(vals), yerr=errs if np.any(errs > 0) else None, capsize=4,
                  color=[_w2_color(l) for l in labels], edgecolor="0.2", lw=0.6)
    for b, v, e in zip(bars, vals, errs):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e, f"  {v:.3g}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.0, "n/a", ha="center", va="bottom", fontsize=9, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    top = float(np.nanmax(np.nan_to_num(vals) + errs)) if len(vals) else 1.0
    ax.set_ylim(0.0, max(top, 1e-6) * 1.15)
    ax.grid(alpha=0.3, axis="y")
    if title:
        ax.set_title(_wrap(title, 62), fontsize=12)
    return _finish_or_close(fig, path)


# --------------------------------------------------------------------------------------
# MCMC diagnostics (extra, used by ndim.evaluate)
# --------------------------------------------------------------------------------------
@figure_style
def plot_mcmc_traces(
    traces: Dict[str, ArrayLike],
    n_burn_in: Optional[int] = None,
    path: Optional[Path] = None,
    title: str = "",
) -> Optional[Path]:
    """Mean (over chains) log-likelihood trace of MCMC runs, one line per entry of ``traces`` (label -> ``[n_steps]``).

    A trace that has flattened out well before the end indicates the chains reached the posterior's typical
    set; ``n_burn_in`` marks the end of the adaptive burn-in (paper Sec. 4.2 footnote 4).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    palette = ["blue", "red", "black", "purple", "goldenrod", "green", "cyan", "magenta"]
    for i, (label, tr) in enumerate(traces.items()):
        y = _np(tr).reshape(-1)
        if y.size == 0:
            continue
        ax.plot(np.arange(1, y.size + 1), y, lw=1.2, color=palette[i % len(palette)], label=label)
    if n_burn_in is not None and n_burn_in > 0:
        ax.axvline(float(n_burn_in), color="0.3", ls="--", lw=1.0, label="end of burn-in / adaptation")
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel("mean log-likelihood over chains")
    ax.grid(alpha=0.3)
    if traces:
        ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    return _finish_or_close(fig, path)


# --------------------------------------------------------------------------------------
# The coefficient of the diffusion estimated from the observations (ndim.cir_eta)
# --------------------------------------------------------------------------------------
@figure_style
def plot_eta_estimate(estimate: Any, path: Optional[Path] = None, unit: str = "time units of the data", title: Optional[str] = None) -> Optional[Path]:
    """The curve ``eta2(h)`` of ``ndim.cir_eta.estimate_cir_eta`` with its standard errors and the fit whose level at
    ``h = 0`` is the estimate of ``eta^2``. Hollow points are bin widths that the fit leaves out."""
    h, est, se = _np(estimate.widths), _np(estimate.curve), _np(estimate.curve_se)
    used = np.asarray(estimate.used, dtype=bool)
    beta = _np(estimate.coefficients)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.errorbar(h[used], est[used], yerr=se[used], fmt="o", ms=5, color="blue", ecolor="blue", elinewidth=1.2, capsize=2.5,
                label="estimate at bin width h, ± standard error")
    if (~used).any():
        ax.errorbar(h[~used], est[~used], yerr=se[~used], fmt="o", ms=5, mfc="white", color="0.45", ecolor="0.6", elinewidth=1.0,
                    capsize=2.5, label="not used by the fit")
    grid = np.linspace(0.0, float(h[used].max()), 200)
    line = beta[0] + beta[1] * grid
    ax.plot(grid, line, color="black", lw=1.6, ls="--" if estimate.with_groups else "-",
            label=f"eta^2 + c h: eta^2 = {estimate.eta2:.3g} (standard error {estimate.eta2_se:.2g}) at h = 0")
    if estimate.with_groups:
        g = grid[grid >= 0.6 * float(h[used].min())]
        ax.plot(g, beta[0] + beta[1] * g + beta[2] / g ** 2, color="black", lw=1.6, label="the fit, with d / h^2 for events in groups")
    # the range that the fit determines; the smallest bins are noisy and may leave it
    core = se <= 4.0 * np.median(se)
    lo = min(0.0, float((est[core] - se[core]).min()), float(line.min()))
    hi = max(float((est[core] + se[core]).max()), float(line.max()))
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0.0, 1.04 * float(h.max()))
    ax.set_xlabel(f"bin width h ({unit})")
    ax.set_ylabel("estimate of eta^2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(_wrap(title or f"Coefficient of the diffusion: {estimate.summary()}", 90), fontsize=10)
    return _finish_or_close(fig, path)
