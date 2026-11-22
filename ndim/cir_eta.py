"""The coefficient ``eta`` of a CIR diffusion ``eta * sqrt(Z)`` of the intensity, estimated from the observations.

``model.diffusion: {kind: cir, eta_init: auto, trainable: false}`` (the default of ``scripts/fit.py``) asks for it:
``eta`` is estimated once from the training observations, before training, and then kept fixed.

No model of the drift is needed. Let ``N_k`` be the number of events of an observation in bin ``k`` of width ``h``
and ``L_k`` the integral of its intensity over that bin. Given the intensity path the counts are independent Poisson
variables, so for two adjacent bins, with variances taken over the observations (which removes the mean curve),

    Var(N_{k+1} - N_k) = E[N_{k+1} + N_k] + Var(L_{k+1} - L_k).

The Brownian term ``eta * sqrt(Z) dB`` contributes ``eta^2 int w_k(s)^2 E[Z_s] ds`` to ``Var(L_{k+1} - L_k)``, where
the tent weight ``w_k`` rises from 0 to ``h`` over bin ``k`` and falls back to 0 over bin ``k + 1``: about
``(2/3) eta^2 E[Z] h^3``. Whatever is smooth in time (the pull of the drift, a level that differs from one
observation to the next times the slope of the mean curve) contributes at order ``h^4``. Hence

    eta2(h) = sum_k [ s^2(N_{k+1} - N_k) - mean(N_{k+1} + N_k) ] / sum_k int w_k^2 zbar ds  =  eta^2 + c h + ...

and ``eta^2`` is the level of this curve at ``h -> 0``, found by a weighted straight line in ``h``. Two things are
decided from the data: a term ``d / h^2`` is added when events come in groups (several within seconds), which raises
the curve at the smallest bins; and the widest bins are left out while the curve bends, as it does when the
intensity returns to its mean curve faster than the widest bin. Standard errors and the interval come from a
bootstrap over the observations.

What the estimate can and cannot do (``docs/OWN_DATA.md``): it measures the variation of the intensity on short time
scales. Its precision grows with ``eta^2 T^2`` and with the number of observations; where the observations do not
determine it, the interval says so and a floor is used.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_FINE = 1440                                    # the events are counted on this many equal bins of [0, T]
WIDTHS = (5, 10, 15, 20, 30, 45, 60, 90, 120)    # bin widths in fine bins: T/288 ... T/12
MIN_WIDTHS = 5                                   # the fit keeps at least this many (the smallest) widths
Z_KEEP = 2.0                                     # a term is kept, or the curve counted as bent, beyond this many standard errors
FLOOR_CV = 0.1                                   # floor of eta: the diffusion alone moves Z by this share of its mean over [0, T]
MIN_OBSERVATIONS = 3                             # fewer observations: no estimate, the floor is used


@dataclass
class EtaEstimate:
    """Result of ``estimate_cir_eta``. ``eta`` is the value to use; the rest says how well it is determined."""
    eta: float                      # sqrt(eta2), or ``floor`` when the estimate lies below it
    eta2: float                     # the level of eta2(h) at h -> 0; negative values occur when nothing is detected
    eta2_se: float                  # its bootstrap standard error
    interval: Tuple[float, float]   # 95% bootstrap interval of eta (0 where the interval of eta^2 is negative)
    floor: float
    floored: bool                   # eta is the floor
    with_groups: bool               # the fit has the term d / h^2 for events that come in groups
    widths: List[float]             # bin widths h, in the time unit of the data
    curve: List[float]              # eta2(h)
    curve_se: List[float]
    used: List[bool]                # widths that the fit uses
    coefficients: List[float]       # [eta^2, c] or [eta^2, c, d]
    n_observations: int
    mean_events: float

    @property
    def determined(self) -> bool:
        """The interval excludes zero: the observations show a diffusion on short time scales."""
        return self.interval[0] > 0.0

    def summary(self) -> str:
        if self.n_observations < MIN_OBSERVATIONS:
            return (f"eta = {self.eta:.3g}, the floor: {self.n_observations} observations are too few to estimate it "
                    f"(at least {MIN_OBSERVATIONS} are needed)")
        if self.floored:
            return (f"eta = {self.eta:.3g}, the floor: the observations show no variation of the intensity on short time scales "
                    f"(estimate of eta^2 {self.eta2:.3g} with standard error {self.eta2_se:.3g}; 95% interval of eta "
                    f"{self.interval[0]:.3g} to {self.interval[1]:.3g})")
        text = f"eta = {self.eta:.3g} (95% interval {self.interval[0]:.3g} to {self.interval[1]:.3g})"
        if not self.determined:
            text += ": the observations determine it only roughly"
        return text

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["interval"] = list(self.interval)
        d["determined"] = self.determined
        return d


def fine_counts(events: Any, T: float, n_fine: int = N_FINE) -> np.ndarray:
    """Counts of every observation on ``n_fine`` equal bins of ``[0, T]``: ``[n, n_fine]``. ``events`` is a 2-D array
    or tensor padded with ``inf`` / ``nan``, or one sequence of event times per observation."""
    edges = np.linspace(0.0, float(T), int(n_fine) + 1)
    if hasattr(events, "detach"):
        events = events.detach().cpu().numpy()
    out = np.zeros((len(events), int(n_fine)), dtype=np.float64)
    for i, row in enumerate(events):
        e = np.asarray(row, dtype=np.float64)
        out[i] = np.histogram(e[np.isfinite(e)], bins=edges)[0]
    return out


def _tent_weights(m: int, delta: float) -> np.ndarray:
    """Integral of ``w(s)^2`` over each of the ``2m`` fine bins of a pair of adjacent bins of width ``m * delta``."""
    j = np.arange(m, dtype=np.float64)
    g = delta ** 3 * ((j + 1.0) ** 3 - j ** 3) / 3.0
    return np.concatenate([g, g[::-1]])


def eta2_curve(C: np.ndarray, ms: Sequence[int], delta: float, n_offsets: int = 4) -> np.ndarray:
    """``eta2(h)`` for the bin widths ``h = m * delta``: adjacent pairs of bins, pooled over a few shifts of the grid."""
    n, n_fine = C.shape
    zbar = C.mean(axis=0) / delta
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(C, axis=1)], axis=1)
    out = np.empty(len(ms))
    for j, m in enumerate(ms):
        m = int(m)
        g2 = _tent_weights(m, delta)
        num = wt = 0.0
        for o in sorted(set(int(round(a * m / n_offsets)) for a in range(n_offsets))):
            K = (n_fine - o) // m
            if K < 2:
                continue
            edges = o + m * np.arange(K + 1)
            coarse = cum[:, edges[1:]] - cum[:, edges[:-1]]                         # [n, K]
            D, S = coarse[:, 1:] - coarse[:, :-1], coarse[:, 1:] + coarse[:, :-1]
            num += float((D.var(axis=0, ddof=1) - S.mean(axis=0)).sum())
            windows = np.lib.stride_tricks.sliding_window_view(zbar[o:o + K * m], 2 * m)[::m]
            wt += float((windows @ g2).sum())
        out[j] = num / wt if wt > 0 else float("nan")
    return out


def _wls(X: np.ndarray, Y: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Weighted least squares of every row of ``Y`` on the columns of ``X`` (weights ``1 / se^2``): ``[rows, columns]``."""
    w = 1.0 / np.asarray(se, dtype=np.float64) ** 2
    A = X.T @ (X * w[:, None])
    return np.linalg.solve(A, X.T @ (w[:, None] * np.atleast_2d(Y).T)).T


def _design(h: np.ndarray, groups: bool, bend: bool = False) -> np.ndarray:
    cols = [np.ones_like(h), h]
    if groups:
        cols.append(1.0 / h ** 2)
    if bend:
        cols.append(h ** 2)
    return np.column_stack(cols)


def _z(X: np.ndarray, y: np.ndarray, boot: np.ndarray, se: np.ndarray, column: int) -> float:
    """Coefficient ``column`` of the fit over its bootstrap standard deviation."""
    value = _wls(X, y, se)[0, column]
    sd = _wls(X, boot, se)[:, column].std(ddof=1)
    return float(value / sd) if sd > 0 and np.isfinite(sd) else 0.0


def estimate_from_counts(C: np.ndarray, T: float, widths: Sequence[int] = WIDTHS, n_boot: int = 400, seed: int = 0,
                         floor_cv: float = FLOOR_CV) -> EtaEstimate:
    """``estimate_cir_eta`` for counts on a fine grid, ``C[i, j]`` events of observation ``i`` in fine bin ``j``."""
    C = np.asarray(C, dtype=np.float64)
    n, n_fine = C.shape
    if C.sum() <= 0:
        raise ValueError("the observations contain no events")
    delta = float(T) / n_fine
    ms = [int(m) for m in widths if 2 * int(m) <= n_fine]
    h = np.array(ms, dtype=np.float64) * delta
    mean_events = float(C.sum() / n)
    floor = float(floor_cv) * math.sqrt(mean_events / float(T) / float(T))
    if n < MIN_OBSERVATIONS:                     # the estimate rests on the variation between observations
        return EtaEstimate(eta=floor, eta2=0.0, eta2_se=float("inf"), interval=(0.0, float("inf")), floor=floor, floored=True,
                           with_groups=False, widths=[float(v) for v in h], curve=[], curve_se=[], used=[], coefficients=[],
                           n_observations=int(n), mean_events=mean_events)
    curve = eta2_curve(C, ms, delta)
    rng = np.random.default_rng(seed)
    boot = np.stack([eta2_curve(C[rng.integers(0, n, n)], ms, delta) for _ in range(int(n_boot))])
    se = boot.std(axis=0, ddof=1)
    se = np.where(np.isfinite(se) & (se > 0), se, np.nanmax(np.abs(curve)) + 1.0)

    use = np.isfinite(curve)
    while True:
        hu, yu, bu, su = h[use], curve[use], boot[:, use], se[use]
        groups = len(hu) >= 4 and _z(_design(hu, True), yu, bu, su, 2) > Z_KEEP
        enough = int(use.sum()) > max(MIN_WIDTHS, 3 + int(groups))
        if enough and abs(_z(_design(hu, groups, bend=True), yu, bu, su, 2 + int(groups))) > Z_KEEP:
            use[np.flatnonzero(use)[-1]] = False        # the curve bends over these widths: leave out the widest
            continue
        break
    X = _design(hu, groups)
    beta = _wls(X, yu, su)[0]
    intercepts = _wls(X, bu, su)[:, 0]
    eta2, eta2_se = float(beta[0]), float(intercepts.std(ddof=1))
    low, high = (float(v) for v in np.quantile(intercepts, [0.025, 0.975]))
    eta = math.sqrt(eta2) if eta2 > 0 else 0.0
    return EtaEstimate(eta=max(eta, floor), eta2=eta2, eta2_se=eta2_se, interval=(math.sqrt(max(low, 0.0)), math.sqrt(max(high, 0.0))),
                       floor=floor, floored=eta < floor, with_groups=bool(groups), widths=[float(v) for v in h],
                       curve=[float(v) for v in curve], curve_se=[float(v) for v in se], used=[bool(v) for v in use],
                       coefficients=[float(v) for v in beta], n_observations=int(n), mean_events=mean_events)


def estimate_cir_eta(events: Any, T: float, n_fine: int = N_FINE, widths: Sequence[int] = WIDTHS, n_boot: int = 400, seed: int = 0,
                     floor_cv: float = FLOOR_CV) -> EtaEstimate:
    """Estimate ``eta`` of the intensity diffusion ``eta * sqrt(Z)`` from independent observations on ``[0, T]``.

    ``events``: a 2-D array or tensor of event times padded with ``inf`` / ``nan``, or one sequence per observation.
    ``widths`` are the bin widths in units of ``T / n_fine``; the defaults span ``T/288`` to ``T/12``.

    Thinning: keeping every event with probability ``p`` turns the intensity ``Z`` into ``p Z``, whose coefficient is
    ``sqrt(p) * eta``. An estimate made on all events is converted with that factor.
    """
    return estimate_from_counts(fine_counts(events, T, n_fine), T, widths=widths, n_boot=n_boot, seed=seed, floor_cv=floor_cv)


def save_eta_estimate(estimate: EtaEstimate, out_dir: Path, unit: str = "time units of the data", title: Optional[str] = None) -> None:
    """Write ``eta_estimate.json`` and ``eta_estimate.png`` (the curve ``eta2(h)`` and its fit) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eta_estimate.json").write_text(json.dumps(estimate.to_dict(), indent=2))
    if estimate.curve:                           # nothing to draw when there were too few observations for an estimate
        from .plots import plot_eta_estimate

        plot_eta_estimate(estimate, out_dir / "eta_estimate.png", unit=unit, title=title)


def wants_eta_estimate(cfg: Any) -> bool:
    return cfg.model.diffusion.eta_init == "auto"


def resolve_eta(cfg: Any, train: Any, out_dir: Optional[Path] = None) -> Optional[EtaEstimate]:
    """Replace ``model.diffusion.eta_init: auto`` by the estimate from the training observations, so that the config
    saved with the run carries the number. Returns the estimate (``None`` when ``eta_init`` is a number already) and,
    with ``out_dir``, writes ``eta_estimate.json`` / ``.png`` there."""
    if not wants_eta_estimate(cfg):
        return None
    estimate = estimate_cir_eta(train.events, float(cfg.data.T))
    cfg.model.diffusion.eta_init = float(f"{estimate.eta:.4g}")
    if out_dir is not None:
        save_eta_estimate(estimate, out_dir, title=f"{cfg.name}: coefficient of the diffusion from the {len(train)} training observations")
    return estimate
