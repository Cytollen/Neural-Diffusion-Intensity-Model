"""Data: synthetic Cox-process observations (ground-truth CIR intensity paths and exact thinning, paper App. G)
and observations of an unspecified model read from a file (``read_event_file`` / ``load_event_file``).

Observations are stored padded: ``events`` is a ``[n_obs, K_max]`` float tensor padded with ``+inf``
(so ``events <= T'`` masks are trivial) and ``counts`` is ``[n_obs]``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import DataConfig, ExperimentConfig, TruthConfig, model_x_start, model_z0, z0_mode

Tensor = torch.Tensor


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------
def make_truth_drift(truth: TruthConfig) -> Callable[[Tensor, Tensor], Tensor]:
    """``b(z, t) = kappa (theta - z)`` on tensors (broadcasting); the time is accepted and not used."""
    kappa, theta = float(truth.kappa), float(truth.theta)

    def drift(z: Tensor, t: Tensor) -> Tensor:
        return kappa * (theta - z) + 0.0 * t

    return drift


def make_truth_diffusion(truth: TruthConfig) -> Callable[[Tensor, Tensor], Tensor]:
    eta = float(truth.eta)

    def diffusion(z: Tensor, t: Tensor) -> Tensor:
        return eta * torch.sqrt(torch.relu(z))

    return diffusion


def _z0_components(truth: TruthConfig) -> Tuple[Tensor, Tensor]:
    """Means and normalised weights (float64) of the normal components of the truth's initial intensity: one
    component at ``z0``, or the mixture ``z0_modes`` / ``z0_weights``."""
    if truth.z0_modes is None:
        return torch.tensor([float(truth.z0)], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    means = torch.tensor([float(m) for m in truth.z0_modes], dtype=torch.float64)
    w = torch.ones_like(means) if truth.z0_weights is None else torch.tensor([float(v) for v in truth.z0_weights], dtype=torch.float64)
    return means, w / w.sum()


def _z0_bounds(truth: TruthConfig) -> Tuple[float, float]:
    return float(truth.z0_low), float("inf") if truth.z0_high is None else float(truth.z0_high)


def _z0_untruncated_cdf(truth: TruthConfig, x: Tensor) -> Tensor:
    means, w = _z0_components(truth)
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    return (w * normal.cdf((x.unsqueeze(-1) - means) / float(truth.z0_std))).sum(dim=-1)


def truth_z0_cdf(truth: TruthConfig, x: Tensor) -> Tensor:
    """Exact distribution function of the truth's initial intensity at ``x`` (float64): the normal, or the mixture of
    normals, truncated to ``[z0_low, z0_high]``. A fixed start (``z0_std == 0``) gives the step at ``z0``."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if float(truth.z0_std) <= 0:
        return (x >= float(truth.z0)).to(torch.float64)
    lo, hi = _z0_bounds(truth)
    a = _z0_untruncated_cdf(truth, torch.tensor(lo, dtype=torch.float64))
    b = torch.tensor(1.0, dtype=torch.float64) if hi == float("inf") else _z0_untruncated_cdf(truth, torch.tensor(hi, dtype=torch.float64))
    return ((_z0_untruncated_cdf(truth, x.clamp(lo, hi)) - a) / (b - a)).clamp(0.0, 1.0)


def truth_z0_pdf(truth: TruthConfig, x: Tensor) -> Tensor:
    """Exact density of the truth's initial intensity at ``x`` (float64), zero outside ``[z0_low, z0_high]``.
    A fixed start has no density."""
    if float(truth.z0_std) <= 0:
        raise ValueError("a fixed initial intensity (data.truth.z0_std = 0) has no density")
    x = torch.as_tensor(x, dtype=torch.float64)
    means, w = _z0_components(truth)
    sd = float(truth.z0_std)
    lo, hi = _z0_bounds(truth)
    dens = (w * torch.exp(-0.5 * ((x.unsqueeze(-1) - means) / sd) ** 2)).sum(dim=-1) / (sd * float(np.sqrt(2.0 * np.pi)))
    a = _z0_untruncated_cdf(truth, torch.tensor(lo, dtype=torch.float64))
    b = torch.tensor(1.0, dtype=torch.float64) if hi == float("inf") else _z0_untruncated_cdf(truth, torch.tensor(hi, dtype=torch.float64))
    return torch.where((x >= lo) & (x <= hi), dens / (b - a), torch.zeros_like(dens))


def describe_truth_z0(truth: TruthConfig) -> str:
    """The truth's initial intensity in words, for logs: ``Z0 = 5``, or its distribution."""
    if float(truth.z0_std) <= 0:
        return f"Z0 = {truth.z0:g}"
    bounds = f"truncated to [{truth.z0_low:g}, {'inf' if truth.z0_high is None else format(truth.z0_high, 'g')}]"
    if truth.z0_modes is None:
        return f"Z0 ~ normal(mean {truth.z0:g}, sd {truth.z0_std:g}) {bounds}"
    _, w = _z0_components(truth)
    return (f"Z0 ~ mixture of normals at {', '.join(format(float(m), 'g') for m in truth.z0_modes)} (sd {truth.z0_std:g} each, "
            f"weights {', '.join(format(float(v), '.3g') for v in w)}) {bounds}")


def truth_z0_from_normal(truth: TruthConfig, zeta: Tensor) -> Tensor:
    """Initial intensity of the truth as a function of standard-normal draws ``zeta``.

    ``truth.z0_std == 0``: the constant ``truth.z0``. Otherwise the normal with mean ``z0`` and standard deviation
    ``z0_std`` truncated to ``[z0_low, z0_high]``, or with ``z0_modes`` the mixture of such normals truncated as a
    whole, through the quantile function, so that ``zeta ~ N(0, 1)`` gives exactly that distribution (the path-space
    MCMC moves ``zeta`` together with the Brownian noise). The map is increasing in ``zeta``: for a mixture the lower
    modes are reached by negative ``zeta``, and the prior of ``zeta`` stays one normal.
    """
    if float(truth.z0_std) <= 0:
        return torch.full_like(zeta, float(truth.z0))
    mu, sd, lo = float(truth.z0), float(truth.z0_std), float(truth.z0_low)
    hi = float("inf") if truth.z0_high is None else float(truth.z0_high)
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    if truth.z0_modes is not None:                      # a mixture: its truncated distribution function, inverted on a fine grid
        u = normal.cdf(zeta.to(torch.float64)).reshape(-1)
        grid = torch.linspace(lo, hi, 8193, dtype=torch.float64)
        cdf = truth_z0_cdf(truth, grid)
        idx = torch.searchsorted(cdf, u.contiguous()).clamp(1, grid.numel() - 1)
        c0, c1 = cdf[idx - 1], cdf[idx]
        frac = ((u - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0.0, 1.0)
        out = grid[idx - 1] + frac * (grid[idx] - grid[idx - 1])
        return out.reshape(zeta.shape).clamp(lo, hi).to(zeta.dtype)
    a = float(normal.cdf(torch.tensor((lo - mu) / sd, dtype=torch.float64)))
    b = 1.0 if hi == float("inf") else float(normal.cdf(torch.tensor((hi - mu) / sd, dtype=torch.float64)))
    z = zeta.to(torch.float64)
    if a < 1e-12 and b > 1.0 - 1e-12:                   # the bounds are far in the tails: the normal itself
        out = mu + sd * z
    else:
        u = (a + normal.cdf(z) * (b - a)).clamp(1e-15, 1.0 - 1e-15)
        out = mu + sd * normal.icdf(u)
    return out.clamp(lo, hi).to(zeta.dtype)


def sample_truth_z0(truth: TruthConfig, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
    """``n`` initial intensities of the truth (no random number is drawn when the truth starts at a fixed value)."""
    if float(truth.z0_std) <= 0:
        return torch.full((int(n),), float(truth.z0))
    return truth_z0_from_normal(truth, torch.randn(int(n), generator=generator))


@torch.no_grad()
def simulate_truth_paths(
    truth: TruthConfig,
    T: float,
    n_steps: int,
    n_paths: int,
    generator: Optional[torch.Generator] = None,
    brownian: Optional[Tensor] = None,
    substeps: int = 1,
) -> Tensor:
    """Euler–Maruyama (full truncation) paths of the true CIR prior on the grid ``t_k = k T/n_steps``.

    If ``substeps > 1`` the SDE is integrated on a grid ``substeps`` times finer and subsampled.
    Returns ``[n_paths, n_steps + 1]`` (positive part of the state).
    """
    drift = make_truth_drift(truth)
    diffusion = make_truth_diffusion(truth)
    n_fine = int(n_steps) * int(substeps)
    dt_fine = float(T) / n_fine
    if brownian is None:
        brownian = torch.randn(n_paths, n_fine, generator=generator) * np.sqrt(dt_fine)
    else:
        assert brownian.shape == (n_paths, n_fine)
    z = sample_truth_z0(truth, n_paths, generator)       # drawn after the Brownian increments; a constant for a fixed start
    out = torch.empty(n_paths, n_fine + 1)
    out[:, 0] = z
    for k in range(n_fine):
        t = torch.full_like(z, k * dt_fine)
        zp = torch.relu(z)
        z = z + drift(zp, t) * dt_fine + diffusion(zp, t) * brownian[:, k]
        out[:, k + 1] = z
    out = torch.relu(out)
    return out[:, :: int(substeps)]


@torch.no_grad()
def thin_events(paths: Tensor, T: float, generator: Optional[torch.Generator] = None) -> List[np.ndarray]:
    """Exact thinning of an inhomogeneous Poisson process whose intensity is piecewise constant on
    the grid of ``paths`` (``[n_paths, n_grid]``; value on ``[t_k, t_{k+1})`` is ``paths[:, k]``).
    Returns a list of sorted event-time arrays (paper App. G, Algorithm 2).
    """
    n_paths, n_grid = paths.shape
    dt = float(T) / (n_grid - 1)
    lam_max = paths.max(dim=1).values.clamp_min(1e-8)  # [n_paths]
    n_cand = torch.poisson(lam_max * float(T), generator=generator).long()  # [n_paths]
    K = int(n_cand.max().item()) if n_paths > 0 else 0
    if K == 0:
        return [np.zeros(0, dtype=np.float64) for _ in range(n_paths)]
    u = torch.rand(n_paths, K, generator=generator) * float(T)
    idx = torch.clamp((u / dt).floor().long(), max=n_grid - 2)
    lam_at = torch.gather(paths, 1, idx)
    accept = torch.rand(n_paths, K, generator=generator) < (lam_at / lam_max[:, None])
    valid = torch.arange(K)[None, :] < n_cand[:, None]
    keep = accept & valid
    out: List[np.ndarray] = []
    for i in range(n_paths):
        ev = u[i][keep[i]].sort().values.numpy().astype(np.float64)
        out.append(ev)
    return out


# --------------------------------------------------------------------------------------
# Dataset container
# --------------------------------------------------------------------------------------
@dataclass
class EventData:
    """Padded event times for a set of observations."""
    events: Tensor          # [n_obs, K_max], padded with +inf
    counts: Tensor          # [n_obs] long
    T: float

    def __len__(self) -> int:
        return int(self.events.shape[0])

    @property
    def K_max(self) -> int:
        return int(self.events.shape[1])

    def to(self, device: torch.device) -> "EventData":
        return EventData(self.events.to(device), self.counts.to(device), self.T)

    def subset(self, idx: Sequence[int]) -> "EventData":
        idx_t = torch.as_tensor(list(idx), dtype=torch.long, device=self.events.device)
        ev = self.events.index_select(0, idx_t)
        cnt = self.counts.index_select(0, idx_t)
        kmax = int(cnt.max().item()) if len(idx) > 0 else 0
        return EventData(ev[:, : max(kmax, 1)], cnt, self.T)

    def as_lists(self) -> List[np.ndarray]:
        return [self.events[i, : int(self.counts[i])].cpu().numpy() for i in range(len(self))]

    def counts_before(self, t_bar: float) -> Tensor:
        return (self.events <= float(t_bar)).sum(dim=1)

    @staticmethod
    def from_lists(event_lists: Sequence[np.ndarray], T: float) -> "EventData":
        counts = torch.tensor([len(e) for e in event_lists], dtype=torch.long)
        kmax = int(counts.max().item()) if len(event_lists) > 0 else 0
        ev = torch.full((len(event_lists), max(kmax, 1)), float("inf"))
        for i, e in enumerate(event_lists):
            if len(e) > 0:
                ev[i, : len(e)] = torch.as_tensor(np.asarray(e, dtype=np.float64), dtype=torch.float32)
        return EventData(ev, counts, float(T))


@dataclass
class Dataset:
    train: EventData
    test: EventData
    # true latent paths behind the observations, [n, n_steps+1]; None for observations of an unspecified model
    train_paths: Optional[Tensor]
    test_paths: Optional[Tensor]
    cfg: DataConfig

    @property
    def has_truth(self) -> bool:
        """Whether the observations were generated from ``cfg.truth`` (synthetic data), so that a truth exists to compare with."""
        return self.train_paths is not None and self.test_paths is not None

    def save(self, path: Path) -> None:
        if not self.has_truth:
            raise ValueError("only synthetic datasets are cached; observations read from a file stay in their file")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_events=self.train.events.numpy(),
            train_counts=self.train.counts.numpy(),
            test_events=self.test.events.numpy(),
            test_counts=self.test.counts.numpy(),
            train_paths=self.train_paths.numpy(),
            test_paths=self.test_paths.numpy(),
            T=np.array(self.cfg.T),
        )

    @staticmethod
    def load(path: Path, cfg: DataConfig) -> "Dataset":
        z = np.load(path)
        T = float(z["T"])
        return Dataset(
            train=EventData(torch.from_numpy(z["train_events"]), torch.from_numpy(z["train_counts"]), T),
            test=EventData(torch.from_numpy(z["test_events"]), torch.from_numpy(z["test_counts"]), T),
            train_paths=torch.from_numpy(z["train_paths"]),
            test_paths=torch.from_numpy(z["test_paths"]),
            cfg=cfg,
        )


def dataset_filename(cfg: DataConfig) -> str:
    tr = cfg.truth
    start = ""
    if tr.z0_std > 0:           # random initial intensity: its standard deviation and bounds are part of the name
        if tr.z0_modes is not None:     # a mixture: its means, and its weights when they are given
            start += "_z0mix" + "-".join(f"{float(m):g}" for m in tr.z0_modes)
            if tr.z0_weights is not None:
                start += "_w" + "-".join(f"{float(w):g}" for w in tr.z0_weights)
        start += f"_z0sd{tr.z0_std:g}_{tr.z0_low:g}-{'inf' if tr.z0_high is None else format(tr.z0_high, 'g')}"
    return (
        f"cir_k{tr.kappa:g}_th{tr.theta:g}_eta{tr.eta:g}{start}_z0{tr.z0:g}"
        f"_T{cfg.T:g}_N{cfg.n_steps}_sub{cfg.sim_substeps}_ntr{cfg.n_train}_nte{cfg.n_test}_seed{cfg.seed}.npz"
    )


def generate_dataset(cfg: DataConfig) -> Dataset:
    g = torch.Generator().manual_seed(int(cfg.seed))
    n = int(cfg.n_train) + int(cfg.n_test)
    paths = simulate_truth_paths(cfg.truth, cfg.T, cfg.n_steps, n, generator=g, substeps=int(cfg.sim_substeps))
    events = thin_events(paths, cfg.T, generator=g)
    train = EventData.from_lists(events[: cfg.n_train], cfg.T)
    test = EventData.from_lists(events[cfg.n_train :], cfg.T)
    return Dataset(train=train, test=test, train_paths=paths[: cfg.n_train], test_paths=paths[cfg.n_train :], cfg=cfg)


def estimate_initial_intensity(events: Tensor, T: float, window: float = 0.125) -> float:
    """Estimate of a fixed initial intensity ``Z_0`` from observations of an unspecified model: the mean event rate
    over the first ``window`` of ``[0, T]``, i.e. the mean number of events in ``(0, window * T]`` divided by
    ``window * T``. ``events`` is ``[n_obs, K]`` padded with ``+inf``.

    The intensity is taken as constant over that window. If it rises or falls there, the estimate is its average
    over the window, not its value at 0.
    """
    if not 0.0 < float(window) <= 1.0:
        raise ValueError("window must be in (0, 1]")
    if events.shape[0] == 0:
        raise ValueError("no observations to estimate the initial intensity from")
    span = float(window) * float(T)
    counts = ((events > 0) & (events <= span)).sum(dim=1).to(torch.float64)
    return float(counts.mean() / span)


def resolve_z0(cfg: ExperimentConfig, train: EventData, has_truth: bool = True) -> float:
    """Initial intensity of the model for these training observations; an ``estimate`` is computed and written into
    ``cfg.model.z0``, so that the config saved with the run carries the number. Without a truth (``has_truth`` false)
    ``auto`` means ``estimate``."""
    if z0_mode(cfg) == "estimate" or (cfg.model.z0 == "auto" and not has_truth):
        cfg.model.z0 = estimate_initial_intensity(train.events, float(cfg.data.T), float(cfg.model.z0_window))
    return model_z0(cfg)


def resolve_x_start(cfg: ExperimentConfig, train: EventData) -> Optional[float]:
    """Latent state at the start of the burn-in for these training observations (``None`` without a burn-in).

    ``auto`` is replaced in ``cfg.model.x_start`` by the latent value of the mean event rate over the first
    ``model.z0_window`` of the window (over the whole window if no event falls there), so that the config saved with
    the run carries the number and the paths start on the scale of the data."""
    if cfg.burn_steps == 0:
        return None
    if cfg.model.x_start == "auto":
        from .links import build_link

        T = float(cfg.data.T)
        rate = estimate_initial_intensity(train.events, T, float(cfg.model.z0_window))
        if rate <= 0:
            rate = float(train.counts.to(torch.float64).mean()) / T
        if rate <= 0:
            raise ValueError("the training observations contain no events: model.x_start cannot be taken from them")
        cfg.model.x_start = float(build_link(cfg.model.link).inverse_scalar(rate))
    return model_x_start(cfg)


def _event_rows(rows: Any, name: str, T: float) -> List[np.ndarray]:
    """Sorted event times of every observation. ``rows`` is a 2-D array padded with ``inf`` / ``nan``, or a sequence
    with one sequence of event times per observation (they may differ in length)."""
    if isinstance(rows, np.ndarray) and rows.dtype != object:
        if rows.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array, one row of event times per observation (padded with inf or nan)")
        rows = list(rows)
    out = []
    for k, row in enumerate(rows):
        try:
            a = np.asarray(row, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: observation {k} is not a sequence of event times") from exc
        ev = np.sort(a[np.isfinite(a)])
        if ev.size and (ev[0] <= 0.0 or ev[-1] > float(T)):
            raise ValueError(f"{name} has event times outside (0, T] with T = {T:g} (observation {k}: {ev[0]:g} to {ev[-1]:g}). "
                             "T is the length of the observation window (--T of scripts/fit.py, data.T in a config), and the "
                             "times are measured from its start")
        out.append(ev)
    return out


@dataclass
class EventFile:
    """Content of a file of observations, as it was stored (``read_event_file``)."""
    events: Optional[List[np.ndarray]] = None     # all the observations, when the file does not split them
    train: Optional[List[np.ndarray]] = None      # the file's own split
    test: Optional[List[np.ndarray]] = None
    T: Optional[float] = None                     # length of the observation window, when the file records it


_TEXT_SUFFIXES = (".csv", ".txt", ".tsv", ".dat")
_TIME_COLUMNS = ("time", "t", "event_time", "times", "timestamp", "tau")
_ID_COLUMNS = ("observation", "obs", "id", "sequence", "seq", "sample", "path", "unit", "subject", "day", "date", "series")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_event_text(path: Path) -> List[np.ndarray]:
    """Observations from a text file. Lines starting with ``#`` and blank lines are skipped.

    * One observation per line, its event times separated by commas, semicolons or blanks (the lines may differ in
      length; a line holding only a separator is an observation without events).
    * Or, when the first line is a header, one event per line with a column naming the observation and a column
      holding the event time: ``observation,time``. The time column is the one called ``time`` (or ``t``,
      ``event_time``), otherwise the last; the observation column is the one called ``observation`` (or ``obs``,
      ``id``, ``sequence``, ``day``, ...), otherwise the first other column. Observations are kept in their order of
      first appearance; one without any event cannot be written in this layout.
    """
    # utf-8-sig drops the byte-order mark that spreadsheet programs write at the start of a csv
    lines = [(k, ln.strip()) for k, ln in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)]
    lines = [(k, ln) for k, ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"{path} holds no observations")

    def tokens(ln: str, pattern: str = r"[,;\s]+") -> List[str]:
        return [tok.strip().strip('"\'') for tok in re.split(pattern, ln)]

    first = [tok for tok in tokens(lines[0][1]) if tok]
    if first and not all(_is_number(tok) for tok in first):          # a header: one event per line
        pattern = r"[,;\t]" if re.search(r"[,;\t]", lines[0][1]) else r"\s+"
        header = [tok.lower() for tok in tokens(lines[0][1], pattern)]
        if len(header) < 2:
            raise ValueError(f"{path}: with a header the file needs two columns, the observation and the event time")
        t_col = next((header.index(name) for name in _TIME_COLUMNS if name in header), len(header) - 1)
        others = [c for c in range(len(header)) if c != t_col]
        id_col = next((header.index(name) for name in _ID_COLUMNS if name in header and header.index(name) != t_col), others[0])
        groups: Dict[str, List[float]] = {}
        for k, ln in lines[1:]:
            tok = tokens(ln, pattern)
            if len(tok) <= max(t_col, id_col) or not _is_number(tok[t_col]):
                raise ValueError(f"{path}, line {k}: expected the columns {', '.join(header)} with a number in '{header[t_col]}'")
            groups.setdefault(tok[id_col], []).append(float(tok[t_col]))
        if not groups:
            raise ValueError(f"{path} holds a header and no events")
        return [np.asarray(v, dtype=np.float64) for v in groups.values()]
    rows = []
    for k, ln in lines:
        tok = [t for t in tokens(ln) if t]
        if not all(_is_number(t) for t in tok):
            raise ValueError(f"{path}, line {k}: event times must be numbers")
        rows.append(np.asarray([float(t) for t in tok], dtype=np.float64))
    return rows


def read_event_file(path: Union[str, Path]) -> EventFile:
    """Read observations of a point process from a file, without checking them against a window.

    * ``.npz``: the arrays ``train_events`` and ``test_events``, or one array ``events`` (the split is then made by
      ``load_event_file``); one row of event times per observation, padded with ``inf`` or ``nan``. ``T`` is optional.
    * ``.npy``: one such array with all the observations.
    * ``.json``: a list with one list of event times per observation, or an object with the keys of the ``.npz``.
    * ``.csv`` / ``.txt`` / ``.tsv`` / ``.dat``: text, see ``_read_event_text``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data.path not found: {path}")
    suffix = path.suffix.lower()

    def from_mapping(get: Callable[[str], Any], keys: Sequence[str]) -> EventFile:
        T = float(np.asarray(get("T")).reshape(-1)[0]) if "T" in keys else None
        if "train_events" in keys and "test_events" in keys:
            return EventFile(train=list(get("train_events")), test=list(get("test_events")), T=T)
        if "events" in keys:
            return EventFile(events=list(get("events")), T=T)
        raise ValueError(f"{path} needs the entry events (all the observations), or train_events and test_events")

    if suffix == ".npz":
        with np.load(path) as z:
            content = from_mapping(lambda k: np.asarray(z[k]), z.files)
    elif suffix == ".npy":
        content = EventFile(events=list(np.load(path)))
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        content = from_mapping(lambda k: obj[k], list(obj)) if isinstance(obj, dict) else EventFile(events=list(obj))
    elif suffix in _TEXT_SUFFIXES:
        content = EventFile(events=_read_event_text(path))
    else:
        raise ValueError(f"{path}: unknown file type '{suffix}'. Observations are read from .npz, .npy, .json and text files "
                         f"({', '.join(_TEXT_SUFFIXES)})")
    return content


def split_observations(n: int, test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Indices of the training and of the test observations among ``n``: ``round(test_fraction * n)`` of them (at
    least one, at most ``n - 1``) are held out, picked by a permutation seeded with ``seed``. Both lists are sorted."""
    if n < 2:
        raise ValueError("at least two observations are needed: one to train on and one to test on")
    n_test = min(max(int(round(float(test_fraction) * n)), 1), n - 1)
    perm = np.random.default_rng(int(seed)).permutation(n)
    return sorted(int(i) for i in perm[n_test:]), sorted(int(i) for i in perm[:n_test])


def load_event_file(cfg: DataConfig) -> Dataset:
    """Observations of an unspecified model from ``cfg.path`` (formats: ``read_event_file``). Every event time must
    lie in ``(0, T]`` with ``T = cfg.T``; a ``T`` stored in the file must agree with it. A file that does not split
    its observations is split here (``cfg.test_fraction``, ``cfg.seed``). There are no latent paths and no truth."""
    path = Path(cfg.path)
    content = read_event_file(path)
    if content.T is not None and abs(content.T - float(cfg.T)) > 1e-9:
        raise ValueError(f"{path} was recorded on [0, {content.T:g}] but data.T is {float(cfg.T):g}")
    if content.events is not None:
        rows = _event_rows(content.events, "events", cfg.T)
        train_idx, test_idx = split_observations(len(rows), float(cfg.test_fraction), int(cfg.seed))
        train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
    else:
        train, test = _event_rows(content.train, "train_events", cfg.T), _event_rows(content.test, "test_events", cfg.T)
    if not train or not test:
        raise ValueError(f"{path} needs at least one training and one test observation")
    return Dataset(train=EventData.from_lists(train, cfg.T), test=EventData.from_lists(test, cfg.T), train_paths=None, test_paths=None, cfg=cfg)


def suggest_scales(train: EventData) -> Dict[str, float]:
    """The two scales that normalise the network inputs, taken from training observations: ``model.intensity_scale``
    (their mean event rate) and ``model.posterior.count_scale`` (their mean number of events), three significant
    digits each. Returned under their config keys."""
    mean_count = max(float(train.counts.to(torch.float64).mean()), 1.0)

    def three_digits(x: float) -> float:
        return float(f"{x:.3g}")

    return {"model.intensity_scale": three_digits(mean_count / float(train.T)), "model.posterior.count_scale": three_digits(mean_count)}


def load_or_generate_dataset(cfg: ExperimentConfig, root: Optional[Path] = None, force: bool = False) -> Tuple[Dataset, Path]:
    if cfg.data.source == "file":
        return load_event_file(cfg.data), Path(cfg.data.path)
    root = Path(root) if root is not None else Path(cfg.data.cache_dir)
    path = root / dataset_filename(cfg.data)
    if path.exists() and not force:
        return Dataset.load(path, cfg.data), path
    ds = generate_dataset(cfg.data)
    ds.save(path)
    return ds, path


# --------------------------------------------------------------------------------------
# Batch helpers
# --------------------------------------------------------------------------------------
def clip_to_horizon(events: Tensor, t_bar: float) -> Tensor:
    """Replace events after ``t_bar`` by ``+inf`` (keeps shape)."""
    return torch.where(events <= float(t_bar), events, torch.full_like(events, float("inf")))


def future_counts_on_grid(events: Tensor, t_grid: Tensor, t_bar: float) -> Tensor:
    """``r[b, k] = #{i : t_k < tau_{b,i} <= t_bar}`` -> ``[B, len(t_grid)]`` (float)."""
    ev = events[:, None, :]                    # [B, 1, K]
    tg = t_grid[None, :, None]                 # [1, G, 1]
    mask = (ev > tg) & (ev <= float(t_bar))
    return mask.sum(dim=2).to(torch.float32)
