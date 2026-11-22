"""Quantitative evaluation metrics (paper Sec. 4).

All functions run on CPU ``float32`` under ``torch.no_grad()`` and draw every random number from the
``torch.Generator`` passed by the caller. Conventions follow ``docs/DESIGN.md`` Sec. 6: paths are
``[rows, N+1]`` with row ``b*m + j`` = path ``j`` of observation ``b``; events are ``[n_obs, K]`` padded
with ``+inf``; horizons are grid multiples ``n' = T'/dt``.

Path-space integrals ``int_0^T f(Z_t) dt`` are discretised as ``dt * sum_k f(Z_{t_k})`` over all ``N+1``
grid points, both for the path deviation (reported in its L2 and L1 versions) and for the Wasserstein cost.

Model paths evolve in X-space and are returned as intensities g(X).

The reference posterior of the comparison metrics (posterior paths, Wasserstein distances) is
``cfg.eval.posterior_reference``: the exact posterior of the CIR truth (``analytic``) or path-space MCMC under
the true prior (``mcmc``).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import ExperimentConfig, MCMCConfig, model_z0
from .data import thin_events, make_truth_diffusion, make_truth_drift, sample_truth_z0
from .likelihood import event_grid_index
from .mcmc import DEFAULT_POSITIVITY, sample_posterior_mcmc_batch
from .model import NDIM
from .links import build_link
from .sde import brownian_increments, simulate_prior
from .utils import timer

Tensor = torch.Tensor
DriftFn = Callable[[Tensor, Tensor], Tensor]
ArrayLike = Union[np.ndarray, Tensor, Sequence[float]]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _to_numpy(x: ArrayLike, dtype: Any = np.float64) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _cpu_generator(generator: Optional[torch.Generator]) -> Optional[torch.Generator]:
    """A CPU generator for CPU-only samplers (``thin_events``), derived deterministically from ``generator``."""
    if generator is None or generator.device.type == "cpu":
        return generator
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), device=generator.device, generator=generator).item())
    return torch.Generator(device="cpu").manual_seed(seed)


def resolve_posterior_reference(cfg: ExperimentConfig, reference: Optional[str] = None) -> str:
    """Reference posterior of the comparison figures: ``analytic``, ``mcmc`` or ``none``.

    ``reference`` overrides ``cfg.eval.posterior_reference``. The analytic reference is the posterior under
    the CIR truth in ``cfg.data.truth``, never the learned prior, whether the learned drift is an MLP or
    parametric. It needs a positive immigration rate ``kappa * theta`` and a known initial intensity.
    """
    requested = str(cfg.eval.posterior_reference if reference is None else reference)
    if requested not in {"analytic", "mcmc", "none"}:
        raise ValueError(f"Unknown posterior reference {requested!r}")
    if requested == "analytic":
        truth = cfg.data.truth
        immigration = float(truth.kappa) * float(truth.theta)
        if immigration <= 0.:
            raise ValueError("The CIR posterior reference requires a positive immigration rate kappa * theta")
    return requested


@torch.no_grad()
def posterior_reference_samples(
    events: Tensor, cfg: ExperimentConfig, n_prime: int, n_paths: int,
    device: torch.device, generator: Optional[torch.Generator] = None, *,
    reference: Optional[str] = None, drift: Optional[DriftFn] = None,
    diffusion: Optional[DriftFn] = None, mcmc_cfg: Optional[MCMCConfig] = None,
    mcmc_positivity: str = DEFAULT_POSITIVITY,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Sample a truth-conditioned comparison posterior, with explicit metadata.

    Returns flat intensity paths ``[batch*n_paths,N+1]``.  Analytic reference
    construction uses only ``cfg.data.truth`` and the observations; learned
    model parameters never enter it.  The MCMC reference samples the posterior
    under ``drift`` / ``diffusion`` (the truth by default).
    """
    kind = resolve_posterior_reference(cfg, reference)
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if kind == "none":
        raise ValueError("Posterior reference sampling is disabled")
    if kind == "analytic":
        from .cir_posterior import sample_true_cir_posterior
        # the reference is sampled on the CPU in float64, whatever the device of the model
        paths, metadata = sample_true_cir_posterior(
            events.detach().cpu(), cfg, int(n_prime), n_paths,
            device=torch.device("cpu"), generator=_cpu_generator(generator))
        metadata = dict(metadata)
        metadata.setdefault("kind", "analytic")
        metadata.setdefault("label", "True CIR analytic posterior")
        metadata.setdefault("sampling_device", "cpu")
        return torch.as_tensor(paths).to(device).reshape(-1, cfg.data.n_steps + 1), metadata
    used = dataclasses.replace(mcmc_cfg or cfg.eval.mcmc, n_chains=n_paths)
    paths, diagnostics = sample_posterior_mcmc_batch(
        make_truth_drift(cfg.data.truth) if drift is None else drift,
        make_truth_diffusion(cfg.data.truth) if diffusion is None else diffusion,
        events, cfg, int(n_prime), used, device, generator, positivity=mcmc_positivity)
    rows = [item.to_dict(include_traces=True) for item in diagnostics]
    metadata = {"kind": "mcmc", "label": "MCMC posterior", "exact": False,
                "target": "Euler prior and discrete observation likelihood",
                "mcmc_diagnostics": rows,
                "acceptance": float(np.mean([item.acceptance_rate for item in diagnostics])) if diagnostics else float("nan"),
                "beta_final": float(np.mean([item.beta_final for item in diagnostics])) if diagnostics else float("nan")}
    return torch.as_tensor(paths).to(device).reshape(-1, cfg.data.n_steps + 1), metadata


def learned_prior_reference_samples(
    model: NDIM, events: Tensor, cfg: ExperimentConfig, n_prime: int, n_paths: int,
    device: torch.device, generator: Optional[torch.Generator] = None, mcmc_cfg: Optional[MCMCConfig] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """MCMC posterior samples under the *learned* prior of ``model``: its drift, diffusion and link, started where
    the model starts (at ``t = -burn_in`` from ``x_start`` with a burn-in). This is the reference for observations of
    an unspecified model, where no truth exists: it checks the amortised posterior against a sampler for the same
    prior. Returns flat intensity paths ``[n_obs * n_paths, N + 1]`` on the observation window and metadata."""
    used = dataclasses.replace(mcmc_cfg or cfg.eval.mcmc, n_chains=int(n_paths))
    burn = model.n_burn > 0
    paths, diagnostics = sample_posterior_mcmc_batch(
        model.drift, model.diffusion, events, cfg, int(n_prime), used, device, generator,
        z0=0.0 if burn else model_z0(cfg), positivity=cfg.model.positivity, link=model.link if model.real_states else None,
        n_lead=model.n_burn, t0=model.t_start, initial_raw=model.initial_state if burn else None)
    metadata = {"kind": "mcmc_learned_prior", "label": "MCMC posterior (learned prior)", "exact": False,
                "target": "learned Euler prior and discrete observation likelihood",
                "mcmc_diagnostics": [item.to_dict(include_traces=True) for item in diagnostics],
                "acceptance": float(np.mean([item.acceptance_rate for item in diagnostics])) if diagnostics else float("nan"),
                "beta_final": float(np.mean([item.beta_final for item in diagnostics])) if diagnostics else float("nan")}
    return torch.as_tensor(paths).to(device).reshape(-1, cfg.data.n_steps + 1), metadata


def _prior_paths(
    drift: DriftFn,
    diffusion: DriftFn,
    cfg: ExperimentConfig,
    brownian: Tensor,
    n_steps: Optional[int] = None,
    positivity: Optional[str] = None,
    state_space: Optional[str] = None,
    z0: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    """Euler–Maruyama paths of ``dZ = drift dt + diffusion dB`` from ``z0`` on the config grid (``[R, n_steps+1]``).

    ``positivity`` defaults to ``cfg.model.positivity`` (the scheme the model was trained with); the MCMC
    reference functions pass ``"full_truncation"``, the scheme of the data-generating truth. ``z0`` defaults to
    the initial intensity of the model; paths of the truth pass ``cfg.data.truth.z0``.
    """
    n_steps = int(cfg.data.n_steps if n_steps is None else n_steps)
    pos = str(cfg.model.positivity if positivity is None else positivity)
    real = (cfg.model.state_space if state_space is None else state_space) == "real"
    return simulate_prior(
        drift, diffusion, model_z0(cfg) if z0 is None else z0 if isinstance(z0, torch.Tensor) else float(z0), n_steps, cfg.dt, brownian,
        positivity=pos, softplus_scale=cfg.model.softplus_scale,
        link=build_link(cfg.model.link) if real else None,
    )


def _posterior_paths(
    model: NDIM,
    events: Tensor,
    cfg: ExperimentConfig,
    n_prime: int,
    n_paths: int,
    brownian: Tensor,
    n_steps: Optional[int] = None,
) -> Tensor:
    """VI posterior paths ``[n_obs*n_paths, n_steps+1]``: score on for ``k < n_prime``, prior afterwards (Eq. 3).

    ``n_prime == 0`` means "no conditioning": the posterior is the prior, so the (learned) prior is
    simulated directly and the score network is never called on an empty grid.
    """
    n_steps = int(cfg.data.n_steps if n_steps is None else n_steps)
    if n_prime <= 0:
        return model.simulate_prior(brownian, n_steps)
    sim = model.simulate_posterior(events, int(n_prime), brownian, n_steps,
                                   max_abs_score=cfg.train.max_abs_score, max_abs_control=cfg.train.max_abs_control)
    return sim.Z[:, sim.n_burn:]                     # the observation window; the burn-in comes before it


# --------------------------------------------------------------------------------------
# Expected path deviation between learned and true prior under common noise (paper Table 1)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def expected_path_deviation(
    learned_drift: DriftFn,
    true_drift: DriftFn,
    diffusion_fn: DriftFn,
    cfg: ExperimentConfig,
    n_paths: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    true_diffusion_fn: Optional[DriftFn] = None,
    return_paths: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Tensor]]]:
    """Expected path deviation between the learned and the true prior under *common* Brownian noise (paper Table 1).

    Both priors are integrated from ``z0`` on the config grid with the same
    increments ``dB`` and returned in intensity coordinates. The truth uses
    intensity Euler with full truncation; learned paths use their configured
    state coordinates and positivity scheme. The grid integrals

    * ``l2   = E int_0^T (Z^learned_t - Z^true_t)^2 dt``   (the quantity reported in Table 1),
    * ``l1   = E int_0^T |Z^learned_t - Z^true_t| dt``,
    * ``rmse = sqrt(l2 / T)``                               (root-mean-square deviation per unit time)

    are estimated with ``n_paths`` Monte-Carlo paths (``dt * sum`` over the ``N+1`` grid points; the ``k=0``
    term vanishes since both paths start at ``z0``).

    ``diffusion_fn`` is used for the learned prior; the true prior uses ``true_diffusion_fn`` when given
    (experiment 3, trainable ``eta``) and ``diffusion_fn`` otherwise. Extra keys: ``l2_stderr``, ``l1_stderr``
    (Monte-Carlo standard errors) and ``n_paths``.

    ``return_paths=True`` additionally returns ``(metrics, {"learned": paths,
    "true": paths})`` for plotting the exact same common-noise simulation.
    The default remains a scalar metrics dictionary. Linked learned models
    use latent Euler steps; the truth uses intensity Euler, so path deviation
    also includes this discretization difference.
    """
    n_paths = int(n_paths)
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if true_diffusion_fn is None:
        true_diffusion_fn = make_truth_diffusion(cfg.data.truth) if cfg.model.state_space == "real" else diffusion_fn
    N, dt, T = int(cfg.data.n_steps), cfg.dt, float(cfg.data.T)
    W = brownian_increments(n_paths, N, dt, device, generator)
    if cfg.burn_steps > 0 or float(cfg.data.truth.z0_std) > 0:
        # both priors start at the same draw of the truth's Z_0: the deviation measures the dynamics on the window
        z0_learned = z0_true = sample_truth_z0(cfg.data.truth, n_paths, _cpu_generator(generator)).to(device)
    else:
        z0_learned, z0_true = None, float(cfg.data.truth.z0)
    Z_learned = _prior_paths(learned_drift, diffusion_fn, cfg, W, z0=z0_learned)
    Z_true = _prior_paths(true_drift, true_diffusion_fn, cfg, W, state_space="intensity", positivity=DEFAULT_POSITIVITY, z0=z0_true)
    diff = (Z_learned - Z_true).to(torch.float64)
    l2_paths = (diff ** 2).sum(dim=1) * dt                            # [n_paths]
    l1_paths = diff.abs().sum(dim=1) * dt
    l2 = float(l2_paths.mean())
    l1 = float(l1_paths.mean())
    sq = math.sqrt(n_paths)
    result = {
        "l2": l2,
        "l1": l1,
        "rmse": math.sqrt(max(l2, 0.0) / T),
        "l2_stderr": float(l2_paths.std(unbiased=True) / sq) if n_paths > 1 else float("nan"),
        "l1_stderr": float(l1_paths.std(unbiased=True) / sq) if n_paths > 1 else float("nan"),
        "n_paths": float(n_paths),
    }
    if return_paths:
        return result, {"learned": Z_learned, "true": Z_true}
    return result


# --------------------------------------------------------------------------------------
# Fig. 5: exact Wasserstein distance between empirical path measures
# --------------------------------------------------------------------------------------
def wasserstein2_paths(paths_a: ArrayLike, paths_b: ArrayLike, dt: float, p: int = 2) -> float:
    """Exact ``p``-Wasserstein distance between two *uniform* empirical measures on discretised paths (paper Fig. 5).

    ``paths_a, paths_b`` are ``[n, G]`` (same ``n``). Ground cost between path ``i`` of ``a`` and ``j`` of ``b``:

    * ``p = 2``: ``c_ij = dt * sum_k (a_ik - b_jk)^2`` (squared ``L2([0,T])`` metric); returns ``sqrt(mean matched cost)``,
    * ``p = 1``: ``c_ij = dt * sum_k |a_ik - b_jk|``   (``L1([0,T])`` metric);   returns ``mean matched cost``.

    With equal cardinalities the optimal coupling is a permutation (Birkhoff), found exactly with
    ``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm, ``O(n^3)``).
    """
    a = _to_numpy(paths_a)
    b = _to_numpy(paths_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"paths must be 2-D [n, G], got shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"paths_a and paths_b must have the same shape (same n and grid), got {a.shape} vs {b.shape}")
    if a.shape[0] == 0:
        return 0.0
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paths contain non-finite values")
    if p == 2:
        cost = cdist(a, b, metric="sqeuclidean")
    elif p == 1:
        cost = cdist(a, b, metric="cityblock")
    else:
        raise ValueError(f"p must be 1 or 2, got {p}")
    cost = np.maximum(cost, 0.0) * float(dt)
    rows, cols = linear_sum_assignment(cost)
    mean_cost = float(cost[rows, cols].mean())
    return math.sqrt(max(mean_cost, 0.0)) if p == 2 else mean_cost


@torch.no_grad()
def posterior_wasserstein(
    model: NDIM,
    drift_true: DriftFn,
    diffusion_true: DriftFn,
    events: Tensor,
    cfg: ExperimentConfig,
    n_prime: int,
    n_paths: int,
    mcmc_cfg: MCMCConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    p: int = 2,
    mcmc_positivity: str = DEFAULT_POSITIVITY,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-observation W2 against the configured true posterior reference.

    For every observation ``b`` in ``events [n_obs, K]``:

    * VI samples: ``n_paths`` paths of the drift-corrected SDE (Eq. 3) conditioned on ``(0, n_prime dt]``,
      simulated to ``T`` (prior dynamics after ``T'``), via :func:`ndim.sde.simulate_posterior`;
    * Reference samples: ``cfg.eval.posterior_reference`` unless ``reference`` is given (the exact CIR
      posterior or path-space MCMC under the true prior).
    * prior baseline: ``n_paths`` learned-prior paths (same Brownian noise as the VI samples) vs the reference samples,
      i.e. the distance one gets without any posterior correction.

    Returns ``{"w2_vi": [n_obs], "w2_prior_baseline": [n_obs], "mean_w2_vi", "mean_w2_prior_baseline"}`` plus
    ``std_w2_vi``, ``std_w2_prior_baseline``, ``reference`` metadata,
    ``seconds_vi``, ``seconds_reference``, ``n_obs``, ``n_paths``, ``n_prime``, ``p``.
    MCMC-specific fields appear only for the MCMC reference.
    """
    model.eval()
    events = events.to(device)
    n_obs = int(events.shape[0])
    n_paths = int(n_paths)
    N, dt = int(cfg.data.n_steps), cfg.dt
    if n_obs == 0:
        return {"w2_vi": [], "w2_prior_baseline": [], "mean_w2_vi": float("nan"), "mean_w2_prior_baseline": float("nan")}

    with timer() as t_vi:
        W = brownian_increments(n_obs * n_paths, model.n_burn + N, dt, device, generator)
        Z_vi = _posterior_paths(model, events, cfg, n_prime, n_paths, W).view(n_obs, n_paths, N + 1).cpu()
        Z_prior = model.simulate_prior(W).view(n_obs, n_paths, N + 1).cpu()

    with timer() as t_reference:
        Z_reference, reference_metadata = posterior_reference_samples(
            events, cfg, int(n_prime), n_paths, device, generator,
            reference=reference, drift=drift_true, diffusion=diffusion_true,
            mcmc_cfg=mcmc_cfg, mcmc_positivity=mcmc_positivity)
    Z_reference = Z_reference.detach().cpu().view(n_obs, n_paths, N + 1)

    w2_vi = [wasserstein2_paths(Z_vi[b], Z_reference[b], dt, p=p) for b in range(n_obs)]
    w2_prior = [wasserstein2_paths(Z_prior[b], Z_reference[b], dt, p=p) for b in range(n_obs)]
    result = {
        "w2_vi": w2_vi,
        "w2_prior_baseline": w2_prior,
        "mean_w2_vi": float(np.mean(w2_vi)),
        "mean_w2_prior_baseline": float(np.mean(w2_prior)),
        "std_w2_vi": float(np.std(w2_vi)),
        "std_w2_prior_baseline": float(np.std(w2_prior)),
        "seconds_vi": float(t_vi["seconds"]),
        "seconds_reference": float(t_reference["seconds"]),
        "reference": reference_metadata,
        "n_obs": n_obs,
        "n_paths": n_paths,
        "n_prime": int(n_prime),
        "p": int(p),
    }
    if reference_metadata["kind"] == "mcmc":
        result.update(mcmc_acceptance=reference_metadata["acceptance"],
                      seconds_mcmc=float(t_reference["seconds"]))
    return result


# --------------------------------------------------------------------------------------
# Fig. 3: cumulative counts and samples from the learned generative model
# --------------------------------------------------------------------------------------
def cumulative_count_curves(events_list: Sequence[ArrayLike], t_grid: ArrayLike) -> np.ndarray:
    """Cumulative counting processes ``N_b(t) = #{i : tau_{b,i} <= t}`` on ``t_grid`` -> ``[n_obs, len(t_grid)]`` (float).

    Each entry of ``events_list`` is a 1-D array of event times (unsorted and ``+inf``-padded inputs are
    accepted; non-finite entries are ignored). Used for the mean +/- std bands of paper Fig. 3.
    """
    tg = _to_numpy(t_grid).reshape(-1)
    out = np.zeros((len(events_list), tg.shape[0]), dtype=np.float64)
    for b, ev in enumerate(events_list):
        e = _to_numpy(ev).reshape(-1)
        e = np.sort(e[np.isfinite(e)])
        out[b] = np.searchsorted(e, tg, side="right")
    return out


@torch.no_grad()
def generate_observations_from_model(
    model: NDIM,
    cfg: ExperimentConfig,
    n_obs: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[np.ndarray], Tensor]:
    """Sample ``n_obs`` synthetic observations from the *learned* generative model (paper Fig. 3, App. G Alg. 2).

    Intensity paths follow the learned prior ``dZ = b_theta dt + sigma_eta dB`` from ``z0`` on the config grid
    (:func:`ndim.sde.simulate_prior`); events are drawn by exact thinning of the piecewise-constant intensity
    (:func:`ndim.data.thin_events`). Returns ``(events_list, paths)`` with ``events_list`` a list of sorted
    ``np.ndarray`` event times and ``paths`` the ``[n_obs, N+1]`` intensity paths (on CPU, like the dataset paths).
    """
    model.eval()
    n_obs = int(n_obs)
    N, dt, T = int(cfg.data.n_steps), cfg.dt, float(cfg.data.T)
    W = brownian_increments(n_obs, model.n_burn + N, dt, device, generator)
    paths = model.simulate_prior(W).detach().cpu()           # the observation window of the learned prior
    events_list = thin_events(paths, T, generator=_cpu_generator(generator))
    return events_list, paths


# alias
expected_l2_deviation = expected_path_deviation

# --------------------------------------------------------------------------------------
# Stiffness of the learned drift on the Euler grid
# --------------------------------------------------------------------------------------
#: a share of steps with a gain below -1 above this is reported as a drift that is too stiff for the grid
STIFF_SHARE = 0.005


def euler_gain(model: NDIM, paths: Tensor, t_grid: Tensor, dt: float, max_paths: int = 500) -> Tensor:
    """Gain of one Euler step of the learned drift, ``1 + dt * db/dx``, at every step of the first ``max_paths``
    intensity paths ``paths [n, G + 1]`` on ``t_grid``: ``[n, G]``.

    One step turns a small deviation ``d`` of the state into ``(1 + dt * db/dx) d``. A gain between 0 and 1 shrinks it
    smoothly; between -1 and 0 it shrinks it with a change of sign, a zig-zag; below -1 the deviation grows from step
    to step: the drift is too stiff for the step size there and paths can run off. A drift that pulls hard where the
    intensity is low does this on a coarse grid."""
    z = paths[: int(max_paths), :-1].to(torch.float32)
    state = model.link.inverse(z.clamp_min(1e-30)) if model.real_states else z
    state = state.detach().clone().requires_grad_(True)
    t = torch.as_tensor(t_grid, dtype=torch.float32, device=state.device)[:-1].expand_as(state)
    with torch.enable_grad():
        (slope,) = torch.autograd.grad(model.drift(state.reshape(-1), t.reshape(-1)).sum(), state)
    return (1.0 + float(dt) * slope).detach()


def euler_gain_summary(model: NDIM, paths: Tensor, t_grid: Tensor, dt: float, n_burn: int = 0) -> Dict[str, Any]:
    """``euler_gain`` in numbers: the smallest gain and the share of steps below -1 (and below 0), over all steps and
    apart for the burn-in and the observation window.

    When every path starts from the same state (a fixed initial intensity, or the start of a burn-in), the first step
    is left out: no deviation exists there that the step could enlarge."""
    gain = euler_gain(model, paths, t_grid, dt)
    gain = gain[torch.isfinite(gain).all(dim=1)]
    start = paths[:, 0]
    first = int(gain.shape[1] > 1 and bool((start == start[0]).all()))
    nb = max(int(n_burn), first)
    steps = gain[:, first:]

    def share(g: Tensor, level: float) -> Optional[float]:
        return float((g < level).double().mean()) if g.numel() else None

    return {"min": float(steps.min()) if steps.numel() else None, "share_below_minus_one": share(steps, -1.0), "share_negative": share(steps, 0.0),
            "share_below_minus_one_burn_in": share(gain[:, first:int(n_burn)], -1.0), "share_below_minus_one_window": share(gain[:, nb:], -1.0),
            "first_step_left_out": bool(first), "n_paths": int(gain.shape[0])}


# --------------------------------------------------------------------------------------
# Likelihood of observations under a learned prior, and two reference models (observations without a truth)
# --------------------------------------------------------------------------------------
def events_per_step(events: Tensor, dt: float, n_steps: int) -> Tensor:
    """Number of events of every observation in every Euler step, ``[B, n_steps]`` (float64), with the convention of
    the likelihood: an event at ``tau`` belongs to step ``floor(tau / dt)``, and one at ``T`` to the last step."""
    ev = events.detach().cpu()
    idx, valid = event_grid_index(ev, float(dt), int(n_steps), float(n_steps) * float(dt))       # the arithmetic of the likelihood itself
    out = torch.zeros(ev.shape[0], int(n_steps), dtype=torch.float64)
    out.scatter_add_(1, idx, valid.to(torch.float64))
    return out


@torch.no_grad()
def marginal_loglik_particle_filter(model: NDIM, events: Tensor, n_particles: int = 2000,
                                    generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
    """Marginal log-likelihood ``log p(X)`` of every observation under the learned prior, by a bootstrap particle
    filter.

    On the Euler grid the model is a state-space model: the state moves by the prior transition (from the start of
    the burn-in when there is one), and the events of step ``k`` contribute ``n_k log Z_k - Z_k dt``. The filter
    propagates ``n_particles`` particles with the prior, weights them by that term and resamples at every step; the
    product of the mean weights is an unbiased estimate of ``p(X)``. Unlike the ELBO it does not depend on the
    posterior network, so it compares priors (two diffusions, say) directly. Importance sampling of whole paths
    estimates the same number but degenerates when the paths are rough.

    Returns ``(log p(X) [B] float64, smallest effective sample size over the steps [B])``.
    """
    cfg = model.cfg
    N, nb, dt = int(cfg.data.n_steps), int(model.n_burn), float(cfg.dt)
    counts = events_per_step(events, dt, N)
    B, P = counts.shape[0], int(n_particles)
    device = next(model.parameters()).device
    g = _cpu_generator(generator)
    raw = torch.full((B * P,), float(model.initial_state), device=device)

    def step(state: Tensor, k: int) -> Tensor:
        t = torch.full_like(state, model.t_start + k * dt)
        coefficient = model.coefficient_state(state)
        noise = torch.randn(state.shape, generator=g).to(device) * math.sqrt(dt)
        return state + model.drift(coefficient, t) * dt + model.diffusion(coefficient, t) * noise

    for k in range(nb):                                  # the burn-in: nothing is observed, so no weights
        raw = step(raw, k)
    logp = torch.zeros(B, dtype=torch.float64)
    ess_min = torch.full((B,), float(P), dtype=torch.float64)
    eps = float(cfg.likelihood.eps)
    for k in range(N):
        Z = model.to_intensity(raw).detach().cpu().to(torch.float64).view(B, P)
        logw = counts[:, k:k + 1] * torch.log(Z.clamp_min(eps)) - Z * dt
        logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -float("inf")))      # a particle that ran off has no weight
        total = torch.logsumexp(logw, dim=1)
        if not torch.isfinite(total).all():
            raise FloatingPointError(f"no particle of the learned prior has a finite weight at step {k}: its paths are not finite")
        logp += total - math.log(P)
        w = torch.softmax(logw, dim=1)
        ess_min = torch.minimum(ess_min, 1.0 / (w ** 2).sum(dim=1))
        if k + 1 < N:
            idx = torch.multinomial(w.to(torch.float32), P, replacement=True, generator=g).to(device)
            raw = step(raw.view(B, P).gather(1, idx).reshape(-1), nb + k)
    return logp, ess_min


def reference_logliks(train_events: Tensor, test_events: Tensor, dt: float, n_steps: int) -> Dict[str, Any]:
    """Log-likelihood of every test observation under two models without a latent path, fitted to the training
    observations on the same grid and with the same likelihood convention as the model:

    * ``poisson``: a Poisson process whose rate is constant on every Euler step (the mean count of the step in the
      training observations, with half an event added so that no rate is zero);
    * ``poisson_gamma``: the same rate multiplied by one factor per observation, gamma distributed with mean 1 (a
      "busy day" effect, constant over the window). Its shape is fitted by maximum likelihood on a grid; the
      coefficient of variation of the factor is ``1 / sqrt(shape)``.

    A model with a latent intensity should beat the first when the observations are overdispersed, and the second
    when their intensity varies within the window in a way that one factor per observation does not describe.
    """
    c_train, c_test = events_per_step(train_events, dt, n_steps).numpy(), events_per_step(test_events, dt, n_steps).numpy()
    rate = (c_train.sum(0) + 0.5) / (c_train.shape[0] * float(dt))
    total = float((rate * float(dt)).sum())
    log_rate = np.log(rate)

    def poisson(c: np.ndarray) -> np.ndarray:
        return (c * log_rate[None, :]).sum(1) - total

    lgamma = np.vectorize(math.lgamma)

    def with_factor(c: np.ndarray, shape: float) -> np.ndarray:
        n = c.sum(1)
        return (c * log_rate[None, :]).sum(1) + lgamma(shape + n) - math.lgamma(shape) + shape * math.log(shape) - (shape + n) * np.log(shape + total)

    grid = np.exp(np.linspace(0.0, math.log(1.0e4), 600))
    shape = float(grid[int(np.argmax([with_factor(c_train, a).sum() for a in grid]))])
    return {"poisson": poisson(c_test), "poisson_gamma": with_factor(c_test, shape), "gamma_shape": shape,
            "gamma_factor_cv": 1.0 / math.sqrt(shape), "rate": rate}


__all__ = [
    "expected_path_deviation",
    "expected_l2_deviation",
    "wasserstein2_paths",
    "posterior_wasserstein",
    "posterior_reference_samples",
    "resolve_posterior_reference",
    "cumulative_count_curves",
    "generate_observations_from_model",
    "events_per_step",
    "marginal_loglik_particle_filter",
    "reference_logliks",
]
