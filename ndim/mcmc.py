"""Path-space MCMC baseline for the exact posterior of latent intensity paths (paper Sec. 4.2, footnote 4;
docs/DESIGN.md Sec. 5).

Target
------
For one observation ``X`` on ``(0, T']`` and a *given* prior ``dZ = b(Z,t) dt + sigma(Z,t) dB``, ``Z_0 = z0``
(full-truncation Euler on the grid ``t_k = k dt``), the posterior over the standard-normal Brownian
increments ``xi in R^N`` is

    pi(xi | X)  ∝  N(xi; 0, I) * exp( log p(X | Z(xi)) ),

where ``Z(xi)`` is the Euler map of ``sqrt(dt) * xi`` (``ndim.sde.simulate_prior``) and ``log p(X | Z)`` is the
Poisson log-likelihood on ``(0, T']`` (``ndim.likelihood.poisson_loglik``, paper Eq. 10). Paths are simulated
to the full horizon ``T = N dt``; only events in ``(0, T']`` enter the likelihood, so after ``T'`` the chains
follow the prior dynamics conditioned on their state at ``T'`` -- exactly the quantity drawn in Fig. 4b.

Samplers (``MCMCConfig.kind``)
------------------------------
* ``independence`` (default) -- the paper's sampler (Fig. 4, Tables 2-3): Metropolis–Hastings with a fresh prior
  path proposed at every step, ``xi' = zeta``, ``zeta ~ N(0, I)``, accepted with probability
  ``min(1, exp(ll(xi') - ll(xi)))``. The proposal is the prior itself, so the prior cancels in the acceptance
  ratio and only the likelihood ratio remains. The acceptance rate depends on the observation: about one half
  for an observation that is typical under the prior, a few percent or less for the test observations with the
  fewest and the most events, so the number of steps has to be large (the paper uses 10,000 burn-in steps).
* ``pcn`` -- preconditioned Crank–Nicolson (Cotter, Roberts, Stuart, White 2013): proposal
  ``xi' = sqrt(1 - beta^2) xi + beta zeta`` with the same acceptance probability (the proposal is reversible
  w.r.t. the Gaussian prior, so the prior cancels *exactly* here too). ``beta = 1`` is the independence
  sampler; ``beta`` is adapted during the burn-in towards ``target_accept`` with a Robbins–Monro rule on
  ``log beta`` (one ``beta`` per observation).

All chains (and, in the batch variant, all observations) are updated in one vectorised Euler simulation per
iteration; every random number comes from the ``torch.Generator`` passed by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import ExperimentConfig, MCMCConfig
from .likelihood import poisson_loglik
from .sde import simulate_prior
from .links import IntensityLink
from .utils import get_logger

Tensor = torch.Tensor
CoefficientFn = Callable[[Tensor, Tensor], Tensor]

#: clipping interval for ``beta`` in the adaptive pCN sampler
BETA_MIN: float = 1e-3
BETA_MAX: float = 1.0
#: ``ll' - ll`` is clipped to ``[-LOG_RATIO_CLIP, LOG_RATIO_CLIP]`` before exponentiation
LOG_RATIO_CLIP: float = 60.0
#: smallest positive float32 uniform: an exact u == 0 must never accept a proposal
UNIFORM_MIN: float = 2.0 ** -24
#: positivity scheme when the caller passes none: full truncation, as in the truth
DEFAULT_POSITIVITY: str = "full_truncation"


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------
@dataclass
class MCMCDiagnostics:
    """Summary of one MCMC run (one observation, ``n_chains`` independent chains).

    ``acceptance_rate`` is the mean over all iterations and chains; ``loglik_trace[i]`` is the mean current
    log-likelihood over chains after iteration ``i`` (a flat trace after burn-in indicates the chains have
    reached the posterior's typical set). The remaining fields are extra bookkeeping with defaults.
    """
    acceptance_rate: float
    beta_final: float
    n_steps: int
    n_chains: int
    loglik_trace: np.ndarray                              # [n_steps] mean over chains
    kind: str = "pcn"
    n_burn_in: int = 0
    acceptance_rate_post_burnin: float = float("nan")
    beta_trace: Optional[np.ndarray] = None               # [n_steps] beta used at each iteration
    loglik_final_mean: float = float("nan")               # mean over chains of the final log-likelihood

    def to_dict(self, include_traces: bool = True) -> Dict[str, Any]:
        """JSON-friendly dictionary (traces as lists, or dropped with ``include_traces=False``)."""
        out: Dict[str, Any] = {
            "kind": self.kind,
            "acceptance_rate": float(self.acceptance_rate),
            "acceptance_rate_post_burnin": float(self.acceptance_rate_post_burnin),
            "beta_final": float(self.beta_final),
            "n_steps": int(self.n_steps),
            "n_burn_in": int(self.n_burn_in),
            "n_chains": int(self.n_chains),
            "loglik_final_mean": float(self.loglik_final_mean),
        }
        if include_traces:
            out["loglik_trace"] = np.asarray(self.loglik_trace, dtype=float).tolist()
            out["beta_trace"] = None if self.beta_trace is None else np.asarray(self.beta_trace, dtype=float).tolist()
        return out


def summarize_diagnostics(diags: Sequence[MCMCDiagnostics]) -> Dict[str, float]:
    """Means over a list of per-observation diagnostics (for tables / ``metrics.json``)."""
    if len(diags) == 0:
        return {"acceptance_rate": float("nan"), "acceptance_rate_post_burnin": float("nan"), "beta_final": float("nan")}
    return {
        "acceptance_rate": float(np.mean([d.acceptance_rate for d in diags])),
        "acceptance_rate_post_burnin": float(np.nanmean([d.acceptance_rate_post_burnin for d in diags])),
        "beta_final": float(np.mean([d.beta_final for d in diags])),
        "loglik_final_mean": float(np.mean([d.loglik_final_mean for d in diags])),
    }


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _randn(shape: Sequence[int], device: torch.device, generator: Optional[torch.Generator]) -> Tensor:
    """``N(0, I)`` sample of ``shape`` on ``device`` drawn from ``generator`` (which may live on another device)."""
    if generator is not None and generator.device != device:
        return torch.randn(*shape, generator=generator).to(device)
    return torch.randn(*shape, device=device, generator=generator)


def _rand(shape: Sequence[int], device: torch.device, generator: Optional[torch.Generator]) -> Tensor:
    """``U(0, 1)`` sample of ``shape`` on ``device`` drawn from ``generator``."""
    if generator is not None and generator.device != device:
        return torch.rand(*shape, generator=generator).to(device)
    return torch.rand(*shape, device=device, generator=generator)


def _as_event_matrix(events: Union[Tensor, np.ndarray, Sequence[float]], device: torch.device) -> Tensor:
    """Coerce event times to a float32 ``[n_obs, K]`` matrix padded with ``+inf`` (``K >= 1``)."""
    ev = torch.as_tensor(events, dtype=torch.float32)
    if ev.dim() == 0:
        ev = ev.reshape(1, 1)
    elif ev.dim() == 1:
        ev = ev.unsqueeze(0)
    elif ev.dim() != 2:
        raise ValueError(f"events must be 1-D (one observation) or 2-D [n_obs, K]; got shape {tuple(ev.shape)}")
    if ev.shape[1] == 0:  # no events at all: keep one +inf column so gathers are well defined
        ev = torch.full((ev.shape[0], 1), float("inf"), dtype=torch.float32)
    return ev.to(device)


def metropolis_accept(ll_cur: Tensor, ll_new: Tensor, u: Tensor) -> Tensor:
    """Vectorised Metropolis–Hastings decision for a prior-reversible proposal.

    ``accept = (delta >= 0) or (u < exp(clip(delta, -60, 60)))`` with ``delta = ll_new - ll_cur`` and ``u ~ U(0, 1)``
    (``[R]`` tensors), plus three safety rules:

    * a proposal with a non-finite log-likelihood (``-inf`` / ``NaN`` / ``+inf``) is **never** accepted;
    * a chain whose *current* log-likelihood is non-finite (e.g. a degenerate initial draw) accepts any
      finite proposal, so it cannot get stuck;
    * ``u`` is lifted to ``[UNIFORM_MIN, 1)``: ``torch.rand`` returns an exact ``0`` with probability ``2^-24``,
      which would otherwise accept a proposal whose likelihood ratio is astronomically small. Proposals with
      ``delta < log(UNIFORM_MIN) ~ -16.6`` are therefore rejected outright -- the resolution of float32 uniforms.
    """
    new_ok = torch.isfinite(ll_new)
    cur_ok = torch.isfinite(ll_cur)
    delta = torch.where(cur_ok, ll_new - ll_cur, torch.full_like(ll_new, LOG_RATIO_CLIP))
    delta = torch.nan_to_num(delta, nan=-LOG_RATIO_CLIP, posinf=LOG_RATIO_CLIP, neginf=-LOG_RATIO_CLIP)
    delta = torch.clamp(delta, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    u_pos = torch.clamp(u.to(delta.dtype), min=UNIFORM_MIN)
    return new_ok & ((delta >= 0) | (u_pos < torch.exp(delta)))


def _prepare_run(
    cfg: ExperimentConfig,
    n_prime: int,
    mcmc: MCMCConfig,
    n_steps_total: Optional[int],
    z0: Optional[float],
    positivity: Optional[str],
) -> Tuple[int, int, float, str]:
    """Validate arguments and resolve defaults -> ``(N, n_prime, z0, positivity)``."""
    N = int(cfg.data.n_steps if n_steps_total is None else n_steps_total)
    n_prime = int(n_prime)
    if N <= 0:
        raise ValueError("n_steps_total must be positive")
    if not (1 <= n_prime <= N):
        raise ValueError(f"n_prime={n_prime} must satisfy 1 <= n_prime <= n_steps_total={N}")
    if mcmc.kind not in {"pcn", "independence"}:
        raise ValueError(f"mcmc.kind must be 'pcn' or 'independence', got {mcmc.kind!r}")
    if int(mcmc.n_chains) <= 0:
        raise ValueError("mcmc.n_chains must be positive")
    if int(mcmc.n_steps) < 0:
        raise ValueError("mcmc.n_steps must be non-negative")
    if not (0.0 < float(mcmc.target_accept) < 1.0):
        raise ValueError("mcmc.target_accept must lie in (0, 1)")
    z0_val = float(cfg.data.truth.z0 if z0 is None else z0)
    pos = str(DEFAULT_POSITIVITY if positivity is None else positivity)
    if pos not in {"full_truncation", "reflect", "softplus"}:
        raise ValueError(f"positivity must be full_truncation|reflect|softplus, got {pos!r}")
    return N, n_prime, z0_val, pos


# --------------------------------------------------------------------------------------
# Core sampler (vectorised over observations x chains)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _run_chains(
    drift: CoefficientFn,
    diffusion: CoefficientFn,
    events: Tensor,
    cfg: ExperimentConfig,
    n_prime: int,
    mcmc: MCMCConfig,
    device: torch.device,
    generator: Optional[torch.Generator],
    N: int,
    z0: float,
    positivity: str,
    verbose: bool,
    log_prefix: str = "",
    link: Optional[IntensityLink] = None,
    return_raw: bool = False,
    z0_from_normal: Optional[Callable[[Tensor], Tensor]] = None,
    n_lead: int = 0,
    t0: float = 0.0,
    initial_raw: Optional[float] = None,
) -> Tuple[Tensor, List[MCMCDiagnostics]]:
    """Run ``mcmc.n_steps`` iterations for ``events [n_obs, K]`` -> ``(paths [n_obs, n_chains, N+1], diags)``.

    ``n_lead`` Euler steps before the observation window (a prior with a burn-in: it starts at ``t0 = -n_lead dt``
    from the latent state ``initial_raw``) are simulated and sampled with the rest of the noise; the likelihood and
    the returned paths cover the window only.

    Row layout is ``o * n_chains + c`` (observation ``o``, chain ``c``), matching ``poisson_loglik(.., m=n_chains)``.
    """
    n_obs = int(events.shape[0])
    n_chains = int(mcmc.n_chains)
    n_steps = int(mcmc.n_steps)
    R = n_obs * n_chains
    dt = float(cfg.dt)
    sqrt_dt = math.sqrt(dt)
    kind = str(mcmc.kind)
    adapt = bool(mcmc.adapt) and kind == "pcn"
    # adaptation window of pCN; for the independence sampler it only splits the acceptance diagnostics
    n_burn = max(0, min(int(round(float(mcmc.burn_in_frac) * n_steps)), n_steps))
    target = float(mcmc.target_accept)
    softplus_scale = float(cfg.model.softplus_scale)
    logger = get_logger("ndim.mcmc") if verbose else None

    # a random initial intensity is a function of one more standard normal per chain, moved with the Brownian noise
    random_z0 = z0_from_normal is not None

    n_lead = int(n_lead)

    def euler(xi: Tensor, zeta0: Optional[Tensor] = None) -> Tensor:
        start = z0_from_normal(zeta0) if random_z0 else z0
        paths = simulate_prior(drift, diffusion, start, N + n_lead, dt, xi * sqrt_dt, positivity, softplus_scale,
                               link=link, return_raw=link is not None, t0=t0, initial_raw=initial_raw)
        return paths[:, n_lead:]                        # the observation window

    def loglik(Z: Tensor) -> Tensor:
        intensity = link(Z) if link is not None else Z
        return poisson_loglik(intensity, events, n_chains, dt, n_prime, cfg.likelihood)

    # --- initial state: one prior draw per chain -----------------------------------------------
    xi = _randn((R, N + n_lead), device, generator)
    zeta0 = _randn((R,), device, generator) if random_z0 else None
    Z = euler(xi, zeta0)
    ll = loglik(Z)

    # --- step size (one beta per observation, adapted in log-space, float64 on CPU) -------------
    if kind == "independence":
        log_beta = torch.zeros(n_obs, dtype=torch.float64)           # beta = 1: fresh prior proposal
    else:
        beta0 = min(max(float(mcmc.beta_init), BETA_MIN), BETA_MAX)
        log_beta = torch.full((n_obs,), math.log(beta0), dtype=torch.float64)

    acc_total = torch.zeros(n_obs, dtype=torch.float64)
    acc_post = torch.zeros(n_obs, dtype=torch.float64)
    ll_trace = torch.empty(n_steps, n_obs, dtype=torch.float64)
    beta_trace = torch.empty(n_steps, n_obs, dtype=torch.float64)
    log_every = max(1, n_steps // 10)
    recent_acc = torch.zeros(n_obs, dtype=torch.float64)
    recent_n = 0

    for it in range(n_steps):
        beta = torch.exp(log_beta)                                    # [n_obs] float64 (already clipped)
        beta_trace[it] = beta
        beta_rows = beta.to(device=device, dtype=xi.dtype).repeat_interleave(n_chains).unsqueeze(1)   # [R, 1]
        zeta = _randn((R, N + n_lead), device, generator)
        zeta0_prop = None
        if kind == "independence":
            xi_prop = zeta
            if random_z0:
                zeta0_prop = _randn((R,), device, generator)
        else:
            keep = torch.sqrt(torch.clamp(1.0 - beta_rows ** 2, min=0.0))
            xi_prop = keep * xi + beta_rows * zeta
            if random_z0:
                zeta0_prop = keep.squeeze(1) * zeta0 + beta_rows.squeeze(1) * _randn((R,), device, generator)
        Z_prop = euler(xi_prop, zeta0_prop)
        ll_prop = loglik(Z_prop)
        u = _rand((R,), device, generator)
        accept = metropolis_accept(ll, ll_prop, u)                    # [R] bool

        acc_rows = accept.unsqueeze(1)
        xi = torch.where(acc_rows, xi_prop, xi)
        if random_z0:
            zeta0 = torch.where(accept, zeta0_prop, zeta0)
        Z = torch.where(acc_rows, Z_prop, Z)
        ll = torch.where(accept, ll_prop, ll)

        acc_obs = accept.view(n_obs, n_chains).to(torch.float64).mean(dim=1).cpu()   # [n_obs]
        acc_total += acc_obs
        recent_acc += acc_obs
        recent_n += 1
        if it >= n_burn:
            acc_post += acc_obs
        if adapt and it < n_burn:
            # Robbins–Monro on log beta towards the target acceptance rate
            log_beta = log_beta + (acc_obs - target) / math.sqrt(it + 1.0)
            log_beta = torch.clamp(log_beta, math.log(BETA_MIN), math.log(BETA_MAX))
        ll_trace[it] = ll.view(n_obs, n_chains).to(torch.float64).mean(dim=1).cpu()

        if logger is not None and ((it + 1) % log_every == 0 or it + 1 == n_steps):
            logger.info(
                "%smcmc[%s] iter %d/%d | recent acc %.3f | beta %.4f | mean loglik %.2f%s",
                log_prefix, kind, it + 1, n_steps, float(recent_acc.mean()) / max(recent_n, 1),
                float(torch.exp(log_beta).mean()), float(ll_trace[it].mean()),
                " (burn-in)" if it < n_burn else "",
            )
            recent_acc.zero_()
            recent_n = 0

    beta_final = torch.exp(log_beta)
    n_post = n_steps - n_burn
    diags: List[MCMCDiagnostics] = []
    ll_final = ll.view(n_obs, n_chains).to(torch.float64).mean(dim=1).cpu()
    for o in range(n_obs):
        diags.append(
            MCMCDiagnostics(
                acceptance_rate=float(acc_total[o] / n_steps) if n_steps > 0 else float("nan"),
                beta_final=float(beta_final[o]),
                n_steps=n_steps,
                n_chains=n_chains,
                loglik_trace=ll_trace[:, o].numpy().copy(),
                kind=kind,
                n_burn_in=n_burn,
                acceptance_rate_post_burnin=float(acc_post[o] / n_post) if n_post > 0 else float("nan"),
                beta_trace=beta_trace[:, o].numpy().copy(),
                loglik_final_mean=float(ll_final[o]),
            )
        )
    paths = link(Z) if link is not None and not return_raw else Z
    return paths.view(n_obs, n_chains, N + 1), diags


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def _truth_z0_map(cfg: ExperimentConfig, z0: Optional[float]) -> Optional[Callable[[Tensor], Tensor]]:
    """Map from a standard normal to the initial intensity when the truth's is random and no fixed ``z0`` was passed."""
    if z0 is not None or float(cfg.data.truth.z0_std) <= 0:
        return None
    from .data import truth_z0_from_normal

    return lambda zeta: truth_z0_from_normal(cfg.data.truth, zeta)


def _default_generator(mcmc: MCMCConfig) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(mcmc.seed))


@torch.no_grad()
def sample_posterior_mcmc(
    drift: CoefficientFn,
    diffusion: CoefficientFn,
    events_row: Union[Tensor, np.ndarray, Sequence[float]],
    cfg: ExperimentConfig,
    n_prime: int,
    mcmc: MCMCConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    n_steps_total: Optional[int] = None,
    verbose: bool = False,
    z0: Optional[float] = None,
    positivity: Optional[str] = None,
    link: Optional[IntensityLink] = None,
    return_raw: bool = False,
) -> Tuple[Tensor, MCMCDiagnostics]:
    """Posterior path samples for **one** observation by pCN / independence MCMC (paper Sec. 4.2, Fig. 4).

    Args:
        drift, diffusion: prior coefficients ``(Z, t) -> tensor`` (e.g. ``make_truth_drift`` or ``model.drift``).
        events_row: 1-D event times of the observation (``+inf`` padding allowed; events after ``n_prime*dt`` are ignored).
        cfg: experiment config (``dt``, ``data.n_steps``, ``data.truth.z0``, ``likelihood``, ``model.softplus_scale``).
        n_prime: conditioning horizon in Euler steps (``T' = n_prime * dt``).
        mcmc: sampler settings (``kind``, ``n_chains``, ``n_steps``, ``burn_in_frac``, ``beta_init``, ``target_accept``, ``adapt``).
        device: torch device for the simulation.
        generator: source of all randomness; ``None`` -> a CPU generator seeded with ``mcmc.seed``.
        n_steps_total: Euler steps of the returned paths (default ``cfg.data.n_steps``); must be ``>= n_prime``.
        verbose: log progress every ~10% of the iterations via ``ndim.utils.get_logger("ndim.mcmc")``.
        z0: initial intensity (default ``cfg.data.truth.z0``).
        positivity: positive-part scheme of the Euler map. Default ``full_truncation``, the
            scheme of the data-generating truth, so that with the true drift/diffusion the target is the exact
            reference posterior of Fig. 4; pass ``cfg.model.positivity`` when sampling under a *learned* prior
            trained with another scheme (e.g. the EM baseline).

    Returns:
        ``paths [mcmc.n_chains, n_steps_total + 1]`` -- the final (positive-part) intensity path of every chain;
        the chains are independent, so these are ``n_chains`` approximate posterior draws -- and an ``MCMCDiagnostics``.
    """
    N, n_prime, z0_val, pos = _prepare_run(cfg, n_prime, mcmc, n_steps_total, z0, positivity)
    ev = _as_event_matrix(events_row, device)
    if ev.shape[0] != 1:
        raise ValueError("sample_posterior_mcmc expects a single observation; use sample_posterior_mcmc_batch for several")
    gen = generator if generator is not None else _default_generator(mcmc)
    paths, diags = _run_chains(drift, diffusion, ev, cfg, n_prime, mcmc, device, gen, N, z0_val, pos, verbose, link=link, return_raw=return_raw,
                               z0_from_normal=_truth_z0_map(cfg, z0))
    return paths[0], diags[0]


@torch.no_grad()
def sample_posterior_mcmc_batch(
    drift: CoefficientFn,
    diffusion: CoefficientFn,
    events: Union[Tensor, np.ndarray],
    cfg: ExperimentConfig,
    n_prime: int,
    mcmc: MCMCConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    n_steps_total: Optional[int] = None,
    verbose: bool = False,
    z0: Optional[float] = None,
    positivity: Optional[str] = None,
    chunk_size: Optional[int] = None,
    link: Optional[IntensityLink] = None,
    return_raw: bool = False,
    n_lead: int = 0,
    t0: float = 0.0,
    initial_raw: Optional[float] = None,
) -> Tuple[Tensor, List[MCMCDiagnostics]]:
    """Vectorised version of :func:`sample_posterior_mcmc` for ``events [n_obs, K]`` (padded with ``+inf``).

    ``n_lead``, ``t0`` and ``initial_raw`` describe a prior with a burn-in (``n_lead`` Euler steps before the window,
    started at ``t0`` from the latent state ``initial_raw``): the sampler under a *learned* prior of that kind.

    All observations and chains are advanced in a single Euler simulation per iteration (rows ``o*n_chains + c``,
    likelihood via ``poisson_loglik(Z, events, m=n_chains, ...)``); the pCN step size ``beta`` is adapted
    separately for every observation because the posterior sharpness scales with its event count.
    ``chunk_size`` (optional) processes the observations in consecutive groups to bound memory.

    Returns ``paths [n_obs, n_chains, n_steps_total + 1]`` and one ``MCMCDiagnostics`` per observation.
    """
    N, n_prime, z0_val, pos = _prepare_run(cfg, n_prime, mcmc, n_steps_total, z0, positivity)
    ev = _as_event_matrix(events, device)
    n_obs = int(ev.shape[0])
    gen = generator if generator is not None else _default_generator(mcmc)
    if n_obs == 0:
        return torch.empty(0, int(mcmc.n_chains), N + 1, device=device), []
    step = n_obs if (chunk_size is None or int(chunk_size) <= 0) else int(chunk_size)
    out_paths: List[Tensor] = []
    out_diags: List[MCMCDiagnostics] = []
    for start in range(0, n_obs, step):
        stop = min(start + step, n_obs)
        prefix = f"obs {start}-{stop - 1}/{n_obs} " if (verbose and step < n_obs) else ""
        p, d = _run_chains(drift, diffusion, ev[start:stop], cfg, n_prime, mcmc, device, gen, N, z0_val, pos, verbose, prefix, link=link, return_raw=return_raw,
                           z0_from_normal=None if initial_raw is not None else _truth_z0_map(cfg, z0),
                           n_lead=n_lead, t0=t0, initial_raw=initial_raw)
        out_paths.append(p)
        out_diags.extend(d)
    return torch.cat(out_paths, dim=0), out_diags


__all__ = [
    "MCMCDiagnostics",
    "metropolis_accept",
    "sample_posterior_mcmc",
    "sample_posterior_mcmc_batch",
    "summarize_diagnostics",
    "BETA_MIN",
    "BETA_MAX",
    "LOG_RATIO_CLIP",
    "UNIFORM_MIN",
    "DEFAULT_POSITIVITY",
]
