"""Poisson log-likelihood of point-process observations given intensity paths on the Euler grid.

``log p(X | Z) = sum_{tau_i <= T'} log lambda(tau_i) - int_0^{T'} lambda_t dt``  (paper Eq. 10)

The intensity is piecewise constant on the grid (left convention): ``lambda(t) = Z_k`` for
``t in [t_k, t_{k+1})``. An event at ``tau`` therefore sees ``Z_{floor(tau / dt)}`` and the integral is the
left Riemann sum, consistent with the thinning sampler that generates the synthetic data (paper App. G).

Vectorised over ``B`` observations x ``m`` paths. Paths are ``[B*m, n'+1]`` (or longer; only the
first ``n'`` grid points are used, ``n' = T'/dt``), events are padded ``[B, K]`` with ``+inf``.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .config import LikelihoodConfig

Tensor = torch.Tensor


def event_grid_index(events: Tensor, dt: float, n_prime: int, t_bar: float) -> Tuple[Tensor, Tensor]:
    """Return ``(idx, valid)`` for events: ``lambda(tau) = Z[idx]`` with ``idx = floor(tau / dt)``.

    ``valid`` masks events in ``(0, t_bar]`` (padding is ``+inf`` so it is excluded automatically).
    The index is clamped to ``n' - 1``, so an event exactly at ``t_bar = n' dt`` reads ``Z_{n'-1}``.
    """
    valid = torch.isfinite(events) & (events <= float(t_bar)) & (events >= 0.0)
    pos = torch.where(valid, events, torch.zeros_like(events)) / float(dt)
    idx = torch.clamp(pos.floor().long(), min=0, max=max(n_prime - 1, 0))
    return idx, valid


def intensity_at_events(paths: Tensor, m: int, idx: Tensor, eps: float) -> Tensor:
    """``paths [B*m, G]`` -> intensities at events ``[B*m, K]`` (floored at ``eps``)."""
    Bm = paths.shape[0]
    B = idx.shape[0]
    assert Bm == B * m, f"paths rows {Bm} != B*m = {B}*{m}"
    idx_r = idx.repeat_interleave(m, dim=0)          # [B*m, K]
    return torch.clamp(torch.gather(paths, 1, idx_r), min=float(eps))


def integrated_intensity(paths: Tensor, dt: float, n_prime: int) -> Tensor:
    """``int_0^{T'} lambda dt`` for each row as the left Riemann sum ``dt * sum_{k < n'} Z_k``, ``T' = n' dt``."""
    return paths[:, :n_prime].sum(dim=1) * float(dt)


def poisson_loglik(
    paths: Tensor,
    events: Tensor,
    m: int,
    dt: float,
    n_prime: int,
    cfg: LikelihoodConfig,
) -> Tensor:
    """Per-path Poisson log-likelihood ``[B*m]`` of observations ``events [B, K]`` on ``[0, n' dt]``.

    ``paths`` must already be intensities: g(X) for a latent state, the positive part for an intensity state or the truth.
    """
    t_bar = float(n_prime) * float(dt)
    idx, valid = event_grid_index(events, dt, n_prime, t_bar)
    lam = intensity_at_events(paths, m, idx, cfg.eps)                  # [B*m, K]
    valid_r = valid.repeat_interleave(m, dim=0)
    log_terms = torch.where(valid_r, torch.log(lam), torch.zeros_like(lam)).sum(dim=1)
    integral = integrated_intensity(paths, dt, n_prime)
    return log_terms - integral


def poisson_loglik_window(
    paths: Tensor,
    events: Tensor,
    m: int,
    dt: float,
    n_start: int,
    n_end: int,
    cfg: LikelihoodConfig,
) -> Tensor:
    """Log-likelihood of the events in ``(n_start dt, n_end dt]`` only (the predictive term of the paper's Eq. 9)."""
    full = poisson_loglik(paths, events, m, dt, n_end, cfg)
    head = poisson_loglik(paths, events, m, dt, n_start, cfg) if n_start > 0 else torch.zeros_like(full)
    return full - head
