"""Tests for ``ndim.likelihood`` (paper Eq. 10): closed forms, the left grid convention, windows.

All tests use ``dt = 0.25`` (``T = 4``, 16 intervals) so that grid times are exactly representable in
float32 and events placed on grid points are unambiguous.
"""
from __future__ import annotations

import math

import pytest
import torch

from ndim.config import LikelihoodConfig
from ndim.likelihood import (
    event_grid_index,
    integrated_intensity,
    intensity_at_events,
    poisson_loglik,
    poisson_loglik_window,
)

DT = 0.25
N = 16          # T = 4
INF = float("inf")
CFG = LikelihoodConfig(eps=1e-6)


def _pad(*rows):
    K = max(len(r) for r in rows)
    out = torch.full((len(rows), K), INF)
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = torch.tensor(r)
    return out


# --------------------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_prime", [16, 8, 3])
def test_constant_intensity_closed_form(n_prime: int) -> None:
    """Constant ``lambda = c`` on ``[0, T']`` with ``K`` events: ``log p = K log c - c T'``."""
    c = 7.5
    T_prime = n_prime * DT
    events = _pad([0.1, 0.4, 0.55, 0.7])                    # all inside (0, 0.75] <= T' for every n'
    K = 4
    paths = torch.full((1, N + 1), c)
    ll = poisson_loglik(paths, events, m=1, dt=DT, n_prime=n_prime, cfg=CFG)
    assert ll.shape == (1,)
    assert float(ll[0]) == pytest.approx(K * math.log(c) - c * T_prime, rel=1e-6)


def test_constant_intensity_no_events() -> None:
    c = 3.0
    events = torch.full((1, 1), INF)
    paths = torch.full((1, N + 1), c)
    ll = poisson_loglik(paths, events, 1, DT, N, CFG)
    assert float(ll[0]) == pytest.approx(-c * 4.0, rel=1e-6)


def test_piecewise_constant_closed_form() -> None:
    """Intensity ``Z_k`` piecewise constant on ``[t_k, t_{k+1})``; ``lambda(tau) = Z_{floor(tau/dt)}``."""
    Z = torch.arange(1.0, N + 2.0)                          # Z_k = k + 1
    paths = Z[None, :]
    events = _pad([0.3, 1.0, 2.6])                          # bins 1, 4, 10
    ll = poisson_loglik(paths, events, 1, DT, N, CFG)
    expected = math.log(2.0) + math.log(5.0) + math.log(11.0) - DT * float(Z[:N].sum())
    assert float(ll[0]) == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------------------
# left grid convention
# --------------------------------------------------------------------------------------
def test_event_reads_the_grid_point_to_its_left() -> None:
    """``lambda(tau) = Z_{floor(tau/dt)}``: an event on ``t_k`` reads ``Z_k``, one at ``T' = n' dt`` reads ``Z_{n'-1}``."""
    events = torch.tensor([[0.125, 0.25, 1.0, 2.75, 4.0]])  # inside the first interval, on grid points, at T
    idx, valid = event_grid_index(events, DT, N, 4.0)
    assert idx.tolist() == [[0, 1, 4, 11, N - 1]] and bool(valid.all())
    paths = torch.tensor([[2.0, 6.0] + [6.0] * (N - 1)])
    ll = poisson_loglik(paths, _pad([0.125]), 1, DT, N, CFG)  # mid-interval event sees Z_0 = 2, not an average
    assert float(ll[0]) == pytest.approx(math.log(2.0) - DT * float(paths[0, :N].sum()), rel=1e-6)


def test_integral_is_the_left_riemann_sum() -> None:
    t = torch.arange(N + 1) * DT
    paths = (1.0 + 2.0 * t)[None, :]                        # Z_t = 1 + 2 t
    n_prime = 8                                             # T' = 2
    left = integrated_intensity(paths, DT, n_prime)
    assert float(left[0]) == pytest.approx(DT * float(paths[0, :n_prime].sum()), rel=1e-6)
    assert float(left[0]) < 2.0 + 4.0                       # below int_0^2 (1 + 2t) dt for an increasing path


# --------------------------------------------------------------------------------------
# windows and horizon restriction
# --------------------------------------------------------------------------------------
def test_events_after_horizon_are_ignored() -> None:
    paths = torch.rand(1, N + 1, generator=torch.Generator().manual_seed(1)) * 10 + 1
    n_prime = 8                                             # T' = 2
    inside = [0.3, 1.2, 1.9]
    ll_a = poisson_loglik(paths, _pad(inside), 1, DT, n_prime, CFG)
    ll_b = poisson_loglik(paths, _pad(inside + [2.1, 3.5, 3.99]), 1, DT, n_prime, CFG)
    assert torch.allclose(ll_a, ll_b)
    # ... and the +inf padding never contributes
    ll_c = poisson_loglik(paths, _pad(inside + [INF, INF, INF, INF]), 1, DT, n_prime, CFG)
    assert torch.allclose(ll_a, ll_c)
    # an event exactly at T' is inside the closed interval (0, T']
    ll_d = poisson_loglik(paths, _pad(inside + [2.0]), 1, DT, n_prime, CFG)
    assert float(ll_d[0]) > float(ll_a[0]) - 1e-6 and not torch.allclose(ll_a, ll_d)


def test_window_loglik_sums_and_matches_full() -> None:
    g = torch.Generator().manual_seed(2)
    paths = torch.rand(3, N + 1, generator=g) * 10 + 1
    events = _pad([0.3, 1.2, 1.9, 2.5, 3.7], [0.8, 3.2], [])
    full = poisson_loglik(paths, events, 1, DT, N, CFG)
    w1 = poisson_loglik_window(paths, events, 1, DT, 0, 6, CFG)
    w2 = poisson_loglik_window(paths, events, 1, DT, 6, 12, CFG)
    w3 = poisson_loglik_window(paths, events, 1, DT, 12, N, CFG)
    assert torch.allclose(w1 + w2 + w3, full, atol=1e-5)
    assert torch.allclose(w1, poisson_loglik(paths, events, 1, DT, 6, CFG))
    # the window (n_start dt, n_end dt] only sees its own events and its own integral
    Zc = 4.0
    const = torch.full((1, N + 1), Zc)
    ww = poisson_loglik_window(const, _pad([0.1, 1.6, 2.4, 3.9]), 1, DT, 6, 12, CFG)   # (1.5, 3.0]: events 1.6, 2.4
    assert float(ww[0]) == pytest.approx(2 * math.log(Zc) - Zc * 1.5, rel=1e-6)


def test_m_paths_layout_and_intensity_floor() -> None:
    """Rows ``b*m + j``: identical rows of an observation give identical log-likelihoods; a zero
    intensity at an event is floored at ``eps`` inside the log."""
    B, m = 2, 3
    base = torch.rand(B, N + 1, generator=torch.Generator().manual_seed(4)) * 10 + 1
    paths = base.repeat_interleave(m, dim=0)                # [B*m, N+1]
    events = _pad([0.3, 1.2], [2.9])
    ll = poisson_loglik(paths, events, m, DT, N, CFG)
    assert ll.shape == (B * m,)
    assert torch.allclose(ll.view(B, m), ll.view(B, m)[:, :1].expand(B, m))
    assert torch.allclose(ll.view(B, m)[:, 0], poisson_loglik(base, events, 1, DT, N, CFG))
    # zero intensity at the event: log(eps) instead of -inf
    zero = torch.zeros(1, N + 1)
    ll0 = poisson_loglik(zero, _pad([0.3]), 1, DT, N, CFG)
    assert torch.isfinite(ll0).all()
    assert float(ll0[0]) == pytest.approx(math.log(1e-6), rel=1e-6)
    # shape mismatch between paths and B*m is caught
    with pytest.raises(AssertionError):
        poisson_loglik(base, events, m=2, dt=DT, n_prime=N, cfg=CFG)
    lam = intensity_at_events(paths, m, event_grid_index(events, DT, N, 4.0)[0], eps=1e-6)
    assert lam.shape == (B * m, 2)
