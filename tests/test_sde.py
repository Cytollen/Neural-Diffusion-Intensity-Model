"""Tests for ``ndim.sde``: Brownian increments, prior / posterior Euler–Maruyama simulation (paper Eq. 1, 3)."""
from __future__ import annotations

import math

import pytest
import torch

from ndim.config import ExperimentConfig
from ndim.data import make_truth_diffusion, make_truth_drift, simulate_truth_paths
from ndim.model import NDIM, build_model
from ndim.models.common import positive_part
from ndim.sde import SimOutput, brownian_increments, simulate_posterior, simulate_prior


def _events_batch(smoke_dataset, B: int) -> torch.Tensor:
    return smoke_dataset.test.events[:B]


def _sim(model: NDIM, cfg: ExperimentConfig, events: torch.Tensor, m: int, n_steps: int, n_prime: int, W: torch.Tensor, **kw) -> SimOutput:
    t_grid = torch.arange(n_prime) * cfg.dt
    ctx = model.score_net.prepare(events, t_grid, n_prime * cfg.dt)
    return simulate_posterior(
        model.drift, model.diffusion, model.score_net, ctx, z0=cfg.data.truth.z0, n_steps=n_steps, n_prime=n_prime,
        dt=cfg.dt, brownian=W, **model.sde_kwargs, **kw
    )


# --------------------------------------------------------------------------------------
# Brownian increments
# --------------------------------------------------------------------------------------
def test_brownian_increments_shape_scale_and_reproducibility(device: torch.device) -> None:
    dt = 0.16
    W = brownian_increments(2000, 25, dt, device, torch.Generator().manual_seed(0))
    assert W.shape == (2000, 25) and W.dtype == torch.float32
    assert abs(float(W.mean())) < 5 * math.sqrt(dt / W.numel())
    assert float(W.var()) == pytest.approx(dt, rel=0.05)     # already multiplied by sqrt(dt)
    W2 = brownian_increments(2000, 25, dt, device, torch.Generator().manual_seed(0))
    assert torch.equal(W, W2)
    W3 = brownian_increments(2000, 25, dt, device, torch.Generator().manual_seed(1))
    assert not torch.equal(W, W3)


# --------------------------------------------------------------------------------------
# prior simulation
# --------------------------------------------------------------------------------------
def test_simulate_prior_shapes_positivity_and_start(smoke_cfg: ExperimentConfig, gen: torch.Generator, device) -> None:
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    drift = make_truth_drift(smoke_cfg.data.truth)
    diff = make_truth_diffusion(smoke_cfg.data.truth)
    W = brownian_increments(6, N, dt, device, gen)
    Z = simulate_prior(drift, diff, smoke_cfg.data.truth.z0, N, dt, W)
    assert Z.shape == (6, N + 1)
    assert torch.all(Z >= 0)
    assert torch.allclose(Z[:, 0], torch.full((6,), 5.0))
    assert not Z.requires_grad


def test_simulate_prior_deterministic_drift_only() -> None:
    """With ``sigma = 0`` and constant drift ``b`` the Euler path is exactly ``z0 + b k dt``."""
    b, z0, dt, N = 2.5, 1.0, 0.25, 8
    Z = simulate_prior(lambda z, t: torch.full_like(z, b), lambda z, t: torch.zeros_like(z), z0, N, dt, torch.zeros(3, N))
    expected = z0 + b * dt * torch.arange(N + 1, dtype=torch.float32)
    assert torch.allclose(Z, expected[None, :].expand(3, N + 1))


def test_simulate_prior_matches_truth_simulator_under_common_noise(smoke_cfg: ExperimentConfig, gen, device) -> None:
    """``simulate_prior`` with the truth coefficients reproduces ``simulate_truth_paths`` given the same noise."""
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    W = brownian_increments(5, N, dt, device, gen)
    truth = smoke_cfg.data.truth
    Za = simulate_prior(make_truth_drift(truth), make_truth_diffusion(truth), truth.z0, N, dt, W)
    Zb = simulate_truth_paths(truth, smoke_cfg.data.T, N, 5, brownian=W)
    assert torch.allclose(Za, Zb, atol=1e-5)


@pytest.mark.parametrize("mode", ["full_truncation", "reflect", "softplus"])
def test_positivity_modes(mode: str) -> None:
    raw = torch.tensor([-2.0, -0.1, 0.0, 0.3, 5.0])
    Z = positive_part(raw, mode, softplus_scale=0.5)
    assert torch.all(Z >= 0)
    if mode == "full_truncation":
        assert torch.equal(Z, torch.relu(raw))
    elif mode == "reflect":
        assert torch.equal(Z, raw.abs())
    else:
        assert torch.all(Z > 0) and float(Z[-1]) == pytest.approx(5.0, abs=1e-3)
    with pytest.raises(ValueError):
        positive_part(raw, "clamp")
    # a strongly negative drift drives the raw state below zero but the intensity stays >= 0
    Zp = simulate_prior(lambda z, t: torch.full_like(z, -50.0), lambda z, t: torch.zeros_like(z), 1.0, 5, 0.25,
                        torch.zeros(1, 5), positivity=mode)
    assert torch.all(Zp >= 0)


# --------------------------------------------------------------------------------------
# posterior simulation
# --------------------------------------------------------------------------------------
def test_simulate_posterior_shapes_and_controls_zero_after_horizon(smoke_cfg, smoke_dataset, tiny_model, gen, device) -> None:
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    B, m = 3, 2
    n_prime = smoke_cfg.horizon_steps(0.5)
    events = _events_batch(smoke_dataset, B)
    W = brownian_increments(B * m, N, dt, device, gen)
    sim = _sim(tiny_model, smoke_cfg, events, m, N, n_prime, W)
    assert sim.Z.shape == (B * m, N + 1) and sim.raw.shape == (B * m, N + 1)
    assert sim.u.shape == (B * m, N) and sim.s.shape == (B * m, N)
    assert sim.b.shape == (B * m, N) and sim.sigma.shape == (B * m, N)
    assert sim.n_prime == n_prime
    assert torch.all(sim.Z >= 0)
    assert torch.equal(sim.Z, torch.exp(sim.raw))
    assert torch.allclose(sim.Z[:, 0], torch.full((B * m,), float(smoke_cfg.data.truth.z0)))
    # controls / scores are exactly zero once t >= T'
    assert float(sim.u[:, n_prime:].detach().abs().sum()) == 0.0
    assert float(sim.s[:, n_prime:].detach().abs().sum()) == 0.0
    # ... and non-zero before (the score network of the default model; no jump term is added, jump_term = none)
    assert float(sim.u[:, :n_prime].detach().abs().sum()) > 0.0
    # kl_integral = int_0^{T'} u^2 dt
    assert torch.allclose(sim.kl_integral, (sim.u ** 2).sum(1) * dt)
    # the graph is kept (posterior parameters are trainable)
    assert sim.Z.requires_grad and sim.u.requires_grad


def test_simulate_posterior_reduces_to_prior_without_controls(make_cfg, smoke_dataset, gen, device) -> None:
    """``score_scale = 0`` and ``jump_term = none`` => ``u = 0`` and the posterior path equals the prior path
    simulated with the model's drift / diffusion under the same Brownian increments."""
    cfg = make_cfg("model.posterior.score_scale=0.0", "model.posterior.jump_term=none")
    torch.manual_seed(0)
    model = build_model(cfg)
    N, dt = cfg.data.n_steps, cfg.dt
    B, m = 2, 3
    W = brownian_increments(B * m, N, dt, device, gen)
    with torch.no_grad():
        sim = _sim(model, cfg, _events_batch(smoke_dataset, B), m, N, N, W)
        Zp = simulate_prior(model.drift, model.diffusion, cfg.data.truth.z0, N, dt, W, **model.sde_kwargs)
    assert float(sim.u.detach().abs().max()) == 0.0
    assert torch.allclose(sim.Z, Zp, atol=1e-5)


def test_simulate_posterior_common_noise_reproducible(smoke_cfg, smoke_dataset, tiny_model, device) -> None:
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    B, m = 2, 2
    events = _events_batch(smoke_dataset, B)
    W1 = brownian_increments(B * m, N, dt, device, torch.Generator().manual_seed(5))
    W1b = brownian_increments(B * m, N, dt, device, torch.Generator().manual_seed(5))
    W2 = brownian_increments(B * m, N, dt, device, torch.Generator().manual_seed(6))
    with torch.no_grad():
        a = _sim(tiny_model, smoke_cfg, events, m, N, N, W1).Z
        b = _sim(tiny_model, smoke_cfg, events, m, N, N, W1b).Z
        c = _sim(tiny_model, smoke_cfg, events, m, N, N, W2).Z
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_path_gradient_reaches_drift_and_score_parameters(smoke_cfg, smoke_dataset, tiny_model, gen, device) -> None:
    """The simulated path is differentiable in the prior drift and in the score network, and the recorded
    drift values are the model's drift at the simulated states."""
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    B, m = 2, 2
    events = _events_batch(smoke_dataset, B)
    W = brownian_increments(B * m, N, dt, device, gen)
    drift_params = [p for p in tiny_model.drift.parameters() if p.requires_grad]
    score_params = [p for p in tiny_model.score_net.parameters() if p.requires_grad]
    assert drift_params and score_params

    tiny_model.zero_grad(set_to_none=True)
    sim = _sim(tiny_model, smoke_cfg, events, m, N, N, W)
    (sim.Z.sum() + (sim.u ** 2).sum()).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in drift_params)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in score_params)
    with torch.no_grad():
        t0 = torch.zeros(B * m)
        assert torch.allclose(sim.b[:, 0], tiny_model.drift(tiny_model.coefficient_state(sim.raw[:, 0]), t0))


def test_simulate_posterior_argument_validation(smoke_cfg, smoke_dataset, tiny_model, gen, device) -> None:
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    W = brownian_increments(2, N, dt, device, gen)
    events = _events_batch(smoke_dataset, 1)
    with pytest.raises(ValueError):
        _sim(tiny_model, smoke_cfg, events, 2, n_steps=N - 5, n_prime=N, W=W)      # n_steps < n_prime
    with pytest.raises(ValueError):
        _sim(tiny_model, smoke_cfg, events, 2, n_steps=N, n_prime=N, W=W[:, : N - 1])  # brownian too short


def test_score_and_control_clipping(smoke_cfg, smoke_dataset, tiny_model, gen, device) -> None:
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    B, m = 2, 2
    events = _events_batch(smoke_dataset, B)
    W = brownian_increments(B * m, N, dt, device, gen)
    with torch.no_grad():
        sim = _sim(tiny_model, smoke_cfg, events, m, N, N, W, max_abs_score=0.05, max_abs_control=0.02)
    assert float(sim.s.detach().abs().max()) <= 0.05 + 1e-6
    assert float(sim.u.detach().abs().max()) <= 0.02 + 1e-6
