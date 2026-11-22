"""Tests for ``ndim.elbo``: the Monte-Carlo ELBO (paper Eq. 6) and the training objective."""
from __future__ import annotations

import pytest
import torch

from ndim.config import ExperimentConfig
from ndim.data import Dataset
from ndim.elbo import compute_objective, evaluate_elbo
from ndim.likelihood import poisson_loglik
from ndim.model import NDIM
from ndim.sde import brownian_increments


def test_compute_objective_shapes_and_identities(smoke_cfg: ExperimentConfig, smoke_dataset: Dataset, tiny_model: NDIM, gen, device) -> None:
    B, m = 3, 2
    n_prime = smoke_cfg.horizon_steps(1.0)
    events = smoke_dataset.train.events[:B]
    W = brownian_increments(B * m, n_prime, smoke_cfg.dt, device, gen)
    out = compute_objective(tiny_model, events, n_prime, m, W, smoke_cfg)
    assert out.loglik.shape == (B,) and out.kl.shape == (B,) and out.elbo.shape == (B,)
    assert out.loss.dim() == 0 and out.loss.requires_grad
    assert torch.allclose(out.elbo, out.loglik - out.kl, atol=1e-4)
    assert torch.all(out.kl >= 0)
    # kl_weight = 0.5 (ELBO): loss = -mean elbo ; the reported elbo always uses 1/2
    assert float(out.loss.detach()) == pytest.approx(-float(out.elbo.mean().detach()), rel=1e-5)
    out0 = compute_objective(tiny_model, events, n_prime, m, W, smoke_cfg, kl_weight=0.0)
    assert float(out0.loss.detach()) == pytest.approx(-float(out0.loglik.mean().detach()), rel=1e-5)
    assert torch.allclose(out0.elbo, out.elbo, atol=1e-4)
    # per-observation values are means over the m rows of the row-wise likelihood
    ll_rows = poisson_loglik(out.sim.Z.detach(), events, m, smoke_cfg.dt, n_prime, smoke_cfg.likelihood)
    assert torch.allclose(out.loglik, ll_rows.view(B, m).mean(1), atol=1e-4)
    kl_rows = 0.5 * (out.sim.u.detach() ** 2).sum(1) * smoke_cfg.dt
    assert torch.allclose(out.kl, kl_rows.view(B, m).mean(1), atol=1e-5)
    assert set(out.summary()) == {"loss", "loglik", "kl", "elbo"}


def test_loss_gradient_reaches_prior_and_posterior(smoke_cfg, smoke_dataset, tiny_model, gen, device) -> None:
    """The pathwise ELBO gradient reaches the prior drift through the simulated paths, and the posterior network."""
    B, m = 2, 2
    n_prime = smoke_cfg.horizon_steps(1.0)
    events = smoke_dataset.train.events[:B]
    W = brownian_increments(B * m, n_prime, smoke_cfg.dt, device, gen)
    tiny_model.zero_grad(set_to_none=True)
    compute_objective(tiny_model, events, n_prime, m, W, smoke_cfg).loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in tiny_model.drift.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in tiny_model.score_net.parameters())


def test_longer_simulation_does_not_change_objective(smoke_cfg, smoke_dataset, tiny_model, gen, device) -> None:
    """``n_steps > n_prime`` (posterior predictive setting): controls off after ``n'``; likelihood and KL on
    ``[0, T']`` are identical to the ``n_steps == n_prime`` run under the same noise."""
    B, m = 2, 2
    N = smoke_cfg.data.n_steps
    n_prime = smoke_cfg.horizon_steps(0.5)
    events = smoke_dataset.train.events[:B]
    W = brownian_increments(B * m, N, smoke_cfg.dt, device, gen)
    with torch.no_grad():
        a = compute_objective(tiny_model, events, n_prime, m, W[:, :n_prime], smoke_cfg)
        b = compute_objective(tiny_model, events, n_prime, m, W, smoke_cfg, n_steps=N)
    assert a.sim.Z.shape == (B * m, n_prime + 1) and b.sim.Z.shape == (B * m, N + 1)
    assert torch.allclose(a.sim.Z, b.sim.Z[:, : n_prime + 1])
    assert torch.allclose(a.loglik, b.loglik) and torch.allclose(a.kl, b.kl)
    assert float(b.sim.u[:, n_prime:].abs().sum()) == 0.0


def test_evaluate_elbo_deterministic_with_same_generator(smoke_cfg, smoke_dataset, tiny_model, device) -> None:
    n_prime = smoke_cfg.horizon_steps(1.0)
    events = smoke_dataset.test.events
    r1 = evaluate_elbo(tiny_model, events, n_prime, m=3, cfg=smoke_cfg, device=device, generator=torch.Generator().manual_seed(42))
    r2 = evaluate_elbo(tiny_model, events, n_prime, m=3, cfg=smoke_cfg, device=device, generator=torch.Generator().manual_seed(42))
    assert set(r1) == {"loglik", "kl", "elbo"}
    assert r1 == r2
    assert all(isinstance(v, float) for v in r1.values())
    assert r1["elbo"] == pytest.approx(r1["loglik"] - r1["kl"], rel=1e-4)
    r3 = evaluate_elbo(tiny_model, events, n_prime, m=3, cfg=smoke_cfg, device=device, generator=torch.Generator().manual_seed(43))
    assert r3 != r1
    # does not leave gradients behind and does not require grad mode
    assert all(p.grad is None for p in tiny_model.parameters())


def test_evaluate_elbo_reports_what_the_two_bounds_did(make_cfg, smoke_dataset, device) -> None:
    """``bounds`` is filled with the largest ``|u|`` and the share of the simulated steps on which the bound on the
    control and the bound on the diffusion held; the numbers of the evaluation do not change."""
    from ndim.model import NDIM

    events = smoke_dataset.test.events
    cfg = make_cfg("train.max_abs_control=null")                        # the latent CIR diffusion with its bound, no bound on the control
    n_prime = cfg.horizon_steps(1.0)
    torch.manual_seed(0)
    model = NDIM(cfg)
    bounds: dict = {}
    plain = evaluate_elbo(model, events, n_prime, 3, cfg, device, torch.Generator().manual_seed(1))
    assert evaluate_elbo(model, events, n_prime, 3, cfg, device, torch.Generator().manual_seed(1), bounds=bounds) == plain
    assert bounds["n_steps"] == len(events) * 3 * n_prime and bounds["largest_abs_control"] > 0.0
    assert bounds["control_bound"] is None and bounds["control_bound_share"] is None
    assert bounds["diffusion_bound_share"] == 0.0                       # the paths start at intensity 5, far above eta^2 dt = 0.16
    # a control bound below the controls of the model, and a bound on the diffusion above every intensity it visits
    tight = make_cfg("train.max_abs_control=1.0e-4", "model.diffusion.max_step_std=1.0e-3")
    torch.manual_seed(0)
    bounds = {}
    evaluate_elbo(NDIM(tight), events, n_prime, 3, tight, device, torch.Generator().manual_seed(1), bounds=bounds)
    assert bounds["control_bound"] == 1.0e-4 and bounds["largest_abs_control"] <= 1.0e-4 + 1e-9
    assert bounds["control_bound_share"] > 0.5 and bounds["diffusion_bound_share"] == 1.0
    # the intensity parametrization has no bound on its diffusion
    intensity = make_cfg("model.state_space=intensity", "train.max_abs_control=null")
    bounds = {}
    evaluate_elbo(NDIM(intensity), events, n_prime, 3, intensity, device, torch.Generator().manual_seed(1), bounds=bounds)
    assert bounds["diffusion_bound_share"] is None and bounds["control_bound_share"] is None


def test_evaluate_elbo_is_mean_over_observations(smoke_cfg, smoke_dataset, tiny_model, device) -> None:
    """Chunked evaluation averages per-observation values: with one chunk it equals ``compute_objective``
    on the same Brownian draw."""
    n_prime = smoke_cfg.horizon_steps(1.0)
    events = smoke_dataset.test.events[:4]
    m = 2
    g = torch.Generator().manual_seed(3)
    r = evaluate_elbo(tiny_model, events, n_prime, m, smoke_cfg, device, generator=g, chunk_size=32)
    W = brownian_increments(4 * m, n_prime, smoke_cfg.dt, device, torch.Generator().manual_seed(3))
    with torch.no_grad():
        out = compute_objective(tiny_model, events, n_prime, m, W, smoke_cfg)
    assert r["loglik"] == pytest.approx(float(out.loglik.mean()), rel=1e-5)
    assert r["kl"] == pytest.approx(float(out.kl.mean()), rel=1e-5)
    assert r["elbo"] == pytest.approx(float(out.elbo.mean()), rel=1e-5)
