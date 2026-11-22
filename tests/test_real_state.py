"""Real latent states, linked likelihoods, Ito comparisons and replaceable corrections."""
from dataclasses import dataclass
import sys
import types

import pytest
import torch
from torch import nn

from ndim.elbo import compute_objective
from ndim.gradients import sensitivity_backward
from ndim.links import build_link, intensity_coefficients
from ndim.mcmc import sample_posterior_mcmc
from ndim.model import NDIM
from ndim.models.posterior import PosteriorCorrection, register_posterior
from ndim.sde import simulate_prior


@pytest.mark.parametrize("kind", ["exponential", "softplus"])
def test_link_inverse_and_ito_derivatives(kind):
    link = build_link(kind)
    x = torch.tensor([-8., -2., 0., 2., 8.], dtype=torch.float64, requires_grad=True)
    z = link(x)
    assert torch.all(z > 0)
    assert torch.allclose(link.inverse(z), x, atol=1e-12)
    first = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    assert torch.allclose(first, link.first(x), atol=1e-12)
    assert torch.allclose(second, link.second(x), atol=1e-12)
    assert torch.allclose(link.log_derivative(x), first / z, atol=1e-12)
    with pytest.raises(ValueError):
        link.inverse_scalar(0.)


def test_unprojected_prior_uses_negative_states_and_links_only_outputs():
    link = build_link()
    visited = []
    def drift(x, t):
        visited.append(x.clone())
        return -torch.ones_like(x)
    W = torch.zeros(2, 4, dtype=torch.float64)
    z = simulate_prior(drift, lambda x, t: x * 0., 1., 4, 1., W, link=link)
    expected_x = -torch.arange(5, dtype=torch.float64)
    assert torch.equal(torch.stack(visited, dim=1), expected_x[:-1].expand(2, 4))
    assert torch.allclose(z, expected_x.exp().expand(2, 5))
    raw = simulate_prior(drift, lambda x, t: x * 0., 1., 4, 1., W, link=link, return_raw=True)
    assert torch.equal(raw, expected_x.expand(2, 5))


@pytest.mark.parametrize("kind", ["exponential", "softplus"])
def test_ito_coefficients_include_noise_drift_correction(kind):
    link = build_link(kind)
    x = torch.tensor([-1., 0., 2.], dtype=torch.float64)
    z, t = link(x), torch.zeros_like(x)
    drift = lambda x, t: 0.3 - 0.2 * x
    diffusion = lambda x, t: torch.full_like(x, .4)
    b, sigma = intensity_coefficients(link, drift, diffusion, z, t)
    expected_b = link.first(x) * drift(x, t) + .5 * link.second(x) * .4**2
    assert torch.allclose(b, expected_b, atol=1e-12)
    assert torch.allclose(sigma, link.first(x) * .4, atol=1e-12)
    if kind == "exponential":
        assert torch.allclose(b, z * (drift(z.log(), t) + .5 * .4**2))
        assert torch.allclose(sigma, z * .4)


@pytest.mark.parametrize("kind", ["exponential", "softplus"])
def test_posterior_simulates_x_and_reports_intensity(make_cfg, kind):
    cfg = make_cfg(f"model.link={kind}", "data.truth.z0=0.2", "model.posterior.score_scale=0", "model.posterior.jump_term=none")
    model = NDIM(cfg).double()
    events = torch.tensor([[.5, 1.]], dtype=torch.float64)
    W = torch.zeros(2, cfg.data.n_steps, dtype=torch.float64)
    out = compute_objective(model, events, cfg.horizon_steps(.5), 2, W, cfg, n_steps=cfg.data.n_steps)
    assert torch.all(out.sim.raw[:, 0] < 0)
    assert torch.allclose(out.sim.Z, model.link(out.sim.raw))
    assert torch.allclose(out.sim.Z[:, 0], torch.full((2,), .2, dtype=torch.float64))
    assert torch.equal(out.sim.u, torch.zeros_like(out.sim.u))
    prior = simulate_prior(model.drift, model.diffusion, .2, cfg.data.n_steps, cfg.dt, W, **model.sde_kwargs)
    assert torch.allclose(out.sim.Z, prior)
    assert torch.isfinite(out.loss)


@pytest.mark.parametrize("kind", ["exponential", "softplus"])
@pytest.mark.parametrize("arch", ["deepset", "remaining_count"])
def test_jump_is_in_latent_coordinates(make_cfg, kind, arch):
    cfg = make_cfg(f"model.link={kind}", f"model.posterior.kind={arch}", "model.posterior.jump_term=raw", "model.posterior.score_scale=0")
    model = NDIM(cfg).double()
    state = torch.tensor([-2., 0., 1.], dtype=torch.float64)
    events = torch.tensor([[1., 2.]], dtype=torch.float64)
    ctx = model.score_net.prepare(events, torch.tensor([.5, 1.], dtype=torch.float64), 2.)
    drop = model.score_net.score(ctx, 0, state) - model.score_net.score(ctx, 1, state)
    assert torch.allclose(drop, model.link.log_derivative(state))
    assert torch.isfinite(drop).all()


@dataclass
class CustomContext:
    B: int
    t_grid: torch.Tensor
    counts: torch.Tensor


class LinearCorrection(PosteriorCorrection):
    """Example architecture using its own context, independent of ScoreContext and Deep Sets."""
    def __init__(self, cfg, T, link):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(cfg.options.get("weight", .02))))
        self.T = T

    def prepare(self, events, t_grid, t_bar, precompute_pool=True):
        counts = ((events > 0) & (events <= t_bar)).sum(1).to(events.dtype)
        return CustomContext(events.shape[0], t_grid, counts)

    def row_inputs(self, ctx, k, m):
        return {"t": ctx.t_grid[k], "counts": ctx.counts.repeat_interleave(m)}

    def forward(self, state, inputs):
        return self.weight * (inputs["counts"] / 100. - state + inputs["t"] / self.T)


def test_registered_and_importable_correction_checkpoint_and_gradients(make_cfg, monkeypatch, tmp_path):
    import ndim.models.posterior as posterior
    monkeypatch.setattr(posterior, "_POSTERIORS", dict(posterior._POSTERIORS))
    register_posterior("linear_test", LinearCorrection)
    with pytest.raises(ValueError):
        register_posterior("linear_test", LinearCorrection)
    cfg = make_cfg("model.posterior.kind=linear_test", "model.posterior.options.weight=0.03", "model.diffusion.trainable=true")
    model = NDIM(cfg).double()
    assert isinstance(model.posterior_correction, LinearCorrection)
    # Compare backprop with sensitivity for a context that has no r, R0, phi, or pooled e.
    events = torch.tensor([[.25, .5], [.75, float('inf')]], dtype=torch.float64)
    W = torch.randn(4, 5, dtype=torch.float64) * .1
    out = compute_objective(model, events, 5, 2, W, cfg)
    out.loss.backward()
    expected = {n: p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    sensitivity_backward(model, events, 5, 2, W, cfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.grad, expected[n], atol=1e-9, rtol=1e-7), n
    # An importable factory makes custom checkpoint reconstruction independent of registration.
    module = types.ModuleType("custom_posterior_test")
    module.factory = LinearCorrection
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg.model.posterior.factory = "custom_posterior_test:factory"
    model = NDIM(cfg)
    path = tmp_path/'custom.pt'
    model.save_checkpoint(path)
    loaded, _ = NDIM.load_checkpoint(path)
    assert isinstance(loaded.posterior_correction, LinearCorrection)
    assert loaded.cfg.model.posterior.options['weight'] == .03
    assert loaded.cfg.model.state_space == "real"
    assert torch.equal(loaded.score_net.weight, model.score_net.weight)


def test_injected_module_can_train(make_cfg):
    cfg = make_cfg()
    model = NDIM(cfg, posterior_correction=LinearCorrection(cfg.model.posterior, cfg.data.T, build_link()))
    events = torch.tensor([[.2, .4]])
    out = compute_objective(model, events, 4, 2, torch.zeros(2, 4), cfg)
    out.loss.backward()
    assert torch.isfinite(model.score_net.weight.grad)
    assert model.score_net.weight in model.param_groups()['posterior']


@pytest.mark.parametrize("kind", ["exponential", "softplus"])
def test_neural_diffusion_gradients_and_sensitivity(make_cfg, kind):
    cfg = make_cfg(f"model.link={kind}", "model.diffusion.kind=mlp", "model.diffusion.hidden=[4]", "model.diffusion.trainable=true")
    torch.manual_seed(3)
    model = NDIM(cfg).double()
    events = torch.tensor([[.25, .5]], dtype=torch.float64)
    W = torch.randn(2, 5, dtype=torch.float64) * .1
    assert torch.all(model.diffusion(torch.tensor([-5., -1., 1.], dtype=torch.float64), torch.zeros(3)) > 0)
    out = compute_objective(model, events, 5, 2, W, cfg)
    out.loss.backward()
    expected = {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    sensitivity_backward(model, events, 5, 2, W, cfg)
    for n,p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad, expected[n], atol=1e-9, rtol=1e-7), n


def test_mcmc_linked_prior_and_latent_returns(make_cfg):
    cfg = make_cfg("eval.mcmc.n_steps=0")
    model = NDIM(cfg)
    events = torch.tensor([.5])
    args = (model.drift, model.diffusion, events, cfg, 5, cfg.eval.mcmc, torch.device('cpu'))
    z, _ = sample_posterior_mcmc(*args, link=model.link, generator=torch.Generator().manual_seed(4))
    x, _ = sample_posterior_mcmc(*args, link=model.link, return_raw=True, generator=torch.Generator().manual_seed(4))
    assert torch.equal(z, model.link(x))
    assert torch.allclose(z[:, 0], torch.full((cfg.eval.mcmc.n_chains,), cfg.data.truth.z0))


def test_a_checkpoint_keeps_the_intensity_state_space(make_cfg, tmp_path):
    cfg = make_cfg("model.state_space=intensity", "model.diffusion.kind=cir_sqrt", "model.diffusion.eta_init=1", "model.drift.theta_init=40")
    model = NDIM(cfg)
    path = tmp_path/'intensity.pt'
    model.save_checkpoint(path)
    loaded, _ = NDIM.load_checkpoint(path)
    assert loaded.cfg.model.state_space == 'intensity' and loaded.cfg.to_dict() == cfg.to_dict()
    raw = torch.tensor([-1., 0., 5.])
    assert torch.equal(loaded.to_intensity(raw), raw.relu())
