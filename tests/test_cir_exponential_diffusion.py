"""CIR diffusion in exponential coordinates: coefficients, gradients, the bound at low intensities and I/O."""
import math

import pytest
import torch

from ndim.config import load_config
from ndim.elbo import compute_objective
from ndim.gradients import sensitivity_backward
from ndim.model import NDIM


@pytest.mark.parametrize('trainable', [False, True])
def test_cir_diffusion_and_ito_drift_in_intensity_units(make_cfg, trainable):
    cfg = make_cfg('model.diffusion.kind=cir_exponential',
                   'model.diffusion.eta_init=0.7', 'model.diffusion.max_step_std=null',     # the form itself, without its bound
                   f'model.diffusion.trainable={str(trainable).lower()}')
    model = NDIM(cfg).double()
    x = torch.tensor([-8., -1., 0., 2., 8.], dtype=torch.float64, requires_grad=True)
    z, t = x.exp(), torch.zeros_like(x)
    eta = model.diffusion.eta
    assert model.comparable_truth_scalars() == {'eta': cfg.data.truth.eta}
    sigma = model.diffusion(x, t)
    assert torch.all(sigma > 0)
    assert model.diffusion.log_eta.requires_grad == trainable
    assert torch.allclose(model.intensity_diffusion(z, t), eta * z.sqrt(), atol=1e-12)
    # g''(X)*sigma_X^2/2 is eta^2/2, including at negative latent states.
    expected_drift = z * model.drift(x, t) + 0.5 * eta.square()
    assert torch.allclose(model.intensity_drift(z, t), expected_drift, atol=1e-12)
    derivative = torch.autograd.grad(sigma.sum(), x, retain_graph=True)[0]
    assert torch.allclose(derivative, -0.5 * sigma, atol=1e-12)
    if trainable:
        grad_eta = torch.autograd.grad(sigma.sum(), model.diffusion.log_eta)[0]
        assert torch.allclose(grad_eta, sigma.sum(), atol=1e-12)


@pytest.mark.parametrize('link', ['exponential', 'softplus'])
@pytest.mark.parametrize('step_std', [1.0, 2.0])
def test_cir_diffusion_of_a_latent_state_is_bounded_at_low_intensities(make_cfg, link, step_std):
    """Above the intensity ``eta^2 dt / max_step_std^2`` the diffusion is ``eta sqrt(Z)``; below it the diffusion of the
    intensity per unit of intensity is held at ``max_step_std / sqrt(dt)``, and neither values nor gradients overflow."""
    cfg = make_cfg('model.diffusion.kind=cir', 'model.diffusion.eta_init=0.7', 'model.diffusion.trainable=true',
                   f'model.link={link}', f'model.diffusion.max_step_std={step_std}')
    model = NDIM(cfg).double()
    eta = model.diffusion.eta.detach()
    cap = step_std / math.sqrt(cfg.dt)
    floor = float(eta) ** 2 * cfg.dt / step_std ** 2
    above = torch.tensor([1.5, 5., 80.], dtype=torch.float64) * floor
    below = torch.tensor([1e-9, 1e-3, 0.5], dtype=torch.float64) * floor
    zero = torch.zeros(3, dtype=torch.float64)
    assert torch.allclose(model.intensity_diffusion(above, zero), eta * above.sqrt(), rtol=1e-12, atol=0)
    assert torch.allclose(model.intensity_diffusion(below, zero), cap * below, rtol=1e-9, atol=0)
    inverse = model.link.inverse
    assert not model.diffusion.bound_active(inverse(above)).any() and model.diffusion.bound_active(inverse(below)).all()
    # far below, where exp(-x/2) and 1 / sqrt(g(x)) overflow: the coefficient and its gradients stay finite
    x = torch.tensor([-5000., -800., -100.], dtype=torch.float64, requires_grad=True)
    sigma = model.diffusion(x, zero)
    assert torch.allclose(sigma, torch.full_like(sigma, cap), rtol=1e-9, atol=0)
    grad_x, grad_eta = torch.autograd.grad(sigma.sum(), [x, model.diffusion.log_eta])
    assert torch.equal(grad_x, torch.zeros_like(grad_x)) and float(grad_eta) == 0.0
    # the bound follows the Euler step and can be switched off
    assert model.diffusion.log_cap == pytest.approx(math.log(cap))
    free = NDIM(make_cfg('model.diffusion.kind=cir', 'model.diffusion.eta_init=0.7', f'model.link={link}',
                         'model.diffusion.max_step_std=null')).double()
    assert free.diffusion.log_cap is None and not free.diffusion.bound_active(free.link.inverse(below)).any()
    assert torch.allclose(free.intensity_diffusion(below, zero), free.diffusion.eta.detach() * below.sqrt(), rtol=1e-9, atol=0)


def test_the_bound_on_the_diffusion_needs_the_euler_step_and_a_positive_value(make_cfg):
    from ndim.config import DiffusionConfig
    from ndim.models.diffusion import build_diffusion
    with pytest.raises(ValueError, match='needs the step'):
        build_diffusion(DiffusionConfig(kind='cir_exponential', eta_init=1.0))
    assert build_diffusion(DiffusionConfig(kind='cir_exponential', eta_init=1.0, max_step_std=None)).log_cap is None
    assert build_diffusion(DiffusionConfig(kind='cir_exponential', eta_init=1.0), dt=0.25).log_cap == pytest.approx(math.log(2.0))
    for bad in ('0', '-1', 'true'):
        with pytest.raises(ValueError, match='max_step_std must be a positive number'):
            make_cfg('model.diffusion.kind=cir', f'model.diffusion.max_step_std={bad}')
    with pytest.raises(ValueError, match='train.max_abs_control must be a positive number'):
        make_cfg('train.max_abs_control=0')


@pytest.mark.parametrize('step_std', ['null', '1.0', '0.5'])
def test_a_saved_run_keeps_its_bound_on_the_diffusion(make_cfg, tmp_path, step_std):
    cfg = make_cfg('model.diffusion.kind=cir', f'model.diffusion.max_step_std={step_std}')
    model = NDIM(cfg)
    model.save_checkpoint(tmp_path / 'bound.pt')
    loaded, _ = NDIM.load_checkpoint(tmp_path / 'bound.pt')
    assert loaded.cfg.model.diffusion.max_step_std == cfg.model.diffusion.max_step_std
    assert loaded.diffusion.log_cap == model.diffusion.log_cap
    assert (model.diffusion.log_cap is None) == (step_std == 'null')


class _HostileScore:
    """A posterior correction that pushes the latent state down, harder the lower it is: what an untrained network
    can do when it extrapolates."""

    def score(self, ctx, k, x):
        return -(5.0 + x.square())


@pytest.mark.parametrize('step_std, control, finite', [(1.0, 30.0, True), (None, 30.0, False), (1.0, None, False), (None, None, False)])
def test_the_two_bounds_together_keep_every_euler_step_bounded(step_std, control, finite):
    """With the bound on the diffusion and the bound on the control, one Euler step moves the latent state by at most
    ``|b| dt + max_step_std * (max_abs_control * sqrt(dt) + |xi|)`` whatever the correction returns, so no path runs off.
    Either bound alone does not give this."""
    from ndim.config import DiffusionConfig
    from ndim.links import ExponentialLink
    from ndim.models.diffusion import build_diffusion
    from ndim.sde import brownian_increments, simulate_posterior
    dt, n = 0.04, 125
    diffusion = build_diffusion(DiffusionConfig(kind='cir_exponential', eta_init=1.0, trainable=False, max_step_std=step_std), dt=dt)
    W = brownian_increments(64, n, dt, torch.device('cpu'), torch.Generator().manual_seed(0))
    sim = simulate_posterior(lambda x, t: torch.zeros_like(x), diffusion, _HostileScore(), None, z0=0.0, n_steps=n, n_prime=n, dt=dt,
                             brownian=W, max_abs_control=control, link=ExponentialLink(), initial_raw=0.0)
    assert bool(torch.isfinite(sim.raw).all()) == finite
    if finite:
        steps = (sim.raw[:, 1:] - sim.raw[:, :-1]).abs()
        assert torch.all(steps <= step_std * (control * math.sqrt(dt) + W.abs() / math.sqrt(dt)) + 1e-4)
        assert float(sim.u.abs().max()) <= control and float(sim.sigma.max()) <= step_std / math.sqrt(dt) + 1e-6


@pytest.mark.parametrize('name, bound', [('base', 30.0), ('smoke', 30.0), ('state_space/cir_burn_in', 30.0), ('observations', 30.0),
                                         ('intensity/cir_mlp', None)])
def test_the_state_space_configs_bound_the_control(name, bound):
    """Every config of the state-space parametrization carries both bounds (they come from ``configs/base.yaml``); the
    intensity parametrization of example 1 switches the bound on the control off and has no bound on its diffusion."""
    cfg = load_config(name, overrides=['data.path=observations.npz'] if name == 'observations' else [])
    assert cfg.train.max_abs_control == bound
    assert cfg.model.diffusion.max_step_std == 1.0


def test_cir_diffusion_objective_gradients(make_cfg):
    cfg = make_cfg('model.diffusion.kind=cir_exponential', 'model.diffusion.trainable=true',
                   'model.posterior.kind=remaining_count')
    torch.manual_seed(40)
    model = NDIM(cfg).double()
    events = torch.tensor([[.25, .5], [.35, float('inf')]], dtype=torch.float64)
    W = torch.randn(4, 5, dtype=torch.float64) * .1
    obj = compute_objective(model, events, 5, 2, W, cfg)
    obj.loss.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
    assert expected['diffusion.log_eta'].abs() > 1e-8
    model.zero_grad(set_to_none=True)
    sensitivity_backward(model, events, 5, 2, W, cfg)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad, expected[name], atol=1e-9, rtol=1e-7), name
    # Independent finite difference of the full objective.
    eta_raw = model.diffusion.log_eta.detach().clone()
    losses = []
    with torch.no_grad():
        for delta in [-1e-4, 1e-4]:
            model.diffusion.log_eta.copy_(eta_raw + delta)
            losses.append(float(compute_objective(model, events, 5, 2, W, cfg).loss))
        model.diffusion.log_eta.copy_(eta_raw)
    finite_diff = (losses[1] - losses[0]) / 2e-4
    assert finite_diff == pytest.approx(float(expected['diffusion.log_eta']), rel=1e-6, abs=1e-8)


@pytest.mark.parametrize('link', ['exponential', 'exp'])
def test_cir_exponential_checkpoint_roundtrip(make_cfg, tmp_path, link):
    cfg = make_cfg('model.diffusion.kind=cir_exponential', 'model.diffusion.trainable=true', f'model.link={link}')
    model = NDIM(cfg)
    with torch.no_grad():
        model.diffusion.log_eta.add_(0.4)
    path = tmp_path / 'linked_cir.pt'
    model.save_checkpoint(path)
    loaded, _ = NDIM.load_checkpoint(path)
    assert loaded.cfg.model.diffusion.kind == 'cir_exponential'
    z = torch.tensor([0.01, 5., 60.])
    t = torch.zeros_like(z)
    assert torch.equal(loaded.intensity_diffusion(z, t), model.intensity_diffusion(z, t))


@pytest.mark.parametrize('override', ['model.link=softplus', 'model.state_space=intensity'])
def test_cir_exponential_rejects_wrong_coordinates(make_cfg, override):
    with pytest.raises(ValueError, match='real states and an exponential link'):
        make_cfg('model.diffusion.kind=cir_exponential', override)


@pytest.mark.parametrize('overrides, form', [
    ((), 'CIRExponentialDiffusion'),
    (('model.link=softplus',), 'CIRLinkDiffusion'),
    (('model.state_space=intensity',), 'CIRSqrtDiffusion'),
])
def test_cir_diffusion_follows_the_parametrization(make_cfg, overrides, form):
    """``cir`` is ``eta * sqrt(Z)`` in intensity terms whichever state space and link are chosen."""
    cfg = make_cfg('model.diffusion.kind=cir', 'model.diffusion.eta_init=0.7', *overrides)
    model = NDIM(cfg).double()
    assert type(model.diffusion).__name__ == form
    eta = model.diffusion.eta.detach()                      # 0.7 up to the float32 rounding of its initial value
    assert float(eta) == pytest.approx(0.7, rel=1e-6)
    z = torch.tensor([0.5, 5., 25., 80., 150.], dtype=torch.float64)
    assert torch.allclose(model.intensity_diffusion(z, torch.zeros_like(z)), eta * z.sqrt(), rtol=1e-12, atol=0)


def test_bare_cir_kind_needs_the_state_space():
    from ndim.config import DiffusionConfig
    from ndim.models.diffusion import build_diffusion
    with pytest.raises(ValueError, match='follows the state space'):
        build_diffusion(DiffusionConfig(kind='cir'))


@pytest.mark.parametrize('name, form', [
    ('intensity/cir_mlp', 'CIRSqrtDiffusion'),              # intensity parametrization: the state is the intensity
    ('state_space/cir_burn_in', 'CIRExponentialDiffusion'), # state-space parametrization: the state is X = log Z
])
def test_both_parts_use_the_same_fixed_intensity_diffusion(name, form):
    """The two examples use the same diffusion in different coordinates: the intensity diffusion is the known ``sqrt(Z)`` in both,
    and it has no trainable parameter. For the latent state that is ``exp(-X/2)``, not a constant."""
    model = NDIM(load_config(name, profile='laptop')).double()
    assert type(model.diffusion).__name__ == form
    assert not any(p.requires_grad for p in model.diffusion.parameters())
    z = torch.tensor([0.5, 5., 25., 80., 150.], dtype=torch.float64)
    t = torch.zeros_like(z)
    assert torch.allclose(model.intensity_diffusion(z, t), z.sqrt(), rtol=1e-12, atol=0)
    if model.real_states:
        x = z.log()
        assert torch.allclose(model.diffusion(x, t), torch.exp(-x / 2), rtol=1e-12, atol=0)
