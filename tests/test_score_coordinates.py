"""Coordinate-consistent posterior scores under a real-state intensity link."""
import copy

import pytest
import torch

from ndim.model import NDIM
from ndim.elbo import compute_objective
from ndim.gradients import sensitivity_backward


@pytest.mark.parametrize('link', ['exponential', 'softplus'])
@pytest.mark.parametrize('kind,state_in_phi', [('remaining_count', False), ('deepset', False), ('deepset', True)])
@pytest.mark.parametrize('jump', ['none', 'raw', 'baseline_removed'])
def test_intensity_encoder_chain_rule(make_cfg, link, kind, state_in_phi, jump):
    cfg = make_cfg(f'model.link={link}', f'model.posterior.kind={kind}',
                   'model.posterior.score_space=intensity', 'model.posterior.z_scale=80',
                   f'model.posterior.phi_uses_state={str(state_in_phi).lower()}',
                   f'model.posterior.jump_term={jump}')
    real = NDIM(cfg).double()
    intensity_cfg = copy.deepcopy(cfg)
    intensity_cfg.model.state_space = 'intensity'
    intensity_cfg.model.diffusion.kind = 'cir_sqrt'
    intensity_model = NDIM(intensity_cfg).double()
    intensity_model.score_net.load_state_dict(real.score_net.state_dict())
    events = torch.tensor([[.1, .4, 1.], [.2, .6, float('inf')]], dtype=torch.float64)
    times = torch.tensor([0., .3, .8], dtype=torch.float64)
    cx = real.score_net.prepare(events, times, 1.)
    cz = intensity_model.score_net.prepare(events, times, 1.)
    z = torch.tensor([5., 10., 30., 70.], dtype=torch.float64)
    x = real.link.inverse(z)
    for k in range(len(times)):
        sx = real.score_net.score(cx, k, x)
        sz = intensity_model.score_net.score(cz, k, z)
        assert torch.allclose(sx, real.link.first(x) * sz, atol=1e-10, rtol=1e-10)


def test_cir_control_and_kl_are_coordinate_invariant(make_cfg):
    cfg = make_cfg('model.diffusion.kind=cir_exponential', 'model.diffusion.eta_init=1.2',
                   'model.posterior.score_space=intensity', 'model.posterior.z_scale=80')
    real = NDIM(cfg).double()
    intensity_cfg = copy.deepcopy(cfg)
    intensity_cfg.model.state_space = 'intensity'
    intensity_cfg.model.diffusion.kind = 'cir_sqrt'
    intensity_model = NDIM(intensity_cfg).double()
    intensity_model.score_net.load_state_dict(real.score_net.state_dict())
    z = torch.tensor([5., 10., 30., 70.], dtype=torch.float64)
    x, t = z.log(), torch.zeros_like(z)
    events = torch.tensor([[.1, .4], [.2, .6]], dtype=torch.float64)
    sx = real.score_net.score(real.score_net.prepare(events, t[:1], 1.), 0, x)
    sz = intensity_model.score_net.score(intensity_model.score_net.prepare(events, t[:1], 1.), 0, z)
    ux, uz = real.diffusion(x,t)*sx, intensity_model.diffusion(z,t)*sz
    assert torch.allclose(ux, uz, atol=1e-12)
    assert torch.allclose(ux.square(), uz.square(), atol=1e-12)
    assert torch.allclose(z*real.diffusion(x,t)*ux, intensity_model.diffusion(z,t)*uz, atol=1e-12)


def test_intensity_score_sensitivity_matches_backprop(make_cfg):
    cfg = make_cfg('model.diffusion.kind=cir_exponential', 'model.diffusion.trainable=true',
                   'model.posterior.score_space=intensity', 'model.posterior.z_scale=80',
                   'model.posterior.phi_uses_state=true')
    torch.manual_seed(7)
    model = NDIM(cfg).double()
    events = torch.tensor([[.25, .5]], dtype=torch.float64)
    w = torch.randn(2,5,dtype=torch.float64)*.1
    compute_objective(model, events, 5, 2, w, cfg).loss.backward()
    grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    sensitivity_backward(model, events, 5, 2, w, cfg)
    for n,p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.grad,grads[n],atol=1e-9,rtol=1e-7), n


def test_score_space_checkpoint_default_and_reload(make_cfg, tmp_path):
    cfg = make_cfg('model.posterior.score_space=intensity', 'model.posterior.z_scale=80')
    model = NDIM(cfg)
    p = tmp_path/'score.pt'
    model.save_checkpoint(p)
    loaded,_ = NDIM.load_checkpoint(p)
    assert loaded.score_net.score_space == 'intensity'
    payload = torch.load(p,weights_only=False)
    del payload['config']['model']['posterior']['score_space']
    torch.save(payload,p)
    loaded,_ = NDIM.load_checkpoint(p)
    assert loaded.score_net.score_space == 'state'


def test_invalid_score_space(make_cfg):
    with pytest.raises(ValueError,match='score_space'):
        make_cfg('model.posterior.score_space=invalid')
