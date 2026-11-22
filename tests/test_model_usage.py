"""Using a trained model: ``NDIM.times``, ``prior_paths``, ``posterior_paths`` and ``generate_observations`` against the
simulators they wrap (the README section "Using a trained model")."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ndim.data import thin_events
from ndim.model import NDIM, build_model
from ndim.sde import brownian_increments

CPU = torch.device("cpu")
BURN = ["model.burn_in=0.96", "model.x_start=1.6", "model.drift.time_input=true"]       # 6 steps of the smoke grid (dt = 0.16)
EVENTS = [0.3, 0.9, 1.4, 2.2, 2.25, 3.1, 3.7]


def _model(make_cfg, *overrides: str) -> NDIM:
    torch.manual_seed(0)
    return build_model(make_cfg(*overrides))


@pytest.mark.parametrize("overrides", [[], BURN])
def test_paths_of_the_prior_and_of_the_posterior(make_cfg, overrides) -> None:
    model = _model(make_cfg, *overrides)
    cfg = model.cfg
    N, dt, T, nb = cfg.data.n_steps, cfg.dt, cfg.data.T, model.n_burn
    assert nb == (6 if overrides else 0)
    t = model.times
    assert t.shape == (N + 1,) and float(t[0]) == 0.0 and float(t[-1]) == pytest.approx(T) and float(t[1]) == pytest.approx(dt)

    # the prior: the low-level simulation with the same noise, on the observation window
    prior = model.prior_paths(7, seed=3)
    W = brownian_increments(7, nb + N, dt, CPU, torch.Generator().manual_seed(3))
    assert prior.shape == (7, N + 1) and torch.equal(prior, model.simulate_prior(W))
    assert torch.equal(model.prior_paths(7, seed=3), prior) and not torch.equal(model.prior_paths(7, seed=4), prior)
    assert bool((prior > 0).all()) and model.prior_paths().shape == (100, N + 1) and not prior.requires_grad

    # the posterior given one observation: the low-level simulation with the bounds of the config
    events = torch.tensor([EVENTS])
    post = model.posterior_paths(EVENTS, 5, seed=1)
    W = brownian_increments(5, nb + N, dt, CPU, torch.Generator().manual_seed(1))
    sim = model.simulate_posterior(events, N, W, N, max_abs_score=cfg.train.max_abs_score, max_abs_control=cfg.train.max_abs_control)
    assert post.shape == (5, N + 1) and torch.equal(post, sim.Z[:, nb:]) and not post.requires_grad
    assert torch.equal(model.posterior_paths(np.array(EVENTS[::-1]), 5, seed=1), post)           # any order, any array type
    assert torch.equal(model.posterior_paths(torch.tensor(EVENTS + [float("inf")]), 5, seed=1), post)   # padding is ignored

    # observed on [0, T/2] only: the later events are not used, and the path after T/2 follows the prior dynamics
    half = model.posterior_paths(EVENTS, 5, observed_until=T / 2, seed=1)
    sim = model.simulate_posterior(events, N // 2, W, N, max_abs_score=cfg.train.max_abs_score, max_abs_control=cfg.train.max_abs_control)
    assert half.shape == (5, N + 1) and torch.equal(half, sim.Z[:, nb:])
    early = [e for e in EVENTS if e <= (N // 2) * dt]
    assert 0 < len(early) < len(EVENTS) and torch.equal(model.posterior_paths(early, 5, observed_until=T / 2, seed=1), half)
    assert not torch.equal(half, post)
    # no event at all is an observation too
    assert model.posterior_paths([], 3, seed=0).shape == (3, N + 1)


def test_the_bound_on_the_control_is_the_one_of_the_config(make_cfg) -> None:
    """``posterior_paths`` simulates the posterior the model was trained as: with a tight bound the control is cut."""
    free = _model(make_cfg, "train.max_abs_control=null")
    tight = _model(make_cfg, "train.max_abs_control=1.0e-6")
    tight.load_state_dict(free.state_dict())
    a, b = free.posterior_paths(EVENTS, 4, seed=2), tight.posterior_paths(EVENTS, 4, seed=2)
    assert not torch.equal(a, b)
    assert torch.allclose(b, tight.prior_paths(4, seed=2), rtol=1e-3, atol=1e-3)                  # no control left: the prior paths


def test_new_observations(make_cfg) -> None:
    model = _model(make_cfg, *BURN)
    T = model.cfg.data.T
    new = model.generate_observations(4, seed=5)
    assert len(new) == 4 and all(isinstance(e, np.ndarray) for e in new)
    assert all(np.all(np.diff(e) >= 0) and (e.size == 0 or (e.min() >= 0.0 and e.max() <= T)) for e in new)
    again = model.generate_observations(4, seed=5)
    assert all(np.array_equal(a, b) for a, b in zip(new, again))
    # the same draws as the prior paths followed by the thinning
    g = torch.Generator().manual_seed(5)
    paths = model.simulate_prior(brownian_increments(4, model.n_burn + model.cfg.data.n_steps, model.cfg.dt, CPU, g))
    assert all(np.array_equal(a, b) for a, b in zip(new, thin_events(paths, T, generator=g)))
    assert len(model.generate_observations()) == 1


def test_inputs_that_cannot_be_used_are_refused(make_cfg) -> None:
    model = _model(make_cfg)
    T = model.cfg.data.T
    for bad in ([0.5, T + 0.1], [0.0, 1.0], [-0.2]):
        with pytest.raises(ValueError, match="event times must lie in"):
            model.posterior_paths(bad, 2)
    for bad in (0.0, -1.0, T + 0.5, 1e-6):
        with pytest.raises(ValueError, match="observed_until must lie in"):
            model.posterior_paths(EVENTS, 2, observed_until=bad)
    with pytest.raises(ValueError, match="at least 1"):
        model.prior_paths(0)
