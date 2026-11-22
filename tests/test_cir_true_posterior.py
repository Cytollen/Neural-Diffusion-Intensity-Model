"""Exact CIR posterior path sampling checked against analytic path moments."""
import math
from types import SimpleNamespace

import pytest
import torch

from ndim.cir_posterior import (
    build_cir_future_likelihood, cir_affine_step, sample_true_cir_posterior,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _cfg(n_steps=4):
    truth = SimpleNamespace(kappa=.3, theta=80., eta=1., z0=5.)
    return SimpleNamespace(data=SimpleNamespace(T=4., n_steps=n_steps, truth=truth))


def _moment(events, insert_times, horizon=4.):
    base = build_cir_future_likelihood([events], [0.], T=horizon).log_marginal()
    inserted = build_cir_future_likelihood([list(events) + list(insert_times)], [0.], T=horizon).log_marginal()
    return float((inserted - base).exp())


def test_no_event_posterior_terminal_transition_matches_killed_affine_law():
    count = 40000
    cfg = _cfg()
    paths, metadata = sample_true_cir_posterior(torch.empty((1, 0)), cfg, 4, count,
                                               generator=torch.Generator().manual_seed(901))
    _, _, log_D, log_U = cir_affine_step(0., 4., 1.)
    q, D, r = .5 * log_U.exp(), log_D.exp(), 48.
    mean = float(r * q + D * 5.)
    variance = float(r * q.square() + 2 * q * D * 5.)
    actual = paths[:, -1]
    assert abs(float(actual.mean()) - mean) < 6 * math.sqrt(variance / count)
    assert float(actual.var()) == pytest.approx(variance, rel=.04)
    assert metadata["exact_finite_dimensional"] is True
    assert metadata["independent_paths"] is True
    assert metadata["method"] == "exact_affine_polynomial"
    assert paths.dtype == torch.float64 and tuple(paths.shape) == (count, 5)


def test_event_tilted_path_moments_and_cross_time_covariance_are_exact():
    events = [.6, 2., 3.1]
    count = 50000
    paths, _ = sample_true_cir_posterior(torch.tensor([events], dtype=torch.float64), _cfg(), 4, count,
                                        generator=torch.Generator().manual_seed(163))
    means = {}
    for time in (1, 2, 4):
        mean, second = _moment(events, [time]), _moment(events, [time, time])
        variance = second - mean * mean
        means[time] = mean
        assert abs(float(paths[:, time].mean()) - mean) < 6 * math.sqrt(variance / count)
        assert float(paths[:, time].var()) == pytest.approx(variance, rel=.05)
    expected_cross = _moment(events, [1., 4.]) - means[1] * means[4]
    products = (paths[:, 1] - means[1]) * (paths[:, 4] - means[4])
    cross_se = float(products.std()) / math.sqrt(count)
    assert abs(float(products.mean()) - expected_cross) < 6 * cross_se
    # There is no Markov-chain correlation across the independently drawn paths.
    correlation = torch.corrcoef(torch.stack((paths[:-1, -1], paths[1:, -1])))[0, 1]
    assert abs(float(correlation)) < .04


def test_partial_horizon_includes_endpoint_event_then_uses_exact_prior():
    count = 40000
    paths, metadata = sample_true_cir_posterior(torch.tensor([[2., 3.]], dtype=torch.float64),
                                               _cfg(), 2, count, generator=torch.Generator().manual_seed(73))
    mean2 = _moment([2.], [2.], horizon=2.)
    var2 = _moment([2.], [2., 2.], horizon=2.) - mean2 * mean2
    assert abs(float(paths[:, 2].mean()) - mean2) < 6 * math.sqrt(var2 / count)
    prior_expected = 80. + (mean2 - 80.) * math.exp(-.3 * 2)
    assert abs(float(paths[:, -1].mean()) - prior_expected) < 6 * float(paths[:, -1].std()) / math.sqrt(count)
    assert metadata["conditioning_horizon"] == 2.
    assert metadata["events_conditioned"] == [1]


def test_local_generator_is_reproducible_and_future_events_have_no_influence():
    cfg = _cfg(10)
    state = torch.random.get_rng_state().clone()
    a, _ = sample_true_cir_posterior(torch.tensor([[.2, 3.9]], dtype=torch.float64), cfg, 5, 8,
                                    generator=torch.Generator().manual_seed(123))
    b, _ = sample_true_cir_posterior(torch.tensor([[.2, 2.3]], dtype=torch.float64), cfg, 5, 8,
                                    generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(a, b, atol=0., rtol=0.)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_zero_horizon_is_unconditional_exact_cir_prior_and_batch_order_is_preserved():
    count = 20000
    cfg = _cfg(2)
    paths, metadata = sample_true_cir_posterior(torch.tensor([[.1], [3.]]), cfg, 0, count,
                                               generator=torch.Generator().manual_seed(553))
    mean = 80 + (5 - 80) * math.exp(-.3 * 4)
    for batch in range(2):
        terminal = paths[batch * count:(batch + 1) * count, -1]
        assert abs(float(terminal.mean()) - mean) < 6 * float(terminal.std()) / math.sqrt(count)
    assert metadata["events_conditioned"] == [0, 0]
    assert torch.all(paths[:, 0] == 5.)


def test_duplicate_events_at_grid_time_use_the_full_event_multiplicity():
    count = 25000
    paths, _ = sample_true_cir_posterior(torch.tensor([[2., 2.]], dtype=torch.float64), _cfg(), 2, count,
                                        generator=torch.Generator().manual_seed(830))
    mean = _moment([2., 2.], [2.], horizon=2.)
    assert abs(float(paths[:, 2].mean()) - mean) < 6 * float(paths[:, 2].std()) / math.sqrt(count)
