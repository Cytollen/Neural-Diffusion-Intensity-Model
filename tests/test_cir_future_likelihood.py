"""Independent checks of the exact CIR/Cox polynomial recursion."""
import math

import pytest
import torch

from ndim.cir_posterior import (
    build_cir_future_likelihood, cir_affine_step, propagate_log_polynomial,
)


DTYPE = torch.float64


def _bond(tau, eta, kappa=.3, theta=80.):
    gamma = math.sqrt(kappa * kappa + 2 * eta * eta)
    denominator = (gamma + kappa) * math.expm1(gamma * tau) + 2 * gamma
    B = 2 * math.expm1(gamma * tau) / denominator
    log_A = (2 * kappa * theta / eta ** 2) * (
        math.log(2 * gamma) + (kappa + gamma) * tau / 2 - math.log(denominator))
    return B, log_A


def test_no_events_matches_cir_bond_formula_and_terminal_score():
    times = torch.tensor([0., .1, 1., 3.99, 4.], dtype=DTYPE)
    result = build_cir_future_likelihood([[], []], times, .8)
    z = torch.tensor([5., 80.], dtype=DTYPE)
    for index, time in enumerate(times.tolist()):
        B, log_A = _bond(4 - time, .8)
        torch.testing.assert_close(result.log_future_likelihood(index, z), log_A - B * z,
                                   rtol=2e-13, atol=3e-13)
        torch.testing.assert_close(result.score(index, z), torch.full_like(z, -B),
                                   rtol=2e-13, atol=3e-13)


def test_one_future_event_matches_affine_polynomial_formula_and_boundary_limit():
    eta, tau, time = 1.1, 2.7, .4
    result = build_cir_future_likelihood([[tau]], [time, tau, 3.], eta)
    tail_B, tail_log_A = _bond(4 - tau, eta)
    B, log_A, log_D, log_U = cir_affine_step(tail_B, tau - time, eta)
    z = torch.tensor([[1e-9], [5.], [80.]], dtype=DTYPE)
    D, U = log_D.exp(), log_U.exp()
    expected_score = D / (.3 * 80 * U + D * z) - B
    expected_loglik = tail_log_A + log_A - B * z + torch.log(.3 * 80 * U + D * z)
    torch.testing.assert_close(result.score(0, z), expected_score, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(result.log_future_likelihood(0, z), expected_loglik,
                               rtol=2e-13, atol=2e-13)
    # Immigration gives a positive constant polynomial term before the event:
    # the K/z asymptotic does not hold at strictly earlier times.
    assert abs(float(result.score(0, z)[0])) < 2
    B_after, _ = _bond(4 - tau, eta)
    torch.testing.assert_close(result.score(1, z), torch.full_like(z, -B_after))


def test_segment_propagation_matches_independent_triangular_ode():
    scipy_integrate = pytest.importorskip("scipy.integrate")
    import numpy as np
    initial = np.array([.2, .7, .1, .5, .04])
    eta, b, duration = 1.2, .7, .31
    def ode(_, state):
        B, A = state[:2]
        coeff = state[2:]
        derivative = -np.arange(len(coeff)) * (.3 + eta ** 2 * B) * coeff
        derivative[:-1] += (np.arange(1, len(coeff)) *
                            (.3 * 80 + .5 * eta ** 2 * np.arange(len(coeff) - 1)) * coeff[1:])
        return np.r_[1 - .3 * B - .5 * eta ** 2 * B * B, -.3 * 80 * B, derivative]
    solved = scipy_integrate.solve_ivp(ode, [0, duration], np.r_[b, 0., initial],
                                       rtol=2e-12, atol=1e-13).y[:, -1]
    coeff, B, scale = propagate_log_polynomial(torch.tensor(initial, dtype=DTYPE).log(),
                                              b, duration, eta, z_scale=1.)
    torch.testing.assert_close(B, torch.tensor(solved[0], dtype=DTYPE), rtol=3e-11, atol=1e-11)
    actual = (coeff + scale).exp()
    expected = torch.tensor(solved[2:] * math.exp(solved[1]), dtype=DTYPE)
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=1e-11)


def test_recursion_is_semigroup_consistent_and_score_is_state_gradient():
    events = [[.2, .3, 1.7, 3.8], [.05, 1.4, 2.9]]
    coarse = build_cir_future_likelihood(events, [0., 2.], .9)
    refined = build_cir_future_likelihood(events, [0., .11, .41, 1., 2., 2.8, 3.9], .9)
    z = torch.tensor([5., 70.], dtype=DTYPE, requires_grad=True)
    torch.testing.assert_close(coarse.log_future_likelihood(0, z),
                               refined.log_future_likelihood(0, z), rtol=2e-13, atol=3e-13)
    torch.testing.assert_close(coarse.score(0, z), refined.score(0, z), rtol=2e-13, atol=3e-13)
    derivative = torch.autograd.grad(coarse.log_future_likelihood(0, z).sum(), z)[0]
    torch.testing.assert_close(coarse.score(0, z), derivative, rtol=2e-13, atol=3e-13)
    torch.testing.assert_close(coarse.logcoordinate_score(0, z), coarse.score(0, z) * z,
                               rtol=2e-13, atol=3e-13)


def test_polynomial_basis_rescaling_preserves_likelihood_and_score():
    events = [[.2, .3, 1.7, 3.8], [.05, 1.4, 2.9]]
    reference = build_cir_future_likelihood(events, [0., 1.], .9, z_scale=1.)
    rescaled = build_cir_future_likelihood(events, [0., 1.], .9, z_scale=500.)
    z = torch.tensor([5., 70.], dtype=DTYPE)
    torch.testing.assert_close(reference.log_future_likelihood(0, z),
                               rescaled.log_future_likelihood(0, z), rtol=2e-13, atol=3e-13)
    torch.testing.assert_close(reference.score(0, z), rescaled.score(0, z),
                               rtol=2e-13, atol=3e-13)


def test_terminal_event_is_included_before_T_and_excluded_at_postevent_T():
    eta, T = .8, 4.
    result = build_cir_future_likelihood([[T]], [0., T], eta, T=T)
    B, log_A, log_D, log_U = cir_affine_step(0., T, eta)
    z = torch.tensor([5.], dtype=DTYPE)
    expected = log_A - B * z + torch.log(.3 * 80 * log_U.exp() + log_D.exp() * z)
    torch.testing.assert_close(result.log_future_likelihood(0, z), expected)
    torch.testing.assert_close(result.log_future_likelihood(1, z), torch.zeros_like(z))


def test_eta_gradient_of_exact_likelihood_matches_finite_difference():
    events = [[.2, .3, 1.7, 3.8], [.05, 1.4, 2.9]]
    log_eta = torch.tensor(math.log(.85), dtype=DTYPE, requires_grad=True)
    value = build_cir_future_likelihood(events, [0.], log_eta.exp()).log_marginal().sum()
    gradient = torch.autograd.grad(value, log_eta)[0]
    epsilon = 1e-5
    plus = build_cir_future_likelihood(events, [0.], (log_eta.detach() + epsilon).exp()).log_marginal().sum()
    minus = build_cir_future_likelihood(events, [0.], (log_eta.detach() - epsilon).exp()).log_marginal().sum()
    torch.testing.assert_close(gradient, (plus - minus) / (2 * epsilon), rtol=2e-8, atol=2e-8)


def test_hundreds_of_events_stay_finite_with_differentiable_state_score():
    torch.set_num_threads(1)
    events = [torch.linspace(.01, 3.99, 220, dtype=DTYPE)]
    with torch.no_grad():
        result = build_cir_future_likelihood(events, [0., 1., 2., 3., 4.])
    z = torch.tensor([[5.], [80.], [200.]], dtype=DTYPE, requires_grad=True)
    values = result.log_future_likelihood(0, z)
    scores = result.score(0, z)
    assert torch.isfinite(values).all()
    assert torch.isfinite(scores).all()
    assert torch.isfinite(torch.autograd.grad(scores.sum(), z)[0]).all()


@pytest.mark.parametrize("events,times", [([[]], []), ([[0.]], [0.]), ([[4.1]], [0.]), ([[]], [1., 0.])])
def test_invalid_event_or_query_times_are_rejected(events, times):
    with pytest.raises(ValueError):
        build_cir_future_likelihood(events, times)
