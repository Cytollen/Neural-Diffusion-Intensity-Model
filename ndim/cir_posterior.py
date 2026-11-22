"""Exact continuous-time CIR/Cox future likelihood and posterior score.

For event times ``tau > t``, ``F(t,z)`` is the expectation of the future
Poisson likelihood under ``dZ=kappa*(theta-Z)dt+eta*sqrt(Z)dW``.  Its form is
``exp(log_normalizer-B*z) * sum_j exp(log_coefficients[j])*(z/z_scale)**j``.
The positive triangular polynomial recursion is propagated exactly between
events, with a log-domain representation to support hundreds of events.

The posterior physical drift correction is
``eta**2*z*score``; freezing a score alone while changing eta does NOT freeze
that physical correction.  A fixed-reference correction must retain its
reference eta in that multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def _scalar64(value, *, device=None):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def cir_affine_step(B, duration, eta, *, kappa=.3, theta=80.):
    """Exact Riccati flow and polynomial time change for one backward step.

    Returns ``B_new, log_affine_multiplier, log_D, log_U``.  The zero-duration
    step has ``U=0`` and is handled explicitly by the polynomial propagator.
    ``eta`` can be a differentiable scalar tensor.
    """
    eta = _scalar64(eta)
    B = _scalar64(B, device=eta.device)
    duration = _scalar64(duration, device=eta.device)
    if eta.numel() != 1 or bool(eta.detach() <= 0):
        raise ValueError("eta must be a positive scalar")
    if bool(duration.detach() < 0):
        raise ValueError("duration must be nonnegative")
    c = eta.square() / 2
    gamma = (kappa * kappa + 4 * c).sqrt()
    delta = -torch.expm1(-gamma * duration)
    denominator_minus_one = (kappa + 2 * c * B - gamma) * delta / (2 * gamma)
    log_denominator = torch.log1p(denominator_minus_one)
    denominator = 1 + denominator_minus_one
    B_new = (B * (1 - (kappa + gamma) * delta / (2 * gamma)) + delta / gamma) / denominator
    # (gamma-kappa)/(2*c) = 2/(gamma+kappa) avoids subtractive cancellation.
    integral_B = 2 * duration / (gamma + kappa) + log_denominator / c
    log_affine = -kappa * theta * integral_B
    log_D = -gamma * duration - 2 * log_denominator
    log_U = torch.log(delta) - torch.log(gamma) - log_denominator
    return B_new, log_affine, log_D, log_U


def _log_triangular_weights(degree, eta, kappa, theta, device):
    """Eta-dependent, duration-independent coefficients for exp(U*L)."""
    order = torch.arange(degree + 1, dtype=torch.float64, device=device)
    j, n = order[:, None], order[None, :]
    difference = n - j
    r = 2 * kappa * theta / eta.square()
    # n choose j times Gamma(r+n)/Gamma(r+j); masked invalid entries
    # never participate in the sum, and their lgamma arguments stay positive.
    weights = (torch.lgamma(n + 1) - torch.lgamma(j + 1)
               - torch.lgamma(difference.clamp_min(0) + 1)
               + torch.lgamma(r + n) - torch.lgamma(r + j))
    return weights.masked_fill(difference < 0, -torch.inf), difference.clamp_min(0)


def propagate_log_polynomial(log_coefficients, B, duration, eta, *, kappa=.3,
                             theta=80., z_scale=80., triangular_weights=None):
    """Propagate a normalized polynomial exactly over an event-free interval.

    Returns ``(new_log_coefficients, new_B, log_scale_increment)``.  The
    logarithmic scale includes the affine killing factor.  No coefficient
    truncation or numerical ODE integration is used.
    """
    eta = _scalar64(eta, device=log_coefficients.device)
    if float(torch.as_tensor(duration).detach()) == 0:
        return log_coefficients, _scalar64(B, device=eta.device), eta * 0
    new_B, log_affine, log_D, log_U = cir_affine_step(
        B, duration, eta, kappa=kappa, theta=theta)
    degree = log_coefficients.numel() - 1
    if triangular_weights is None:
        weights, difference = _log_triangular_weights(degree, eta, kappa, theta, eta.device)
    else:
        weights, difference = (x[:degree + 1, :degree + 1] for x in triangular_weights)
    log_cU = torch.log(eta.square() / (2 * z_scale)) + log_U
    result = torch.logsumexp(log_coefficients[None, :] + weights + difference * log_cU,
                            dim=-1)
    result = result + torch.arange(degree + 1, dtype=torch.float64,
                                   device=eta.device) * log_D
    offset = result.max()
    return result - offset, new_B, log_affine + offset


@dataclass
class CIRFutureLikelihood:
    """Analytic future likelihood at fixed query times and event sequences.

    Tensor shapes are ``[time, batch, degree]`` for polynomial coefficients
    and ``[time,batch]`` for exponential coefficients and normalizers.  All
    score methods accept ``z`` with batch as its final dimension, optionally
    preceded by Monte Carlo path dimensions.
    """
    times: torch.Tensor
    log_coefficients: torch.Tensor
    B: torch.Tensor
    log_normalizer: torch.Tensor
    eta: torch.Tensor
    z_scale: float
    events: tuple
    T: float
    kappa: float
    theta: float

    @property
    def batch_size(self):
        return self.log_coefficients.shape[1]

    def _terms(self, index, z):
        z = _scalar64(z, device=self.log_coefficients.device)
        if bool((z.detach() <= 0).any()):
            raise ValueError("CIR likelihood/score evaluation requires z>0")
        order = torch.arange(self.log_coefficients.shape[-1], dtype=z.dtype,
                             device=z.device)
        terms = self.log_coefficients[index] + torch.log(z[..., None] / self.z_scale) * order
        return z, order, terms

    def score(self, index, z):
        """Return d/dz log F(t,z) with state derivatives retained."""
        z, order, terms = self._terms(index, z)
        mean_degree = (torch.softmax(terms, dim=-1) * order).sum(dim=-1)
        return mean_degree / z - self.B[index]

    def logcoordinate_score(self, index, z):
        """Return d/dlog(z) log F, avoiding cancellation through 1/z."""
        z, order, terms = self._terms(index, z)
        mean_degree = (torch.softmax(terms, dim=-1) * order).sum(dim=-1)
        return mean_degree - self.B[index] * z

    def log_future_likelihood(self, index, z):
        z, _, terms = self._terms(index, z)
        return self.log_normalizer[index] - self.B[index] * z + torch.logsumexp(terms, dim=-1)

    def log_marginal(self, z0=5.):
        if float(self.times[0]) != 0:
            raise ValueError("log_marginal requires the first query time to be zero")
        return self.log_future_likelihood(0, z0)


def build_cir_future_likelihood(events: Sequence, times, eta=1., *, T=4.,
                                kappa=.3, theta=80., z_scale=80.):
    """Construct exact polynomial coefficients for each event sequence.

    ``events`` is a sequence of unpadded one-dimensional event tensors.  All
    events must be in ``(0,T]``.  At a query exactly equal to an event time,
    the event is excluded (the right-continuous, post-event score).  Use
    ``torch.no_grad()`` when constructing a fixed-reference posterior; eta
    derivatives through hundreds of event recursions otherwise retain a
    large autograd graph.
    """
    eta = _scalar64(eta)
    times = _scalar64(times, device=eta.device)
    if times.ndim != 1 or times.numel() == 0 or bool((times[1:] < times[:-1]).any()):
        raise ValueError("times must be a nonempty sorted one-dimensional grid")
    if bool((times < 0).any()) or bool((times > T).any()):
        raise ValueError("query times must lie in [0,T]")
    if kappa <= 0 or theta <= 0 or z_scale <= 0 or T <= 0:
        raise ValueError("kappa, theta, z_scale, T must be positive")
    event_arrays = []
    for sequence in events:
        array = _scalar64(sequence, device=eta.device).detach().flatten().sort().values
        if bool(((array <= 0) | (array > T)).any()):
            raise ValueError("event times must lie in (0,T]")
        event_arrays.append(array)
    if not event_arrays:
        raise ValueError("at least one event sequence is required")
    max_degree = max(x.numel() for x in event_arrays)
    triangular = _log_triangular_weights(max_degree, eta, kappa, theta, eta.device)
    all_coefficients, all_B, all_normalizers = [], [], []
    for sequence in event_arrays:
        coefficients = torch.zeros(1, dtype=torch.float64, device=eta.device)
        B, normalizer = eta * 0, eta * 0
        cursor, event_index = float(T), sequence.numel() - 1
        coefficients_at_time, B_at_time, normalizers_at_time = [], [], []
        for query in reversed(times.detach().tolist()):
            while event_index >= 0 and float(sequence[event_index]) > query:
                event_time = float(sequence[event_index])
                coefficients, B, extra = propagate_log_polynomial(
                    coefficients, B, cursor - event_time, eta, kappa=kappa,
                    theta=theta, z_scale=z_scale, triangular_weights=triangular)
                normalizer = normalizer + extra
                coefficients = torch.cat((coefficients.new_full((1,), -torch.inf),
                                          coefficients + math.log(z_scale)))
                cursor, event_index = event_time, event_index - 1
            coefficients, B, extra = propagate_log_polynomial(
                coefficients, B, cursor - query, eta, kappa=kappa, theta=theta,
                z_scale=z_scale, triangular_weights=triangular)
            normalizer = normalizer + extra
            padded = torch.nn.functional.pad(coefficients, (0, max_degree + 1 - coefficients.numel()),
                                             value=-torch.inf)
            coefficients_at_time.append(padded)
            B_at_time.append(B)
            normalizers_at_time.append(normalizer)
            cursor = query
        all_coefficients.append(torch.stack(coefficients_at_time[::-1]))
        all_B.append(torch.stack(B_at_time[::-1]))
        all_normalizers.append(torch.stack(normalizers_at_time[::-1]))
    return CIRFutureLikelihood(times, torch.stack(all_coefficients, dim=1),
                               torch.stack(all_B, dim=1), torch.stack(all_normalizers, dim=1),
                               eta, float(z_scale), tuple(event_arrays), float(T),
                               float(kappa), float(theta))


def _gamma_poisson_transition(z, r, q, D, generator):
    """Noncentral chi-square in the gamma/Poisson representation (float64)."""
    rate = D * z / q
    poisson = torch.poisson(rate, generator=generator)
    # torch.distributions.Gamma.sample does not accept a local generator.
    # The same PyTorch gamma primitive accepts one without changing global RNG.
    return q * torch._standard_gamma(r + poisson, generator=generator)


def _sample_polynomial_tilt(z, log_coefficients, B, duration, eta, *, kappa,
                            theta, z_scale, generator, triangular_weights):
    """Exact killed-CIR transition tilted by exp(-B*z') P(z').

    The exponential tilt has a gamma/Poisson transition with ``q=c*U`` and
    noncentrality ``lambda=D*z/q``.  For a polynomial degree n its additional
    size bias is a finite positive mixture over j<=n.  Draw that pair first,
    then ``M~Poisson(lambda)`` and ``z'=q*Gamma(r+M+j+n)``.  This samples the
    transition itself, rather than Euler integration of the posterior drift.
    """
    _, _, log_D, log_U = cir_affine_step(B, duration, eta, kappa=kappa, theta=theta)
    c = eta.square() / 2
    r, q, D = kappa * theta / c, c * log_U.exp(), log_D.exp()
    degree = log_coefficients.numel() - 1
    if degree == 0:
        return _gamma_poisson_transition(z, r, q, D, generator)
    weights, difference = (x[:degree + 1, :degree + 1] for x in triangular_weights)
    conditional_degree = log_coefficients[None, :] + weights + difference * torch.log(q / z_scale)
    order = torch.arange(degree + 1, dtype=z.dtype, device=z.device)
    propagated = torch.logsumexp(conditional_degree, dim=1) + order * log_D
    log_j = propagated + torch.log(z[:, None] / z_scale) * order
    j = torch.multinomial(torch.softmax(log_j, dim=1), 1, generator=generator).squeeze(1)
    log_n = conditional_degree.index_select(0, j)
    n = torch.multinomial(torch.softmax(log_n, dim=1), 1, generator=generator).squeeze(1)
    poisson = torch.poisson(D * z / q, generator=generator)
    return q * torch._standard_gamma(r + poisson + j + n, generator=generator)


@torch.no_grad()
def sample_true_cir_posterior(events, cfg, n_prime, n_paths, device="cpu", generator=None):
    """Sample independent true CIR posterior paths on the experiment grid.

    Returns intensity paths ``[batch*n_paths, n_steps+1]`` and method metadata.
    The constant-immigration CIR branch has exact finite-dimensional posterior
    transitions (up to float64/random-number arithmetic), including survival
    and events between grid nodes.  It conditions only on events through
    ``n_prime*dt``; after that horizon it uses exact CIR prior transitions.
    The learned model is never consulted.
    """
    truth = cfg.data.truth
    n_steps, T = int(cfg.data.n_steps), float(cfg.data.T)
    if int(n_prime) != n_prime or not 0 <= n_prime <= n_steps or int(n_paths) != n_paths or n_paths < 1:
        raise ValueError("n_prime must be in [0,n_steps] and n_paths must be positive integers")
    n_prime, n_paths = int(n_prime), int(n_paths)
    kappa, theta, eta_value, z0 = map(float, (truth.kappa, truth.theta, truth.eta, truth.z0))
    if min(kappa, theta, eta_value, z0, T, n_steps) <= 0:
        raise ValueError("CIR truth coefficients, initial state and grid must be positive")
    dev = torch.device(device)
    if generator is not None and torch.device(generator.device).type != dev.type:
        raise ValueError("The posterior generator must use the sampling device")
    raw_events = events.events if hasattr(events, "events") else events
    padded = torch.as_tensor(raw_events, dtype=torch.float64, device=dev)
    if padded.ndim == 1:
        padded = padded.unsqueeze(0)
    if padded.ndim != 2 or padded.shape[0] == 0:
        raise ValueError("events must be a nonempty padded [batch,event] tensor")
    if bool((torch.isnan(padded) | (padded <= 0)).any()):
        raise ValueError("Event entries must be positive times or +inf padding")
    dt, horizon = T / n_steps, n_prime * T / n_steps
    grid = [index * dt for index in range(n_steps + 1)]
    grid[-1] = T
    horizon = grid[n_prime]
    result = torch.empty((len(padded) * n_paths, n_steps + 1), dtype=torch.float64, device=dev)
    result[:, 0] = z0
    eta = _scalar64(eta_value, device=dev)
    c = eta.square() / 2
    r = kappa * theta / c
    used_counts = []
    z_scale = max(theta, z0, 1.)
    for observation, event_row in enumerate(padded):
        sequence = event_row[torch.isfinite(event_row) & (event_row <= horizon)].sort().values
        used_counts.append(int(sequence.numel()))
        event_count = {}
        for event in sequence.tolist():
            event_count[event] = event_count.get(event, 0) + 1
        timeline = sorted(set(grid[:n_prime + 1] + sequence.tolist()))
        cache = (build_cir_future_likelihood([sequence], timeline, eta, T=horizon,
                                            kappa=kappa, theta=theta, z_scale=z_scale)
                 if n_prime else None)
        triangular = _log_triangular_weights(int(sequence.numel()), eta, kappa, theta, dev)
        output = result[observation * n_paths:(observation + 1) * n_paths]
        grid_index = {time: index for index, time in enumerate(grid)}
        z = torch.full((n_paths,), z0, dtype=torch.float64, device=dev)
        previous = 0.
        for index, time in enumerate(timeline[1:], start=1):
            coefficients = cache.log_coefficients[index, 0]
            last = int(torch.isfinite(coefficients).nonzero()[-1])
            coefficients = coefficients[:last + 1]
            count = event_count.get(time, 0)
            if count:
                coefficients = torch.cat((coefficients.new_full((count,), -torch.inf),
                                          coefficients + count * math.log(z_scale)))
            z = _sample_polynomial_tilt(z, coefficients, cache.B[index, 0], time - previous,
                                       eta, kappa=kappa, theta=theta, z_scale=z_scale,
                                       generator=generator, triangular_weights=triangular)
            if time in grid_index:
                output[:, grid_index[time]] = z
            previous = time
        for index in range(n_prime + 1, n_steps + 1):
            duration = grid[index] - grid[index - 1]
            D = math.exp(-kappa * duration)
            q = c * (-math.expm1(-kappa * duration)) / kappa
            z = _gamma_poisson_transition(z, r, q, D, generator)
            output[:, index] = z
    if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise FloatingPointError("Nonpositive or nonfinite exact CIR posterior samples")
    metadata = {
        "kind": "analytic", "method": "exact_affine_polynomial", "label": "Exact CIR posterior",
        "exact_finite_dimensional": True, "exact_grid_paths": True, "independent_paths": True,
        "conditioning_horizon": horizon, "n_prime": n_prime, "n_paths": n_paths,
        "n_paths_per_observation": n_paths,
        "n_steps": n_steps, "events_conditioned": used_counts, "precision": "float64",
        "truth": {"kappa": kappa, "theta": theta, "eta": eta_value, "z0": z0},
        "reference_model": "Continuous-time true CIR/Cox model; survival and event times are handled exactly.",
        "transition": "Positive polynomial tilt of the killed-CIR gamma/Poisson mixture; exact CIR prior after horizon.",
        "numerical_precision": "Float64 coefficients and random variates; no posterior SDE time discretization.",
    }
    return result, metadata
