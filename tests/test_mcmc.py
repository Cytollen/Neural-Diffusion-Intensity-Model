"""Tests for ``ndim.mcmc`` (path-space pCN / independence sampler; DESIGN.md Sec. 5).

Everything runs on ``configs/smoke.yaml`` (25 Euler steps on ``[0, 4]``) with the *true* CIR drift/diffusion,
a handful of chains and at most a few hundred iterations, so the whole file takes a few seconds on a laptop.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from ndim import mcmc
from ndim.config import ExperimentConfig, MCMCConfig
from ndim.data import EventData, make_truth_diffusion, make_truth_drift
from ndim.sde import brownian_increments, simulate_prior

INF = float("inf")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _truth(cfg: ExperimentConfig):
    return make_truth_drift(cfg.data.truth), make_truth_diffusion(cfg.data.truth)


def _mcmc_cfg(**kw) -> MCMCConfig:
    base = dict(kind="pcn", n_chains=4, n_steps=30, burn_in_frac=0.5, beta_init=0.2, target_accept=0.25, adapt=True, seed=123)
    base.update(kw)
    return MCMCConfig(**base)


def _prior_mean(cfg: ExperimentConfig, device: torch.device, n_paths: int = 1024, seed: int = 0) -> float:
    """Monte-Carlo mean intensity of the true prior over the grid points ``k < N`` (left-Riemann convention)."""
    drift, diff = _truth(cfg)
    N, dt = cfg.data.n_steps, cfg.dt
    W = brownian_increments(n_paths, N, dt, device, torch.Generator().manual_seed(seed))
    return float(simulate_prior(drift, diff, cfg.data.truth.z0, N, dt, W)[:, :N].mean())


def _check_diag(diag, n_steps: int, n_chains: int) -> None:
    assert isinstance(diag, mcmc.MCMCDiagnostics)
    assert 0.0 <= diag.acceptance_rate <= 1.0
    assert diag.n_steps == n_steps and diag.n_chains == n_chains
    assert 1e-3 <= diag.beta_final <= 1.0
    trace = np.asarray(diag.loglik_trace)
    assert trace.shape == (n_steps,)
    assert np.all(np.isfinite(trace))


# --------------------------------------------------------------------------------------
# Metropolis decision (pure function)
# --------------------------------------------------------------------------------------
def test_metropolis_accept_handles_nonfinite_logliks() -> None:
    """Non-finite proposals are never accepted; a chain with a non-finite *current* value accepts any finite proposal;
    otherwise ``accept = u < exp(ll_new - ll_cur)`` (the Gaussian prior cancels for pCN proposals)."""
    ll_cur = torch.tensor([0.0, 0.0, -INF, 0.0, 0.0])
    ll_new = torch.tensor([-INF, float("nan"), 0.0, math.log(0.5), math.log(0.5)])
    u = torch.tensor([0.0, 0.0, 0.999, 0.4, 0.6])
    acc = mcmc.metropolis_accept(ll_cur, ll_new, u)
    assert acc.dtype == torch.bool
    assert acc.tolist() == [False, False, True, True, False]
    # a (much) better proposal is always accepted, even after clipping the log-ratio to [-60, 60]
    assert bool(mcmc.metropolis_accept(torch.zeros(2), torch.full((2,), 1e6), torch.tensor([0.0, 0.999])).all())
    # a (much) worse finite proposal is never accepted
    assert not bool(mcmc.metropolis_accept(torch.zeros(2), torch.full((2,), -1e6), torch.tensor([0.0, 1e-30])).any())


# --------------------------------------------------------------------------------------
# single-observation sampler
# --------------------------------------------------------------------------------------
def test_sample_posterior_mcmc_shapes_and_diagnostics(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[0]
    mc = _mcmc_cfg(n_chains=4, n_steps=30)
    paths, diag = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, n_prime=N, mcmc=mc, device=device, generator=gen)
    assert paths.shape == (4, N + 1)
    assert paths.dtype == torch.float32
    assert torch.all(paths >= 0) and torch.isfinite(paths).all()
    assert torch.allclose(paths[:, 0], torch.full((4,), float(smoke_cfg.data.truth.z0)))
    assert not paths.requires_grad
    _check_diag(diag, 30, 4)
    assert diag.kind == "pcn" and diag.n_burn_in == 15


def test_partial_horizon_and_custom_total_length(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    n_prime = smoke_cfg.horizon_steps(0.5)
    row = smoke_dataset.test.events[1]
    mc = _mcmc_cfg(n_chains=3, n_steps=20)
    paths, diag = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, n_prime, mc, device, gen)
    assert paths.shape == (3, N + 1)                         # simulated to the full horizon T (Fig. 4b)
    paths2, _ = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, n_prime, mc, device,
                                           torch.Generator().manual_seed(0), n_steps_total=n_prime)
    assert paths2.shape == (3, n_prime + 1)
    _check_diag(diag, 20, 3)


def test_acceptance_strictly_between_zero_and_one(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    """With a moderate pCN step some proposals are accepted and some rejected."""
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[2]
    mc = _mcmc_cfg(n_chains=8, n_steps=200, beta_init=0.2, adapt=True)
    _, diag = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, gen)
    assert 0.0 < diag.acceptance_rate < 1.0
    assert 0.0 < diag.acceptance_rate_post_burnin < 1.0
    _check_diag(diag, 200, 8)


def test_independence_kind_runs_with_beta_one(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    """The independence sampler (the paper's) is pCN with ``beta = 1`` (fresh prior proposal every step), never adapted."""
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[0]
    mc = _mcmc_cfg(kind="independence", n_chains=4, n_steps=40, beta_init=0.2, adapt=True)
    paths, diag = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, gen)
    assert paths.shape == (4, N + 1)
    assert diag.kind == "independence"
    assert diag.beta_final == pytest.approx(1.0)
    assert np.allclose(np.asarray(diag.beta_trace), 1.0)
    _check_diag(diag, 40, 4)


def test_pcn_adapts_towards_target_and_independence_acceptance_is_small(smoke_cfg: ExperimentConfig, device) -> None:
    """Informative observation (300 evenly spread events, ~2x the prior's typical intensity): the Robbins-Monro
    adaptation moves the pCN step ``beta`` away from its initial value to where the acceptance rate is close to
    ``target_accept``, and freezes it after the burn-in. The independence sampler (``beta = 1``, the default kernel)
    accepts only a few percent of its proposals on the same observation (why the paper uses 10,000 burn-in steps)."""
    drift, diff = _truth(smoke_cfg)
    N, T = smoke_cfg.data.n_steps, smoke_cfg.data.T
    many = torch.linspace(0.01, T, 300)
    n_steps, beta_init, target = 300, 0.3, 0.25
    mc = _mcmc_cfg(n_chains=8, n_steps=n_steps, beta_init=beta_init, target_accept=target, adapt=True)
    _, d = mcmc.sample_posterior_mcmc(drift, diff, many, smoke_cfg, N, mc, device, torch.Generator().manual_seed(1))
    n_burn = int(round(0.5 * n_steps))
    assert d.n_burn_in == n_burn
    bt = np.asarray(d.beta_trace)
    assert bt.shape == (n_steps,)
    assert bt[0] == pytest.approx(beta_init)                                  # first proposal uses beta_init
    assert d.beta_final != pytest.approx(beta_init)                          # adaptation moved the step ...
    assert mcmc.BETA_MIN <= d.beta_final < mcmc.BETA_MAX                     # ... without hitting the clip
    assert np.all(bt[n_burn:] == bt[n_burn]) and bt[-1] == pytest.approx(d.beta_final)   # frozen after burn-in
    assert abs(d.acceptance_rate_post_burnin - target) < 0.12
    # same observation, independence sampler
    ind = _mcmc_cfg(kind="independence", n_chains=8, n_steps=n_steps)
    _, di = mcmc.sample_posterior_mcmc(drift, diff, many, smoke_cfg, N, ind, device, torch.Generator().manual_seed(1))
    assert di.beta_final == pytest.approx(1.0)
    assert di.acceptance_rate < 0.1
    assert di.acceptance_rate < d.acceptance_rate


def test_pcn_without_adaptation_keeps_beta_fixed(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    mc = _mcmc_cfg(n_chains=4, n_steps=30, beta_init=0.37, adapt=False)
    _, d = mcmc.sample_posterior_mcmc(drift, diff, smoke_dataset.test.events[0], smoke_cfg, N, mc, device, gen)
    assert d.beta_final == pytest.approx(0.37)
    assert np.allclose(np.asarray(d.beta_trace), 0.37)


def test_reproducible_with_generator(smoke_cfg: ExperimentConfig, smoke_dataset, device) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[0]
    mc = _mcmc_cfg(n_chains=3, n_steps=25)
    a, da = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(7))
    b, db = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert da.acceptance_rate == db.acceptance_rate
    assert np.array_equal(da.loglik_trace, db.loglik_trace)
    # a different seed gives different chains
    c, _ = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(8))
    assert not torch.equal(a, c)


def test_posterior_mean_moves_towards_the_data(smoke_cfg: ExperimentConfig, device) -> None:
    """Many events => posterior mean intensity over [0, T'] above the prior mean; no events => below."""
    drift, diff = _truth(smoke_cfg)
    N, T = smoke_cfg.data.n_steps, smoke_cfg.data.T
    prior_mean = _prior_mean(smoke_cfg, device, n_paths=512)
    mc = _mcmc_cfg(n_chains=8, n_steps=300, beta_init=0.3)
    # ~75 events per unit time, evenly spread: far above the prior's typical intensity early on
    many = torch.linspace(0.01, T, 300)
    paths_hi, diag_hi = mcmc.sample_posterior_mcmc(drift, diff, many, smoke_cfg, N, mc, device, torch.Generator().manual_seed(1))
    assert float(paths_hi[:, :N].mean()) > prior_mean
    assert diag_hi.loglik_trace[-1] > diag_hi.loglik_trace[0]      # chains climbed from the initial prior draw
    # zero events on [0, T]: the likelihood -int Z dt pulls the paths down
    none = torch.full((3,), INF)
    paths_lo, diag_lo = mcmc.sample_posterior_mcmc(drift, diff, none, smoke_cfg, N, mc, device, torch.Generator().manual_seed(2))
    assert float(paths_lo[:, :N].mean()) < prior_mean
    assert diag_lo.loglik_trace[-1] > diag_lo.loglik_trace[0]


def test_posterior_mean_tracks_event_rate_of_test_observations(smoke_cfg: ExperimentConfig, smoke_dataset, device) -> None:
    """Fig. 4a on the smoke test set: the observation with the fewest events pulls the posterior mean over ``[0, T]``
    below the prior mean, the one with the most events pushes it above."""
    drift, diff = _truth(smoke_cfg)
    N, T = smoke_cfg.data.n_steps, smoke_cfg.data.T
    counts = smoke_dataset.test.counts
    i_min, i_max = int(counts.argmin()), int(counts.argmax())
    prior_mean = _prior_mean(smoke_cfg, device)
    rate_min, rate_max = float(counts[i_min]) / T, float(counts[i_max]) / T
    if not (rate_min < prior_mean < rate_max):  # pragma: no cover - precondition on the cached dataset
        pytest.skip("smoke test set does not bracket the prior mean intensity")
    mc = _mcmc_cfg(n_chains=16, n_steps=300, beta_init=0.2)
    lo, _ = mcmc.sample_posterior_mcmc(drift, diff, smoke_dataset.test.events[i_min], smoke_cfg, N, mc, device, torch.Generator().manual_seed(3))
    hi, _ = mcmc.sample_posterior_mcmc(drift, diff, smoke_dataset.test.events[i_max], smoke_cfg, N, mc, device, torch.Generator().manual_seed(3))
    assert float(lo[:, :N].mean()) < prior_mean < float(hi[:, :N].mean())


def test_positivity_default_is_full_truncation_and_validated(smoke_cfg: ExperimentConfig, smoke_dataset, device) -> None:
    """The Euler map uses full truncation (the truth's scheme) unless told otherwise."""
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[0]
    mc = _mcmc_cfg(n_chains=3, n_steps=10)
    a, _ = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(5))
    b, _ = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(5), positivity="full_truncation")
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(5), positivity="bogus")


def test_invalid_arguments_raise(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[0]
    with pytest.raises(ValueError):  # returned paths shorter than the conditioning horizon
        mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, _mcmc_cfg(n_steps=2), device, gen, n_steps_total=N // 2)
    with pytest.raises(ValueError):  # several observations passed to the single-observation API
        mcmc.sample_posterior_mcmc(drift, diff, smoke_dataset.test.events[:2], smoke_cfg, N, _mcmc_cfg(n_steps=2), device, gen)
    with pytest.raises(ValueError):  # n_prime must be >= 1
        mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, 0, _mcmc_cfg(n_steps=2), device, gen)


# --------------------------------------------------------------------------------------
# batch sampler
# --------------------------------------------------------------------------------------
def test_batch_sampler_shapes(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    events = smoke_dataset.test.events[:3]
    mc = _mcmc_cfg(n_chains=4, n_steps=15)
    paths, diags = mcmc.sample_posterior_mcmc_batch(drift, diff, events, smoke_cfg, N, mc, device, gen)
    assert paths.shape == (3, 4, N + 1)
    assert torch.all(paths >= 0) and torch.isfinite(paths).all()
    assert torch.allclose(paths[:, :, 0], torch.full((3, 4), float(smoke_cfg.data.truth.z0)))
    assert isinstance(diags, list) and len(diags) == 3
    for d in diags:
        _check_diag(d, 15, 4)


def test_batch_with_one_observation_matches_single(smoke_cfg: ExperimentConfig, smoke_dataset, device) -> None:
    """Row layout ``o * n_chains + c``: a one-observation batch reproduces the single-observation sampler exactly."""
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    row = smoke_dataset.test.events[1]
    mc = _mcmc_cfg(n_chains=3, n_steps=12)
    single, ds = mcmc.sample_posterior_mcmc(drift, diff, row, smoke_cfg, N, mc, device, torch.Generator().manual_seed(9))
    batch, db = mcmc.sample_posterior_mcmc_batch(drift, diff, row[None, :], smoke_cfg, N, mc, device, torch.Generator().manual_seed(9))
    assert batch.shape == (1, 3, N + 1)
    assert torch.equal(batch[0], single)
    assert len(db) == 1 and db[0].acceptance_rate == ds.acceptance_rate


def test_batch_sampler_handles_empty_observation(smoke_cfg: ExperimentConfig, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    events = EventData.from_lists([np.array([0.5, 1.0, 2.5]), np.array([])], T=smoke_cfg.data.T).events
    mc = _mcmc_cfg(n_chains=2, n_steps=10)
    paths, diags = mcmc.sample_posterior_mcmc_batch(drift, diff, events, smoke_cfg, N, mc, device, gen)
    assert paths.shape == (2, 2, N + 1) and torch.isfinite(paths).all()
    assert len(diags) == 2
    # chunked processing gives the same shapes and one diagnostics object per observation
    paths_c, diags_c = mcmc.sample_posterior_mcmc_batch(drift, diff, events, smoke_cfg, N, mc, device, gen, chunk_size=1)
    assert paths_c.shape == paths.shape and len(diags_c) == 2


def test_diagnostics_serialisation_and_summary(smoke_cfg: ExperimentConfig, smoke_dataset, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    N = smoke_cfg.data.n_steps
    _, diags = mcmc.sample_posterior_mcmc_batch(drift, diff, smoke_dataset.test.events[:2], smoke_cfg, N, _mcmc_cfg(n_chains=2, n_steps=6), device, gen)
    d = diags[0].to_dict()
    json.dumps(d)                                             # JSON-friendly (traces as lists)
    assert len(d["loglik_trace"]) == 6 and len(d["beta_trace"]) == 6
    assert "loglik_trace" not in diags[0].to_dict(include_traces=False)
    s = mcmc.summarize_diagnostics(diags)
    assert set(s) >= {"acceptance_rate", "beta_final"}
    assert 0.0 <= s["acceptance_rate"] <= 1.0
    assert math.isnan(mcmc.summarize_diagnostics([])["acceptance_rate"])
