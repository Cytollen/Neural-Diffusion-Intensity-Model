"""Tests for ``ndim.metrics``: exact Wasserstein-2 between path sets, expected path deviation, cumulative counts and
samples from the learned generative model (Fig. 3).

All model-based tests use the freshly initialised smoke model (``tiny_model``) and the true CIR drift/diffusion;
MCMC-based metrics run with a few chains and a handful of steps.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ndim import metrics
from ndim.config import ExperimentConfig, MCMCConfig
from ndim.data import EventData, make_truth_diffusion, make_truth_drift
from ndim.model import NDIM


def _truth(cfg: ExperimentConfig):
    return make_truth_drift(cfg.data.truth), make_truth_diffusion(cfg.data.truth)


def _mcmc_cfg(**kw) -> MCMCConfig:
    base = dict(kind="pcn", n_chains=4, n_steps=10, burn_in_frac=0.5, beta_init=0.2, target_accept=0.25, adapt=True, seed=1)
    base.update(kw)
    return MCMCConfig(**base)


# --------------------------------------------------------------------------------------
# Wasserstein (Fig. 5)
# --------------------------------------------------------------------------------------
def test_wasserstein_identical_sets_is_zero_and_symmetric() -> None:
    g = torch.Generator().manual_seed(0)
    a = torch.rand(6, 11, generator=g) * 10
    b = torch.rand(6, 11, generator=g) * 10
    dt = 0.4
    assert metrics.wasserstein2_paths(a, a, dt) == pytest.approx(0.0, abs=1e-6)
    assert metrics.wasserstein2_paths(a, a.clone(), dt, p=1) == pytest.approx(0.0, abs=1e-6)
    wab = metrics.wasserstein2_paths(a, b, dt)
    wba = metrics.wasserstein2_paths(b, a, dt)
    assert wab == pytest.approx(wba, rel=1e-6)
    assert wab > 0.0
    # invariant under permutation of the rows of one set (empirical measures)
    perm = torch.randperm(6, generator=g)
    assert metrics.wasserstein2_paths(a, b[perm], dt) == pytest.approx(wab, rel=1e-6)
    # W1 is a different metric but shares the same invariances
    w1 = metrics.wasserstein2_paths(a, b, dt, p=1)
    assert w1 > 0.0 and metrics.wasserstein2_paths(b[perm], a, dt, p=1) == pytest.approx(w1, rel=1e-6)


def test_wasserstein_constant_shift_closed_form() -> None:
    """Every pairing has the same cost, so W2 = c sqrt(dt G) and W1 = c dt G."""
    n, G, dt, c = 5, 9, 0.25, 3.0
    a = torch.zeros(n, G)
    b = torch.full((n, G), c)
    assert metrics.wasserstein2_paths(a, b, dt) == pytest.approx(c * math.sqrt(dt * G), rel=1e-6)
    assert metrics.wasserstein2_paths(a, b, dt, p=1) == pytest.approx(c * dt * G, rel=1e-6)
    # the optimal assignment is used (not the identity pairing): for a = [0, c], b = [c, 0] the cost is 0
    a2 = torch.tensor([[0.0] * G, [c] * G])
    b2 = torch.tensor([[c] * G, [0.0] * G])
    assert metrics.wasserstein2_paths(a2, b2, dt) == pytest.approx(0.0, abs=1e-6)
    # numpy inputs are accepted too
    assert metrics.wasserstein2_paths(a.numpy(), b.numpy(), dt) == pytest.approx(c * math.sqrt(dt * G), rel=1e-6)
    # scaling: W2 scales linearly with the paths and with sqrt(dt)
    assert metrics.wasserstein2_paths(2 * a, 2 * b, 4 * dt) == pytest.approx(4 * c * math.sqrt(dt * G), rel=1e-6)


def test_wasserstein_requires_same_number_of_paths() -> None:
    with pytest.raises(Exception):
        metrics.wasserstein2_paths(torch.zeros(4, 5), torch.zeros(3, 5), 0.1)


def test_wasserstein_rejects_nonfinite_values_and_bad_p() -> None:
    a = torch.zeros(3, 4)
    b = a.clone()
    b[0, 0] = float("inf")
    with pytest.raises(ValueError):
        metrics.wasserstein2_paths(a, b, 0.1)
    with pytest.raises(ValueError):
        metrics.wasserstein2_paths(a, a, 0.1, p=3)


# --------------------------------------------------------------------------------------
# expected path deviation (Table 1)
# --------------------------------------------------------------------------------------
def test_expected_path_deviation_identical_drifts_is_zero(cfg_copy: ExperimentConfig, device, gen) -> None:
    cfg_copy.model.state_space = "intensity"  # both coefficient sets in this test describe intensities
    drift, diff = _truth(cfg_copy)
    out = metrics.expected_path_deviation(drift, drift, diff, cfg_copy, n_paths=16, device=device, generator=gen)
    assert set(out) >= {"l2", "l1", "rmse"}
    assert out["l2"] == pytest.approx(0.0, abs=1e-8)
    assert out["l1"] == pytest.approx(0.0, abs=1e-8)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-8)


def test_expected_path_deviation_positive_and_consistent(cfg_copy: ExperimentConfig, device) -> None:
    cfg_copy.model.state_space = "intensity"  # both coefficient sets in this test describe intensities
    true_drift, diff = _truth(cfg_copy)
    other = lambda z, t: 0.3 * (40.0 - z)                    # wrong mean-reversion level
    out = metrics.expected_path_deviation(other, true_drift, diff, cfg_copy, 32, device, torch.Generator().manual_seed(1))
    assert out["l2"] > 0 and out["l1"] > 0 and out["rmse"] > 0
    assert out["rmse"] == pytest.approx(math.sqrt(out["l2"] / cfg_copy.data.T), rel=1e-5)
    # Cauchy-Schwarz: (E int |d| dt)^2 <= T * E int d^2 dt
    assert out["l1"] ** 2 <= cfg_copy.data.T * out["l2"] * (1 + 1e-6)
    assert out["l2_stderr"] > 0 and out["n_paths"] == 32
    # common noise + fixed seed => deterministic
    out2 = metrics.expected_path_deviation(other, true_drift, diff, cfg_copy, 32, device, torch.Generator().manual_seed(1))
    assert out2["l2"] == pytest.approx(out["l2"], rel=1e-6)


def test_expected_path_deviation_true_diffusion_override(cfg_copy: ExperimentConfig, device) -> None:
    """Experiment 3 (trainable eta): the true prior may use a different diffusion than the learned one."""
    cfg_copy.model.state_space = "intensity"  # both coefficient sets in this test describe intensities
    drift, diff = _truth(cfg_copy)
    diff2 = lambda z, t: 2.0 * diff(z, t)
    same = metrics.expected_path_deviation(drift, drift, diff, cfg_copy, 16, device, torch.Generator().manual_seed(0), true_diffusion_fn=diff)
    other = metrics.expected_path_deviation(drift, drift, diff, cfg_copy, 16, device, torch.Generator().manual_seed(0), true_diffusion_fn=diff2)
    assert same["l2"] == pytest.approx(0.0, abs=1e-8)
    assert other["l2"] > 0 and other["l1"] > 0
    with pytest.raises(ValueError):
        metrics.expected_path_deviation(drift, drift, diff, cfg_copy, 0, device)


# --------------------------------------------------------------------------------------
# cumulative counts and generated observations (Fig. 3)
# --------------------------------------------------------------------------------------
def test_cumulative_count_curves() -> None:
    ev = [np.array([0.5, 1.5, 1.5, 3.0]), np.array([]), np.array([2.0])]
    t_grid = np.array([0.0, 1.0, 1.5, 2.0, 4.0])
    C = metrics.cumulative_count_curves(ev, t_grid)
    assert C.shape == (3, 5)
    assert np.array_equal(C[0], [0, 1, 3, 3, 4])             # counts of events <= t
    assert np.array_equal(C[1], [0, 0, 0, 0, 0])
    assert np.array_equal(C[2], [0, 0, 0, 1, 1])


def test_cumulative_count_curves_accepts_padded_tensor_rows() -> None:
    ev = EventData.from_lists([np.array([0.5, 2.5]), np.array([])], T=4.0)
    C = metrics.cumulative_count_curves([ev.events[0], ev.events[1]], torch.tensor([0.0, 1.0, 3.0, 4.0]))
    assert np.array_equal(C, [[0, 1, 2, 2], [0, 0, 0, 0]])


def test_generate_observations_from_model(smoke_cfg: ExperimentConfig, tiny_model: NDIM, device, gen) -> None:
    n_obs = 5
    events_list, paths = metrics.generate_observations_from_model(tiny_model, smoke_cfg, n_obs, device, gen)
    assert isinstance(events_list, list) and len(events_list) == n_obs
    assert paths.shape == (n_obs, smoke_cfg.data.n_steps + 1)
    assert paths.device.type == "cpu" and torch.all(paths >= 0)
    for e in events_list:
        e = np.asarray(e)
        assert e.ndim == 1
        assert np.all(np.diff(e) >= 0) and np.all((e >= 0) & (e <= smoke_cfg.data.T))


def test_generated_counts_consistent_with_paths(smoke_cfg: ExperimentConfig, tiny_model: NDIM, device) -> None:
    """Thinning consistency (paper App. G, Alg. 2): ``E[count | Z] = int_0^T Z dt`` (left Riemann on the grid),
    so the mean of ``count - int Z dt`` over observations is zero up to Poisson noise."""
    n_obs = 64
    events_list, paths = metrics.generate_observations_from_model(tiny_model, smoke_cfg, n_obs, device, torch.Generator().manual_seed(0))
    N, dt = smoke_cfg.data.n_steps, smoke_cfg.dt
    integral = (paths[:, :N].sum(dim=1) * dt).numpy()
    counts = np.array([len(e) for e in events_list], dtype=float)
    se = math.sqrt(float(integral.mean()) / n_obs)          # Var(count - int Z) = E int Z  (conditionally Poisson)
    assert abs(float((counts - integral).mean())) < 5 * se


# --------------------------------------------------------------------------------------
def test_posterior_wasserstein_keys(smoke_cfg: ExperimentConfig, smoke_dataset, tiny_model: NDIM, device, gen) -> None:
    drift, diff = _truth(smoke_cfg)
    events = smoke_dataset.test.events[:2]
    mc = _mcmc_cfg(n_chains=4, n_steps=10)
    out = metrics.posterior_wasserstein(tiny_model, drift, diff, events, smoke_cfg, smoke_cfg.data.n_steps, 4, mc, device, gen)
    assert set(out) >= {"w2_vi", "w2_prior_baseline", "mean_w2_vi", "mean_w2_prior_baseline"}
    assert len(out["w2_vi"]) == 2 and len(out["w2_prior_baseline"]) == 2
    assert all(math.isfinite(float(v)) and float(v) >= 0 for v in list(out["w2_vi"]) + list(out["w2_prior_baseline"]))
    assert out["mean_w2_vi"] == pytest.approx(float(np.mean(out["w2_vi"])), rel=1e-6)
    assert out["mean_w2_prior_baseline"] == pytest.approx(float(np.mean(out["w2_prior_baseline"])), rel=1e-6)
    assert out["n_paths"] == 4 and out["n_obs"] == 2 and out["reference"]["kind"] == "analytic"
    assert out["reference"]["exact_finite_dimensional"]
    assert "mcmc_acceptance" not in out and "seconds_mcmc" not in out
    # n_paths sets equal cardinalities for the empirical assignment.
    out2 = metrics.posterior_wasserstein(tiny_model, drift, diff, events, smoke_cfg, smoke_cfg.horizon_steps(0.5), 3, mc, device,
                                         torch.Generator().manual_seed(0))
    assert len(out2["w2_vi"]) == 2 and out2["n_paths"] == 3 and out2["n_prime"] == smoke_cfg.horizon_steps(0.5)


def test_euler_gain_of_a_linear_drift(make_cfg) -> None:
    """A drift -c x of the latent state has the gain 1 - c dt at every step: below -1 when c dt exceeds 2."""
    from ndim.metrics import STIFF_SHARE, euler_gain, euler_gain_summary
    from ndim.model import build_model

    cfg = make_cfg("model.burn_in=0.48", "model.x_start=0.0")                # 3 burn-in steps of 0.16
    model = build_model(cfg)
    dt, nb = cfg.dt, cfg.burn_steps
    t_grid = model.t_start + torch.arange(nb + cfg.data.n_steps + 1) * dt
    paths = torch.rand(7, nb + cfg.data.n_steps + 1) * 50.0 + 0.5             # intensity paths

    class Linear(torch.nn.Module):
        def __init__(self, c: float) -> None:
            super().__init__()
            self.c = c

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return -self.c * x * (t >= 0) - 0.5 * self.c * x * (t < 0)        # half as stiff on the burn-in

    model.drift = Linear(5.0)
    gain = euler_gain(model, paths, t_grid, dt)
    assert gain.shape == (7, nb + cfg.data.n_steps)
    torch.testing.assert_close(gain[:, nb:], torch.full_like(gain[:, nb:], 1.0 - 5.0 * dt), atol=1e-5, rtol=0)        # 0.2
    torch.testing.assert_close(gain[:, :nb], torch.full_like(gain[:, :nb], 1.0 - 2.5 * dt), atol=1e-5, rtol=0)
    calm = euler_gain_summary(model, paths, t_grid, dt, nb)
    assert calm["share_below_minus_one"] == 0.0 and calm["share_negative"] == 0.0 and calm["min"] == pytest.approx(0.2, abs=1e-5)
    model.drift = Linear(15.0)                                                 # 1 - 2.4 = -1.4 on the window, -0.2 on the burn-in
    stiff = euler_gain_summary(model, paths, t_grid, dt, nb)
    assert stiff["min"] == pytest.approx(-1.4, abs=1e-5) and stiff["share_below_minus_one_window"] == 1.0
    assert stiff["share_below_minus_one_burn_in"] == 0.0 and stiff["share_negative"] == 1.0
    assert stiff["share_below_minus_one"] == pytest.approx(cfg.data.n_steps / (nb + cfg.data.n_steps))
    assert stiff["share_below_minus_one"] > STIFF_SHARE and not stiff["first_step_left_out"]

    class FirstStep(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return -15.0 * x * (t < model.t_start + 0.5 * dt)                 # stiff at the first step only

    model.drift = FirstStep()
    counted = euler_gain_summary(model, paths, t_grid, dt, nb)
    assert counted["min"] == pytest.approx(-1.4, abs=1e-5) and counted["share_below_minus_one_burn_in"] == pytest.approx(1.0 / nb)
    # when every path starts from the same state, the first step cannot enlarge a deviation and is left out
    same_start = paths.clone()
    same_start[:, 0] = 3.0
    left_out = euler_gain_summary(model, same_start, t_grid, dt, nb)
    assert left_out["first_step_left_out"] and left_out["share_below_minus_one"] == 0.0 and left_out["min"] == pytest.approx(1.0, abs=1e-5)
    assert left_out["share_below_minus_one_burn_in"] == 0.0 and left_out["share_below_minus_one_window"] == 0.0
