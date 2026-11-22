"""The coefficient ``eta`` of the CIR diffusion estimated from the observations (``ndim.cir_eta``)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ndim.cir_eta import MIN_OBSERVATIONS, N_FINE, estimate_cir_eta, estimate_from_counts, eta2_curve, fine_counts, resolve_eta
from ndim.model import build_model

AUTO = "model.diffusion={kind: cir, eta_init: auto, trainable: false}"


def _cox_counts(n: int, T: float, kappa: float, theta: float, eta: float, seed: int, sub: int = 2) -> np.ndarray:
    """Counts on the fine grid of ``n`` Cox processes with ``dZ = kappa (theta - Z) dt + eta sqrt(Z) dB``, ``Z_0 = theta``
    (Euler with full truncation on a grid ``sub`` times finer than the fine grid)."""
    rng = np.random.default_rng(seed)
    steps = N_FINE * sub
    dt = T / steps
    z = np.full(n, float(theta))
    lam = np.zeros((n, N_FINE))
    for k in range(steps):
        zp = np.maximum(z, 0.0)
        lam[:, k // sub] += zp * dt
        z = z + kappa * (theta - zp) * dt + eta * np.sqrt(zp) * rng.normal(size=n) * math.sqrt(dt)
    return rng.poisson(lam).astype(np.float64)


def test_fine_counts_reads_padded_arrays_tensors_and_lists() -> None:
    rows = [[0.5, 1.5, 3.9], [0.2]]
    padded = np.array([[0.5, 1.5, 3.9], [0.2, np.inf, np.nan]])
    for events in (rows, padded, torch.tensor(padded)):
        C = fine_counts(events, 4.0, 8)                                   # bins of width 0.5
        assert C.shape == (2, 8) and C.sum() == 4
        assert list(C[0]) == [0, 1, 0, 1, 0, 0, 0, 1] and list(C[1]) == [1, 0, 0, 0, 0, 0, 0, 0]


def test_the_curve_is_zero_in_expectation_for_a_poisson_process() -> None:
    """Without a random intensity the variance of a difference of counts is the mean of their sum."""
    rng = np.random.default_rng(0)
    rate = 10.0 + 8.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, N_FINE))     # events per unit time on [0, 24]
    C = rng.poisson(np.tile(rate * 24.0 / N_FINE, (2000, 1))).astype(np.float64)
    curve = eta2_curve(C, [30, 60, 120], 24.0 / N_FINE)
    assert np.all(np.abs(curve) < 0.05)
    estimate = estimate_from_counts(C[:300], 24.0, n_boot=100)
    assert abs(estimate.eta2) < 3.0 * estimate.eta2_se and not estimate.determined           # nothing but sampling error
    assert estimate.floor == pytest.approx(0.1 * math.sqrt(C[:300].sum() / 300 / 24.0 / 24.0))
    assert estimate.eta >= estimate.floor and estimate.floored == (estimate.eta == estimate.floor)
    # an estimate below the floor is replaced by it
    floored = estimate_from_counts(C[:300], 24.0, n_boot=100, floor_cv=10.0)
    assert floored.floored and floored.eta == floored.floor == pytest.approx(100.0 * estimate.floor) and "the floor" in floored.summary()


def test_the_coefficient_of_a_cir_intensity_is_recovered() -> None:
    C = _cox_counts(600, 24.0, 0.3, 12.0, 1.0, seed=1)
    estimate = estimate_from_counts(C, 24.0, n_boot=100)
    assert 0.85 < estimate.eta < 1.15 and estimate.interval[0] < 1.0 < estimate.interval[1]
    assert estimate.determined and not estimate.floored and not estimate.with_groups
    assert estimate.n_observations == 600 and len(estimate.curve) == len(estimate.widths) == len(estimate.used)
    assert estimate.eta == pytest.approx(math.sqrt(estimate.eta2)) and "95% interval" in estimate.summary()
    json.dumps(estimate.to_dict())                                         # plain numbers and lists
    # thinning with probability p turns the intensity into p Z, whose coefficient is sqrt(p) eta
    thinned = np.random.default_rng(2).binomial(C.astype(np.int64), 0.5).astype(np.float64)
    half = estimate_from_counts(thinned, 24.0, n_boot=100)
    assert half.eta == pytest.approx(math.sqrt(0.5) * estimate.eta, rel=0.2)
    # every event doubled: the coefficient of the doubled intensity is sqrt(2) eta, and the pairs show as events in groups
    doubled = estimate_from_counts(2.0 * C, 24.0, n_boot=100)
    assert doubled.with_groups and len(doubled.coefficients) == 3 and doubled.coefficients[2] > 0
    assert doubled.eta == pytest.approx(math.sqrt(2.0) * estimate.eta, rel=0.2)


def test_event_times_and_counts_give_the_same_estimate() -> None:
    rng = np.random.default_rng(3)
    C = _cox_counts(60, 24.0, 0.3, 12.0, 1.0, seed=4)
    delta = 24.0 / N_FINE
    rows = [np.sort(np.concatenate([(j + rng.random(int(c))) * delta for j, c in enumerate(row) if c])) for row in C]
    a, b = estimate_cir_eta(rows, 24.0, n_boot=20), estimate_from_counts(C, 24.0, n_boot=20)
    assert a.eta == pytest.approx(b.eta) and a.interval == pytest.approx(b.interval)


def test_too_few_observations_use_the_floor() -> None:
    C = _cox_counts(MIN_OBSERVATIONS - 1, 24.0, 0.3, 12.0, 1.0, seed=5)
    estimate = estimate_from_counts(C, 24.0)
    assert estimate.floored and estimate.eta == estimate.floor > 0 and not estimate.determined and estimate.curve == []
    assert "too few" in estimate.summary()
    with pytest.raises(ValueError, match="no events"):
        estimate_from_counts(np.zeros((5, N_FINE)), 24.0)


def test_auto_is_resolved_from_the_training_observations(make_cfg, smoke_dataset, tmp_path: Path) -> None:
    cfg = make_cfg(AUTO)
    assert cfg.model.diffusion.eta_init == "auto"
    with pytest.raises(ValueError, match="resolve_eta"):                   # a model cannot be built before the number is known
        build_model(cfg)
    estimate = resolve_eta(cfg, smoke_dataset.train, tmp_path)
    assert estimate is not None and estimate.eta > 0 and cfg.model.diffusion.eta_init == pytest.approx(estimate.eta, rel=1e-3)
    saved = json.loads((tmp_path / "eta_estimate.json").read_text())
    assert saved["eta"] == pytest.approx(estimate.eta) and saved["n_observations"] == len(smoke_dataset.train)
    assert (tmp_path / "eta_estimate.png").stat().st_size > 0
    assert resolve_eta(cfg, smoke_dataset.train, tmp_path) is None         # a number is left as it is
    build_model(cfg)
    for bad in ("model.diffusion={kind: cir, eta_init: auto, trainable: true}", "model.diffusion={kind: mlp, eta_init: auto, trainable: false}",
                "model.diffusion.eta_init=0", "model.diffusion.eta_init=large"):
        with pytest.raises(ValueError, match="eta_init"):
            make_cfg(bad)
