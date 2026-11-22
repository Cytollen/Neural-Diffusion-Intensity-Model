"""Forward-sensitivity estimator (paper Sec. 3.2, App. F.2, Eq. 12-13) vs reverse-mode autodiff.

Both estimators differentiate the *same* discretised objective, so in float64 they must agree to
round-off (relative difference ~1e-15; the test bound is 1e-6). Checked for four configurations:
CIR drift + remaining-count posterior, CIR + Deep Sets, MLP drift + Deep Sets with
``phi_uses_state``, and a trainable diffusion scale ``eta``.
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from ndim.config import ExperimentConfig
from ndim.data import Dataset
from ndim.elbo import compute_objective
from ndim.gradients import sensitivity_backward
from ndim.model import NDIM, build_model
from ndim.sde import brownian_increments

#: the parametric drift needs the intensity as the state; the MLP cases use the default latent state
INTENSITY_CIR = ["model.state_space=intensity", "model.drift.kind=cir"]
CONFIGS: Dict[str, List[str]] = {
    "cir+remaining_count": [
        *INTENSITY_CIR, "model.posterior.kind=remaining_count",
    ],
    "cir+deepset": [
        *INTENSITY_CIR, "model.posterior.kind=deepset",
    ],
    "cir+trainable_eta": [
        *INTENSITY_CIR, "model.posterior.kind=deepset", "model.diffusion.trainable=true",
        "model.diffusion.eta_init=1.5",
    ],
    "mlp+deepset(phi_uses_state)": [
        "model.drift.kind=mlp", "model.drift.hidden=[8, 8]", "model.posterior.kind=deepset",
        "model.posterior.phi_uses_state=true",
    ],
    "mlp+trainable_eta": [
        "model.drift.kind=mlp", "model.drift.hidden=[8, 8]", "model.posterior.kind=deepset",
        "model.diffusion.trainable=true", "model.diffusion.eta_init=1.5",
    ],
}


def _grads(model: NDIM) -> Dict[str, torch.Tensor]:
    return {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in model.named_parameters() if p.requires_grad}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(float(a.norm()), float(b.norm()), 1e-300)
    return float((a - b).norm()) / denom


def _setup(cfg: ExperimentConfig, ds: Dataset, B: int, m: int, n_prime: int, seed: int = 0):
    torch.manual_seed(seed)
    model = build_model(cfg).double()
    events = ds.train.events[:B].double()
    g = torch.Generator().manual_seed(seed + 1)
    W = brownian_increments(B * m, n_prime, cfg.dt, torch.device("cpu"), g).double()
    return model, events, W


@pytest.mark.parametrize("name", list(CONFIGS))
def test_sensitivity_matches_backprop_float64(make_cfg, smoke_dataset: Dataset, name: str) -> None:
    cfg = make_cfg(*CONFIGS[name])
    B, m = 2, 2
    n_prime = cfg.horizon_steps(1.0)
    model, events, W = _setup(cfg, smoke_dataset, B, m, n_prime)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if name.endswith("trainable_eta"):
        assert "diffusion.log_eta" in trainable
    else:
        assert "diffusion.log_eta" not in trainable
    if name.startswith("mlp"):
        assert any(n.startswith("drift.net") for n in trainable)
    else:
        assert {"drift.raw_kappa", "drift.theta_unit"} <= trainable

    # reverse mode
    model.zero_grad(set_to_none=True)
    out_bp = compute_objective(model, events, n_prime, m, W, cfg)
    out_bp.loss.backward()
    g_bp = _grads(model)

    # forward sensitivities
    model.zero_grad(set_to_none=True)
    out_s = sensitivity_backward(model, events, n_prime, m, W, cfg)
    g_s = _grads(model)

    assert float(out_bp.loss.detach()) == pytest.approx(float(out_s.loss.detach()), rel=1e-10)
    assert torch.allclose(out_bp.loglik, out_s.loglik, rtol=1e-10, atol=1e-10)
    assert torch.allclose(out_bp.kl, out_s.kl, rtol=1e-10, atol=1e-10)
    assert torch.allclose(out_bp.sim.Z.detach(), out_s.sim.Z, rtol=1e-10, atol=1e-10)

    assert set(g_bp) == set(g_s) == trainable
    worst = 0.0
    for n in trainable:
        assert torch.isfinite(g_s[n]).all(), n
        assert float(g_bp[n].norm()) > 0.0, f"backprop gradient of {n} is identically zero; test is vacuous"
        rel = _rel(g_bp[n], g_s[n])
        worst = max(worst, rel)
        assert rel < 1e-6, f"{name}: parameter {n} rel diff {rel:.3e}"
    assert worst < 1e-6


def test_sensitivity_partial_horizon_and_accumulation(make_cfg, smoke_dataset: Dataset) -> None:
    """A partial horizon ``n' < N`` and gradient accumulation into existing ``.grad`` buffers."""
    cfg = make_cfg(*CONFIGS["cir+deepset"])
    B, m = 3, 2
    n_prime = cfg.horizon_steps(0.5)
    model, events, W = _setup(cfg, smoke_dataset, B, m, n_prime, seed=4)
    model.zero_grad(set_to_none=True)
    compute_objective(model, events, n_prime, m, W, cfg).loss.backward()
    g_bp = _grads(model)
    # accumulate: pre-fill grads with ones, expect ones + sensitivity gradient
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    sensitivity_backward(model, events, n_prime, m, W, cfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert _rel(p.grad - 1.0, g_bp[n]) < 1e-6, n


def test_sensitivity_float32_agrees_loosely(make_cfg, smoke_dataset: Dataset) -> None:
    """In float32 the two estimators agree to single precision (the mode used in training)."""
    cfg = make_cfg(*CONFIGS["cir+remaining_count"])
    B, m = 2, 2
    n_prime = cfg.horizon_steps(1.0)
    torch.manual_seed(0)
    model = build_model(cfg)
    events = smoke_dataset.train.events[:B]
    W = brownian_increments(B * m, n_prime, cfg.dt, torch.device("cpu"), torch.Generator().manual_seed(1))
    model.zero_grad(set_to_none=True)
    compute_objective(model, events, n_prime, m, W, cfg).loss.backward()
    g_bp = _grads(model)
    model.zero_grad(set_to_none=True)
    sensitivity_backward(model, events, n_prime, m, W, cfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert _rel(g_bp[n], p.grad) < 1e-3, n
