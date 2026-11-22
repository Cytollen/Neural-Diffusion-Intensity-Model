"""Forward-sensitivity gradient estimator (paper Sec. 3.2 and App. F.2, Eq. 12-13).

Instead of reverse-mode autodiff through the Euler–Maruyama recursion (``train.grad_mode=backprop``),
this module propagates the parameter sensitivities ``J_k = d z_k / d p`` forward alongside the path::

    z_{k+1} = z_k + F(z_k; p) dt + G(z_k; p) dW_k,     F = b + 1{k<n'} sigma u,   G = sigma
    J_{k+1} = J_k + [d_z F J_k + d_p F] dt + [d_z G J_k + d_p G] dW_k,     J_0 = 0

and assembles the ELBO gradient with the chain rule

    d L / d p = sum_k (dL/dz_k) J_k + sum_k (dL/du_k) (d_z u_k J_k + d_p u_k),

where ``dL/dz_k`` and ``dL/du_k`` come from autograd applied to the *discretised* objective as a
function of the path and the controls (so the discretisation is identical to the backprop mode, and
the two estimators agree up to floating point; see ``tests/test_gradients.py``).

The per-step local Jacobians are computed with ``torch.func`` (``vmap`` over rows of ``jacrev`` w.r.t. the
parameters and the state), so the cost per step is O(P) and this mode is intended for small models
(the parametric CIR drift, small posterior networks). It generalises the paper's formulas to any
trainable parameter, including the diffusion scale ``eta``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.func import functional_call, jacrev, vmap

from .config import ExperimentConfig
from .elbo import ObjectiveOutput
from .likelihood import poisson_loglik
from .model import NDIM
from .sde import SimOutput

Tensor = torch.Tensor


def _split_params(model: NDIM) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], List[str]]:
    """Trainable parameters as sub-module dicts (names relative to each sub-module) plus the global order."""
    drift_p, diff_p, score_p, order = {}, {}, {}, []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("drift."):
            drift_p[name[len("drift."):]] = p
            order.append(name)
        elif name.startswith("diffusion."):
            diff_p[name[len("diffusion."):]] = p
            order.append(name)
        elif name.startswith("score_net."):
            score_p[name[len("score_net."):]] = p
            order.append(name)
    return drift_p, diff_p, score_p, order


def _flatten(jac: Dict[str, Tensor], order_local: List[str], lead_shape: Tuple[int, ...]) -> Tensor:
    """Concatenate a dict of jacobians ``[..., *param.shape]`` into ``[..., P]`` following ``order_local``."""
    parts = [jac[k].reshape(*lead_shape, -1) for k in order_local]
    if not parts:
        return torch.zeros(*lead_shape, 0)
    return torch.cat(parts, dim=-1)


def sensitivity_backward(
    model: NDIM,
    events: Tensor,
    n_prime: int,
    m: int,
    brownian: Tensor,
    cfg: ExperimentConfig,
    kl_weight: Optional[float] = None,
) -> ObjectiveOutput:
    """Populate ``p.grad`` for all trainable parameters with the forward-sensitivity estimate of the
    gradient of ``loss = -mean_rows[ loglik - kl_weight * int u^2 dt ]``.

    Returns an ``ObjectiveOutput`` whose ``loss`` is detached (for logging).
    """
    dt = cfg.dt
    device = brownian.device
    B = events.shape[0]
    R = B * m
    n = int(n_prime)
    kl_weight = float(cfg.train.kl_weight if kl_weight is None else kl_weight)
    z0 = model.initial_state

    drift, diffusion, score_net = model.drift, model.diffusion, model.score_net
    drift_p, diff_p, score_p, order = _split_params(model)
    all_p = {**{f"drift.{k}": v for k, v in drift_p.items()}, **{f"diffusion.{k}": v for k, v in diff_p.items()},
             **{f"score_net.{k}": v for k, v in score_p.items()}}
    P = sum(v.numel() for v in all_p.values())
    detached = {k: v.detach() for k, v in all_p.items()}

    t_grid = torch.arange(n, device=device, dtype=brownian.dtype) * dt
    with torch.no_grad():
        ctx = score_net.prepare(events, t_grid, float(n) * dt, precompute_pool=False)

    # ---------------------------------------------------------------- per-row step function
    def step_fn(params: Dict[str, Tensor], z_raw: Tensor, t: Tensor, inp: Dict[str, Tensor], controlled: Tensor) -> Tensor:
        Z = model.coefficient_state(z_raw)
        dp = {k[len("drift."):]: v for k, v in params.items() if k.startswith("drift.")}
        gp = {k[len("diffusion."):]: v for k, v in params.items() if k.startswith("diffusion.")}
        sp = {k[len("score_net."):]: v for k, v in params.items() if k.startswith("score_net.")}
        b = functional_call(drift, dp, (Z, t))
        sig = functional_call(diffusion, gp, (Z, t))
        s = functional_call(score_net, sp, (Z, inp))
        if cfg.train.max_abs_score is not None:
            s = torch.clamp(s, -float(cfg.train.max_abs_score), float(cfg.train.max_abs_score))
        u = controlled * sig * s
        if cfg.train.max_abs_control is not None:
            u = torch.clamp(u, -float(cfg.train.max_abs_control), float(cfg.train.max_abs_control))
        F = b + sig * u
        return torch.stack([F, sig, u])

    jac_fn = jacrev(step_fn, argnums=(0, 1))

    def in_dims_for(inp: Dict[str, Tensor]) -> Dict[str, Optional[int]]:
        return {k: (None if v.dim() == 0 else 0) for k, v in inp.items()}

    # ---------------------------------------------------------------- pass 1: simulate path (no grad)
    with torch.no_grad():
        z = torch.full((R,), z0, device=device, dtype=brownian.dtype)
        raws, us, bs, sigs = [z], [], [], []
        for k in range(n):
            t = t_grid[k]
            inp = score_net.row_inputs(ctx, k, m)
            Z = model.coefficient_state(z)
            tt = torch.full((R,), float(t), device=device, dtype=z.dtype)
            b = drift(Z, tt)
            sig = diffusion(Z, tt)
            s = score_net.forward(Z, inp)
            if cfg.train.max_abs_score is not None:
                s = torch.clamp(s, -float(cfg.train.max_abs_score), float(cfg.train.max_abs_score))
            u = sig * s
            if cfg.train.max_abs_control is not None:
                u = torch.clamp(u, -float(cfg.train.max_abs_control), float(cfg.train.max_abs_control))
            z = z + (b + sig * u) * dt + sig * brownian[:, k]
            raws.append(z)
            us.append(u)
            bs.append(b)
            sigs.append(sig)
        raw = torch.stack(raws, dim=1)          # [R, n+1]
        u_all = torch.stack(us, dim=1)          # [R, n]
        b_all = torch.stack(bs, dim=1)
        sig_all = torch.stack(sigs, dim=1)

    # ---------------------------------------------------------------- dL/dz, dL/du by autograd on the discretised objective
    raw_leaf = raw.detach().clone().requires_grad_(True)
    u_leaf = u_all.detach().clone().requires_grad_(True)
    Z_leaf = model.to_intensity(raw_leaf)
    loglik_rows = poisson_loglik(Z_leaf, events, m, dt, n, cfg.likelihood)
    kl_raw_rows = (u_leaf ** 2).sum(dim=1) * dt
    L_rows = loglik_rows - kl_weight * kl_raw_rows
    L = L_rows.sum()
    gZ, gu = torch.autograd.grad(L, (raw_leaf, u_leaf))       # [R, n+1], [R, n]

    # ---------------------------------------------------------------- pass 2: sensitivities
    grad_flat = torch.zeros(P, device=device, dtype=brownian.dtype)
    if P > 0:
        J = torch.zeros(R, P, device=device, dtype=brownian.dtype)
        z = raw[:, 0]
        for k in range(n):
            t = t_grid[k]
            inp = dict(score_net.row_inputs(ctx, k, m))   # plain dict: RowInputs is not a registered pytree
            controlled = torch.ones((), device=device, dtype=z.dtype)
            in_dims = (None, 0, None, in_dims_for(inp), None)
            jac_p, jac_z = vmap(jac_fn, in_dims=in_dims)(detached, z, t, inp, controlled)
            # jac_p: dict name -> [R, 3, *shape]; jac_z: [R, 3]
            Jp = _flatten(jac_p, list(detached.keys()), (R, 3))    # [R, 3, P]
            dF_dp, dG_dp, du_dp = Jp[:, 0], Jp[:, 1], Jp[:, 2]
            dF_dz, dG_dz, du_dz = jac_z[:, 0], jac_z[:, 1], jac_z[:, 2]
            # energy term at step k uses J_k (state before the step)
            grad_flat += (gu[:, k, None] * (du_dz[:, None] * J + du_dp)).sum(dim=0)
            J = J + (dF_dz[:, None] * J + dF_dp) * dt + (dG_dz[:, None] * J + dG_dp) * brownian[:, k, None]
            grad_flat += (gZ[:, k + 1, None] * J).sum(dim=0)
            z = raw[:, k + 1]
        grad_flat = -grad_flat / R      # loss = -mean over rows

    # ---------------------------------------------------------------- write gradients
    offset = 0
    for name in order:
        p = all_p[name]
        g = grad_flat[offset : offset + p.numel()].view_as(p)
        p.grad = g.detach().clone() if p.grad is None else p.grad + g.detach()
        offset += p.numel()

    with torch.no_grad():
        Zpos = model.to_intensity(raw)
        sim = SimOutput(raw=raw, Z=Zpos, b=b_all, sigma=sig_all, s=torch.zeros_like(u_all), u=u_all, n_prime=n)
        sim._dt = dt
        return ObjectiveOutput(
            loss=-(L_rows.mean()).detach(),
            loglik=loglik_rows.detach().view(B, m).mean(dim=1),
            kl=(0.5 * kl_raw_rows.detach()).view(B, m).mean(dim=1),
            elbo=(loglik_rows - 0.5 * kl_raw_rows).detach().view(B, m).mean(dim=1),
            sim=sim,
        )
