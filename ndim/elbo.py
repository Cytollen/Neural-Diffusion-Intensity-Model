"""Monte-Carlo ELBO (paper Eq. 6) and the training objective for one mini-batch.

For ``B`` observations x ``m`` common-random-number paths on horizon ``T' = n' dt``::

    loglik_r = sum_i log Z_{tau_i} - int_0^{T'} Z dt          (per path r)
    kl_r     = int_0^{T'} u^2 dt                              (raw Girsanov integral; KL = kl/2)
    elbo_b   = mean_j [ loglik_{b,j} - kl_{b,j}/2 ]

Every parameter (prior drift, diffusion, posterior network) receives the pathwise gradient of this
objective through the simulated posterior paths (paper Sec. 3.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .config import ExperimentConfig, resolved_diffusion_kind
from .likelihood import poisson_loglik
from .model import NDIM
from .sde import SimOutput

Tensor = torch.Tensor


@dataclass
class ObjectiveOutput:
    loss: Tensor                 # scalar to minimise
    loglik: Tensor               # [B] per-observation Monte-Carlo estimates
    kl: Tensor                   # [B]  (1/2 int u^2)
    elbo: Tensor                 # [B]
    sim: SimOutput

    def summary(self) -> Dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "loglik": float(self.loglik.detach().mean()),
            "kl": float(self.kl.detach().mean()),
            "elbo": float(self.elbo.detach().mean()),
        }


def compute_objective(
    model: NDIM,
    events: Tensor,
    n_prime: int,
    m: int,
    brownian: Tensor,
    cfg: ExperimentConfig,
    kl_weight: Optional[float] = None,
    n_steps: Optional[int] = None,
) -> ObjectiveOutput:
    """Simulate the posterior and evaluate the objective for ``events [B, K]``.

    ``brownian`` is ``[B*m, n_burn + n_steps]`` (``n_steps`` defaults to ``n_prime``; ``n_burn = model.n_burn`` is
    0 without a burn-in). ``kl_weight`` multiplies the
    raw integral ``int u^2 dt`` in the *training* objective (0.5 = ELBO); the reported ``elbo`` always
    uses 1/2.
    """
    dt = cfg.dt
    B = events.shape[0]
    kl_weight = float(cfg.train.kl_weight if kl_weight is None else kl_weight)

    # with a burn-in the likelihood covers the window only, the control energy every controlled step
    sim = model.simulate_posterior(events, n_prime, brownian, n_steps,
                                   max_abs_score=cfg.train.max_abs_score, max_abs_control=cfg.train.max_abs_control)
    nb = sim.n_burn
    loglik_rows = poisson_loglik(sim.Z[:, nb:], events, m, dt, n_prime, cfg.likelihood)   # [B*m]
    kl_raw_rows = (sim.u[:, : nb + n_prime] ** 2).sum(dim=1) * dt                         # [B*m]
    elbo_rows = loglik_rows - 0.5 * kl_raw_rows
    train_rows = loglik_rows - kl_weight * kl_raw_rows
    loss = -train_rows.mean()

    return ObjectiveOutput(
        loss=loss,
        loglik=loglik_rows.view(B, m).mean(dim=1),
        kl=(0.5 * kl_raw_rows).view(B, m).mean(dim=1),
        elbo=elbo_rows.view(B, m).mean(dim=1),
        sim=sim,
    )


@torch.no_grad()
def evaluate_elbo(
    model: NDIM,
    events: Tensor,
    n_prime: int,
    m: int,
    cfg: ExperimentConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    chunk_size: int = 32,
    bounds: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Monte-Carlo ELBO / log-lik / KL on a set of observations (mean over observations), in chunks.

    A dict passed as ``bounds`` is filled with what the two bounds of the state-space parametrization did on the
    simulated posterior paths (``bounds_summary``)."""
    from .sde import brownian_increments

    model.eval()
    n_obs = events.shape[0]
    tot = {"loglik": 0.0, "kl": 0.0, "elbo": 0.0}
    count = {"steps": 0, "control": 0, "diffusion": 0, "max_u": 0.0}
    for start in range(0, n_obs, chunk_size):
        ev = events[start : start + chunk_size].to(device)
        Bc = ev.shape[0]
        W = brownian_increments(Bc * m, model.n_burn + n_prime, cfg.dt, device, generator)
        out = compute_objective(model, ev, n_prime, m, W, cfg)
        tot["loglik"] += float(out.loglik.sum())
        tot["kl"] += float(out.kl.sum())
        tot["elbo"] += float(out.elbo.sum())
        if bounds is not None:
            controlled = out.sim.n_burn + n_prime
            u = out.sim.u[:, :controlled].abs()
            count["steps"] += int(u.numel())
            count["max_u"] = max(count["max_u"], float(u.max()))
            if cfg.train.max_abs_control is not None:
                count["control"] += int((u >= float(cfg.train.max_abs_control)).sum())
            count["diffusion"] += int(model.diffusion.bound_active(out.sim.raw[:, :controlled]).sum())
    if bounds is not None:
        bounds.update(bounds_summary(cfg, count["steps"], count["control"], count["diffusion"], count["max_u"]))
    return {k: v / n_obs for k, v in tot.items()}


def bounds_summary(cfg: ExperimentConfig, n_steps: int, n_control: int, n_diffusion: int, max_abs_u: float) -> Dict[str, Any]:
    """What the bound on the control (``train.max_abs_control``) and the bound on the CIR diffusion of a latent state
    (``model.diffusion.max_step_std``) did on ``n_steps`` simulated steps: the largest ``|u|``, and the share of the
    steps on which each bound held (None for a bound that is not set). On a fitted model both shares should be 0."""
    n = max(int(n_steps), 1)
    latent_cir = resolved_diffusion_kind(cfg.model) in {"cir_exponential", "cir_link"} and cfg.model.diffusion.max_step_std is not None
    return {"largest_abs_control": float(max_abs_u), "control_bound": cfg.train.max_abs_control,
            "control_bound_share": None if cfg.train.max_abs_control is None else n_control / n,
            "diffusion_bound_share": n_diffusion / n if latent_cir else None, "n_steps": int(n_steps)}
