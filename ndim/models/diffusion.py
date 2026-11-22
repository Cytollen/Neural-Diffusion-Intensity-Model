"""Diffusion coefficient of the simulated state.

* ``mlp`` (default): a learned positive network, for when no structure is assumed.
* ``constant``: a positive scalar.
* ``cir``: the intensity diffusion ``eta * sqrt(Z)``, built in the coordinates of the model's state space
  (``CIRSqrtDiffusion`` for intensity states, ``CIRExponentialDiffusion`` for ``X = log Z``,
  ``CIRLinkDiffusion`` for a latent state with another link). For a latent state the coefficient is bounded where
  the intensity is very low (``model.diffusion.max_step_std``, see ``CIRExponentialDiffusion``).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import DiffusionConfig, required_scale
from ..links import IntensityLink
from .common import MLP

Tensor = torch.Tensor


class Diffusion(nn.Module):
    """Coefficient interface: forward(state, time), trainable, named_scalars(), bound_active(state)."""
    trainable: bool = False

    def named_scalars(self) -> Dict[str, float]:
        return {}

    def bound_active(self, state: Tensor) -> Tensor:
        """Where a bound of the coefficient holds (boolean, like ``state``). Only the CIR diffusion of a latent state
        has one (``model.diffusion.max_step_std``)."""
        return torch.zeros_like(state, dtype=torch.bool)


class CIRSqrtDiffusion(Diffusion):
    """``sigma(z, t) = eta * sqrt(relu(z))`` with ``eta = exp(log_eta)`` optionally trainable.

    The CIR diffusion for intensity states. The ``sqrt(z)`` structure is assumed known; the scale ``eta``
    is fixed or learned.
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.log_eta = nn.Parameter(torch.log(torch.tensor(float(cfg.eta_init))), requires_grad=bool(cfg.trainable))
        self.trainable = bool(cfg.trainable)

    @property
    def eta(self) -> Tensor:
        return torch.exp(self.log_eta)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        # z is already the positive part; relu guards against tiny negatives from other positivity modes
        return self.eta * torch.sqrt(torch.relu(z) + 0.0)

    def named_scalars(self) -> Dict[str, float]:
        return {"eta": float(self.eta.detach())}


class ConstantDiffusion(Diffusion):
    """Positive, optionally learned constant noise scale, defined on all of R."""
    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.log_eta = nn.Parameter(torch.log(torch.tensor(float(cfg.eta_init))), requires_grad=cfg.trainable)
        self.trainable = bool(cfg.trainable)

    @property
    def eta(self) -> Tensor:
        return torch.exp(self.log_eta)

    def named_scalars(self) -> Dict[str, float]:
        return {"eta": float(self.eta.detach())}

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        return self.eta.expand_as(z)


class CIRExponentialDiffusion(ConstantDiffusion):
    """``sigma_X(x) = min(eta * exp(-x/2), cap)`` for the intensity link ``lambda = exp(x)``.

    Ito's diffusion transformation gives ``sigma_lambda = exp(x) * sigma_X(x) = eta * sqrt(lambda)``, so eta has the
    original CIR intensity-scale meaning.

    ``eta * exp(-x/2)`` grows without bound as the intensity falls. On an Euler grid one step from a low intensity
    can then move the state arbitrarily far, and the posterior correction ``sigma_X^2 s`` grows with it: the paths
    of an untrained model run off. ``cap = max_step_std / sqrt(dt)`` bounds the standard deviation of the noise that
    one Euler step adds to ``X = log lambda`` by ``max_step_std`` (``model.diffusion.max_step_std``, 1 by default).
    The coefficient is then the CIR form above the intensity ``eta^2 dt / max_step_std^2`` and constant below it,
    and the bound disappears as the step goes to 0. It plays the part that the truncation at 0 plays for the Euler
    scheme of the intensity itself. ``cap=None`` is the unbounded form. The state is not projected.
    """

    def __init__(self, cfg: DiffusionConfig, cap: Optional[float] = None) -> None:
        super().__init__(cfg)
        self.log_cap = None if cap is None else math.log(float(cap))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        log_sigma = self.log_eta - 0.5 * x
        if self.log_cap is not None:   # bounded before the exponential: no overflow, and no gradient where the bound holds
            log_sigma = torch.clamp(log_sigma, max=self.log_cap)
        return torch.exp(log_sigma)

    def bound_active(self, x: Tensor) -> Tensor:
        """States whose intensity lies below ``eta^2 dt / max_step_std^2``, where the coefficient is held at ``cap``."""
        return super().bound_active(x) if self.log_cap is None else (self.log_eta - 0.5 * x).detach() > self.log_cap


class CIRLinkDiffusion(ConstantDiffusion):
    """``sigma_X(x) = eta * sqrt(g(x)) / g'(x)`` for a latent state with intensity link ``lambda = g(x)``.

    Ito's diffusion transformation gives ``sigma_lambda = g'(x) * sigma_X(x) = eta * sqrt(lambda)``: the CIR
    intensity diffusion for any link. ``CIRExponentialDiffusion`` is its closed form for ``g = exp``, and the bound
    is the same: the diffusion of the intensity per unit of intensity, ``eta / sqrt(lambda)``, is at most ``cap``,
    so the coefficient is the CIR form above the intensity ``eta^2 dt / max_step_std^2``.
    """

    def __init__(self, cfg: DiffusionConfig, link: IntensityLink, cap: Optional[float] = None) -> None:
        super().__init__(cfg)
        self.link = link
        self.log_cap = None if cap is None else math.log(float(cap))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # sqrt(g) / g' written as (eta / sqrt(g)) / (g'/g): the links evaluate log g and g'/g stably for very negative states
        log_relative = self.log_eta - 0.5 * self.link.log_intensity(x)
        if self.log_cap is not None:
            log_relative = torch.clamp(log_relative, max=self.log_cap)
        return torch.exp(log_relative) / self.link.log_derivative(x)

    def bound_active(self, x: Tensor) -> Tensor:
        if self.log_cap is None:
            return super().bound_active(x)
        return (self.log_eta - 0.5 * self.link.log_intensity(x)).detach() > self.log_cap


class MLPDiffusion(Diffusion):
    """Learned state/time-dependent diffusion, parametrised by the Cholesky factor of ``a = sigma sigma^T``.

    The factor is lower triangular with a positive diagonal, which makes it the unique square root of ``a``.
    The state is one-dimensional here, so the factor is the positive scalar
    ``sigma = min_scale + eta_init * softplus(net) / log 2`` (equal to ``min_scale + eta_init`` at initialisation).
    """
    def __init__(self, cfg: DiffusionConfig, T: float) -> None:
        super().__init__()
        self.trainable = bool(cfg.trainable)
        self.time_input = cfg.time_input
        self.z_scale = required_scale(cfg.z_scale, "model.diffusion.z_scale")
        self.T, self.min_scale = float(T), cfg.min_scale
        self.net = MLP(2 if cfg.time_input else 1, cfg.hidden, 1, cfg.activation, 0.01)
        self.register_buffer("initial_scale", torch.tensor(float(cfg.eta_init)))
        for p in self.parameters():
            p.requires_grad_(self.trainable)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        cols = [z / self.z_scale]
        if self.time_input:
            cols.append(torch.broadcast_to(t, z.shape) / self.T)
        raw = self.net(torch.stack(cols, dim=-1)).squeeze(-1)
        return self.min_scale + self.initial_scale * torch.nn.functional.softplus(raw) / 0.6931471805599453

    def named_scalars(self) -> Dict[str, float]:
        return {}


def relative_diffusion_cap(cfg: DiffusionConfig, dt: Optional[float]) -> Optional[float]:
    """Bound on the CIR diffusion of a latent state, ``max_step_std / sqrt(dt)`` (None: no bound).

    It is the largest diffusion of the intensity per unit of intensity, which for ``X = log Z`` is the diffusion of X.
    """
    if cfg.max_step_std is None:
        return None
    if dt is None or dt <= 0:
        raise ValueError("model.diffusion.max_step_std bounds the noise of one Euler step, so the CIR diffusion of a latent "
                         "state needs the step: pass dt (NDIM does), or set max_step_std to null for the unbounded form")
    return float(cfg.max_step_std) / math.sqrt(float(dt))


def build_diffusion(cfg: DiffusionConfig, T: float = 1.0, kind: Optional[str] = None,
                    link: Optional[IntensityLink] = None, dt: Optional[float] = None) -> Diffusion:
    """Build the diffusion module. ``kind`` overrides ``cfg.kind``.

    ``cir`` has no coordinates of its own: pass ``kind=resolved_diffusion_kind(model_cfg)`` (as ``NDIM`` does)
    so that it becomes ``cir_sqrt``, ``cir_exponential`` or ``cir_link`` according to the state space and link.
    ``cir_link`` needs the ``link`` module, and the two forms for a latent state need the Euler step ``dt`` for
    their bound (``relative_diffusion_cap``).
    """
    kind = cfg.kind if kind is None else kind
    if cfg.eta_init == "auto":
        raise ValueError("model.diffusion.eta_init is to be estimated from the observations: call "
                         "ndim.cir_eta.resolve_eta(cfg, train_events) first (ndim.train.train does), or set it to a number")
    if kind == "cir":
        raise ValueError("diffusion kind 'cir' follows the state space; pass kind=resolved_diffusion_kind(model_cfg)")
    if kind == "cir_sqrt":
        return CIRSqrtDiffusion(cfg)
    if kind == "constant":
        return ConstantDiffusion(cfg)
    if kind == "cir_exponential":
        return CIRExponentialDiffusion(cfg, relative_diffusion_cap(cfg, dt))
    if kind == "cir_link":
        if link is None:
            raise ValueError("diffusion kind 'cir_link' needs the intensity link")
        return CIRLinkDiffusion(cfg, link, relative_diffusion_cap(cfg, dt))
    if kind == "mlp":
        return MLPDiffusion(cfg, T)
    raise ValueError(f"unknown diffusion kind {kind}")
