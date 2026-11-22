"""Prior drift ``b(state, t)``: an MLP on the state, or the parametric CIR drift ``kappa (theta - Z)`` on the intensity."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DriftConfig, required_scale
from .common import MLP

Tensor = torch.Tensor


class Drift(nn.Module):
    """Interface: ``forward(z, t) -> b`` with ``z, t`` broadcastable tensors (same shape output)."""

    def forward(self, z: Tensor, t: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def named_scalars(self) -> Dict[str, float]:
        """Interpretable scalar parameters for logging (empty for MLPs)."""
        return {}


class MLPDrift(Drift):
    """``b(z, t) = drift_scale * MLP([z / z_scale, t / T])``.

    Inputs are normalised to O(1) and the output is rescaled so the network works in a
    well-conditioned range (the true CIR drift takes values of order tens).
    """

    def __init__(self, cfg: DriftConfig, T: float) -> None:
        super().__init__()
        self.time_input = bool(cfg.time_input)
        self.z_scale = required_scale(cfg.z_scale, "model.drift.z_scale")
        self.t_scale = float(T)
        self.drift_scale = float(cfg.drift_scale)
        in_dim = 2 if self.time_input else 1
        self.net = MLP(in_dim, cfg.hidden, 1, cfg.activation, cfg.init_output_std)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        zn = (z / self.z_scale).unsqueeze(-1)
        if self.time_input:
            tn = torch.broadcast_to(t, z.shape).to(z.dtype) / self.t_scale
            x = torch.cat([zn, tn.unsqueeze(-1)], dim=-1)
        else:
            x = zn
        return self.drift_scale * self.net(x).squeeze(-1)


class CIRDrift(Drift):
    """Parametric mean-reverting drift ``b(z) = kappa (theta - z)`` of the CIR intensity, ``kappa = softplus(raw)``.

    This is the paper's ``theta_1 (theta_2 - z)`` (Fig. 11). It is a drift of the intensity ``Z``, so it is used with
    ``model.state_space: intensity`` (``validate_config`` rejects it for a latent state).

    ``theta`` is an intensity level, so it is trained in the units in which the networks see the state: the parameter
    is ``theta_unit = theta / z_scale``. An optimiser step of size ``lr`` then changes ``theta`` by ``z_scale * lr``,
    and the learning rate of the networks suits it (a raw ``theta`` of order 80 needs a rate about 40 times larger).
    ``cfg.theta_init`` and the reported ``theta`` stay in intensity units.
    """

    def __init__(self, cfg: DriftConfig) -> None:
        super().__init__()
        self.z_scale = required_scale(cfg.z_scale, "model.drift.z_scale")
        raw_kappa = torch.log(torch.expm1(torch.tensor(float(cfg.kappa_init))))
        self.raw_kappa = nn.Parameter(raw_kappa, requires_grad=bool(cfg.learn_kappa))
        self.theta_unit = nn.Parameter(torch.tensor(float(cfg.theta_init) / self.z_scale), requires_grad=bool(cfg.learn_theta))

    @property
    def kappa(self) -> Tensor:
        return F.softplus(self.raw_kappa)

    @property
    def theta(self) -> Tensor:
        """The level the drift pulls toward, in intensity units."""
        return self.z_scale * self.theta_unit

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        return self.kappa * (self.theta - z)

    def named_scalars(self) -> Dict[str, float]:
        return {"kappa": float(self.kappa.detach()), "theta": float(self.theta.detach())}


def build_drift(cfg: DriftConfig, T: float) -> Drift:
    if cfg.kind == "mlp":
        return MLPDrift(cfg, T)
    if cfg.kind == "cir":
        return CIRDrift(cfg)
    raise ValueError(f"unknown drift kind {cfg.kind!r}: use mlp or cir")
