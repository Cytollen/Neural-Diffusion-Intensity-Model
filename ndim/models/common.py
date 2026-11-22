"""Shared building blocks for the networks."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor

_ACTS = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
}


def make_activation(name: str) -> nn.Module:
    try:
        return _ACTS[name.lower()]()
    except KeyError as e:
        raise ValueError(f"unknown activation '{name}'. Options: {sorted(_ACTS)}") from e


class MLP(nn.Module):
    """Plain fully-connected network ``in -> hidden... -> out``.

    Hidden layers use the requested activation; the output layer is linear and (optionally)
    initialised with a small std so the network starts close to zero.
    """

    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int, activation: str = "silu", init_output_std: float = 0.01) -> None:
        super().__init__()
        layers = []
        d = int(in_dim)
        for h in hidden:
            lin = nn.Linear(d, int(h))
            nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)  # PyTorch default; keep explicit
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(make_activation(activation))
            d = int(h)
        out = nn.Linear(d, int(out_dim))
        if init_output_std is not None and init_output_std > 0:
            nn.init.normal_(out.weight, std=float(init_output_std))
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def positive_part(z: Tensor, mode: str = "full_truncation", softplus_scale: float = 0.5) -> Tensor:
    """Map the raw Euler state to a non-negative intensity.

    * ``full_truncation``: ``relu(z)`` (Lord–Koekkoek–van Dijk scheme for square-root diffusions).
    * ``reflect``: ``|z|``.
    * ``softplus``: ``s * softplus(z / s)`` (smooth, ~z for z >> s).
    """
    if mode == "full_truncation":
        return torch.relu(z)
    if mode == "reflect":
        return torch.abs(z)
    if mode == "softplus":
        s = float(softplus_scale)
        return s * torch.nn.functional.softplus(z / s)
    raise ValueError(f"unknown positivity mode {mode}")
