"""Euler–Maruyama simulation of the prior SDE (Eq. 1) and the drift-corrected posterior SDE (Eq. 3).

State layout: ``[R, n+1]`` with ``R = B*m`` rows (row ``b*m + j`` = path ``j`` of observation ``b``).
With a link, all coefficients and corrections see the real latent state X; only the
likelihood/output sees Z=g(X). Without a link the state is the intensity itself and the
coefficients take its positive part: the intensity parametrization (``model.state_space: intensity``),
also used to simulate the synthetic truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from .models.common import positive_part
from .models.posterior import PosteriorCorrection
from .links import IntensityLink

Tensor = torch.Tensor


def brownian_increments(n_rows: int, n_steps: int, dt: float, device: torch.device, generator: Optional[torch.Generator] = None) -> Tensor:
    """``sqrt(dt) * xi``, shape ``[n_rows, n_steps]``."""
    if generator is not None and generator.device != device:
        xi = torch.randn(n_rows, n_steps, generator=generator).to(device)
    else:
        xi = torch.randn(n_rows, n_steps, device=device, generator=generator)
    return xi * (float(dt) ** 0.5)


def _initial_state(z0: Union[float, Tensor], link: Optional[IntensityLink], initial_raw: Optional[Union[float, Tensor]],
                   R: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Starting value of the Euler state for ``R`` rows: ``initial_raw`` itself when given (a latent state), otherwise
    the intensity ``z0`` (a number, or one value per row) mapped through the inverse link."""
    if initial_raw is not None:
        return torch.as_tensor(initial_raw, device=device, dtype=dtype).expand(R).clone()
    if isinstance(z0, torch.Tensor):
        z = z0.to(device=device, dtype=dtype).expand(R).clone()
        return link.inverse(z) if link is not None else z
    return torch.full((R,), link.inverse_scalar(z0) if link is not None else float(z0), device=device, dtype=dtype)


@dataclass
class SimOutput:
    raw: Tensor                    # [R, n+1] latent X (unprojected Euler state)
    Z: Tensor                      # [R, n+1] linked intensity g(X) (intensity state: its positive part)
    b: Tensor                      # [R, n]   prior drift used in the update
    sigma: Tensor                  # [R, n]   diffusion coefficient
    s: Tensor                      # [R, n]   score (zero for k >= n')
    u: Tensor                      # [R, n]   Brownian-space control (zero for k >= n')
    n_prime: int
    n_burn: int = 0                # leading steps that lie before the observation window (burn-in); set by the model

    @property
    def kl_integral(self) -> Tensor:
        """``int_0^{T'} u^2 dt`` per row (multiply by 1/2 for the KL)."""
        return (self.u ** 2).sum(dim=1) * self._dt

    _dt: float = 0.0


@torch.no_grad()
def simulate_prior(
    drift: Callable[[Tensor, Tensor], Tensor],
    diffusion: Callable[[Tensor, Tensor], Tensor],
    z0: Union[float, Tensor],
    n_steps: int,
    dt: float,
    brownian: Tensor,
    positivity: str = "full_truncation",
    softplus_scale: float = 0.5,
    link: Optional[IntensityLink] = None,
    return_raw: bool = False,
    t0: float = 0.0,
    initial_raw: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    """Intensity paths [R,n_steps+1] on the grid ``t0 + k dt``; with a link evolve X from g^{-1}(z0).

    ``z0`` is a number or one initial intensity per row. ``initial_raw`` starts the Euler state at that latent value
    instead (the burn-in start). return_raw returns the unprojected Euler states instead of intensities.
    """
    R = brownian.shape[0]
    device = brownian.device
    z = _initial_state(z0, link, initial_raw, R, device, brownian.dtype)
    out = torch.empty(R, n_steps + 1, device=device, dtype=brownian.dtype)
    def intensity(state):
        return link(state) if link is not None else positive_part(state, positivity, softplus_scale)
    out[:, 0] = z if return_raw else intensity(z)
    for k in range(n_steps):
        t = torch.full((R,), float(t0) + k * float(dt), device=device, dtype=brownian.dtype)
        Z = z if link is not None else intensity(z)
        z = z + drift(Z, t) * float(dt) + diffusion(Z, t) * brownian[:, k]
        out[:, k + 1] = z if return_raw else intensity(z)
    return out


def simulate_posterior(
    drift: nn.Module,
    diffusion: nn.Module,
    score_net: PosteriorCorrection,
    ctx: Any,
    z0: Union[float, Tensor],
    n_steps: int,
    n_prime: int,
    dt: float,
    brownian: Tensor,
    positivity: str = "full_truncation",
    softplus_scale: float = 0.5,
    max_abs_score: Optional[float] = None,
    max_abs_control: Optional[float] = None,
    link: Optional[IntensityLink] = None,
    t0: float = 0.0,
    initial_raw: Optional[Union[float, Tensor]] = None,
) -> SimOutput:
    """Simulate ``dX = [b + 1{k<n'} sigma u] dt + sigma dB`` on the grid ``t0 + k dt`` and output Z=g(X).

    The first ``n_prime`` steps are controlled, whatever ``t0`` is: with a burn-in the caller counts its steps in
    ``n_prime`` and ``n_steps``. Without a link the state is the intensity and the coefficients take its positive part.
    """
    if n_steps < n_prime:
        raise ValueError("n_steps must be >= n_prime")
    if brownian.shape[1] < n_steps:
        raise ValueError(f"brownian has {brownian.shape[1]} steps < n_steps={n_steps}")
    R = brownian.shape[0]
    device = brownian.device
    dtype = brownian.dtype

    z = _initial_state(z0, link, initial_raw, R, device, dtype)
    def intensity(state):
        return link(state) if link is not None else positive_part(state, positivity, softplus_scale)
    raws = [z]
    Zs = [intensity(z)]
    bs, sigmas, ss, us = [], [], [], []
    zero = torch.zeros(R, device=device, dtype=dtype)
    for k in range(n_steps):
        Z = z if link is not None else Zs[-1]
        t = torch.full((R,), float(t0) + k * float(dt), device=device, dtype=dtype)
        b = drift(Z, t)
        sig = diffusion(Z, t)
        if k < n_prime:
            s = score_net.score(ctx, k, Z)
            if max_abs_score is not None:
                s = torch.clamp(s, -float(max_abs_score), float(max_abs_score))
            u = sig * s
            if max_abs_control is not None:
                u = torch.clamp(u, -float(max_abs_control), float(max_abs_control))
            corr = sig * u
        else:
            s, u, corr = zero, zero, zero
        z = z + (b + corr) * float(dt) + sig * brownian[:, k]
        raws.append(z)
        Zs.append(intensity(z))
        bs.append(b)
        sigmas.append(sig)
        ss.append(s)
        us.append(u)
    out = SimOutput(
        raw=torch.stack(raws, dim=1),
        Z=torch.stack(Zs, dim=1),
        b=torch.stack(bs, dim=1),
        sigma=torch.stack(sigmas, dim=1),
        s=torch.stack(ss, dim=1),
        u=torch.stack(us, dim=1),
        n_prime=int(n_prime),
    )
    out._dt = float(dt)
    return out
