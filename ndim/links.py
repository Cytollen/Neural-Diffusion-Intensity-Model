"""Prespecified C2 bijections from real latent states to positive intensities."""
from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class IntensityLink(nn.Module):
    """A fixed link, its inverse, and the derivatives required by Ito's formula."""

    def inverse(self, intensity: Tensor) -> Tensor:
        raise NotImplementedError

    def first(self, state: Tensor) -> Tensor:
        raise NotImplementedError

    def second(self, state: Tensor) -> Tensor:
        raise NotImplementedError

    def log_derivative(self, state: Tensor) -> Tensor:
        """d log(g(x)) / dx, used by analytic posterior jump terms."""
        return self.first(state) / self(state)

    def log_intensity(self, state: Tensor) -> Tensor:
        """log(g(x)), used by the CIR diffusion of a latent state."""
        return torch.log(self(state))

    def inverse_scalar(self, intensity: float) -> float:
        if intensity <= 0:
            raise ValueError("Initial intensity must be strictly positive for a real-state link")
        return float(self.inverse(torch.tensor(intensity, dtype=torch.float64)))


class ExponentialLink(IntensityLink):
    def forward(self, state: Tensor) -> Tensor:
        return torch.exp(state)

    def inverse(self, intensity: Tensor) -> Tensor:
        return torch.log(intensity)

    def first(self, state: Tensor) -> Tensor:
        return self(state)

    def second(self, state: Tensor) -> Tensor:
        return self(state)

    def log_derivative(self, state: Tensor) -> Tensor:
        return torch.ones_like(state)

    def log_intensity(self, state: Tensor) -> Tensor:
        return state


class SoftplusLink(IntensityLink):
    def forward(self, state: Tensor) -> Tensor:
        # logaddexp implements the smooth function without a thresholded linear branch.
        return torch.logaddexp(state, torch.zeros_like(state))

    def inverse(self, intensity: Tensor) -> Tensor:
        return intensity + torch.log(-torch.expm1(-intensity))

    def first(self, state: Tensor) -> Tensor:
        return torch.sigmoid(state)

    def second(self, state: Tensor) -> Tensor:
        p = self.first(state)
        return p * (1 - p)

    def log_derivative(self, state: Tensor) -> Tensor:
        # At very negative states g(x) ~ exp(x), so this ratio tends to one.
        safe = torch.clamp(state, min=-20.0)
        ratio = torch.sigmoid(safe) / self(safe)
        return torch.where(state < -20.0, torch.ones_like(state), ratio)

    def log_intensity(self, state: Tensor) -> Tensor:
        # At very negative states g(x) ~ exp(x), whose logarithm is x; exp(x) itself underflows there.
        safe = torch.clamp(state, min=-30.0)
        return torch.where(state < -30.0, state, torch.log(self(safe)))


def build_link(kind: str = "exponential") -> IntensityLink:
    if kind in {"exponential", "exp"}:
        return ExponentialLink()
    if kind == "softplus":
        return SoftplusLink()
    raise ValueError(f"Unknown intensity link {kind!r}; use exponential or softplus")


def intensity_coefficients(link: IntensityLink, drift, diffusion, intensity: Tensor, time: Tensor):
    """Evaluate the Ito-transformed prior coefficients at z > 0, x = g^{-1}(z).

    b_Z = g'(x)b_X(x,t) + g''(x)sigma_X(x,t)^2/2;
    sigma_Z = g'(x)sigma_X(x,t). No clipping or projection is applied.
    """
    state = link.inverse(intensity)
    sigma = diffusion(state, time)
    first = link.first(state)
    return first * drift(state, time) + 0.5 * link.second(state) * sigma.square(), first * sigma
