"""Container tying prior drift, diffusion and posterior score network together, plus checkpoint I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import (ExperimentConfig, config_from_dict, model_x_start, model_z0, resolve_input_scales, resolved_diffusion_kind,
                     validate_config)
from .models.diffusion import Diffusion, build_diffusion
from .models.drift import Drift, build_drift
from .models.posterior import PosteriorCorrection, build_posterior
from .links import build_link, intensity_coefficients
from .models.common import positive_part
from .sde import SimOutput, brownian_increments, simulate_posterior, simulate_prior

Tensor = torch.Tensor


def _generator(seed: Optional[int]) -> Optional[torch.Generator]:
    return None if seed is None else torch.Generator().manual_seed(int(seed))


class NDIM(nn.Module):
    """Neural Diffusion Intensity Model: ``(b_theta, sigma_eta, s_beta)``."""

    def __init__(self, cfg: ExperimentConfig, posterior_correction: Optional[PosteriorCorrection] = None) -> None:
        super().__init__()
        resolve_input_scales(cfg.model)   # scales left as null follow the state space (no-op for a loaded config)
        validate_config(cfg)
        self.cfg = cfg
        T = float(cfg.data.T)
        link = build_link(cfg.model.link)
        self.drift: Drift = build_drift(cfg.model.drift, T)
        self.diffusion: Diffusion = build_diffusion(cfg.model.diffusion, T, resolved_diffusion_kind(cfg.model), link=link,
                                                    dt=float(cfg.dt))
        self.link = link
        self.score_net = posterior_correction if posterior_correction is not None else build_posterior(
            cfg.model.posterior, T, self.link if self.real_states else None)

    @property
    def real_states(self) -> bool:
        return self.cfg.model.state_space == "real"

    @property
    def posterior_correction(self) -> PosteriorCorrection:
        return self.score_net

    @property
    def n_burn(self) -> int:
        """Euler steps of the burn-in ``[-burn_in, 0)`` that precede the observation window (0 without a burn-in)."""
        return int(self.cfg.burn_steps)

    @property
    def t_start(self) -> float:
        """Time at which the diffusion of the model starts: ``-burn_in`` (0 without a burn-in)."""
        return -self.n_burn * float(self.cfg.dt)

    @property
    def initial_state(self) -> float:
        """Euler state at ``t_start``: the latent ``model.x_start`` with a burn-in, otherwise the fixed initial
        intensity of the model in the coordinates of its state."""
        if self.n_burn > 0:
            return model_x_start(self.cfg)
        z0 = model_z0(self.cfg)
        return self.link.inverse_scalar(z0) if self.real_states else z0

    def _start_kwargs(self) -> Dict[str, Any]:
        """Where and when the simulators start, plus the state-space settings (arguments of ``ndim.sde``)."""
        return {"z0": 0.0, "initial_raw": self.initial_state, "t0": self.t_start, **self.sde_kwargs}

    def simulate_posterior(self, events: Tensor, n_prime: int, brownian: Tensor, n_steps: Optional[int] = None,
                           max_abs_score: Optional[float] = None, max_abs_control: Optional[float] = None) -> SimOutput:
        """Posterior paths for ``events [B, K]`` conditioned on ``(0, n' dt]``.

        The grid starts at ``t_start``: the ``n_burn`` burn-in steps, then ``n_steps`` steps of the observation window
        (default ``n_prime``). The correction is on for the burn-in and for the first ``n_prime`` window steps, so
        ``brownian`` is ``[B * m, n_burn + n_steps]``. The returned paths have ``n_burn + n_steps + 1`` points and
        ``sim.n_burn`` says where the window begins (``sim.Z[:, sim.n_burn:]``).
        """
        nb, dt = self.n_burn, float(self.cfg.dt)
        n_steps = int(n_prime if n_steps is None else n_steps)
        if brownian.shape[1] < nb + n_steps:
            raise ValueError(f"brownian has {brownian.shape[1]} steps but the model needs {nb} burn-in + {n_steps} window steps")
        t_grid = self.t_start + torch.arange(nb + int(n_prime), device=brownian.device, dtype=brownian.dtype) * dt
        ctx = self.score_net.prepare(events, t_grid, float(n_prime) * dt)
        sim = simulate_posterior(self.drift, self.diffusion, self.score_net, ctx, n_steps=nb + n_steps, n_prime=nb + int(n_prime),
                                 dt=dt, brownian=brownian, max_abs_score=max_abs_score, max_abs_control=max_abs_control,
                                 **self._start_kwargs())
        sim.n_burn = nb
        return sim

    @torch.no_grad()
    def simulate_prior(self, brownian: Tensor, n_steps: Optional[int] = None, full: bool = False) -> Tensor:
        """Intensity paths of the learned prior for ``brownian [R, n_burn + n_steps]`` (``n_steps`` defaults to the
        grid of the config): the window ``[R, n_steps + 1]`` starting at ``t = 0``, or with ``full`` the whole path from
        ``t_start`` (``[R, n_burn + n_steps + 1]``)."""
        nb = self.n_burn
        n_steps = int(self.cfg.data.n_steps if n_steps is None else n_steps)
        if brownian.shape[1] < nb + n_steps:
            raise ValueError(f"brownian has {brownian.shape[1]} steps but the model needs {nb} burn-in + {n_steps} window steps")
        paths = simulate_prior(self.drift, self.diffusion, n_steps=nb + n_steps, dt=float(self.cfg.dt), brownian=brownian, **self._start_kwargs())
        return paths if full else paths[:, nb:]

    # ------------------------------------------------------------------ using a trained model
    @property
    def times(self) -> Tensor:
        """Times of the columns of a path on the observation window: ``k * dt`` for ``k = 0, ..., N`` (``[N + 1]``)."""
        return torch.arange(int(self.cfg.data.n_steps) + 1, dtype=torch.float32) * float(self.cfg.dt)

    def _noise(self, n_paths: int, generator: Optional[torch.Generator]) -> Tensor:
        """Brownian increments for ``n_paths`` paths from the start of the model (its burn-in included) to ``T``."""
        if int(n_paths) < 1:
            raise ValueError("the number of paths must be at least 1")
        p = next(self.parameters())
        return brownian_increments(int(n_paths), self.n_burn + int(self.cfg.data.n_steps), float(self.cfg.dt), p.device, generator).to(p.dtype)

    @torch.no_grad()
    def prior_paths(self, n_paths: int = 100, seed: Optional[int] = None) -> Tensor:
        """``n_paths`` intensity paths of the learned prior on ``[0, T]``: ``[n_paths, N + 1]``, column ``k`` at time
        ``times[k]``. ``seed`` makes the draw reproducible."""
        return self.simulate_prior(self._noise(n_paths, _generator(seed)))

    @torch.no_grad()
    def posterior_paths(self, events: Any, n_paths: int = 100, observed_until: Optional[float] = None,
                        seed: Optional[int] = None) -> Tensor:
        """``n_paths`` intensity paths on ``[0, T]`` given the events of one observation: ``[n_paths, N + 1]``.

        ``events`` are the event times of the observation, in ``(0, T]``. With ``observed_until = t`` the observation
        is known on ``[0, t]`` only (``t`` is rounded to the Euler grid): later events are not used, and after ``t``
        the paths follow the dynamics of the learned prior, which makes that part a forecast. The default is the
        whole window. The paths are simulated with the bounds on the score and on the control of the config.
        """
        cfg = self.cfg
        N, dt, T = int(cfg.data.n_steps), float(cfg.dt), float(cfg.data.T)
        p = next(self.parameters())
        ev = torch.as_tensor(events, dtype=torch.float32).flatten()
        ev = ev[torch.isfinite(ev)]
        if ev.numel() and (float(ev.min()) <= 0.0 or float(ev.max()) > T):
            raise ValueError(f"the event times must lie in (0, T] with T = {T:g}: got {float(ev.min()):g} to {float(ev.max()):g}")
        horizon = T if observed_until is None else float(observed_until)
        n_prime = int(round(horizon / dt))
        if not 0.0 < horizon <= T or n_prime < 1:
            raise ValueError(f"observed_until must lie in (0, T] with T = {T:g} and cover at least one Euler step (dt = {dt:g})")
        ev = ev[ev <= n_prime * dt].sort().values
        if ev.numel() == 0:                                  # nothing observed: the row is all padding
            ev = torch.full((1,), float("inf"))
        sim = self.simulate_posterior(ev.unsqueeze(0).to(p.device, p.dtype), n_prime, self._noise(n_paths, _generator(seed)), N,
                                      max_abs_score=cfg.train.max_abs_score, max_abs_control=cfg.train.max_abs_control)
        return sim.Z[:, sim.n_burn:]

    @torch.no_grad()
    def generate_observations(self, n: int = 1, seed: Optional[int] = None) -> List[np.ndarray]:
        """``n`` new observations of the learned model: for each, a path of the learned prior and the event times of a
        Poisson process with that intensity (sorted, on ``[0, T]``)."""
        from .data import thin_events   # imported here: the model itself does not need the data module

        generator = _generator(seed)
        paths = self.simulate_prior(self._noise(n, generator)).float().cpu()
        return thin_events(paths, float(self.cfg.data.T), generator=generator)

    def coefficient_state(self, raw: Tensor) -> Tensor:
        return raw if self.real_states else positive_part(raw, self.cfg.model.positivity, self.cfg.model.softplus_scale)

    def to_intensity(self, raw: Tensor) -> Tensor:
        return self.link(raw) if self.real_states else self.coefficient_state(raw)

    @property
    def sde_kwargs(self) -> Dict[str, Any]:
        return {"link": self.link if self.real_states else None,
                "positivity": self.cfg.model.positivity, "softplus_scale": self.cfg.model.softplus_scale}

    def intensity_drift(self, intensity: Tensor, time: Tensor) -> Tensor:
        if not self.real_states:
            return self.drift(intensity, time)
        return intensity_coefficients(self.link, self.drift, self.diffusion, intensity, time)[0]

    def intensity_diffusion(self, intensity: Tensor, time: Tensor) -> Tensor:
        if not self.real_states:
            return self.diffusion(intensity, time)
        return intensity_coefficients(self.link, self.drift, self.diffusion, intensity, time)[1]

    # ------------------------------------------------------------------ parameter groups
    def param_groups(self) -> Dict[str, list]:
        groups = {
            "prior": [p for p in self.drift.parameters() if p.requires_grad],
            "posterior": [p for p in self.score_net.parameters() if p.requires_grad],
            "diffusion": [p for p in self.diffusion.parameters() if p.requires_grad],
        }
        return {k: v for k, v in groups.items() if len(v) > 0}

    def named_scalars(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update({f"drift.{k}": v for k, v in self.drift.named_scalars().items()})
        if self.diffusion.trainable:
            out.update({f"diffusion.{k}": v for k, v in self.diffusion.named_scalars().items()})
        return out

    def comparable_truth_scalars(self) -> Dict[str, float]:
        """Truth references only for parameters retaining their intensity units."""
        truth = self.cfg.data.truth
        if self.cfg.data.source != "synthetic":      # observations of an unspecified model: no truth
            return {}
        if not self.real_states:
            return {"kappa": float(truth.kappa), "theta": float(truth.theta), "eta": float(truth.eta)}
        if resolved_diffusion_kind(self.cfg.model) == "cir_exponential":
            return {"eta": float(truth.eta)}
        return {}

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.cfg.to_dict(), "state_dict": self.state_dict(), "extra": extra or {}}
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: Path, map_location: str = "cpu") -> Tuple["NDIM", Dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        cfg = config_from_dict(payload["config"])
        model = NDIM(cfg)
        model.load_state_dict(payload["state_dict"])
        return model, payload.get("extra", {})


def build_model(cfg: ExperimentConfig, posterior_correction: Optional[PosteriorCorrection] = None) -> NDIM:
    return NDIM(cfg, posterior_correction=posterior_correction)
