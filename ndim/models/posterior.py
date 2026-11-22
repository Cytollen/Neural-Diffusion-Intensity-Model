"""Amortised posterior-correction (encoder) networks (paper Sec. 3.3).

Both built-in networks output the score with respect to the simulator state (real X by default,
the intensity Z with ``model.state_space: intensity``). The simulator forms the Brownian-space control ``u = sigma s`` and the drift correction
``sigma u = sigma^2 s``.
With ``score_space=intensity``, the network internally uses intensity inputs and an
intensity score, then returns ``g'(X) * s_intensity`` by the chain rule. The simulator
and custom-correction protocol still always receive a score in simulator coordinates.

Evaluation protocol for one mini-batch of ``B`` observations x ``m`` paths on a grid with ``n'`` controlled steps:

1. ``ctx = net.prepare(events, t_grid, t_bar)`` precomputes everything that depends only on the
   observations and the grid (future counts, per-event features, and -- when ``phi`` does not see the
   state -- the pooled Deep Sets embeddings for every grid time in one batched call).
2. ``s = net.score(ctx, k, Z)`` for the current step ``k`` and states ``Z`` of shape ``[B*m]`` (row
   ``b*m + j`` is path ``j`` of observation ``b``).

``forward(Z, inputs)`` is a pure function of the module parameters and per-row inputs so that
``torch.func`` (vmap/grad/functional_call) can be used for the forward-sensitivity gradient mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from importlib import import_module
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from ..config import PosteriorConfig, required_scale
from ..links import IntensityLink
from .common import MLP

Tensor = torch.Tensor


@dataclass
class ScoreContext:
    """Observation/grid dependent quantities for one mini-batch (see ``ScoreNetwork.prepare``)."""
    t_grid: Tensor              # [G] grid times at which controls are evaluated (G = n')
    t_bar: float
    T: float
    r: Tensor                   # [B, G] future counts #{tau in (t_k, t_bar]}
    R0: Tensor                  # [B]    total counts on (0, t_bar]
    feats: Optional[Tensor]     # [B, G, K, F] per-event features (None for remaining_count)
    gate: Optional[Tensor]      # [B, G, K] future-event weights (0/1 or smooth)
    e: Optional[Tensor]         # [B, G, d] pooled embeddings (None if phi uses the state)

    @property
    def B(self) -> int:
        return int(self.r.shape[0])


class RowInputs(Dict[str, Tensor]):
    """Per-row inputs for ``ScoreNetwork.forward``: keys ``t``, ``t_bar`` (0-d tensors), ``r``, ``R0`` ``[R]``,
    optionally ``e`` ``[R, d]`` or ``feats`` ``[R, K, F]`` + ``gate`` ``[R, K]``."""


class PosteriorCorrection(nn.Module, ABC):
    """Replaceable latent-score interface, independent of observation encoding.

    prepare creates a context exposing B; row_inputs returns tensors for each row;
    forward(state, inputs) is a pure score function (also used by torch.func).
    Scores are with respect to the simulator state X, not the intensity g(X).
    """
    @abstractmethod
    def prepare(self, events: Tensor, t_grid: Tensor, t_bar: float, precompute_pool: bool = True) -> Any:
        raise NotImplementedError

    @abstractmethod
    def row_inputs(self, ctx: Any, k: int, m: int) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: Tensor, inputs: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def score(self, ctx: Any, k: int, state: Tensor) -> Tensor:
        if state.shape[0] % ctx.B:
            raise ValueError("State rows must be a multiple of the context batch size")
        return self(state, self.row_inputs(ctx, k, state.shape[0] // ctx.B))


class ScoreNetwork(PosteriorCorrection):
    """Base class implementing the hard-coded jump term and the ``prepare``/``score`` protocol."""

    def __init__(self, cfg: PosteriorConfig, T: float, link: Optional[IntensityLink] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.link = link
        self.score_space = cfg.score_space
        self.T = float(T)
        self.z_scale = required_scale(cfg.z_scale, "model.posterior.z_scale")
        self.count_scale = float(cfg.count_scale)
        self.score_scale = float(cfg.score_scale)
        self.jump_term = str(cfg.jump_term)
        self.jump_z_floor = float(cfg.jump_z_floor)

    # ------------------------------------------------------------------ shared pieces
    def _network_state(self, state: Tensor) -> Tensor:
        if self.score_space == "intensity" and self.link is not None:
            return self.link(state)
        return state

    def _network_score(self, state: Tensor, score: Tensor) -> Tensor:
        if self.score_space == "intensity" and self.link is not None:
            return self.link.first(state) * score
        return score

    def _base_inputs(self, Z: Tensor, inp: Dict[str, Tensor]) -> Tensor:
        """``[Z/z_scale, t/T, t_bar/T, r/count_scale]`` -> ``[R, 4]`` (or ``[4]`` under vmap)."""
        t = inp["t"].to(Z.dtype) / self.T
        tb = inp["t_bar"].to(Z.dtype) / self.T
        r = inp["r"].to(Z.dtype) / self.count_scale
        cols = [self._network_state(Z) / self.z_scale, torch.broadcast_to(t, Z.shape), torch.broadcast_to(tb, Z.shape), r]
        return torch.stack(cols, dim=-1)

    def _jump(self, Z: Tensor, inp: Dict[str, Tensor]) -> Tensor:
        """Latent jump (r-baseline)g'/g; intensity-state jump (r-baseline)/max(Z,floor)."""
        if self.jump_term == "none":
            return torch.zeros_like(Z)
        r = inp["r"].to(Z.dtype)
        if self.jump_term == "raw":
            num = r
        elif self.jump_term == "baseline_removed":
            t = inp["t"].to(Z.dtype)
            tb = inp["t_bar"].to(Z.dtype)
            w = torch.clamp((tb - t) / torch.clamp(tb, min=1e-12), min=0.0, max=1.0)
            num = r - w * inp["R0"].to(Z.dtype)
        else:
            raise ValueError(self.jump_term)
        if self.link is not None:
            return num * self.link.log_derivative(Z)
        return num / torch.clamp(Z, min=self.jump_z_floor)

    # ------------------------------------------------------------------ protocol
    def prepare(self, events: Tensor, t_grid: Tensor, t_bar: float, precompute_pool: bool = True) -> ScoreContext:  # pragma: no cover
        raise NotImplementedError

    def row_inputs(self, ctx: ScoreContext, k: int, m: int) -> RowInputs:  # pragma: no cover
        raise NotImplementedError

    def forward(self, Z: Tensor, inp: Dict[str, Tensor]) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _counts(events: Tensor, t_grid: Tensor, t_bar: float) -> "tuple[Tensor, Tensor]":
        ev = events[:, None, :]
        valid = torch.isfinite(events) & (events <= float(t_bar)) & (events > 0.0)   # [B, K]
        fut = valid[:, None, :] & (ev > t_grid[None, :, None])                        # [B, G, K]
        r = fut.sum(dim=2).to(torch.float32)
        R0 = valid.sum(dim=1).to(torch.float32)
        return r, R0


class RemainingCountScore(ScoreNetwork):
    """Paper Sec. 3.3.1 "remaining count": ``s = score_scale * MLP([Z/z_s, t/T, T'/T, r_t/c_s]) + jump``.

    The MLP has the hidden widths ``cfg.count_hidden`` (4 -> 32 -> 16 -> 1 by default).
    """

    def __init__(self, cfg: PosteriorConfig, T: float, link: Optional[IntensityLink] = None) -> None:
        super().__init__(cfg, T, link)
        self.rho = MLP(4, cfg.count_hidden, 1, cfg.activation, cfg.init_output_std)

    def prepare(self, events: Tensor, t_grid: Tensor, t_bar: float, precompute_pool: bool = True) -> ScoreContext:
        r, R0 = self._counts(events, t_grid, t_bar)
        return ScoreContext(t_grid=t_grid, t_bar=float(t_bar), T=self.T, r=r, R0=R0, feats=None, gate=None, e=None)

    def row_inputs(self, ctx: ScoreContext, k: int, m: int) -> RowInputs:
        return RowInputs(
            t=ctx.t_grid[k],
            t_bar=torch.as_tensor(ctx.t_bar, dtype=ctx.r.dtype, device=ctx.r.device),
            r=ctx.r[:, k].repeat_interleave(m),
            R0=ctx.R0.repeat_interleave(m),
        )

    def forward(self, Z: Tensor, inp: Dict[str, Tensor]) -> Tensor:
        x = self._base_inputs(Z, inp)
        s = self.score_scale * self.rho(x).squeeze(-1)
        return self._network_score(Z, s) + self._jump(Z, inp)


class DeepSetScore(ScoreNetwork):
    """Paper Sec. 3.3.1 "Deep Sets":

    ``s = score_scale * rho([Z/z_s, t/T, T'/T, r_t/c_s, e]) + jump``,
    ``e = sum_i gate_i * phi([(tau_i - t)/T, (T' - tau_i)/T, Delta_i/g_s, (Z/z_s)])``

    over future events ``t < tau_i <= T'``. ``Delta_i`` is the gap to the previous future event (to
    ``t`` for the first one). Every input is divided by a fixed scale so that it is of order one; the gap is
    measured in units of ``g_s = cfg.gap_scale``, by default the typical gap between events ``T / c_s``
    (dividing it by ``T`` would leave it a hundred times smaller than the other inputs). ``Z`` enters ``phi`` by default
    (``cfg.phi_uses_state``, the paper's form), so the pooling is evaluated for every path at every step. With
    ``phi_uses_state=false`` the pooled embedding depends only on the observation and the grid time and is precomputed
    for all grid times in ``prepare`` (optionally with activation checkpointing to save memory); the state then
    enters through ``rho`` only. That form is about five times faster and was less accurate when the two were compared.
    """

    def __init__(self, cfg: PosteriorConfig, T: float, link: Optional[IntensityLink] = None) -> None:
        super().__init__(cfg, T, link)
        self.phi_uses_state = bool(cfg.phi_uses_state)
        self.gap_feature = bool(cfg.gap_feature)
        self.gap_scale = float(cfg.gap_scale) if cfg.gap_scale is not None else self.T / self.count_scale
        self.gate_width = float(cfg.future_gate_width)
        self.checkpoint_phi = bool(cfg.checkpoint_phi)
        self.n_feat = 2 + (1 if self.gap_feature else 0)
        phi_in = self.n_feat + (1 if self.phi_uses_state else 0)
        self.phi = MLP(phi_in, cfg.phi_hidden, cfg.phi_out, cfg.activation, init_output_std=None)
        self.rho = MLP(4 + cfg.phi_out, cfg.rho_hidden, 1, cfg.activation, cfg.init_output_std)

    # -------------------------------------------------------------- features
    def event_features(self, events: Tensor, t_grid: Tensor, t_bar: float) -> "tuple[Tensor, Tensor]":
        """``feats [B, G, K, F]`` and ``gate [B, G, K]`` for events (padded with +inf) and grid times."""
        B, K = events.shape
        G = t_grid.shape[0]
        valid = torch.isfinite(events) & (events <= float(t_bar)) & (events > 0.0)          # [B, K]
        ev = torch.where(valid, events, torch.zeros_like(events))                            # finite copy
        rel = ev[:, None, :] - t_grid[None, :, None]                                         # [B, G, K]
        fut = valid[:, None, :] & (rel > 0)
        if self.gate_width > 0:
            gate = torch.tanh(torch.clamp(rel, min=0.0) / self.gate_width) ** 2 * fut.to(events.dtype)
        else:
            gate = fut.to(events.dtype)
        rem = (float(t_bar) - ev)[:, None, :].expand(B, G, K)
        cols = [rel / self.T, rem / self.T]
        if self.gap_feature:
            prev = torch.cat([torch.zeros_like(ev[:, :1]), ev[:, :-1]], dim=1)                 # tau_{i-1}, 0 for i=0
            prev = torch.where(valid, prev, torch.zeros_like(prev))
            gap = ev[:, None, :] - torch.maximum(prev[:, None, :], t_grid[None, :, None])     # tau_i - max(tau_{i-1}, t)
            cols.append(torch.clamp(gap, min=0.0) / self.gap_scale)
        feats = torch.stack(cols, dim=-1)
        feats = feats * fut.unsqueeze(-1).to(feats.dtype)      # zero-out non-future entries
        return feats, gate

    def _pool(self, feats: Tensor, gate: Tensor) -> Tensor:
        """``sum_i gate_i phi(feats_i)`` over the last event axis. ``feats [..., K, F]``, ``gate [..., K]``."""
        h = self.phi(feats)                                    # [..., K, d]
        return (h * gate.unsqueeze(-1)).sum(dim=-2)

    def _pool_ckpt(self, feats: Tensor, gate: Tensor) -> Tensor:
        if self.checkpoint_phi and self.training and torch.is_grad_enabled():
            return ckpt.checkpoint(self._pool, feats, gate, use_reentrant=False)
        return self._pool(feats, gate)

    # -------------------------------------------------------------- protocol
    def prepare(self, events: Tensor, t_grid: Tensor, t_bar: float, precompute_pool: bool = True) -> ScoreContext:
        """``precompute_pool=False`` keeps the raw per-event features so that ``forward`` recomputes the
        pooling per row (needed when differentiating with ``torch.func`` in the sensitivity gradient mode)."""
        r, R0 = self._counts(events, t_grid, t_bar)
        feats, gate = self.event_features(events, t_grid, t_bar)
        e = None
        if not self.phi_uses_state and precompute_pool:
            e = self._pool_ckpt(feats, gate)                   # [B, G, d]
            feats, gate = None, None
        return ScoreContext(t_grid=t_grid, t_bar=float(t_bar), T=self.T, r=r, R0=R0, feats=feats, gate=gate, e=e)

    def row_inputs(self, ctx: ScoreContext, k: int, m: int) -> RowInputs:
        inp = RowInputs(
            t=ctx.t_grid[k],
            t_bar=torch.as_tensor(ctx.t_bar, dtype=ctx.r.dtype, device=ctx.r.device),
            r=ctx.r[:, k].repeat_interleave(m),
            R0=ctx.R0.repeat_interleave(m),
        )
        if ctx.e is not None:
            inp["e"] = ctx.e[:, k].repeat_interleave(m, dim=0)
        else:
            inp["feats"] = ctx.feats[:, k].repeat_interleave(m, dim=0)   # [B*m, K, F]
            inp["gate"] = ctx.gate[:, k].repeat_interleave(m, dim=0)     # [B*m, K]
        return inp

    def forward(self, Z: Tensor, inp: Dict[str, Tensor]) -> Tensor:
        x = self._base_inputs(Z, inp)
        if "e" in inp:
            e = inp["e"].to(Z.dtype)
        else:
            feats, gate = inp["feats"], inp["gate"]
            if self.phi_uses_state:
                zcol = (self._network_state(Z) / self.z_scale).unsqueeze(-1).unsqueeze(-1).expand(*feats.shape[:-1], 1)
                feats = torch.cat([feats, zcol], dim=-1)
            e = self._pool(feats, gate)
        s = self.score_scale * self.rho(torch.cat([x, e], dim=-1)).squeeze(-1)
        return self._network_score(Z, s) + self._jump(Z, inp)


PosteriorFactory = Callable[[PosteriorConfig, float, Optional[IntensityLink]], PosteriorCorrection]
_POSTERIORS: Dict[str, PosteriorFactory] = {
    "deepset": DeepSetScore,
    "remaining_count": RemainingCountScore,
}


def register_posterior(name: str, factory: PosteriorFactory) -> None:
    """Register an architecture; duplicate names are rejected to prevent accidental replacement."""
    if not name or name in _POSTERIORS:
        raise ValueError(f"Posterior architecture {name!r} is empty or already registered")
    _POSTERIORS[name] = factory


def build_posterior(cfg: PosteriorConfig, T: float, link: Optional[IntensityLink] = None) -> PosteriorCorrection:
    if cfg.factory:
        module, name = cfg.factory.split(":", 1)
        factory = getattr(import_module(module), name)
    else:
        try:
            factory = _POSTERIORS[cfg.kind]
        except KeyError as e:
            raise ValueError(f"Unknown posterior architecture {cfg.kind!r}; registered: {sorted(_POSTERIORS)}") from e
    correction = factory(cfg, T, link)
    if not isinstance(correction, PosteriorCorrection):
        raise TypeError("Posterior factory must return a PosteriorCorrection module")
    return correction
