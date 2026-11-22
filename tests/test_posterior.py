"""Tests for the posterior score networks in ``ndim/models/posterior.py`` (paper Sec. 3.3.1).

The key structural property (Theorem 2.1 / "hard-coded constraints"): the score ``d/dz log h`` jumps by
exactly ``1/Z`` every time ``t`` passes an event, i.e. ``s(t_k) - s(t_{k+1}) = #{tau in (t_k, t_{k+1}]} / Z``.
The code implements the jump term as ``r_t / max(Z, jump_z_floor)`` (``jump_term=raw``) or with a continuous
baseline subtracted (``baseline_removed``, DESIGN.md Sec. 3.3).

These tests use the intensity-score convention. Real-state link-score
checks live in test_real_state.py. All tests use ``data.n_steps=16`` (``dt = 0.25``) so grid times are exact in float32.
"""
from __future__ import annotations

from typing import Callable

import pytest
import torch

from ndim.config import ExperimentConfig
from ndim.model import NDIM, build_model
from ndim.models.posterior import DeepSetScore, RemainingCountScore, ScoreNetwork

INF = float("inf")
T = 4.0
DT = 0.25
N = 16
KINDS = ["remaining_count", "deepset"]


def _build(make_cfg: Callable[..., ExperimentConfig], *overrides: str, seed: int = 0) -> NDIM:
    cfg = make_cfg("data.n_steps=16", "model.state_space=intensity", *overrides)
    torch.manual_seed(seed)
    model = build_model(cfg)
    model.eval()
    return model


def _events() -> torch.Tensor:
    # two observations; second has events straddling grid points and a double crossing in (1.5, 1.75]
    return torch.tensor(
        [
            [0.5, 1.7, 1.75, 3.9, INF, INF],
            [0.1, 0.25, 1.0, 2.2, 2.3, 3.99],
        ]
    )


def _crossed(events: torch.Tensor, t_grid: torch.Tensor, t_bar: float) -> torch.Tensor:
    """``c[b, k] = #{tau_b in (t_k, t_{k+1}]}`` for ``k < G-1`` (events beyond ``t_bar`` never count)."""
    ev = events[:, None, :]
    lo, hi = t_grid[:-1][None, :, None], t_grid[1:][None, :, None]
    return ((ev > lo) & (ev <= hi) & (ev <= t_bar)).sum(dim=2).float()


def _scores(net: ScoreNetwork, events: torch.Tensor, t_grid: torch.Tensor, t_bar: float, Z_rows: torch.Tensor) -> torch.Tensor:
    """``[G, B*m]`` scores at every grid time for fixed states ``Z_rows``."""
    with torch.no_grad():
        ctx = net.prepare(events, t_grid, t_bar)
        return torch.stack([net.score(ctx, k, Z_rows) for k in range(t_grid.shape[0])], dim=0)


# --------------------------------------------------------------------------------------
# jump structure
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_raw_jump_equals_events_crossed_over_floor(make_cfg, kind: str) -> None:
    """``jump_term=raw`` with the network switched off (``score_scale=0``): the drop of the score between
    consecutive grid steps is ``(#events crossed) / max(Z, jump_z_floor)`` exactly."""
    floor = 1.0
    model = _build(make_cfg, f"model.posterior.kind={kind}", "model.posterior.jump_term=raw",
                   "model.posterior.score_scale=0.0", f"model.posterior.jump_z_floor={floor}")
    net = model.score_net
    events = _events()
    B = events.shape[0]
    t_grid = torch.arange(N + 1) * DT                       # include t_bar so every crossing is visible
    t_bar = T
    Zvals = torch.tensor([0.3, 1.0, 7.0])                   # below, at and above the floor
    m = Zvals.numel()
    Z_rows = Zvals.repeat(B)                                # row b*m + j -> Zvals[j]
    S = _scores(net, events, t_grid, t_bar, Z_rows)         # [G, B*m]
    drops = S[:-1] - S[1:]                                  # [G-1, B*m]
    crossed = _crossed(events, t_grid, t_bar)               # [B, G-1]
    denom = torch.clamp(Z_rows, min=floor)                  # [B*m]
    expected = crossed.repeat_interleave(m, dim=0).T / denom[None, :]
    assert torch.allclose(drops, expected, atol=1e-6)
    assert float(crossed.sum()) == 10.0                     # all events of both observations are crossed once
    # the score itself is r_t / max(Z, floor) and vanishes at t = t_bar
    assert torch.allclose(S[-1], torch.zeros(B * m))
    assert torch.allclose(S[0], events.isfinite().sum(1).float().repeat_interleave(m) / denom)


@pytest.mark.parametrize("kind", KINDS)
def test_raw_minus_none_is_exactly_the_jump_term(make_cfg, kind: str) -> None:
    """With identical network weights, ``raw`` and ``none`` differ by ``r_t / max(Z, floor)`` only."""
    floor = 2.0
    common = [f"model.posterior.kind={kind}", f"model.posterior.jump_z_floor={floor}", "model.posterior.score_scale=1.0"]
    m_raw = _build(make_cfg, *common, "model.posterior.jump_term=raw", seed=7)
    m_none = _build(make_cfg, *common, "model.posterior.jump_term=none", seed=7)
    for (n1, p1), (n2, p2) in zip(m_raw.score_net.named_parameters(), m_none.score_net.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    events = _events()
    t_grid = torch.arange(N) * DT
    Z_rows = torch.tensor([0.5, 3.0, 40.0]).repeat(events.shape[0])
    S_raw = _scores(m_raw.score_net, events, t_grid, T, Z_rows)
    S_none = _scores(m_none.score_net, events, t_grid, T, Z_rows)
    with torch.no_grad():
        r, _ = ScoreNetwork._counts(events, t_grid, T)      # [B, G]
    expected = r.repeat_interleave(3, dim=0).T / torch.clamp(Z_rows, min=floor)[None, :]
    assert torch.allclose(S_raw - S_none, expected, atol=1e-5)
    # the network part is not identically zero here (score_scale = 1, random init)
    assert float(S_none.abs().max()) > 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_baseline_removed_equals_raw_plus_continuous_function(make_cfg, kind: str) -> None:
    """``baseline_removed`` = ``raw`` - ``w(t) R_0 / max(Z, floor)`` with ``w(t) = (T' - t)/T'``: a function that
    is linear (hence continuous) in ``t`` and does not depend on where the events are -- so the jump
    structure of ``raw`` is preserved exactly (DESIGN.md Sec. 3.3)."""
    floor = 1.0
    common = [f"model.posterior.kind={kind}", f"model.posterior.jump_z_floor={floor}"]
    m_raw = _build(make_cfg, *common, "model.posterior.jump_term=raw", seed=3)
    m_br = _build(make_cfg, *common, "model.posterior.jump_term=baseline_removed", seed=3)
    events = _events()
    B = events.shape[0]
    t_bar = 3.0                                             # partial horizon: R_0 counts events in (0, 3]
    G = int(round(t_bar / DT)) + 1
    t_grid = torch.arange(G) * DT
    Zvals = torch.tensor([0.4, 12.0])
    m = Zvals.numel()
    Z_rows = Zvals.repeat(B)
    S_raw = _scores(m_raw.score_net, events, t_grid, t_bar, Z_rows)
    S_br = _scores(m_br.score_net, events, t_grid, t_bar, Z_rows)
    diff = S_br - S_raw                                     # [G, B*m]
    R0 = ((events > 0) & (events <= t_bar)).sum(dim=1).float()          # [B]
    w = torch.clamp((t_bar - t_grid) / t_bar, 0.0, 1.0)                 # [G]
    expected = -w[:, None] * R0.repeat_interleave(m)[None, :] / torch.clamp(Z_rows, min=floor)[None, :]
    assert torch.allclose(diff, expected, atol=1e-5)
    # linear in t: second differences along the grid vanish, first differences are constant
    first = diff[1:] - diff[:-1]
    assert torch.allclose(first, first[:1].expand_as(first), atol=1e-5)
    assert torch.allclose(first[0], (DT / t_bar) * R0.repeat_interleave(m) / torch.clamp(Z_rows, min=floor), atol=1e-5)
    # the baseline vanishes at t = T' so both variants agree there
    assert torch.allclose(S_br[-1], S_raw[-1], atol=1e-5)
    # consequently the per-step drops of baseline_removed differ from raw by a constant (no extra jumps)
    d_raw = S_raw[:-1] - S_raw[1:]
    d_br = S_br[:-1] - S_br[1:]
    assert torch.allclose(d_br - d_raw, (d_br - d_raw)[:1].expand_as(d_br), atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_network_part_is_smooth_between_events(make_cfg, kind: str) -> None:
    """``jump_term=none``: the learned part is a smooth function of ``(Z, t)``; over a tiny time step that
    crosses no event, and over a tiny change of ``Z``, it barely moves (no hidden jumps)."""
    model = _build(make_cfg, f"model.posterior.kind={kind}", "model.posterior.jump_term=none",
                   "model.posterior.score_scale=1.0", seed=5)
    net = model.score_net
    events = _events()
    B = events.shape[0]
    eps = 1e-4
    t0 = 1.5                                                # no event of either observation in (1.5, 1.5 + eps]
    assert not bool(((events > t0) & (events <= t0 + eps)).any())
    t_grid = torch.tensor([t0, t0 + eps])
    Z_rows = torch.tensor([0.5, 30.0]).repeat(B)
    S = _scores(net, events, t_grid, T, Z_rows)
    assert float((S[1] - S[0]).abs().max()) < 1e-3
    # smooth in Z as well (no floor kink without the jump term)
    with torch.no_grad():
        ctx = net.prepare(events, torch.tensor([t0]), T)
        s_a = net.score(ctx, 0, Z_rows)
        s_b = net.score(ctx, 0, Z_rows + eps)
    assert float((s_a - s_b).abs().max()) < 1e-3
    # sanity: the same tiny step *with* the raw jump term across an event changes the score by ~1/Z
    m_raw = _build(make_cfg, f"model.posterior.kind={kind}", "model.posterior.jump_term=raw",
                   "model.posterior.score_scale=0.0", "model.posterior.jump_z_floor=0.1", seed=5)
    tg = torch.tensor([1.0 - eps, 1.0])                     # observation 2 has an event exactly at 1.0
    S2 = _scores(m_raw.score_net, events, tg, T, Z_rows)
    drop = (S2[0] - S2[1]).view(B, 2)
    assert torch.allclose(drop[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(drop[1], 1.0 / torch.tensor([0.5, 30.0]), atol=1e-5)


# --------------------------------------------------------------------------------------
# counts helper
# --------------------------------------------------------------------------------------
def test_future_counts_semantics() -> None:
    events = torch.tensor([[0.0, 0.5, 1.0, 2.5, INF]])      # an event at exactly 0 is not counted
    t_grid = torch.tensor([0.0, 0.5, 1.0, 2.0])
    r, R0 = ScoreNetwork._counts(events, t_grid, t_bar=2.0)
    assert r.shape == (1, 4) and R0.shape == (1,)
    assert r[0].tolist() == [2.0, 1.0, 0.0, 0.0]            # t_k < tau <= 2.0
    assert float(R0[0]) == 2.0                              # 0.5 and 1.0 (2.5 > t_bar, 0.0 excluded)
    assert r.dtype == torch.float32


# --------------------------------------------------------------------------------------
# shapes: remaining_count, deepset, phi_uses_state
# --------------------------------------------------------------------------------------
def test_default_architectures(make_cfg) -> None:
    """Defaults: the remaining-count MLP is 4 -> 32 -> 16 -> 1 and the Deep Sets ``phi`` sees the state."""
    count = _build(make_cfg, "model.posterior.kind=remaining_count").score_net
    assert count.rho.in_dim == 4
    assert [layer.out_features for layer in count.rho.net if isinstance(layer, torch.nn.Linear)] == [32, 16, 1]
    deep = _build(make_cfg, "model.posterior.kind=deepset").score_net
    assert deep.phi_uses_state and deep.phi.in_dim == deep.n_feat + 1          # the paper's form: the state is an input of phi
    plain = _build(make_cfg, "model.posterior.kind=deepset", "model.posterior.phi_uses_state=false").score_net
    assert plain.phi.in_dim == plain.n_feat                                    # the faster form: the state enters through rho only


def test_remaining_count_shapes(make_cfg) -> None:
    model = _build(make_cfg, "model.posterior.kind=remaining_count", "model.posterior.count_hidden=[8]")
    net = model.score_net
    assert isinstance(net, RemainingCountScore)
    assert net.rho.in_dim == 4                              # [Z/z_s, t/T, T'/T, r/c_s]
    events = _events()
    B, m, G = events.shape[0], 3, 9
    t_grid = torch.arange(G) * DT
    ctx = net.prepare(events, t_grid, t_bar=G * DT)
    assert ctx.r.shape == (B, G) and ctx.R0.shape == (B,)
    assert ctx.feats is None and ctx.gate is None and ctx.e is None
    assert ctx.B == B
    inp = net.row_inputs(ctx, 2, m)
    assert inp["r"].shape == (B * m,) and inp["R0"].shape == (B * m,)
    assert inp["t"].dim() == 0 and float(inp["t"]) == pytest.approx(2 * DT)
    Z = torch.rand(B * m) * 50
    s = net.score(ctx, 2, Z)
    assert s.shape == (B * m,)
    assert torch.isfinite(s).all()
    # forward is a pure function of (Z, inputs): same result
    assert torch.allclose(s, net.forward(Z, inp))


@pytest.mark.parametrize("phi_uses_state", [False, True])
@pytest.mark.parametrize("gap_feature", [True, False])
def test_deepset_shapes(make_cfg, phi_uses_state: bool, gap_feature: bool) -> None:
    model = _build(
        make_cfg,
        "model.posterior.kind=deepset",
        f"model.posterior.phi_uses_state={str(phi_uses_state).lower()}",
        f"model.posterior.gap_feature={str(gap_feature).lower()}",
        "model.posterior.phi_out=5",
    )
    net = model.score_net
    assert isinstance(net, DeepSetScore)
    F = 2 + (1 if gap_feature else 0)
    assert net.n_feat == F
    assert net.phi.in_dim == F + (1 if phi_uses_state else 0)
    assert net.rho.in_dim == 4 + 5
    events = _events()
    B, K = events.shape
    m, G = 2, 7
    t_grid = torch.arange(G) * DT
    t_bar = 3.0
    ctx = net.prepare(events, t_grid, t_bar)
    assert ctx.r.shape == (B, G)
    if phi_uses_state:
        # phi needs Z: per-event features are kept, pooling happens per row in forward
        assert ctx.e is None
        assert ctx.feats.shape == (B, G, K, F) and ctx.gate.shape == (B, G, K)
        inp = net.row_inputs(ctx, 3, m)
        assert inp["feats"].shape == (B * m, K, F) and inp["gate"].shape == (B * m, K)
        assert "e" not in inp
    else:
        assert ctx.e.shape == (B, G, 5) and ctx.feats is None and ctx.gate is None
        inp = net.row_inputs(ctx, 3, m)
        assert inp["e"].shape == (B * m, 5) and "feats" not in inp
    Z = torch.rand(B * m) * 50
    s = net.score(ctx, 3, Z)
    assert s.shape == (B * m,) and torch.isfinite(s).all()
    # gate is a hard 0/1 future mask by default; non-future features are zeroed
    feats, gate = net.event_features(events, t_grid, t_bar)
    assert feats.shape == (B, G, K, F) and gate.shape == (B, G, K)
    fut = (events[:, None, :] > t_grid[None, :, None]) & (events[:, None, :] <= t_bar)
    assert torch.equal(gate.bool(), fut)
    assert float(feats[~fut].abs().sum()) == 0.0
    assert torch.equal(gate.sum(2), ctx.r)                  # pooled weights count the future events


def test_gap_feature_is_measured_in_units_of_the_typical_gap(make_cfg) -> None:
    """Every input of ``phi`` is of order one: the gap is divided by ``T / count_scale``, not by ``T``."""
    net = _build(make_cfg, "model.posterior.kind=deepset").score_net
    typical = net.T / net.count_scale
    assert net.gap_scale == pytest.approx(typical)
    events = (torch.arange(1, 9, dtype=torch.float32) * typical).unsqueeze(0)      # eight events one typical gap apart
    feats, _ = net.event_features(events, torch.tensor([0.0]), T)
    assert torch.allclose(feats[0, 0, :, 2], torch.ones(8), atol=1e-5)
    assert float(feats[0, 0, :, :2].abs().max()) <= 1.0                            # the two time features are fractions of T
    # the unit is a setting: here T itself
    in_T = _build(make_cfg, "model.posterior.kind=deepset", f"model.posterior.gap_scale={T}").score_net
    feats_T, _ = in_T.event_features(events, torch.tensor([0.0]), T)
    assert torch.allclose(feats_T[0, 0, :, 2], torch.full((8,), typical / T), atol=1e-7)


def test_deepset_precomputed_pool_matches_per_row_pool(make_cfg) -> None:
    """``prepare(precompute_pool=False)`` (used by the sensitivity mode) must give the same score."""
    model = _build(make_cfg, "model.posterior.kind=deepset", "model.posterior.phi_uses_state=false")
    net = model.score_net
    events = _events()
    t_grid = torch.arange(N) * DT
    Z = torch.rand(events.shape[0] * 2) * 60
    with torch.no_grad():
        ctx_a = net.prepare(events, t_grid, T, precompute_pool=True)
        ctx_b = net.prepare(events, t_grid, T, precompute_pool=False)
        assert ctx_a.e is not None and ctx_b.e is None
        for k in (0, 5, N - 1):
            assert torch.allclose(net.score(ctx_a, k, Z), net.score(ctx_b, k, Z), atol=1e-5)


def test_deepset_smooth_future_gate(make_cfg) -> None:
    model = _build(make_cfg, "model.posterior.kind=deepset", "model.posterior.future_gate_width=0.08")
    net = model.score_net
    events = _events()
    t_grid = torch.arange(N) * DT
    feats, gate = net.event_features(events, t_grid, T)
    assert torch.all((gate >= 0) & (gate <= 1))
    fut = (events[:, None, :] > t_grid[None, :, None]) & (events[:, None, :] <= T)
    assert float(gate[~fut].abs().sum()) == 0.0
    # far-future events are fully gated in, an event just after t is (smoothly) suppressed
    rel = events[:, None, :] - t_grid[None, :, None]
    assert torch.all(gate[fut & (rel > 1.0)] > 0.99)
    assert float(gate[1, 3, 2]) > 0.95                      # tau = 1.0, t_3 = 0.75 -> rel = 0.25 > 3 widths: gated in
    # an event less than one width after t is strongly suppressed: tanh(0.5)^2 ~ 0.21
    with torch.no_grad():
        _, g1 = net.event_features(events, torch.tensor([0.96]), T)
    assert 0.1 < float(g1[1, 0, 2]) < 0.5
    k_just_before = int(torch.argmin((t_grid - 0.99).abs()))
    with torch.no_grad():
        tg2 = torch.tensor([0.99])
        _, g2 = net.event_features(events, tg2, T)
    assert float(g2[1, 0, 2]) < 0.05                        # tau = 1.0 only 0.01 after t


def test_score_scale_multiplies_network_part_only(make_cfg) -> None:
    common = ["model.posterior.kind=remaining_count", "model.posterior.jump_term=raw"]
    m1 = _build(make_cfg, *common, "model.posterior.score_scale=1.0", seed=9)
    m2 = _build(make_cfg, *common, "model.posterior.score_scale=3.0", seed=9)
    m0 = _build(make_cfg, *common, "model.posterior.score_scale=0.0", seed=9)
    events = _events()
    t_grid = torch.arange(N) * DT
    Z = torch.tensor([2.0, 20.0]).repeat(events.shape[0])
    S1, S2, S0 = (_scores(m.score_net, events, t_grid, T, Z) for m in (m1, m2, m0))
    assert torch.allclose(S2 - S0, 3.0 * (S1 - S0), atol=1e-5)
