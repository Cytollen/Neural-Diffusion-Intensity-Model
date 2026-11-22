"""Tests for ``ndim.data``: truth simulation, exact thinning (paper App. G, Alg. 2), padding, subsets, caching."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ndim.config import ExperimentConfig, TruthConfig
from ndim.data import (
    Dataset,
    EventData,
    clip_to_horizon,
    dataset_filename,
    future_counts_on_grid,
    generate_dataset,
    load_or_generate_dataset,
    make_truth_diffusion,
    make_truth_drift,
    simulate_truth_paths,
    thin_events,
)


# --------------------------------------------------------------------------------------
# ground truth coefficients and paths
# --------------------------------------------------------------------------------------
def test_truth_drift_and_diffusion_formulas() -> None:
    truth = TruthConfig(kappa=0.3, theta=80.0, eta=2.0, z0=5.0)
    b = make_truth_drift(truth)
    s = make_truth_diffusion(truth)
    z = torch.tensor([0.0, 4.0, 100.0])
    t = torch.tensor([0.0, 1.0, 2.0])
    # b(z, t) = kappa (theta - z), whatever the time
    assert torch.allclose(b(z, t), 0.3 * (80.0 - z))
    assert b(z[:, None], t[None, :]).shape == (3, 3)              # and it broadcasts over a mesh of states and times
    # sigma(z, t) = eta sqrt(relu(z)), and negative states are truncated
    assert torch.allclose(s(z, t), 2.0 * torch.sqrt(z))
    assert float(s(torch.tensor(-3.0), torch.tensor(0.0))) == 0.0
    # broadcasting: scalar t against a vector z
    assert b(z, torch.tensor(0.0)).shape == z.shape


def test_simulate_truth_paths_shapes_and_positivity(gen: torch.Generator) -> None:
    truth = TruthConfig()
    paths = simulate_truth_paths(truth, T=4.0, n_steps=25, n_paths=7, generator=gen)
    assert paths.shape == (7, 26)
    assert paths.dtype == torch.float32
    assert torch.all(paths >= 0)                            # positive part is returned
    assert torch.allclose(paths[:, 0], torch.full((7,), 5.0))
    # substeps > 1: simulated on a finer grid and subsampled back to n_steps + 1 points
    fine = simulate_truth_paths(truth, T=4.0, n_steps=25, n_paths=3, generator=gen, substeps=4)
    assert fine.shape == (3, 26)


def test_simulate_truth_paths_common_noise_is_deterministic() -> None:
    truth = TruthConfig()
    W = torch.randn(5, 25, generator=torch.Generator().manual_seed(3)) * math.sqrt(4.0 / 25)
    a = simulate_truth_paths(truth, 4.0, 25, 5, brownian=W)
    b = simulate_truth_paths(truth, 4.0, 25, 5, brownian=W)
    assert torch.equal(a, b)
    # with zero noise the Euler path is deterministic mean reversion towards theta
    Z0 = simulate_truth_paths(truth, 4.0, 25, 1, brownian=torch.zeros(1, 25))[0]
    assert torch.all(Z0[1:] > Z0[:-1])                     # z0 = 5 < theta = 80: increasing
    assert float(Z0[-1]) < truth.theta


# --------------------------------------------------------------------------------------
# thinning
# --------------------------------------------------------------------------------------
def test_thinning_constant_intensity_mean_count(gen: torch.Generator) -> None:
    """For a constant path the count is Poisson(c T): the sample mean must be within 5 s.e. of c T."""
    c, T, n_paths, n_grid = 10.0, 4.0, 3000, 26
    paths = torch.full((n_paths, n_grid), c)
    ev = thin_events(paths, T, generator=gen)
    assert len(ev) == n_paths
    counts = np.array([len(e) for e in ev], dtype=float)
    expected = c * T                                        # = sum_k Z_k dt with dt = T / (n_grid - 1)
    se = math.sqrt(expected / n_paths)
    assert abs(counts.mean() - expected) < 5 * se
    for e in ev[:50]:
        assert np.all(np.diff(e) >= 0)                      # sorted
        assert np.all((e >= 0) & (e <= T))


def test_thinning_piecewise_constant_mean_count(gen: torch.Generator) -> None:
    """Non-constant path: the expected count is the left-Riemann integral ``sum_k Z_k dt`` (grid convention
    ``value on [t_k, t_{k+1}) = paths[:, k]``)."""
    T, n_paths, n_grid = 4.0, 3000, 26
    dt = T / (n_grid - 1)
    row = torch.cat([torch.full((13,), 2.0), torch.full((13,), 20.0)])   # low first half, high second half
    paths = row[None, :].expand(n_paths, n_grid).contiguous()
    ev = thin_events(paths, T, generator=gen)
    counts = np.array([len(e) for e in ev], dtype=float)
    expected = float(row[:-1].sum()) * dt
    se = math.sqrt(expected / n_paths)
    assert abs(counts.mean() - expected) < 5 * se
    # events concentrate in the high-intensity half: fraction in the second half ~ 20 / 22
    all_ev = np.concatenate(ev)
    frac_late = np.mean(all_ev > T / 2)
    assert abs(frac_late - 20.0 / 22.0) < 0.03


def test_thinning_zero_intensity_gives_no_events(gen: torch.Generator) -> None:
    paths = torch.zeros(4, 26)
    ev = thin_events(paths, 4.0, generator=gen)
    assert all(len(e) == 0 for e in ev)
    assert all(e.dtype == np.float64 for e in ev)


def test_thinning_reproducible_with_generator() -> None:
    paths = torch.full((5, 26), 7.0)
    a = thin_events(paths, 4.0, generator=torch.Generator().manual_seed(11))
    b = thin_events(paths, 4.0, generator=torch.Generator().manual_seed(11))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------------------------
# EventData: padding with +inf, subsets, helpers
# --------------------------------------------------------------------------------------
def test_event_data_padding_and_lists() -> None:
    lists = [np.array([0.5, 1.5, 3.0]), np.array([]), np.array([2.0])]
    ed = EventData.from_lists(lists, T=4.0)
    assert ed.events.shape == (3, 3) and ed.K_max == 3
    assert ed.events.dtype == torch.float32 and ed.counts.dtype == torch.long
    assert ed.counts.tolist() == [3, 0, 1]
    assert len(ed) == 3
    assert torch.isinf(ed.events[1]).all()                  # empty observation: all padding
    assert torch.isinf(ed.events[2, 1:]).all() and float(ed.events[2, 0]) == 2.0
    assert torch.all(ed.events[:, :-1] <= ed.events[:, 1:])  # +inf padding keeps rows sorted
    back = ed.as_lists()
    for x, y in zip(lists, back):
        assert np.allclose(x, y)
    assert ed.counts_before(1.6).tolist() == [2, 0, 0]
    assert ed.counts_before(4.0).tolist() == [3, 0, 1]


def test_event_data_all_empty_has_one_padding_column() -> None:
    ed = EventData.from_lists([np.array([]), np.array([])], T=1.0)
    assert ed.events.shape == (2, 1)
    assert torch.isinf(ed.events).all()
    assert ed.counts.tolist() == [0, 0]


def test_event_data_subset() -> None:
    lists = [np.array([0.5, 1.5, 3.0, 3.5]), np.array([0.1]), np.array([2.0, 2.5]), np.array([])]
    ed = EventData.from_lists(lists, T=4.0)
    sub = ed.subset([1, 2])
    assert len(sub) == 2
    assert sub.counts.tolist() == [1, 2]
    assert sub.K_max == 2                                   # padding shrinks to the subset's max count
    assert sub.T == ed.T
    for x, y in zip([lists[1], lists[2]], sub.as_lists()):
        assert np.allclose(x, y)
    # subset of a single empty observation keeps one column
    e = ed.subset([3])
    assert e.events.shape == (1, 1) and torch.isinf(e.events).all()
    # torch index tensors are accepted as well
    sub2 = ed.subset(torch.tensor([0, 3]))
    assert sub2.counts.tolist() == [4, 0]


def test_event_data_to_device_noop_on_cpu(device: torch.device) -> None:
    ed = EventData.from_lists([np.array([1.0])], T=4.0)
    ed2 = ed.to(device)
    assert torch.equal(ed.events, ed2.events) and torch.equal(ed.counts, ed2.counts)


def test_clip_to_horizon_and_future_counts() -> None:
    events = torch.tensor([[0.5, 1.5, 3.0, float("inf")], [2.0, float("inf"), float("inf"), float("inf")]])
    clipped = clip_to_horizon(events, 2.0)
    assert clipped.shape == events.shape
    assert clipped[0].tolist()[:2] == [0.5, 1.5] and torch.isinf(clipped[0, 2:]).all()
    assert float(clipped[1, 0]) == 2.0                      # t_bar itself is kept (closed interval)
    t_grid = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r = future_counts_on_grid(events, t_grid, t_bar=3.0)    # r[b,k] = #{t_k < tau <= 3}
    assert r.shape == (2, 4)
    assert r[0].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert r[1].tolist() == [1.0, 1.0, 0.0, 0.0]


# --------------------------------------------------------------------------------------
# dataset generation and caching
# --------------------------------------------------------------------------------------
def test_generate_dataset_shapes(smoke_cfg: ExperimentConfig) -> None:
    ds = generate_dataset(smoke_cfg.data)
    N = smoke_cfg.data.n_steps
    assert len(ds.train) == smoke_cfg.data.n_train and len(ds.test) == smoke_cfg.data.n_test
    assert ds.train_paths.shape == (smoke_cfg.data.n_train, N + 1)
    assert ds.test_paths.shape == (smoke_cfg.data.n_test, N + 1)
    assert torch.all(ds.train_paths >= 0)
    assert ds.train.T == pytest.approx(smoke_cfg.data.T)
    # padding is +inf beyond each observation's count
    K = ds.train.K_max
    ar = torch.arange(K)[None, :]
    assert torch.isinf(ds.train.events[ar >= ds.train.counts[:, None]]).all()
    assert torch.isfinite(ds.train.events[ar < ds.train.counts[:, None]]).all()
    assert torch.all(ds.train.events[torch.isfinite(ds.train.events)] <= smoke_cfg.data.T)


def test_generate_dataset_is_deterministic(smoke_cfg: ExperimentConfig) -> None:
    a = generate_dataset(smoke_cfg.data)
    b = generate_dataset(smoke_cfg.data)
    assert torch.equal(a.train.events, b.train.events)
    assert torch.equal(a.train_paths, b.train_paths)
    assert torch.equal(a.test.counts, b.test.counts)


def test_cache_round_trip(smoke_cfg: ExperimentConfig, tmp_path: Path) -> None:
    root = tmp_path / "cache"
    ds1, path1 = load_or_generate_dataset(smoke_cfg, root=root)
    assert path1.exists() and path1.suffix == ".npz"
    assert path1.name == dataset_filename(smoke_cfg.data)
    mtime = path1.stat().st_mtime_ns
    ds2, path2 = load_or_generate_dataset(smoke_cfg, root=root)   # second call loads from disk
    assert path2 == path1 and path1.stat().st_mtime_ns == mtime
    assert torch.equal(ds1.train.events, ds2.train.events)
    assert torch.equal(ds1.train.counts, ds2.train.counts)
    assert torch.equal(ds1.test.events, ds2.test.events)
    assert torch.equal(ds1.train_paths, ds2.train_paths)
    assert torch.equal(ds1.test_paths, ds2.test_paths)
    assert ds2.train.events.dtype == torch.float32 and ds2.train.counts.dtype == torch.long
    assert ds2.train.T == pytest.approx(smoke_cfg.data.T)
    # force regenerates (deterministic seed -> identical content)
    ds3, _ = load_or_generate_dataset(smoke_cfg, root=root, force=True)
    assert torch.equal(ds1.train.events, ds3.train.events)
    # explicit save/load round trip
    p = tmp_path / "manual.npz"
    ds1.save(p)
    ds4 = Dataset.load(p, smoke_cfg.data)
    assert torch.equal(ds1.test_paths, ds4.test_paths)


def test_dataset_filename_encodes_generating_parameters(smoke_cfg: ExperimentConfig) -> None:
    name = dataset_filename(smoke_cfg.data)
    assert "N25" in name and "ntr16" in name and "nte8" in name and "seed0" in name
    import copy

    d2 = copy.deepcopy(smoke_cfg.data)
    d2.truth.kappa = 0.5
    assert dataset_filename(d2) != name and "_k0.5_" in dataset_filename(d2)
