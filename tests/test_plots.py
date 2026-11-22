"""Tests for ``ndim.plots``: the training-time figures and the evaluation figures.

Every figure function must accept numpy *and* torch inputs, write a non-empty PNG when ``path`` is given and
return that path (``None`` when no path is given). Rendering uses the Agg backend, so no display is needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from ndim import plots
from ndim.config import ExperimentConfig
from ndim.data import make_truth_drift
from ndim.model import NDIM


def _history():
    return [
        {"epoch": 1, "train_elbo": 10.0, "train_loglik": 12.0, "train_kl": 2.0, "test_elbo": 9.0, "test_loglik": 11.0, "test_kl": 2.0,
         "eval_train_elbo": 9.5, "eval_train_loglik": 11.5, "eval_train_kl": 2.0, "drift.kappa": 0.5, "drift.theta": 40.0},
        {"epoch": 2, "train_elbo": 11.0, "train_loglik": 13.0, "train_kl": 2.0, "drift.kappa": 0.45, "drift.theta": 45.0},
        {"epoch": 3, "train_elbo": 12.0, "train_loglik": 13.5, "train_kl": 1.5, "test_elbo": 10.0, "test_loglik": 11.8, "test_kl": 1.8,
         "eval_train_elbo": 10.5, "eval_train_loglik": 12.0, "eval_train_kl": 1.5, "drift.kappa": 0.4, "drift.theta": 50.0},
    ]


def _ok(p, expected: Path) -> None:
    assert p is not None
    assert Path(p) == expected
    assert expected.exists() and expected.stat().st_size > 0


def test_agg_backend_and_colors_present() -> None:
    assert matplotlib.get_backend().lower() == "agg"
    for k in ("prior", "mcmc", "vi", "true_prior", "learned_prior", "true_drift", "learned_drift", "train", "test", "events"):
        assert k in plots.COLORS


# --------------------------------------------------------------------------------------
# training-time figures
# --------------------------------------------------------------------------------------
def test_training_curves_and_parameter_history(tmp_path: Path) -> None:
    p = plots.plot_training_curves(_history(), tmp_path / "curves.png", title="t")
    _ok(p, tmp_path / "curves.png")
    p2 = plots.plot_parameter_history(_history(), ["drift.kappa", "drift.theta", "diffusion.eta"],
                                      truth={"kappa": 0.3, "theta": 80.0}, path=tmp_path / "params.png")
    _ok(p2, tmp_path / "params.png")
    assert plots.plot_parameter_history(_history(), ["diffusion.eta"], path=tmp_path / "none.png") is None


def test_drift_snapshot(smoke_cfg: ExperimentConfig, tiny_model: NDIM, smoke_dataset, tmp_path: Path) -> None:
    truth = make_truth_drift(smoke_cfg.data.truth)
    p = plots.plot_drift_snapshot(lambda z, t: tiny_model.drift(z, t), truth, smoke_cfg.eval.drift_z_max, smoke_cfg.data.T,
                                  occupancy=smoke_dataset.train_paths, path=tmp_path / "drift.png", title="drift")
    _ok(p, tmp_path / "drift.png")
    assert plots.plot_drift_snapshot(lambda z, t: tiny_model.drift(z, t), None, 100.0, 4.0) is None   # no path -> no file


# --------------------------------------------------------------------------------------
# evaluation figures
# --------------------------------------------------------------------------------------
def test_plot_cumulative_counts(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    true_ev = [np.sort(rng.uniform(0, 4, size=rng.integers(0, 30))) for _ in range(6)]
    gen_ev = [torch.as_tensor(np.sort(rng.uniform(0, 4, size=rng.integers(0, 30)))) for _ in range(6)]   # torch inputs
    p = plots.plot_cumulative_counts(true_ev, gen_ev, 4.0, n_grid=50, show_individual=True, path=tmp_path / "cum.png", title="Fig. 3")
    _ok(p, tmp_path / "cum.png")
    # padded (+inf) rows and an empty observation are tolerated; no path -> None
    padded = [np.array([0.5, 1.0, np.inf, np.inf]), np.zeros(0)]
    assert plots.plot_cumulative_counts(padded, gen_ev, 4.0, n_grid=20) is None


def test_cumulative_counts_helper_matches_definition() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    N = plots._cumulative_counts([np.array([0.5, 1.0, 2.5]), np.array([np.inf, 3.0])], t)
    assert N.shape == (2, 4)
    assert N[0].tolist() == [0.0, 2.0, 2.0, 3.0]     # counts of events <= t (right-continuous)
    assert N[1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_plot_posterior_comparison(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(0)
    t = np.linspace(0, 4, 26)
    prior, mc, vi = (torch.rand(5, 26, generator=g) * 50 for _ in range(3))
    events = np.array([0.5, 1.2, 3.3])
    p = plots.plot_posterior_comparison(t, prior, mc, vi, events, t_bar=2.0, path=tmp_path / "post.png", title="Fig. 4b", show_individual=True)
    _ok(p, tmp_path / "post.png")
    p2 = plots.plot_posterior_comparison(t, prior.numpy(), mc.numpy(), vi.numpy(), events, None, path=tmp_path / "post2.png")
    _ok(p2, tmp_path / "post2.png")
    # MCMC disabled (None), torch t_grid and padded events
    p3 = plots.plot_posterior_comparison(torch.as_tensor(t), prior, None, vi, np.array([0.5, np.inf]), 4.0, path=tmp_path / "post3.png")
    _ok(p3, tmp_path / "post3.png")
    with pytest.raises(ValueError):
        plots.plot_posterior_comparison(t, prior[:, :-1], mc, vi, events, None)
    # the true latent path of the observation (synthetic data): one more line, and its length is checked
    p4 = plots.plot_posterior_comparison(t, prior, mc, vi, events, None, path=tmp_path / "post4.png", true_path=prior[0])
    _ok(p4, tmp_path / "post4.png")
    with pytest.raises(ValueError):
        plots.plot_posterior_comparison(t, prior, mc, vi, events, None, true_path=prior[0, :-1])


def test_posterior_comparison_keeps_coinciding_sets_visible(tmp_path: Path) -> None:
    """A model without spread gives one curve for the prior and both posteriors, and the set drawn last would hide the
    other two: their means are drawn as alternating dashes and the legend says so. Ordinary sets are left alone."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 24.0, 121)
    curve = 3.0 + 20.0 * np.exp(-0.5 * ((t - 10.0) / 3.0) ** 2)
    one_curve = [curve + 1e-3 * rng.standard_normal((16, 121)) for _ in range(3)]
    spread = [curve * np.exp(0.4 * rng.standard_normal((16, 1))) for _ in range(3)]

    def dashed(sets):
        fig, ax = plt.subplots()
        drawn = [(name, ax.plot(t, P.mean(axis=0))[0], P.mean(axis=0), P.std(axis=0)) for name, P in zip("abc", sets)]
        note = plots._dash_coinciding(drawn, tol=0.08)
        out = [d[1].is_dashed() for d in drawn]
        plt.close(fig)
        return note, out

    assert dashed(one_curve) == ("a, b and c coincide (bands thinner than a line): drawn as alternating dashes", [True, True, True])
    assert dashed(spread) == ("", [False, False, False])
    # two thin sets on one curve and a third with a spread: only the two are changed
    assert dashed([spread[0], one_curve[1], one_curve[2]]) == ("b and c coincide (bands thinner than a line): drawn as alternating dashes",
                                                              [False, True, True])
    # a thin set away from the others hides nothing
    assert dashed([one_curve[0], one_curve[1] + 5.0, spread[2]]) == ("", [False, False, False])

    def colours(path):                                         # inside the axes: away from the title, the frame and the legend
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        r, g, b = (img[int(0.12 * h):int(0.70 * h), int(0.10 * w):int(0.90 * w), k] for k in range(3))
        return {"black": int(((r < 0.12) & (g < 0.12) & (b < 0.12)).sum()), "red": int(((r > 0.85) & (g < 0.25) & (b < 0.25)).sum()),
                "blue": int(((b > 0.85) & (r < 0.25) & (g < 0.25)).sum())}

    p = plots.plot_posterior_comparison(t, one_curve[0], one_curve[2], one_curve[1], np.array([]), None, path=tmp_path / "one_curve.png")
    seen = colours(p)
    assert min(seen.values()) > 200, seen                      # prior (black), learned posterior (red) and reference (blue) all show
    # with a spread nothing changes: the figure is the one drawn before this rule existed (solid means, bands)
    q = plots.plot_posterior_comparison(t, spread[0], spread[2], spread[1], np.array([]), None, path=tmp_path / "spread.png")
    _ok(q, tmp_path / "spread.png")


def test_plot_drift_comparison_and_heatmap(smoke_cfg: ExperimentConfig, tiny_model: NDIM, smoke_dataset, tmp_path: Path) -> None:
    truth = make_truth_drift(smoke_cfg.data.truth)
    learned = lambda z, t: tiny_model.drift(z, t)
    p = plots.plot_drift_comparison(learned, truth, 100.0, 4.0, occupancy=smoke_dataset.train_paths, path=tmp_path / "cmp.png", title="Fig. 2")
    _ok(p, tmp_path / "cmp.png")
    # numpy occupancy, no truth, custom times
    p1 = plots.plot_drift_comparison(learned, None, 100.0, 4.0, times=(0.25, 0.75), occupancy=smoke_dataset.train_paths.numpy(), path=tmp_path / "cmp2.png")
    _ok(p1, tmp_path / "cmp2.png")
    p2 = plots.plot_drift_heatmap(learned, truth, 100.0, 4.0, smoke_dataset.train_paths, path=tmp_path / "heat.png", title="heat")
    _ok(p2, tmp_path / "heat.png")
    assert plots.plot_drift_heatmap(learned, None, 100.0, 4.0, None) is None


def test_mass_levels_and_occupation_density(smoke_dataset) -> None:
    tc, zc, H = plots._occupation_density(smoke_dataset.train_paths, 4.0, 100.0, n_z=20)
    assert H.shape == (20, tc.shape[0]) and zc.shape == (20,)
    assert np.all(H >= 0) and H.sum() > 0
    level_fracs = plots._mass_levels(H)
    levels = [l for l, _ in level_fracs]
    fracs = [f for _, f in level_fracs]
    assert levels == sorted(levels) and all(l > 0 for l in levels)
    assert fracs == sorted(fracs, reverse=True)                       # outermost (lowest) level encloses the most mass
    # the super-level set of each level encloses (at least) its fraction of the mass
    for l, f in level_fracs:
        assert H[H >= l].sum() / H.sum() >= f - 1e-9
    assert plots._mass_levels(np.zeros((3, 3))) == []


def test_plot_drift_3d_html(smoke_cfg: ExperimentConfig, tiny_model: NDIM, tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    truth = make_truth_drift(smoke_cfg.data.truth)
    p = plots.plot_drift_3d_html(lambda z, t: tiny_model.drift(z, t), truth, 100.0, 4.0, tmp_path / "d3.html", n_z=8, n_t=6)
    _ok(p, tmp_path / "d3.html")
    assert "plotly" in (tmp_path / "d3.html").read_text(errors="ignore")[:200000].lower()


def test_drift_3d_figure_is_drawn_as_in_the_paper(smoke_cfg: ExperimentConfig, tiny_model: NDIM) -> None:
    """Fig. 2 as a 3-D figure: Z on x, t on y, solid purple and orange surfaces, opacity 0.7."""
    pytest.importorskip("plotly")
    truth = make_truth_drift(smoke_cfg.data.truth)                      # 0.3 (80 - z)
    fig = plots.drift_3d_figure(lambda z, t: tiny_model.drift(z, t), truth, 100.0, 4.0, n_z=8, n_t=6)
    learned, true = fig.data
    assert (learned.name, true.name) == ("Neural Network Drift", "True Drift")
    for surface, colour in ((learned, "purple"), (true, "orange")):
        assert surface.type == "surface" and surface.opacity == 0.7 and surface.showscale is False
        assert [c for _, c in surface.colorscale] == [colour, colour]    # one colour: a solid surface
        assert surface.x.shape == surface.y.shape == surface.z.shape == (6, 8)   # [n_t, n_z]
        assert surface.x[0, 0] == 0.0 and surface.x[0, -1] == 100.0 and (surface.x[:, 3] == surface.x[0, 3]).all()   # x is Z
        assert surface.y[0, 0] == 0.0 and surface.y[-1, 0] == 4.0 and (surface.y[2, :] == surface.y[2, 0]).all()     # y is t
    assert true.z[0, 0] == pytest.approx(24.0) and true.z[-1, -1] == pytest.approx(-6.0, abs=1e-4)   # 0.3 (80 - Z)
    scene = fig.layout.scene
    assert (scene.xaxis.title.text, scene.yaxis.title.text, scene.zaxis.title.text) == ("Z", "t", "Drift b(Z, t)")
    assert fig.layout.title.text == "Learned Neural Network Drift" and fig.layout.legend.title.text == "Legend"
    assert len(plots.drift_3d_figure(lambda z, t: tiny_model.drift(z, t), None, 100.0, 4.0, n_z=4, n_t=4).data) == 1
    assert plots.drift_3d_figure(truth, truth, 100.0, 4.0).data[0].z.shape == (100, 100)    # the paper's grid


def test_plot_drift_3d_html_without_plotly(monkeypatch: pytest.MonkeyPatch, tiny_model: NDIM, tmp_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("no plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = plots.plot_drift_3d_html(lambda z, t: tiny_model.drift(z, t), None, 100.0, 4.0, tmp_path / "missing.html")
    assert out is None and not (tmp_path / "missing.html").exists()


def test_plot_wasserstein_bars(tmp_path: Path) -> None:
    p = plots.plot_wasserstein_bars({"train VI": 1.0, "test VI": 1.5, "test prior baseline": 3.0}, path=tmp_path / "bars.png",
                                    errors={"train VI": 0.2, "test VI": 0.3}, title="W2")
    _ok(p, tmp_path / "bars.png")
    assert plots.plot_wasserstein_bars({"a": float("nan"), "b": 1.0}) is None       # NaN entries are tolerated


def test_plot_mcmc_traces(tmp_path: Path) -> None:
    traces = {"a": np.linspace(-10, 0, 50), "b": torch.linspace(-20, -5, 50)}
    p = plots.plot_mcmc_traces(traces, n_burn_in=25, path=tmp_path / "tr.png", title="traces")
    _ok(p, tmp_path / "tr.png")
    assert plots.plot_mcmc_traces({}) is None


def test_posterior_comparison_draws_the_true_latent_path(monkeypatch):
    import matplotlib.pyplot as plt
    seen = {}
    def inspect_and_close(fig, path):
        seen["labels"] = fig.axes[0].get_legend_handles_labels()[1]
        seen["ylim"] = fig.axes[0].get_ylim()
        plt.close(fig)
    monkeypatch.setattr(plots, "_finish_or_close", inspect_and_close)
    t = np.linspace(0, 4, 5)
    P = np.full((3, 5), 10.0) + np.arange(3)[:, None]
    plots.plot_posterior_comparison(t, P, P, P, [1.0], None)
    assert "true latent path of this observation" not in seen["labels"]
    plots.plot_posterior_comparison(t, P, P, P, [1.0], None, true_path=np.array([30.0, 28.0, 25.0, 22.0, 40.0]))
    assert "true latent path of this observation" in seen["labels"]
    assert seen["ylim"][0] < 10.0 and seen["ylim"][1] > 40.0          # the axis makes room for the path


def test_analytic_reference_labels_do_not_say_mcmc(monkeypatch):
    import matplotlib.pyplot as plt
    labels = []
    def inspect_and_close(fig, path):
        for axis in fig.axes:
            labels.extend(axis.get_legend_handles_labels()[1])
            labels.append(axis.get_ylabel())
        plt.close(fig)
    monkeypatch.setattr(plots, "_finish_or_close", inspect_and_close)
    times = np.linspace(0, 4, 3)
    paths = np.ones((2, 3))
    plots.plot_posterior_comparison(times, paths, paths, paths, [], None,
                                    reference_label="true posterior (exact CIR)")
    assert any("exact CIR" in label for label in labels)
    assert not any("MCMC" in label for label in labels)


def test_no_figures_leak(tmp_path: Path) -> None:
    """Every function closes its figure (saved or not), so long evaluation runs do not accumulate figures."""
    import matplotlib.pyplot as plt

    plt.close("all")
    t = np.linspace(0, 4, 11)
    P = np.random.default_rng(0).random((3, 11))
    plots.plot_posterior_comparison(t, P, P, P, np.array([1.0]), None)
    plots.plot_posterior_comparison(t, P, P, P, np.array([1.0]), None, path=tmp_path / "x.png")
    plots.plot_cumulative_counts([np.array([1.0])], [np.array([2.0])], 4.0, n_grid=5)
    plots.plot_wasserstein_bars({"a": 1.0})
    assert len(plt.get_fignums()) == 0
