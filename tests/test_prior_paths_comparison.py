"""The prior comparison must visualize the same paired paths as its metric."""
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ndim import evaluate, metrics, plots
from ndim.data import make_truth_diffusion, make_truth_drift


def test_optional_paths_reuse_one_brownian_bank_and_preserve_metrics(cfg_copy, device, monkeypatch):
    cfg_copy.model.state_space = "intensity"
    drift = make_truth_drift(cfg_copy.data.truth)
    diffusion = make_truth_diffusion(cfg_copy.data.truth)
    different = lambda z, t: drift(z, t) - 2.
    original = metrics._prior_paths
    banks = []
    def capture(drift_fn, diffusion_fn, cfg, W, **kwargs):
        banks.append(W)
        return original(drift_fn, diffusion_fn, cfg, W, **kwargs)
    monkeypatch.setattr(metrics, "_prior_paths", capture)
    result, paths = metrics.expected_path_deviation(different, drift, diffusion, cfg_copy, 8, device,
                                                    torch.Generator().manual_seed(182), return_paths=True)
    assert len(banks) == 2 and banks[0] is banks[1]
    assert paths["true"].shape == paths["learned"].shape == (8, cfg_copy.data.n_steps + 1)
    measured = ((paths["learned"] - paths["true"]).double().square().sum(1) * cfg_copy.dt).mean()
    assert result["l2"] == pytest.approx(float(measured), abs=1e-12)
    default = metrics.expected_path_deviation(different, drift, diffusion, cfg_copy, 8, device,
                                              torch.Generator().manual_seed(182))
    assert result == default


def test_identical_prior_coefficients_return_identical_paired_paths(cfg_copy, device):
    cfg_copy.model.state_space = "intensity"
    drift, diffusion = make_truth_drift(cfg_copy.data.truth), make_truth_diffusion(cfg_copy.data.truth)
    result, paths = metrics.expected_path_deviation(drift, drift, diffusion, cfg_copy, 16, device,
                                                    torch.Generator().manual_seed(8), return_paths=True)
    torch.testing.assert_close(paths["true"], paths["learned"], atol=0., rtol=0.)
    assert result["l2"] == 0.


def test_l2_stage_writes_plot_and_only_small_json_metadata(smoke_cfg, tiny_model, smoke_dataset, tmp_path):
    ctx = evaluate._Ctx(cfg=smoke_cfg, model=tiny_model, ds=smoke_dataset,
                        truth_drift=make_truth_drift(smoke_cfg.data.truth),
                        truth_diffusion=make_truth_diffusion(smoke_cfg.data.truth), device=torch.device("cpu"),
                        out_dir=tmp_path, run_dir=tmp_path, logger=logging.getLogger("test.prior_paths"),
                        history=[], ckpt_extra={})
    result = evaluate._stage_l2(ctx)
    metadata = result["path_comparison"]
    assert metadata["shared_brownian_increments"] is True
    assert metadata["true_scheme"] == "Intensity Euler (full truncation)"
    assert "exponential" in metadata["learned_scheme"]
    assert (tmp_path / "prior_paths_comparison.png").stat().st_size > 1000
    encoded = json.dumps(ctx.metrics)
    assert len(encoded) < 2500
    assert "true_terminal_mean" in encoded and "learned_terminal_mean" in encoded
    assert "\"true\": [[" not in encoded and "\"learned\": [[" not in encoded


def test_plot_supports_numpy_and_tensor_inputs_and_single_pair(tmp_path):
    times = np.linspace(0., 1., 5)
    truth = torch.tensor([[5., 10., 12., 15., 17.]])
    learned = np.array([[5., 9., 11., 14., 16.]])
    path = tmp_path / "paired.png"
    assert plots.plot_prior_paths_comparison(times, truth, learned, path=path,
                                             learned_scheme="Latent Euler with exponential link") == path
    assert path.stat().st_size > 1000
    assert plots.plot_prior_paths_comparison(times, truth, learned) is None
    assert not plt.get_fignums()


@pytest.mark.parametrize("bad", [np.ones((2, 4)), np.full((1, 5), np.nan)])
def test_plot_rejects_unpaired_or_nonfinite_paths(bad):
    with pytest.raises(ValueError):
        plots.plot_prior_paths_comparison(np.arange(5), np.ones((1, 5)), bad)
