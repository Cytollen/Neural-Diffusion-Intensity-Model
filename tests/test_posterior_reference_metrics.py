"""The comparison reference follows ``eval.posterior_reference`` and the CIR truth."""
import copy

import numpy as np
import pytest
import torch

from ndim import cir_posterior, metrics
from ndim.mcmc import MCMCDiagnostics


def _forbidden(*args, **kwargs):
    raise AssertionError("The analytic CIR comparison must not invoke MCMC")


@pytest.fixture
def analytic_stub(monkeypatch):
    calls = []

    def sample(events, cfg, n_prime, n_paths, device="cpu", generator=None):
        calls.append((n_prime, n_paths, cfg.data.truth.eta))
        paths = torch.full((len(events) * n_paths, cfg.data.n_steps + 1), 10.,
                           dtype=torch.float64, device=device)
        return paths, {"label": "Exact CIR posterior", "exact_finite_dimensional": True,
                       "truth_eta": cfg.data.truth.eta}

    monkeypatch.setattr(cir_posterior, "sample_true_cir_posterior", sample)
    monkeypatch.setattr(metrics, "sample_posterior_mcmc_batch", _forbidden)
    return calls


def test_reference_resolution_follows_the_config_and_checks_the_truth(smoke_cfg):
    cfg = copy.deepcopy(smoke_cfg)
    assert cfg.model.drift.kind == "mlp"                     # the learned prior is not CIR: the reference is the truth's
    assert metrics.resolve_posterior_reference(cfg) == "analytic"
    cfg.data.truth.theta = 0.                                # no immigration: the CIR posterior formulas do not apply
    with pytest.raises(ValueError, match="positive immigration"):
        metrics.resolve_posterior_reference(cfg)
    assert metrics.resolve_posterior_reference(cfg, "mcmc") == "mcmc"
    cfg.eval.posterior_reference = "none"
    assert metrics.resolve_posterior_reference(cfg) == "none"
    with pytest.raises(ValueError, match="Unknown posterior reference"):
        metrics.resolve_posterior_reference(cfg, "auto")


def test_reference_samples_ignore_learned_coefficients(smoke_cfg, analytic_stub, device):
    cfg = copy.deepcopy(smoke_cfg)
    cfg.data.truth.eta = .85
    events = torch.tensor([[.2, torch.inf], [torch.inf, torch.inf]])
    paths, metadata = metrics.posterior_reference_samples(events, cfg, 3, 5, device,
                                                           drift=_forbidden, diffusion=_forbidden)
    assert paths.shape == (10, cfg.data.n_steps + 1)
    assert analytic_stub == [(3, 5, .85)]
    assert metadata["kind"] == "analytic" and metadata["exact_finite_dimensional"]


def test_analytic_reference_uses_cpu_generator_independently_of_output_device(monkeypatch, smoke_cfg):
    generator = torch.Generator(device="cpu").manual_seed(17)
    observed = []

    def sample(events, cfg, n_prime, n_paths, device="cpu", generator=None):
        observed.append((events.device, torch.device(device), generator))
        return torch.ones(len(events) * n_paths, cfg.data.n_steps + 1), {
            "label": "Exact CIR posterior", "exact_finite_dimensional": True}

    monkeypatch.setattr(cir_posterior, "sample_true_cir_posterior", sample)
    # A meta destination exercises device separation without requiring CUDA.
    paths, metadata = metrics.posterior_reference_samples(
        torch.tensor([[.2]]), smoke_cfg, 3, 2, torch.device("meta"), generator)
    assert observed == [(torch.device("cpu"), torch.device("cpu"), generator)]
    assert paths.device.type == "meta" and paths.shape == (2, smoke_cfg.data.n_steps + 1)
    assert metadata["sampling_device"] == "cpu"


def test_wasserstein_uses_the_configured_reference_and_has_no_mcmc_fields(
        smoke_cfg, tiny_model, analytic_stub, device):
    events = torch.tensor([[.2], [.4]])
    result = metrics.posterior_wasserstein(
        tiny_model, _forbidden, _forbidden, events, smoke_cfg, smoke_cfg.data.n_steps,
        2, smoke_cfg.eval.mcmc, device, torch.Generator().manual_seed(4))
    assert len(result["w2_vi"]) == 2
    assert result["reference"]["kind"] == "analytic"
    assert "seconds_reference" in result
    assert "seconds_mcmc" not in result and "mcmc_acceptance" not in result


def test_mcmc_is_explicit_and_preserves_trace_metadata(monkeypatch, smoke_cfg, device):
    calls = []

    def sample(drift, diffusion, events, cfg, n_prime, used, device, generator, **kwargs):
        calls.append(used.n_chains)
        diagnostics = [MCMCDiagnostics(.4, .2, 3, used.n_chains, np.arange(3.)) for _ in events]
        return torch.ones(len(events), used.n_chains, cfg.data.n_steps + 1), diagnostics

    monkeypatch.setattr(metrics, "sample_posterior_mcmc_batch", sample)
    paths, metadata = metrics.posterior_reference_samples(
        torch.tensor([[.2], [.4]]), smoke_cfg, 4, 3, device, reference="mcmc")
    assert calls == [3] and paths.shape[0] == 6
    assert metadata["kind"] == "mcmc" and metadata["acceptance"] == .4
    assert metadata["mcmc_diagnostics"][0]["loglik_trace"] == [0., 1., 2.]
