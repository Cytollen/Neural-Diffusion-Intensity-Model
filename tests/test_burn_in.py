"""Burn-in (``model.burn_in``): the diffusion of the model starts before the observation window, from a fixed latent
state, so that the distribution of the initial intensity ``Z_0`` is learned; and the truth's random ``Z_0``
(``data.truth.z0_std``). See docs/BURN_IN.md."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ndim.config import MCMCConfig
from ndim.data import (dataset_filename, generate_dataset, load_or_generate_dataset, make_truth_diffusion, make_truth_drift,
                       sample_truth_z0, simulate_truth_paths, truth_z0_cdf, truth_z0_from_normal, truth_z0_pdf)
from ndim.elbo import compute_objective, evaluate_elbo
from ndim.likelihood import poisson_loglik
from ndim.mcmc import sample_posterior_mcmc_batch
from ndim.metrics import expected_path_deviation, generate_observations_from_model
from ndim.model import NDIM, build_model
from ndim.sde import brownian_increments, simulate_prior
from ndim.train import train

CPU = torch.device("cpu")
BURN = ["model.burn_in=0.96", "model.x_start=0.0", "model.drift.time_input=true"]      # 6 steps of the smoke grid (dt = 0.16)
RANDOM_Z0 = ["data.truth.z0=50.0", "data.truth.z0_std=1.0", "data.truth.z0_low=20.0", "data.truth.z0_high=80.0", "eval.posterior_reference=mcmc"]
# a two-mode initial intensity: normals at 35 and at 65 with standard deviation 5, truncated to [20, 80]
BIMODAL = ["data.truth.z0=50.0", "data.truth.z0_modes=[35.0, 65.0]", "data.truth.z0_std=5.0", "data.truth.z0_low=20.0", "data.truth.z0_high=80.0",
           "eval.posterior_reference=mcmc"]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", [
    ["model.burn_in=-1.0"],
    ["model.burn_in=0.01"],                                        # shorter than half a step of the smoke grid
    ["model.burn_in=0.96", "model.state_space=intensity"],        # the burn-in is for the latent state
    ["model.burn_in=0.96", "train.grad_mode=sensitivity"],
    ["data.truth.z0_std=-1.0"],
    ["data.truth.z0_std=1.0", "data.truth.z0_low=60.0", "data.truth.z0_high=40.0", "eval.posterior_reference=mcmc"],
    ["data.truth.z0_std=1.0"],                                     # the exact CIR posterior of base.yaml needs a fixed Z_0
])
def test_validation(make_cfg, overrides) -> None:
    with pytest.raises(ValueError):
        make_cfg(*overrides)


# --------------------------------------------------------------------------------------
# Truth with a random initial intensity
# --------------------------------------------------------------------------------------
def test_truth_initial_intensity_distribution(make_cfg) -> None:
    truth = make_cfg(*RANDOM_Z0).data.truth
    z = sample_truth_z0(truth, 20000, torch.Generator().manual_seed(0))
    assert float(z.mean()) == pytest.approx(50.0, abs=0.03) and float(z.std()) == pytest.approx(1.0, abs=0.03)
    assert float(z.min()) >= 20.0 and float(z.max()) <= 80.0
    zeta = torch.linspace(-3, 3, 7)
    assert torch.allclose(truth_z0_from_normal(truth, zeta), 50.0 + zeta)          # the bounds are 30 sd away: the normal itself
    tight = make_cfg("data.truth.z0=50.0", "data.truth.z0_std=10.0", "data.truth.z0_low=45.0", "data.truth.z0_high=60.0",
                     "eval.posterior_reference=mcmc").data.truth
    t = sample_truth_z0(tight, 20000, torch.Generator().manual_seed(0))
    assert float(t.min()) >= 45.0 and float(t.max()) <= 60.0 and 50.0 < float(t.mean()) < 53.5   # mass cut off on both sides
    mapped = truth_z0_from_normal(tight, torch.linspace(-4, 4, 41))
    assert torch.all(mapped[1:] >= mapped[:-1])                                     # a quantile map: monotone
    fixed = make_cfg().data.truth                                                   # z0_std = 0: every path starts at z0
    g = torch.Generator().manual_seed(3)
    before = g.get_state()
    assert torch.all(sample_truth_z0(fixed, 5, g) == 5.0) and torch.equal(g.get_state(), before)   # and no random number is used


def test_dataset_with_a_random_initial_intensity(make_cfg) -> None:
    cfg = make_cfg(*RANDOM_Z0, "data.n_train=200")
    ds = generate_dataset(cfg.data)
    z0 = ds.train_paths[:, 0]
    assert float(z0.mean()) == pytest.approx(50.0, abs=0.3) and 0.8 < float(z0.std()) < 1.2
    assert "z0sd1_20-80" in dataset_filename(cfg.data) and dataset_filename(cfg.data) != dataset_filename(make_cfg("data.truth.z0=50.0").data)
    fixed = make_cfg()
    assert "z0sd" not in dataset_filename(fixed.data)                               # the fixed-start datasets keep their file names
    paths = simulate_truth_paths(fixed.data.truth, 4.0, 25, 7, torch.Generator().manual_seed(0))
    assert torch.all(paths[:, 0] == 5.0)


# --------------------------------------------------------------------------------------
# Model: simulation from t = -burn_in
# --------------------------------------------------------------------------------------
def test_times_start_at_minus_burn_in() -> None:
    seen = []

    def drift(z, t):
        seen.append(float(t[0]))
        return torch.zeros_like(z)

    W = torch.zeros(2, 5)
    out = simulate_prior(drift, lambda z, t: torch.zeros_like(z), 3.0, 5, 0.25, W, t0=-0.5)
    assert seen == pytest.approx([-0.5, -0.25, 0.0, 0.25, 0.5]) and torch.all(out == 3.0)
    raw = simulate_prior(drift, lambda z, t: torch.zeros_like(z), 3.0, 5, 0.25, W, initial_raw=-1.5, return_raw=True)
    assert torch.all(raw == -1.5)                                                   # a latent start value replaces z0
    per_row = simulate_prior(drift, lambda z, t: torch.zeros_like(z), torch.tensor([2.0, 7.0]), 5, 0.25, W)
    assert per_row[:, 0].tolist() == [2.0, 7.0]                                     # one initial intensity per path


def test_posterior_paths_cover_the_burn_in(make_cfg, smoke_dataset) -> None:
    cfg = make_cfg(*BURN)
    torch.manual_seed(0)
    model = build_model(cfg)
    nb, N, n_p, B, m = model.n_burn, 25, 10, 3, 2
    assert nb == 6 and model.t_start == pytest.approx(-0.96) and model.initial_state == 0.0
    events = smoke_dataset.train.events[:B]
    W = brownian_increments(B * m, nb + N, cfg.dt, CPU, torch.Generator().manual_seed(1))
    sim = model.simulate_posterior(events, n_p, W, N)
    assert sim.n_burn == nb and sim.Z.shape == (B * m, nb + N + 1) and sim.u.shape == (B * m, nb + N)
    assert torch.all(sim.raw[:, 0] == 0.0) and torch.all(sim.Z[:, 0] == 1.0)        # X = 0 at t = -burn_in: intensity exp(0) = 1
    assert float(sim.u[:, :nb].abs().sum()) > 0.0                                   # the correction acts on the burn-in
    assert float(sim.u[:, nb : nb + n_p].abs().sum()) > 0.0 and float(sim.u[:, nb + n_p :].abs().sum()) == 0.0   # and stops at T'
    with pytest.raises(ValueError, match="burn-in"):
        model.simulate_posterior(events, n_p, W[:, : N], N)                         # noise for the burn-in steps is missing
    # the score network is asked at the grid times -0.96, -0.80, ...: all events are still in the future there
    ctx = model.score_net.prepare(events, model.t_start + torch.arange(nb + n_p) * cfg.dt, n_p * cfg.dt)
    seen = (events <= n_p * cfg.dt).sum(dim=1).float()
    assert float(ctx.t_grid[0]) == pytest.approx(-0.96) and torch.equal(ctx.r[:, :nb], seen[:, None].expand(B, nb))
    # with the score switched off the posterior is the prior on the whole grid (small noise keeps the untrained paths in range)
    off = build_model(make_cfg(*BURN, "model.posterior.score_scale=0.0"))
    off.drift.load_state_dict(model.drift.state_dict())
    quiet = 0.05 * W
    sim0 = off.simulate_posterior(events, n_p, quiet, N)
    assert float(sim0.u.abs().max()) == 0.0 and torch.allclose(sim0.Z, off.simulate_prior(quiet, N, full=True), atol=1e-5)
    assert off.simulate_prior(quiet, N).shape == (B * m, N + 1)                     # the observation window alone


def test_elbo_uses_the_window_likelihood_and_the_whole_control_energy(make_cfg, smoke_dataset) -> None:
    cfg = make_cfg(*BURN)
    torch.manual_seed(0)
    model = build_model(cfg)
    nb, n_p, B, m = model.n_burn, 25, 3, 2
    events = smoke_dataset.train.events[:B]
    W = 0.05 * brownian_increments(B * m, nb + n_p, cfg.dt, CPU, torch.Generator().manual_seed(1))   # small noise, see above
    out = compute_objective(model, events, n_p, m, W, cfg)
    sim = out.sim
    window = poisson_loglik(sim.Z[:, nb:], events, m, cfg.dt, n_p, cfg.likelihood).view(B, m).mean(dim=1)
    assert torch.allclose(out.loglik, window, atol=1e-4)                            # events and integral on [0, T'] only
    whole = poisson_loglik(sim.Z, events, m, cfg.dt, n_p, cfg.likelihood).view(B, m).mean(dim=1)
    assert not torch.allclose(out.loglik, whole, atol=1e-2)                         # not the path read from t = -burn_in
    energy = 0.5 * (sim.u ** 2).sum(dim=1) * cfg.dt
    assert torch.allclose(out.kl, energy.view(B, m).mean(dim=1), atol=1e-5)         # the Girsanov integral from -burn_in to T'
    on_window = 0.5 * (sim.u[:, nb:] ** 2).sum(dim=1) * cfg.dt
    assert float(energy.mean()) > float(on_window.mean())                           # the burn-in control is paid for
    assert torch.allclose(out.elbo, out.loglik - out.kl, atol=1e-4)
    out.loss.backward()
    assert all(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params) for params in model.param_groups().values())
    res = evaluate_elbo(model, events, n_p, 2, cfg, CPU, torch.Generator().manual_seed(0))
    assert set(res) == {"loglik", "kl", "elbo"}                                    # draws its own noise for burn-in + window


def test_without_a_burn_in_nothing_changes(make_cfg, smoke_dataset) -> None:
    cfg = make_cfg()
    torch.manual_seed(0)
    model = build_model(cfg)
    assert model.n_burn == 0 and model.t_start == 0.0 and model.initial_state == pytest.approx(math.log(5.0))
    events = smoke_dataset.train.events[:2]
    W = brownian_increments(4, 25, cfg.dt, CPU, torch.Generator().manual_seed(1))
    out = compute_objective(model, events, 25, 2, W, cfg)
    assert out.sim.n_burn == 0 and out.sim.Z.shape == (4, 26) and torch.allclose(out.sim.Z[:, 0], torch.full((4,), 5.0))


# --------------------------------------------------------------------------------------
# Reference sampler and metrics with a random initial intensity
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["independence", "pcn"])
def test_mcmc_moves_the_initial_intensity(make_cfg, kind: str) -> None:
    cfg = make_cfg("data.truth.z0=50.0", "data.truth.z0_std=8.0", "data.truth.z0_low=20.0", "data.truth.z0_high=80.0", "eval.posterior_reference=mcmc")
    ds = generate_dataset(cfg.data)
    truth = cfg.data.truth
    order = torch.argsort(ds.train_paths[:, 0])
    pick = torch.stack([order[0], order[-1]])                                       # observations whose true Z_0 is lowest / highest
    mcmc = MCMCConfig(kind=kind, n_chains=48, n_steps=300)
    paths, diags = sample_posterior_mcmc_batch(make_truth_drift(truth), make_truth_diffusion(truth), ds.train.events[pick], cfg, 25, mcmc, CPU,
                                               torch.Generator().manual_seed(0))
    z0 = paths[:, :, 0]
    assert paths.shape == (2, 48, 26) and torch.isfinite(paths).all()
    assert float(z0.min()) >= 20.0 and float(z0.max()) <= 80.0 and float(z0.std(dim=1).min()) > 0.5   # Z_0 is sampled, not fixed
    assert float(z0[1].mean()) > float(z0[0].mean())                                # and follows the observation
    fixed, _ = sample_posterior_mcmc_batch(make_truth_drift(truth), make_truth_diffusion(truth), ds.train.events[pick], cfg, 25, mcmc, CPU,
                                           torch.Generator().manual_seed(0), z0=33.0)
    assert torch.all(fixed[:, :, 0] == 33.0)                                        # a given z0 switches the draws off


def test_metrics_with_a_burn_in(make_cfg) -> None:
    cfg = make_cfg(*BURN, *RANDOM_Z0)
    torch.manual_seed(0)
    model = build_model(cfg)
    events, paths = generate_observations_from_model(model, cfg, 6, CPU, torch.Generator().manual_seed(0))
    assert len(events) == 6 and paths.shape == (6, 26)                              # the observation window of the learned prior
    truth = cfg.data.truth
    _, both = expected_path_deviation(model.drift, make_truth_drift(truth), model.diffusion, cfg, 8, CPU, torch.Generator().manual_seed(0),
                                      true_diffusion_fn=make_truth_diffusion(truth), return_paths=True)
    # the dynamics are compared from a common draw of the truth's Z_0, not from the model's burn-in
    assert torch.allclose(both["learned"][:, 0], both["true"][:, 0], atol=1e-4) and 45.0 < float(both["true"][:, 0].mean()) < 55.0


# --------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------
def test_train_and_evaluate_with_a_burn_in(make_cfg) -> None:
    from ndim.evaluate import evaluate_run

    cfg = make_cfg("name=smoke_pytest_burn_in", *BURN, *RANDOM_Z0, "train.max_abs_control=30", "eval.n_initial=200")
    res = train(cfg)
    assert len(res.history) == 2 and all(h["n_updates"] + h["n_skipped"] == 10 for h in res.history)
    assert "burn-in: the diffusion of the model starts at t = -0.96" in (res.output_dir / "train.log").read_text()
    loaded, _ = NDIM.load_checkpoint(res.output_dir / "ckpt_last.pt")
    assert loaded.n_burn == 6 and loaded.cfg.data.truth.z0_std == 1.0               # the checkpoint carries the burn-in and the truth

    metrics = evaluate_run(res.output_dir, ckpt="last", which=["curves", "initial", "posterior"])
    assert metrics["errors"] == [], metrics["errors"]
    initial = metrics["initial"]
    assert initial["burn_in"] == pytest.approx(0.96) and initial["x_start"] == 0.0 and initial["n_samples"] == 200
    assert initial["true"]["mean"] == pytest.approx(50.0, abs=0.3) and 0.8 < initial["true"]["std"] < 1.2
    assert initial["learned"]["std"] > 0 and initial["wasserstein1"] >= 0 and 0.0 <= initial["ks_statistic"] <= 1.0
    assert 0.0 <= initial["ks_statistic_exact"] <= 1.0 and "modes" not in initial        # against the exact truncated normal
    assert (res.output_dir / "eval" / "initial_distribution.png").stat().st_size > 0
    case = metrics["posterior"]["cases"]["most_events_full"]
    assert case["z0_vi"]["std"] > 0 and case["z0_reference"]["std"] > 0             # the posterior of Z_0 is a distribution on both sides
    assert 20.0 <= case["true_z0"] <= 80.0 and metrics["posterior"]["reference_kind"] == "mcmc"

    plain = train(make_cfg("name=smoke_pytest_no_burn_in", "train.epochs=1", "train.eval_paths=1"))
    skipped = evaluate_run(plain.output_dir, ckpt="last", which=["initial"])
    assert skipped["stages"]["initial"]["status"] == "skipped" and not (plain.output_dir / "eval" / "initial_distribution.png").exists()


# --------------------------------------------------------------------------------------
# A truth whose initial intensity has two modes
# --------------------------------------------------------------------------------------
def test_two_mode_initial_intensity(make_cfg) -> None:
    truth = make_cfg(*BIMODAL).data.truth
    z = sample_truth_z0(truth, 40000, torch.Generator().manual_seed(0)).to(torch.float64)
    x = torch.linspace(20.0, 80.0, 6001, dtype=torch.float64)
    pdf = truth_z0_pdf(truth, x)
    mean = float(torch.trapz(x * pdf, x))
    sd = float(torch.trapz((x - mean) ** 2 * pdf, x).sqrt())
    assert float(torch.trapz(pdf, x)) == pytest.approx(1.0, abs=1e-6) and mean == pytest.approx(50.0, abs=1e-6)
    assert sd == pytest.approx(math.sqrt(15.0 ** 2 + 5.0 ** 2), abs=0.1)                # little is cut off: 3 sd beyond each mode
    assert float(z.min()) >= 20.0 and float(z.max()) <= 80.0
    assert float(z.mean()) == pytest.approx(50.0, abs=0.3) and float(z.std()) == pytest.approx(sd, abs=0.15)
    assert float((z < 50.0).to(torch.float64).mean()) == pytest.approx(0.5, abs=0.01)   # half of the mass on each mode
    valley = float(truth_z0_cdf(truth, torch.tensor(55.0)) - truth_z0_cdf(truth, torch.tensor(45.0)))
    assert valley == pytest.approx(0.0227, abs=0.0005)                                  # and almost none between them
    assert float(((z > 45.0) & (z < 55.0)).to(torch.float64).mean()) == pytest.approx(valley, abs=0.004)
    # the sampler follows the exact distribution function, and the map from a standard normal is increasing
    zs = torch.sort(z).values
    steps = torch.arange(zs.numel() + 1, dtype=torch.float64) / zs.numel()
    cdf = truth_z0_cdf(truth, zs)
    assert float(torch.maximum((steps[1:] - cdf).abs(), (steps[:-1] - cdf).abs()).max()) < 0.01
    mapped = truth_z0_from_normal(truth, torch.linspace(-4, 4, 201))
    assert torch.all(mapped[1:] >= mapped[:-1]) and float(mapped[0]) < 35.0 < 65.0 < float(mapped[-1])
    assert float(truth_z0_cdf(truth, torch.tensor(20.0))) == 0.0 and float(truth_z0_cdf(truth, torch.tensor(80.0))) == 1.0
    assert float(truth_z0_pdf(truth, torch.tensor(19.0))) == 0.0 and float(truth_z0_pdf(truth, torch.tensor(81.0))) == 0.0
    # unequal weights
    uneven = make_cfg(*BIMODAL, "data.truth.z0_weights=[3.0, 1.0]").data.truth
    u = sample_truth_z0(uneven, 40000, torch.Generator().manual_seed(1))
    assert float((u < 50.0).to(torch.float64).mean()) == pytest.approx(0.75, abs=0.01)
    assert "_w3-1_" in dataset_filename(make_cfg(*BIMODAL, "data.truth.z0_weights=[3.0, 1.0]").data)
    # one normal: the exact functions are those of the truncated normal, and a fixed start is a step
    single = make_cfg(*RANDOM_Z0).data.truth
    assert float(truth_z0_cdf(single, torch.tensor(50.0))) == pytest.approx(0.5) and float(truth_z0_pdf(single, torch.tensor(50.0))) == pytest.approx(1 / math.sqrt(2 * math.pi))
    fixed = make_cfg().data.truth
    assert truth_z0_cdf(fixed, torch.tensor([4.9, 5.0, 5.1])).tolist() == [0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        truth_z0_pdf(fixed, torch.tensor(5.0))


@pytest.mark.parametrize("overrides", [
    ["data.truth.z0_modes=[35.0]", "data.truth.z0_std=5.0", "data.truth.z0_high=80.0"],                    # one mean is not a mixture
    ["data.truth.z0_modes=[35.0, 65.0]", "data.truth.z0_high=80.0"],                                        # no standard deviation
    ["data.truth.z0_modes=[35.0, 65.0]", "data.truth.z0_std=5.0"],                                          # no upper bound
    ["data.truth.z0_modes=[35.0, 95.0]", "data.truth.z0_std=5.0", "data.truth.z0_low=20.0", "data.truth.z0_high=80.0"],   # a mode outside
    ["data.truth.z0_modes=[35.0, 65.0]", "data.truth.z0_weights=[1.0]", "data.truth.z0_std=5.0", "data.truth.z0_high=80.0"],
    ["data.truth.z0_modes=[35.0, 65.0]", "data.truth.z0_weights=[1.0, -1.0]", "data.truth.z0_std=5.0", "data.truth.z0_high=80.0"],
    ["data.truth.z0_weights=[1.0, 1.0]", "data.truth.z0_std=5.0", "data.truth.z0_high=80.0"],              # weights without modes
])
def test_two_mode_validation(make_cfg, overrides) -> None:
    with pytest.raises(ValueError):
        make_cfg("data.truth.z0=50.0", *overrides, "eval.posterior_reference=mcmc")


def test_mcmc_with_two_modes(make_cfg) -> None:
    cfg = make_cfg(*BIMODAL, "data.n_train=40")
    ds = generate_dataset(cfg.data)
    truth = cfg.data.truth
    z_true = ds.train_paths[:, 0]
    assert float(z_true.min()) >= 20.0 and float(z_true.max()) <= 80.0 and int((z_true < 50).sum()) > 5 and int((z_true > 50).sum()) > 5
    order = torch.argsort(z_true)
    pick = torch.stack([order[0], order[-1]])                                       # one observation from each mode
    paths, _ = sample_posterior_mcmc_batch(make_truth_drift(truth), make_truth_diffusion(truth), ds.train.events[pick], cfg, 25,
                                           MCMCConfig(kind="pcn", n_chains=48, n_steps=300), CPU, torch.Generator().manual_seed(0))
    z0 = paths[:, :, 0]
    assert torch.isfinite(paths).all() and float(z0.min()) >= 20.0 and float(z0.max()) <= 80.0
    assert float(z0[0].mean()) < 50.0 < float(z0[1].mean())                         # the chains settle on the mode of the observation


def test_evaluation_reports_the_modes(make_cfg) -> None:
    from ndim.evaluate import evaluate_run

    cfg = make_cfg("name=smoke_pytest_burn_in_bimodal", *BURN, *BIMODAL, "train.epochs=1", "train.eval_paths=1", "train.max_abs_control=30",
                   "eval.n_initial=300")
    res = train(cfg)
    metrics = evaluate_run(res.output_dir, ckpt="last", which=["initial"])
    assert metrics["errors"] == [], metrics["errors"]
    initial = metrics["initial"]
    modes = initial["modes"]
    assert modes["means"] == [35.0, 65.0] and modes["cuts"] == [50.0] and modes["std"] == 5.0
    assert modes["mass_true"] == pytest.approx([0.5, 0.5], abs=1e-6) and sum(modes["mass_learned"]) == pytest.approx(1.0)
    assert modes["valley_mass_true"] == pytest.approx([0.0227], abs=0.0005) and 0.0 <= modes["valley_mass_learned"][0] <= 1.0
    assert 0.0 <= initial["ks_statistic_exact"] <= 1.0 and (res.output_dir / "eval" / "initial_distribution.png").stat().st_size > 0


# --------------------------------------------------------------------------------------
# The burn-in by default: observations of an unspecified model
# --------------------------------------------------------------------------------------
def test_burn_in_is_automatic_for_observations_of_an_unspecified_model(make_cfg, smoke_dataset, tmp_root: Path) -> None:
    from ndim.config import burn_in_length, model_x_start
    from ndim.data import estimate_initial_intensity, resolve_x_start

    synthetic = make_cfg()
    assert synthetic.model.burn_in == "auto" and synthetic.model.x_start == "auto"
    assert burn_in_length(synthetic) == 0.0 and synthetic.burn_steps == 0             # a known truth: the model starts at its Z_0
    path = tmp_root / "events_for_the_burn_in.npz"
    np.savez(path, train_events=smoke_dataset.train.events.numpy(), test_events=smoke_dataset.test.events.numpy(), T=np.array(4.0))
    file_args = ["data.source=file", f"data.path={json.dumps(str(path))}"]
    cfg = make_cfg(*file_args)
    assert burn_in_length(cfg) == pytest.approx(1.0) and cfg.burn_steps == 6          # a quarter of T = 4 (dt = 0.16)
    assert make_cfg(*file_args, "model.burn_in_frac=0.4").burn_steps == 10
    assert make_cfg(*file_args, "model.burn_in=0").burn_steps == 0                    # switched off: a fixed, estimated Z_0
    assert make_cfg(*file_args, "model.burn_in=0.48").burn_steps == 3                 # a length of one's own
    assert make_cfg(*file_args, "model.state_space=intensity").burn_steps == 0        # it needs the latent state: auto means off
    with pytest.raises(ValueError):
        make_cfg(*file_args, "model.state_space=intensity", "model.burn_in=0.96")     # asked for explicitly: an error
    # the start value is taken from the observations: the event rate over the first eighth of the window
    with pytest.raises(ValueError, match="resolve_x_start"):
        model_x_start(cfg)
    ds, _ = load_or_generate_dataset(cfg)
    rate = estimate_initial_intensity(ds.train.events, 4.0, 0.125)
    assert rate > 0 and resolve_x_start(cfg, ds.train) == pytest.approx(math.log(rate))
    assert cfg.model.x_start == pytest.approx(math.log(rate)) and build_model(cfg).initial_state == pytest.approx(math.log(rate))
    assert resolve_x_start(make_cfg(), smoke_dataset.train) is None                   # no burn-in: nothing to resolve
    assert resolve_x_start(make_cfg(*file_args, "model.x_start=0.5"), ds.train) == 0.5   # a number is kept
    for bad in ("model.x_start=low", "model.burn_in=long", "model.burn_in_frac=0.0"):
        with pytest.raises(ValueError):
            make_cfg(*file_args, bad)


def test_mcmc_under_the_learned_prior(make_cfg, smoke_dataset) -> None:
    from ndim.metrics import learned_prior_reference_samples

    events = smoke_dataset.test.events[:2]
    mcmc = MCMCConfig(kind="pcn", n_chains=6, n_steps=20)
    cfg = make_cfg("model.burn_in=0.96", "model.x_start=2.0", "model.drift.time_input=true")
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    paths, meta = learned_prior_reference_samples(model, events, cfg, 25, 6, CPU, torch.Generator().manual_seed(0), mcmc_cfg=mcmc)
    assert paths.shape == (2 * 6, 26) and torch.isfinite(paths).all() and float(paths.min()) > 0     # intensities on the window only
    assert meta["kind"] == "mcmc_learned_prior" and 0.0 <= meta["acceptance"] <= 1.0
    assert float(paths[:, 0].std()) > 0                                              # Z_0 is sampled: the burn-in spreads it
    plain = make_cfg()                                                               # no burn-in: every path starts at the model's Z_0
    torch.manual_seed(0)
    fixed, _ = learned_prior_reference_samples(build_model(plain).eval(), events, plain, 25, 6, CPU, torch.Generator().manual_seed(0), mcmc_cfg=mcmc)
    assert fixed.shape == (12, 26) and torch.allclose(fixed[:, 0], torch.full((12,), 5.0), atol=1e-4)

