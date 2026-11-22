"""Tests for ``ndim.config``: ``extends`` chains, hardware profiles, ``key=value`` overrides, validation."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from ndim.config import (
    CONFIG_DIR,
    DiffusionConfig,
    DriftConfig,
    ExperimentConfig,
    PosteriorConfig,
    config_from_dict,
    load_config,
    model_z0,
    parse_overrides,
    resolved_diffusion_kind,
    save_config,
    state_input_scale,
    validate_config,
)

#: the two synthetic examples: example 1 (intensity parametrization) and example 2 (state space, with a burn-in)
INTENSITY = ["intensity/cir_mlp"]
STATE_SPACE = ["state_space/cir_burn_in"]


# --------------------------------------------------------------------------------------
# extends
# --------------------------------------------------------------------------------------
def test_smoke_extends_base(smoke_cfg: ExperimentConfig) -> None:
    # values set in smoke.yaml and in tiny.yaml, the second file it extends
    assert smoke_cfg.name == "smoke"
    assert smoke_cfg.data.n_steps == 25
    assert smoke_cfg.data.n_train == 16 and smoke_cfg.data.n_test == 8
    assert smoke_cfg.model.drift.kind == "mlp" and smoke_cfg.model.drift.hidden == [8]
    assert smoke_cfg.model.state_space == "real" and smoke_cfg.model.link == "exponential"   # the default model
    assert smoke_cfg.model.posterior.rho_hidden == [16]
    assert smoke_cfg.train.epochs == 2
    # values inherited from base.yaml (paper Sec. 4.1 setting)
    assert smoke_cfg.data.T == pytest.approx(4.0)
    assert smoke_cfg.data.truth.kappa == pytest.approx(0.3)
    assert smoke_cfg.data.truth.theta == pytest.approx(80.0)
    assert smoke_cfg.data.truth.z0 == pytest.approx(5.0)
    assert smoke_cfg.model.posterior.kind == "deepset"
    assert smoke_cfg.model.posterior.jump_term == "none"           # DESIGN.md Sec. 3.3 default
    assert smoke_cfg.train.horizons == [1.0, 1.0, 0.75, 0.5, 0.25]
    # the nested dict of tiny.yaml is *merged* into base (not replaced): kind stays "pcn"
    assert smoke_cfg.eval.mcmc.kind == "pcn"
    assert smoke_cfg.eval.mcmc.n_chains == 4 and smoke_cfg.eval.mcmc.n_steps == 20
    assert smoke_cfg.eval.posterior_reference == "analytic"        # stated explicitly by base.yaml for the CIR truth


@pytest.mark.parametrize("value", ["analytic", "mcmc", "none"])
def test_posterior_reference_config_choices(value):
    cfg = config_from_dict({"eval": {"posterior_reference": value}})
    validate_config(cfg)
    assert cfg.eval.posterior_reference == value


def test_posterior_reference_defaults_to_mcmc_without_a_config() -> None:
    assert ExperimentConfig().eval.posterior_reference == "mcmc"


def test_mcmc_kernel_defaults_to_pcn() -> None:
    assert ExperimentConfig().eval.mcmc.kind == "pcn"                # without a config
    assert load_config("base").eval.mcmc.kind == "pcn"               # and in the shipped base config
    independence = config_from_dict({"eval": {"mcmc": {"kind": "independence"}}})    # the paper's sampler stays selectable
    validate_config(independence)
    assert independence.eval.mcmc.kind == "independence"


@pytest.mark.parametrize("patch", [{"posterior_reference": "unknown"}, {"posterior_reference": "auto"}])
def test_invalid_posterior_reference_options_are_rejected(patch):
    with pytest.raises(ValueError):
        config_from_dict({"eval": patch})


def test_load_by_name_suffix_and_absolute_path() -> None:
    a = load_config("smoke")
    b = load_config("smoke.yaml")
    c = load_config(CONFIG_DIR / "smoke.yaml")
    assert a.to_dict() == b.to_dict() == c.to_dict()


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("this_config_does_not_exist")


@pytest.mark.parametrize("name", INTENSITY + STATE_SPACE)
def test_experiment_configs_load(name: str) -> None:
    cfg = load_config(name)
    assert cfg.name == name.replace("/", "_")                  # the run directory: outputs/<parametrization>_<experiment>
    validate_config(cfg)
    # both examples: the grid and sizes of the paper's data and a known CIR diffusion
    assert (cfg.data.n_train, cfg.data.n_test, cfg.data.n_steps, cfg.data.T) == (256, 128, 100, 4.0)
    assert cfg.model.diffusion.kind == "cir" and not cfg.model.diffusion.trainable
    assert cfg.eval.posterior_reference == ("analytic" if cfg.data.truth.z0_std == 0 else "mcmc")


@pytest.mark.parametrize("name", INTENSITY)
def test_intensity_parametrization_is_the_model_of_the_paper(name: str) -> None:
    cfg = load_config(name)
    assert cfg.model.state_space == "intensity"                # the SDE state is the intensity Z
    assert resolved_diffusion_kind(cfg.model) == "cir_sqrt"    # sqrt(Z)
    assert cfg.eval.posterior_reference == "analytic"          # the exact CIR posterior: the initial intensity is known
    assert (cfg.train.epochs, cfg.train.batch_size, cfg.train.m_paths) == (100, 16, 8)
    assert cfg.train.horizons == [1.0, 1.0, 0.75, 0.5, 0.25]   # every mini-batch is trained at each of them
    assert cfg.model.drift.activation == cfg.model.posterior.activation == "silu"         # as in every example


def test_silu_is_the_activation_of_every_example(base_overrides) -> None:
    """SiLU is the default and what every shipped example uses; ReLU stays selectable and reaches the networks."""
    import torch.nn as nn

    from ndim.model import build_model

    assert DriftConfig().activation == PosteriorConfig().activation == DiffusionConfig().activation == "silu"
    for name in ["base", *INTENSITY, *STATE_SPACE]:
        cfg = load_config(name)
        assert cfg.model.drift.activation == cfg.model.posterior.activation == "silu"
    relu = ["model.drift.activation=relu", "model.posterior.activation=relu"]
    for name, extra, kind in ((INTENSITY[0], [], nn.SiLU), (STATE_SPACE[0], [], nn.SiLU), (INTENSITY[0], relu, nn.ReLU)):
        model = build_model(load_config(name, profile="tiny", overrides=[*base_overrides, *extra]))
        networks = [model.score_net.phi, model.score_net.rho, model.drift.net]
        for network in networks:
            activations = [m for m in network.net if not isinstance(m, nn.Linear)]
            assert activations and all(type(m) is kind for m in activations), (name, network)


def test_example_1_has_an_mlp_drift_and_a_known_initial_intensity() -> None:
    cfg = load_config(INTENSITY[0])
    assert cfg.model.drift.kind == "mlp" and cfg.model.drift.time_input            # b_theta(z, t)
    assert cfg.data.truth.z0 == 5.0 and cfg.data.truth.z0_std == 0.0               # every path of the truth starts at 5
    assert cfg.burn_steps == 0 and model_z0(cfg) == 5.0                            # and so does every path of the model


def test_example_2_is_the_state_space_parametrization_with_a_burn_in() -> None:
    cfg = load_config(STATE_SPACE[0])
    assert cfg.model.state_space == "real" and cfg.model.link == "exponential"     # X = log Z
    assert resolved_diffusion_kind(cfg.model) == "cir_exponential"                 # sqrt(Z) written for X: exp(-X/2)
    assert not cfg.model.diffusion.trainable
    assert cfg.model.drift.kind == "mlp" and cfg.model.drift.time_input            # one network on the burn-in and the window
    assert (cfg.model.burn_in, cfg.model.x_start, cfg.burn_steps) == (1.0, 0.0, 25)    # from X = 0 (intensity 1) at t = -1
    truth = cfg.data.truth
    assert (truth.z0, truth.z0_std, truth.z0_low, truth.z0_high) == (40.0, 10.0, 5.0, 75.0)    # the unknown initial intensity
    assert (cfg.eval.posterior_reference, cfg.eval.mcmc.kind) == ("mcmc", "pcn")   # no exact posterior for a random Z_0
    assert cfg.train.epochs == 150
    one, two = load_config(INTENSITY[0]).to_dict()["data"], cfg.to_dict()["data"]
    assert {k: v for k, v in one.items() if k != "truth"} == {k: v for k, v in two.items() if k != "truth"}    # same grid and sizes


def test_parametric_drift_is_cir_on_the_intensity(make_cfg) -> None:
    assert DriftConfig().theta_init == 40.0                                        # an intensity level
    cfg = make_cfg("model.state_space=intensity", "model.drift.kind=cir")
    assert cfg.model.drift.kind == "cir"
    with pytest.raises(ValueError, match="state_space: intensity"):                # not a drift of a latent state
        make_cfg("model.drift.kind=cir")
    with pytest.raises(ValueError, match="must be mlp"):                           # the latent linear drift is gone
        make_cfg("model.drift.kind=ou")


def test_parametric_drift_uses_the_learning_rate_of_the_networks() -> None:
    """``theta`` is trained in normalised units, so the parametric drift needs no learning rate of its own."""
    parametric, mlp = load_config("intensity/cir_mlp", overrides=["model.drift.kind=cir"]), load_config("intensity/cir_mlp")
    assert parametric.optim.lr_prior == mlp.optim.lr_prior == 5e-3
    assert parametric.model.drift.z_scale == 80.0


def test_network_input_scales_must_be_positive(make_cfg) -> None:
    assert PosteriorConfig().gap_scale is None                    # default: the typical gap between events, T / count_scale
    assert make_cfg("model.posterior.gap_scale=0.5").model.posterior.gap_scale == 0.5
    for override in ("model.posterior.gap_scale=0", "model.posterior.count_scale=0", "model.posterior.z_scale=-1",
                     "model.drift.z_scale=0", "model.diffusion.z_scale=0", "model.intensity_scale=0"):
        with pytest.raises(ValueError, match="positive"):
            make_cfg(override)


def test_state_scales_follow_the_coordinates_of_each_network(make_cfg) -> None:
    """One number, ``model.intensity_scale``, sets the scale of every state input (``state_input_scale``)."""
    latent = math.log(80.0)
    assert DriftConfig().z_scale is None and DiffusionConfig().z_scale is None and PosteriorConfig().z_scale is None
    default = make_cfg()                                                   # X = log Z: the latent state at 80
    assert default.model.intensity_scale == 80.0
    for part in (default.model.drift, default.model.diffusion, default.model.posterior):
        assert part.z_scale == pytest.approx(latent)
    intensity = make_cfg("model.state_space=intensity").model              # the state is the intensity
    assert [intensity.drift.z_scale, intensity.diffusion.z_scale, intensity.posterior.z_scale] == [80.0] * 3
    softplus = make_cfg("model.link=softplus").model                       # this latent state is on the intensity scale
    assert softplus.drift.z_scale == pytest.approx(80.0) and softplus.posterior.z_scale == pytest.approx(80.0)
    score = make_cfg("model.posterior.score_space=intensity").model        # the score network works with the intensity
    assert score.posterior.z_scale == 80.0 and score.drift.z_scale == pytest.approx(latent)
    assert make_cfg("model.state_space=intensity", "model.intensity_scale=500").model.posterior.z_scale == 500.0
    explicit = make_cfg("model.state_space=intensity", "model.posterior.z_scale=20").model   # a number in the config wins
    assert explicit.posterior.z_scale == 20.0 and explicit.drift.z_scale == 80.0
    # |log 1.5| < 1: a latent state that is already of order one is left as it is
    assert state_input_scale(make_cfg("model.intensity_scale=1.5").model) == 1.0


@pytest.mark.parametrize("name", INTENSITY + STATE_SPACE)
def test_shipped_experiments_leave_the_state_scale_to_the_base_code(name: str) -> None:
    cfg = load_config(name)
    expected = 80.0 if name in INTENSITY else math.log(80.0)
    assert cfg.model.intensity_scale == 80.0
    for part in (cfg.model.drift, cfg.model.diffusion, cfg.model.posterior):
        assert part.z_scale == pytest.approx(expected)


def test_saved_runs_keep_the_scales_they_were_trained_with(smoke_cfg: ExperimentConfig) -> None:
    saved = smoke_cfg.to_dict()
    assert saved["model"]["posterior"]["z_scale"] == pytest.approx(math.log(80.0))   # a run records the resolved number
    saved["model"]["intensity_scale"] = 12.0                   # the recorded scales are kept, not derived again
    cfg = config_from_dict(saved)
    assert cfg.model.drift.z_scale == cfg.model.diffusion.z_scale == cfg.model.posterior.z_scale == pytest.approx(math.log(80.0))
    assert config_from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()                # a saved config is read back as it is


# --------------------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------------------
def test_profile_layer_applies_on_top(make_cfg) -> None:
    cfg = make_cfg(profile="laptop")
    assert cfg.name == "smoke"                       # profile does not rename the run
    assert cfg.train.eval_paths == 16                # laptop.yaml overrides smoke.yaml
    assert (cfg.train.batch_size, cfg.train.m_paths) == (8, 2)   # ... but not the training settings of the config
    assert cfg.train.device == "cpu"
    assert cfg.train.num_threads == 8
    assert cfg.model.posterior.rho_hidden == [64, 64]
    assert cfg.model.posterior.phi_out == 16
    assert cfg.eval.mcmc.n_chains == 32 and cfg.eval.mcmc.n_steps == 3000
    assert cfg.eval.mcmc.kind == "pcn"               # untouched keys survive the merge
    assert cfg.data.n_steps == 25                    # keys not in the profile are untouched


@pytest.mark.parametrize("profile", ["laptop", "gpu"])
@pytest.mark.parametrize("name", INTENSITY + STATE_SPACE)
def test_hardware_profiles_keep_the_training_settings_of_the_experiment(name: str, profile: str) -> None:
    """A profile must not change how an experiment trains: batch size, paths per observation, epochs, optimiser."""
    plain, with_profile = load_config(name), load_config(name, profile=profile)
    for key in ("epochs", "batch_size", "m_paths", "horizons", "kl_weight", "grad_mode"):
        assert getattr(with_profile.train, key) == getattr(plain.train, key), key
    assert with_profile.to_dict()["optim"] == plain.to_dict()["optim"]
    assert with_profile.to_dict()["data"] == plain.to_dict()["data"]
    assert with_profile.to_dict()["model"]["drift"] == plain.to_dict()["model"]["drift"]
    assert (with_profile.train.batch_size, with_profile.train.m_paths) == (16, 8)


def test_gpu_profile_loads_without_device_check() -> None:
    cfg = load_config("smoke", profile="gpu")
    assert cfg.train.device == "cuda"                # only resolved later by resolve_device
    assert cfg.model.posterior.checkpoint_phi is False
    assert cfg.eval.mcmc.n_chains == 128


# --------------------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------------------
def test_parse_overrides_types() -> None:
    d = parse_overrides(["train.epochs=3", "model.drift.hidden=[8, 8]", "model.diffusion.trainable=true",
                         "optim.lr_prior=1e-3", "name=abc", "eval.mcmc.kind=independence"])
    assert d["train"]["epochs"] == 3 and isinstance(d["train"]["epochs"], int)
    assert d["model"]["drift"]["hidden"] == [8, 8]
    assert d["model"]["diffusion"]["trainable"] is True
    assert d["optim"]["lr_prior"] == pytest.approx(1e-3)
    assert d["name"] == "abc"
    assert d["eval"]["mcmc"]["kind"] == "independence"


def test_parse_overrides_requires_equals() -> None:
    with pytest.raises(ValueError):
        parse_overrides(["train.epochs"])


def test_overrides_applied_last(make_cfg) -> None:
    cfg = make_cfg("train.epochs=7", "model.drift.kind=mlp", "model.drift.hidden=[8,8]",
                   "model.diffusion.trainable=true", "train.eval_paths=3", profile="laptop")
    assert cfg.train.epochs == 7
    assert cfg.model.drift.kind == "mlp" and cfg.model.drift.hidden == [8, 8]
    assert cfg.model.diffusion.trainable is True
    assert cfg.train.eval_paths == 3                 # override beats the laptop profile (16)


def test_unknown_key_raises(make_cfg) -> None:
    with pytest.raises(KeyError):
        make_cfg("train.no_such_key=1")
    with pytest.raises(KeyError):
        make_cfg("nosuchsection.x=1")


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "override",
    [
        "model.drift.kind=foo",
        "model.posterior.kind=''",
        "model.state_space=foo",
        "model.link=square",
        "model.diffusion.kind=cir_sqrt",
        "data.truth.z0=0",
        "model.posterior.jump_term=smooth",
        "model.positivity=clamp",
        "train.grad_mode=adjoint",
        "train.horizons=[]",
        "train.horizons=[1.5]",
        "train.horizons=[0.0]",
        "train.horizons=[0.75, 0.5]",         # the full horizon must be trained
        "eval.mcmc.kind=hmc",
        "data.n_steps=0",
        "data.T=-1.0",
    ],
)
def test_validation_errors(make_cfg, override: str) -> None:
    with pytest.raises(ValueError):
        make_cfg(override)


def test_likelihood_grid_convention_is_not_configurable(make_cfg) -> None:
    with pytest.raises(KeyError):
        make_cfg("likelihood.interp=linear")
    with pytest.raises(KeyError):
        make_cfg("likelihood.quadrature=trapezoid")


def test_the_3d_drift_figure_spans_the_range_of_the_paper() -> None:
    assert load_config("base").eval.drift_3d_z_max == 100.0 and load_config(INTENSITY[0]).eval.drift_3d_z_max == 100.0
    assert load_config("base").eval.drift_z_max == 160.0        # the slices and the heat-map keep their wider range


def test_defaults_when_no_structure_is_assumed() -> None:
    diffusion = DiffusionConfig()
    assert diffusion.kind == "mlp" and diffusion.trainable            # a learned diffusion unless a config says otherwise
    posterior = PosteriorConfig()
    assert posterior.phi_uses_state                                   # the paper's Deep Sets form
    assert posterior.count_hidden == [32, 16]


@pytest.mark.parametrize("name", ["base", *INTENSITY, *STATE_SPACE])
def test_cir_experiments_use_the_known_cir_diffusion(name: str) -> None:
    cfg = load_config(name)
    assert cfg.model.diffusion.kind == "cir" and not cfg.model.diffusion.trainable
    assert cfg.model.diffusion.eta_init == cfg.data.truth.eta == 1.0


@pytest.mark.parametrize("override", ["train.no_such_setting=3", "eval.no_such_setting=true", "likelihood.quadrature=trapezoid"])
def test_settings_that_do_not_exist_are_rejected(make_cfg, smoke_cfg: ExperimentConfig, override: str) -> None:
    """A config is read strictly, from a file, from ``--set`` and from the dict saved with a run."""
    with pytest.raises(KeyError, match="Unknown config key"):
        make_cfg(override)
    section, key = override.split("=")[0].split(".")
    saved = smoke_cfg.to_dict()
    saved[section][key] = 1
    with pytest.raises(KeyError, match="Unknown config key"):
        config_from_dict(saved)


# --------------------------------------------------------------------------------------
# derived helpers and round trips
# --------------------------------------------------------------------------------------
def test_derived_helpers(smoke_cfg: ExperimentConfig, tmp_root: Path) -> None:
    assert smoke_cfg.dt == pytest.approx(4.0 / 25)
    assert smoke_cfg.horizon_steps(1.0) == 25
    assert smoke_cfg.horizon_steps(0.25) == round(0.25 * 25)
    assert smoke_cfg.horizon_steps(0.5) == round(0.5 * 25)
    assert smoke_cfg.horizon_steps(1e-9) == 1                     # never fewer than one interval
    assert smoke_cfg.horizon_steps(5.0) == 25                     # never beyond the grid
    assert smoke_cfg.output_dir == Path(smoke_cfg.output_root) / "smoke"
    assert str(smoke_cfg.output_dir).startswith(str(tmp_root))    # redirected by the conftest


def test_dict_round_trip(smoke_cfg: ExperimentConfig) -> None:
    d = smoke_cfg.to_dict()
    cfg2 = config_from_dict(d)
    assert cfg2.to_dict() == d
    assert cfg2.data.truth.theta == pytest.approx(smoke_cfg.data.truth.theta)


def test_save_and_reload(smoke_cfg: ExperimentConfig, tmp_path: Path) -> None:
    p = tmp_path / "sub" / "config.yaml"
    save_config(smoke_cfg, p)
    assert p.exists()
    cfg2 = load_config(p)
    assert cfg2.to_dict() == smoke_cfg.to_dict()


def test_config_from_dict_validates() -> None:
    d = load_config("smoke").to_dict()
    d["model"]["drift"]["kind"] = "nope"
    with pytest.raises(ValueError):
        config_from_dict(d)
