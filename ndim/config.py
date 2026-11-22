"""Configuration dataclasses, YAML loading, hardware profiles and ``key=value`` overrides.

A run is described by a YAML file (see ``configs/``). Files can ``extends`` other files
(relative to the ``configs`` directory or absolute); later files override earlier ones.
Hardware profiles (``configs/laptop.yaml``, ``configs/gpu.yaml``) are just another layer applied
on top, and ``--set a.b.c=value`` overrides are applied last (values parsed as YAML).
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------
@dataclass
class TruthConfig:
    """Ground-truth data-generating CIR process ``dZ = kappa (theta - Z) dt + eta sqrt(Z) dB``."""
    kappa: float = 0.3
    theta: float = 80.0
    eta: float = 1.0
    z0: float = 5.0          # initial intensity; with ``z0_std > 0`` the mean of its distribution
    # z0_std > 0: Z_0 is normal(z0, z0_std) truncated to [z0_low, z0_high]; 0: every path starts at z0
    z0_std: float = 0.0
    z0_low: float = 0.0
    z0_high: Optional[float] = None
    # z0_modes: a mixture of normals with these means (weights z0_weights, equal by default) in place of the one normal
    z0_modes: Optional[List[float]] = None
    z0_weights: Optional[List[float]] = None


@dataclass
class DataConfig:
    T: float = 4.0
    n_steps: int = 100            # Euler intervals on [0, T]; grid has n_steps + 1 points
    n_train: int = 256
    n_test: int = 128
    seed: int = 0
    sim_substeps: int = 1         # >1 simulates the truth on a finer grid when generating data
    cache_dir: str = "data"
    truth: TruthConfig = field(default_factory=TruthConfig)
    # synthetic: generated from ``truth`` and cached | file: observations read from ``path``, without a truth
    source: str = "synthetic"
    path: Optional[str] = None
    # file only: share of the observations held out for testing when the file does not split them
    test_fraction: float = 0.25


@dataclass
class DriftConfig:
    # mlp | cir (the parametric drift kappa (theta - Z) of the intensity; needs model.state_space: intensity)
    kind: str = "mlp"
    hidden: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "silu"
    time_input: bool = True
    # scale of the state input (and of the cir ``theta``); None -> derived from ``model.intensity_scale``
    z_scale: Optional[float] = None
    drift_scale: float = 1.0
    init_output_std: float = 0.01
    # cir only: initial values (theta is an intensity level) and which of the two parameters are trained
    kappa_init: float = 0.5
    theta_init: float = 40.0
    learn_theta: bool = True
    learn_kappa: bool = True


@dataclass
class DiffusionConfig:
    # mlp | constant | cir. cir is eta * sqrt(Z) of the intensity, in the coordinates of ``model.state_space``
    kind: str = "mlp"
    # initial scale (mlp, constant) or the CIR eta; ``auto``: estimated from the training observations (``ndim.cir_eta``)
    eta_init: Union[float, str] = 0.2
    trainable: bool = True
    hidden: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "silu"
    time_input: bool = True
    z_scale: Optional[float] = None   # mlp only: scale of the state input; None -> ``state_input_scale``
    min_scale: float = 1e-4
    # cir for a latent state: the diffusion of X is capped at max_step_std / sqrt(dt), which changes it only below the
    # intensity eta^2 dt / max_step_std^2. None: no cap
    max_step_std: Optional[float] = 1.0


@dataclass
class PosteriorConfig:
    kind: str = "deepset"         # registered architecture name
    score_space: str = "state"   # built-in feature/output coordinates: state | intensity
    factory: Optional[str] = None # optional import path "package.module:factory"
    options: Dict[str, Any] = field(default_factory=dict)
    rho_hidden: List[int] = field(default_factory=lambda: [64, 64])    # deepset: rho
    phi_hidden: List[int] = field(default_factory=lambda: [32, 32])    # deepset: phi
    phi_out: int = 16
    count_hidden: List[int] = field(default_factory=lambda: [32, 16])  # remaining_count: its MLP, 4 -> 32 -> 16 -> 1
    activation: str = "silu"
    z_scale: Optional[float] = None       # scale of the state input; None -> ``state_input_scale`` (follows score_space)
    count_scale: float = 100.0            # typical number of events of an observation: scale of the remaining count
    score_scale: float = 1.0
    jump_term: str = "none"  # "none" | "raw" | "baseline_removed"
    jump_z_floor: float = 1.0
    phi_uses_state: bool = True           # deepset: phi sees the state (paper Sec. 3.3.1); false precomputes the pooling: faster, less accurate
    future_gate_width: float = 0.0        # 0 -> hard mask
    gap_feature: bool = True              # include Delta_i (gap to previous future event / to t)
    gap_scale: Optional[float] = None     # time unit of the gap feature; None -> T / count_scale, the typical gap between events
    checkpoint_phi: bool = True           # only when phi_uses_state is false: recompute the precomputed pooling in backward to save memory
    init_output_std: float = 0.01


@dataclass
class ModelConfig:
    # real: the state is X with Z = link(X) (default X = log Z) | intensity: the state is Z itself, as in the paper
    state_space: str = "real"
    link: str = "exponential"    # real states only: exponential | softplus; fixed, not trained
    # typical intensity of the data (events per unit time): the networks scale their state input with it
    intensity_scale: float = 80.0
    drift: DriftConfig = field(default_factory=DriftConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    positivity: str = "full_truncation"   # intensity states only: full_truncation | reflect | softplus
    softplus_scale: float = 0.5           # only for positivity == "softplus"
    # initial intensity of a model without burn-in: auto (the truth's for synthetic data, else estimate) | truth |
    # estimate (mean event rate over the first ``z0_window`` of [0, T]) | a number
    z0: Union[str, float] = "auto"
    z0_window: float = 0.125
    # burn-in: the model starts at t = -burn_in from the latent state ``x_start``, nothing is observed before 0, and the
    # distribution of Z_0 is learned. auto (``burn_in_frac * T`` for observations from a file, off for synthetic data) |
    # a length in time units (0: no burn-in)
    burn_in: Union[str, float] = "auto"
    burn_in_frac: float = 0.25
    # latent state at the start of the burn-in: auto (from the event rate at the start of the window) | a number
    x_start: Union[str, float] = "auto"


def resolved_diffusion_kind(model: ModelConfig) -> str:
    """Coordinate form of ``model.diffusion.kind``.

    ``cir`` always means the intensity diffusion ``eta * sqrt(Z)``, so the same physical diffusion is used
    whichever parametrization ``model.state_space`` selects:

    * ``cir_sqrt`` when the state is the intensity itself;
    * ``cir_exponential`` (``sigma_X = eta * exp(-X/2)``) when the state is ``X = log Z``;
    * ``cir_link`` (``sigma_X = eta * sqrt(g(X)) / g'(X)``) for a latent state with any other link ``g``.
    """
    kind = model.diffusion.kind
    if kind != "cir":
        return kind
    if model.state_space == "intensity":
        return "cir_sqrt"
    return "cir_exponential" if model.link in {"exponential", "exp"} else "cir_link"


# --------------------------------------------------------------------------------------
# Normalisation of the network inputs
# --------------------------------------------------------------------------------------
def _latent_state(link: str, intensity: float) -> float:
    """The latent state ``g^{-1}(intensity)`` of a real-state model."""
    from .links import build_link  # imported here: the configuration itself does not need torch

    return build_link(link).inverse_scalar(float(intensity))


def state_input_scale(model: ModelConfig, coordinates: str = "state") -> float:
    """Fixed scale by which a network divides its state input, so that the input is of order one.

    ``model.intensity_scale`` is the typical intensity of the data. An input on the intensity scale -- the state
    of ``state_space: intensity``, or the intensity that a ``score_space: intensity`` network works with
    (``coordinates="intensity"``) -- is divided by it. A latent state is divided by its own value at that
    intensity, ``|g^{-1}(intensity_scale)|`` (never less than 1): about 4.4 for ``X = log Z`` at 80 events per
    unit time, and the intensity scale itself for the softplus link, whose latent state is on the intensity scale.
    """
    scale = float(model.intensity_scale)
    if scale <= 0:
        raise ValueError("model.intensity_scale must be positive: it sets the scale of the state inputs of the networks")
    if model.state_space == "intensity" or coordinates == "intensity":
        return scale
    return max(abs(_latent_state(model.link, scale)), 1.0)


def resolve_input_scales(model: ModelConfig) -> None:
    """Fill every ``z_scale`` left as ``None`` with ``state_input_scale`` (in place; explicit values are kept).

    Called when a config is loaded and when a model is built, so the networks always receive a number and a
    saved run records the scales it was trained with.
    """
    if model.drift.z_scale is None:
        model.drift.z_scale = state_input_scale(model)
    if model.diffusion.z_scale is None:
        model.diffusion.z_scale = state_input_scale(model)
    if model.posterior.z_scale is None:
        model.posterior.z_scale = state_input_scale(model, model.posterior.score_space)


def burn_in_length(cfg: "ExperimentConfig") -> float:
    """Length of the burn-in in time units: ``model.burn_in``, with ``auto`` resolved (see ``ModelConfig``)."""
    b = cfg.model.burn_in
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return float(b)
    if b != "auto":
        raise ValueError("model.burn_in must be auto or a non-negative number")
    if cfg.data.source != "synthetic" and cfg.model.state_space == "real" and cfg.train.grad_mode == "backprop":
        return float(cfg.model.burn_in_frac) * float(cfg.data.T)
    return 0.0


def model_x_start(cfg: "ExperimentConfig") -> float:
    """Latent state at the start of the burn-in (``model.x_start``). ``auto`` has to be resolved against the training
    observations first (``ndim.data.resolve_x_start``, which ``ndim.train.train`` calls)."""
    x = cfg.model.x_start
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    raise ValueError("model.x_start is to be taken from the observations: call ndim.data.resolve_x_start(cfg, train_events) "
                     "first (ndim.train.train does), or set model.x_start to a number")


def z0_mode(cfg: "ExperimentConfig") -> str:
    """How the initial intensity of the model is obtained: ``value`` (a number is set), ``truth`` or ``estimate``."""
    z0 = cfg.model.z0
    if isinstance(z0, (int, float)) and not isinstance(z0, bool):
        return "value"
    if z0 == "auto":
        return "truth" if cfg.data.source == "synthetic" else "estimate"
    return str(z0)


def model_z0(cfg: "ExperimentConfig") -> float:
    """Initial intensity ``Z_0`` of the model (``model.z0``); the truth has its own, ``data.truth.z0``.

    ``estimate`` has to be resolved against the training observations first (``ndim.data.resolve_z0``, which
    ``ndim.train.train`` calls): a model cannot be simulated before its starting value is known.
    """
    mode = z0_mode(cfg)
    if mode == "value":
        return float(cfg.model.z0)
    if mode == "truth":
        return float(cfg.data.truth.z0)
    raise ValueError("model.z0 is to be estimated from the observations: call ndim.data.resolve_z0(cfg, train_events) first "
                     "(ndim.train.train does), or set model.z0 to a number")


def required_scale(value: Optional[float], name: str) -> float:
    """``value`` as a float; a clear error when a scale was never resolved (a sub-config built by hand)."""
    if value is None:
        raise ValueError(f"{name} is not set. Load the config with load_config / config_from_dict or build the model "
                         "with NDIM (they derive it from model.intensity_scale), or pass a number.")
    return float(value)


@dataclass
class LikelihoodConfig:
    """Poisson likelihood on the Euler grid, left convention: the intensity on ``[t_k, t_{k+1})`` is ``Z_k``."""
    eps: float = 1e-6                     # floor of the intensity inside the log


@dataclass
class OptimConfig:
    lr_prior: float = 5e-3
    lr_posterior: float = 5e-3
    lr_diffusion: float = 1e-2
    weight_decay: float = 0.0
    clip_prior: Optional[float] = 10.0
    clip_posterior: Optional[float] = 10.0
    clip_diffusion: Optional[float] = 10.0
    lr_schedule: str = "cosine"           # "cosine" | "exponential" | "plateau" | "none"; stepped once per epoch
    lr_min_factor: float = 0.1            # cosine / exponential: the rate at the last epoch, as a fraction of the initial rate
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    m_paths: int = 8
    # fractions of T; every mini-batch is trained at each of them in turn. The full horizon 1.0 must be in the list
    horizons: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.75, 0.5, 0.25])
    grad_mode: str = "backprop"           # "backprop" | "sensitivity" (same pathwise ELBO gradient, paper Eq. 12-13)
    kl_weight: float = 0.5                # 0.5 gives the ELBO
    kl_warmup_epochs: int = 0
    eval_every: int = 5
    eval_paths: int = 32
    eval_horizon_fracs: List[float] = field(default_factory=lambda: [1.0])
    plot_every: int = 5
    checkpoint_every: int = 5
    early_stopping_patience: int = 0      # 0 disables
    log_every_batches: int = 0            # 0 -> once per epoch
    seed: int = 1
    device: str = "cpu"
    num_threads: int = 0                  # 0 -> torch default
    detect_anomaly: bool = False
    skip_nonfinite: bool = True
    # off | auto: when the untrained model has no finite loss, start at a smaller diffusion coefficient and raise it
    warm_start: str = "off"
    max_abs_score: Optional[float] = None
    # bound on the posterior control u = sigma * s, in training and in evaluation (the KL term uses the bounded control)
    max_abs_control: Optional[float] = None


@dataclass
class MCMCConfig:
    kind: str = "pcn"                     # "pcn" (preconditioned Crank-Nicolson) | "independence" (prior-path proposals)
    n_chains: int = 64
    n_steps: int = 5000
    burn_in_frac: float = 0.5             # pcn: adaptation window; independence: only splits the acceptance diagnostics
    beta_init: float = 0.2                # pcn only: initial step size
    target_accept: float = 0.25           # pcn only: acceptance rate targeted by the step-size adaptation
    adapt: bool = True                    # pcn only: adapt the step size during the burn-in
    seed: int = 123


@dataclass
class EvalConfig:
    n_generated: int = 128
    n_initial: int = 4000  # draws of the learned prior for the figures of its distribution
    n_particles: int = 2000  # particles of the filter for the marginal log-likelihood (observations without a truth)
    n_paths_posterior: int = 64
    partial_horizon_frac: float = 0.5
    wasserstein_n_obs: int = 16
    wasserstein_n_paths: int = 64
    l2_n_paths: int = 256
    drift_z_max: float = 160.0
    drift_3d_html: bool = True
    drift_3d_z_max: float = 100.0   # z range of the 3-D surface figure; the paper's Fig. 2 spans 0 to 100
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    # reference posterior of the comparison figures: analytic (exact CIR posterior, known Z_0) | mcmc | none
    posterior_reference: str = "mcmc"
    run_wasserstein: bool = True


@dataclass
class ExperimentConfig:
    name: str = "exp"
    output_root: str = "outputs"
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # derived helpers -----------------------------------------------------------------
    @property
    def dt(self) -> float:
        return float(self.data.T) / int(self.data.n_steps)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.name

    @property
    def burn_steps(self) -> int:
        """Number of Euler intervals of the burn-in ``[-burn_in, 0)``; 0 without a burn-in."""
        return int(round(burn_in_length(self) / self.dt))

    def horizon_steps(self, frac: float) -> int:
        """Number of Euler intervals covered by horizon ``frac * T`` (rounded to the grid)."""
        n = int(round(float(frac) * self.data.n_steps))
        return max(1, min(n, int(self.data.n_steps)))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def typical_input_sizes(cfg: ExperimentConfig, mean_count: float) -> Dict[str, float]:
    """Typical size of each scaled network input for observations with ``mean_count`` events on ``[0, T]``.

    Only the observations are used: their mean event rate ``mean_count / T`` stands for the intensity and
    ``T / mean_count`` for the gap between two events. A value far from one means that a scale does not fit the
    data (``off_scale_inputs``).
    """
    model, post, T = cfg.model, cfg.model.posterior, float(cfg.data.T)
    mean_count = max(float(mean_count), 1e-12)
    rate = mean_count / T

    def state_size(scale: Optional[float], coordinates: str = "state") -> float:
        on_intensity_scale = model.state_space == "intensity" or coordinates == "intensity"
        level = rate if on_intensity_scale else abs(_latent_state(model.link, rate))
        return level / required_scale(scale, "z_scale")

    sizes: Dict[str, float] = {}
    if model.drift.kind == "mlp":
        sizes["drift state / z_scale"] = state_size(model.drift.z_scale)
    if model.diffusion.kind == "mlp":
        sizes["diffusion state / z_scale"] = state_size(model.diffusion.z_scale)
    if not post.factory:
        sizes["posterior state / z_scale"] = state_size(post.z_scale, post.score_space)
        sizes["count / count_scale"] = mean_count / float(post.count_scale)
        if post.kind == "deepset" and post.gap_feature:
            gap_scale = float(post.gap_scale) if post.gap_scale is not None else T / float(post.count_scale)
            sizes["gap / gap_scale"] = (T / mean_count) / gap_scale
    return sizes


def off_scale_inputs(cfg: ExperimentConfig, sizes: Dict[str, float], low: float = 0.2, high: float = 5.0) -> List[str]:
    """Names of the inputs in ``sizes`` whose typical size is outside ``[low, high]``.

    A latent state can legitimately sit near zero (``X = log Z`` at one event per unit time), so only ``high``
    applies to it.
    """
    latent = cfg.model.state_space == "real"
    out = []
    for name, size in sizes.items():
        is_latent_state = latent and "state" in name and not (name.startswith("posterior") and cfg.model.posterior.score_space == "intensity")
        if size > high or (size < low and not is_latent_state):
            out.append(name)
    return out


# --------------------------------------------------------------------------------------
# Loading / merging
# --------------------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_path(p: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    candidates = []
    if relative_to is not None:
        candidates.append(relative_to / p)
    candidates.append(CONFIG_DIR / p)
    candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.exists():
            return c
    if p.suffix == "" :
        return _resolve_path(p.with_suffix(".yaml"), relative_to)
    raise FileNotFoundError(f"Config file not found: {p} (tried {', '.join(str(c) for c in candidates)})")


def load_yaml_with_extends(path: Union[str, Path]) -> Dict[str, Any]:
    path = _resolve_path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("extends", None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = _deep_update(merged, load_yaml_with_extends(_resolve_path(b, path.parent)))
    return _deep_update(merged, raw)


def _from_dict(cls: Any, d: Dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return d
    kwargs: Dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}
    for k, v in (d or {}).items():
        if k not in valid:
            raise KeyError(f"Unknown config key '{k}' for {cls.__name__}. Valid keys: {sorted(valid)}")
        ftype = valid[k].type
        target = _dataclass_for_field(cls, k)
        if target is not None and isinstance(v, dict):
            kwargs[k] = _from_dict(target, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _dataclass_for_field(cls: Any, name: str) -> Optional[Any]:
    for f in fields(cls):
        if f.name == name:
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None  # type: ignore[misc]
            if default is not None and is_dataclass(default):
                return type(default)
    return None


_NUMERIC_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _parse_override_value(value: str) -> Any:
    """Parse a ``--set`` value as YAML; coerce numeric-looking strings such as ``1e-3``.

    PyYAML implements YAML 1.1, whose float regex requires a dot in scientific notation, so
    ``yaml.safe_load("1e-3")`` returns the *string* ``"1e-3"``.  A learning rate passed on the
    command line must become a float, so such strings are converted here.
    """
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str) and _NUMERIC_RE.match(parsed.strip()):
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed


def parse_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """``["train.epochs=10", "model.drift.hidden=[32,32]"]`` -> nested dict (values parsed as YAML)."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must look like key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        parsed = _parse_override_value(value)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return out


def load_config(
    path: Union[str, Path],
    profile: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
) -> ExperimentConfig:
    d = load_yaml_with_extends(path)
    if profile:
        d = _deep_update(d, load_yaml_with_extends(profile))
    if overrides:
        d = _deep_update(d, parse_overrides(overrides))
    cfg = _from_dict(ExperimentConfig, d)
    resolve_input_scales(cfg.model)
    validate_config(cfg)
    return cfg


def config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    """The config of a dict, such as the one saved with a run: every key must be one of the current settings."""
    cfg = _from_dict(ExperimentConfig, copy.deepcopy(d))
    resolve_input_scales(cfg.model)
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.data.n_steps <= 0 or cfg.data.T <= 0:
        raise ValueError("data.n_steps and data.T must be positive")
    if cfg.model.drift.kind not in {"mlp", "cir"}:
        raise ValueError("model.drift.kind must be mlp|cir")
    if not cfg.model.posterior.kind:
        raise ValueError("model.posterior.kind must be a non-empty architecture name")
    if cfg.model.posterior.score_space not in {"state", "intensity"}:
        raise ValueError("model.posterior.score_space must be state|intensity")
    if cfg.model.posterior.factory and ":" not in cfg.model.posterior.factory:
        raise ValueError("model.posterior.factory must be package.module:factory")
    if cfg.model.state_space not in {"real", "intensity"}:
        raise ValueError("model.state_space must be real|intensity")
    if cfg.model.link not in {"exponential", "exp", "softplus"}:
        raise ValueError("model.link must be exponential|softplus")
    if cfg.data.source not in {"synthetic", "file"}:
        raise ValueError("data.source must be synthetic|file")
    if cfg.data.source == "file" and not cfg.data.path:
        raise ValueError("data.source: file needs data.path, the file with the observations")
    if not 0.0 < float(cfg.data.test_fraction) < 1.0:
        raise ValueError("data.test_fraction must be strictly between 0 and 1: the share of the observations held out for testing")
    z0 = cfg.model.z0
    if isinstance(z0, bool) or not (isinstance(z0, (int, float)) or z0 in {"auto", "truth", "estimate"}):
        raise ValueError("model.z0 must be auto|truth|estimate or a number")
    if isinstance(z0, (int, float)) and z0 < 0:
        raise ValueError("model.z0 must be non-negative: it is an intensity")
    if z0 == "truth" and cfg.data.source != "synthetic":
        raise ValueError("model.z0: truth needs synthetic data; observations read from a file have no known truth")
    if not 0.0 < cfg.model.z0_window <= 1.0:
        raise ValueError("model.z0_window must be in (0, 1]: the fraction of [0, T] whose event rate estimates Z_0")
    if not 0.0 < cfg.model.burn_in_frac <= 4.0:
        raise ValueError("model.burn_in_frac must be in (0, 4]: the length of an automatic burn-in as a fraction of T")
    if burn_in_length(cfg) < 0:
        raise ValueError("model.burn_in must be non-negative: the length of the interval before the observation window")
    if burn_in_length(cfg) > 0 and cfg.burn_steps < 1:
        raise ValueError(f"model.burn_in = {burn_in_length(cfg):g} is shorter than half an Euler step (dt = {cfg.dt:g})")
    x_start = cfg.model.x_start
    if not (x_start == "auto" or (isinstance(x_start, (int, float)) and not isinstance(x_start, bool))):
        raise ValueError("model.x_start must be auto or a number: the latent state at the start of the burn-in")
    if cfg.burn_steps > 0 and cfg.model.state_space != "real":
        raise ValueError("the burn-in is implemented for the state-space parametrization: it needs model.state_space: real")
    if cfg.burn_steps > 0 and cfg.train.grad_mode != "backprop":
        raise ValueError("the burn-in is trained by backpropagation: train.grad_mode must be backprop")
    truth = cfg.data.truth
    if truth.z0_std < 0:
        raise ValueError("data.truth.z0_std must be non-negative")
    if truth.z0_std > 0:
        high = float("inf") if truth.z0_high is None else float(truth.z0_high)
        if not 0.0 <= float(truth.z0_low) < high:
            raise ValueError("data.truth.z0_low and z0_high must satisfy 0 <= z0_low < z0_high: an intensity is not negative")
        if cfg.eval.posterior_reference == "analytic":
            raise ValueError("the exact CIR posterior assumes a fixed initial intensity: with data.truth.z0_std > 0 use "
                             "eval.posterior_reference: mcmc (or none)")
    if truth.z0_modes is not None:
        modes = [float(v) for v in truth.z0_modes]
        if len(modes) < 2:
            raise ValueError("data.truth.z0_modes needs at least two means (one normal is data.truth.z0 with z0_std)")
        if truth.z0_std <= 0 or truth.z0_high is None:
            raise ValueError("data.truth.z0_modes needs z0_std > 0 (the standard deviation of every mode) and a finite z0_high")
        if any(not float(truth.z0_low) < m < float(truth.z0_high) for m in modes):
            raise ValueError("every mean in data.truth.z0_modes must lie strictly between z0_low and z0_high")
        if truth.z0_weights is not None and (len(truth.z0_weights) != len(modes) or any(float(w) <= 0 for w in truth.z0_weights)):
            raise ValueError("data.truth.z0_weights needs one positive weight per mean in z0_modes")
    elif truth.z0_weights is not None:
        raise ValueError("data.truth.z0_weights is only used together with data.truth.z0_modes")
    if (cfg.burn_steps == 0 and cfg.model.state_space == "real" and z0_mode(cfg) != "estimate" and model_z0(cfg) <= 0):
        raise ValueError("the initial intensity (model.z0, by default data.truth.z0) must be strictly positive for a real-state link")
    if cfg.model.drift.kind == "cir" and cfg.model.state_space != "intensity":
        raise ValueError("the cir drift kappa (theta - Z) is a drift of the intensity: it needs model.state_space: intensity")
    if cfg.model.diffusion.kind not in {"mlp", "constant", "cir", "cir_sqrt", "cir_exponential"}:
        raise ValueError("model.diffusion.kind must be mlp|constant|cir (or its coordinate forms cir_sqrt|cir_exponential)")
    if cfg.model.state_space == "real" and cfg.model.diffusion.kind == "cir_sqrt":
        raise ValueError("cir_sqrt diffusion requires intensity states; use cir, which follows model.state_space")
    if cfg.model.diffusion.kind == "cir_exponential" and (
        cfg.model.state_space != "real" or cfg.model.link not in {"exponential", "exp"}
    ):
        raise ValueError("cir_exponential diffusion requires real states and an exponential link")
    eta_init = cfg.model.diffusion.eta_init
    if eta_init == "auto":
        if cfg.model.diffusion.kind not in {"cir", "cir_sqrt", "cir_exponential"} or cfg.model.diffusion.trainable:
            raise ValueError("model.diffusion.eta_init: auto estimates the coefficient of a CIR diffusion that is not trained: "
                             "it needs model.diffusion.kind: cir and trainable: false")
    elif isinstance(eta_init, bool) or not isinstance(eta_init, (int, float)) or eta_init <= 0:
        raise ValueError("model.diffusion.eta_init must be a positive number (or auto for a CIR diffusion that is not trained)")
    if cfg.model.diffusion.min_scale <= 0:
        raise ValueError("diffusion scales must be positive")
    step_std = cfg.model.diffusion.max_step_std
    if step_std is not None and (isinstance(step_std, bool) or not isinstance(step_std, (int, float)) or step_std <= 0):
        raise ValueError("model.diffusion.max_step_std must be a positive number (or null for no bound): the largest standard "
                         "deviation of the noise that one Euler step adds to the log-intensity under the CIR diffusion")
    for name in ("max_abs_score", "max_abs_control"):
        bound = getattr(cfg.train, name)
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or bound <= 0):
            raise ValueError(f"train.{name} must be a positive number (or null for no bound)")
    if cfg.train.warm_start not in {"off", "auto"}:
        raise ValueError("train.warm_start must be off|auto")
    if cfg.model.posterior.jump_term not in {"none", "raw", "baseline_removed"}:
        raise ValueError("model.posterior.jump_term must be none|raw|baseline_removed")
    if cfg.model.intensity_scale <= 0:
        raise ValueError("model.intensity_scale must be positive: it sets the scale of the state inputs of the networks")
    if cfg.model.posterior.count_scale <= 0:
        raise ValueError("model.posterior.count_scale must be positive: it normalises a network input")
    if cfg.model.posterior.gap_scale is not None and cfg.model.posterior.gap_scale <= 0:
        raise ValueError("model.posterior.gap_scale must be positive (or null for T / count_scale)")
    for part in ("drift", "diffusion", "posterior"):
        z_scale = getattr(cfg.model, part).z_scale
        if z_scale is not None and z_scale <= 0:   # None: not resolved yet, see resolve_input_scales
            raise ValueError(f"model.{part}.z_scale must be positive (or null to derive it from model.intensity_scale): "
                             "it normalises a network input")
    if cfg.model.positivity not in {"full_truncation", "reflect", "softplus"}:
        raise ValueError("model.positivity must be full_truncation|reflect|softplus")
    if cfg.train.grad_mode not in {"backprop", "sensitivity"}:
        raise ValueError("train.grad_mode must be backprop|sensitivity")
    if not cfg.train.horizons:
        raise ValueError("train.horizons must be non-empty")
    for h in cfg.train.horizons + cfg.train.eval_horizon_fracs:
        if not (0.0 < h <= 1.0):
            raise ValueError(f"horizon fractions must be in (0, 1], got {h}")
    if all(cfg.horizon_steps(h) < cfg.horizon_steps(1.0) for h in cfg.train.horizons):
        raise ValueError(f"train.horizons must include the full horizon 1.0, got {cfg.train.horizons}: the monitoring "
                         "and the evaluation use the posterior on the whole observation window")
    validate_lr_schedule(cfg.optim)
    if cfg.eval.mcmc.kind not in {"pcn", "independence"}:
        raise ValueError("eval.mcmc.kind must be pcn|independence")
    if cfg.eval.posterior_reference not in {"analytic", "mcmc", "none"}:
        raise ValueError("eval.posterior_reference must be analytic|mcmc|none")


def validate_lr_schedule(optim: OptimConfig) -> None:
    if optim.lr_schedule not in {"none", "cosine", "exponential", "plateau"}:
        raise ValueError("optim.lr_schedule must be none|cosine|exponential|plateau")
    low_ok = optim.lr_min_factor > 0 if optim.lr_schedule == "exponential" else optim.lr_min_factor >= 0
    if not (low_ok and optim.lr_min_factor <= 1):
        raise ValueError("optim.lr_min_factor must be in (0, 1] (0 is allowed for the cosine schedule): it is the final "
                         "learning rate as a fraction of the initial one")


def save_config(cfg: ExperimentConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
