"""Evaluation suite for a finished training run (docs/DESIGN.md Sec. 5).

``evaluate_run(run_dir)`` loads ``run_dir/config.yaml`` and a checkpoint, re-creates the cached dataset and the
ground-truth prior, and reproduces the paper's figures and tables into ``run_dir/eval/``:

======================  ====================================================================  =====================
stage (``which``)       outputs                                                               paper
======================  ====================================================================  =====================
``curves``              ``training_curves.png``, ``param_history.png``, final test ELBO       training diagnostics
                        (``evaluate_elbo`` on the whole test set)
``drift``               ``drift_comparison.png``, ``drift_heatmap.png``, ``drift_3d.html``,   Fig. 2
                        learned scalar parameters vs truth
``l2``                  ``expected_path_deviation``, ``prior_paths_comparison.png``          Table 1
``samples``             ``generated_vs_true_counts.png`` (+ ``_paths.png``)                   Fig. 3
``posterior``           ``posterior_{fewest,most}_events_{full,partial}.png``,                Fig. 4a / 4b
                        reference metadata (MCMC traces for the MCMC reference)
``initial``             ``initial_distribution.png``: learned against true distribution of    (with a burn-in)
                        ``Z_0``
``wasserstein``         ``wasserstein.png``, W2 summary (train / test / prior baseline)       Fig. 5 (single ``n``)
======================  ====================================================================  =====================

Observations of an unspecified model (``data.source: file``) have no truth. The stages that compare with one
(``l2``, ``wasserstein``) are skipped and the others change their reference:

======================  ====================================================================
``drift``               ``learned_model.png``: the learned prior against the observations (event rate over
                        time, spread between observations, events per observation), and the learned drift
                        and diffusion of the intensity
``curves``              also the marginal log-likelihood of the test observations (particle filter) next to
                        the ELBO and to two reference models without a latent path (``test_likelihood``)
``samples``             ``generated_vs_true_counts.png``: generated against the observed test data
``initial``             ``initial_distribution.png``: the learned distribution of ``Z_0`` (with a burn-in)
``posterior``           the learned posterior against MCMC under the learned prior
======================  ====================================================================

Everything is recorded in ``metrics.json`` together with the resolved config, per-stage timings and -- for stages
that failed -- the error and traceback (a failing stage never aborts the others). Progress is logged to
``eval.log``. Learned paths use their configured simulation scheme; analytic CIR reference paths use float64.
Sampling runs under ``torch.no_grad()`` and draws randomness from
explicit ``torch.Generator`` objects (one per stage, seeded from the config) so that any subset of stages is
reproducible on its own.

``ndim.mcmc`` and ``ndim.metrics`` are imported lazily inside the stages that need them.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from .config import ExperimentConfig, MCMCConfig, config_from_dict, load_yaml_with_extends, model_z0, parse_overrides
from .data import (Dataset, estimate_initial_intensity, load_or_generate_dataset, make_truth_diffusion, make_truth_drift, sample_truth_z0,
                   truth_z0_cdf, truth_z0_pdf)
from .elbo import evaluate_elbo
from .model import NDIM
from .plots import (
    plot_cumulative_counts,
    plot_drift_3d_html,
    plot_drift_comparison,
    plot_drift_heatmap,
    plot_initial_distribution,
    plot_learned_model,
    plot_mcmc_traces,
    plot_parameter_history,
    plot_posterior_comparison,
    plot_prior_paths_comparison,
    plot_training_curves,
    plot_wasserstein_bars,
)
from .sde import brownian_increments, simulate_prior
from .utils import ensure_dir, get_logger, load_json, resolve_device, save_json, timer

Tensor = torch.Tensor
DriftFn = Callable[[Tensor, Tensor], Tensor]

#: evaluation stages in execution order (cheap diagnostics first, posterior sampling stages last)
STAGES: Tuple[str, ...] = ("curves", "drift", "l2", "samples", "initial", "posterior", "wasserstein")
#: stages that run without a truth (observations of an unspecified model)
TRUTH_FREE_STAGES: Tuple[str, ...] = ("curves", "drift", "samples", "initial", "posterior")
#: only these config sections may be changed at evaluation time (the architecture is fixed by the checkpoint)
EVAL_OVERRIDE_KEYS: Tuple[str, ...] = ("eval", "train.device", "train.num_threads")


# --------------------------------------------------------------------------------------
# Config / checkpoint / history loading
# --------------------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _eval_patch(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the ``eval`` section and ``train.device`` / ``train.num_threads`` of a config dict."""
    patch: Dict[str, Any] = {}
    if isinstance(d.get("eval"), dict):
        patch["eval"] = copy.deepcopy(d["eval"])
    tr = d.get("train") or {}
    if isinstance(tr, dict):
        tr_patch = {k: tr[k] for k in ("device", "num_threads") if k in tr}
        if tr_patch:
            patch["train"] = tr_patch
    return patch


def _is_eval_override(key: str) -> bool:
    key = key.strip()
    return any(key == k or key.startswith(k + ".") for k in EVAL_OVERRIDE_KEYS)


def load_eval_config(
    run_dir: Path,
    profile: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> ExperimentConfig:
    """``run_dir/config.yaml`` with a hardware ``profile`` and ``key=value`` ``overrides`` applied to the ``eval``
    section and ``train.device`` / ``train.num_threads`` only (everything else -- data, model, training -- is what
    the checkpoint was trained with). Ignored overrides are logged as warnings.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found: run_dir must be the output directory of ndim.train.train")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}
    if profile:
        raw = _deep_update(raw, _eval_patch(load_yaml_with_extends(profile)))
    if overrides:
        kept = [o for o in overrides if "=" in o and _is_eval_override(o.split("=", 1)[0])]
        dropped = [o for o in overrides if o not in kept]
        if dropped and logger is not None:
            logger.warning("ignoring non-evaluation overrides %s (only eval.*, train.device, train.num_threads apply)", dropped)
        if kept:
            raw = _deep_update(raw, parse_overrides(kept))
    return config_from_dict(raw)


def _checkpoint_path(run_dir: Path, ckpt: str) -> Path:
    p = Path(run_dir) / (f"ckpt_{ckpt}.pt" if not str(ckpt).endswith(".pt") else str(ckpt))
    if not p.exists():
        avail = sorted(q.name for q in Path(run_dir).glob("ckpt_*.pt"))
        raise FileNotFoundError(f"checkpoint {p} not found (available: {avail})")
    return p


def _load_history(run_dir: Path, extra: Dict[str, Any]) -> List[Dict[str, float]]:
    hp = Path(run_dir) / "history.json"
    if hp.exists():
        try:
            return list(load_json(hp))
        except Exception:  # pragma: no cover - corrupt file, fall back to the checkpoint copy
            pass
    return list(extra.get("history", []) or [])


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-serialisable Python objects; non-finite floats become ``None``."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, torch.Tensor):
        return _json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _subsample(x: Any, n_max: int = 200) -> List[float]:
    """Evenly subsample a 1-D trace to at most ``n_max`` points (keeps ``metrics.json`` small)."""
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size <= n_max:
        return a.tolist()
    idx = np.linspace(0, a.size - 1, n_max).round().astype(int)
    return a[idx].tolist()


def _configure_posterior_reference(metrics: Dict[str, Any], cfg: ExperimentConfig,
                                    out_dir: Path, has_truth: bool = True) -> str:
    """Record the reference of the comparison figures and discard results computed with another one.

    The posterior plots and Wasserstein distances use ``cfg.eval.posterior_reference``. Reusing untouched stages of
    an earlier report that used a different reference would silently mix methods.
    """
    from .metrics import resolve_posterior_reference
    # without a truth: MCMC under the learned prior
    kind = resolve_posterior_reference(cfg) if has_truth else "mcmc_learned_prior"
    previous = metrics.get("posterior_reference", {}).get("kind")
    stale = {"posterior", "wasserstein"} if previous is not None and previous != kind else set()
    for stage in stale:
        metrics.pop(stage, None)
        metrics.get("stages", {}).pop(stage, None)
        metrics.get("timings", {}).pop(stage, None)
    metrics["errors"] = [error for error in metrics.get("errors", []) if error.get("stage") not in stale]
    if kind != "mcmc":
        (Path(out_dir) / "mcmc_traces.png").unlink(missing_ok=True)
    labels = {"analytic": "Analytic CIR posterior", "mcmc": "MCMC posterior", "none": "No posterior reference",
              "mcmc_learned_prior": "MCMC posterior (learned prior)"}
    truth = cfg.data.truth
    metrics["posterior_reference"] = {
        "kind": kind, "label": labels[kind],
        "truth": {key: float(getattr(truth, key)) for key in ("kappa", "theta", "eta", "z0")} if has_truth else None,
        "uses_learned_parameters": not has_truth,
    }
    metrics["mcmc"] = {"config": dataclasses.asdict(cfg.eval.mcmc), "comparison_figures": kind == "mcmc"}
    return kind


# --------------------------------------------------------------------------------------
# Shared evaluation state
# --------------------------------------------------------------------------------------
@dataclass
class _Ctx:
    cfg: ExperimentConfig
    model: NDIM
    ds: Dataset
    truth_drift: DriftFn
    truth_diffusion: DriftFn
    device: torch.device
    out_dir: Path
    run_dir: Path
    logger: logging.Logger
    history: List[Dict[str, float]]
    ckpt_extra: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)

    # convenience -----------------------------------------------------------------
    @property
    def N(self) -> int:
        return int(self.cfg.data.n_steps)

    @property
    def dt(self) -> float:
        return float(self.cfg.dt)

    @property
    def T(self) -> float:
        return float(self.cfg.data.T)

    @property
    def z0(self) -> float:
        """Initial intensity of the model (the truth has ``cfg.data.truth.z0``)."""
        return model_z0(self.cfg)

    def gen(self, offset: int) -> torch.Generator:
        """Fresh CPU generator for one stage / sub-task (``train.seed`` based; VI and prior sampling)."""
        return torch.Generator().manual_seed(int(self.cfg.train.seed) * 1000 + 7919 + int(offset))

    def mcmc_gen(self, offset: int) -> torch.Generator:
        """Fresh CPU generator for the truth reference (``eval.mcmc.seed`` based)."""
        return torch.Generator().manual_seed(int(self.cfg.eval.mcmc.seed) * 1000 + int(offset))

    def mcmc_cfg(self, n_chains: Optional[int] = None) -> MCMCConfig:
        m = self.cfg.eval.mcmc
        return dataclasses.replace(m, n_chains=int(n_chains)) if n_chains is not None else dataclasses.replace(m)

    def truth_scalars(self) -> Dict[str, float]:
        tr = self.cfg.data.truth
        return {"kappa": float(tr.kappa), "theta": float(tr.theta), "eta": float(tr.eta), "z0": float(tr.z0)}

    def save(self) -> None:
        save_json(_json_safe(self.metrics), self.out_dir / "metrics.json")


def _prior_paths(ctx: _Ctx, drift: DriftFn, diffusion: DriftFn, W: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Paths of the true prior on the observation window: the truth's drift and diffusion from the truth's initial
    intensity (one draw per path when it is random; ``generator`` is used for those draws only)."""
    z0 = sample_truth_z0(ctx.cfg.data.truth, W.shape[0], generator).to(W.device)
    return simulate_prior(drift, diffusion, z0, ctx.N, ctx.dt, W, "full_truncation", ctx.cfg.model.softplus_scale)


@torch.no_grad()
def _vi_paths(ctx: _Ctx, events: Tensor, n_prime: int, W: Tensor) -> Tensor:
    """Amortised posterior paths on the observation window for ``events [B, K]`` with ``W [B*m, n_burn + N]``: the
    correction is on for the burn-in and for ``k < n_prime`` window steps, the prior dynamics run after (Eq. 3)."""
    cfg = ctx.cfg
    sim = ctx.model.simulate_posterior(events.to(ctx.device), int(n_prime), W, ctx.N,
                                       max_abs_score=cfg.train.max_abs_score, max_abs_control=cfg.train.max_abs_control)
    return sim.Z[:, sim.n_burn:]


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------
def _stage_curves(ctx: _Ctx) -> Dict[str, Any]:
    """Training curves + parameter trajectories + final test ELBO (``evaluate_elbo`` on the whole test set)."""
    cfg, out = ctx.cfg, ctx.out_dir
    res: Dict[str, Any] = {}
    if ctx.history:
        plot_training_curves(ctx.history, out / "training_curves.png", title=f"{cfg.name}: training curves (train vs test)")
        keys = [k for k in ("drift.kappa", "drift.theta", "diffusion.eta") if any(k in h for h in ctx.history)]
        if keys:
            plot_parameter_history(ctx.history, keys, truth=ctx.model.comparable_truth_scalars(), path=out / "param_history.png",
                                   title=f"{cfg.name}: learned scalar parameters")
        else:
            ctx.logger.info("no scalar parameters in the history (MLP drift, fixed eta): param_history.png skipped")
        last = ctx.history[-1]
        res["history_last"] = {k: v for k, v in last.items() if isinstance(v, (int, float))}
        res["epochs"] = int(len(ctx.history))
    else:
        ctx.logger.warning("empty training history: training_curves.png / param_history.png skipped")
    res["best_epoch"] = ctx.ckpt_extra.get("best_epoch")
    res["best_test_elbo"] = ctx.ckpt_extra.get("best_test_elbo")
    res["checkpoint_epoch"] = ctx.ckpt_extra.get("epoch")

    m = int(cfg.train.eval_paths)
    bounds: Dict[str, Any] = {}
    with timer() as t:
        te = evaluate_elbo(ctx.model, ctx.ds.test.events, ctx.N, m, cfg, ctx.device, ctx.gen(1), bounds=bounds)
    res["test_elbo"] = {"elbo": te["elbo"], "loglik": te["loglik"], "kl": te["kl"], "n_obs": int(len(ctx.ds.test)), "n_paths": m,
                        "horizon_frac": 1.0, "seconds": t["seconds"], "bounds": bounds}
    ctx.logger.info("final test ELBO %.3f (loglik %.3f, KL %.3f) on %d observations x %d paths", te["elbo"], te["loglik"], te["kl"], len(ctx.ds.test), m)
    if bounds["control_bound_share"] is not None or bounds["diffusion_bound_share"] is not None:
        def percent(share: Optional[float]) -> str:
            return "not set" if share is None else f"held on {100 * share:.2f}% of the steps"
        ctx.logger.info("bounds on the posterior paths of the test observations: largest |u| %.2f; bound on the control (%s) %s; bound on "
                        "the diffusion %s", bounds["largest_abs_control"], bounds["control_bound"], percent(bounds["control_bound_share"]),
                        percent(bounds["diffusion_bound_share"]))
    ctx.metrics["test_elbo"] = res["test_elbo"]
    if not ctx.ds.has_truth:
        res["test_likelihood"] = _test_likelihood(ctx, float(te["elbo"]))
    return res


def _test_likelihood(ctx: _Ctx, elbo: float) -> Dict[str, Any]:
    """Observations of an unspecified model: the marginal log-likelihood of the test observations under the learned
    prior (particle filter), next to the ELBO and to two reference models without a latent path (a Poisson process
    with the event rate of the training observations, and the same with one gamma factor per observation).

    The ELBO is a lower bound of the marginal log-likelihood; their difference is what the amortised posterior
    loses. Models are compared by the marginal log-likelihood, and against the references observation by
    observation, because the log-likelihood of an observation grows with its number of events."""
    from .metrics import marginal_loglik_particle_filter, reference_logliks

    cfg = ctx.cfg
    n_particles = int(cfg.eval.n_particles)
    with timer() as t:
        logp, ess = marginal_loglik_particle_filter(ctx.model, ctx.ds.test.events, n_particles, ctx.gen(8))
    logp = logp.numpy()
    ref = reference_logliks(ctx.ds.train.events, ctx.ds.test.events, ctx.dt, ctx.N)
    n = len(logp)

    def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        d = a - b
        return {"mean": float(d.mean()), "se": float(d.std() / math.sqrt(n)) if n > 1 else float("nan")}

    res = {
        "n_obs": n, "n_particles": n_particles, "seconds": t["seconds"],
        "marginal_loglik": float(logp.mean()), "elbo": elbo, "elbo_gap": float(logp.mean()) - elbo,
        "smallest_effective_sample_size_median": float(ess.median()),
        "poisson": float(ref["poisson"].mean()), "poisson_gamma": float(ref["poisson_gamma"].mean()),
        "gamma_shape": float(ref["gamma_shape"]), "gamma_factor_cv": float(ref["gamma_factor_cv"]),
        "marginal_minus_poisson": paired(logp, ref["poisson"]), "marginal_minus_poisson_gamma": paired(logp, ref["poisson_gamma"]),
        "per_observation": {"marginal_loglik": [float(v) for v in logp], "poisson": [float(v) for v in ref["poisson"]],
                            "poisson_gamma": [float(v) for v in ref["poisson_gamma"]], "n_events": [int(v) for v in ctx.ds.test.counts]},
    }
    ctx.metrics["test_likelihood"] = res
    ctx.logger.info("marginal log-likelihood per test observation %.3f (particle filter, %d particles; ELBO %.3f, %.3f below) | Poisson process "
                    "with the rate of the training observations %.3f (model %+.3f +- %.3f) | with one gamma factor per observation %.3f "
                    "(model %+.3f +- %.3f; factor CV %.3f)", res["marginal_loglik"], n_particles, elbo, res["elbo_gap"], res["poisson"],
                    res["marginal_minus_poisson"]["mean"], res["marginal_minus_poisson"]["se"], res["poisson_gamma"],
                    res["marginal_minus_poisson_gamma"]["mean"], res["marginal_minus_poisson_gamma"]["se"], res["gamma_factor_cv"])
    return res


def _stage_learned_model(ctx: _Ctx) -> Dict[str, Any]:
    """Observations of an unspecified model: the learned prior against the observations, and the learned drift and
    diffusion (``learned_model.png``).

    Nothing can be compared with a truth, so the model is checked against the data it was fitted to: the mean
    intensity of the learned prior against the event rate of the training observations, bin by bin, and the spread
    of the number of events per bin between observations, which is the Poisson noise plus the variation of the
    intensity between observations. The number of events of the model in a bin is Poisson given the path, so its
    variance is ``E[L] + Var[L]`` with ``L`` the integral of the intensity over the bin."""
    from .metrics import STIFF_SHARE, euler_gain_summary, events_per_step

    cfg, out, model = ctx.cfg, ctx.out_dir, ctx.model
    N, nb, dt, T = ctx.N, model.n_burn, ctx.dt, ctx.T
    n_paths = max(int(cfg.eval.n_initial), 2)
    W = brownian_increments(n_paths, nb + N, dt, ctx.device, ctx.gen(6))
    paths = model.simulate_prior(W, full=True).detach().cpu().to(torch.float64)                      # [n, nb + N + 1] from t_start
    finite = torch.isfinite(paths).all(dim=1)
    n_bad = int((~finite).sum())
    if n_bad:
        ctx.logger.warning("%d of %d prior paths are not finite and are left out", n_bad, n_paths)
    if int(finite.sum()) < 2:
        raise FloatingPointError("the learned prior produced no finite paths")
    paths = paths[finite]
    window = paths[:, nb:]                                                                          # [n, N + 1] on [0, T]
    t_grid = model.t_start + np.arange(nb + N + 1) * dt

    # time bins of a whole number of Euler steps: at most 48, with about 50 events per bin at least
    train = ctx.ds.train
    total = float(train.counts.sum())
    target = int(min(48, N, max(6, total // 50)))
    per_bin = min((d for d in range(1, N + 1) if N % d == 0), key=lambda d: (abs(N // d - target), d))
    k_edges = np.arange(0, N + 1, per_bin)
    edges = k_edges * dt
    per_step = events_per_step(train.events, dt, N)                                                 # the likelihood's own assignment of events to steps
    counts = torch.stack([per_step[:, a:b].sum(dim=1) for a, b in zip(k_edges[:-1], k_edges[1:])], dim=1)       # [n_obs, n_bins]
    widths = np.diff(edges)
    lam = torch.stack([window[:, a:b].sum(dim=1) * dt for a, b in zip(k_edges[:-1], k_edges[1:])], dim=1)       # [n, n_bins], left convention
    observed = {"rate": (counts.mean(0).numpy() / widths), "count_sd": counts.std(0, unbiased=False).numpy(),
                "totals": train.counts.to(torch.float64).numpy()}
    lam_total = window[:, :-1].sum(dim=1) * dt
    generated = torch.poisson(lam_total.clamp_min(0.0), generator=ctx.gen(7))
    fitted = {"rate": (lam.mean(0).numpy() / widths), "count_sd": torch.sqrt(lam.mean(0) + lam.var(0, unbiased=False)).numpy(),
              "totals": generated.numpy()}
    plot_learned_model(t_grid, paths.numpy(), edges, observed, fitted, model.intensity_drift, model.intensity_diffusion, T,
                       path=out / "learned_model.png", device=ctx.device,
                       title=f"{cfg.name}: the learned model against the {len(train)} training observations")

    # the diffusion of the state along the paths of the learned prior, on the window
    tw = torch.as_tensor(t_grid[nb:], dtype=torch.float32, device=ctx.device).expand(window.shape[0], -1)
    zw = window.clamp_min(1e-12).to(torch.float32).to(ctx.device)
    sig_z = model.intensity_diffusion(zw.reshape(-1), tw.reshape(-1)).to(torch.float64).cpu()
    rel = (sig_z / zw.reshape(-1).to(torch.float64).cpu().clamp_min(1e-12))                          # relative: per unit of intensity
    q = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    mean_path, sd_path = window.mean(0), window.std(0)
    res: Dict[str, Any] = {
        "n_prior_paths": int(paths.shape[0]), "n_nonfinite_prior_paths": n_bad,
        "bin_edges": [float(v) for v in edges],
        "rate_observed": [float(v) for v in observed["rate"]], "rate_learned_prior": [float(v) for v in fitted["rate"]],
        "count_sd_observed": [float(v) for v in observed["count_sd"]], "count_sd_learned_model": [float(v) for v in fitted["count_sd"]],
        "rate_rmse": float(np.sqrt(np.mean((fitted["rate"] - observed["rate"]) ** 2))),
        "rate_rmse_relative": float(np.sqrt(np.mean((fitted["rate"] - observed["rate"]) ** 2)) / max(float(np.mean(observed["rate"])), 1e-12)),
        "count_sd_ratio_mean": float(np.mean(fitted["count_sd"] / np.maximum(observed["count_sd"], 1e-12))),
        "events_per_observation": {"observed_mean": float(observed["totals"].mean()), "observed_std": float(observed["totals"].std()),
                                   "learned_mean": float(generated.mean()), "learned_std": float(generated.std(unbiased=False))},
        "intensity_cv_mean_over_time": float((sd_path / mean_path.clamp_min(1e-12)).mean()),
        "euler_gain": euler_gain_summary(model, paths.to(ctx.device), t_grid, dt, nb),
        # share of the steps of the prior paths (burn-in included) on which the bound on the diffusion holds
        "diffusion_bound_share": float(model.diffusion.bound_active(
            model.link.inverse(paths[:, :-1].clamp_min(1e-300)).to(torch.float32).to(ctx.device)).double().mean()) if model.real_states else None,
        "intensity_diffusion_quantiles_5_50_95": [float(v) for v in torch.quantile(sig_z, q)],
        "relative_diffusion_quantiles_5_50_95": [float(v) for v in torch.quantile(rel, q)],
        "learned_scalars": model.named_scalars(),
    }
    ctx.metrics["learned_model"] = res
    e = res["events_per_observation"]
    ctx.logger.info("learned prior against the training observations: event rate off by %.1f%% (RMSE over %d bins, relative to the mean rate) | "
                    "events per observation: mean %.1f (observed %.1f), std %.1f (observed %.1f) | std of the events per bin: %.2f of the observed",
                    100 * res["rate_rmse_relative"], len(widths), e["learned_mean"], e["observed_mean"], e["learned_std"], e["observed_std"],
                    res["count_sd_ratio_mean"])
    gain = res["euler_gain"]
    if (gain["share_below_minus_one"] or 0.0) > STIFF_SHARE:
        ctx.logger.warning("the learned drift is too stiff for the Euler grid on %.1f%% of the steps of the prior paths (gain "
                           "1 + dt db/dx below -1, smallest %.2f): the fit is fragile there; use a finer grid (data.n_steps)",
                           100 * gain["share_below_minus_one"], gain["min"])
    return res


def _stage_drift(ctx: _Ctx) -> Dict[str, Any]:
    """Learned vs true drift figures (Fig. 2) and learned scalar parameters vs truth. Without a truth:
    the learned model against the observations (``_stage_learned_model``)."""
    if not ctx.ds.has_truth:
        return _stage_learned_model(ctx)
    cfg, out = ctx.cfg, ctx.out_dir
    learned = ctx.model.intensity_drift
    z_min = 1e-3 if ctx.model.real_states else 0.0
    z_max, T = float(cfg.eval.drift_z_max), ctx.T
    occ = ctx.ds.train_paths
    plot_drift_comparison(learned, ctx.truth_drift, z_max, T, times=(0.0, 0.5, 1.0), occupancy=occ, path=out / "drift_comparison.png",
                          title=f"{cfg.name}: intensity drift vs truth b(z, t)", device=ctx.device, z_min=z_min)
    plot_drift_heatmap(learned, ctx.truth_drift, z_max, T, occ, path=out / "drift_heatmap.png",
                       title=f"{cfg.name}: intensity drift over (t, z) with occupation of the truth", device=ctx.device, z_min=z_min)
    res: Dict[str, Any] = {}
    if cfg.eval.drift_3d_html:
        p = plot_drift_3d_html(learned, ctx.truth_drift, float(cfg.eval.drift_3d_z_max), T, out / "drift_3d.html",
                               device=ctx.device, z_min=z_min)
        res["drift_3d_html"] = None if p is None else str(p)
    # scalar parameters vs truth
    learned_scalars = ctx.model.named_scalars()
    truth = ctx.truth_scalars()
    params: Dict[str, Any] = {"learned": learned_scalars, "truth": {k: truth[k] for k in ("kappa", "theta", "eta")}}
    rel: Dict[str, float] = {}
    comparable_truth = ctx.model.comparable_truth_scalars()
    for k, v in learned_scalars.items():
        name = k.split(".")[-1]
        if name in comparable_truth and comparable_truth[name] != 0:
            rel[k] = float(v - comparable_truth[name]) / abs(comparable_truth[name])
    params["relative_error"] = rel
    params["learned_state_space"] = cfg.model.state_space
    res["parameters"] = params
    ctx.metrics["parameters"] = params
    # drift error where the data live: RMSE of b_learned - b_true at the (Z_k, t_k) visited by the true training paths
    with torch.no_grad():
        P = occ.to(ctx.device)
        tg = (torch.arange(P.shape[1], device=ctx.device, dtype=P.dtype) * ctx.dt).expand_as(P)
        z, t = P.reshape(-1), tg.reshape(-1)
        valid = z > 0 if ctx.model.real_states else torch.ones_like(z, dtype=torch.bool)
        d = (learned(z[valid], t[valid]) - ctx.truth_drift(z[valid], t[valid])).to(torch.float64)
        res["drift_error_on_occupied_states"] = {
            "rmse": float(torch.sqrt((d ** 2).mean())),
            "mean_abs": float(d.abs().mean()),
            "max_abs": float(d.abs().max()),
            "n_points": int(d.numel()),
            "n_zero_intensities_excluded": int((~valid).sum()),
        }
    if cfg.model.diffusion.trainable:
        plot_drift_comparison(ctx.model.intensity_diffusion, ctx.truth_diffusion, z_max, T,
                              occupancy=occ, path=out / "diffusion_comparison.png", device=ctx.device,
                              title=f"{cfg.name}: intensity diffusion vs truth", z_min=z_min, quantity="diffusion")
        with torch.no_grad():
            diff = (ctx.model.intensity_diffusion(z[valid], t[valid]) - ctx.truth_diffusion(z[valid], t[valid])).double()
            res["diffusion_error_on_occupied_states"] = {
                "rmse": float(diff.square().mean().sqrt()), "mean_abs": float(diff.abs().mean()),
                "n_points": int(diff.numel()), "n_zero_intensities_excluded": int((~valid).sum())}
        ctx.metrics["diffusion_error_on_occupied_states"] = res["diffusion_error_on_occupied_states"]
    ctx.metrics["drift_error_on_occupied_states"] = res["drift_error_on_occupied_states"]
    ctx.logger.info("drift RMSE on occupied states %.3f | learned scalars %s", res["drift_error_on_occupied_states"]["rmse"], learned_scalars)
    return res


def _stage_l2(ctx: _Ctx) -> Dict[str, Any]:
    """Expected path deviation between learned and true prior under common Brownian noise."""
    from .metrics import expected_path_deviation

    cfg = ctx.cfg
    n_paths = int(cfg.eval.l2_n_paths)
    kwargs: Dict[str, Any] = {}
    try:
        parameters = inspect.signature(expected_path_deviation).parameters
        if "true_diffusion_fn" in parameters:
            kwargs["true_diffusion_fn"] = ctx.truth_diffusion
        if "return_paths" in parameters:
            kwargs["return_paths"] = True
    except (TypeError, ValueError):  # pragma: no cover - builtins without signature
        pass
    with torch.no_grad():
        output = expected_path_deviation(ctx.model.drift, ctx.truth_drift, ctx.model.diffusion, cfg, n_paths, ctx.device, ctx.gen(2), **kwargs)
    if isinstance(output, tuple):
        dev, paths = output
    else:
        dev, paths = output, None
    dev = dict(dev)
    dev["n_paths"] = n_paths
    dev["true_diffusion_used_for_truth"] = "true_diffusion_fn" in kwargs
    if paths is not None:
        truth, learned = paths["true"], paths["learned"]
        true_scheme = "Intensity Euler (full truncation)"
        learned_scheme = (("Log-state Euler (exponential link)" if cfg.model.link in {"exponential", "exp"}
                           else f"Latent Euler ({cfg.model.link} link)") if ctx.model.real_states
                          else f"Intensity Euler ({cfg.model.positivity})")
        plot_path = ctx.out_dir / "prior_paths_comparison.png"
        plot_prior_paths_comparison(np.linspace(0., ctx.T, ctx.N + 1), truth, learned,
                                    path=plot_path, title=f"{cfg.name}: prior SDE paths under common noise",
                                    true_scheme=true_scheme, learned_scheme=learned_scheme)
        dev["path_comparison"] = {
            "plot": str(plot_path), "shared_brownian_increments": True,
            "true_scheme": true_scheme, "learned_scheme": learned_scheme,
            "n_steps": ctx.N, "dt": ctx.dt,
            "true_terminal_mean": float(truth[:, -1].mean()),
            "learned_terminal_mean": float(learned[:, -1].mean()),
            "true_terminal_std": float(truth[:, -1].std(unbiased=False)),
            "learned_terminal_std": float(learned[:, -1].std(unbiased=False)),
        }
    ctx.metrics["expected_path_deviation"] = dev
    ctx.logger.info("expected path deviation: L2 %.4g | L1 %.4g | rmse %.4g (n_paths=%d)", dev.get("l2", float("nan")), dev.get("l1", float("nan")),
                    dev.get("rmse", float("nan")), n_paths)
    return dev


def _stage_samples(ctx: _Ctx) -> Dict[str, Any]:
    """Fig. 3: cumulative counts of observations generated by the learned model vs the true (test) data."""
    from .metrics import cumulative_count_curves, generate_observations_from_model

    cfg, out = ctx.cfg, ctx.out_dir
    n_gen = int(cfg.eval.n_generated)
    with torch.no_grad():
        gen_events, gen_paths = generate_observations_from_model(ctx.model, cfg, n_gen, ctx.device, ctx.gen(3))
    true_events = ctx.ds.test.as_lists()
    kind = "true" if ctx.ds.has_truth else "observed"
    title = f"{cfg.name}: cumulative counts, {kind} test data (n={len(true_events)}) vs generated (n={len(gen_events)})"
    plot_cumulative_counts(true_events, gen_events, ctx.T, n_grid=1000, show_individual=False, path=out / "generated_vs_true_counts.png", title=title,
                           true_label=f"{kind} data (test set)")
    plot_cumulative_counts(true_events, gen_events, ctx.T, n_grid=1000, show_individual=True, max_individual=64,
                           path=out / "generated_vs_true_counts_paths.png", title=title, true_label=f"{kind} data (test set)")
    t_grid = np.linspace(0.0, ctx.T, 200)
    Nt = np.asarray(cumulative_count_curves(true_events, t_grid), dtype=float)
    Ng = np.asarray(cumulative_count_curves(gen_events, t_grid), dtype=float)
    ct = np.array([len(e) for e in true_events], dtype=float)
    cg = np.array([len(e) for e in gen_events], dtype=float)
    res: Dict[str, Any] = {
        "n_true": int(len(true_events)),
        "n_generated": int(len(gen_events)),
        "count_mean_true": float(ct.mean()) if ct.size else None,
        "count_std_true": float(ct.std()) if ct.size else None,
        "count_mean_generated": float(cg.mean()) if cg.size else None,
        "count_std_generated": float(cg.std()) if cg.size else None,
        "max_abs_diff_of_mean_curves": float(np.max(np.abs(Nt.mean(0) - Ng.mean(0)))) if (Nt.size and Ng.size) else None,
        "generated_path_mean_final": float(gen_paths[:, -1].mean()) if gen_paths.numel() else None,
        "true_path_mean_final": float(ctx.ds.test_paths[:, -1].mean()) if ctx.ds.has_truth else None,
    }
    ctx.metrics["samples"] = res
    ctx.logger.info("generated samples: count mean %.2f (true %.2f) | std %.2f (true %.2f)", res["count_mean_generated"] or float("nan"),
                    res["count_mean_true"] or float("nan"), res["count_std_generated"] or float("nan"), res["count_std_true"] or float("nan"))
    return res


def _stage_initial(ctx: _Ctx) -> Dict[str, Any]:
    """Distribution of the initial intensity ``Z_0``: the truth's against the one the learned prior produces at
    ``t = 0`` after its burn-in (``initial_distribution.png``). Skipped without a burn-in, where ``Z_0`` is a fixed number."""
    cfg, out, model = ctx.cfg, ctx.out_dir, ctx.model
    nb = model.n_burn
    if nb == 0:
        ctx.logger.info("no burn-in: the model starts at a fixed initial intensity; initial_distribution.png skipped")
        return {"skipped": True, "reason": "no burn-in"}
    n = max(int(cfg.eval.n_initial), 2)
    W = brownian_increments(n, nb, ctx.dt, ctx.device, ctx.gen(4))
    burn = model.simulate_prior(W, n_steps=0, full=True).detach().cpu().to(torch.float64)      # [n, nb + 1] on [-burn_in, 0]
    # paths that are not finite are counted and left out
    finite = torch.isfinite(burn).all(dim=1)
    n_bad = int((~finite).sum())
    if n_bad:
        ctx.logger.warning("%d of %d prior paths are not finite on the burn-in and are left out", n_bad, n)
    if int(finite.sum()) < 2:
        raise FloatingPointError("the learned prior produced no finite paths on the burn-in")
    burn = burn[finite]
    n = int(burn.shape[0])
    learned = burn[:, -1]
    burn_len, x_start = -float(model.t_start), float(model.initial_state)
    if not ctx.ds.has_truth:
        # without a truth: the learned distribution, with the event rate of the data at the start of the window
        qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
        rate = float(estimate_initial_intensity(ctx.ds.train.events, ctx.T, float(cfg.model.z0_window)))
        res = {"n_samples": n, "n_nonfinite_prior_paths": n_bad,
               "learned": {"mean": float(learned.mean()), "std": float(learned.std()),
                           "quantiles_5_25_50_75_95": [float(v) for v in torch.quantile(learned, qs)]},
               "data_event_rate_at_start": rate, "data_event_rate_window": float(cfg.model.z0_window),
               "burn_in": burn_len, "x_start": x_start}
        plot_initial_distribution(None, learned.numpy(), model.t_start + np.arange(nb + 1) * ctx.dt, burn.numpy(),
                                  path=out / "initial_distribution.png", data_rate=rate,
                                  title=f"{cfg.name}: learned distribution of the initial intensity Z_0")
        ctx.metrics["initial"] = res
        ctx.logger.info("initial intensity: learned mean %.3f sd %.3f | mean event rate of the training observations over the first "
                        "%g of [0, T]: %.3f", res["learned"]["mean"], res["learned"]["std"], float(cfg.model.z0_window), rate)
        return res
    true = sample_truth_z0(cfg.data.truth, n, ctx.gen(5)).to(torch.float64)
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)

    def describe(x: Tensor) -> Dict[str, Any]:
        return {"mean": float(x.mean()), "std": float(x.std()), "quantiles_5_25_50_75_95": [float(v) for v in torch.quantile(x, q)]}

    ls, ts = torch.sort(learned).values, torch.sort(true).values
    grid = torch.sort(torch.cat([ls, ts])).values
    ks = float((torch.searchsorted(ls, grid, right=True) - torch.searchsorted(ts, grid, right=True)).abs().max()) / n
    res: Dict[str, Any] = {"n_samples": n, "n_nonfinite_prior_paths": n_bad, "learned": describe(learned), "true": describe(true),
                           "data": describe(ctx.ds.train_paths[:, 0].to(torch.float64)),       # the Z_0 behind the training observations
                           "wasserstein1": float((ls - ts).abs().mean()), "ks_statistic": ks,
                           "burn_in": burn_len, "x_start": x_start}
    # the exact distribution of the truth
    truth = cfg.data.truth
    cdf_at = truth_z0_cdf(truth, ls)
    steps = torch.arange(n + 1, dtype=torch.float64) / n
    res["ks_statistic_exact"] = float(torch.maximum((steps[1:] - cdf_at).abs(), (steps[:-1] - cdf_at).abs()).max())
    true_modes = None
    if truth.z0_modes is not None:  # a mixture: the mass of each mode and between them
        means = sorted(float(m) for m in truth.z0_modes)
        sd = float(truth.z0_std)
        cuts = torch.tensor([0.5 * (a + b) for a, b in zip(means[:-1], means[1:])], dtype=torch.float64)
        edges = torch.cat([torch.zeros(1, dtype=torch.float64), truth_z0_cdf(truth, cuts), torch.ones(1, dtype=torch.float64)])
        learned_mass = torch.bincount(torch.bucketize(learned.contiguous(), cuts), minlength=len(means)).to(torch.float64) / n
        res["modes"] = {
            "means": means, "std": sd, "cuts": [float(c) for c in cuts],
            "mass_true": [float(v) for v in torch.diff(edges)], "mass_learned": [float(v) for v in learned_mass],
            # the mass within one standard deviation of each cut: small for separated modes
            "valley_mass_true": [float(truth_z0_cdf(truth, c + sd) - truth_z0_cdf(truth, c - sd)) for c in cuts],
            "valley_mass_learned": [float(((learned > c - sd) & (learned < c + sd)).to(torch.float64).mean()) for c in cuts],
        }
        true_modes = [(m, sd) for m in means]
    lo_x, hi_x = float(min(ls[0], ts[0])), float(max(ls[-1], ts[-1]))
    x_dens = torch.linspace(lo_x, hi_x, 600, dtype=torch.float64)
    t_burn = model.t_start + np.arange(nb + 1) * ctx.dt
    plot_initial_distribution(true.numpy(), learned.numpy(), t_burn, burn.numpy(), path=out / "initial_distribution.png",
                              title=f"{cfg.name}: distribution of the initial intensity Z_0, truth against learned prior",
                              true_density=(x_dens.numpy(), truth_z0_pdf(truth, x_dens).numpy()), true_modes=true_modes)
    ctx.metrics["initial"] = res
    ctx.logger.info("initial intensity: learned mean %.2f sd %.2f | true mean %.2f sd %.2f | W1 %.3f | KS %.3f (exact truth: %.3f)",
                    res["learned"]["mean"], res["learned"]["std"], res["true"]["mean"], res["true"]["std"], res["wasserstein1"], ks,
                    res["ks_statistic_exact"])
    if "modes" in res:
        ctx.logger.info("initial intensity, modes at %s: mass learned %s, true %s | mass between the modes learned %s, true %s",
                        res["modes"]["means"], np.round(res["modes"]["mass_learned"], 3).tolist(), np.round(res["modes"]["mass_true"], 3).tolist(),
                        np.round(res["modes"]["valley_mass_learned"], 3).tolist(), np.round(res["modes"]["valley_mass_true"], 3).tolist())
    return res


def _stage_posterior_learned_prior(ctx: _Ctx) -> Dict[str, Any]:
    """Observations of an unspecified model: the amortised posterior against MCMC under the learned prior.

    There is no truth to compare with, so the reference is a sampler for the posterior of the *same* learned prior
    (its drift, diffusion and burn-in). Agreement says that the posterior network does its job; it says nothing
    about the prior itself, which is judged by the generated counts (``samples`` stage) and the test ELBO."""
    from .metrics import learned_prior_reference_samples, wasserstein2_paths
    cfg, out, model = ctx.cfg, ctx.out_dir, ctx.model
    n_paths = int(cfg.eval.n_paths_posterior)
    counts = ctx.ds.test.counts
    cases = {"fewest_events": int(torch.argmin(counts).item()), "most_events": int(torch.argmax(counts).item())}
    horizons = {"full": ctx.N, "partial": cfg.horizon_steps(float(cfg.eval.partial_horizon_frac))}
    t_grid = np.arange(ctx.N + 1) * ctx.dt
    label = "MCMC posterior (learned prior)"
    res: Dict[str, Any] = {"n_paths": n_paths, "cases": {}, "reference_kind": "mcmc_learned_prior", "reference_label": label}
    k = 0
    for case, idx in cases.items():
        ev_row = ctx.ds.test.events[idx]
        ev_np = ev_row[torch.isfinite(ev_row)].cpu().numpy()
        for hname, n_prime in horizons.items():
            k += 1
            t_bar = float(n_prime) * ctx.dt
            n_seen = int((ev_row <= t_bar).sum().item())
            W = brownian_increments(n_paths, model.n_burn + ctx.N, ctx.dt, ctx.device, ctx.gen(10 + k))
            prior = model.simulate_prior(W).detach().cpu()                 # the learned prior on the observation window
            vi = _vi_paths(ctx, ev_row[None, :], n_prime, W).detach().cpu()
            with timer() as tm:
                reference, metadata = learned_prior_reference_samples(model, ev_row[None, :].to(ctx.device), cfg, int(n_prime), n_paths,
                                                                      ctx.device, ctx.mcmc_gen(10 + k), mcmc_cfg=ctx.mcmc_cfg(n_paths))
            reference = reference.detach().cpu()
            # a second, independent MCMC run: its distance to the first is the sampling error
            second, _ = learned_prior_reference_samples(model, ev_row[None, :].to(ctx.device), cfg, int(n_prime), n_paths,
                                                        ctx.device, ctx.mcmc_gen(60 + k), mcmc_cfg=ctx.mcmc_cfg(n_paths))
            second = second.detach().cpu()
            metadata = dict(metadata)
            diagnostics = metadata.pop("mcmc_diagnostics", [])
            metadata["seconds"] = tm["seconds"]
            rv, vv = reference.to(torch.float64).mean(0), vi.to(torch.float64).mean(0)
            entry: Dict[str, Any] = {
                "test_index": idx, "n_events_total": int(counts[idx].item()), "n_events_observed": n_seen, "n_prime": int(n_prime),
                "t_bar": t_bar, "reference": metadata,
                "w2_vi_vs_reference": float(wasserstein2_paths(vi, reference, ctx.dt)),
                "w2_prior_vs_reference": float(wasserstein2_paths(prior, reference, ctx.dt)),
                "w2_between_two_reference_runs": float(wasserstein2_paths(second, reference, ctx.dt)),
                "rmse_of_means_vi_vs_reference": float(torch.sqrt(((rv - vv) ** 2).mean())),
                "mean_at_T_vi": float(vi[:, -1].mean()), "mean_at_T_reference": float(rv[-1]), "mean_at_T_prior": float(prior[:, -1].mean()),
                "z0_vi": {"mean": float(vi[:, 0].mean()), "std": float(vi[:, 0].std())},
                "z0_reference": {"mean": float(reference[:, 0].mean()), "std": float(reference[:, 0].std())},
                "z0_prior": {"mean": float(prior[:, 0].mean()), "std": float(prior[:, 0].std())},
            }
            if diagnostics:
                d = dict(diagnostics[0])
                trace = d.pop("loglik_trace", None)
                if trace is not None:
                    d["loglik_trace_subsampled"] = _subsample(trace)
                entry["mcmc"] = d
            title = (f"{cfg.name}: test observation {idx} ({case.replace('_', ' ')}, {int(counts[idx])} events)\n"
                     + ("posteriors conditioned on the full observation [0, T]" if hname == "full"
                        else f"posteriors conditioned on [0, T'] = [0, {t_bar:g}]; prior dynamics after T'"))
            plot_posterior_comparison(t_grid, prior, reference, vi, ev_np, t_bar if hname == "partial" else None,
                                      path=out / f"posterior_{case}_{hname}.png", title=title, show_individual=False,
                                      reference_label=label, prior_label="learned prior")
            res["cases"][f"{case}_{hname}"] = entry
            ctx.logger.info("posterior %s/%s: obs %d, %d events seen | W2 learned to MCMC %.3f (between two MCMC runs %.3f, learned prior "
                            "to MCMC %.3f) | mean(T) learned %.2f, MCMC %.2f", case, hname, idx, n_seen, entry["w2_vi_vs_reference"],
                            entry["w2_between_two_reference_runs"], entry["w2_prior_vs_reference"], entry["mean_at_T_vi"],
                            entry["mean_at_T_reference"])
    res["mcmc_config"] = dataclasses.asdict(ctx.mcmc_cfg(n_paths))
    ctx.metrics["posterior"] = res
    return res


def _stage_posterior(ctx: _Ctx) -> Dict[str, Any]:
    """Compare learned posterior paths with the truth-conditioned reference (or, without a truth, with MCMC under the
    learned prior: ``_stage_posterior_learned_prior``)."""
    if not ctx.ds.has_truth:
        return _stage_posterior_learned_prior(ctx)
    from .metrics import posterior_reference_samples, resolve_posterior_reference, wasserstein2_paths
    cfg, out = ctx.cfg, ctx.out_dir
    kind = resolve_posterior_reference(cfg)
    n_paths = int(cfg.eval.n_paths_posterior)
    counts = ctx.ds.test.counts
    cases = {"fewest_events": int(torch.argmin(counts).item()), "most_events": int(torch.argmax(counts).item())}
    horizons = {"full": ctx.N, "partial": cfg.horizon_steps(float(cfg.eval.partial_horizon_frac))}
    t_grid = np.arange(ctx.N + 1) * ctx.dt
    res: Dict[str, Any] = {"n_paths": n_paths, "cases": {}, "reference_kind": kind}
    traces: Dict[str, np.ndarray] = {}
    n_burn: Optional[int] = None
    k = 0
    for case, idx in cases.items():
        ev_row = ctx.ds.test.events[idx]
        ev_np = ev_row[torch.isfinite(ev_row)].cpu().numpy()
        for hname, n_prime in horizons.items():
            k += 1
            t_bar = float(n_prime) * ctx.dt
            n_seen = int((ev_row <= t_bar).sum().item())
            W = brownian_increments(n_paths, ctx.model.n_burn + ctx.N, ctx.dt, ctx.device, ctx.gen(10 + k))
            prior = _prior_paths(ctx, ctx.truth_drift, ctx.truth_diffusion, W[:, ctx.model.n_burn:], ctx.gen(500 + k))
            vi = _vi_paths(ctx, ev_row[None, :], n_prime, W)
            entry: Dict[str, Any] = {"test_index": idx, "n_events_total": int(counts[idx].item()),
                                     "n_events_observed": n_seen, "n_prime": int(n_prime), "t_bar": t_bar}
            reference_paths, label = None, "No posterior reference"
            if kind != "none":
                with timer() as tm:
                    reference_paths, metadata = posterior_reference_samples(
                        ev_row[None, :], cfg, int(n_prime), n_paths, ctx.device,
                        ctx.mcmc_gen(10 + k), reference=kind, drift=ctx.truth_drift,
                        diffusion=ctx.truth_diffusion, mcmc_cfg=ctx.mcmc_cfg(n_paths))
                reference_paths = reference_paths.detach().cpu()
                metadata = dict(metadata)
                metadata["seconds"] = tm["seconds"]
                label = metadata.get("label", "True analytic CIR posterior" if kind == "analytic" else "MCMC posterior")
                if kind == "analytic":
                    label = "true posterior (exact CIR)" if metadata.get("exact_finite_dimensional") else label
                diagnostics = metadata.pop("mcmc_diagnostics", [])
                entry["reference"] = metadata
                if kind == "mcmc" and diagnostics:
                    d = dict(diagnostics[0])
                    trace = d.pop("loglik_trace", None)
                    if trace is not None:
                        d["loglik_trace_subsampled"] = _subsample(trace)
                        traces[f"{case}, {hname} (obs {idx}, {n_seen} events seen)"] = np.asarray(trace, dtype=float)
                    n_burn = d.get("n_burn_in", n_burn)
                    entry["mcmc"] = d
                entry["z0_reference"] = {"mean": float(reference_paths[:, 0].mean()), "std": float(reference_paths[:, 0].std())}
                entry["w2_vi_vs_reference"] = float(wasserstein2_paths(vi.cpu(), reference_paths, ctx.dt))
                entry["w2_prior_vs_reference"] = float(wasserstein2_paths(prior.cpu(), reference_paths, ctx.dt))
                rv, vv = reference_paths.to(torch.float64).mean(0), vi.cpu().to(torch.float64).mean(0)
                entry["rmse_of_means_vi_vs_reference"] = float(torch.sqrt(((rv - vv) ** 2).mean()))
                entry["mean_at_T_reference"] = float(rv[-1])
                if kind == "mcmc":
                    for target, source in (("w2_vi_vs_mcmc", "w2_vi_vs_reference"),
                                           ("w2_prior_vs_mcmc", "w2_prior_vs_reference"),
                                           ("rmse_of_means_vi_vs_mcmc", "rmse_of_means_vi_vs_reference"),
                                           ("mean_at_T_mcmc", "mean_at_T_reference")):
                        entry[target] = entry[source]
                res["reference_label"] = label
            # the initial intensity given this observation: a point without a burn-in, a distribution with one
            entry["z0_vi"] = {"mean": float(vi[:, 0].mean()), "std": float(vi[:, 0].std())}
            entry["z0_prior"] = {"mean": float(prior[:, 0].mean()), "std": float(prior[:, 0].std())}
            entry["true_z0"] = float(ctx.ds.test_paths[idx, 0])
            entry["mean_at_T_vi"] = float(vi[:, -1].mean())
            entry["mean_at_T_prior"] = float(prior[:, -1].mean())
            entry["true_path_at_T"] = float(ctx.ds.test_paths[idx, -1])
            title = (f"{cfg.name}: test observation {idx} ({case.replace('_', ' ')}, {int(counts[idx])} events)\n"
                     + ("posteriors conditioned on the full observation [0, T]" if hname == "full"
                        else f"posteriors conditioned on [0, T'] = [0, {t_bar:g}]; prior dynamics after T'"))
            plot_posterior_comparison(t_grid, prior, reference_paths, vi, ev_np, t_bar if hname == "partial" else None,
                                      path=out / f"posterior_{case}_{hname}.png", title=title,
                                      show_individual=False, reference_label=label,
                                      true_path=ctx.ds.test_paths[idx] if ctx.ds.test_paths is not None else None)
            res["cases"][f"{case}_{hname}"] = entry
            ctx.logger.info("posterior %s/%s: obs %d, %d events seen | learned mean(T) %.2f | %s %s",
                            case, hname, idx, n_seen, entry["mean_at_T_vi"], label,
                            f"mean(T) {entry['mean_at_T_reference']:.2f}" if reference_paths is not None else "off")
    if traces:
        plot_mcmc_traces(traces, n_burn_in=n_burn, path=out / "mcmc_traces.png",
                         title=f"{cfg.name}: MCMC log-likelihood traces ({cfg.eval.mcmc.kind}, {n_paths} chains)")
    elif kind != "mcmc":
        (out / "mcmc_traces.png").unlink(missing_ok=True)
    res["mcmc_config"] = dataclasses.asdict(ctx.mcmc_cfg(n_paths)) if kind == "mcmc" else None
    ctx.metrics["posterior"] = res
    return res


def _stage_wasserstein(ctx: _Ctx) -> Dict[str, Any]:
    """W2 between learned and truth-reference posterior samples (+ prior baseline)."""
    from .metrics import posterior_wasserstein, resolve_posterior_reference

    cfg, out = ctx.cfg, ctx.out_dir
    kind = resolve_posterior_reference(cfg)
    if not cfg.eval.run_wasserstein or kind == "none":
        ctx.logger.info("Wasserstein evaluation or posterior reference is disabled")
        return {"skipped": True}
    n_paths = int(cfg.eval.wasserstein_n_paths)
    rng = np.random.default_rng(int(cfg.eval.mcmc.seed))
    res: Dict[str, Any] = {"n_paths": n_paths, "n_prime": ctx.N, "reference_kind": kind}
    label = "True analytic CIR posterior" if kind == "analytic" else "MCMC posterior"
    summary: Dict[str, float] = {}
    errors: Dict[str, float] = {}
    for j, (split, data) in enumerate((("train", ctx.ds.train), ("test", ctx.ds.test))):
        n_obs = min(int(cfg.eval.wasserstein_n_obs), len(data))
        idx = sorted(rng.choice(len(data), size=n_obs, replace=False).tolist()) if n_obs < len(data) else list(range(len(data)))
        ev = data.subset(idx).events
        with torch.no_grad():
            r = posterior_wasserstein(ctx.model, ctx.truth_drift, ctx.truth_diffusion, ev, cfg, ctx.N, n_paths, ctx.mcmc_cfg(n_paths),
                                      ctx.device, ctx.mcmc_gen(50 + j), reference=kind)
        r = dict(r)
        label = r.get("reference", {}).get("label", label)
        r["indices"] = idx
        res[split] = r
        w_vi = np.asarray(r.get("w2_vi", []), dtype=float)
        w_pr = np.asarray(r.get("w2_prior_baseline", []), dtype=float)
        summary[f"{split} VI"] = float(r.get("mean_w2_vi", np.mean(w_vi) if w_vi.size else np.nan))
        errors[f"{split} VI"] = float(np.std(w_vi)) if w_vi.size > 1 else 0.0
        if split == "test":
            summary["test prior baseline"] = float(r.get("mean_w2_prior_baseline", np.mean(w_pr) if w_pr.size else np.nan))
            errors["test prior baseline"] = float(np.std(w_pr)) if w_pr.size > 1 else 0.0
        else:
            res["train_prior_baseline_mean"] = float(r.get("mean_w2_prior_baseline", np.mean(w_pr) if w_pr.size else np.nan))
        ctx.logger.info("wasserstein %s (%d obs x %d paths): VI %.4g | prior baseline %.4g", split, n_obs, n_paths, summary[f"{split} VI"],
                        float(r.get("mean_w2_prior_baseline", float("nan"))))
    res["summary"] = summary
    res["summary_std_over_obs"] = errors
    res["reference_label"] = label
    res["mcmc_config"] = dataclasses.asdict(ctx.mcmc_cfg(n_paths)) if kind == "mcmc" else None
    plot_wasserstein_bars(summary, path=out / "wasserstein.png", errors=errors,
                          ylabel=f"2-Wasserstein distance to {label}",
                          title=f"{cfg.name}: W2(learned posterior, {label}), n_train={len(ctx.ds.train)} "
                                f"({cfg.eval.wasserstein_n_obs} obs x {n_paths} paths; bars: std over observations)")
    ctx.metrics["wasserstein"] = res
    return res


_STAGE_FNS: Dict[str, Callable[[_Ctx], Dict[str, Any]]] = {
    "curves": _stage_curves,
    "drift": _stage_drift,
    "l2": _stage_l2,
    "samples": _stage_samples,
    "initial": _stage_initial,
    "posterior": _stage_posterior,
    "wasserstein": _stage_wasserstein,
}

#: top-level ``metrics.json`` keys written by each stage (dropped before the stage is re-run)
_STAGE_RESULT_KEYS: Dict[str, Tuple[str, ...]] = {
    "curves": ("test_elbo", "test_likelihood"),
    "drift": ("parameters", "drift_error_on_occupied_states", "diffusion_error_on_occupied_states", "learned_model"),
    "l2": ("expected_path_deviation",),
    "samples": ("samples",),
    "initial": ("initial",),
    "posterior": ("posterior",),
    "wasserstein": ("wasserstein",),
}


def _mcmc_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate the MCMC diagnostics scattered over the stage results (posterior cases, W2)."""
    out: Dict[str, Any] = {}
    cases = ((metrics.get("posterior") or {}).get("cases") or {})
    accs = [c["mcmc"]["acceptance_rate"] for c in cases.values() if isinstance(c.get("mcmc"), dict) and c["mcmc"].get("acceptance_rate") is not None]
    betas = [c["mcmc"]["beta_final"] for c in cases.values() if isinstance(c.get("mcmc"), dict) and c["mcmc"].get("beta_final") is not None]
    if accs:
        out["posterior_cases"] = {"mean_acceptance_rate": float(np.mean(accs)), "min_acceptance_rate": float(np.min(accs)),
                                  "max_acceptance_rate": float(np.max(accs)), "mean_beta_final": float(np.mean(betas)) if betas else None,
                                  "n_steps": next(iter(cases.values()))["mcmc"].get("n_steps"), "n_chains": next(iter(cases.values()))["mcmc"].get("n_chains")}
    ws = metrics.get("wasserstein") or {}
    w_acc = {s: ws[s].get("mcmc_acceptance") for s in ("train", "test") if isinstance(ws.get(s), dict) and ws[s].get("mcmc_acceptance") is not None}
    if w_acc:
        out["wasserstein_acceptance"] = w_acc
    return out


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def _release_file_handlers(logger: logging.Logger, log_file: Path) -> None:
    """Detach the handler ``get_logger`` attached for this run so repeated calls do not cross-write logs."""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve():
            logger.removeHandler(h)
            h.close()


def evaluate_run(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    profile: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    ckpt: str = "best",
    which: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the evaluation suite on a finished training run and return the ``metrics.json`` contents.

    Args:
        run_dir: output directory of :func:`ndim.train.train` (``config.yaml``, ``ckpt_*.pt``, ``history.json``).
        out_dir: where figures, ``metrics.json`` and ``eval.log`` go (default ``run_dir/eval``).
        profile: hardware profile (``configs/laptop.yaml`` / ``gpu`` / path); only its ``eval`` section and
            ``train.device`` / ``train.num_threads`` are applied.
        overrides: ``key=value`` overrides, same restriction (e.g. ``eval.mcmc.n_steps=500``).
        ckpt: ``"best"`` (best test ELBO) or ``"last"``.
        which: subset of :data:`STAGES` (default all): ``curves``, ``drift``, ``l2``, ``samples``, ``initial``,
            ``posterior``, ``wasserstein``.
        device: torch device string; default ``cfg.train.device``.

    Each stage is timed and isolated: an exception is logged with its traceback and recorded under
    ``metrics["errors"]`` (list of ``{"stage", "error", "traceback"}``) without stopping the other stages.
    ``metrics.json`` is rewritten after every stage so partial results survive an interruption. When
    ``out_dir/metrics.json`` already exists (e.g. from an earlier run with other stages), the results of the
    stages that are *not* re-run are kept and the re-run stages' entries (results, timings, errors) are replaced,
    so ``which=["posterior"]`` with a longer MCMC updates only the posterior part of the report.
    """
    import datetime

    run_dir = Path(run_dir)
    out_dir = ensure_dir(Path(out_dir) if out_dir is not None else run_dir / "eval")
    log_file = out_dir / "eval.log"
    logger = get_logger("ndim.eval", log_file)
    stages = list(STAGES) if which is None else [str(s) for s in which]
    unknown = [s for s in stages if s not in _STAGE_FNS]
    if unknown:
        _release_file_handlers(logger, log_file)
        raise ValueError(f"unknown evaluation stage(s) {unknown}; choose from {list(STAGES)}")
    stages = [s for s in STAGES if s in stages]  # canonical order

    # merge with an earlier report in the same out_dir (keep stages that are not re-run)
    prev: Dict[str, Any] = {}
    if (out_dir / "metrics.json").exists():
        try:
            loaded = load_json(out_dir / "metrics.json")
            prev = loaded if isinstance(loaded, dict) else {}
        except Exception as e:  # noqa: BLE001 - a corrupt previous report is simply replaced
            logger.warning("could not read existing %s (%s); starting a fresh report", out_dir / "metrics.json", e)
    metrics: Dict[str, Any] = dict(prev)
    for s in stages:
        for key in _STAGE_RESULT_KEYS[s]:
            metrics.pop(key, None)
    metrics.update({
        "run_dir": str(run_dir), "out_dir": str(out_dir), "stages_requested": stages,
        "last_run": {"time": datetime.datetime.now().isoformat(timespec="seconds"), "stages": stages, "ckpt": ckpt},
        "stages": {k: v for k, v in (prev.get("stages") or {}).items() if k not in stages},
        "timings": {k: v for k, v in (prev.get("timings") or {}).items() if k not in stages and k != "total"},
        "errors": [e for e in (prev.get("errors") or []) if isinstance(e, dict) and e.get("stage") not in stages],
    })
    try:
        with timer() as t_total:
            cfg = load_eval_config(run_dir, profile=profile, overrides=overrides, logger=logger)
            if cfg.train.num_threads and cfg.train.num_threads > 0:
                torch.set_num_threads(int(cfg.train.num_threads))
            dev = resolve_device(device or cfg.train.device)
            ckpt_path = _checkpoint_path(run_dir, ckpt)
            model, extra = NDIM.load_checkpoint(ckpt_path)
            model = model.to(dev)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            ds, data_path = load_or_generate_dataset(cfg)
            truth_drift = make_truth_drift(cfg.data.truth)
            truth_diffusion = make_truth_diffusion(cfg.data.truth)
            if not ds.has_truth:
                # without a truth only the stages that need none can run
                cfg.eval.posterior_reference = "none"
                skipped = [s for s in stages if s not in TRUTH_FREE_STAGES]
                stages = [s for s in stages if s in TRUTH_FREE_STAGES]
                metrics["stages_skipped_without_truth"] = skipped
                if skipped:
                    logger.info("the observations come from an unspecified model (%s): stages %s compare with a truth and are skipped",
                                data_path, skipped)
            metrics["truth_available"] = bool(ds.has_truth)
            history = _load_history(run_dir, extra)
            logger.info("evaluate %s | ckpt %s (epoch %s, best epoch %s, best test ELBO %s) | device %s | stages %s",
                        run_dir, ckpt_path.name, extra.get("epoch"), extra.get("best_epoch"),
                        f"{extra['best_test_elbo']:.3f}" if isinstance(extra.get("best_test_elbo"), (int, float)) else extra.get("best_test_elbo"),
                        dev, stages)
            logger.info("data %s (train %d, test %d, K_max %d) | N=%d dt=%.4g T=%g | learned scalars %s",
                        data_path, len(ds.train), len(ds.test), ds.train.K_max, cfg.data.n_steps, cfg.dt, cfg.data.T, model.named_scalars())
            metrics.update({
                "config": cfg.to_dict(),
                "checkpoint": {"path": str(ckpt_path), "epoch": extra.get("epoch"), "best_epoch": extra.get("best_epoch"),
                               "best_test_elbo": extra.get("best_test_elbo")},
                "device": str(dev),
                "dataset": {"path": str(data_path), "n_train": len(ds.train), "n_test": len(ds.test), "K_max_train": ds.train.K_max,
                            "K_max_test": ds.test.K_max, "mean_count_train": float(ds.train.counts.float().mean()),
                            "mean_count_test": float(ds.test.counts.float().mean())},
                "learned_scalars": model.named_scalars(),
            })
            _configure_posterior_reference(metrics, cfg, out_dir, has_truth=bool(ds.has_truth))
            ctx = _Ctx(cfg=cfg, model=model, ds=ds, truth_drift=truth_drift, truth_diffusion=truth_diffusion, device=dev, out_dir=out_dir,
                       run_dir=run_dir, logger=logger, history=history, ckpt_extra=dict(extra), metrics=metrics)
            ctx.save()

            for stage in stages:
                logger.info("---- stage %s ----", stage)
                status = "ok"
                with timer() as t:
                    try:
                        with torch.no_grad():
                            out = _STAGE_FNS[stage](ctx)
                        if isinstance(out, dict) and out.get("skipped"):
                            status = "skipped"
                    except Exception as e:  # noqa: BLE001 - a failing stage must not abort the others
                        status = "failed"
                        tb = traceback.format_exc()
                        logger.error("stage %s failed: %s\n%s", stage, e, tb)
                        metrics["errors"].append({"stage": stage, "error": f"{type(e).__name__}: {e}", "traceback": tb})
                metrics["timings"][stage] = t["seconds"]
                metrics["stages"][stage] = {"status": status, "seconds": t["seconds"]}
                logger.info("stage %s: %s in %.1fs", stage, status, t["seconds"])
                ctx.save()
        metrics["timings"]["total"] = t_total["seconds"]
        metrics["mcmc"]["diagnostics"] = _mcmc_summary(metrics)
        save_json(_json_safe(metrics), out_dir / "metrics.json")
        n_err = len(metrics["errors"])
        logger.info("evaluation done in %.1fs -> %s%s", t_total["seconds"], out_dir, f" | {n_err} stage(s) FAILED" if n_err else "")
    finally:
        _release_file_handlers(logger, log_file)
    return _json_safe(metrics)


__all__ = ["evaluate_run", "load_eval_config", "STAGES"]
