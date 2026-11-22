"""Training loop with train/test monitoring, checkpoints and diagnostic plots.

Usage (see ``scripts/train.py``)::

    cfg = load_config("configs/intensity/cir_mlp.yaml", profile="configs/laptop.yaml")
    result = train(cfg)

Outputs in ``cfg.output_dir``: ``config.yaml``, ``train.log``, ``history.csv`` / ``history.json``,
``training_curves.png``, ``drift_snapshot.png`` (+ ``param_history.png``), ``ckpt_last.pt``, ``ckpt_best.pt``.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .cir_eta import resolve_eta
from .config import ExperimentConfig, off_scale_inputs, save_config, typical_input_sizes, z0_mode
from .data import Dataset, load_or_generate_dataset, make_truth_drift, resolve_x_start, resolve_z0
from .elbo import compute_objective, evaluate_elbo
from .links import build_link
from .model import NDIM, build_model
from .plots import plot_drift_snapshot, plot_parameter_history, plot_training_curves
from .sde import brownian_increments
from .utils import count_parameters, ensure_dir, get_logger, resolve_device, save_json, set_seed


@dataclass
class TrainResult:
    model: NDIM
    history: List[Dict[str, float]]
    output_dir: Path
    best_epoch: int
    best_test_elbo: float
    interrupted: bool = False
    broke_down_at: Optional[int] = None       # epoch from which no update had a finite loss (training was stopped)
    dataset: Optional[Dataset] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _make_optimizer(model: NDIM, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    groups = model.param_groups()
    lrs = {"prior": cfg.optim.lr_prior, "posterior": cfg.optim.lr_posterior, "diffusion": cfg.optim.lr_diffusion}
    param_groups = [{"params": ps, "lr": float(lrs[name]), "name": name} for name, ps in groups.items()]
    return torch.optim.Adam(param_groups, betas=tuple(cfg.optim.betas), weight_decay=float(cfg.optim.weight_decay))


def lr_factor(kind: str, epoch: int, epochs: int, floor: float) -> float:
    """Learning rate after ``epoch`` finished epochs, as a fraction of the initial rate.

    Both schedules start at 1 and reach ``floor`` (``optim.lr_min_factor``) after ``epochs`` epochs:
    ``cosine`` along half a cosine, ``exponential`` by a constant factor per epoch.
    """
    x = min(max(int(epoch), 0), int(epochs)) / max(int(epochs), 1)
    if kind == "cosine":
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * x))
    if kind == "exponential":
        return floor ** x
    raise ValueError(f"no learning-rate factor for lr_schedule {kind}")


def _make_scheduler(opt: torch.optim.Optimizer, cfg: ExperimentConfig, start_epoch: int = 0):
    """Scheduler stepped once per epoch. ``start_epoch`` > 0 (a resumed run) continues ``cosine`` / ``exponential``
    from that epoch; ``plateau`` keeps no record of the past and starts again from the configured rates."""
    kind = cfg.optim.lr_schedule
    if kind == "none":
        return None
    if kind in ("cosine", "exponential"):
        floor, epochs = float(cfg.optim.lr_min_factor), int(cfg.train.epochs)
        lam = lambda e: lr_factor(kind, e, epochs, floor)
        if start_epoch > 0:
            for group in opt.param_groups:
                group.setdefault("initial_lr", group["lr"])
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lam] * len(opt.param_groups), last_epoch=int(start_epoch) - 1)
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=float(cfg.optim.plateau_factor), patience=int(cfg.optim.plateau_patience))
    raise ValueError(f"unknown lr_schedule {kind}")


def _clip(model: NDIM, cfg: ExperimentConfig) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    clips = {"prior": cfg.optim.clip_prior, "posterior": cfg.optim.clip_posterior, "diffusion": cfg.optim.clip_diffusion}
    for name, params in model.param_groups().items():
        c = clips.get(name)
        if c is not None and c > 0:
            norms[name] = float(torch.nn.utils.clip_grad_norm_(params, float(c)))
        else:
            tot = 0.0
            for p in params:
                if p.grad is not None:
                    tot += float(p.grad.detach().norm() ** 2)
            norms[name] = math.sqrt(tot)
    return norms


def _grads_finite(model: NDIM) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _write_history(history: List[Dict[str, float]], out_dir: Path) -> None:
    save_json(history, out_dir / "history.json")
    keys: List[str] = []
    for h in history:
        for k in h:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in history:
            w.writerow(h)


def _fixed_eval_sets(ds: Dataset, cfg: ExperimentConfig, rng: np.random.Generator):
    n = min(len(ds.train), len(ds.test))
    idx = rng.choice(len(ds.train), size=n, replace=False)
    return ds.train.subset(sorted(idx.tolist())), ds.test


#: training stops after this many epochs in a row in which no update had a finite loss
DEAD_EPOCHS = 2


def _best_so_far(path: Path, cfg: ExperimentConfig, start_epoch: int) -> Tuple[float, int]:
    """Best test ELBO and its epoch as recorded by ``ckpt_best.pt`` of the run that is being continued, so that a
    later, worse evaluation does not replace that checkpoint. ``(-inf, -1)`` when the folder holds no checkpoint of
    this model from an epoch up to ``start_epoch`` (a run continued in another folder, for instance)."""
    none = (-float("inf"), -1)
    if not Path(path).exists():
        return none
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    elbo, epoch = extra.get("best_test_elbo"), extra.get("epoch")
    from .config import config_from_dict
    try:
        same_model = config_from_dict(payload["config"]).to_dict()["model"] == cfg.to_dict()["model"]
    except Exception:                                    # a checkpoint of something else
        same_model = False
    if not same_model or elbo is None or epoch is None or not math.isfinite(float(elbo)) or int(epoch) > int(start_epoch):
        return none
    return float(elbo), int(epoch)


def train(cfg: ExperimentConfig, dataset: Optional[Dataset] = None, resume_from: Optional[Path] = None,
          init_state: Optional[Dict[str, torch.Tensor]] = None) -> TrainResult:
    """Train the model of ``cfg``. ``resume_from`` continues a run from its checkpoint (epochs, history and schedule
    go on); ``init_state`` starts a new run from the drift and posterior networks of a state dict (the warm start)."""
    out_dir = ensure_dir(cfg.output_dir)
    logger = get_logger("ndim.train", out_dir / "train.log")
    set_seed(int(cfg.train.seed))
    if cfg.train.num_threads and cfg.train.num_threads > 0:
        torch.set_num_threads(int(cfg.train.num_threads))
    device = resolve_device(cfg.train.device)
    if cfg.train.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------------ data
    if dataset is None:
        dataset, data_path = load_or_generate_dataset(cfg)
        logger.info("data: %s (train %d, test %d, K_max %d)", data_path, len(dataset.train), len(dataset.test), dataset.train.K_max)
    # where the paths start: at t = -burn_in from a latent state, or at t = 0 from a fixed intensity
    if cfg.burn_steps > 0:
        from_data = cfg.model.x_start == "auto"
        x_start = resolve_x_start(cfg, dataset.train)
        resolve_z0(cfg, dataset.train, dataset.has_truth)      # not used with a burn-in; kept as a number in the saved config
        logger.info("burn-in: the diffusion of the model starts at t = %g from the latent state %.4g, intensity %.4g%s (%d Euler steps "
                    "before the observation window); the distribution of Z_0 is learned", -cfg.burn_steps * cfg.dt, x_start,
                    float(build_link(cfg.model.link)(torch.tensor(float(x_start)))),
                    f": the mean event rate of the training observations over the first {cfg.model.z0_window:g} of [0, T]" if from_data else "",
                    cfg.burn_steps)
    else:
        estimated = z0_mode(cfg) == "estimate" or (cfg.model.z0 == "auto" and not dataset.has_truth)
        z0 = resolve_z0(cfg, dataset.train, dataset.has_truth)
        logger.info("initial intensity Z_0 = %.4g (%s)", z0,
                    f"mean event rate of the training observations over the first {cfg.model.z0_window:g} of [0, T]" if estimated
                    else "set in the config" if isinstance(cfg.model.z0, (int, float)) else "the truth's")
    # model.diffusion.eta_init: auto
    eta_estimate = resolve_eta(cfg, dataset.train, out_dir)
    if eta_estimate is not None:
        logger.info("diffusion eta * sqrt(Z): the coefficient was estimated from the training observations and is kept fixed: %s",
                    eta_estimate.summary())
    save_config(cfg, out_dir / "config.yaml")
    # warm start, when the untrained model cannot start at its diffusion coefficient
    warm = None
    if resume_from is None and init_state is None and cfg.train.warm_start == "auto":
        from .warm_start import warm_start

        warm = warm_start(cfg, dataset, out_dir, logger, device)
        set_seed(int(cfg.train.seed))  # the run itself starts from its seed
        if warm is not None:
            init_state = warm.pop("state")
            logger.info("warm start: reached the diffusion coefficient %g in %d stages (%d epochs); the %d epochs of the run start "
                        "from these networks", warm["target"], len(warm["stages"]), warm["epochs"], cfg.train.epochs)
    if warm is None and resume_from is None:
        (out_dir / "warm_start.json").unlink(missing_ok=True)       # of an earlier run in this folder
    train_events = dataset.train.events.to(device)
    n_train = train_events.shape[0]
    rng = np.random.default_rng(int(cfg.train.seed))
    eval_train_set, eval_test_set = _fixed_eval_sets(dataset, cfg, rng)
    eval_gen = torch.Generator().manual_seed(int(cfg.train.seed) + 12345)

    # ------------------------------------------------------------------ model
    model = build_model(cfg).to(device)
    start_epoch = 0
    history: List[Dict[str, float]] = []
    if resume_from is not None:
        payload = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        from .config import config_from_dict
        saved_cfg = config_from_dict(payload["config"])
        if saved_cfg.to_dict()["model"] != cfg.to_dict()["model"]:
            raise ValueError("Resume model configuration differs from the checkpoint (including state space/link)")
        model.load_state_dict(payload["state_dict"])
        start_epoch = int(payload.get("extra", {}).get("epoch", 0))
        history = list(payload.get("extra", {}).get("history", []))
        logger.info("resumed from %s at epoch %d", resume_from, start_epoch)
    elif init_state is not None:
        from .warm_start import load_networks

        load_networks(model, init_state)
    opt = _make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg, start_epoch)
    logger.info("model: drift=%s (%d params) | posterior=%s (%d params) | eta trainable=%s | device=%s",
                cfg.model.drift.kind, count_parameters(model.drift), cfg.model.posterior.kind,
                count_parameters(model.score_net), cfg.model.diffusion.trainable, device)
    logger.info("train: epochs=%d batch=%d m=%d grad_mode=%s | every mini-batch at the horizons %s: %d updates per epoch",
                cfg.train.epochs, cfg.train.batch_size, cfg.train.m_paths, cfg.train.grad_mode, cfg.train.horizons,
                -(-n_train // int(cfg.train.batch_size)) * len(cfg.train.horizons))
    # check the input scales against the observations
    sizes = typical_input_sizes(cfg, float(dataset.train.counts.float().mean()))
    logger.info("network inputs at the mean event rate of the training observations: %s",
                ", ".join(f"{name} = {size:.2f}" for name, size in sizes.items()))
    for name in off_scale_inputs(cfg, sizes):
        logger.warning("%s is %.3g for these observations, far from one: set model.intensity_scale to their typical "
                       "intensity and model.posterior.count_scale to their typical number of events", name, sizes[name])
    if cfg.train.grad_mode == "sensitivity":
        from .gradients import sensitivity_backward  # noqa: F401  (import check)

    truth_drift = make_truth_drift(cfg.data.truth) if dataset.has_truth else None     # no truth line for an unspecified model
    n_full = cfg.horizon_steps(1.0)
    B = int(cfg.train.batch_size)
    m = int(cfg.train.m_paths)
    horizon_steps = [cfg.horizon_steps(h) for h in cfg.train.horizons]
    best_test_elbo, best_epoch = -float("inf"), -1
    if resume_from is not None:                              # keep the best state of the run that is continued
        best_test_elbo, best_epoch = _best_so_far(out_dir / "ckpt_best.pt", cfg, start_epoch)
        if best_epoch >= 0:
            logger.info("best test ELBO so far: %.3f at epoch %d (ckpt_best.pt is replaced only by a better one)", best_test_elbo, best_epoch)
    bad_epochs = 0
    interrupted = False
    dead_epochs, broke_down_at = 0, None        # epochs in a row without a single finite update
    t_start = time.time()

    def do_eval(epoch: int, row: Dict[str, float]) -> None:
        for frac in cfg.train.eval_horizon_fracs:
            n_p = cfg.horizon_steps(frac)
            suffix = "" if abs(frac - 1.0) < 1e-9 else f"_h{frac:g}"
            g = torch.Generator().manual_seed(int(eval_gen.initial_seed()))
            te = evaluate_elbo(model, eval_test_set.events, n_p, int(cfg.train.eval_paths), cfg, device, g)
            g = torch.Generator().manual_seed(int(eval_gen.initial_seed()))
            tr = evaluate_elbo(model, eval_train_set.events, n_p, int(cfg.train.eval_paths), cfg, device, g)
            for k, v in te.items():
                row[f"test_{k}{suffix}"] = v
            for k, v in tr.items():
                row[f"eval_train_{k}{suffix}"] = v
        model.train()

    def do_plots() -> None:
        plot_training_curves(history, out_dir / "training_curves.png", title=cfg.name)
        model.eval()
        plot_drift_snapshot(
            model.intensity_drift, truth_drift, cfg.eval.drift_z_max, cfg.data.T, times=(0.0, 0.5, 1.0),
            occupancy=dataset.train_paths, path=out_dir / "drift_snapshot.png",
            z_min=1e-3 if model.real_states else 0.0,
            title=f"{cfg.name}: intensity drift{' vs truth' if dataset.has_truth else ''} (epoch {history[-1]['epoch'] if history else 0})", device=device,
        )
        model.train()
        keys = [k for k in ("drift.kappa", "drift.theta", "diffusion.eta") if any(k in h for h in history)]
        if keys:
            plot_parameter_history(history, keys, truth=model.comparable_truth_scalars(), path=out_dir / "param_history.png", title=cfg.name)

    def save_ckpt(name: str, epoch: int) -> None:
        model.save_checkpoint(out_dir / name, extra={"epoch": epoch, "history": history, "best_epoch": best_epoch, "best_test_elbo": best_test_elbo})

    # ------------------------------------------------------------------ loop
    try:
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            model.train()
            t0 = time.time()
            kl_weight = float(cfg.train.kl_weight)
            if cfg.train.kl_warmup_epochs > 0 and epoch < cfg.train.kl_warmup_epochs:
                kl_weight = kl_weight * float(epoch + 1) / float(cfg.train.kl_warmup_epochs)
            perm = torch.randperm(n_train)
            agg_full = {"loglik": 0.0, "kl": 0.0, "elbo": 0.0, "n": 0}
            agg_all = {"loglik": 0.0, "kl": 0.0, "elbo": 0.0, "n": 0}
            grad_norms_acc: Dict[str, float] = {}
            n_updates, n_skipped = 0, 0
            for bi, start in enumerate(range(0, n_train, B)):
                idx = perm[start : start + B]
                events = train_events.index_select(0, idx.to(device))
                Bc = events.shape[0]
                # every mini-batch is trained at every horizon: observations cut at T', new Brownian paths, one optimizer step
                for n_p in horizon_steps:
                    W = brownian_increments(Bc * m, model.n_burn + n_p, cfg.dt, device)

                    opt.zero_grad(set_to_none=True)
                    if cfg.train.grad_mode == "backprop":
                        out = compute_objective(model, events, n_p, m, W, cfg, kl_weight=kl_weight)
                        if not torch.isfinite(out.loss):
                            n_skipped += 1
                            if cfg.train.skip_nonfinite:
                                continue
                            raise FloatingPointError("non-finite loss")
                        out.loss.backward()
                    else:
                        from .gradients import sensitivity_backward
                        out = sensitivity_backward(model, events, n_p, m, W, cfg, kl_weight=kl_weight)
                    if not _grads_finite(model):
                        n_skipped += 1
                        if cfg.train.skip_nonfinite:
                            opt.zero_grad(set_to_none=True)
                            continue
                        raise FloatingPointError("non-finite gradient")
                    norms = _clip(model, cfg)
                    opt.step()
                    n_updates += 1
                    for k, v in norms.items():
                        grad_norms_acc[k] = grad_norms_acc.get(k, 0.0) + v
                    s = out.summary()
                    tgt = agg_full if n_p == n_full else None
                    for agg in ([agg_all] + ([tgt] if tgt is not None else [])):
                        agg["loglik"] += s["loglik"] * Bc
                        agg["kl"] += s["kl"] * Bc
                        agg["elbo"] += s["elbo"] * Bc
                        agg["n"] += Bc
                    if cfg.train.log_every_batches and (bi + 1) % cfg.train.log_every_batches == 0:
                        logger.info("  epoch %d batch %d | T'=%.2f | elbo %.2f loglik %.2f kl %.3f", epoch + 1, bi + 1, n_p * cfg.dt, s["elbo"], s["loglik"], s["kl"])

            row: Dict[str, float] = {"epoch": epoch + 1, "kl_weight": kl_weight, "time_s": time.time() - t0,
                                     "n_updates": n_updates, "n_skipped": n_skipped}
            for k in ("loglik", "kl", "elbo"):
                row[f"train_{k}"] = agg_full[k] / max(agg_full["n"], 1) if agg_full["n"] else float("nan")
                row[f"train_{k}_allhorizons"] = agg_all[k] / max(agg_all["n"], 1)
            for k, v in grad_norms_acc.items():
                row[f"gradnorm_{k}"] = v / max(n_updates, 1)
            for g in opt.param_groups:
                row[f"lr_{g['name']}"] = float(g["lr"])
            row.update(model.named_scalars())

            is_eval = ((epoch + 1) % max(cfg.train.eval_every, 1) == 0) or (epoch + 1 == cfg.train.epochs)
            if is_eval:
                do_eval(epoch + 1, row)
                if row.get("test_elbo", -float("inf")) > best_test_elbo:
                    best_test_elbo, best_epoch, bad_epochs = float(row["test_elbo"]), epoch + 1, 0
                    save_ckpt("ckpt_best.pt", epoch + 1)
                else:
                    bad_epochs += 1
                if sched is not None and isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(row["test_elbo"])
            if sched is not None and not isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step()
            history.append(row)

            msg = (f"epoch {epoch + 1:4d}/{cfg.train.epochs} | train elbo {row['train_elbo']:9.3f} loglik {row['train_loglik']:9.3f} kl {row['train_kl']:7.3f}")
            if is_eval:
                msg += f" | test elbo {row['test_elbo']:9.3f} loglik {row['test_loglik']:9.3f} kl {row['test_kl']:7.3f} | train-subset elbo {row['eval_train_elbo']:9.3f}"
            scal = model.named_scalars()
            if scal:
                msg += " | " + " ".join(f"{k.split('.')[-1]}={v:.4g}" for k, v in scal.items())
            msg += f" | {row['time_s']:.1f}s"
            if n_skipped:
                msg += f" | skipped {n_skipped} of {n_updates + n_skipped} updates (non-finite)"
            logger.info(msg)

            _write_history(history, out_dir)
            if (epoch + 1) % max(cfg.train.plot_every, 1) == 0 or epoch + 1 == cfg.train.epochs:
                do_plots()
            if (epoch + 1) % max(cfg.train.checkpoint_every, 1) == 0 or epoch + 1 == cfg.train.epochs:
                save_ckpt("ckpt_last.pt", epoch + 1)
            if cfg.train.early_stopping_patience and bad_epochs >= cfg.train.early_stopping_patience:
                logger.info("early stopping: no test-ELBO improvement for %d evaluations (best epoch %d)", bad_epochs, best_epoch)
                break
            # no finite loss in a whole epoch: no update can bring the model back, so stop
            dead_epochs = dead_epochs + 1 if (n_updates == 0 and n_skipped > 0) else 0
            if dead_epochs >= DEAD_EPOCHS:
                broke_down_at = epoch + 2 - dead_epochs
                logger.error("training has broken down: no update since epoch %d had a finite loss, because the paths of the model run "
                             "off. Stopped at epoch %d of %d. %s Two causes are known. In the first epochs: the networks have not "
                             "yet learned to hold the paths where the intensity is low. The bounds of the state-space "
                             "parametrization prevent this (train.max_abs_control, model.diffusion.max_step_std): set them if the "
                             "config does not, or fit again with another seed (train.seed). "
                             "Later: a learned drift that is too stiff for the Euler step where the intensity is low; fit again "
                             "with a finer grid (data.n_steps; --steps of scripts/fit.py).",
                             broke_down_at, epoch + 1, cfg.train.epochs,
                             f"ckpt_best.pt holds the state of epoch {best_epoch}." if best_epoch >= 0 else "No state was evaluated before.")
                break
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("training interrupted by user; saving current state")

    save_ckpt("ckpt_last.pt", len(history))
    if best_epoch < 0:  # never evaluated
        best_epoch = len(history)
        save_ckpt("ckpt_best.pt", best_epoch)
    _write_history(history, out_dir)
    if history:
        do_plots()
    total = time.time() - t_start
    n_run = len(history) - start_epoch
    spent = f"({n_run} of them in this run, in {total / 60:.1f} min)" if start_epoch else f"in {total / 60:.1f} min"
    logger.info("done: %d epochs %s | best test ELBO %.3f at epoch %d | outputs in %s", len(history), spent, best_test_elbo, best_epoch, out_dir)
    summary = {"epochs": len(history), "train_time_s": total, "best_epoch": best_epoch, "best_test_elbo": best_test_elbo, "interrupted": interrupted}
    if start_epoch:
        summary["epochs_this_run"] = n_run                  # train_time_s is the time of these
    if broke_down_at is not None:
        summary["broke_down_at"] = broke_down_at
    if warm is not None:
        summary["warm_start_epochs"] = warm["epochs"]
    save_json(summary, out_dir / "train_summary.json")
    extras: Dict[str, Any] = {}
    if warm is not None:
        extras["warm_start"] = warm
    if eta_estimate is not None:
        extras["eta_estimate"] = eta_estimate
    return TrainResult(model=model, history=history, output_dir=out_dir, best_epoch=best_epoch, best_test_elbo=best_test_elbo,
                       interrupted=interrupted, broke_down_at=broke_down_at, dataset=dataset, extras=extras)
