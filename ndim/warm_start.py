"""Warm start for a fixed diffusion coefficient that an untrained model cannot start with (``train.warm_start: auto``).

Written for the latent state ``X = log Z``, the CIR diffusion ``eta * exp(-X/2)`` grows without bound as the
intensity falls. With a coefficient of realistic size and an untrained drift, a share of the paths runs off within
the window, the loss of every mini-batch is non-finite and training cannot begin. A trained drift keeps the paths in
range, so the model is trained at a smaller coefficient first:

1. the untrained model is tried at the coefficient of the config; if most mini-batches give a finite loss, nothing
   is done;
2. otherwise the coefficient is lowered by the factor 1.4 until the untrained model can start; the model is trained
   there for a few epochs (the drift learns the mean curve of the observations);
3. the coefficient is raised in steps: at every step the largest of the factors 1.4, 1.2, 1.1 is taken for which the
   current networks give finite losses on most mini-batches, and the networks are trained for a few epochs at the
   new value;
4. at the coefficient of the config the networks are handed to the run proper, which trains its epochs from them
   with its own learning-rate schedule.

The stages are ordinary training runs with a constant learning rate, written below ``<run>/warm_start/``; their
list is saved as ``<run>/warm_start.json``.
"""
from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import ExperimentConfig
from .data import Dataset
from .elbo import evaluate_elbo
from .model import NDIM, build_model
from .utils import get_logger, save_json, set_seed

START_EPOCHS = 8                  # epochs at the first coefficient
STEP_EPOCHS = 3                   # epochs after every raise
FACTORS = (1.4, 1.2, 1.1)         # candidate raises, the largest first
MIN_FINITE = 0.75                 # share of the probe mini-batches with a finite loss at which training can proceed
N_PROBE = 16                      # probe mini-batches
MAX_STAYS = 3                     # times in a row that the coefficient may stay where it is
MAX_LOWERINGS = 12                # the start is searched down to target / 1.4^12


class WarmStartError(RuntimeError):
    """The warm start did not reach the configured diffusion coefficient."""

def finite_share(model: NDIM, dataset: Dataset, cfg: ExperimentConfig, device: torch.device) -> float:
    """Share of ``N_PROBE`` mini-batches of training observations on which ``model`` gives a finite ELBO. The
    mini-batches and the Brownian paths are the same at every call; the global random state is not used."""
    train = dataset.train
    n, batch = len(train), min(int(cfg.train.batch_size), len(train))
    floor = -1e4 * max(float(train.counts.double().mean()), 1.0)          # far below any ELBO of a model that works
    rng = np.random.default_rng(12345)
    was_training = model.training
    good = 0
    with torch.no_grad():
        for b in range(N_PROBE):
            idx = torch.as_tensor(np.sort(rng.choice(n, size=batch, replace=False)))
            out = evaluate_elbo(model, train.events.index_select(0, idx), cfg.horizon_steps(1.0), int(cfg.train.m_paths), cfg, device,
                                torch.Generator().manual_seed(1000 + b))
            value = float(out["elbo"])
            good += int(math.isfinite(value) and value > floor)
    model.train(was_training)
    return good / N_PROBE


def load_networks(model: NDIM, state: Dict[str, torch.Tensor]) -> None:
    """Load the drift and posterior networks of ``state`` into ``model``; the diffusion of ``model`` is kept."""
    networks = {k: v for k, v in state.items() if not k.startswith("diffusion.")}
    missing, unexpected = model.load_state_dict(networks, strict=False)
    if unexpected or any(not k.startswith("diffusion.") for k in missing):
        raise RuntimeError(f"the networks to start from do not fit the model: missing {list(missing)}, unexpected {list(unexpected)}")


def _model_at(cfg: ExperimentConfig, eta: float, device: torch.device, state: Optional[Dict[str, torch.Tensor]] = None) -> NDIM:
    """The model of ``cfg`` with the diffusion coefficient ``eta``: untrained as training would build it, or with the
    networks of ``state``."""
    c = copy.deepcopy(cfg)
    c.model.diffusion.eta_init = float(eta)
    set_seed(int(cfg.train.seed))
    model = build_model(c).to(device)
    if state is not None:
        load_networks(model, state)
    return model


def warm_start(cfg: ExperimentConfig, dataset: Dataset, out_dir: Path, logger: logging.Logger, device: torch.device) -> Optional[Dict[str, Any]]:
    """Networks trained up to the diffusion coefficient of ``cfg`` (``{"state": ..., "target": ..., "stages": [...]}``),
    or ``None`` when the untrained model can start at that coefficient. ``cfg`` must be resolved (``x_start``,
    ``eta_init`` numbers). Raises ``WarmStartError`` when no coefficient lets the model start or the target is not reached."""
    diffusion = cfg.model.diffusion
    target = float(diffusion.eta_init)
    share = finite_share(_model_at(cfg, target, device), dataset, cfg, device)
    if share >= MIN_FINITE:
        logger.info("warm start: not needed (the untrained model gives a finite loss on %.0f%% of %d probe mini-batches)", 100 * share, N_PROBE)
        return None
    if diffusion.trainable or diffusion.kind == "mlp":
        logger.warning("the untrained model gives a finite loss on %.0f%% of the probe mini-batches only; the warm start is made for a "
                       "diffusion coefficient that is not trained, so training starts as it is", 100 * share)
        return None

    eta, tried = target, [f"{target:.4g}: {share:.0%}"]
    for _ in range(MAX_LOWERINGS):
        eta = eta / FACTORS[0]
        share = finite_share(_model_at(cfg, eta, device), dataset, cfg, device)
        tried.append(f"{eta:.4g}: {share:.0%}")
        if share >= MIN_FINITE:
            break
    else:
        raise WarmStartError(f"the untrained model gives no finite loss even with the diffusion coefficient lowered from {target:g} to "
                           f"{eta:.3g}: check the event times and T, or use more Euler steps (data.n_steps)")
    logger.info("warm start: the untrained model cannot start at the diffusion coefficient %g (share of probe mini-batches with a finite "
                "loss by coefficient: %s). Training starts at %.4g and the coefficient is raised in steps; stages in %s",
                target, ", ".join(tried), eta, Path(out_dir) / "warm_start")

    from .train import train                    # imported here: ndim.train imports this module

    stages: List[Dict[str, Any]] = []
    state: Optional[Dict[str, torch.Tensor]] = None

    def run_stage(value: float, epochs: int) -> None:
        nonlocal state
        c = copy.deepcopy(cfg)
        c.model.diffusion.eta_init = float(value)
        c.train.epochs = int(epochs)
        c.train.warm_start = "off"
        c.train.eval_every = c.train.plot_every = c.train.checkpoint_every = int(epochs)
        c.train.early_stopping_patience = 0
        c.optim.lr_schedule = "none"
        c.output_root = str(Path(out_dir) / "warm_start")
        c.name = f"stage_{len(stages):02d}_eta_{value:.4g}"
        result = train(c, dataset=dataset, init_state=state)
        get_logger(logger.name, Path(out_dir) / "train.log")                 # the stage logged to its own file
        if result.broke_down_at is not None:
            raise WarmStartError(f"warm start: training broke down at the diffusion coefficient {value:.4g} (configured: {target:g}): "
                                 f"no update had a finite loss any more, because the paths of the model ran off. The stage is in "
                                 f"{result.output_dir}. Fit again with another seed (train.seed) or with more Euler steps "
                                 "(data.n_steps; --steps of scripts/fit.py)")
        state = {k: v.detach().cpu().clone() for k, v in result.model.state_dict().items()}
        last = result.history[-1] if result.history else {}
        stages.append({"eta": float(value), "epochs": len(result.history),
                       "updates": sum(int(h.get("n_updates", 0)) for h in result.history),
                       "skipped": sum(int(h.get("n_skipped", 0)) for h in result.history),
                       "train_elbo": float(last.get("train_elbo", float("nan"))), "test_elbo": float(last.get("test_elbo", float("nan")))})
        logger.info("warm start: stage %d at the coefficient %.4g: %d epochs, %d of %d updates skipped as non-finite, train ELBO %.2f",
                    len(stages), value, stages[-1]["epochs"], stages[-1]["skipped"], stages[-1]["updates"] + stages[-1]["skipped"],
                    stages[-1]["train_elbo"])

    run_stage(eta, START_EPOCHS)
    stays = 0
    while eta < target * (1.0 - 1e-9):
        chosen = None
        for factor in FACTORS:
            candidate = min(target, eta * factor)
            if finite_share(_model_at(cfg, candidate, device, state), dataset, cfg, device) >= MIN_FINITE:
                chosen = candidate
                break
        if chosen is None:
            stays += 1
            if stays > MAX_STAYS:
                raise WarmStartError(f"warm start: the diffusion coefficient could not be raised above {eta:.4g} (configured: {target:g}); "
                                   f"the stages are in {Path(out_dir) / 'warm_start'}. A smaller coefficient, more Euler steps "
                                   "(data.n_steps) or a diffusion that is trained are the alternatives")
            chosen = eta
        else:
            stays = 0
        eta = chosen
        run_stage(eta, STEP_EPOCHS)

    info = {"target": target, "stages": stages, "epochs": sum(s["epochs"] for s in stages)}
    save_json(info, Path(out_dir) / "warm_start.json")
    return {"state": state, **info}
