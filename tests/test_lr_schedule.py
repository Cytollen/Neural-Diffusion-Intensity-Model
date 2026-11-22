"""Learning-rate schedules (``optim.lr_schedule``): the cosine and exponential factors, their use by the training
loop, and a resumed run continuing the schedule."""
from __future__ import annotations

import math

import pytest
import torch

from ndim.config import config_from_dict
from ndim.train import _make_scheduler, lr_factor, train


@pytest.mark.parametrize("kind", ["cosine", "exponential"])
def test_factor_starts_at_one_and_ends_at_the_floor(kind: str) -> None:
    epochs, floor = 100, 0.1
    factors = [lr_factor(kind, e, epochs, floor) for e in range(epochs + 1)]
    assert factors[0] == pytest.approx(1.0) and factors[-1] == pytest.approx(floor)
    assert all(a > b for a, b in zip(factors, factors[1:]))                      # decreasing all the way
    assert lr_factor(kind, epochs + 7, epochs, floor) == pytest.approx(floor)    # and constant afterwards
    if kind == "exponential":
        ratios = [b / a for a, b in zip(factors, factors[1:])]
        assert max(ratios) == pytest.approx(min(ratios)) and ratios[0] == pytest.approx(floor ** (1 / epochs))   # one factor per epoch
        assert factors[50] == pytest.approx(math.sqrt(floor))
    else:
        assert factors[50] == pytest.approx((1 + floor) / 2)                     # half-way between the two ends at mid-run


def test_cosine_keeps_more_of_the_step_budget_than_exponential() -> None:
    """Sum of the per-epoch rates relative to a constant rate, for the same end point of 10%."""
    epochs, floor = 100, 0.1
    budget = {kind: sum(lr_factor(kind, e, epochs, floor) for e in range(epochs)) / epochs for kind in ("cosine", "exponential")}
    assert budget["cosine"] == pytest.approx(0.55, abs=0.01) and budget["exponential"] == pytest.approx(0.395, abs=0.01)


def test_schedule_validation() -> None:
    for kind in ("none", "cosine", "exponential", "plateau"):
        assert config_from_dict({"optim": {"lr_schedule": kind}}).optim.lr_schedule == kind
    for patch in ({"lr_schedule": "linear"}, {"lr_schedule": "exponential", "lr_min_factor": 0.0}, {"lr_schedule": "cosine", "lr_min_factor": 1.5}):
        with pytest.raises(ValueError):
            config_from_dict({"optim": patch})
    assert config_from_dict({"optim": {"lr_schedule": "cosine", "lr_min_factor": 0.0}}).optim.lr_min_factor == 0.0


def test_cosine_is_the_default_schedule() -> None:
    from ndim.config import load_config

    assert config_from_dict({}).optim.lr_schedule == "cosine" and config_from_dict({}).optim.lr_min_factor == 0.1
    for name in ("base", "intensity/cir_mlp", "state_space/cir_burn_in"):
        assert load_config(name).optim.lr_schedule == "cosine", name


def test_scheduler_resumes_at_the_given_epoch(make_cfg) -> None:
    cfg = make_cfg("optim.lr_schedule=exponential", "train.epochs=10")
    for start in (0, 4):
        opt = torch.optim.Adam([{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 5e-3}])
        sched = _make_scheduler(opt, cfg, start)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-3 * lr_factor("exponential", start, 10, 0.1))
        opt.step()
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-3 * lr_factor("exponential", start + 1, 10, 0.1))
    assert _make_scheduler(torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]), make_cfg("optim.lr_schedule=none")) is None


@pytest.mark.parametrize("kind", ["cosine", "exponential"])
def test_training_applies_the_schedule_and_a_resumed_run_continues_it(make_cfg, kind: str) -> None:
    name = f"name=smoke_pytest_schedule_{kind}"
    cfg = make_cfg(name, f"optim.lr_schedule={kind}", "train.epochs=3", "train.eval_paths=1")
    res = train(cfg)
    for group, base in (("prior", cfg.optim.lr_prior), ("posterior", cfg.optim.lr_posterior)):
        used = [h[f"lr_{group}"] for h in res.history]                           # the rate each epoch was trained at
        assert used == pytest.approx([base * lr_factor(kind, e, 3, 0.1) for e in range(3)])
    # resumed with a longer horizon, epochs 4 and 5 continue the schedule of the 5-epoch run instead of restarting it
    longer = make_cfg(name, f"optim.lr_schedule={kind}", "train.epochs=5", "train.eval_paths=1")
    resumed = train(longer, resume_from=res.output_dir / "ckpt_last.pt")
    assert [h["epoch"] for h in resumed.history] == [1, 2, 3, 4, 5]
    assert [h["lr_prior"] for h in resumed.history[3:]] == pytest.approx([cfg.optim.lr_prior * lr_factor(kind, e, 5, 0.1) for e in (3, 4)])
