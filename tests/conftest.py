"""Shared pytest fixtures for the ``ndim`` test-suite.

* Inserts the repository root into ``sys.path`` so ``import ndim`` works no matter where pytest is started from.
* Every configuration used by the tests is derived from ``configs/smoke.yaml`` through
  ``load_config("smoke", overrides=[...])`` with ``output_root`` and ``data.cache_dir`` redirected to a
  session-scoped temporary directory, so the suite never writes into the repository.
* Session-scoped fixtures cache the (tiny) smoke dataset and a single 2-epoch smoke training run that
  ``test_train_smoke.py`` and ``test_evaluate_smoke.py`` share.

The whole suite is meant to run in well under three minutes on an 8 GB laptop CPU.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  (after sys.path manipulation)

from ndim.config import ExperimentConfig, load_config  # noqa: E402
from ndim.data import Dataset, load_or_generate_dataset  # noqa: E402
from ndim.model import NDIM, build_model  # noqa: E402

torch.set_default_dtype(torch.float32)

CPU = torch.device("cpu")


def yaml_quote(value: object) -> str:
    """Render ``value`` as a YAML scalar that ``parse_overrides`` (``yaml.safe_load``) parses back verbatim.

    Paths are JSON-quoted so characters such as ``:`` or spaces cannot be mis-parsed.
    """
    return json.dumps(str(value))


# --------------------------------------------------------------------------------------
# Session-scoped scratch space and base overrides
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One temporary directory for the whole session (outputs + dataset cache)."""
    return tmp_path_factory.mktemp("ndim_tests")


@pytest.fixture(scope="session")
def base_overrides(tmp_root: Path) -> List[str]:
    """Overrides that redirect all file output of the smoke config into ``tmp_root``."""
    return [
        f"output_root={yaml_quote(tmp_root / 'outputs')}",
        f"data.cache_dir={yaml_quote(tmp_root / 'data')}",
    ]


def _load_smoke(base: Sequence[str], overrides: Iterable[str] = (), profile: Optional[str] = None) -> ExperimentConfig:
    return load_config("smoke", profile=profile, overrides=list(base) + list(overrides))


@pytest.fixture(scope="session")
def smoke_cfg(base_overrides: List[str]) -> ExperimentConfig:
    """``configs/smoke.yaml`` with a temporary ``output_root`` / ``data.cache_dir``.

    Session-scoped: tests must not mutate it (use ``make_cfg`` for variants).
    """
    return _load_smoke(base_overrides)


@pytest.fixture
def make_cfg(base_overrides: List[str]) -> Callable[..., ExperimentConfig]:
    """Factory: ``make_cfg("model.drift.kind=mlp", "train.epochs=1", profile=None)`` -> fresh smoke config."""

    def _make(*overrides: str, profile: Optional[str] = None) -> ExperimentConfig:
        return _load_smoke(base_overrides, overrides, profile=profile)

    return _make


@pytest.fixture
def cfg_copy(smoke_cfg: ExperimentConfig) -> ExperimentConfig:
    """A deep copy of the smoke config that a test may mutate freely."""
    return copy.deepcopy(smoke_cfg)


# --------------------------------------------------------------------------------------
# Data / model / randomness
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def smoke_dataset(smoke_cfg: ExperimentConfig) -> Dataset:
    """The cached smoke dataset (16 train / 8 test observations, 25 Euler steps on [0, 4])."""
    ds, _ = load_or_generate_dataset(smoke_cfg)
    return ds


@pytest.fixture
def tiny_model(smoke_cfg: ExperimentConfig) -> NDIM:
    """Freshly initialised smoke model (latent state, tiny MLP drift and Deep Sets posterior), deterministic init."""
    torch.manual_seed(0)
    return build_model(smoke_cfg)


@pytest.fixture
def gen() -> torch.Generator:
    """A CPU ``torch.Generator`` with a fixed seed."""
    return torch.Generator().manual_seed(0)


@pytest.fixture
def device() -> torch.device:
    return CPU


# --------------------------------------------------------------------------------------
# One shared smoke training run (2 epochs, ~2 s) for the train / evaluate smoke tests
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def trained_smoke_run(base_overrides: List[str]):
    """Train the smoke config once per session; returns ``ndim.train.TrainResult``."""
    from ndim.train import train

    cfg = _load_smoke(base_overrides, ["name=smoke_pytest_run"])
    return train(cfg)
