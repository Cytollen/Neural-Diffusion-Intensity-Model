"""Small utilities: seeding, device selection, logging, timing."""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a torch device.

    ``auto`` prefers CUDA, then CPU. MPS is only used when requested explicitly because for the
    small sequential models used here it is usually slower than the CPU on Apple silicon.
    """
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def get_logger(name: str = "ndim", log_file: Optional[PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # one file handler per logger, so that a later run does not write into an earlier log
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) != log_file.resolve():
                logger.removeHandler(h)
                h.close()
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    """``with timer() as t: ...; t['seconds']``"""
    out: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start


def save_json(obj: Any, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


