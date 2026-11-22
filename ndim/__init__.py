"""ndim — Neural Diffusion Intensity Models for point-process data.

Paper: Du, Honnappa, Rao. "Neural Diffusion Intensity Models for Point Process Data", UAI 2026 (PMLR 337).
See docs/DESIGN.md for the math -> code mapping.
"""
from .config import ExperimentConfig, load_config, config_from_dict, save_config  # noqa: F401
from .data import Dataset, EventData, generate_dataset, load_or_generate_dataset  # noqa: F401
from .model import NDIM, build_model  # noqa: F401
from .links import IntensityLink, build_link, intensity_coefficients  # noqa: F401
from .models.posterior import PosteriorCorrection, register_posterior  # noqa: F401
from .elbo import compute_objective, evaluate_elbo  # noqa: F401
from .train import train, TrainResult  # noqa: F401

__version__ = "0.1.0"
