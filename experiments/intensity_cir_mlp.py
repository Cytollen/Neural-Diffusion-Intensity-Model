#!/usr/bin/env python3
"""Example 1: a known initial intensity, the model as the paper writes it (Sec. 4.1).

Data: dZ = 0.3(80 - Z) dt + sqrt(Z) dB with Z_0 = 5 on [0, 4]; 256 training and 128 test observations.
Model: the state of the SDE is the intensity Z (the intensity parametrization), every path starts at the known Z_0,
the drift b_theta(z, t) is an MLP, and the diffusion sqrt(Z) is known and not trained.

Stages
  data    generate (or reuse) the synthetic dataset
  train   train, then evaluate: learned against true drift (Fig. 2), generated against test counts (Fig. 3), and
          posterior paths against the exact CIR posterior, with the intensity path that generated the observation
          (Fig. 4)

Results go to outputs/intensity_cir_mlp/ (figures and metrics.json in its eval/ folder).

Examples
  python experiments/intensity_cir_mlp.py
  python experiments/intensity_cir_mlp.py --profile laptop
  python experiments/intensity_cir_mlp.py --smoke
"""
from __future__ import annotations

from typing import List

from _runner import Experiment, Stage, run

CONFIG = "configs/intensity/cir_mlp.yaml"


def stages(experiment: Experiment) -> List[Stage]:
    return [
        experiment.data(CONFIG),
        experiment.train(CONFIG, title="train, then evaluate (Fig. 2 drift, Fig. 3 generated counts, Fig. 4 posterior paths)"),
    ]


if __name__ == "__main__":
    raise SystemExit(run(__doc__, stages))
