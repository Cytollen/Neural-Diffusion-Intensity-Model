#!/usr/bin/env python3
"""Example 2: an unknown initial intensity, learned through a burn-in (state-space parametrization).

Data: Z_0 normal with mean 40 and standard deviation 10 truncated to [5, 75], then dZ = 0.3(80 - Z) dt + sqrt(Z) dB
on [0, 4]; 256 training and 128 test observations.
Model: the SDE is written for X = log Z. Its diffusion starts at t = -1, before the observation window, from the
latent state 0 (intensity 1), and nothing is observed until t = 0. The distribution of the initial intensity Z_0 is
whatever the learned dynamics produce by then, and the posterior correction acts on the burn-in too, so the posterior
of Z_0 depends on the observation. The drift b_theta(x, t) is an MLP; the diffusion sqrt(Z), exp(-X/2) for X, is
known and not trained.

Stages
  data    generate (or reuse) the synthetic dataset
  train   train, then evaluate: true against learned distribution of Z_0 (initial_distribution.png), learned against
          true drift, generated against test counts, and posterior paths against pCN MCMC under the true prior, with
          the intensity path that generated the observation

Results go to outputs/state_space_cir_burn_in/ (figures and metrics.json in its eval/ folder).

Examples
  python experiments/state_space_cir_burn_in.py
  python experiments/state_space_cir_burn_in.py --profile laptop
  python experiments/state_space_cir_burn_in.py --smoke
"""
from __future__ import annotations

from typing import List

from _runner import Experiment, Stage, run

CONFIG = "configs/state_space/cir_burn_in.yaml"


def stages(experiment: Experiment) -> List[Stage]:
    return [
        experiment.data(CONFIG),
        experiment.train(CONFIG, title="train, then evaluate (initial distribution, drift, generated counts, posterior paths)"),
    ]


if __name__ == "__main__":
    raise SystemExit(run(__doc__, stages))
