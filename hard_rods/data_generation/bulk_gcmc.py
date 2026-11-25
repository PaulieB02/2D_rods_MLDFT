"""
GCMC of the homogeneous bulk at every chemical potential of the equation of state (Sec. 4.2).

Writes bulk/eos_gcmc.npy, [2, n_mu]: beta*mu, eta.
"""

import os

import numpy as np

from ..config import EOS_MU_VALUES
from ..utils import packing_fraction
from .gcmc import simulate_runs


def eos_gcmc_path(cfg):
    return cfg.bulk_path("eos_gcmc.npy")


def run_bulk_gcmc_scan(cfg):
    """Simulate a uniform bulk at every mu and save its packing fraction."""
    k = cfg.k
    grid_height = grid_width = cfg.eos_gcmc_grid_size
    n_steps, parallel_simulations = cfg.eos_gcmc_steps, cfg.eos_gcmc_parallel_runs
    warmup_steps = n_steps // 2        # discard the first half of every bulk run

    eta = np.empty(len(EOS_MU_VALUES))
    for i, mu in enumerate(EOS_MU_VALUES):
        u = np.full((2, grid_height, grid_width), -float(mu))          # uniform field u = -beta mu
        runs = simulate_runs(u, grid_height, grid_width, n_steps,
                             parallel_simulations, k, warmup_steps)          # [runs, 2, H, W]
        eta[i] = packing_fraction(runs.mean(axis=0), k)
        print(f"  mu = {mu:+.3f}  ->  eta = {eta[i]:.4f}")

    path = eos_gcmc_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.vstack((EOS_MU_VALUES, eta)))
    print(f"Saved: {path}")
