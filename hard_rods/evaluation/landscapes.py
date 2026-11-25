"""
Bulk free-energy landscapes beta f(S) at fixed eta (Sec. 4.3). Writes
bulk/free_energy_landscapes.npz: "eta" and "curves" [n_eta, 4, n_S] (the S grid, then the
Phi-network and the direct-c1 network along the simultaneous and the sequential path).
"""

import os

import numpy as np

from ..functionals.neural import load_network
from . import bulk_free_energy as bulk

eta_values = [0.3, 0.7, 0.8]
S_min, S_max, num_points = -1.0, 1.0, 150


def landscapes_path(cfg):
    return cfg.bulk_path("free_energy_landscapes.npz")


def landscapes(net_phi, net_c1, eta, k, offset):
    """The S grid and the three beta f(S) curves at fixed eta."""
    S = np.linspace(S_min, S_max, num_points)
    return (S,
            bulk.free_energy_curve_phi(net_phi, eta, S, k) - offset,
            bulk.free_energy_curve_c1(net_c1, eta, S, k, num_alphas=bulk.NUM_ALPHAS),
            bulk.free_energy_curve_c1_alt(net_c1, eta, S, k, num_alphas=bulk.NUM_ALPHAS))


def compute_landscapes(cfg, device):
    """Compute the three landscapes at every eta and save them."""
    net_phi = load_network(cfg, "phi", device)
    net_c1 = load_network(cfg, "c1", device)
    offset = bulk.phi_gauge_constant(net_phi, cfg.k)
    print(f"Phi-network gauge constant (beta f at rho = 0): {offset:+.6f}")

    curves = np.empty((len(eta_values), 4, num_points))
    for i, eta in enumerate(eta_values):
        curves[i] = landscapes(net_phi, net_c1, eta, cfg.k, offset)
        print(f"eta = {eta} done")

    path = landscapes_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, eta=np.array(eta_values), curves=curves)
    print(f"Saved: {path}  shape {curves.shape}")
    return path
