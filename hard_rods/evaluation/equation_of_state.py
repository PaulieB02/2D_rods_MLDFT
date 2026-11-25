"""
Bulk equation of state eta(mu) of the networks and of LC (Sec. 4.2), by Picard iteration of a
uniform periodic box. Writes bulk/eos_networks.npy, [4, n_mu] (beta*mu, then eta of the phi, c1
and c1_regularised networks), and bulk/eos_lc.npy, [3, n_mu] (beta*mu, eta, converged flag).
"""

import os

import numpy as np

from ..config import EOS_MU_VALUES, NETWORK_TYPES
from ..functionals.lafuente_cuesta import c1_lafuente_cuesta
from ..functionals.neural import c1_for_picard, load_network
from ..utils import packing_fraction
from .picard import picard_minimise


# floor of the mixing parameter for LC, whose ln(1 - n_0) rejects overshooting steps
LC_ALPHA_MIN = 0.01


def eos_networks_path(cfg):
    return cfg.bulk_path("eos_networks.npy")


def eos_lc_path(cfg):
    return cfg.bulk_path("eos_lc.npy")


def bulk_scan(c1_of_rho, window, k, settings, device, alpha_min=None):
    """eta and a 0/1 converged flag at every mu, for a uniform periodic box window x window."""
    eta = np.empty(len(EOS_MU_VALUES))
    converged = np.empty(len(EOS_MU_VALUES))
    for i, mu in enumerate(EOS_MU_VALUES):
        u = np.full((2, window, window), -float(mu), dtype=np.float32)
        rho, steps = picard_minimise(c1_of_rho, u, settings.alpha, settings.maxiter, settings.tol,
                                     device, alpha_min=alpha_min, S_initial=settings.S_initial)
        eta[i] = packing_fraction(rho.cpu().numpy(), k)
        converged[i] = steps < settings.maxiter
    return eta, converged


def network_equation_of_state(cfg, device):
    """Bulk fixed point of every network at every mu; saved to bulk/eos_networks.npy."""
    etas = []
    for network_type in NETWORK_TYPES:
        net = load_network(cfg, network_type, device)
        # a uniform box of half the kernel width, the smallest that circular padding allows
        window = net.conv1.weight.shape[-1] // 2
        eta, _ = bulk_scan(c1_for_picard(net, network_type), window, cfg.k, cfg.picard_eos, device)
        etas.append(eta)
        print(f"  {network_type:15s} done")

    path = eos_networks_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.vstack([EOS_MU_VALUES] + etas))
    print(f"Saved: {path}")
    return path


def lafuente_cuesta_equation_of_state(cfg, device):
    """Bulk fixed point of LC at every mu (box 3k x 3k); saved to bulk/eos_lc.npy."""
    eta, converged = bulk_scan(c1_lafuente_cuesta(cfg.k), 3 * cfg.k, cfg.k, cfg.picard_eos_lc,
                               device, alpha_min=LC_ALPHA_MIN)
    print(f"  LC              done ({int((1 - converged).sum())} did not converge)")

    path = eos_lc_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.vstack((EOS_MU_VALUES, eta, converged)))
    print(f"Saved: {path}")
    return path
