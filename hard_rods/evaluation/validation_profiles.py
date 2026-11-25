"""
Self-consistent profiles of all validation potentials for each functional (Sec. 4.1).

Writes evaluation/<functional>/rho/rho_###.npy (Picard solution) and c1/c1_###.npy (c1 on the
GCMC density).
"""

import os

import numpy as np
import torch

from ..config import profile_path
from ..functionals.lafuente_cuesta import c1_lafuente_cuesta
from ..functionals.neural import c1_for_picard, load_network
from .picard import picard_minimise


def evaluate_functional(cfg, functional, c1_of_rho, device):
    """Minimise every validation profile with c1_of_rho and save the results; returns the non-converged indices."""
    settings = cfg.picard_validation
    out_dir = cfg.evaluation_dir(functional)
    os.makedirs(os.path.join(out_dir, "rho"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "c1"), exist_ok=True)
    validation_dir = cfg.profile_dir("validation")

    not_converged = []
    for i in range(cfg.n_validation):
        u = np.load(profile_path(validation_dir, "vext", i))
        rho_mc = np.load(profile_path(validation_dir, "rho", i))

        rho_min, steps = picard_minimise(c1_of_rho, u, settings.alpha, settings.maxiter,
                                         settings.tol, device)
        converged = steps < settings.maxiter
        if not converged:
            not_converged.append(i)
        rho_min = rho_min.detach().cpu().numpy()

        # c1 on the GCMC density, both species in one call
        rho_mc_tensor = torch.from_numpy(np.float32(rho_mc)).to(device)
        c1_mc = c1_of_rho(rho_mc_tensor).cpu().numpy()

        # non-finite results are never written, so that they cannot enter any average
        if np.isfinite(c1_mc).all():
            np.save(profile_path(out_dir, "c1", i), c1_mc)
        else:
            print(f"profile {i:03d}: c1 contains non-finite values -- not saved")
        if np.isfinite(rho_min).all():
            np.save(profile_path(out_dir, "rho", i), rho_min)
        else:
            print(f"profile {i:03d}: minimisation produced a non-finite density -- not saved")

        status = f"converged in {steps} steps" if converged else f"not converged in {steps} steps"
        print(f"{functional}: profile {i:03d} evaluation completed ({status})")

    print(f"{functional}: {len(not_converged)}/{cfg.n_validation} profiles did not converge"
          + (f": {not_converged}" if not_converged else ""))
    return not_converged


def evaluate_network(cfg, network_type, device):
    """Validation profiles of one trained network ("phi", "c1" or "c1_regularised")."""
    net = load_network(cfg, network_type, device)
    return evaluate_functional(cfg, network_type, c1_for_picard(net, network_type), device)


def evaluate_lafuente_cuesta(cfg, device):
    """Validation profiles of the LC reference functional."""
    return evaluate_functional(cfg, "lafuente_cuesta", c1_lafuente_cuesta(cfg.k), device)
