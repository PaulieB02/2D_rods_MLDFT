"""Small helpers shared by several subpackages."""

import numpy as np
import torch


def packing_fraction(rho, k):
    """Packing fraction eta = k (rho_0 + rho_1) averaged over all sites (walls included); rho is [2, H, W]."""
    n_bins = rho.size                  # both species, so 2 * H * W
    if n_bins == 0:
        return 0.0
    return np.sum(rho) / n_bins * k * 2


def default_device():
    """The fastest available torch device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
