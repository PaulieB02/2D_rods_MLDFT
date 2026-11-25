"""
Violation of the c2 exchange symmetry c2_ij(r, r') = c2_ji(r', r) (Appendix B), measured with
the mixed-derivative penalty on 2000 random (4k+1) windows of the validation profiles (seed 0).
The Phi-network's values (~1e-14) are float32 round-off and not bit-reproducible on a GPU.
Writes bulk/c2_symmetry_losses.npz: per network, [n_windows, 3] (inter, intra, total).
"""

import os

import numpy as np
import torch
from torch.func import grad, jvp, vjp

from ..config import profile_path
from ..functionals.neural import c1_direct, load_network, phi_field

n_windows = 2000
seed = 0
# order in which the networks are evaluated (and drawn)
NETWORK_ORDER = ("c1", "c1_regularised", "phi")


def c2_losses_path(cfg):
    return cfg.bulk_path("c2_symmetry_losses.npz")


def random_windows(cfg, device):
    """n_windows random [2, 4k+1, 4k+1] windows cut from the validation profiles."""
    window_size = 4 * cfg.k + 1
    directory = cfg.profile_dir("validation")
    profiles = [np.load(profile_path(directory, "rho", i)) for i in range(cfg.n_validation)]

    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(n_windows):
        rho = profiles[rng.integers(len(profiles))]
        top = rng.integers(rho.shape[1] - window_size + 1)
        left = rng.integers(rho.shape[2] - window_size + 1)
        windows.append(rho[:, top:top + window_size, left:left + window_size])
    return torch.tensor(np.array(windows), dtype=torch.float32, device=device)


def c1_from_phi_window(net, window, k):
    """c1 of the Phi-network on the central 2k+1 sites of a window (padding off)."""
    c1 = -grad(lambda w: phi_field(net, w).sum())(window)
    return c1[:, k:-k, k:-k]


def jacobian_slices(c1_of, window, k):
    """
    Row d c1_i(centre)/d rho_j(r) and column d c1_j(r)/d rho_i(centre) of c2 over the central
    2k+1 sites; c1_of maps a window to c1 on those sites. Both are [2, 2, 2k+1, 2k+1].
    """
    c1, pullback = vjp(c1_of, window)       # pullback(v) = v . d c1 / d rho
    c1_centre = c1.shape[-1] // 2           # the centre in the c1 field (k)
    rho_centre = window.shape[-1] // 2      # the centre in the window (2k)

    rows, columns = [], []
    for i in range(2):
        # reverse mode, from c1_i at the centre to every density
        pick_c1 = torch.zeros_like(c1)
        pick_c1[i, c1_centre, c1_centre] = 1.0
        row = pullback(pick_c1)[0]                       # [2, 4k+1, 4k+1]
        rows.append(row[:, k:-k, k:-k])

        # forward mode, from the density rho_i at the centre to every c1
        nudge_rho = torch.zeros_like(window)
        nudge_rho[i, rho_centre, rho_centre] = 1.0
        _, column = jvp(c1_of, (window,), (nudge_rho,))  # [2, 2k+1, 2k+1]
        columns.append(column)

    return torch.stack(rows), torch.stack(columns)


def curl_losses(c1_of, window, k):
    """The inter-species, the intra-species and the summed penalty of one window."""
    row, column = jacobian_slices(c1_of, window, k)
    squared_difference = (row - column) ** 2
    inter = (squared_difference[0, 1].mean() + squared_difference[1, 0].mean()).item()
    intra = (squared_difference[0, 0].mean() + squared_difference[1, 1].mean()).item()
    return inter, intra, inter + intra


def compute_c2_losses(cfg, device):
    """Evaluate every network on the same random windows and save the losses."""
    k = cfg.k
    windows = random_windows(cfg, device)
    print(f"{n_windows} windows of {4 * k + 1}x{4 * k + 1}")

    losses = {}
    for network_type in NETWORK_ORDER:
        net = load_network(cfg, network_type, device)
        net.disable_padding()
        if network_type == "phi":
            c1_of = lambda w: c1_from_phi_window(net, w, k)
        else:
            c1_of = lambda w: c1_direct(net, w)
        losses[network_type] = np.array([curl_losses(c1_of, window, k) for window in windows])

    for n, part in enumerate(("inter-species", "intra-species", "total")):
        print(f"[{part}]")
        for network_type, loss in losses.items():
            print(f"  {network_type:15s} mean {loss[:, n].mean():.3e}  median {np.median(loss[:, n]):.3e}")

    path = c2_losses_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **losses)
    print(f"Saved: {path}")
    return path
