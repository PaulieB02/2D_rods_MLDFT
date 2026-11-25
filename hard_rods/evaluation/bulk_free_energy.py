"""
Bulk free-energy density beta f(S) per site at fixed packing fraction (Sec. 4.3), with
rho_0,1 = (1 +/- S) eta / (2k): from the Phi-network directly, and from the direct-c1 network
by integrating c1 along the simultaneous or the sequential path.
"""

import numpy as np
import torch

from ..functionals.neural import c1_direct, phi_field

# integration steps of the two c1 paths
NUM_ALPHAS = 1000


def bulk_window_size(net):
    """Smallest uniform window circular padding allows (kernel // 2)."""
    return net.conv1.weight.shape[-1] // 2


def bulk_windows(net, eta, S_array, k, window_size):
    """(rho0, rho1, uniform windows [len(S), 2, w, w]) for every S at fixed eta."""
    if window_size is None:
        window_size = bulk_window_size(net)

    rho_bulk = eta / k
    S = np.asarray(S_array, dtype=np.float64)
    rho0 = (1.0 + S) * rho_bulk / 2.0
    rho1 = (1.0 - S) * rho_bulk / 2.0

    device = next(net.parameters()).device
    rho = torch.empty(len(S), 2, window_size, window_size, device=device)
    rho[:, 0] = torch.as_tensor(rho0, dtype=torch.float32, device=device)[:, None, None]
    rho[:, 1] = torch.as_tensor(rho1, dtype=torch.float32, device=device)[:, None, None]
    return rho0, rho1, rho


def ideal_free_energy_density(rho0, rho1):
    """Ideal free-energy density per site, sum_i rho_i (ln rho_i - 1), with 0 ln 0 = 0."""
    def xlogx(x):
        x = np.asarray(x, dtype=np.float64)
        return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)
    return xlogx(rho0) - np.asarray(rho0) + xlogx(rho1) - np.asarray(rho1)


def free_energy_curve_phi(net, eta, S_array, k, window_size=None):
    """beta f(S) at packing fraction eta for every S, from the Phi-network."""
    rho0, rho1, rho = bulk_windows(net, eta, S_array, k, window_size)

    with torch.no_grad():
        phi = phi_field(net, rho)                          # [len(S), 1, h, w]
    F_ex = phi.sum(dim=(1, 2, 3)).double().cpu().numpy()

    # per site of the phi field (one site larger than the window for an even kernel)
    n_sites = phi.shape[-1] * phi.shape[-2]

    return F_ex / n_sites + ideal_free_energy_density(rho0, rho1)


def phi_gauge_constant(net, k, window_size=None):
    """beta f of the Phi-network at zero density, the constant fixed by F_ex[0] = 0."""
    return float(free_energy_curve_phi(net, 0.0, [0.0], k, window_size)[0])


def free_energy_curve_c1(net, eta, S_array, k, window_size=None, num_alphas=NUM_ALPHAS):
    """beta f(S) from the direct-c1 network along the simultaneous path rho -> a rho."""
    if window_size is None:
        window_size = bulk_window_size(net)
    device = next(net.parameters()).device
    rho0, rho1, _ = bulk_windows(net, eta, S_array, k, window_size)
    alphas = torch.linspace(0.0, 1.0, num_alphas, device=device)
    F_ex = np.empty(len(rho0))

    with torch.no_grad():
        for i in range(len(rho0)):
            # one uniform window per a, both species scaled together
            rho = torch.empty(num_alphas, 2, window_size, window_size, device=device)
            rho[:, 0] = (alphas * float(rho0[i]))[:, None, None]
            rho[:, 1] = (alphas * float(rho1[i]))[:, None, None]

            c1 = c1_direct(net, rho)                          # [num_alphas, 2, h, w]
            c1_0, c1_1 = c1[:, 0], c1[:, 1]
            n_sites = c1_0.shape[-1] * c1_0.shape[-2]

            integrand = (float(rho0[i]) * c1_0 + float(rho1[i]) * c1_1).double().sum(dim=(1, 2))
            F_ex[i] = -float(torch.trapezoid(integrand, x=alphas.double())) / n_sites

    return F_ex + ideal_free_energy_density(rho0, rho1)


def free_energy_curve_c1_alt(net, eta, S_array, k, window_size=None, num_alphas=NUM_ALPHAS):
    """beta f(S) from the direct-c1 network along (0, 0) -> (rho_0, 0) -> (rho_0, rho_1)."""
    if window_size is None:
        window_size = bulk_window_size(net)
    device = next(net.parameters()).device
    rho0, rho1, _ = bulk_windows(net, eta, S_array, k, window_size)
    alphas = torch.linspace(0.0, 1.0, num_alphas, device=device)
    F_ex = np.empty(len(rho0))

    with torch.no_grad():
        for i in range(len(rho0)):
            # leg 1: species 0 from empty, species 1 absent
            rho = torch.zeros(num_alphas, 2, window_size, window_size, device=device)
            rho[:, 0] = (alphas * float(rho0[i]))[:, None, None]
            c1_0 = c1_direct(net, rho)[:, 0]

            # leg 2: species 0 at full density, species 1 from empty
            rho[:, 0] = float(rho0[i])
            rho[:, 1] = (alphas * float(rho1[i]))[:, None, None]
            c1_1 = c1_direct(net, rho)[:, 1]

            n_sites = c1_0.shape[-1] * c1_0.shape[-2]
            leg1 = (float(rho0[i]) * c1_0).double().sum(dim=(1, 2))
            leg2 = (float(rho1[i]) * c1_1).double().sum(dim=(1, 2))
            F_ex[i] = -float(torch.trapezoid(leg1 + leg2, x=alphas.double())) / n_sites

    return F_ex + ideal_free_energy_density(rho0, rho1)
