"""
Landau analysis (Sec. 4.5): a_n = d^n(beta f)/dS^n at S = 0 (n = 2, 4, 6) of both networks, by
central differences in float64; a_2 of the direct-c1 network also directly from c1; and |S| of
the sixth-order Landau polynomial near the Phi-network's onset of order.
"""

import os

import numpy as np
import torch

from ..functionals.neural import c1_direct, load_network, phi_field
from . import phase_diagram

eta_values = np.linspace(0.02, 0.94, 200)
h = 0.01                                    # step of the S grid
S = np.linspace(-6 * h, 6 * h, 13)          # the 13 points the six derivatives at S = 0 need
num_alphas = 1000                           # integration steps for the direct-c1 network

# Landau polynomial: eta window around the Phi-network's onset of order, and the S grid on
# which the polynomial's minimum is read off
landau_eta_values = np.linspace(0.57, 0.61, 201)
landau_S_grid = np.linspace(0.0, 1.0, 2001)


def landau_path(cfg):
    return cfg.bulk_path("landau_coefficients.npz")


def bulk_window(net, rho):
    """A uniform window with the densities rho = (rho0, rho1), as small as circular padding allows."""
    size = net.conv1.weight.shape[-1] // 2
    return rho.view(1, 2, 1, 1).expand(1, 2, size, size)


def f_ideal(eta, S_values, k):
    """beta f_id per site, for every S."""
    rho = eta / (2 * k) * np.stack((1 + S_values, 1 - S_values))
    return (rho * (np.log(rho) - 1)).sum(axis=0)


def f_excess(net, network_type, eta, S_value, k, device):
    """beta f_ex per site of the bulk at (eta, S_value)."""
    rho = eta / (2 * k) * torch.tensor([1 + S_value, 1 - S_value], dtype=torch.float64, device=device)
    window = bulk_window(net, rho)

    if network_type == "phi":
        return phi_field(net, window).mean().item()

    alphas = torch.linspace(0.0, 1.0, num_alphas, dtype=torch.float64, device=device)
    c1 = c1_direct(net, alphas[:, None, None, None] * window).mean(dim=(2, 3))    # [num_alphas, 2]
    integrand = c1 @ rho                                                         # sum_i rho_i c1_i
    return -torch.trapezoid(integrand, alphas).item()


def landscape(net, network_type, eta, k, device):
    """beta f per site on the grid S."""
    with torch.no_grad():
        f_exc = np.array([f_excess(net, network_type, eta, s, k, device) for s in S])
    return f_ideal(eta, S, k) + f_exc


def even_derivatives(f):
    """d^2f/dS^2, d^4f/dS^4 and d^6f/dS^6 of the landscape f at S = 0, by central differences."""
    d1 = np.gradient(f, S)
    d2 = np.gradient(d1, S)
    d3 = np.gradient(d2, S)
    d4 = np.gradient(d3, S)
    d5 = np.gradient(d4, S)
    d6 = np.gradient(d5, S)
    centre = len(S) // 2
    return d2[centre], d4[centre], d6[centre]


def second_derivative_from_c1(net, eta, k, device):
    """a_2 directly from the direct-c1 network: eta/k - eta/(2k) d(c1_0 - c1_1)/dS at S = 0."""
    S_value = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    rho = eta / (2 * k) * torch.stack((1 + S_value, 1 - S_value))
    c1 = c1_direct(net, bulk_window(net, rho)).mean(dim=(0, 2, 3))     # [2]
    dc1_dS = torch.autograd.grad(c1[0] - c1[1], S_value)[0].item()
    return eta / k - eta / (2 * k) * dc1_dS


def landau_order_parameter(a2, a4, a6):
    """|S| at the global minimum over [0, 1] of the sixth-order Landau polynomial."""
    f = a2 * landau_S_grid**2 / 2 + a4 * landau_S_grid**4 / 24 + a6 * landau_S_grid**6 / 720
    return landau_S_grid[np.argmin(f)]


def jumps_in(S_values):
    """The indices j where |S| jumps (by more than 0.1) between neighbouring eta."""
    return np.flatnonzero(np.abs(np.diff(S_values)) > 0.1)


def landau_curve(net, k, device):
    """|S| of the Landau polynomial at every landau_eta_values (float64 Phi-network)."""
    coefficients = np.array([even_derivatives(landscape(net, "phi", eta, k, device))
                             for eta in landau_eta_values])
    S_landau = np.array([landau_order_parameter(*a) for a in coefficients])

    for j in jumps_in(S_landau):
        print(f"Landau jump between eta {landau_eta_values[j]:.4f} and "
              f"{landau_eta_values[j + 1]:.4f}: |S| {S_landau[j]:.3f} -> {S_landau[j + 1]:.3f}")
    return S_landau


def compute_landau_analysis(cfg, device):
    """
    Compute and save bulk/landau_coefficients.npz with the keys eta, phi_a2/a4/a6, c1_a2/a4/a6,
    c1_a2_from_c1, eta_jump and phi_at_jump (the Phi-network's first jump in |S| and its
    coefficients there), landau_eta and landau_S. Needs the global-minimum table.
    """
    k = cfg.k
    # the Phi-network's first jump in |S|, midway between the two eta either side of it
    eta_grid, S_grid = np.load(phase_diagram.global_minimum_path(cfg))[:2]
    j = np.flatnonzero(np.abs(np.diff(S_grid)) > 0.1)[0]
    eta_jump = (eta_grid[j] + eta_grid[j + 1]) / 2
    print(f"Phi-network's first |S| jump at eta = {eta_jump:.4f}")

    results = {"eta": eta_values, "eta_jump": eta_jump}
    for network_type in ("phi", "c1"):
        net = load_network(cfg, network_type, device).double()
        d2, d4, d6 = np.array([even_derivatives(landscape(net, network_type, eta, k, device))
                               for eta in eta_values]).T
        results.update({f"{network_type}_a2": d2, f"{network_type}_a4": d4,
                        f"{network_type}_a6": d6})

        if network_type == "phi":
            # at the jump itself, so that the marker sits on the curve
            results["phi_at_jump"] = np.array(
                even_derivatives(landscape(net, network_type, eta_jump, k, device)))
            results["landau_eta"] = landau_eta_values
            results["landau_S"] = landau_curve(net, k, device)

        if network_type == "c1":
            results["c1_a2_from_c1"] = np.array([second_derivative_from_c1(net, eta, k, device)
                                                 for eta in eta_values])

    path = landau_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **results)
    print(f"Saved: {path}")
    return path
