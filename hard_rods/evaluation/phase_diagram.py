"""
Orientational order |S|(eta) of the Phi- and the direct-c1 network (Sec. 4.4) by two methods:
the global minimum of beta f(S), and the minima where d(beta f)/dS, computed from c1, crosses
zero from below (first-derivative method). Also the fits and jumps drawn in the figure.

Writes bulk/phase_diagram_global_min.npy and bulk/phase_diagram_derivative.npy: one (eta, |S|)
row pair per network (Phi-network first), NaN-padded.
"""

import os

import numpy as np
import torch
from scipy.optimize import curve_fit

from ..functionals.neural import c1_direct, c1_from_phi, load_network
from . import bulk_free_energy as bulk

# the eta axis shared by both methods
eta_values = np.linspace(0.05, 1.0, 200)

# global-minimum method: S grid, one resolution per route
S_min, S_max, num_points_c1, num_points_phi = -1.0, 1.0, 200, 2000

# first-derivative method: S grid in [0, 1] on which the zero crossings are located
num_points_derivative = 2000

# the network each (eta, |S|) row pair of the saved tables belongs to, in row order
SERIES = ("phi", "c1")

# fits: only eta up to here enters, and |S| above S_ORDERED counts as ordered
ETA_FIT_MAX = 0.8
S_ORDERED = 0.1


def global_minimum_path(cfg):
    return cfg.bulk_path("phase_diagram_global_min.npy")


def derivative_path(cfg):
    return cfg.bulk_path("phase_diagram_derivative.npy")


def diagram_table(*series):
    """Pack (eta, |S|) point sets into one NaN-padded array, rows eta, |S| per series."""
    n = max(len(eta) for eta, _ in series)
    rows = []
    for eta, S in series:
        pad = np.full(n - len(eta), np.nan)
        rows.append(np.concatenate((np.asarray(eta, dtype=float), pad)))
        rows.append(np.concatenate((np.asarray(S, dtype=float), pad)))
    return np.vstack(rows)


def _save(path, table):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, table)
    print(f"Saved: {path}  shape {table.shape}")
    return path


# ---------------------------------------------------------------------------
# global-minimum method
# ---------------------------------------------------------------------------
def order_parameters(curve_fn, num_points):
    """|S| at the global minimum of f(S), one per eta, for a free-energy curve function."""
    S = np.linspace(S_min, S_max, num_points)
    return np.array([abs(S[np.argmin(curve_fn(eta, S))]) for eta in eta_values])


def global_minimum_diagram(cfg, device):
    """Global minimum of f(S) at every eta for both networks; saved to the bulk folder."""
    k = cfg.k
    net_phi = load_network(cfg, "phi", device)
    net_c1 = load_network(cfg, "c1", device)

    print(f"global minimum: {len(eta_values)} eta values, {num_points_phi} S points for the "
          f"Phi-network, {num_points_c1} for the direct-c1 network")
    S_phi = order_parameters(lambda eta, S: bulk.free_energy_curve_phi(net_phi, eta, S, k),
                             num_points_phi)
    S_c1 = order_parameters(
        lambda eta, S: bulk.free_energy_curve_c1(net_c1, eta, S, k, num_alphas=bulk.NUM_ALPHAS),
        num_points_c1)
    return _save(global_minimum_path(cfg),
                 diagram_table((eta_values, S_phi), (eta_values, S_c1)))


# ---------------------------------------------------------------------------
# first-derivative method
# ---------------------------------------------------------------------------
def bulk_c1_phi(net, rho):
    """c1 per site of both species from the Phi-network for uniform windows, [N, 2]."""
    rho = rho.detach().requires_grad_(True)
    with torch.enable_grad():
        c1 = c1_from_phi(net, rho, local_learning_mode=False, create_graph=False)
    kernel = net.conv1.weight.shape[-1]
    n_sites = (rho.shape[-1] + 2 * (kernel // 2) - kernel + 1) ** 2
    return c1.detach().sum(dim=(2, 3)) / n_sites


def bulk_c1_c1(net, rho):
    """c1 per site of both species from the direct-c1 network, [N, 2]."""
    with torch.no_grad():
        return c1_direct(net, rho).mean(dim=(2, 3))


def free_energy_derivative_curve(net, c1_of_rho, eta, S_array, k, window_size=None):
    """d(beta f)/dS at fixed eta for every S, from c1 alone (c1_of_rho: bulk_c1_phi or bulk_c1_c1)."""
    if window_size is None:
        window_size = bulk.bulk_window_size(net)
    rho0, rho1, rho = bulk.bulk_windows(net, eta, S_array, k, window_size)

    c1 = c1_of_rho(net, rho).double().cpu().numpy()          # [N, 2]
    return eta / k / 2.0 * ((np.log(rho0) - np.log(rho1)) - (c1[:, 0] - c1[:, 1]))


def all_minima(net, c1_of_rho, k):
    """Every minimum of f at every eta as flat (eta, |S|) arrays; an eta with two minima appears twice."""
    S = np.linspace(0.0, 1.0, num_points_derivative)
    eta_points, S_points = [], []

    for eta in eta_values:
        dF = free_energy_derivative_curve(net, c1_of_rho, float(eta), S, k)

        # S = 0 is stationary by symmetry; float32 noise must not decide the sign of dF there
        dF[0] = 0.0

        # f rising immediately above S = 0 makes the isotropic state a minimum
        roots = [0.0] if dF[1] > 0 else []
        # every crossing from below, linearly interpolated between the grid points
        for j in np.flatnonzero((dF[:-1] < 0) & (dF[1:] >= 0)):
            roots.append(S[j] - dF[j] * (S[j + 1] - S[j]) / (dF[j + 1] - dF[j]))

        if not roots:
            raise RuntimeError(
                f"eta = {eta}: no minimum found -- d(beta f)/dS never crosses zero from "
                f"below, which the ideal-gas divergence at S = 1 should make impossible "
                f"(dF[1] = {dF[1]:.3e}, dF[-1] = {dF[-1]:.3e})")

        for root in roots:
            eta_points.append(float(eta))
            S_points.append(root)
    return np.array(eta_points), np.array(S_points)


def first_derivative_diagram(cfg, device):
    """Every minimum at every eta for both networks; saved to the bulk folder."""
    k = cfg.k
    net_phi = load_network(cfg, "phi", device)
    net_c1 = load_network(cfg, "c1", device)

    print(f"first derivative: {len(eta_values)} eta values, {num_points_derivative} S points each")
    phi_points = all_minima(net_phi, bulk_c1_phi, k)
    c1_points = all_minima(net_c1, bulk_c1_c1, k)
    return _save(derivative_path(cfg), diagram_table(phi_points, c1_points))


# ---------------------------------------------------------------------------
# fits and jumps, for the figure
# ---------------------------------------------------------------------------
def tanh_model(eta, a, b, eta_c, w):
    return a + b * np.tanh((eta - eta_c) / w)


def largest_S_per_eta(eta, S):
    """One point per eta, sorted, keeping the largest |S| there. Drops the NaN padding."""
    keep = np.isfinite(eta) & np.isfinite(S)
    eta, S = eta[keep], S[keep]
    unique = np.unique(eta)
    return unique, np.array([S[eta == e].max() for e in unique])


def fit_tanh(eta, S):
    """tanh fit, started from plateaus at 0 and max |S| and a narrow width."""
    half = S.max() / 2
    p0 = (half, half, eta[np.argmax(S > half)], 0.02)
    (a, b, eta_c, w), _ = curve_fit(tanh_model, eta, S, p0=p0)
    return {"eta_c": eta_c, "w": w, "a": a, "b": b}


def fit_step(eta, S):
    """Step between the last disordered and the first ordered point."""
    first = np.argmax(S > S_ORDERED)
    S_below, S_above = S[first - 1], S[first]
    return {"eta_c": (eta[first - 1] + eta[first]) / 2, "w": 0.0,
            "a": (S_above + S_below) / 2, "b": (S_above - S_below) / 2}


def fit_all(cfg):
    """
    Fits keyed "<method>_<network>" (method "global" or "derivative"): a tanh for the
    Phi-network, a step for the direct-c1 network; each a dict of eta_c, w, a and b.
    """
    fits = {}
    for method, path in (("global", global_minimum_path(cfg)), ("derivative", derivative_path(cfg))):
        if not os.path.exists(path):
            continue
        table = np.load(path)
        for row, network in enumerate(SERIES):
            name = f"{method}_{network}"
            eta, S = largest_S_per_eta(table[2 * row], table[2 * row + 1])
            inside = eta <= ETA_FIT_MAX
            eta, S = eta[inside], S[inside]
            if not (S > S_ORDERED).any():
                print(f"  {name:<15s} never ordered, no fit")
                continue
            fit = fit_step if network == "c1" else fit_tanh
            fits[name] = fit(eta, S)
            print(f"  {name:<15s} eta_c = {fits[name]['eta_c']:.4f}  w = {fits[name]['w']:.5f}")
    return fits


def phi_jumps(cfg, eta_max=0.8):
    """(eta, lower |S|, upper |S|) wherever the derivative method finds two Phi-network minima."""
    eta, S = np.load(derivative_path(cfg))[:2]      # the Phi-network's rows
    keep = np.isfinite(eta) & (eta < eta_max)
    eta, S = eta[keep], S[keep]
    values, counts = np.unique(eta, return_counts=True)
    return [(e, S[eta == e].min(), S[eta == e].max()) for e in values[counts > 1]]
