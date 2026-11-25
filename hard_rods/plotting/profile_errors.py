"""
Errors of the self-consistent validation profiles against GCMC versus packing fraction
(Sec. 4.1), for the three networks, LC and the GCMC noise. quantity="c1" compares c1 on the
GCMC densities instead (supplementary figure).
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from ..config import NETWORK_TYPES, profile_path
from ..utils import packing_fraction
from .style import SINGLE_COLUMN, figure_path, paper_style

# sites whose GCMC relative standard error exceeds this are left out
max_relative_error = 0.25

# drawing order and legend: the networks, then LC and the GCMC noise
STYLES = {
    "c1": {"label": r"direct-$c^{(1)}$ network", "color": "tab:red", "marker": "o", "s": 6,
           "linewidths": 0},
    "c1_regularised": {"label": r"direct-$c^{(1)}$ network, regularized", "color": "tab:green",
                       "marker": "o", "s": 6, "linewidths": 0},
    "phi": {"label": r"$\Phi$-network", "color": "tab:blue", "marker": "o", "s": 6,
            "linewidths": 0},
    "LC": {"label": "LC", "color": "k", "marker": "x", "s": 8, "linewidths": 0.5},
    "noise": {"label": r"$\sigma_\mathrm{GCMC}$", "facecolors": "none", "edgecolors": "0.45",
              "marker": "o", "s": 8, "linewidths": 0.5},
}

# evaluation folder of every compared source
SOURCES = {**{network_type: network_type for network_type in NETWORK_TYPES}, "LC": "lafuente_cuesta"}

# axis labels (absolute, relative) and figure name; quotients instead of \frac keep scripts legible
QUANTITIES = {
    "rho": {"labels": (r"$\left\langle \left|\rho_\mathrm{GCMC}-\rho_\mathrm{DFT}\right|\right\rangle$",
                       r"$\left\langle \left|\rho_\mathrm{GCMC}-\rho_\mathrm{DFT}\right|"
                       r" / \rho_\mathrm{GCMC}\right\rangle$"),
            "figure": "profile_deviations"},
    "c1": {"labels": (r"$\left\langle \left|c^{(1),\mathrm{GCMC}}-c^{(1),\mathrm{DFT}}\right|\right\rangle$",
                      r"$\left\langle \left|c^{(1),\mathrm{GCMC}}-c^{(1),\mathrm{DFT}}\right|"
                      r" / \left|c^{(1),\mathrm{GCMC}}\right|\right\rangle$"),
           "figure": "network_comparison_c1"},
}


def _load(path):
    """The array saved at path, or None if there is no such file."""
    return np.load(path) if os.path.exists(path) else None


def abs_and_rel_means(values, target):
    """The mean absolute size of values, and its mean size relative to the target."""
    return np.mean(np.abs(values)), np.mean(np.abs(values / target))


def collect_points(cfg, quantity):
    """(eta, E_abs, E_rel) per validation profile and source; also the number of profiles above the eta cut."""
    k = cfg.k
    validation = cfg.profile_dir("validation")
    points = {source: [] for source in STYLES}
    above_eta_cap = 0
    for i in range(cfg.n_validation):
        rho = _load(profile_path(validation, "rho", i))
        if rho is None:
            continue
        rho_err = np.load(profile_path(validation, "rho_sem", i))
        target = np.load(profile_path(validation, quantity, i))

        eta = packing_fraction(rho, k)
        if eta > cfg.max_packing_fraction_comparison:
            above_eta_cap += 1
            continue

        # compared sites: non-zero, well-sampled GCMC density
        bins = (rho > 0) & (rho_err <= max_relative_error * rho)
        if not bins.any():
            continue

        for source, functional in SOURCES.items():
            prediction = _load(profile_path(cfg.evaluation_dir(functional), quantity, i))
            if prediction is None:       # this functional has not been evaluated
                continue
            ok = bins & np.isfinite(prediction)
            if ok.any():
                points[source].append((eta, *abs_and_rel_means(prediction[ok] - target[ok],
                                                               target[ok])))

        # GCMC standard error on the same sites (relative error of rho for c1 = ln(rho) + u)
        sigma = rho_err[bins] if quantity == "rho" else rho_err[bins] / rho[bins]
        points["noise"].append((eta, *abs_and_rel_means(sigma, target[bins])))

    return points, above_eta_cap


def plot_profile_errors(cfg, figure_dir, quantity="rho"):
    """Draw k<k>_profile_deviations.pdf (or k<k>_network_comparison_c1.pdf) into figure_dir."""
    points, above_eta_cap = collect_points(cfg, quantity)
    print(f"{quantity}: {above_eta_cap} profiles skipped above eta = "
          f"{cfg.max_packing_fraction_comparison}")
    if not any(points[network_type] for network_type in NETWORK_TYPES):
        print("no network had data -- comparison plot not written")
        return None

    q = QUANTITIES[quantity]
    with paper_style():
        # stacked panels at single-column width, sharing the eta axis
        fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(SINGLE_COLUMN, 5.1), sharex=True,
                                             layout="constrained")
        for source in STYLES:
            if not points[source]:
                print(f"  {source}: no data -- left off the plot")
                continue
            eta, absolute, relative = np.array(points[source]).T
            ax_abs.scatter(eta, absolute, alpha=0.6, **STYLES[source])
            ax_rel.scatter(eta, relative, alpha=0.6, **STYLES[source])
            print(f"  {source}: {eta.size} profiles, mean deviation abs {absolute.mean():.4g}, "
                  f"rel {relative.mean():.3f}")

        ax_abs.set_ylabel(q["labels"][0])
        ax_rel.set_ylabel(q["labels"][1])
        ax_rel.set_xlabel(r"$\eta$")
        for ax in (ax_abs, ax_rel):
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.7)
        # one legend below both panels, filled column by column
        handles, labels = ax_abs.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False,
                   handletextpad=0.3, labelspacing=0.3, columnspacing=1.0)

        out_path = figure_path(figure_dir, f"k{cfg.k}_{q['figure']}.pdf")
        # no bbox_inches="tight", which would change the width away from the column width
        fig.savefig(out_path, format="pdf", dpi=300, transparent=True)
        plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
