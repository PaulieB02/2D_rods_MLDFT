"""
Bulk equation of state (Sec. 4.2): eta(mu) of GCMC, the networks and LC, and the deviations
from GCMC; for k = 8 a dashed line marks the GCMC transition.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from ..config import NETWORK_TYPES, read_critical_packing_fraction
from ..data_generation.bulk_gcmc import eos_gcmc_path
from ..evaluation.equation_of_state import eos_lc_path, eos_networks_path
from .style import SINGLE_COLUMN, figure_path, paper_style

# label, colour, marker; GCMC is drawn in black
STYLES = {"phi": (r"$\Phi$-network", "tab:blue", "^"),
          "c1": (r"direct-$c^{(1)}$ network", "tab:red", "o"),
          "c1_regularised": (r"direct-$c^{(1)}$ network, regularized", "tab:green", "v"),
          "LC": ("LC", "tab:purple", "x")}


def plot_equation_of_state(cfg, figure_dir):
    """Draw k<k>_equation_of_state.pdf into figure_dir. Returns the path."""
    with paper_style():
        fig, (ax, axres) = plt.subplots(2, 1, figsize=(SINGLE_COLUMN, 3.9), sharex=True,
                                        height_ratios=(2, 1), layout="constrained")

        # GCMC, networks and LC share the mu grid config.EOS_MU_VALUES
        mu, eta_gcmc = np.load(eos_gcmc_path(cfg))
        ax.plot(mu, eta_gcmc, color="black", marker="s", markersize=3, markerfacecolor="none",
                markeredgewidth=0.7, linewidth=0.8, zorder=4, label="GCMC")

        # rows: mu, then one eta per network
        table = np.load(eos_networks_path(cfg))
        series = dict(zip(NETWORK_TYPES, table[1:1 + len(NETWORK_TYPES)]))
        if os.path.exists(eos_lc_path(cfg)):
            series["LC"] = np.load(eos_lc_path(cfg))[1]          # rows: mu, eta, converged
        for name, eta in series.items():
            label, colour, marker = STYLES[name]
            # open markers, so that coinciding curves stay visible
            style = {"color": colour, "marker": marker, "markersize": 3, "markerfacecolor": "none",
                     "markeredgewidth": 0.7, "linewidth": 0.8}
            ax.plot(mu, eta, label=label, **style)
            axres.plot(mu, eta - eta_gcmc, **style)

        # the residual panel is scaled to the networks alone
        residuals = np.array([series[name] - eta_gcmc for name in NETWORK_TYPES])
        pad = 0.1 * (residuals.max() - residuals.min())
        axres.set_ylim(residuals.min() - pad, residuals.max() + pad)

        # GCMC transition, at the mu where the GCMC eta(mu) reaches eta_c
        if cfg.eta_c_file is not None:
            eta_c = read_critical_packing_fraction(cfg.eta_c_file)
            mu_c = np.interp(eta_c, eta_gcmc, mu)
            print(f"  gcmc: transition eta_c = {eta_c:.4f} at mu = {mu_c:.3f}")
            ax.axvline(mu_c, color="black", linestyle="--", linewidth=0.75, zorder=1,
                       label=r"$\mu_\mathrm{c}^\mathrm{GCMC}$")
            axres.axvline(mu_c, color="black", linestyle="--", linewidth=0.75, zorder=1)

        ax.set_ylabel(r"$\eta$")
        ax.grid(True, linestyle="--", alpha=0.7)
        # eta rises with mu, so the upper left stays empty
        ax.legend(loc="upper left", framealpha=0.0, handletextpad=0.5, labelspacing=0.3,
                  borderpad=0.3)

        axres.axhline(0.0, color="black", linewidth=0.5, zorder=1)
        axres.set_ylabel(r"$\eta - \eta_\mathrm{GCMC}$")
        axres.set_xlabel(r"$\mu$")
        axres.grid(True, linestyle="--", alpha=0.7)

        out_path = figure_path(figure_dir, f"k{cfg.k}_equation_of_state.pdf")
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
