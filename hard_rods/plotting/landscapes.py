"""
Bulk free-energy landscapes beta f(S) (Sec. 4.3), one panel per eta; the dashed line is the
exact beta f at S = +-1 (one-dimensional rods, for which LC is exact).
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from ..evaluation.bulk_free_energy import ideal_free_energy_density
from ..evaluation.landscapes import landscapes_path
from ..functionals.lafuente_cuesta import lafuente_cuesta_excess
from .style import DOUBLE_COLUMN, figure_path, paper_style

# legend labels of the three curves (rows 1-3 of the saved array) and their style
LABELS = (r"$\Phi$-network", r"direct-$c^{(1)}$ network: standard integration path",
          r"direct-$c^{(1)}$ network: alternative integration path")
STYLES = (("tab:blue", "-", 1.2), ("tab:red", "--", 1.0), ("tab:orange", ":", 1.0))


def exact_free_energy_aligned(eta, k):
    """beta f per site with all rods in one orientation: LC (exact in one dimension) plus the ideal part."""
    rho = torch.zeros(2, 1, 1)
    rho[0] = eta / k
    return lafuente_cuesta_excess(rho, k).item() + ideal_free_energy_density(eta / k, 0.0)


def plot_landscapes(cfg, figure_dir):
    """Draw k<k>_free_energy_landscapes.pdf into figure_dir. Returns the path."""
    data = np.load(landscapes_path(cfg))
    eta_values, curves = data["eta"], data["curves"]

    with paper_style():
        # no shared y axis: every eta sits at its own free energy
        fig, axes = plt.subplots(1, len(eta_values), figsize=(DOUBLE_COLUMN, 2.4),
                                 squeeze=False, layout="constrained")
        for col, (ax, eta) in enumerate(zip(axes[0], eta_values)):
            S = curves[col, 0]
            for row, (colour, style, width) in enumerate(STYLES, start=1):
                ax.plot(S, curves[col, row], color=colour, linestyle=style, linewidth=width,
                        label=LABELS[row - 1])
            f_aligned = exact_free_energy_aligned(eta, cfg.k)
            print(f"  eta = {eta}: exact beta f at S = +-1: {f_aligned}")
            ax.axhline(f_aligned, color="k", linestyle="--", linewidth=0.75,
                       label=r"Exact $\beta f$ for $S = \pm 1$")

            ax.axvline(0.0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(rf"$\eta = {eta}$")
            ax.set_xlabel(r"$S$")
            if col == 0:
                ax.set_ylabel(r"$\beta f$")

        # one legend for the figure, below the panels
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=len(labels),
                   framealpha=0.0, handletextpad=0.5, columnspacing=1.5)

        out_path = figure_path(figure_dir, f"k{cfg.k}_free_energy_landscapes.pdf")
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
