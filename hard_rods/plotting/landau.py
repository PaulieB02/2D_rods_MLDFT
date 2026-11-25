"""
Landau coefficients a_2, a_4 and a_6 against eta (Sec. 4.5), with the Phi-network's first jump
marked and, as an inset, the Landau polynomial's |S| against the full landscape's.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from ..evaluation import landau, phase_diagram
from .style import SINGLE_COLUMN, figure_path, paper_style

# colour and legend label per network, in drawing order
NETWORKS = {"phi": ("tab:blue", r"$\Phi$-network"),
            "c1": ("tab:red", r"direct-$c^{(1)}$ network via $f(S)$")}
# a_2 from the orientational balance, directly from c1
A2_FROM_C1_LABEL = r"direct-$c^{(1)}$ network via Eq. 21"


def draw_landau_comparison(ax, cfg, eta_values, S_landau, labels, marker_size):
    """The Landau polynomial's |S| (line) against the full landscape's global minimum (markers)."""
    eta_min, eta_max = eta_values[0], eta_values[-1]
    eta_grid, S_full = np.load(phase_diagram.global_minimum_path(cfg))[:2]
    window = (eta_grid >= eta_min) & (eta_grid <= eta_max)

    # the line is broken at every jump, and the two sides joined by a dotted line
    jumps = landau.jumps_in(S_landau)
    ax.plot(np.insert(eta_values, jumps + 1, np.nan), np.insert(S_landau, jumps + 1, np.nan),
            color="k", linewidth=1.0, label=labels[0])
    for j in jumps:
        ax.vlines((eta_values[j] + eta_values[j + 1]) / 2, S_landau[j], S_landau[j + 1],
                  color="k", linestyle=(0, (1, 1)), linewidth=0.75)
    ax.scatter(eta_grid[window], S_full[window], marker="^", s=marker_size, color="tab:blue",
               linewidths=0, zorder=3, label=labels[1])
    ax.set_xlim(eta_min, eta_max)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$|S|$")
    ax.grid(True, linestyle="--", alpha=0.7)


def plot_landau(cfg, figure_dir):
    """Draw k<k>_free_energy_derivatives_vs_eta.pdf into figure_dir. Returns the path."""
    data = np.load(landau.landau_path(cfg))
    eta_values = data["eta"]

    with paper_style():
        fig, (ax2, ax4, ax6) = plt.subplots(3, 1, figsize=(SINGLE_COLUMN, 6.4), sharex=True,
                                            layout="constrained")
        fig.get_layout_engine().set(hspace=0.08)

        for network_type, (color, label) in NETWORKS.items():
            ax2.plot(eta_values, data[f"{network_type}_a2"], color=color, linewidth=1.0, label=label)
            ax4.plot(eta_values, data[f"{network_type}_a4"], color=color, linewidth=1.0, label=label)
            ax6.plot(eta_values, data[f"{network_type}_a6"], color=color, linewidth=1.0, label=label)
            if network_type == "c1":
                ax2.plot(eta_values, data["c1_a2_from_c1"], color=color, linewidth=1.0,
                         linestyle="--", label=A2_FROM_C1_LABEL)

        ax2.set_ylabel(r"$\partial^2 \beta f / \partial S^2 \,|_{S=0}$")
        ax4.set_ylabel(r"$\partial^4 \beta f / \partial S^4 \,|_{S=0}$")
        ax6.set_ylabel(r"$\partial^6 \beta f / \partial S^6 \,|_{S=0}$")
        ax6.set_xlabel(r"$\eta$")
        for ax, letter in zip((ax2, ax4, ax6), "abc"):
            # panel letter just in from the figure's left edge, level with the top of the axes
            ax.text(0.04, 1.0, f"({letter})", ha="left", va="top",
                    transform=blended_transform_factory(fig.transFigure, ax.transAxes))
            ax.axhline(0.0, color="k", linewidth=0.5)
            ax.grid(True, linestyle="--", alpha=0.7)
        eta_jump = float(data["eta_jump"])
        for ax, value in zip((ax2, ax4, ax6), data["phi_at_jump"]):
            ax.plot(eta_jump, value, linestyle="none", marker="x", color="k", markersize=4,
                    markeredgewidth=0.75, zorder=4, label=r"$\Phi$-network jump in $|S|$")
        ax2.legend(loc="upper left", framealpha=0.0, handletextpad=0.3, borderpad=0.3,
                   labelspacing=0.3)

        # inset in the lower left of (c), which the curves leave empty
        inset = ax6.inset_axes([0.13, 0.14, 0.42, 0.33])
        # transparent=True clears every axes background; a white rectangle keeps the inset opaque
        inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes, facecolor="white",
                                  edgecolor="none", zorder=-1))
        draw_landau_comparison(inset, cfg, data["landau_eta"], data["landau_S"],
                               labels=(r"Landau, to $S^6$", r"$\Phi$-network"), marker_size=4)
        inset.set_xticks([0.57, 0.59, 0.61])
        inset.set_yticks([0.0, 0.5, 1.0])
        inset.tick_params(labelsize=5.5, length=2, pad=1.5)
        inset.xaxis.labelpad = 1
        inset.yaxis.labelpad = 1
        inset.xaxis.label.set_size(6)
        inset.yaxis.label.set_size(6)
        inset.legend(loc="upper left", fontsize=5, framealpha=0.0, handlelength=1.2,
                     handletextpad=0.3, borderpad=0.2, labelspacing=0.2)

        out_path = figure_path(figure_dir, f"k{cfg.k}_free_energy_derivatives_vs_eta.pdf")
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
