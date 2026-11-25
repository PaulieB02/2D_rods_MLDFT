"""Histogram of the per-window mixed-derivative penalty (Appendix B) of the three networks."""

import numpy as np
from matplotlib import pyplot as plt

from ..evaluation.c2_symmetry import NETWORK_ORDER, c2_losses_path
from .style import SINGLE_COLUMN, figure_path, paper_style

COLORS_AND_LABELS = {"c1": ("tab:red", r"direct-$c^{(1)}$ network"),
                     "c1_regularised": ("tab:green", r"direct-$c^{(1)}$ network, regularized"),
                     "phi": ("tab:blue", r"$\Phi$-network")}
LOSS_LABEL = r"$\mathcal{L}_{c^{(2)}}$"

# TeX's calligraphic L for the loss symbol (cmsy10 ships with matplotlib; the sans family has none)
STYLE_OVERRIDES = {"mathtext.cal": "cmsy10"}


def plot_c2_histogram(cfg, figure_dir):
    """Draw k<k>_c2_symmetry_histogram.pdf into figure_dir. Returns the path."""
    data = np.load(c2_losses_path(cfg))
    losses = {network_type: data[network_type][:, 2] for network_type in NETWORK_ORDER}   # total

    # bins spanning every positive loss (a log axis has no room for zero)
    values = np.concatenate(list(losses.values()))
    values = values[values > 0]
    bins = np.logspace(*np.log10((values.min(), values.max())), 60)

    with paper_style(STYLE_OVERRIDES):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.4), layout="constrained")
        for network_type in NETWORK_ORDER:
            color, label = COLORS_AND_LABELS[network_type]
            ax.hist(losses[network_type], bins=bins, alpha=0.55, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel(f"per-window {LOSS_LABEL}")
        ax.set_ylabel("number of windows")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        # the (2) in L_c(2) is a second-level script; 9.5 pt keeps it above 4.5 pt
        ax.xaxis.label.set_fontsize(9.5)

        out_path = figure_path(figure_dir, f"k{cfg.k}_c2_symmetry_histogram.pdf")
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
