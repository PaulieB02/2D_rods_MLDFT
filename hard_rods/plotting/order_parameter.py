"""
Orientational order |S|(eta) (Sec. 4.4): (a) global minimum, (b) first-derivative method, with
the fits, the Phi-network's jumps and the GCMC critical packing fraction.
"""

import numpy as np
from matplotlib import pyplot as plt

from ..config import read_critical_packing_fraction
from ..evaluation import phase_diagram
from .style import SINGLE_COLUMN, figure_path, paper_style

# legend labels and markers, in the row order of the saved tables (Phi-network first)
SERIES_LABELS = (r"$\Phi$-network", r"direct-$c^{(1)}$ network")
STYLES = (
    {"marker": "^", "s": 7, "color": "tab:blue", "linewidths": 0, "zorder": 3},
    {"marker": "o", "s": 4, "color": "tab:red", "alpha": 0.7, "linewidths": 0, "zorder": 2},
)

# the Phi-network's jumps are drawn only below this eta (far outside the training range above)
JUMP_ETA_MAX = 0.8


def plot_order_parameter(cfg, figure_dir):
    """Draw k<k>_order_parameter_two_routes.pdf into figure_dir. Returns the path."""
    panels = (("Global Minimum Method", phase_diagram.global_minimum_path(cfg), "global"),
              ("First Derivative Method", phase_diagram.derivative_path(cfg), "derivative"))

    fits = phase_diagram.fit_all(cfg)
    eta_line = np.linspace(phase_diagram.eta_values[0], phase_diagram.ETA_FIT_MAX, 2000)
    eta_c = read_critical_packing_fraction(cfg.eta_c_file) if cfg.eta_c_file else None

    def draw_fit(ax, name, color):
        """A tanh fit as a thin line, a step (w = 0) as a dashed line at eta_c."""
        if name not in fits:
            return
        p = fits[name]
        if p["w"] == 0:
            ax.vlines(p["eta_c"], 0.0, p["a"] + p["b"], color=color, linestyle="--",
                      linewidth=0.75, zorder=1)
        else:
            ax.plot(eta_line, phase_diagram.tanh_model(eta_line, p["a"], p["b"], p["eta_c"], p["w"]),
                    color=color, linewidth=0.75, zorder=1)

    jumps = phase_diagram.phi_jumps(cfg, eta_max=JUMP_ETA_MAX)
    for eta, lower, upper in jumps:
        print(f"  Phi-network jump at eta = {eta:.4f}: |S| {lower:.3f} -> {upper:.3f}")

    with paper_style():
        fig, axes = plt.subplots(len(panels), 1, figsize=(SINGLE_COLUMN, 4.4), sharex=True,
                                 sharey=True, layout="constrained")
        fig.get_layout_engine().set(hspace=0.08)
        for ax, letter, (title, path, method) in zip(axes, "ab", panels):
            # rows come in (eta, |S|) pairs, one pair per network; the NaN padding is not drawn
            table = np.load(path)
            for series, style in enumerate(STYLES):
                ax.scatter(table[2 * series], table[2 * series + 1],
                           label=SERIES_LABELS[series], **style)

            draw_fit(ax, f"{method}_phi", "tab:blue")
            draw_fit(ax, f"{method}_c1", "tab:red")
            for eta, lower, upper in jumps:
                ax.vlines(eta, lower, upper, color="tab:blue", linestyle=(0, (1, 1)),
                          linewidth=0.75, zorder=1)
                if method == "global":
                    # the end of the jump that is only a local minimum is drawn hollow
                    S_global = table[1][np.isclose(table[0], eta)]
                    local = lower if np.abs(S_global - upper).min() < np.abs(S_global - lower).min() \
                        else upper
                    ax.scatter(eta, local, marker="^", s=7, facecolors="white",
                               edgecolors="tab:blue", linewidths=0.5, zorder=3)

            if eta_c is not None:
                ax.axvline(eta_c, color="k", linestyle="--", linewidth=0.75, zorder=1,
                           label=r"$\eta_\mathrm{c}^\mathrm{GCMC}$")

            ax.set_title(title)
            ax.set_title(f"({letter})", loc="left")
            ax.set_ylabel(r"$|S|$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.03, 1.0)
            ax.grid(True, linestyle="--", alpha=0.7)
        axes[-1].set_xlabel(r"$\eta$")

        # one legend, in the upper panel's top left corner (|S| = 0 below the transition)
        axes[0].legend(loc="upper left", framealpha=0.0, handletextpad=0.3, borderpad=0.3,
                       labelspacing=0.3)

        out_path = figure_path(figure_dir, f"k{cfg.k}_order_parameter_two_routes.pdf")
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
