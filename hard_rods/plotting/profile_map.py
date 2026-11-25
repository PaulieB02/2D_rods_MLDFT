"""
One validation profile (Sec. 4.1): external field, GCMC density, a network's density and their
difference, for both species.
"""

import numpy as np
from matplotlib import pyplot as plt

from ..config import profile_path
from .style import figure_path, paper_style

color_map = "viridis"     # perceptually uniform, legible in greyscale
diff_color_map = "bwr"    # diverging: blue negative, white zero, red positive
vext_clip = 6             # hard walls are +inf, so clip for a usable colour scale

# superscript of the network's density in the panel titles
NETWORK_SYMBOL = {"phi": r"\Phi", "c1": r"\mathrm{dir}", "c1_regularised": r"\mathrm{reg}"}


def plot_profile_map(cfg, figure_dir, profile=3, network_type="phi"):
    """Draw k<k>_<network_type>_profile###.pdf into figure_dir."""
    validation = cfg.profile_dir("validation")
    u = np.load(profile_path(validation, "vext", profile))
    rho_mc = np.load(profile_path(validation, "rho", profile))
    rho_neural = np.load(profile_path(cfg.evaluation_dir(network_type), "rho", profile))
    net = NETWORK_SYMBOL[network_type]

    # row by row: external field, GCMC, network, difference; species 0 left, species 1 right
    panels = []
    for s in (0, 1):
        panels.append((rf"$\beta(V_{s} - \mu)$", np.minimum(u[s], vext_clip), color_map, False))
    for s in (0, 1):
        panels.append((rf"$\rho_{s}^\text{{GCMC}}$", rho_mc[s], color_map, False))
    for s in (0, 1):
        panels.append((rf"$\rho_{s}^{net}$", rho_neural[s], color_map, False))
    for s in (0, 1):
        panels.append((rf"$\rho_{s}^{net} - \rho_{s}^\text{{GCMC}}$",
                       rho_neural[s] - rho_mc[s], diff_color_map, True))

    with paper_style():
        # 5.0 x 7.2 in (360 x 518 pt), inside the double-column limits
        fig, axes = plt.subplots(4, 2, figsize=(5.0, 7.2), layout="constrained")
        for n, (title, data, cmap, center_zero) in enumerate(panels):
            ax = axes[n // 2][n % 2]
            ax.set_title(title)
            # a diverging map reads correctly only if zero sits at its white midpoint
            vmax = float(np.nanmax(np.abs(data))) or 1.0
            limits = {"vmin": -vmax, "vmax": vmax} if center_zero else {}
            # arrays are [species, y, x], so imshow with origin="lower" puts y up the page
            im = ax.imshow(data, origin="lower", cmap=cmap, **limits)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            # axis labels only on the outer edge
            if n // 2 == 3:
                ax.set_xlabel("$x$")
            if n % 2 == 0:
                ax.set_ylabel("$y$")

        out_path = figure_path(figure_dir, f"k{cfg.k}_{network_type}_profile{profile:03d}.pdf")
        fig.savefig(out_path, format="pdf", dpi=300)
        plt.close(fig)

    dev = np.abs(rho_neural - rho_mc)
    print(f"profile {profile}, {network_type}: mean |rho_network - rho_GCMC| = {dev.mean():.4e}, "
          f"max = {dev.max():.4e}")
    print(f"Saved: {out_path}")
    return out_path
