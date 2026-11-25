"""
Common figure style: final size for JCTC (single column 3.33 in, double column 7 in), 8 pt
Helvetica/Arial-compatible sans-serif lettering, TrueType fonts embedded in the PDFs. Every
figure is drawn inside paper_style(), with its own overrides where it deviates.
"""

import contextlib
import os

import matplotlib

SINGLE_COLUMN = 3.33     # inches
DOUBLE_COLUMN = 7.0      # inches

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    # no calligraphic face in this family: \mathcal falls back to italic sans
    "mathtext.cal": "sans:italic",
    "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
}


@contextlib.contextmanager
def paper_style(overrides=None):
    """Context in which a figure is created and saved: STYLE, updated by overrides."""
    with matplotlib.rc_context({**STYLE, **(overrides or {})}):
        yield


def figure_path(figure_dir, name):
    """Path of a figure file, creating the folder."""
    os.makedirs(figure_dir, exist_ok=True)
    return os.path.join(figure_dir, name)
