"""Matplotlib appearance of the figures, scoped to each plot (no global rcParams changes)."""
from functools import wraps

import matplotlib

# 12-inch figures and Matplotlib's original typography.
SAVE_DPI = 300
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.8",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.axisbelow": True,
    "grid.color": "#b0b0b0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def figure_style(fn):
    """Use these defaults while creating and saving one figure."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with matplotlib.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapped
