"""Format consistent Matplotlib figures.

Import style() at the top of any script.
"""
import matplotlib as plt

# Readable and pretty colors for the plots :)
INK      = "#1b2a3a"
BLUE     = "#1a4d7a" 
ORANGE   = "#c9772e"
RED       = "#8a2846"
GREEN    = "#2e7d4f"
GREY     = "#9aa3ab"

# Call this once at the top of a script to give every figure the same fonts, sizes, and grid.
# It mutates matplotlib's global rcParams rather than returning anything, so the settings apply to
# every plot made afterward — that's what keeps the four result figures looking like one set.
def style():
    """Apply the shared Matplotlib look (fonts, sizes, grid) for all figures.

    Returns:
        None. Mutates the global matplotlib rcParams in place.
    """
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
