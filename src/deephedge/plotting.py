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

# pretty fonts and figure sizes.
def style():
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
