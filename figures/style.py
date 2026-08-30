"""Shared figure style for the manuscript.

Palette sampled from the original artwork so regenerated figures sit beside
the untouched ones without a visible seam:

    blue   #2A78D6   primary data
    orange #EB6834   second category
    red    #E34948   the value the reader must not miss
    grey   #898781   recessive / comparator
    ink    #0B0B0B   text
    bg     #FCFCFB   surface

Orange and red are never used as adjacent categories in one panel: they
separate by only dE 7.1 at normal vision, below the dE 15 floor at which
full-colour readers can still tell a pair apart. Where both appear in a figure
they carry unrelated roles in different panels, and the bottom legend names
each one.

House rule for these figures: no prose annotation inside the axes. Identity is
carried by a legend beneath the panels; interpretation belongs to the figure
legend in the manuscript, which has room to qualify it properly.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2A78D6"
ORANGE = "#EB6834"
RED = "#E34948"
MUTE = "#898781"
INK = "#0B0B0B"
SUB = "#52514E"
GRID = "#E1E0D9"
BG = "#FCFCFB"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "axes.edgecolor": MUTE,
    "axes.linewidth": 0.9,
    "font.size": 11,
})


def style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter, title, pad=11, letter_x=-0.085):
    """Panel letter and heading sharing one baseline.

    Both are drawn with the same offset above the axes and va="baseline", so
    the letter and the heading sit on exactly the same line however their font
    sizes differ. Setting the letter's y in axes coordinates while the heading
    comes from set_title puts them on two different baselines, which is what
    an earlier version did.
    """
    ax.annotate(letter, xy=(letter_x, 1.0), xycoords="axes fraction",
                xytext=(0, pad), textcoords="offset points",
                fontsize=13.5, fontweight="bold", va="baseline", ha="left",
                color=INK, annotation_clip=False)
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, pad), textcoords="offset points",
                fontsize=12.5, fontweight="bold", va="baseline", ha="left",
                color=INK, annotation_clip=False)


def bottom_legend(fig, handles, ncol=2, y=0.005, fontsize=11.5):
    """One legend for the whole figure, beneath every panel."""
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=fontsize, handlelength=1.8, handleheight=1.0,
               columnspacing=2.6, labelspacing=0.7,
               bbox_to_anchor=(0.5, y))
