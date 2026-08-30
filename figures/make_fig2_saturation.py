"""Figure 2: evidence for benchmark exhaustion in brain tumour classification.

Regenerated from the values tabulated in Table 3 of the manuscript, so the
figure and the table cannot disagree. Every point below is traceable to a row
of that table; nothing is estimated from the previous rendering.

Panel (b) is computed, not tabulated: the Wilson score interval width at the
stated accuracy and test-set size.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from style import (BG, BLUE, GRID, INK, ORANGE, RED, bottom_legend,
                   figure_dir, panel_label, style)

OUT = os.path.join(figure_dir(os.path.dirname(os.path.abspath(__file__))), "Fig2_saturation.png")

# (dataset size, accuracy, study key) -- all from Table 3, tab:braintumour.
# Split by the UNIT the dataset size is quoted in, which is the point of the
# panel: 3264 images is not 3264 patients, and comparing the two as though
# they were the same quantity is part of how the literature reads as saturated.
SUBJECTS = [
    (66,   98.5,  "arif2022brain"),        # Private, 66 patients
    (338,  96.55, "wozniak2021deep"),      # Kaggle, 338 patients
    (1016, 83.0,  "li2022molecular"),      # Private, 1016 subjects
]
IMAGES = [
    (253,  90.0,  "sharma2022enhanced"),   # Kaggle, 253 images
    (3064, 99.51, "noreen2020deep"),       # Figshare/CE-MRI, 3064 slices
    (3064, 99.67, "raza2022hybrid"),       # CE-MRI (Cheng), 3064 images
    (3264, 92.13, "khan2022intelligent"),  # Kaggle, 3264 images
]


def wilson_width(p: float, n: int, z: float = 1.96) -> float:
    """Width of the Wilson score interval, in percentage points."""
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = centre - half, centre + half
    return (hi - lo) * 100


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 5.3))

    # ---- (a) reported accuracy against reported dataset size ------------
    a.scatter([x for x, _, _ in SUBJECTS], [y for _, y, _ in SUBJECTS],
              s=170, marker="o", color=BLUE, edgecolor=BG, linewidth=1.4,
              zorder=3)
    a.scatter([x for x, _, _ in IMAGES], [y for _, y, _ in IMAGES],
              s=170, marker="s", color=ORANGE, edgecolor=BG, linewidth=1.4,
              zorder=3)
    a.set_xscale("log")
    a.set_xticks([50, 100, 250, 500, 1000, 3000])
    a.set_xticklabels(["50", "100", "250", "500", "1,000", "3,000"])
    a.set_xlim(45, 4600)
    a.set_ylim(78, 102)
    a.set_xlabel("Reported dataset size (log scale)")
    a.set_ylabel("Reported accuracy (%)")
    style(a)
    panel_label(a, "a", "Reported accuracy against reported dataset size")

    # ---- (b) sampling uncertainty at these sample sizes -----------------
    ns = list(range(20, 1001))
    b.plot(ns, [wilson_width(0.95, n) for n in ns], color=BLUE, lw=2.4, zorder=3)
    b.plot(ns, [wilson_width(0.99, n) for n in ns], color=ORANGE, lw=2.4, zorder=3)
    b.axhline(2.0, color=RED, lw=1.8, ls=(0, (4, 3)), zorder=4)
    b.set_xscale("log")
    b.set_xticks([20, 50, 100, 250, 500, 1000])
    b.set_xticklabels(["20", "50", "100", "250", "500", "1,000"])
    # A log axis keeps its own minor ticks, which here print 3x10^1, 4x10^1 ...
    # straight through the chosen labels.
    b.xaxis.set_minor_formatter(NullFormatter())
    b.tick_params(axis="x", which="minor", length=0)
    b.set_xlim(20, 1000)
    b.set_ylim(0, 24)
    b.set_xlabel("Test-set size (log scale)")
    b.set_ylabel("Width of 95% Wilson interval (pts)")
    style(b)
    panel_label(b, "b", "Uncertainty at these sample sizes")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markersize=11, label="Patients / subjects (a)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE,
               markersize=11, label="Images / slices (a)"),
        Line2D([0], [0], color=BLUE, lw=2.4, label="95% accuracy (b)"),
        Line2D([0], [0], color=ORANGE, lw=2.4, label="99% accuracy (b)"),
        Line2D([0], [0], color=RED, lw=1.8, ls=(0, (4, 3)),
               label="2-point difference (b)"),
    ]
    bottom_legend(fig, handles, ncol=len(handles), y=0.005, fontsize=11)

    fig.tight_layout(pad=2.0, w_pad=3.4, rect=(0, 0.10, 1, 1))
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")
    for n in (66, 253, 1016, 3264):
        print(f"  Wilson width at 95% acc, n={n:>5}: {wilson_width(0.95, n):5.2f} pts")


if __name__ == "__main__":
    main()
