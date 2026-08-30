"""Figure 4: deployed performance, and the methodological base of the evidence.

Panel (a) reproduces the subgroup sensitivities reported for one regulatorily
cleared intracranial haemorrhage detector across 101,944 head CT examinations
at 17 facilities (manuscript Section 8).

Panel (b) reproduces the column totals of Table 2, the methodological
appraisal of the 52 studies scored in this review.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from style import (BG, BLUE, INK, MUTE, RED, bottom_legend, figure_dir,
                   panel_label, style)

OUT = os.path.join(figure_dir(os.path.dirname(os.path.abspath(__file__))), "Fig4_evidence.png")

# Deployed sensitivity by subgroup, in percent (Section 8).
SUBGROUPS = [
    ("Overall", 82.2, False),
    ("≤ 10 mm\nlesions", 74.8, False),
    ("Out-\npatients", 72.2, False),
    ("Chronic\nhaemorrhage", 54.8, True),   # the subgroup that fails
]

# Column totals of Table 2 (tab:quality), out of 52 appraised studies.
CRITERIA = [
    ("Sample size\nreported", 36),
    ("Reader\ncomparison", 7),
    ("External\nvalidation", 2),
    ("Patient-level\nsplit stated", 0),
    ("Prospective\ndesign", 0),
    ("Clinical\nendpoint", 0),
    ("Confidence\ninterval", 0),
]
N_STUDIES = 52


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # ---- (a) deployed sensitivity by subgroup ---------------------------
    names = [n for n, _, _ in SUBGROUPS]
    vals = [v for _, v, _ in SUBGROUPS]
    cols = [RED if flag else BLUE for _, _, flag in SUBGROUPS]
    bars = a.bar(names, vals, color=cols, width=0.62, zorder=3)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 1.6, f"{v:.1f}",
               ha="center", fontsize=12, fontweight="bold", color=INK)
    a.set_ylim(0, 100)
    a.set_ylabel("Sensitivity (%)")
    style(a)
    panel_label(a, "a", "Deployed sensitivity for intracranial haemorrhage")

    # ---- (b) methodological criteria met --------------------------------
    labels = [n for n, _ in CRITERIA][::-1]
    counts = [c for _, c in CRITERIA][::-1]
    # One series, one colour. The four zero rows carry the panel's point by
    # being zero; painting them a warning hue would colour by rank.
    bars = b.barh(labels, counts, color=BLUE, height=0.6, zorder=3)
    for bar, c in zip(bars, counts):
        b.text(c + 0.8, bar.get_y() + bar.get_height() / 2, f"{c}",
               va="center", fontsize=12, fontweight="bold", color=INK)
    b.set_xlim(0, N_STUDIES)
    b.set_xticks([0, 13, 26, 39, 52])
    b.set_xlabel(f"Studies meeting criterion (of {N_STUDIES})")
    style(b, grid_axis="x")
    panel_label(b, "b", "Methodological criteria met, 52 appraised studies")

    handles = [
        Patch(facecolor=BLUE, edgecolor="none",
              label="Reported value (a) / criterion met (b)"),
        Patch(facecolor=RED, edgecolor="none",
              label="Worst subgroup (a)"),
    ]
    bottom_legend(fig, handles, ncol=len(handles), y=0.005, fontsize=11)

    fig.tight_layout(pad=2.0, w_pad=3.6, rect=(0, 0.085, 1, 1))
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")
    print(f"  panel a  headline {vals[0]}%  worst subgroup {vals[-1]}%"
          f"  spread {vals[0] - vals[-1]:.1f} pts")
    print(f"  panel b  criteria met by no study: "
          f"{sum(1 for _, c in CRITERIA if c == 0)} of {len(CRITERIA)}")


if __name__ == "__main__":
    main()
