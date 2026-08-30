"""Generate the manuscript figure from the harness output.

Run after worked_example.py:  python examples/make_figure.py

Reads examples/output/results.json and writes Fig3_evaluation.png into the
manuscript directory. The figure is generated from the run rather than
transcribed from it, so the panel values cannot drift from the code that
produced them.

Palette matches the manuscript's existing figures: blue #2A78D6 carries the
data, red #E34948 is reserved for the single value the reader must not miss,
and everything else is recessive grey. Orange (#EB6834, used elsewhere in the
paper) is deliberately not used here: against #E34948 it fails a normal-vision
separation check at dE 7.1, and these panels put the two side by side.

No prose annotation is drawn inside the axes. Identity is carried by a single
legend beneath all four panels, and interpretation by the figure legend in the
manuscript. That constraint forces the colours to mean ONE thing across the
whole figure, which an earlier draft did not manage: red marked the grounded
score in (a), critical severity in (b) and perfect agreement in (c), three
unrelated referents sharing a hue. The three roles below are now uniform, so a
reader who learns the legend once can read every panel:

    grey  comparator or reference-standard ceiling
    blue  the system under evaluation
    red   the quantity conventional reporting leaves out

Numeric values stay on the marks. Those are data, not annotation, and a bar
chart in a journal is read for its values.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "output", "results.json")

sys.path.insert(0, os.path.join(HERE, ".."))
from elr.paths import figure_dir
OUT = os.path.join(figure_dir(HERE), "Fig3_evaluation.png")

BLUE = "#2A78D6"
RED = "#E34948"
INK = "#0B0B0B"
SUB = "#52514E"
GRID = "#E1E0D9"
MUTE = "#898781"
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


def main() -> None:
    with open(RESULTS, encoding="utf-8") as fh:
        R = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    (a, b), (c, d) = axes

    # ---- (a) what requiring evidence costs ------------------------------
    g = R["grounded"]
    ib = R["image_blind"]
    names = ["Image-blind\nbaseline", "Model,\nconcept level", "Model,\ngrounded"]
    vals = [ib["image_blind_f1"], g["concept_f1"], g["f1"]]
    cols = [MUTE, BLUE, RED]
    bars = a.bar(names, vals, color=cols, width=0.6, zorder=3)
    # CI on the grounded bar only: it is the estimate the argument rests on
    a.errorbar(2, g["f1"], yerr=[[g["f1"] - g["ci_lo"]], [g["ci_hi"] - g["f1"]]],
               fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=4)
    # The third bar carries a CI, so its label clears the upper cap rather
    # than the bar top; otherwise the whisker strikes through the digits.
    tops = [vals[0], vals[1], g["ci_hi"]]
    for bar, v, top in zip(bars, vals, tops):
        a.text(bar.get_x() + bar.get_width() / 2, top + 0.026, f"{v:.3f}",
               ha="center", fontsize=12, fontweight="bold", color=INK)
    a.set_ylim(0, 1.0)
    a.set_ylabel("Finding-level F1")
    style(a)
    panel_label(a, "a", "What evidence-linking costs")

    # ---- (b) omission by clinical severity ------------------------------
    sev = R["severity"]
    order = ["critical", "urgent", "significant", "routine"]
    rates, labels, counts = [], [], []
    for s in order:
        n, om = sev[s]["reference"], sev[s]["omitted"]
        rates.append(om / n if n else 0.0)
        counts.append(n)
        labels.append(s.capitalize())
    # One series, one colour. Painting the critical bar red would colour by
    # rank rather than by entity, and would break the shared legend: red
    # already means "the quantity conventional reporting leaves out".
    # Critical is the longest bar regardless, so the asymmetry still reads.
    bars = b.barh(labels[::-1], rates[::-1], color=BLUE,
                  height=0.62, zorder=3)
    for bar, v, n in zip(bars, rates[::-1], counts[::-1]):
        b.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
               f"{v:.3f}   (n={n})", va="center", fontsize=11,
               fontweight="bold", color=INK)
    b.set_xlim(0, max(rates) + 0.22)
    b.set_xlabel("Proportion of reference findings omitted")
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    style(b, grid_axis="x")
    panel_label(b, "b", "Omission by clinical severity")
    # No in-panel caption here. The earlier draft asserted that the most
    # consequential findings are missed most often, which the routine row
    # (0.417, inflated by reference content carrying no image evidence)
    # contradicts. The qualification belongs in the figure legend, where
    # there is room to state it accurately.

    # ---- (c) the ceiling ------------------------------------------------
    ce = R["ceiling"]
    ypos = [1, 0]
    c.barh(ypos, [ce["inter_reader_f1"], ce["model_f1"]],
           color=[MUTE, BLUE], height=0.42, zorder=3)
    c.set_yticks(ypos)
    c.set_yticklabels(["Two radiologists,\nagainst each other", "Model,\ngrounded"])
    for y, v in zip(ypos, [ce["inter_reader_f1"], ce["model_f1"]]):
        c.text(v - 0.02, y, f"{v:.3f}", va="center", ha="right",
               fontsize=12, fontweight="bold", color="white")
    # The dotted line at perfect agreement carries the panel's point without a
    # caption: the grey ceiling bar stops well short of it, so the distance a
    # paper would report as "remaining gap" is visibly the wrong distance.
    c.axvline(1.0, color=RED, lw=1.6, ls=(0, (2, 2)), zorder=4)
    c.set_xlim(0, 1.06)
    c.set_ylim(-0.55, 1.55)
    c.set_xlabel("Finding-level F1")
    style(c, grid_axis="x")
    panel_label(c, "c", "The ceiling the reference standard permits")

    # ---- (d) risk-coverage ----------------------------------------------
    rc = sorted(R["risk_coverage"], key=lambda p: p["coverage"])
    cov = [p["coverage"] for p in rc]
    risk = [p["risk"] for p in rc]
    d.plot(cov, risk, color=BLUE, lw=2.0, marker="o", markersize=6,
           markerfacecolor=BLUE, markeredgecolor=BG, markeredgewidth=1.4,
           zorder=3)
    # The no-abstention operating point: where a system that cannot decline is
    # forced to sit, and the only point on this curve a conventional report
    # would show.
    full = max(rc, key=lambda p: p["coverage"])
    d.scatter([full["coverage"]], [full["risk"]], s=170, color=RED,
              edgecolor=BG, linewidth=1.6, zorder=5)
    d.set_xlabel("Coverage (proportion of findings asserted)")
    d.set_ylabel("Risk (error rate among asserted)")
    d.set_xlim(0, 1.05)
    d.set_ylim(0, 1.0)
    style(d, grid_axis="both")
    panel_label(d, "d", "Risk against coverage when the system may decline")

    # ---- shared legend, beneath all four panels -------------------------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=MUTE, edgecolor="none",
              label="Comparator / ceiling"),
        Patch(facecolor=BLUE, edgecolor="none",
              label="System under evaluation"),
        Patch(facecolor=RED, edgecolor="none",
              label="Conventional reporting omits"),
        Line2D([0], [0], color=RED, lw=1.6, ls=(0, (2, 2)),
               label="Perfect agreement (c)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=11, handlelength=1.6,
               handleheight=1.0, columnspacing=2.6, labelspacing=0.75,
               bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(pad=2.0, w_pad=3.4, h_pad=3.6, rect=(0, 0.062, 1, 1))
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")
    print(f"  panel a  cost of evidence  {g['cost_of_evidence']:.3f}")
    print(f"  panel b  critical omission {rates[0]:.3f}")
    print(f"  panel c  true gap          {ce['distance_to_ceiling']:.3f}"
          f"  vs apparent {ce['apparent_gap_to_one']:.3f}")
    print(f"  panel d  risk at full cov  {full['risk']:.3f}")


if __name__ == "__main__":
    main()
