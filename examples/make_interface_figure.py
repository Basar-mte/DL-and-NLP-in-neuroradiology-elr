"""Figure: the interface the formulation requires, on a real image.

One synthetic demonstration finding is emitted through the actual DICOM SR and
FHIR emitters, and the three artefacts are shown side by side: the referenced
image region, the SR content items, and the FHIR Observation. Panels (b) and
(c) are rendered from the emitted objects, not typed, so the figure cannot
drift from what the code produces.

The background image is a T1-weighted axial slice of a healthy adult volunteer
from OpenNeuro dataset ds000228 (CC0; see assets/ATTRIBUTION.md). The
annotation is the synthetic demonstration finding of the worked example and
does not describe this image or this person.

Run:  python examples/make_interface_figure.py   (writes Fig3_interface.png)
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from elr import Finding, Measurement, Region, Report
from elr.emit import build_fhir, build_sr

BLUE = "#2A78D6"
RED = "#E34948"
INK = "#0B0B0B"
SUB = "#52514E"
MUTE = "#898781"
PANEL = "#F4F3EE"
BG = "#FCFCFB"

ASSET = os.path.join(HERE, "assets", "t1w_ds000228_sub-pixar155_ax148.png")

# The demonstration finding. Region coordinates are in the pixel space of the
# displayed slice and are drawn verbatim in panel (a), so the numbers the SR
# carries are the numbers the reader sees.
REGION = Region(series_uid="1.2.826.0.1.99", frame=148,
                bbox=(118.0, 96.0, 158.0, 150.0))
FINDING = Finding(
    code="21454007", display="Subarachnoid haemorrhage",
    text="Acute subarachnoid haemorrhage in the left sylvian fissure.",
    severity="critical", region=REGION,
    measurement=Measurement("Lesion diameter", 21.4, "mm", "mm"),
    confidence=0.91,
)
REPORT = Report(study_uid="1.2.826.0.1.3680043.8.498.11111",
                narrative=FINDING.text, findings=[FINDING])


def sr_lines(ds) -> list[str]:
    """Walk the emitted SR dataset and render its content items as text."""
    out = ["Comprehensive 3D SR  (SOP class ...88.34)",
           "CONTAINER  Radiology Report"]
    for grp in ds.ContentSequence:
        vt = getattr(grp, "ValueType", "")
        if vt == "TEXT":
            continue
        if vt != "CONTAINER":
            continue
        items = list(grp.ContentSequence)
        for i, it in enumerate(items):
            branch = "└" if i == len(items) - 1 else "├"
            ivt = it.ValueType
            name = it.ConceptNameCodeSequence[0].CodeMeaning
            if ivt == "CODE":
                v = it.ConceptCodeSequence[0]
                out.append(f" {branch} CODE    {name}: "
                           f"{v.CodingSchemeDesignator} {v.CodeValue} "
                           f"{v.CodeMeaning}")
            elif ivt == "TEXT":
                out.append(f" {branch} TEXT    {name}:")
                out.append(f" │          “{it.TextValue}”")
            elif ivt == "NUM":
                mv = it.MeasuredValueSequence[0]
                unit = mv.MeasurementUnitsCodeSequence[0].CodeValue
                shown = "" if unit == "1" else f" {unit}"
                out.append(f" {branch} NUM     {name}: {mv.NumericValue}{shown}")
            elif ivt == "SCOORD":
                pts = len(it.GraphicData) // 2
                ref = it.ContentSequence[0].ReferencedSOPSequence[0]
                out.append(f" {branch} SCOORD  {it.GraphicType}, {pts} points")
                out.append(f"            → IMAGE series {ref.ReferencedSOPInstanceUID},"
                           f" frame {ref.ReferencedFrameNumber}")
    return out


def fhir_lines(bundle) -> list[str]:
    """A compact excerpt, every value read from the emitted Observation."""
    obs = bundle["entry"][1]["resource"]
    c = obs["code"]["coding"][0]
    q = obs["valueQuantity"]
    d = obs["derivedFrom"][0]
    return [
        "FHIR Observation (excerpt)",
        '{',
        f' "resourceType": "{obs["resourceType"]}",',
        ' "code": {"coding": [{',
        f'   "system": "{c["system"]}",',
        f'   "code": "{c["code"]}", "display": "{c["display"]}"}}]}},',
        f' "valueQuantity": {{"value": {q["value"]}, "unit": "{q["unit"]}",'
        f' "code": "{q["code"]}"}},',
        ' "derivedFrom": [{',
        f'   "reference": "{d["reference"]}",',
        f'   "display": "{d["display"]}"}}]',
        '}',
    ]


def text_panel(ax, title, lines, fontsize=8.6):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(MUTE)
        sp.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.965, title, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="top", color=INK, family="DejaVu Sans")
    ax.text(0.03, 0.885, "\n".join(lines[1:]), transform=ax.transAxes,
            fontsize=fontsize, va="top", family="DejaVu Sans Mono",
            color=SUB, linespacing=1.42)


def main() -> None:
    img = Image.open(ASSET)
    w, h = img.size

    fig = plt.figure(figsize=(12.8, 5.4), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.55],
                          left=0.035, right=0.985, top=0.90, bottom=0.045,
                          hspace=0.16, wspace=0.10)
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 1])

    # (a) the referenced image region
    a.imshow(img, cmap="gray", vmin=0, vmax=255)
    x0, y0, x1, y1 = REGION.bbox
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                          edgecolor=BLUE, linewidth=2.2))
    a.text(x0, y0 - 5, "1", fontsize=11, fontweight="bold", color=BG,
           ha="center", va="bottom",
           bbox=dict(boxstyle="circle,pad=0.32", fc=BLUE, ec="none"))
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(MUTE)
    a.set_title("a   Referenced image region", fontsize=11.5,
                fontweight="bold", loc="left", color=INK, pad=8)

    ds = build_sr(REPORT, verifying_observer="Radiologist^Verifying")
    bundle = build_fhir(REPORT)

    text_panel(b, "b   DICOM Comprehensive 3D SR — content items",
               [""] + sr_lines(ds)[1:], fontsize=8.4)
    text_panel(c, "c   HL7 FHIR Observation (excerpt) — same finding, same codes",
               [""] + fhir_lines(bundle)[1:], fontsize=8.4)

    out_dir = os.environ.get("ELR_FIGURE_DIR")
    if not out_dir:
        probe = HERE
        for _ in range(4):
            probe = os.path.dirname(probe)
            if os.path.isfile(os.path.join(probe, "main.tex")):
                out_dir = probe
                break
    if not out_dir:
        out_dir = os.path.join(HERE, "output")
        os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Fig3_interface.png")
    fig.savefig(out, dpi=600, facecolor=BG)
    print("wrote", out)
    print("  SR content lines :", len(sr_lines(ds)))
    print("  region drawn     :", REGION.bbox, "frame", REGION.frame)


if __name__ == "__main__":
    main()
