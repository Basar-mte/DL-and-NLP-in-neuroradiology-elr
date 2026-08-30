"""Core data model for evidence-linked radiology reporting.

A generated report is not a string. It is a set of findings, each of which
carries a coded concept, the image evidence that supports it, and optionally a
measurement. This module defines that structure, because every metric in
``elr.metrics`` is computed over findings rather than over text.

The design commitment stated in Section 7.1 of the manuscript is that a
globally pooled latent carries no spatial index and therefore cannot support
evidence linkage. The consequence at the data level is this: a Finding without
a ``region`` is not evidence-linked, and the metrics treat it as ungrounded
rather than silently scoring it as if it were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Consequence weights for omission, keyed by severity. These are the
# "clinical severity" weights required by requirement 4 of the evaluation
# standard: missing a developing herniation and missing a chronic
# microvascular change are not commensurable errors, so they must not be
# averaged as if they were.
#
# These particular values are a DEFAULT, not a validated instrument. Any
# study using this harness must state the weights it used and justify them;
# the harness records them in its output so that they cannot be left implicit.
DEFAULT_SEVERITY_WEIGHTS = {
    "critical": 10.0,   # actionable within minutes (herniation, acute large-vessel occlusion)
    "urgent": 5.0,      # actionable within hours (acute haemorrhage, abscess)
    "significant": 2.0, # changes management but not emergently (new mass lesion)
    "routine": 1.0,     # documented, rarely changes management (chronic microvascular change)
}

VALID_SEVERITIES = tuple(DEFAULT_SEVERITY_WEIGHTS)


@dataclass(frozen=True)
class Region:
    """A reference to the image evidence supporting a finding.

    ``frame`` identifies the slice or frame within the referenced series;
    ``bbox`` is (x0, y0, x1, y1) in pixel coordinates on that frame.
    ``series_uid`` ties the region back to the acquisition it came from, which
    is what makes the reference resolvable in PACS rather than decorative.
    """

    series_uid: str
    frame: int
    bbox: tuple[float, float, float, float]

    def iou(self, other: "Region") -> float:
        """Intersection-over-union with another region.

        Returns 0.0 when the regions are on different series or frames: a
        correct-looking box on the wrong slice is not partial credit, it is a
        different location.
        """
        if self.series_uid != other.series_uid or self.frame != other.frame:
            return 0.0
        ax0, ay0, ax1, ay1 = self.bbox
        bx0, by0, bx1, by1 = other.bbox
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Measurement:
    """A quantitative value retained as a number, not as prose.

    Requirement of Section 7.3: measurements must survive into the structured
    report as machine-readable values, so that a follow-up rule can act on
    them without re-parsing the narrative.
    """

    name: str
    value: float
    unit: str
    ucum: str = ""  # UCUM code, e.g. "mm" -> "mm", "mL" -> "mL"


@dataclass
class Finding:
    """One assertion in a report, with the evidence that supports it."""

    code: str                       # SNOMED CT concept id
    display: str                    # human-readable concept name
    text: str                       # the sentence as it appears in the report
    severity: str = "routine"
    region: Optional[Region] = None
    measurement: Optional[Measurement] = None
    confidence: Optional[float] = None   # model's own probability, for calibration
    absent: bool = False            # explicit negation ("no acute haemorrhage")

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {VALID_SEVERITIES}, got {self.severity!r}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence}")

    @property
    def grounded(self) -> bool:
        """True when the finding carries a resolvable image reference.

        An ungrounded finding may still be correct. It is simply not auditable,
        which is the property the whole formulation exists to provide.
        """
        return self.region is not None


@dataclass
class Report:
    """A report as a structured artefact rather than a block of text."""

    study_uid: str
    findings: list[Finding] = field(default_factory=list)
    narrative: str = ""
    abstained: bool = False   # system declined to report (requirement 6)

    def positive(self) -> list[Finding]:
        """Findings actually asserted, excluding explicit negations."""
        return [f for f in self.findings if not f.absent]

    def by_code(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.positive():
            out.setdefault(f.code, []).append(f)
        return out

    @property
    def grounded_fraction(self) -> float:
        pos = self.positive()
        if not pos:
            return float("nan")
        return sum(f.grounded for f in pos) / len(pos)
