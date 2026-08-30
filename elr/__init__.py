"""elr: evidence-linked reporting.

A reference implementation of the evaluation standard and interoperability
substrate specified in "From Pixels to Patterns: Deep Learning and Natural
Language Processing in Neuroradiology" (Section 6).

This package deliberately contains no report-generation model. The manuscript's
argument is that the field's bottleneck is evaluation and accountability rather
than architecture, and shipping another unvalidated generator alongside that
argument would contradict it. What is provided instead is the tooling a study
needs in order to report its results in a way that can be compared against
another study's.
"""

from .model import (
    DEFAULT_SEVERITY_WEIGHTS,
    Finding,
    Measurement,
    Region,
    Report,
)
from .metrics import (
    bootstrap_ci,
    comparator_panel,
    consequence_weighted_errors,
    faithfulness_by_intervention,
    finding_level_agreement,
    image_blind_delta,
    match_findings,
    reader_study_summary,
    reference_standard_ceiling,
    risk_coverage,
    unsupported_fraction,
)

__version__ = "0.1.0"

__all__ = [
    "Finding", "Measurement", "Region", "Report", "DEFAULT_SEVERITY_WEIGHTS",
    "match_findings", "finding_level_agreement", "consequence_weighted_errors",
    "image_blind_delta", "comparator_panel", "faithfulness_by_intervention",
    "risk_coverage", "reference_standard_ceiling", "unsupported_fraction",
    "reader_study_summary", "bootstrap_ci",
]
