"""Tests for the evaluation harness.

Run:  python -m pytest tests/ -q      (or: python tests/test_metrics.py)

These are behavioural tests for the properties the manuscript claims the
metrics have. Two of them exist specifically because the first draft of this
harness got them wrong.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elr import (
    Finding,
    Measurement,
    Region,
    Report,
    consequence_weighted_errors,
    faithfulness_by_intervention,
    finding_level_agreement,
    image_blind_delta,
    match_findings,
    reader_study_summary,
    risk_coverage,
    unsupported_fraction,
)

R1 = Region("s1", 5, (10.0, 10.0, 30.0, 30.0))
R1_NEAR = Region("s1", 5, (12.0, 12.0, 32.0, 32.0))     # high IoU
R1_FAR = Region("s1", 5, (100.0, 100.0, 120.0, 120.0))  # zero IoU
R1_OTHER_SLICE = Region("s1", 9, (10.0, 10.0, 30.0, 30.0))


def f(code="21454007", region=R1, severity="critical", **kw) -> Finding:
    return Finding(code=code, display="Test finding", text="Test.",
                   severity=severity, region=region, **kw)


def rep(*findings) -> Report:
    return Report(study_uid="1.2.3", findings=list(findings))


# -- region geometry -------------------------------------------------------

def test_iou_same_region_is_one():
    assert R1.iou(R1) == 1.0


def test_iou_zero_across_slices():
    """A correct-looking box on the wrong slice is not partial credit."""
    assert R1.iou(R1_OTHER_SLICE) == 0.0


def test_iou_zero_across_series():
    other = Region("s2", 5, (10.0, 10.0, 30.0, 30.0))
    assert R1.iou(other) == 0.0


# -- matching --------------------------------------------------------------

def test_right_concept_wrong_place_is_not_a_hit():
    """The property that separates this from ordinary label matching."""
    r = finding_level_agreement(rep(f(region=R1)), rep(f(region=R1_FAR)))
    assert r.hits == 0
    assert r.omissions == 1
    assert r.fabrications == 1


def test_right_concept_right_place_is_a_hit():
    r = finding_level_agreement(rep(f(region=R1)), rep(f(region=R1_NEAR)))
    assert r.hits == 1
    assert r.omissions == 0 and r.fabrications == 0
    assert r.mean_iou > 0.5


def test_ungrounded_candidate_fails_grounded_matching():
    r = finding_level_agreement(rep(f(region=R1)), rep(f(region=None)))
    assert r.hits == 0


def test_ungrounded_candidate_matches_at_concept_level():
    r = finding_level_agreement(rep(f(region=R1)), rep(f(region=None)),
                                require_grounding=False)
    assert r.hits == 1


def test_mean_iou_is_nan_not_one_for_ungrounded_matches():
    """Regression: ungrounded matching used to report a fabricated IoU of 1.0,
    which made the column meaningless exactly where honesty matters."""
    r = finding_level_agreement(rep(f(region=R1)), rep(f(region=None)),
                                require_grounding=False)
    assert r.hits == 1
    assert math.isnan(r.mean_iou)


def test_grounded_pairing_preferred_over_ungrounded():
    ref = rep(f(region=R1))
    cand = rep(f(region=None), f(region=R1_NEAR))
    matches = [m for m in match_findings(ref, cand, require_grounding=False) if m.is_hit]
    assert len(matches) == 1
    assert matches[0].candidate.region is not None


# -- image-blind baseline --------------------------------------------------

def test_image_blind_baseline_is_scored_at_concept_level():
    """Regression: scoring the blind arm with grounding required forced it to
    zero by construction, manufacturing a delta regardless of the image."""
    refs = [rep(f(region=R1))]
    with_img = [rep(f(region=R1_NEAR))]
    blind = [rep(f(region=None))]          # a blind decoder cannot localise
    res = image_blind_delta(refs, with_img, blind)
    assert res.without_image.f1 == 1.0     # it named the finding correctly
    assert res.delta_f1 == 0.0
    assert res.uses_image is False


def test_image_blind_delta_positive_when_image_helps():
    refs = [rep(f(code="A"), f(code="B"))]
    with_img = [rep(f(code="A"), f(code="B"))]
    blind = [rep(f(code="A"))]
    res = image_blind_delta(refs, with_img, blind)
    assert res.delta_f1 > 0
    assert res.uses_image is True


def test_image_blind_requires_equal_lengths():
    try:
        image_blind_delta([rep()], [rep()], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")


# -- consequence weighting -------------------------------------------------

def test_critical_omission_outweighs_routine_omission():
    ref = rep(f(code="CRIT", severity="critical"),
              f(code="ROUT", severity="routine"))
    miss_critical = consequence_weighted_errors(ref, rep(f(code="ROUT", severity="routine")))
    miss_routine = consequence_weighted_errors(ref, rep(f(code="CRIT", severity="critical")))
    assert miss_critical.weighted_omission > miss_routine.weighted_omission
    assert miss_critical.weighted_omission == 10.0
    assert miss_routine.weighted_omission == 1.0


def test_weights_are_returned_with_the_score():
    """An unpublished weight table makes a weighted rate irreproducible."""
    c = consequence_weighted_errors(rep(f()), rep(f()))
    assert "critical" in c.weights and c.weights["critical"] == 10.0


def test_custom_weights_are_honoured():
    ref = rep(f(severity="critical"))
    c = consequence_weighted_errors(ref, rep(), weights={"critical": 99.0})
    assert c.weighted_omission == 99.0


# -- faithfulness ----------------------------------------------------------

def test_unchanged_statement_under_masking_is_unfaithful():
    r = rep(f())
    res = faithfulness_by_intervention(r, regenerate=lambda x: x)
    assert res.tested == 1
    assert res.faithfulness_rate == 0.0


def test_withdrawn_statement_under_masking_is_faithful():
    r = rep(f())
    res = faithfulness_by_intervention(r, regenerate=lambda x: None)
    assert res.faithfulness_rate == 1.0


def test_ungrounded_findings_are_not_faithfulness_testable():
    r = rep(f(region=None))
    res = faithfulness_by_intervention(r, regenerate=lambda x: None)
    assert res.tested == 0


# -- abstention ------------------------------------------------------------

def test_coverage_falls_as_threshold_rises():
    refs = [rep(f(code="A"), f(code="B"))]
    cand = [rep(f(code="A", confidence=0.9), f(code="B", confidence=0.2))]
    pts = risk_coverage(refs, cand, thresholds=[0.0, 0.5])
    assert pts[0].coverage > pts[1].coverage


# -- reference standard ----------------------------------------------------

def test_unsupported_fraction_counts_ungrounded_content():
    reports = [rep(f(region=R1), f(region=None))]
    assert unsupported_fraction(reports) == 0.5


# -- reader study ----------------------------------------------------------

def test_automation_bias_rate():
    accepted = [True, True, False, True]
    correct = [True, False, False, False]
    rs = reader_study_summary(accepted, correct)
    assert rs.incorrect_statements_shown == 3
    assert rs.incorrect_statements_accepted == 2
    assert abs(rs.automation_bias_rate - 2 / 3) < 1e-9


def test_reader_study_rejects_mismatched_lengths():
    try:
        reader_study_summary([True], [True, False])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# -- model validation ------------------------------------------------------

def test_invalid_severity_rejected():
    try:
        Finding(code="X", display="X", text="X", severity="catastrophic")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown severity")


def test_confidence_out_of_range_rejected():
    try:
        Finding(code="X", display="X", text="X", confidence=1.4)
    except ValueError:
        return
    raise AssertionError("expected ValueError on out-of-range confidence")


def test_negated_findings_excluded_from_positives():
    r = rep(f(), Finding(code="Z", display="Z", text="No Z.", absent=True))
    assert len(r.positive()) == 1


if __name__ == "__main__":
    ns = dict(globals())
    tests = [(n, o) for n, o in ns.items() if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
