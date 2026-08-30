"""Reference implementation of the evaluation standard (manuscript Section 6.2).

Each public function here corresponds to one numbered requirement. The point
of the module is not that these are difficult computations. It is that the
computations are *specified*: a study reporting "our model achieved BLEU 0.41"
cannot be compared against another, whereas two studies reporting
``consequence_weighted_errors`` with their weight table attached can be.

Requirement coverage
--------------------
1. image-blind baseline .......... image_blind_delta
2. additional comparators ........ comparator_panel
3. factuality not similarity ..... finding_level_agreement
4. fabrication and omission ...... consequence_weighted_errors
5. demonstrated faithfulness ..... faithfulness_by_intervention
6. calibration and abstention .... risk_coverage
7. reference-standard ceiling .... reference_standard_ceiling
8. prospective in-workflow ....... reader_study_summary
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .model import DEFAULT_SEVERITY_WEIGHTS, Finding, Report

# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

DEFAULT_IOU_THRESHOLD = 0.3


@dataclass
class Match:
    reference: Optional[Finding]
    candidate: Optional[Finding]
    iou: float = 0.0

    @property
    def is_hit(self) -> bool:
        return self.reference is not None and self.candidate is not None

    @property
    def is_omission(self) -> bool:
        return self.reference is not None and self.candidate is None

    @property
    def is_fabrication(self) -> bool:
        return self.reference is None and self.candidate is not None


def match_findings(
    reference: Report,
    candidate: Report,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    require_grounding: bool = True,
) -> list[Match]:
    """Align candidate findings to reference findings.

    A candidate matches a reference when the coded concept agrees AND, if
    ``require_grounding``, the referenced regions overlap at or above
    ``iou_threshold``. The grounding condition is what separates this from
    ordinary label matching: a system that says "left frontal mass" while
    pointing at the right occipital lobe has not made a correct statement about
    the study, and scoring it as a hit is exactly the failure the evaluation
    standard exists to prevent.

    Greedy assignment by descending IoU. Each reference is consumed once.
    """
    unmatched_ref = list(reference.positive())
    unmatched_cand = list(candidate.positive())

    scored: list[tuple[float, Finding, Finding]] = []
    for ref in unmatched_ref:
        for cand in unmatched_cand:
            if ref.code != cand.code:
                continue
            # The IoU is computed whenever both sides carry a region, whether
            # or not it gates the match. Reporting a fabricated 1.0 for
            # ungrounded matching would make mean_iou meaningless exactly
            # where the reader most needs it to be honest.
            if ref.region is not None and cand.region is not None:
                iou = ref.region.iou(cand.region)
            else:
                iou = float("nan")
            if require_grounding:
                if iou != iou or iou < iou_threshold:  # NaN or below threshold
                    continue
            scored.append((iou, ref, cand))

    # NaN (one side ungrounded) sorts last, so a grounded pairing always wins
    # the greedy assignment over an ungrounded one for the same concept.
    scored.sort(key=lambda t: (t[0] != t[0], -t[0] if t[0] == t[0] else 0.0))
    matches: list[Match] = []
    used_ref: set[int] = set()
    used_cand: set[int] = set()
    for iou, ref, cand in scored:
        if id(ref) in used_ref or id(cand) in used_cand:
            continue
        used_ref.add(id(ref))
        used_cand.add(id(cand))
        matches.append(Match(reference=ref, candidate=cand, iou=iou))

    matches += [Match(reference=r, candidate=None) for r in unmatched_ref if id(r) not in used_ref]
    matches += [Match(reference=None, candidate=c) for c in unmatched_cand if id(c) not in used_cand]
    return matches


# --------------------------------------------------------------------------
# requirement 3: factuality rather than similarity
# --------------------------------------------------------------------------

@dataclass
class AgreementResult:
    hits: int
    omissions: int
    fabrications: int
    grounded_hits: int
    mean_iou: float

    @property
    def precision(self) -> float:
        d = self.hits + self.fabrications
        return self.hits / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.hits + self.omissions
        return self.hits / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "omissions": self.omissions,
            "fabrications": self.fabrications,
            "grounded_hits": self.grounded_hits,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
        }


def finding_level_agreement(
    reference: Report,
    candidate: Report,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    require_grounding: bool = True,
) -> AgreementResult:
    """Requirement 3. Entity-level factual agreement, replacing n-gram overlap.

    Note what is deliberately absent: there is no BLEU, ROUGE or BERTScore
    anywhere in this module. A system can score well on those while asserting
    the opposite of the truth, which is why the manuscript rejects them.
    """
    matches = match_findings(reference, candidate, iou_threshold, require_grounding)
    hits = [m for m in matches if m.is_hit]
    ious = [m.iou for m in hits if m.iou == m.iou]  # drop NaN (ungrounded pairs)
    return AgreementResult(
        hits=len(hits),
        omissions=sum(1 for m in matches if m.is_omission),
        fabrications=sum(1 for m in matches if m.is_fabrication),
        grounded_hits=sum(1 for m in hits if m.candidate and m.candidate.grounded),
        mean_iou=sum(ious) / len(ious) if ious else float("nan"),
    )


# --------------------------------------------------------------------------
# requirement 4: fabrication and omission, weighted by consequence
# --------------------------------------------------------------------------

@dataclass
class ConsequenceResult:
    weighted_omission: float
    weighted_fabrication: float
    total_reference_weight: float
    weights: dict
    by_severity: dict

    @property
    def weighted_omission_rate(self) -> float:
        return (
            self.weighted_omission / self.total_reference_weight
            if self.total_reference_weight
            else float("nan")
        )

    def as_dict(self) -> dict:
        return {
            "weighted_omission": self.weighted_omission,
            "weighted_fabrication": self.weighted_fabrication,
            "weighted_omission_rate": self.weighted_omission_rate,
            "weights_used": self.weights,
            "by_severity": self.by_severity,
        }


def consequence_weighted_errors(
    reference: Report,
    candidate: Report,
    weights: Optional[dict] = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    require_grounding: bool = True,
) -> ConsequenceResult:
    """Requirement 4. Omission and fabrication weighted by clinical severity.

    The weight table is returned alongside the score. This is deliberate: an
    unweighted error rate hides which errors were made, and a weighted rate
    whose weights are not published is not reproducible.
    """
    w = dict(weights or DEFAULT_SEVERITY_WEIGHTS)
    matches = match_findings(reference, candidate, iou_threshold, require_grounding)

    by_sev: dict[str, dict[str, int]] = {}
    for sev in w:
        by_sev[sev] = {"reference": 0, "hit": 0, "omitted": 0, "fabricated": 0}

    weighted_omission = 0.0
    weighted_fabrication = 0.0
    total_ref_weight = 0.0

    for m in matches:
        if m.reference is not None:
            sev = m.reference.severity
            by_sev.setdefault(sev, {"reference": 0, "hit": 0, "omitted": 0, "fabricated": 0})
            by_sev[sev]["reference"] += 1
            total_ref_weight += w.get(sev, 1.0)
            if m.is_omission:
                by_sev[sev]["omitted"] += 1
                weighted_omission += w.get(sev, 1.0)
            else:
                by_sev[sev]["hit"] += 1
        elif m.is_fabrication and m.candidate is not None:
            sev = m.candidate.severity
            by_sev.setdefault(sev, {"reference": 0, "hit": 0, "omitted": 0, "fabricated": 0})
            by_sev[sev]["fabricated"] += 1
            weighted_fabrication += w.get(sev, 1.0)

    return ConsequenceResult(
        weighted_omission=weighted_omission,
        weighted_fabrication=weighted_fabrication,
        total_reference_weight=total_ref_weight,
        weights=w,
        by_severity=by_sev,
    )


# --------------------------------------------------------------------------
# requirement 1: the image-blind baseline
# --------------------------------------------------------------------------

@dataclass
class ImageBlindResult:
    with_image: AgreementResult
    without_image: AgreementResult

    @property
    def delta_f1(self) -> float:
        return self.with_image.f1 - self.without_image.f1

    @property
    def uses_image(self) -> bool:
        """Whether the system demonstrably uses the image at all."""
        return self.delta_f1 > 0

    def as_dict(self) -> dict:
        return {
            "with_image_f1": self.with_image.f1,
            "image_blind_f1": self.without_image.f1,
            "delta_f1": self.delta_f1,
            "uses_image": self.uses_image,
        }


def image_blind_delta(
    references: Sequence[Report],
    with_image: Sequence[Report],
    image_blind: Sequence[Report],
    **kw,
) -> ImageBlindResult:
    """Requirement 1, and the one the manuscript calls most informative.

    ``image_blind`` is the same decoder given only the clinical indication, or
    only the structured measurements, with no image. Much report content is
    predictable from priors alone. A system that does not beat this baseline
    is not using the image, however good its headline number looks.

    Both arms are scored WITHOUT the grounding requirement, and the reason is
    not a concession. A decoder that never saw the image cannot emit an image
    region, so requiring one would score the baseline at zero by construction
    and manufacture a difference no matter how little the image contributed.
    The comparison that carries information is concept-level: how much of the
    report is recoverable from priors alone? Grounding is a separate property,
    measured against the image-using arm by ``finding_level_agreement`` and
    ``faithfulness_by_intervention``, and it is not what this contrast is for.
    """
    if not (len(references) == len(with_image) == len(image_blind)):
        raise ValueError("references, with_image and image_blind must be the same length")
    kw = {**kw, "require_grounding": False}
    return ImageBlindResult(
        with_image=_pooled_agreement(references, with_image, **kw),
        without_image=_pooled_agreement(references, image_blind, **kw),
    )


def _pooled_agreement(references, candidates, **kw) -> AgreementResult:
    hits = omissions = fabrications = grounded = 0
    ious: list[float] = []
    for ref, cand in zip(references, candidates):
        r = finding_level_agreement(ref, cand, **kw)
        hits += r.hits
        omissions += r.omissions
        fabrications += r.fabrications
        grounded += r.grounded_hits
        if not math.isnan(r.mean_iou):
            ious.append(r.mean_iou)
    return AgreementResult(
        hits=hits,
        omissions=omissions,
        fabrications=fabrications,
        grounded_hits=grounded,
        mean_iou=sum(ious) / len(ious) if ious else float("nan"),
    )


# --------------------------------------------------------------------------
# requirement 2: additional comparators
# --------------------------------------------------------------------------

def comparator_panel(
    references: Sequence[Report],
    systems: dict[str, Sequence[Report]],
    **kw,
) -> dict[str, dict]:
    """Requirement 2. Score every system in the panel on the same footing.

    The manuscript names two comparators a generative system must beat:
    nearest-neighbour report retrieval, and segmentation followed by a
    deterministic template verbaliser. The latter is trivially faithful and
    quantitative, so it is the honest comparator any generative system must
    beat to justify its additional risk. Pass them in here alongside the model.
    """
    return {
        name: _pooled_agreement(references, preds, **kw).as_dict()
        for name, preds in systems.items()
    }


# --------------------------------------------------------------------------
# requirement 5: demonstrated faithfulness, by intervention
# --------------------------------------------------------------------------

@dataclass
class FaithfulnessResult:
    tested: int
    changed_or_withdrawn: int

    @property
    def faithfulness_rate(self) -> float:
        return self.changed_or_withdrawn / self.tested if self.tested else float("nan")

    def as_dict(self) -> dict:
        return {
            "statements_tested": self.tested,
            "changed_or_withdrawn": self.changed_or_withdrawn,
            "faithfulness_rate": self.faithfulness_rate,
        }


def faithfulness_by_intervention(
    report: Report,
    regenerate: Callable[[Finding], Optional[Finding]],
) -> FaithfulnessResult:
    """Requirement 5. The causal test, not the attention map.

    ``regenerate`` receives a finding whose referenced region has been masked
    or perturbed, and returns the finding the system emits under that
    intervention, or None if it withdraws the statement.

    A faithful system changes or withdraws the statement when the evidence it
    claims to rest on is removed. A system whose output is unchanged was not
    using that region, whatever its attention map displays. Attention is
    softmax-normalised and therefore always sums to one over whatever visual
    keys exist, so its mere presence carries no information.
    """
    tested = changed = 0
    for f in report.positive():
        if not f.grounded:
            continue
        tested += 1
        after = regenerate(f)
        if after is None or after.code != f.code or after.text != f.text:
            changed += 1
    return FaithfulnessResult(tested=tested, changed_or_withdrawn=changed)


# --------------------------------------------------------------------------
# requirement 6: calibration and abstention
# --------------------------------------------------------------------------

@dataclass
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float


def risk_coverage(
    references: Sequence[Report],
    candidates: Sequence[Report],
    thresholds: Optional[Sequence[float]] = None,
    **kw,
) -> list[RiskCoveragePoint]:
    """Requirement 6. Selective prediction: what does the system get right
    when it is allowed to decline?

    Coverage is the fraction of findings the system asserts at a given
    confidence threshold. Risk is the fabrication rate AMONG THOSE ASSERTED,
    the standard selective-prediction semantics: a statement the system
    declined to make cannot be wrong. Omissions are deliberately not counted
    here; the cost of abstaining is priced, severity-weighted, by requirement
    4, and the two curves must be read together. An earlier version folded
    omissions into risk, which made abstention look harmful by construction
    and contradicted the axis label it was plotted under; the regression
    tests pin the corrected semantics.
    """
    grid = list(thresholds or [i / 10 for i in range(0, 10)])
    out: list[RiskCoveragePoint] = []
    for t in grid:
        gated: list[Report] = []
        for c in candidates:
            keep = [
                f for f in c.findings
                if f.confidence is None or f.confidence >= t
            ]
            gated.append(Report(study_uid=c.study_uid, findings=keep,
                                narrative=c.narrative, abstained=not keep))
        total_asserted = sum(len(g.positive()) for g in gated)
        total_possible = sum(len(c.positive()) for c in candidates)
        agg = _pooled_agreement(references, gated, **kw)
        asserted = agg.hits + agg.fabrications
        out.append(RiskCoveragePoint(
            threshold=t,
            coverage=total_asserted / total_possible if total_possible else float("nan"),
            risk=agg.fabrications / asserted if asserted else float("nan"),
        ))
    return out


# --------------------------------------------------------------------------
# requirement 7: the reference standard's own reliability
# --------------------------------------------------------------------------

def reference_standard_ceiling(
    radiologist_a: Sequence[Report],
    radiologist_b: Sequence[Report],
    **kw,
) -> AgreementResult:
    """Requirement 7. What two independent radiologists achieve against each
    other on the same studies.

    This is the ceiling the reference standard permits. A model reported at
    F1 0.72 against a single-reader reference standard is not 28 points from
    perfect; it is some distance from a ceiling that must be measured rather
    than assumed. Inter-radiologist report variability is larger and less well
    characterised than inter-rater segmentation agreement.
    """
    return _pooled_agreement(radiologist_a, radiologist_b, **kw)


def unsupported_fraction(reports: Sequence[Report]) -> float:
    """Also requirement 7: the fraction of reference-report content that is
    not derivable from the image.

    Reference reports contain clinical history, comparison with priors and
    answers to referrer questions. A model trained on them is trained to
    fabricate exactly that content, so the unsupported fraction of the
    training corpus bounds what any faithful system can be expected to
    reproduce. Findings without a region are counted as unsupported.
    """
    total = sum(len(r.positive()) for r in reports)
    if not total:
        return float("nan")
    ungrounded = sum(1 for r in reports for f in r.positive() if not f.grounded)
    return ungrounded / total


# --------------------------------------------------------------------------
# requirement 8: prospective in-workflow assessment
# --------------------------------------------------------------------------

@dataclass
class ReaderStudyResult:
    drafts_shown: int
    incorrect_statements_shown: int
    incorrect_statements_accepted: int
    median_reporting_time_s: Optional[float] = None

    @property
    def automation_bias_rate(self) -> float:
        """The rate at which readers accept incorrect drafted statements.

        The manuscript identifies this as the dominant risk, and as
        qualitatively worse for drafting than for detection: an uncorrected
        generated sentence enters the permanent medical record.
        """
        return (
            self.incorrect_statements_accepted / self.incorrect_statements_shown
            if self.incorrect_statements_shown
            else float("nan")
        )

    def as_dict(self) -> dict:
        return {
            "drafts_shown": self.drafts_shown,
            "incorrect_statements_shown": self.incorrect_statements_shown,
            "incorrect_statements_accepted": self.incorrect_statements_accepted,
            "automation_bias_rate": self.automation_bias_rate,
            "median_reporting_time_s": self.median_reporting_time_s,
        }


def reader_study_summary(
    accepted_flags: Sequence[bool],
    statement_correct: Sequence[bool],
    reporting_times_s: Optional[Sequence[float]] = None,
    drafts_shown: Optional[int] = None,
) -> ReaderStudyResult:
    """Requirement 8. Summarise a prospective in-workflow reader study.

    ``accepted_flags[i]`` records whether the reader let drafted statement i
    stand; ``statement_correct[i]`` whether it was in fact correct.
    """
    if len(accepted_flags) != len(statement_correct):
        raise ValueError("accepted_flags and statement_correct must be the same length")
    incorrect = [i for i, ok in enumerate(statement_correct) if not ok]
    accepted_incorrect = sum(1 for i in incorrect if accepted_flags[i])
    median_t = None
    if reporting_times_s:
        s = sorted(reporting_times_s)
        n = len(s)
        median_t = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return ReaderStudyResult(
        drafts_shown=drafts_shown if drafts_shown is not None else len(accepted_flags),
        incorrect_statements_shown=len(incorrect),
        incorrect_statements_accepted=accepted_incorrect,
        median_reporting_time_s=median_t,
    )


# --------------------------------------------------------------------------
# bootstrap confidence intervals
# --------------------------------------------------------------------------

def bootstrap_ci(
    references: Sequence[Report],
    candidates: Sequence[Report],
    statistic: Callable[[AgreementResult], float] = lambda a: a.f1,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    **kw,
) -> tuple[float, float, float]:
    """Study-level bootstrap CI for any agreement statistic.

    Included because Table 2 of the manuscript records that *no* appraised
    study reported a confidence interval on any metric, and because item M-16
    of the SIIM reviewer checklist asks for the statistical method. A harness
    that makes the interval free removes the usual excuse for omitting it.

    Returns (point_estimate, lower, upper).
    """
    import random

    rng = random.Random(seed)
    n = len(references)
    point = statistic(_pooled_agreement(references, candidates, **kw))
    if n < 2:
        return point, float("nan"), float("nan")
    draws: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = statistic(_pooled_agreement([references[i] for i in idx],
                                        [candidates[i] for i in idx], **kw))
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return point, float("nan"), float("nan")
    draws.sort()
    lo = draws[max(0, int((alpha / 2) * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return point, lo, hi
