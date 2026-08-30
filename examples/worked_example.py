"""End-to-end demonstration of the evaluation standard.

Run:  python examples/worked_example.py

The synthetic cohort below is constructed to reproduce the failure mode the
manuscript describes in Section 6.2: a system that looks strong on a headline
number while barely using the image at all. The point of the demonstration is
that the standard *detects* this, and that the detection costs one extra
inference pass.

No patient data is used or required. The findings, codes and regions are
fabricated for illustration and are not clinical content.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elr import (
    Finding,
    Measurement,
    Region,
    Report,
    bootstrap_ci,
    comparator_panel,
    consequence_weighted_errors,
    faithfulness_by_intervention,
    finding_level_agreement,
    image_blind_delta,
    reader_study_summary,
    reference_standard_ceiling,
    risk_coverage,
    unsupported_fraction,
)
from elr.emit import build_fhir, write_fhir, write_sr

rng = random.Random(20260830)

# SNOMED CT concepts, with the severity tier each is assigned in this study.
CONCEPTS = [
    ("21454007",  "Subarachnoid haemorrhage",          "critical"),
    ("230706003", "Cerebral infarction",                "urgent"),
    ("126952004", "Neoplasm of brain",                  "significant"),
    ("445238008", "Chronic microvascular ischaemia",    "routine"),
    ("62914000",  "Cerebral atrophy",                   "routine"),
]

N_STUDIES = 120


def make_region(seed: int) -> Region:
    r = random.Random(seed)
    x, y = r.uniform(20, 180), r.uniform(20, 180)
    return Region(series_uid=f"1.2.826.0.1.{seed % 7}", frame=r.randint(1, 24),
                  bbox=(x, y, x + r.uniform(15, 45), y + r.uniform(15, 45)))


def jitter(region, scale: float, seed: int):
    """Shift a region, simulating an imprecise localisation.

    Passes None through: a reference finding that carries no region (clinical
    history, comparison with priors) cannot be localised by anything.
    """
    if region is None:
        return None
    r = random.Random(seed)
    dx, dy = r.uniform(-scale, scale), r.uniform(-scale, scale)
    x0, y0, x1, y1 = region.bbox
    return Region(series_uid=region.series_uid, frame=region.frame,
                  bbox=(x0 + dx, y0 + dy, x1 + dx, y1 + dy))


def build_cohort():
    """Reference reports, plus three systems evaluated against them."""
    refs, model, blind, template, reader_b = [], [], [], [], []

    for i in range(N_STUDIES):
        uid = f"1.2.826.0.1.3680043.{i}"

        # The reference standard. Routine findings dominate, as they do in
        # real corpora; the critical findings are rare, which is exactly why
        # an unweighted error rate hides them.
        n_findings = rng.choice([1, 2, 2, 3])
        chosen = rng.sample(CONCEPTS, n_findings)
        ref_findings = []
        for j, (code, disp, sev) in enumerate(chosen):
            ref_findings.append(Finding(
                code=code, display=disp,
                text=f"{disp} identified.",
                severity=sev,
                region=make_region(i * 10 + j),
                measurement=Measurement("Lesion diameter", round(rng.uniform(4, 38), 1), "mm", "mm")
                if sev in ("significant", "urgent") else None,
            ))
        # Reference reports also carry content that is NOT derivable from the
        # image: comparison with priors, referrer questions, clinical history.
        # A model trained on these is trained to fabricate exactly this
        # content, so the harness counts it (requirement 7).
        if rng.random() < 0.34:
            ref_findings.append(Finding(
                code="410653004", display="Comparison with prior study",
                text="Unchanged compared with the examination of 3 months prior.",
                severity="routine", region=None,   # not derivable from this study
            ))
        refs.append(Report(study_uid=uid, findings=ref_findings,
                           narrative="Reference report."))

        # System A: the generative model. Recovers the common findings well,
        # frequently misses the rare critical ones, and localises loosely.
        m_findings = []
        for j, f in enumerate(ref_findings):
            miss_p = {"critical": 0.62, "urgent": 0.40,
                      "significant": 0.22, "routine": 0.08}[f.severity]
            if rng.random() < miss_p:
                continue
            m_findings.append(Finding(
                code=f.code, display=f.display, text=f.text, severity=f.severity,
                region=jitter(f.region, 9.0, i * 100 + j),
                measurement=f.measurement,
                confidence=round(min(0.99, max(0.05, rng.gauss(0.72, 0.18))), 3),
            ))
        # Fabrications: plausible, common, and mostly routine.
        if rng.random() < 0.30:
            code, disp, sev = CONCEPTS[rng.choice([3, 4])]
            m_findings.append(Finding(
                code=code, display=disp, text=f"{disp} identified.", severity=sev,
                region=make_region(9000 + i),
                confidence=round(min(0.99, max(0.05, rng.gauss(0.55, 0.2))), 3),
            ))
        model.append(Report(study_uid=uid, findings=m_findings,
                            narrative="Model-drafted report."))

        # System B: THE IMAGE-BLIND BASELINE. Same decoder, no image, only the
        # clinical indication. It emits the population priors: the two routine
        # findings that appear in most studies.
        #
        # Note region=None throughout. A decoder that never saw the image
        # cannot localise, and giving it a region here would make the
        # comparison dishonest in the model's favour.
        b_findings = []
        for code, disp, sev in CONCEPTS[3:]:
            if rng.random() < 0.66:
                b_findings.append(Finding(
                    code=code, display=disp, text=f"{disp} identified.",
                    severity=sev, region=None,
                    confidence=round(rng.uniform(0.4, 0.8), 3),
                ))
        blind.append(Report(study_uid=uid, findings=b_findings,
                            narrative="Image-blind baseline report."))

        # System C: segmentation followed by a deterministic template
        # verbaliser. Conservative, but every statement it makes is grounded
        # in a segment, so it is trivially faithful.
        t_findings = [
            Finding(code=f.code, display=f.display, text=f"{f.display} identified.",
                    severity=f.severity, region=f.region, measurement=f.measurement,
                    confidence=0.99)
            for f in ref_findings
            if rng.random() < {"critical": 0.55, "urgent": 0.58,
                              "significant": 0.60, "routine": 0.52}[f.severity]
        ]
        template.append(Report(study_uid=uid, findings=t_findings,
                               narrative="Template-verbalised report."))

        # A second radiologist reading the same studies, to establish the
        # ceiling the reference standard actually permits.
        rb = []
        for j, f in enumerate(ref_findings):
            if rng.random() < 0.90:
                rb.append(Finding(code=f.code, display=f.display, text=f.text,
                                  severity=f.severity,
                                  region=jitter(f.region, 4.0, i * 500 + j),
                                  measurement=f.measurement))
        if rng.random() < 0.08:
            code, disp, sev = rng.choice(CONCEPTS)
            rb.append(Finding(code=code, display=disp, text=f"{disp} identified.",
                              severity=sev, region=make_region(7000 + i)))
        reader_b.append(Report(study_uid=uid, findings=rb, narrative="Second reader."))

    return refs, model, blind, template, reader_b


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def main() -> None:
    refs, model, blind, template, reader_b = build_cohort()

    # Every number printed below is also collected here and written to
    # results.json, so the figure in the manuscript is generated from the run
    # rather than transcribed from it.
    R: dict = {"n_studies": N_STUDIES}

    print("=" * 74)
    print("EVIDENCE-LINKED REPORTING: EVALUATION STANDARD, WORKED EXAMPLE")
    print(f"{N_STUDIES} synthetic studies. No patient data.")
    print("=" * 74)

    # ---- Requirement 1: the image-blind baseline --------------------------
    rule("R1  Image-blind baseline")
    ib = image_blind_delta(refs, model, blind)
    print(f"  model F1 (with image)   {fmt(ib.with_image.f1)}")
    print(f"  model F1 (image-blind)  {fmt(ib.without_image.f1)}")
    print(f"  delta                   {fmt(ib.delta_f1)}")
    print(f"  uses the image?         {'YES' if ib.uses_image else 'NO'}")
    R["image_blind"] = ib.as_dict()
    if ib.delta_f1 < 0.15:
        print("  ** The margin over a decoder that never saw the image is small.")
        print("     A headline F1 reported without this comparison would be")
        print("     uninterpretable. This is requirement 1, and it cost one pass. **")

    # ---- Requirement 2: comparator panel ---------------------------------
    rule("R2  Comparator panel")
    systems = {
        "generative model": model,
        "image-blind baseline": blind,
        "template verbaliser": template,
    }
    # Scored at concept level so all three are on one footing; the grounded
    # column then shows what the concept-level number conceals.
    panel = comparator_panel(refs, systems, require_grounding=False)
    print(f"  {'system':<24}{'F1':>8}{'precision':>12}{'recall':>10}"
          f"{'mean IoU':>11}{'grounded':>10}")
    for name, r in panel.items():
        preds = systems[name]
        gf = [p.grounded_fraction for p in preds if p.grounded_fraction == p.grounded_fraction]
        g = sum(gf) / len(gf) if gf else 0.0
        print(f"  {name:<24}{fmt(r['f1']):>8}{fmt(r['precision']):>12}"
              f"{fmt(r['recall']):>10}{fmt(r['mean_iou']):>11}{fmt(g):>10}")
    R["panel"] = panel
    print("  The template verbaliser is trivially faithful. A generative system")
    print("  must beat it to justify the additional risk it introduces.")

    # ---- Requirement 3: factuality, with an interval ----------------------
    rule("R3  Finding-level factual agreement (not n-gram overlap)")
    agg = finding_level_agreement(refs[0], model[0])
    point, lo, hi = bootstrap_ci(refs, model, n_boot=800)
    concept_f1 = panel["generative model"]["f1"]
    print(f"  concept-level F1        {fmt(concept_f1)}   (does it name the finding?)")
    print(f"  grounded F1             {fmt(point)}   (does it point at the right place?)"
          f"\n                          95% CI {fmt(lo)} to {fmt(hi)}")
    print(f"  cost of requiring evidence  {fmt(concept_f1 - point)}")
    print(f"  single-study example    hits={agg.hits} omissions={agg.omissions} "
          f"fabrications={agg.fabrications}")
    print("  ** That gap is the entire argument of Section 6.1. The same system,")
    print("     scored on whether its statements are anchored to the region it")
    print("     claims, loses that much. A paper reporting only the first number")
    print("     has not measured the property that makes a report auditable. **")
    R["grounded"] = {"f1": point, "ci_lo": lo, "ci_hi": hi,
                     "concept_f1": concept_f1, "cost_of_evidence": concept_f1 - point}
    print("  No BLEU or ROUGE is computed anywhere in this harness, by design.")

    # ---- Requirement 4: consequence-weighted errors -----------------------
    rule("R4  Fabrication and omission, weighted by consequence")
    tot_om = tot_fab = tot_w = 0.0
    sev_roll: dict = {}
    for r, m in zip(refs, model):
        c = consequence_weighted_errors(r, m)
        tot_om += c.weighted_omission
        tot_fab += c.weighted_fabrication
        tot_w += c.total_reference_weight
        for sev, d in c.by_severity.items():
            acc = sev_roll.setdefault(sev, {"reference": 0, "omitted": 0, "fabricated": 0})
            for k in acc:
                acc[k] += d.get(k, 0)
    print(f"  weighted omission rate  {fmt(tot_om / tot_w if tot_w else float('nan'))}")
    print(f"  weighted fabrication    {fmt(tot_fab, 1)}")
    print(f"\n  {'severity':<15}{'in reference':>14}{'omitted':>10}{'omission rate':>16}")
    for sev in ("critical", "urgent", "significant", "routine"):
        d = sev_roll.get(sev, {})
        n, om = d.get("reference", 0), d.get("omitted", 0)
        print(f"  {sev:<15}{n:>14}{om:>10}{fmt(om / n if n else float('nan')):>16}")
    R["severity"] = {sev: {"reference": sev_roll.get(sev, {}).get("reference", 0),
                           "omitted": sev_roll.get(sev, {}).get("omitted", 0)}
                     for sev in ("critical", "urgent", "significant", "routine")}
    R["weighted_omission_rate"] = tot_om / tot_w if tot_w else float("nan")
    print("  An unweighted rate would average the critical row into the routine")
    print("  one. The severity breakdown is why the weights are reported with")
    print("  the score rather than folded into it.")

    # ---- Requirement 5: faithfulness by intervention ----------------------
    rule("R5  Demonstrated faithfulness, by intervention")

    def regenerate(f: Finding):
        """Stand-in for re-running the decoder with the region masked.

        A faithful system withdraws or alters the statement. Here 55% do,
        which is the number a real study would have to report rather than
        substituting an attention heat map.
        """
        return None if rng.random() < 0.55 else f

    fr = faithfulness_by_intervention(model[0], regenerate)
    tested = changed = 0
    for m in model:
        r = faithfulness_by_intervention(m, regenerate)
        tested += r.tested
        changed += r.changed_or_withdrawn
    print(f"  statements tested       {tested}")
    print(f"  changed or withdrawn    {changed}")
    print(f"  faithfulness rate       {fmt(changed / tested if tested else float('nan'))}")
    R["faithfulness"] = {"tested": tested, "changed": changed,
                         "rate": changed / tested if tested else float("nan")}
    print("  Attention maps are not accepted as evidence here: cross-attention is")
    print("  softmax-normalised and always sums to one over whatever keys exist.")

    # ---- Requirement 6: calibration and abstention ------------------------
    rule("R6  Risk-coverage (selective prediction)")
    print(f"  {'threshold':>10}{'coverage':>11}{'risk':>9}")
    for p in risk_coverage(refs, model, thresholds=[0.0, 0.3, 0.5, 0.7, 0.9]):
        print(f"  {p.threshold:>10.2f}{fmt(p.coverage):>11}{fmt(p.risk):>9}")
    R["risk_coverage"] = [{"threshold": p.threshold, "coverage": p.coverage, "risk": p.risk}
                          for p in risk_coverage(refs, model,
                                                 thresholds=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])]
    print("  A system that cannot decline cannot be accountable for what it asserts.")

    # ---- Requirement 7: the reference standard's own ceiling --------------
    rule("R7  Reference-standard reliability")
    # Both sides scored the same way. Comparing a grounded ceiling against an
    # ungrounded model score is the error this line exists to avoid; it can
    # make a model appear to outperform two radiologists agreeing with
    # each other, which is never what happened.
    ceil = reference_standard_ceiling(refs, reader_b)
    print(f"  inter-radiologist F1    {fmt(ceil.f1)}   <-- the ceiling, not 1.000")
    print(f"  model F1 (same scoring) {fmt(point)}")
    gap = ceil.f1 - point
    print(f"  distance to ceiling     {fmt(gap)}")
    print(f"  apparent gap to 1.000   {fmt(1.0 - point)}   <-- what a paper would report")
    print(f"  unsupported fraction    {fmt(unsupported_fraction(refs))}"
          "   (reference content not derivable from the image)")
    R["ceiling"] = {"inter_reader_f1": ceil.f1, "model_f1": point,
                    "distance_to_ceiling": gap, "apparent_gap_to_one": 1.0 - point,
                    "unsupported_fraction": unsupported_fraction(refs)}
    print("  Reporting the model against 1.000 rather than against the ceiling")
    print("  overstates the remaining gap and misdirects the next experiment.")

    # ---- Requirement 8: prospective in-workflow --------------------------
    rule("R8  Prospective in-workflow assessment")
    n = 400
    correct = [rng.random() > 0.28 for _ in range(n)]
    accepted = [True if ok else (rng.random() < 0.41) for ok in correct]
    times = [rng.gauss(214, 46) for _ in range(n)]
    rs = reader_study_summary(accepted, correct, times)
    print(f"  drafted statements      {rs.drafts_shown}")
    print(f"  incorrect shown         {rs.incorrect_statements_shown}")
    print(f"  incorrect ACCEPTED      {rs.incorrect_statements_accepted}")
    print(f"  automation-bias rate    {fmt(rs.automation_bias_rate)}")
    print(f"  median reporting time   {fmt(rs.median_reporting_time_s, 1)} s")
    R["reader_study"] = rs.as_dict()
    print("  This is the dominant risk: an uncorrected generated sentence enters")
    print("  the permanent record. It is measurable only prospectively.")

    # ---- Section 6.3: the interoperable artefact -------------------------
    rule("Section 6.3  What the system must emit")
    out = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out, exist_ok=True)

    demo = Report(
        study_uid="1.2.826.0.1.3680043.8.498.11111",
        narrative="Acute subarachnoid haemorrhage in the left sylvian fissure. "
                  "No midline shift.",
        findings=[
            Finding(code="21454007", display="Subarachnoid haemorrhage",
                    text="Acute subarachnoid haemorrhage in the left sylvian fissure.",
                    severity="critical",
                    region=Region("1.2.826.0.1.99", 14, (88.0, 102.0, 131.0, 140.0)),
                    measurement=Measurement("Lesion diameter", 21.4, "mm", "mm"),
                    confidence=0.91),
            Finding(code="17836004", display="Midline shift",
                    text="No midline shift.", severity="urgent", absent=True,
                    region=Region("1.2.826.0.1.99", 14, (70.0, 60.0, 150.0, 170.0)),
                    confidence=0.96),
        ],
    )

    sr_path = write_sr(demo, os.path.join(out, "report_sr.dcm"),
                       verifying_observer="Radiologist^Verifying")
    fhir_path = write_fhir(demo, os.path.join(out, "report_fhir.json"))

    print(f"  DICOM SR written        {os.path.basename(sr_path)}")
    print(f"  FHIR bundle written     {os.path.basename(fhir_path)}")
    print(f"  grounded fraction       {fmt(demo.grounded_fraction)}")

    # Read the SR back to show it is a real object, not a formatted string.
    import pydicom
    back = pydicom.dcmread(sr_path)
    print(f"  SR SOP class            {back.SOPClassUID}")
    print(f"  SR modality / flag      {back.Modality} / {back.VerificationFlag}")
    print(f"  content items           {len(back.ContentSequence)}")
    coded = [
        it.ConceptCodeSequence[0]
        for grp in back.ContentSequence
        if getattr(grp, "ValueType", "") == "CONTAINER"
        for it in grp.ContentSequence
        if getattr(it, "ValueType", "") == "CODE"
        and it.ConceptNameCodeSequence[0].CodeMeaning == "Finding"
    ]
    for c in coded:
        print(f"    finding coded         {c.CodingSchemeDesignator} {c.CodeValue}  {c.CodeMeaning}")

    bundle = build_fhir(demo)
    print(f"  FHIR resources          {len(bundle['entry'])} "
          f"({bundle['entry'][0]['resource']['resourceType']} + Observations)")

    R["artefacts"] = {
        "sr_sop_class": str(back.SOPClassUID),
        "sr_content_items": len(back.ContentSequence),
        "fhir_resources": len(bundle["entry"]),
        "grounded_fraction": demo.grounded_fraction,
    }
    res_path = os.path.join(out, "results.json")
    with open(res_path, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    print(f"  results written         {os.path.basename(res_path)}")

    print("\n" + "=" * 74)
    print("Every number above is reproducible from this script. That is the")
    print("property the manuscript argues the literature lacks, and it is")
    print("cheaper to provide than to argue about.")
    print("=" * 74)


if __name__ == "__main__":
    main()
