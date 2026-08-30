# elr — evidence-linked reporting

Reference implementation accompanying:

> **Deep Learning and Natural Language Processing in Neuroradiology: From
> Saturated Benchmarks to Evidence-Linked Reporting.**
> Islam A, Siddik AB, Roky MAB, Ray AS, Abhi SH, Rose TH.
> *Manuscript under review.*

## What this is, and what it deliberately is not

This package contains **no report-generation model**, and that is a design
decision rather than an omission.

The manuscript argues that the bottleneck in this field is evaluation and
accountability rather than architecture, and that the literature's failure mode
is "not insufficient ambition but insufficient evidence." Shipping another
unvalidated generator alongside that argument would contradict it. What is
provided instead is the tooling a study needs in order to report its results in
a form that can be compared against another study's.

The claim being demonstrated is narrow and checkable: **the evaluation standard
in Section 7.2 is computable, and the interoperable artefact in Section 7.3 is
constructible with existing standards.** Neither is aspirational. Their absence
from published systems is a choice.

## Install and run

```
pip install pydicom
python examples/worked_example.py     # full worked example, ~2 s
python tests/test_metrics.py          # 25 tests, no pytest required
```

No patient data is used or required. The worked example runs on a synthetic
cohort of 120 studies with fabricated findings, codes and regions.

## The eight requirements

| # | Requirement | Function |
|---|---|---|
| 1 | Image-blind baseline | `image_blind_delta` |
| 2 | Additional comparators | `comparator_panel` |
| 3 | Factuality, not similarity | `finding_level_agreement` |
| 4 | Consequence-weighted omission and fabrication | `consequence_weighted_errors` |
| 5 | Faithfulness by intervention | `faithfulness_by_intervention` |
| 6 | Calibration and abstention | `risk_coverage` |
| 7 | Reference-standard ceiling | `reference_standard_ceiling`, `unsupported_fraction` |
| 8 | Prospective in-workflow assessment | `reader_study_summary` |

Plus `bootstrap_ci`, because Table 2 of the manuscript records that *no*
appraised study reported a confidence interval on any metric. A harness that
makes the interval free removes the usual reason for omitting it.

There is no BLEU, ROUGE or BERTScore anywhere in this package. A system can
score well on those while asserting the opposite of the truth.

## What the worked example shows

Running it on the synthetic cohort produces, among other numbers:

```
concept-level F1        0.809   (does it name the finding?)
grounded F1             0.632   (does it point at the right place?)
cost of requiring evidence  0.177

inter-radiologist F1    0.785   <-- the ceiling, not 1.000
model F1 (same scoring) 0.632
distance to ceiling     0.153
apparent gap to 1.000   0.368   <-- what a paper would report
```

Two things are visible here that a headline accuracy conceals. Requiring a
statement to be anchored to the region it claims costs 0.177 F1 — that gap
*is* the argument of Section 7.1. And a paper reporting distance-to-perfect
would claim a remaining gap of 0.368 when the reference standard only permits
0.153, overstating it by a factor of 2.4 and misdirecting the next experiment.

The severity breakdown shows the same asymmetry the manuscript describes:
critical findings are omitted at roughly twice the rate of significant ones, an
asymmetry an unweighted error rate averages away entirely.

## Data model

A report is not a string. `Report` holds `Finding` objects, each carrying a
SNOMED CT concept, the `Region` of image evidence supporting it, an optional
`Measurement` retained as a number, and an optional model confidence.

A `Finding` without a region is **not evidence-linked**. It may still be
correct; it is simply not auditable, which is the property the whole
formulation exists to provide. The metrics treat it as ungrounded rather than
silently scoring it as though it were grounded.

## Interoperability (Section 7.3)

`elr.emit` produces two artefacts from the same `Report`:

- **DICOM Comprehensive 3D SR** (`1.2.840.10008.5.1.4.1.1.88.34`) — findings as
  coded content items, measurements as `NUM` items with UCUM units, image
  regions as `SCOORD` with `SELECTED FROM` image references, and an explicit
  `VerificationFlag` so a draft is never mistaken for a signed report.
- **HL7 FHIR transaction bundle** — a `DiagnosticReport` with one `Observation`
  per finding, same SNOMED codes, measurements as `valueQuantity`, and the
  evidence link carried in `derivedFrom` so it survives the hop into the
  electronic record.

Both are written to `examples/output/` by the worked example and read back to
show they are real objects rather than formatted strings.

## Two bugs this harness had, and what they illustrate

Recorded because both are easy to make and neither is visible in a headline
number. Both are now regression-tested.

1. **The image-blind baseline was scored with grounding required.** A decoder
   that never saw the image cannot emit a region, so requiring one forced the
   baseline to zero *by construction* and manufactured a large delta no matter
   how little the image actually contributed. The comparison that carries
   information is concept-level.
2. **Ungrounded matches reported a fabricated IoU of 1.0**, making the column
   meaningless precisely where it needed to be honest. It now reports `NaN`.

## Caveats

The severity weights in `DEFAULT_SEVERITY_WEIGHTS` are a **default, not a
validated instrument**. Any study using this harness must state and justify the
weights it used; the harness returns them alongside every score so they cannot
be left implicit.

As the manuscript states, no validated factuality metric exists for
neuroradiology reports — the established instruments were developed on chest
radiograph corpora. Constructing one is a prerequisite for this programme, not
a detail, and this package does not claim to be one. It provides the protocol
into which such a metric would slot.

## Repository layout

```
elr/          the harness: data model, the eight metrics, DICOM SR + FHIR emitters
tests/        25 behavioural tests, no pytest required
examples/     worked example on a synthetic cohort, and the Figure 3 generator
figures/      generators for Figures 2 and 4, from the manuscript's own tables
```

Figures are written to `output/` when the repository is used standalone. Set
`ELR_FIGURE_DIR` to write them somewhere else; if the code sits inside the
manuscript tree it finds it automatically and writes the PNGs in place.

## Licence

MIT, see [LICENSE](LICENSE). See the manuscript for the argument this code
accompanies.
