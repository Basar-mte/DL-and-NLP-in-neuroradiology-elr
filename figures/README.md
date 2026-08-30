# Manuscript figure sources

Regenerate any figure from the manuscript's own tabulated values:

    python figures/make_fig2_saturation.py     # Fig2_saturation.png
    python figures/make_fig5_evidence.py       # Fig5_evidence.png
    python elr/examples/worked_example.py      # writes results.json
    python elr/examples/make_figure.py         # Fig4_evaluation.png
    python elr/examples/make_interface_figure.py   # Fig3_interface.png

`style.py` holds the shared palette, sampled from the original artwork so
regenerated panels sit beside the untouched ones without a visible seam.

## House rules

**No prose annotation inside the axes.** Identity is carried by a single legend
on one line beneath the panels; interpretation belongs to the figure legend in
the manuscript, which has room to qualify it properly. Numeric values stay on
the marks: those are data, not annotation.

**One colour, one meaning, across every panel of a figure.** The evaluation figure previously
used red for the grounded score in (a), critical severity in (b) and perfect
agreement in (c) -- three unrelated referents sharing a hue, which a shared
legend cannot describe. Colour now follows the entity, never its rank, which
is why the severity bars in its omission panel and the zero rows in the criteria panel of Fig5 are a single
colour rather than graded.

**Orange and red are never adjacent categories in one panel.** They separate by
only dE 7.1 at normal vision, below the dE 15 floor at which full-colour
readers can still distinguish a pair. Where both appear in a figure they carry
unrelated roles in different panels, and the legend names each.

## Provenance

Nothing here is estimated from the previous rendering. Fig2 panel (a) plots the
seven rows of Table 3 that state a numeric dataset size; panel (b) is computed
(Wilson score interval width). Fig5 reproduces the subgroup sensitivities in
Section 8 and the column totals of Table 2. Fig4 is generated from
`results.json`, written by the evaluation harness, so the figure and the run
cannot disagree.

Fig1 (workflow) and Fig6 (roadmap) are diagrams, not plots: their text is
content rather than annotation, and they are left as supplied. No source is
available for them.
