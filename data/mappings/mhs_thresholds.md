# MHS severity thresholds

The cutoffs use mean comment-level hate scores. They were fixed on 14 July 2026 from approximate thirds of the hateful range.

| Band | Condition | Recorded share of hateful comments |
|---|---|---:|
| Non-hate | score <= 0.5 | - |
| Low | 0.5 < score <= 1.00 | 33.9% |
| Medium | 1.00 < score <= 1.63 | 32.9% |
| High | score > 1.63 | 33.2% |

These are score-derived perceived-intensity bands, not independently annotated threat levels.

Inspect with `python -m scripts.plots.plot_mhs`. Historical reproduction uses the recorded cutoffs and original export rather than estimating new boundaries.
