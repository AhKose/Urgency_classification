# Inter-rater agreement (rater 1 vs rater 2, binary urgent / non-urgent)

All **287** patients, before adjudication. Rater 3 breaks ties only.

- **Raw agreement**: 95.5% (95% CI 93.0–97.6)
- **Cohen's κ**: 0.880 (95% CI 0.810–0.939)  — "almost perfect" (Landis & Koch)
- **PABAK**: 0.909 (95% CI 0.861–0.951)  — prevalence-adjusted (imbalance-robust)
- **Gwet's AC1**: 0.927 (95% CI 0.885–0.963)
- **Disagreements**: 13 / 287 (4.5%) — the cases rater 3 adjudicated

## Confusion matrix (rater 1 rows, rater 2 columns)

|  | rater 2 urgent | rater 2 non-urgent |
|---|---|---|
| **rater 1 urgent** | 208 | 4 |
| **rater 1 non-urgent** | 9 | 66 |

## Disagreement cases (→ adjudicated by rater 3)

| patient_id | source | rater 1 | rater 2 | consensus (rater 3) |
|---|---|---|---|---|
| case_113 | B | urgent | non_urgent | non_urgent |
| case_120 | B | urgent | non_urgent | non_urgent |
| case_134 | B | urgent | non_urgent | non_urgent |
| case_208 | B | urgent | non_urgent | non_urgent |
| case_221 | A | non_urgent | urgent | urgent |
| case_227 | A | non_urgent | urgent | urgent |
| case_254 | A | non_urgent | urgent | non_urgent |
| case_255 | A | non_urgent | urgent | non_urgent |
| case_258 | A | non_urgent | urgent | non_urgent |
| case_265 | A | non_urgent | urgent | non_urgent |
| case_266 | A | non_urgent | urgent | urgent |
| case_278 | B | non_urgent | urgent | non_urgent |
| case_283 | B | non_urgent | urgent | non_urgent |

_Final consensus label = majority of three. Of the 13 disagreements, rater 3's adjudication changed 7 of rater 1's original labels; the remainder were resolved in rater 1's favour._