# Fusion-rule sensitivity (det_count + YOLO-TL CNN)

Mean ± sd over 5 outer folds. `prob_avg` (equal-weight probability average) is the rule reported in the main results. No tuned weights.

| combination rule | macro-F1 | ROC-AUC | PR-AUC non-urgent | balanced acc | MCC |
|---|---|---|---|---|---|
| probability average (0.5) — reported | 73.6 ± 4.6 | 86.2 ± 5.4 | 70.6 ± 10.6 | 76.4 ± 6.3 | 50.1 ± 9.0 |
| logit average | 73.6 ± 4.6 | 85.2 ± 6.7 | 69.0 ± 11.4 | 76.4 ± 6.3 | 50.1 ± 9.0 |
| rank average (calibration-free) | 68.1 ± 3.0 | 83.9 ± 7.7 | 66.2 ± 13.8 | 75.5 ± 5.3 | 45.1 ± 9.1 |
| isotonic-calibrate then average | 67.1 ± 11.2 | 82.8 ± 7.8 | 62.5 ± 15.2 | 67.1 ± 11.2 | 38.1 ± 20.1 |

macro-F1 spread across the four rules: 6.5 points — non-trivial.