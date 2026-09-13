# Operating-point analysis (triage framing)

Threshold chosen on each fold's `final_val` to reach the target urgent-sensitivity, then applied once to that fold's held-out test set. Mean ± sd over 5 folds.

`NPV` = P(truly non-urgent | model says "can wait") — the safety metric for automatic deferral.
`cleared` = fraction of the caseload the tool would defer.

## YOLO-TL caries-adapted (CNN)

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.81 ± 0.11 | 0.49 ± 0.31 | 0.49 ± 0.13 | 0.83 ± 0.08 | 27% | 40/211 |
| sens>=90 | 0.83 ± 0.17 | 0.40 ± 0.30 | 0.51 ± 0.16 | 0.81 ± 0.07 | 23% | 36/211 |
| sens>=95 | 0.92 ± 0.09 | 0.21 ± 0.26 | 0.60 ± 0.26 | 0.77 ± 0.06 | 11% | 16/211 |

## ResNet50V2 (ImageNet)

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.58 ± 0.28 | 0.52 ± 0.28 | 0.35 ± 0.12 | 0.79 ± 0.08 | 45% | 88/211 |
| sens>=90 | 0.90 ± 0.07 | 0.28 ± 0.06 | 0.53 ± 0.12 | 0.77 ± 0.01 | 15% | 22/211 |
| sens>=95 | 0.93 ± 0.04 | 0.20 ± 0.10 | 0.53 ± 0.08 | 0.76 ± 0.02 | 10% | 14/211 |

## EfficientNet-B3 (ImageNet)

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.71 ± 0.20 | 0.46 ± 0.28 | 0.37 ± 0.06 | 0.79 ± 0.06 | 34% | 62/211 |
| sens>=90 | 0.87 ± 0.04 | 0.21 ± 0.15 | 0.34 ± 0.21 | 0.75 ± 0.04 | 15% | 28/211 |
| sens>=95 | 0.96 ± 0.03 | 0.13 ± 0.13 | 0.54 ± 0.37 | 0.75 ± 0.03 | 7% | 9/211 |

## YOLOv8-L COCO-only

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.77 ± 0.11 | 0.39 ± 0.15 | 0.40 ± 0.13 | 0.78 ± 0.03 | 27% | 49/211 |
| sens>=90 | 0.81 ± 0.13 | 0.39 ± 0.12 | 0.48 ± 0.16 | 0.79 ± 0.02 | 24% | 40/211 |
| sens>=95 | 0.87 ± 0.12 | 0.29 ± 0.13 | 0.51 ± 0.16 | 0.77 ± 0.03 | 17% | 27/211 |

## YOLOv8-L from scratch

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 1.00 ± 0.00 | 0.00 ± 0.00 | nan ± nan | 0.74 ± 0.01 | 0% | 0/211 |
| sens>=90 | 1.00 ± 0.00 | 0.01 ± 0.03 | 1.00 ± nan | 0.74 ± 0.01 | 0% | 0/211 |
| sens>=95 | 1.00 ± 0.00 | 0.01 ± 0.03 | 1.00 ± nan | 0.74 ± 0.01 | 0% | 0/211 |

## Detect-then-count (detector features)

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.76 ± 0.07 | 0.71 ± 0.17 | 0.52 ± 0.04 | 0.89 ± 0.05 | 37% | 51/211 |
| sens>=90 | 0.88 ± 0.06 | 0.61 ± 0.20 | 0.65 ± 0.09 | 0.87 ± 0.06 | 25% | 26/211 |
| sens>=95 | 0.93 ± 0.05 | 0.41 ± 0.24 | 0.68 ± 0.13 | 0.82 ± 0.06 | 16% | 14/211 |

## Detect-then-count + spatial features

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.80 ± 0.13 | 0.68 ± 0.28 | 0.56 ± 0.09 | 0.88 ± 0.07 | 33% | 43/211 |
| sens>=90 | 0.86 ± 0.07 | 0.66 ± 0.19 | 0.64 ± 0.10 | 0.88 ± 0.06 | 28% | 30/211 |
| sens>=95 | 0.93 ± 0.04 | 0.40 ± 0.20 | 0.65 ± 0.14 | 0.81 ± 0.05 | 16% | 15/211 |

## Detect-then-count + CNN (late fusion, proposed)

| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |
|---|---|---|---|---|---|---|
| default_0.5 | 0.80 ± 0.09 | 0.73 ± 0.18 | 0.58 ± 0.09 | 0.90 ± 0.06 | 34% | 42/211 |
| sens>=90 | 0.82 ± 0.10 | 0.62 ± 0.22 | 0.57 ± 0.08 | 0.87 ± 0.07 | 30% | 38/211 |
| sens>=95 | 0.91 ± 0.12 | 0.44 ± 0.31 | 0.73 ± 0.21 | 0.83 ± 0.08 | 18% | 20/211 |
