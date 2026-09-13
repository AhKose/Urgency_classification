# Prevalence-adjusted PPV / NPV (spectrum-bias projection)

The evaluated cohort is referral-skewed (~74% urgent). Sensitivity and specificity at the `sens>=95` operating point are held fixed and PPV/NPV are recomputed by Bayes' rule at lower urgent prevalences representative of an unscreened / general pediatric population.

## YOLO-TL caries-adapted (CNN)  (Se=0.92, Sp=0.21 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.77 | 0.49 |
| 30% | 0.33 | 0.87 |
| 15% | 0.17 | 0.94 |

## ResNet50V2 (ImageNet)  (Se=0.93, Sp=0.20 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.77 | 0.51 |
| 30% | 0.33 | 0.87 |
| 15% | 0.17 | 0.94 |

## EfficientNet-B3 (ImageNet)  (Se=0.96, Sp=0.13 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.76 | 0.52 |
| 30% | 0.32 | 0.88 |
| 15% | 0.16 | 0.95 |

## YOLOv8-L COCO-only  (Se=0.87, Sp=0.29 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.78 | 0.44 |
| 30% | 0.35 | 0.84 |
| 15% | 0.18 | 0.93 |

## YOLOv8-L from scratch  (Se=1.00, Sp=0.01 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.74 | 1.00 |
| 30% | 0.30 | 1.00 |
| 15% | 0.15 | 1.00 |

## Detect-then-count (detector features)  (Se=0.93, Sp=0.41 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.82 | 0.69 |
| 30% | 0.40 | 0.94 |
| 15% | 0.22 | 0.97 |

## Detect-then-count + spatial features  (Se=0.93, Sp=0.40 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.81 | 0.66 |
| 30% | 0.40 | 0.93 |
| 15% | 0.21 | 0.97 |

## Detect-then-count + CNN (late fusion, proposed)  (Se=0.91, Sp=0.44 @ sens>=95)

| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |
|---|---|---|
| 74% (observed) | 0.82 | 0.62 |
| 30% | 0.41 | 0.91 |
| 15% | 0.22 | 0.96 |
