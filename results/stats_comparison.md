# Statistical comparison — proposed (YOLO-TL) vs baselines

**Per-fold paired tests (n = 5 folds — underpowered, directional only)**

### vs YOLO-TL caries-adapted (CNN)

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 62.7 | +10.9 ± 6.8 | 0.023 | 0.062 |
| f1_non_urgent | 63.0 | 44.3 | +18.8 ± 13.4 | 0.035 | 0.062 |
| roc_auc | 86.2 | 75.2 | +11.0 ± 7.9 | 0.036 | 0.062 |
| pr_auc_non_urgent | 70.6 | 54.3 | +16.3 ± 5.7 | 0.003 | 0.062 |
| mcc | 50.1 | 29.9 | +20.2 ± 12.8 | 0.024 | 0.062 |
| balanced_accuracy | 76.4 | 65.0 | +11.4 ± 5.7 | 0.011 | 0.062 |

### vs ResNet50V2 (ImageNet)

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 51.0 | +22.6 ± 9.6 | 0.006 | 0.062 |
| f1_non_urgent | 63.0 | 38.3 | +24.7 ± 12.1 | 0.010 | 0.062 |
| roc_auc | 86.2 | 63.7 | +22.5 ± 10.5 | 0.009 | 0.062 |
| pr_auc_non_urgent | 70.6 | 44.4 | +26.2 ± 12.0 | 0.008 | 0.062 |
| mcc | 50.1 | 12.6 | +37.5 ± 16.0 | 0.006 | 0.062 |
| balanced_accuracy | 76.4 | 55.5 | +20.9 ± 9.1 | 0.007 | 0.062 |

### vs EfficientNet-B3 (ImageNet)

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 55.5 | +18.1 ± 10.6 | 0.019 | 0.062 |
| f1_non_urgent | 63.0 | 38.0 | +25.0 ± 13.6 | 0.015 | 0.062 |
| roc_auc | 86.2 | 64.5 | +21.6 ± 13.6 | 0.024 | 0.062 |
| pr_auc_non_urgent | 70.6 | 42.6 | +28.0 ± 17.0 | 0.021 | 0.062 |
| mcc | 50.1 | 16.5 | +33.6 ± 18.4 | 0.015 | 0.062 |
| balanced_accuracy | 76.4 | 58.5 | +17.9 ± 10.0 | 0.016 | 0.062 |

### vs YOLOv8-L COCO-only

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 57.4 | +16.2 ± 6.0 | 0.004 | 0.062 |
| f1_non_urgent | 63.0 | 37.8 | +25.2 ± 13.6 | 0.014 | 0.062 |
| roc_auc | 86.2 | 66.3 | +19.9 ± 5.2 | 0.001 | 0.062 |
| pr_auc_non_urgent | 70.6 | 48.1 | +22.5 ± 11.3 | 0.011 | 0.062 |
| mcc | 50.1 | 16.7 | +33.4 ± 14.0 | 0.006 | 0.062 |
| balanced_accuracy | 76.4 | 58.0 | +18.4 ± 9.3 | 0.011 | 0.062 |

### vs YOLOv8-L from scratch

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 42.4 | +31.2 ± 4.6 | 0.000 | 0.062 |
| f1_non_urgent | 63.0 | 0.0 | +63.0 ± 7.7 | 0.000 | 0.062 |
| roc_auc | 86.2 | 54.8 | +31.4 ± 16.6 | 0.013 | 0.062 |
| pr_auc_non_urgent | 70.6 | 33.8 | +36.8 ± 16.9 | 0.008 | 0.062 |
| mcc | 50.1 | 0.0 | +50.1 ± 9.0 | 0.000 | 0.062 |
| balanced_accuracy | 76.4 | 50.0 | +26.4 ± 6.3 | 0.001 | 0.062 |

### vs Detect-then-count (detector features)

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 70.2 | +3.4 ± 1.6 | 0.009 | 0.062 |
| f1_non_urgent | 63.0 | 59.1 | +4.0 ± 1.6 | 0.006 | 0.062 |
| roc_auc | 86.2 | 83.6 | +2.6 ± 1.9 | 0.039 | 0.125 |
| pr_auc_non_urgent | 70.6 | 65.4 | +5.2 ± 5.0 | 0.081 | 0.062 |
| mcc | 50.1 | 43.7 | +6.4 ± 2.4 | 0.004 | 0.062 |
| balanced_accuracy | 76.4 | 73.6 | +2.8 ± 1.0 | 0.004 | 0.062 |

### vs Detect-then-count + spatial features

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 70.3 | +3.3 ± 6.8 | 0.336 | 0.465 |
| f1_non_urgent | 63.0 | 57.8 | +5.3 ± 11.7 | 0.369 | 0.465 |
| roc_auc | 86.2 | 83.0 | +3.2 ± 3.1 | 0.084 | 0.062 |
| pr_auc_non_urgent | 70.6 | 65.7 | +4.9 ± 4.8 | 0.085 | 0.125 |
| mcc | 50.1 | 45.4 | +4.7 ± 10.5 | 0.374 | 0.465 |
| balanced_accuracy | 76.4 | 73.7 | +2.7 ± 5.6 | 0.335 | 0.273 |

### vs CNN ensemble (YOLO-TL + ResNet + EffNet)

| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| f1_macro | 73.6 | 67.0 | +6.6 ± 3.9 | 0.019 | 0.125 |
| f1_non_urgent | 63.0 | 51.5 | +11.6 ± 5.7 | 0.011 | 0.062 |
| roc_auc | 86.2 | 76.9 | +9.3 ± 7.7 | 0.055 | 0.062 |
| pr_auc_non_urgent | 70.6 | 59.1 | +11.5 ± 5.1 | 0.007 | 0.062 |
| mcc | 50.1 | 35.6 | +14.5 ± 6.2 | 0.006 | 0.062 |
| balanced_accuracy | 76.4 | 68.1 | +8.3 ± 3.0 | 0.004 | 0.062 |

**Pooled out-of-fold bootstrap (n = 287 patients, 5000 resamples) — primary**

### vs YOLO-TL caries-adapted (CNN)

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +9.3 | [+4.3, +14.5] | 0.001 |
| pr_auc_non_urgent | +15.3 | [+6.6, +23.6] | 0.000 |

### vs ResNet50V2 (ImageNet)

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +24.4 | [+16.1, +32.7] | 0.000 |
| pr_auc_non_urgent | +29.4 | [+16.0, +40.6] | 0.000 |

### vs EfficientNet-B3 (ImageNet)

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +22.2 | [+15.2, +29.4] | 0.000 |
| pr_auc_non_urgent | +27.2 | [+15.1, +38.3] | 0.000 |

### vs YOLOv8-L COCO-only

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +20.8 | [+12.7, +28.9] | 0.000 |
| pr_auc_non_urgent | +25.6 | [+13.4, +37.0] | 0.000 |

### vs YOLOv8-L from scratch

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +34.7 | [+26.0, +43.0] | 0.000 |
| pr_auc_non_urgent | +37.6 | [+24.6, +48.6] | 0.000 |

### vs Detect-then-count (detector features)

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +2.9 | [+0.5, +5.2] | 0.014 |
| pr_auc_non_urgent | +7.5 | [-1.2, +14.4] | 0.087 |

### vs Detect-then-count + spatial features

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +5.1 | [+1.6, +8.6] | 0.005 |
| pr_auc_non_urgent | +10.6 | [-0.2, +19.2] | 0.056 |

### vs CNN ensemble (YOLO-TL + ResNet + EffNet)

| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |
|---|---|---|---|
| roc_auc | +9.2 | [+3.1, +15.5] | 0.004 |
| pr_auc_non_urgent | +11.8 | [+1.9, +21.4] | 0.016 |
