# CV-aware statistical comparison

Reference = **proposed (Detect-then-count + CNN, late fusion)**. Folds share ~3/4 of training data, so per-fold t-tests under-estimate variance; the analyses below respect the CV structure.

## 1. GLMM — `correct ~ model + (1|patient) + (1|fold)`

Odds ratio < 1 ⇒ that baseline is **less** likely to be correct than the proposed model.

| baseline | odds ratio (95% CI) | z | p |
|---|---|---|---|
| Detect-then-count (detector features) | 0.77 [0.50, 1.19] | -1.18 | 0.239 |
| Detect-then-count + spatial features | 0.87 [0.56, 1.36] | -0.60 | 0.552 |
| EfficientNet-B3 (ImageNet) | 0.38 [0.25, 0.59] | -4.41 | 0.000 |
| CNN ensemble (YOLO-TL + ResNet + EffNet) | 0.77 [0.50, 1.19] | -1.18 | 0.239 |
| ResNet50V2 (ImageNet) | 0.25 [0.17, 0.39] | -6.36 | 0.000 |
| YOLOv8-L COCO-only | 0.46 [0.30, 0.70] | -3.59 | 0.000 |
| YOLOv8-L from scratch | 0.71 [0.46, 1.10] | -1.52 | 0.129 |
| YOLO-TL caries-adapted (CNN) | 0.66 [0.43, 1.02] | -1.86 | 0.063 |

Variance components: patient_id=2.307, fold=0.000

## 2. Cluster bootstrap — pooled-OOF AUC difference (resample folds, then patients)

| baseline | Δ ROC-AUC (95% CI) | p | Δ PR-AUC non-urgent (95% CI) | p |
|---|---|---|---|---|
| YOLO-TL caries-adapted (CNN) | +9.3 [+3.5, +17.9] | 0.002 | +15.3 [+5.9, +24.5] | 0.002 |
| ResNet50V2 (ImageNet) | +24.4 [+11.4, +35.4] | 0.000 | +29.4 [+12.8, +42.1] | 0.000 |
| EfficientNet-B3 (ImageNet) | +22.2 [+9.6, +35.9] | 0.000 | +27.2 [+10.7, +45.4] | 0.001 |
| YOLOv8-L COCO-only | +20.8 [+11.0, +30.4] | 0.000 | +25.6 [+10.7, +37.9] | 0.001 |
| YOLOv8-L from scratch | +34.7 [+21.4, +47.2] | 0.000 | +37.6 [+20.0, +53.3] | 0.000 |
| Detect-then-count (detector features) | +2.9 [-0.1, +6.2] | 0.063 | +7.5 [-1.9, +15.1] | 0.150 |
| Detect-then-count + spatial features | +5.1 [+0.4, +9.3] | 0.030 | +10.6 [-1.0, +18.7] | 0.083 |
| CNN ensemble (YOLO-TL + ResNet + EffNet) | +9.2 [+2.3, +17.5] | 0.009 | +11.8 [+1.8, +21.0] | 0.021 |

## 3. Nadeau–Bengio corrected spread — per-fold macro-F1

| model | mean | naive sd | corrected sd |
|---|---|---|---|
| YOLO-TL caries-adapted (CNN) | 62.7 | 9.7 | 14.5 |
| ResNet50V2 (ImageNet) | 51.0 | 9.0 | 13.5 |
| EfficientNet-B3 (ImageNet) | 55.5 | 7.3 | 10.9 |
| YOLOv8-L COCO-only | 57.4 | 5.1 | 7.7 |
| YOLOv8-L from scratch | 42.4 | 0.2 | 0.3 |
| Detect-then-count (detector features) | 70.2 | 3.2 | 4.8 |
| Detect-then-count + spatial features | 70.3 | 9.8 | 14.7 |
| Detect-then-count + CNN (late fusion, proposed) | 73.6 | 4.6 | 6.9 |
| CNN ensemble (YOLO-TL + ResNet + EffNet) | 67.0 | 5.6 | 8.4 |