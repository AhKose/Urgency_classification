# Reproducing the full pipeline from raw images

Most people will not need this: every reported number, table, and figure can
already be reproduced from `runs/` and `splits/` without any raw images (see
the main [README.md](../README.md)). This document is for reproducing
preprocessing and training from scratch.

## 1. Environment

```bash
pip install -r requirements.txt
```

R (>= 4.5) with the `lme4` package is required only for the GLMM analysis
(`src/mixed_models.py`); everything else is pure Python.

## 2. Obtain the raw images

Download the two public collections and the DENTEX detection benchmark (see
[Data availability](../README.md#data-availability) in the main README for
links).

**Important:** our own working copies had been reorganized by hand before
this study (images sorted into `good` / `bad` / `normal` folders, central
crop applied) — this reorganization is *not* itself scripted in this
repository. To rebuild an equivalent working set:

1. Extract both Figshare collections.
2. For each of the 287 cases in `labels/pediatric_urgency_labels.csv`, locate
   the file at `original_subfolder`/`original_filename` inside the
   collection named in `figshare_dataset_title`.
3. Sort each case into a `bad` (urgent) / `good` or `normal` (non-urgent)
   folder under a working directory of your choice, mirroring the structure
   `config.py` expects (`RAW_DATA`, `HEPSI_DIR`, `SRC_A`, `SRC_B` — see the
   comments in `config.py`). The urgency label to use for sorting is exactly
   `consensus_urgency_label` from the labels file; you do not need to
   re-derive it.
4. Apply a central crop to the dento-alveolar region of each image (Methods
   3.2, step 1) and place the result under `CROP_DIR`.
5. Download DENTEX and place it under `DENTEX_DIR` if you also intend to
   retrain the caries detector from scratch (optional — the fine-tuned
   detector is already included at `model_weights/caries_detector_yolov8l.pt`).

## 3. Preprocessing, splitting, and detection features

```bash
python -m src.data_prep          # dedup + manifest + CLAHE/NLM preprocessing
python -m src.splits             # patient-level nested-CV splits + leakage assertions
python -m src.detection_features # run the frozen detector, build the 12-/20-D feature table
```

## 4. Train every model (5 outer folds each)

```bash
python run_all.py --smoke        # sanity check: 1 model, 1 fold, tiny budget
python run_all.py                # yolo_tl, resnet50v2, efficientnet_b3 (resumable)
python run_all.py --models yolo_coco yolo_scratch   # backbone-pretraining ablation
python -m src.run_detfeat        # detect-then-count (LFR_FEATSET=base|spatial)
python -m src.run_fusion         # late fusion + CNN ensemble (post hoc, no training)
```

Every `(model, fold)` writes `runs/<model>/fold_<k>/done.json`; re-running
skips finished work. Force CPU with `LFR_DEVICE=cpu`.

## 5. Aggregate and analyze

```bash
python -m src.aggregate
python -m src.mixed_models
python -m src.stats_compare
python -m src.extra_analyses
python -m src.operating_point
python -m src.fusion_variants
python -m src.fig_results
python -m src.fig_preprocessing
python -m src.spatial_map
python -m src.build_results_xlsx
python -m src.build_label_release   # only needed if you change the labels or manifest
```
