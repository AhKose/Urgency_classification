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

## 2. Obtain the raw images and build the working dataset

Download and extract the two public collections (see
[Data availability](../README.md#data-availability) in the main README for
links):

- **Source A**, "Children's teeth - supplement": extract it, and note the
  folder that directly contains `img/` (the archive wraps everything in one
  extra top-level folder — pass the folder *containing* `img/`, not the
  archive root).
- **Source B**, "Children's Dental Panoramic Radiographs Dataset": extract
  it, and note the folder that directly contains
  `Children's dental caries segmentation dataset/`.

Then build `data/manifest.csv` and the preprocessed image set in one step:

```bash
python -m src.build_dataset_from_release \
    --source-a /path/to/extracted/source_a \
    --source-b /path/to/extracted/source_b
```

This locates every one of the 287 cases by the exact file name published in
each collection (`labels/pediatric_urgency_labels.csv`), applies the same
central crop used throughout the study (`data/crop_lookup.csv`, verified
byte-for-byte against the crops used for the reported results, no manual
step), then the same resize/CLAHE/NLM preprocessing as Methods 3.2, and
writes `data/manifest.csv` with `patient_id` set to the public `case_id` —
no internal identifier is introduced at this stage. Any case it cannot find
is reported by path so you can check `--source-a`/`--source-b`.

Download DENTEX and place it under `DENTEX_DIR` only if you also intend to
retrain the caries detector from scratch (optional — the fine-tuned detector
is already included at `model_weights/caries_detector_yolov8l.pt`).

## 3. Splitting and detection features

```bash
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
python -m src.spatial_map
python -m src.build_results_xlsx
```
