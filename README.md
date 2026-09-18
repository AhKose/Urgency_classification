# Pediatric Dental Caries Urgency Classification

Code, released labels, and full result artifacts for the paper **"DETECT-THEN-COUNT AND LATE FUSION FOR PEDIATRIC CARIES URGENCY CLASSIFICATION IN PANORAMIC RADIOGRAPHS."** The task: classify a pediatric panoramic radiograph as
**urgent** or **non-urgent** from its overall radiographic caries burden,
using only case-level (image-level) labels, no lesion-level annotation.

- **287** pediatric panoramic radiographs, two public sources, **211 urgent /
  76 non-urgent**
- Case-level labels by **three-rater dentist consensus** (Cohen's κ = 0.88)
- Patient-level, **nested** five-fold cross-validation (no leakage: every
  outer test fold is scored exactly once, after all hyperparameter selection)
- Proposed model (**detect-then-count + CNN, late fusion**): **macro-F1
  73.6 ± 4.6%**, **ROC-AUC 86.2 ± 5.4%**, significantly ahead of every
  end-to-end CNN baseline (see `results/stats_comparison.md`)

## Two ways to use this repository

**1. Inspect / re-analyze the reported results — no data download needed.**
Every number, table, and figure in the paper is reproducible from the files
already in this repository (`runs/`, `results/`, `splits/`): per-fold
predictions, configurations, metrics, and all statistical comparisons. See
[Re-running the analysis only](#re-running-the-analysis-only).

**2. Reproduce the full pipeline from raw images.** Requires downloading the
two public radiograph collections yourself (not redistributed here, see
[Data availability](#data-availability)). One command
(`python -m src.build_dataset_from_release`) locates every case by its
published file name, applies the exact preprocessing pipeline (the central
crop is reproduced byte-for-byte, see `data/crop_lookup.csv`), and builds
`data/manifest.csv` with no manual sorting step and no internal identifier —
see [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Repository structure

```
config.py                   central configuration (paths, seeds, search space)
run_all.py                  entry point for the full training pipeline
src/                        all pipeline code (see docs/REPRODUCING.md)
model_weights/
  caries_detector_yolov8l.pt   the domain-adapted caries detector (YOLOv8-L,
                                fine-tuned on DENTEX; used frozen throughout)
splits/nested_cv.json       the exact 5 outer folds (+ inner/final splits)
                             used for every result in the paper
runs/<model>/fold_<k>/      per-fold, per-model: best_config.json, metrics.json,
                             predictions.csv, val_predictions.csv, optuna_trials.csv
                             (no model weights, to keep the repo small)
results/                    aggregated tables, statistical comparisons, figures
labels/                     the public label release (see below)
reports/                    consolidated result workbooks (.xlsx)
data/                       small derived files (detection features, dedup log,
                            crop_lookup.csv used to reproduce preprocessing exactly)
docs/                       data-provenance verification, reproduction guide
```

## Labels

[`labels/pediatric_urgency_labels.csv`](labels/pediatric_urgency_labels.csv)
(also as `.xlsx`) gives the urgent / non-urgent consensus label for every one
of the 287 cases, keyed to the **exact file names published in the two
source Figshare collections** — not to any internal working name. Every
entry was verified by downloading both public archives and matching file
content (CRC32), not by assuming file names had been preserved; see
[`labels/README.md`](labels/README.md) and
[`docs/DATA_PROVENANCE_VERIFICATION.md`](docs/DATA_PROVENANCE_VERIFICATION.md)
for exactly how.

## Re-running the analysis only

No raw images or downloads required; everything below reads from `runs/` and
`splits/`, which are included.

```bash
pip install -r requirements.txt
python -m src.aggregate          # rebuilds results/REPORT.md, aggregated.json, figures
python -m src.mixed_models       # GLMM (needs R + lme4, see requirements.txt) + cluster bootstrap
python -m src.stats_compare      # paired per-fold tests
python -m src.extra_analyses     # calibration, trivial baselines, subgroup, Bayesian t-test
python -m src.fig_results        # Figure: model comparison bars, ROC/PR curves
python -m src.build_results_xlsx # reports/results_all_models.xlsx
```

## Reproducing the full pipeline

See [docs/REPRODUCING.md](docs/REPRODUCING.md): environment setup, how to
obtain the raw images, preprocessing, nested-CV training for every model,
and the detect-then-count / fusion pipeline.

## Data availability

This repository does **not** redistribute the original radiographs. The
pediatric cohort was assembled from two public Figshare collections, and the
domain-adapted detector was fine-tuned on the public DENTEX benchmark:

- Children's teeth - supplement: <https://springernature.figshare.com/articles/dataset/Children_s_teeth_-_supplement/22495645>
- Children's Dental Panoramic Radiographs Dataset: <https://springernature.figshare.com/articles/dataset/Children_s_Dental_Panoramic_Radiographs_Dataset/21621705>
- DENTEX: Hamamci et al., 2023 (arXiv:2305.19112); dataset DOI in the paper's references.

Case-level urgency labels for the 287 pediatric images are released in
`labels/`, keyed to the file names in the two collections above.

## Citation

See [CITATION.cff](CITATION.cff). If you use the code, the released labels,
or the results in this repository, please cite the paper.

## License

Code and released labels: [MIT](LICENSE). This does not cover the original
radiograph images, which remain governed by their own publishers' licences
(links above).
