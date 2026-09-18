"""Central configuration for the leak-free pediatric caries-urgency pipeline.

All paths are relative to the repository root, so this works regardless of
where the repository is cloned.
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT

# Raw source imagery --------------------------------------------------------- #
# NOT included in this repository (the two source collections have their own
# public licences on Figshare; see labels/README.md). To reconstruct the
# working set and re-run preprocessing from scratch, download the two public
# collections and pass their extracted paths to
# `src/build_dataset_from_release.py --source-a ... --source-b ...`
# (see docs/REPRODUCING.md). Most users do not need any of this -- see
# "Using the released results without re-running anything" in README.md.
RAW_DATA = ROOT / "raw_data"
DENTEX_DIR = RAW_DATA / "dentex"                            # DENTEX (adult) detection data, for retraining the detector

# Domain-adapted caries detector (YOLOv8-L, single class 'Caries', mAP@0.5~0.70)
# fine-tuned on DENTEX (see Methods); included in this repository.
DETECTOR_WEIGHTS = ROOT / "model_weights" / "caries_detector_yolov8l.pt"

# COCO-pretrained YOLOv8-L used to initialize the detector and the COCO-only
# ablation. Not shipped here (it's the standard public Ultralytics checkpoint);
# passing the bare name lets `ultralytics` download it automatically on first use.
COCO_WEIGHTS = "yolov8l.pt"

# Outputs ------------------------------------------------------------------- #
DATA_DIR = PROJECT / "data"
PREPROC_DIR = DATA_DIR / "preprocessed"                    # {urgent, non_urgent} -- not shipped, rebuilt by src/build_dataset_from_release.py
MANIFEST_CSV = DATA_DIR / "manifest.csv"                   # not shipped -- rebuilt by src/build_dataset_from_release.py, see README.md
SPLIT_DIR = PROJECT / "splits"
SPLIT_JSON = SPLIT_DIR / "nested_cv.json"                  # included: the exact 5 outer folds used for every result in the paper
FEATURE_DIR = PROJECT / "data" / "features"                # cached YOLO features
RUN_DIR = PROJECT / "runs"                                 # included: per-fold configs, metrics, predictions (no model weights)
RESULT_DIR = PROJECT / "results"                           # included: all aggregate tables, stats, figures
REPORT_DIR = PROJECT / "reports"

for _d in (DATA_DIR, SPLIT_DIR, FEATURE_DIR, RUN_DIR, RESULT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Data / preprocessing
# --------------------------------------------------------------------------- #
CLASS3_TO_URGENCY = {"bad": "urgent", "good": "non_urgent", "normal": "non_urgent"}
URGENCY_TO_INT = {"urgent": 0, "non_urgent": 1}            # URGENT = 0, NON_URGENT = 1 throughout
INT_TO_URGENCY = {v: k for k, v in URGENCY_TO_INT.items()}
MINORITY_URGENCY = "non_urgent"                            # the class we care about

# Preprocessing pipeline (Methods 3.2)
RESIZE_WH = (1200, 800)          # (width, height), bicubic
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
NLM_H = 10                       # cv2.fastNlMeansDenoising strength
NLM_TEMPLATE = 7
NLM_SEARCH = 21

MODEL_INPUT = 512               # network input (square), ImageNet norm.
# A larger 640 input has negligible expected effect at this dataset size;
# 512 keeps nested cross-validation tractable on a single 6 GB GPU.
# Override via env LFR_INPUT if desired.
MODEL_INPUT = int(os.environ.get("LFR_INPUT", MODEL_INPUT))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Nested cross-validation
# --------------------------------------------------------------------------- #
OUTER_FOLDS = 5
INNER_VAL_FRACTION = 0.20        # single stratified hold-out inside each outer-train
FINAL_VAL_FRACTION = 0.20        # train/val split for the final per-fold refit
N_OPTUNA_TRIALS = 15             # identical budget for every model, including baselines
OPTUNA_OBJECTIVE = "f1_macro"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODELS = ("yolo_tl", "resnet50v2", "efficientnet_b3")
# Backbone-pretraining ablation (same YOLOv8-L architecture + head + protocol):
#   yolo_tl      = COCO -> adult-caries-detection fine-tune (transfer-learning classifier)
#   yolo_coco    = COCO only  (isolates the contribution of caries pretraining)
#   yolo_scratch = random init  (isolates the contribution of ANY pretraining)
ABLATION_MODELS = ("yolo_coco", "yolo_scratch")
# Tabular 'detect-then-count' model + late fusion (the proposed configuration)
DETFEAT_MODELS = ("det_count", "det_count_sp", "det_fusion", "ens_cnn")
ALL_MODELS = MODELS + ABLATION_MODELS + DETFEAT_MODELS
MODEL_DISPLAY = {
    "yolo_tl": "YOLO-TL caries-adapted (CNN)",
    "resnet50v2": "ResNet50V2 (ImageNet)",
    "efficientnet_b3": "EfficientNet-B3 (ImageNet)",
    "yolo_coco": "YOLOv8-L COCO-only",
    "yolo_scratch": "YOLOv8-L from scratch",
    "det_count": "Detect-then-count (detector features)",
    "det_count_sp": "Detect-then-count + spatial features",
    "det_fusion": "Detect-then-count + CNN (late fusion, proposed)",
    "ens_cnn": "CNN ensemble (YOLO-TL + ResNet + EffNet)",
}
TIMM_NAME = {
    "resnet50v2": "resnetv2_50",
    "efficientnet_b3": "efficientnet_b3",
}

# Discrete Optuna search space (Methods Table 2, applied identically to every model)
SEARCH_SPACE = {
    "dropout_rate":    [0.3, 0.5],
    "unfreeze_level":  [0, 1, 2],
    "lr":              [1e-5, 3e-5, 1e-4, 3e-4],
    "weight_decay":    [1e-4, 1e-3, 5e-3],
    "label_smoothing": [0.0, 0.1],
    "use_focal_loss":  [False, True],
    "scheduler":       ["onecycle", "plateau"],
    "batch_size":      [8, 16],
}

# Fixed training schedule (progressive 3-stage unfreezing)
STAGE_EPOCHS = (5, 15, 20)       # stage I (head), II (late), III (mid+late)
MAX_EPOCHS_PER_FOLD = 40
GRAD_CLIP_L2 = 1.0
EARLY_STOP_PATIENCE = 10
AMP = True
NUM_WORKERS = 4

DEVICE = os.environ.get("LFR_DEVICE", "cuda")   # set LFR_DEVICE=cpu to force CPU
