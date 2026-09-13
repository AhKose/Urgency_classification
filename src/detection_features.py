"""Detect-then-count features.

Run the domain-adapted caries detector (best.pt, trained on ADULT DENTEX only -
never on these 287 pediatric images or on any urgency label) over every
preprocessed pediatric radiograph, and engineer per-image features that mirror
the clinical labelling rule (number and size/'depth' of visible carious lesions).

No leakage: the detector is fixed and label-agnostic; features are deterministic
and can be precomputed once for all patients, like the preprocessed images.

Output: data/detection_features.csv  (patient_id + FEATURE_COLS)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

CONF_LOW = 0.15          # keep faint detections
CONF_HIGH = 0.40         # "clear" lesion
AREA_LARGE = 0.010       # normalised box area treated as a large / possibly deep lesion
IMGSZ = 1024             # match the detector's training resolution

FEATURE_COLS = [
    "det_n", "det_n_high", "det_n_large",
    "det_conf_max", "det_conf_mean", "det_conf_sum",
    "det_area_max", "det_area_mean", "det_area_sum",
    "det_severity",          # sum(conf * sqrt(area)) - size-weighted burden
    "det_spread", "det_nquad",   # spatial dispersion, number of image quadrants hit
]
SPATIAL_COLS = [
    "reg_max_n",            # most detections in any one quadrant
    "reg_max_confsum",      # highest confidence-sum in any quadrant
    "reg_max_severity",     # highest severity in any quadrant
    "reg_n_active",         # quadrants with >= 2 detections
    "reg_lr_asym",          # |left - right| / total    (left/right arch imbalance)
    "reg_ud_asym",          # |upper - lower| / total    (maxilla vs mandible)
    "reg_posterior_frac",   # fraction of detections in posterior sextants
    "reg_entropy",          # Shannon entropy of the detection count over quadrants
]
ALL_FEATURE_COLS = FEATURE_COLS + SPATIAL_COLS


def _features_from(boxes_xywhn: np.ndarray, confs: np.ndarray) -> dict:
    """boxes_xywhn: (N,4) normalised cx,cy,w,h ; confs: (N,)"""
    if len(confs) == 0:
        return {c: 0.0 for c in ALL_FEATURE_COLS}
    area = boxes_xywhn[:, 2] * boxes_xywhn[:, 3]
    cx, cy = boxes_xywhn[:, 0], boxes_xywhn[:, 1]
    sev = confs * np.sqrt(area)
    quad = (cx >= 0.5).astype(int) + 2 * (cy >= 0.5).astype(int)     # 0..3
    spread = float(np.sqrt(cx.var() + cy.var())) if len(confs) > 1 else 0.0

    # per-quadrant aggregates
    qn = np.array([(quad == q).sum() for q in range(4)], float)
    qcs = np.array([confs[quad == q].sum() for q in range(4)], float)
    qsev = np.array([sev[quad == q].sum() for q in range(4)], float)
    n = float(len(confs))
    left = float((cx < 0.5).sum()); right = n - left
    upper = float((cy < 0.5).sum()); lower = n - upper
    posterior = float((np.abs(cx - 0.5) > 0.25).sum())
    p = qn / qn.sum() if qn.sum() > 0 else np.zeros(4)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    return {
        "det_n": n,
        "det_n_high": float((confs >= CONF_HIGH).sum()),
        "det_n_large": float((area >= AREA_LARGE).sum()),
        "det_conf_max": float(confs.max()),
        "det_conf_mean": float(confs.mean()),
        "det_conf_sum": float(confs.sum()),
        "det_area_max": float(area.max()),
        "det_area_mean": float(area.mean()),
        "det_area_sum": float(area.sum()),
        "det_severity": float(sev.sum()),
        "det_spread": spread,
        "det_nquad": float(len(np.unique(quad))),
        "reg_max_n": float(qn.max()),
        "reg_max_confsum": float(qcs.max()),
        "reg_max_severity": float(qsev.max()),
        "reg_n_active": float((qn >= 2).sum()),
        "reg_lr_asym": abs(left - right) / n,
        "reg_ud_asym": abs(upper - lower) / n,
        "reg_posterior_frac": posterior / n,
        "reg_entropy": entropy,
    }


def build() -> pd.DataFrame:
    from ultralytics import YOLO
    from src.common import imread_gray

    man = pd.read_csv(C.MANIFEST_CSV)
    det = YOLO(str(C.DETECTOR_WEIGHTS))
    rows = []
    for _, r in man.iterrows():
        img = imread_gray(r["preproc_path"])
        img3 = np.repeat(img[:, :, None], 3, axis=2)
        res = det.predict(img3, imgsz=IMGSZ, conf=CONF_LOW, verbose=False, device=C.DEVICE)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xywhn = res.boxes.xywhn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
        else:
            xywhn = np.zeros((0, 4)); confs = np.zeros((0,))
        feat = _features_from(xywhn, confs)
        feat["patient_id"] = r["patient_id"]
        rows.append(feat)
    df = pd.DataFrame(rows)[["patient_id"] + ALL_FEATURE_COLS]
    df.to_csv(C.DATA_DIR / "detection_features.csv", index=False)
    return df


def _sanity(df: pd.DataFrame) -> None:
    man = pd.read_csv(C.MANIFEST_CSV)[["patient_id", "urgency"]]
    m = df.merge(man, on="patient_id")
    y = (m.urgency == "urgent").astype(int)
    from sklearn.metrics import roc_auc_score
    print("univariate ROC-AUC of each detection feature vs urgency (all 287, illustrative):")
    for c in ALL_FEATURE_COLS:
        try:
            a = roc_auc_score(y, m[c])
            print(f"  {c:16s} AUC={max(a, 1-a):.3f}  (mean urgent={m.loc[y==1,c].mean():.3f}  non={m.loc[y==0,c].mean():.3f})")
        except Exception:
            pass
    # a quick logistic-regression 5-fold CV on detection features ALONE
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    X = m[ALL_FEATURE_COLS].to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    pred = cross_val_predict(pipe, X, y, cv=5)
    prob = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
    print(f"\ndetection-features-ONLY, 5-fold logistic regression (illustrative, not the study protocol):")
    print(f"  macro-F1 = {f1_score(y, pred, average='macro'):.3f}   ROC-AUC = {roc_auc_score(y, prob):.3f}")


if __name__ == "__main__":
    df = build()
    print(f"wrote data/detection_features.csv  ({len(df)} rows)")
    print(df[FEATURE_COLS].describe().round(3).to_string())
    print()
    _sanity(df)
