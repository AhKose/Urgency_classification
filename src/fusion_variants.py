"""Sensitivity of the det_count + CNN fusion to the combination rule.

Compared against the reported equal-weight probability average (weight 0.5):
  prob_avg   : 0.5 * P_det + 0.5 * P_cnn                     (reported)
  logit_avg  : sigmoid( mean of the two logits )
  rank_avg   : mean of the two within-fold percentile ranks  (calibration-free)
  calib_avg  : isotonic-calibrate each model on final_val, then average probs

All post-hoc from saved per-fold test predictions (+ det_count val predictions,
+ re-scored YOLO-TL val predictions). No tuned weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.evaluate import compute_metrics


def _clip(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def run() -> dict:
    from src.run_fusion import _val_probs_cnn

    man = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")
    folds = json.loads(C.SPLIT_JSON.read_text())["folds"]
    variants = ["prob_avg", "logit_avg", "rank_avg", "calib_avg"]
    per_fold = {v: [] for v in variants}

    for f in folds:
        k = f["fold"]
        det_te = pd.read_csv(C.RUN_DIR / "det_count" / f"fold_{k}" / "predictions.csv").set_index("patient_id")
        cnn_te = pd.read_csv(C.RUN_DIR / "yolo_tl" / f"fold_{k}" / "predictions.csv").set_index("patient_id")
        ids = list(det_te.index)
        y = man.loc[ids, "urgency_int"].to_numpy(int)
        pd_te = _clip(det_te.loc[ids, "Prob_urgent"].to_numpy())
        pc_te = _clip(cnn_te.loc[ids, "Prob_urgent"].to_numpy())

        det_v = pd.read_csv(C.RUN_DIR / "det_count" / f"fold_{k}" / "val_predictions.csv")
        vids = det_v.patient_id.tolist()
        yv = man.loc[vids, "urgency_int"].to_numpy(int)
        pd_v = _clip(det_v.Prob_urgent.to_numpy())
        cnn_v_map = _val_probs_cnn("yolo_tl", k, vids)
        pc_v = _clip(np.array([cnn_v_map[i] for i in vids]))

        combos = {}
        combos["prob_avg"] = 0.5 * pd_te + 0.5 * pc_te
        combos["logit_avg"] = expit(0.5 * (logit(pd_te) + logit(pc_te)))
        rd = pd.Series(pd_te).rank(pct=True).to_numpy()
        rc = pd.Series(pc_te).rank(pct=True).to_numpy()
        combos["rank_avg"] = 0.5 * rd + 0.5 * rc
        # isotonic calibration: fit P(urgent) -> observed urgent(=1-y since y:0=urgent)
        iso_d = IsotonicRegression(out_of_bounds="clip").fit(pd_v, (yv == 0).astype(int))
        iso_c = IsotonicRegression(out_of_bounds="clip").fit(pc_v, (yv == 0).astype(int))
        combos["calib_avg"] = 0.5 * iso_d.predict(pd_te) + 0.5 * iso_c.predict(pc_te)

        for v, p in combos.items():
            yp = (p < 0.5).astype(int)
            per_fold[v].append(compute_metrics(y, yp, p))

    out = {}
    for v in variants:
        m = per_fold[v]
        agg = {}
        for key in ("f1_macro", "roc_auc", "pr_auc_non_urgent", "balanced_accuracy", "mcc", "f1_non_urgent"):
            vals = np.array([x[key] for x in m], float)
            agg[key] = dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                            values=[round(float(x), 4) for x in vals])
        out[v] = agg
    (C.RESULT_DIR / "fusion_variants.json").write_text(json.dumps(out, indent=2))
    _md(out)
    return out


def _md(out: dict) -> None:
    L = ["# Fusion-rule sensitivity (det_count + YOLO-TL CNN)", "",
         "Mean ± sd over 5 outer folds. `prob_avg` (equal-weight probability "
         "average) is the rule reported in the main results. No tuned weights.", "",
         "| combination rule | macro-F1 | ROC-AUC | PR-AUC non-urgent | balanced acc | MCC |",
         "|---|---|---|---|---|---|"]
    names = {"prob_avg": "probability average (0.5) — reported",
             "logit_avg": "logit average", "rank_avg": "rank average (calibration-free)",
             "calib_avg": "isotonic-calibrate then average"}
    for v, a in out.items():
        L.append(f"| {names[v]} "
                 f"| {a['f1_macro']['mean']*100:.1f} ± {a['f1_macro']['sd']*100:.1f} "
                 f"| {a['roc_auc']['mean']*100:.1f} ± {a['roc_auc']['sd']*100:.1f} "
                 f"| {a['pr_auc_non_urgent']['mean']*100:.1f} ± {a['pr_auc_non_urgent']['sd']*100:.1f} "
                 f"| {a['balanced_accuracy']['mean']*100:.1f} ± {a['balanced_accuracy']['sd']*100:.1f} "
                 f"| {a['mcc']['mean']*100:.1f} ± {a['mcc']['sd']*100:.1f} |")
    L.append("")
    spread = max(a["f1_macro"]["mean"] for a in out.values()) - min(a["f1_macro"]["mean"] for a in out.values())
    L.append(f"macro-F1 spread across the four rules: {spread*100:.1f} points — "
             f"{'within fold-to-fold noise; the simplest rule (probability average) is retained' if spread < 0.02 else 'non-trivial'}.")
    (C.RESULT_DIR / "fusion_variants.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
