"""Metric computation. `urgent` = class 0, `non_urgent` (minority) = class 1."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)

URGENT, NON_URGENT = 0, 1


def compute_metrics(y_true, y_pred, prob_urgent) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    prob_urgent = np.asarray(prob_urgent, dtype=float)
    prob_non = 1.0 - prob_urgent

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[URGENT, NON_URGENT], zero_division=0
    )
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_urgent": float(p[0]), "recall_urgent": float(r[0]),
        "f1_urgent": float(f[0]), "support_urgent": int(s[0]),
        "precision_non_urgent": float(p[1]), "recall_non_urgent": float(r[1]),
        "f1_non_urgent": float(f[1]), "support_non_urgent": int(s[1]),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob_non))            # score for class 1
        out["pr_auc_non_urgent"] = float(average_precision_score(y_true == NON_URGENT, prob_non))
        out["pr_auc_urgent"] = float(average_precision_score(y_true == URGENT, prob_urgent))
    else:
        out["roc_auc"] = out["pr_auc_non_urgent"] = out["pr_auc_urgent"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[URGENT, NON_URGENT])
    out["confusion_matrix"] = cm.tolist()   # rows = true [urgent, non_urgent], cols = pred
    return out


def aggregate(fold_metrics: list[dict]) -> dict:
    """mean, sd, and 95% CI (t, df=n-1) over folds for every scalar metric."""
    from scipy import stats

    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    n = len(fold_metrics)
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            agg[k] = dict(mean=float("nan"), sd=float("nan"), ci95=[float("nan")] * 2, values=[])
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            half = float(stats.t.ppf(0.975, len(vals) - 1) * sd / np.sqrt(len(vals)))
        else:
            half = 0.0
        agg[k] = dict(mean=mean, sd=sd, ci95=[mean - half, mean + half],
                      values=[float(v) for v in vals])
    cms = np.array([m["confusion_matrix"] for m in fold_metrics])
    agg["confusion_matrix_sum"] = cms.sum(0).tolist()
    agg["confusion_matrix_mean"] = cms.mean(0).tolist()
    agg["n_folds"] = n
    return agg
