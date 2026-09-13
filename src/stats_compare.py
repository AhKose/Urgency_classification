"""Statistical comparison of the proposed model vs each baseline.

1. Paired t-test + Wilcoxon on per-fold metrics (n = 5 folds, same folds).
2. Bootstrap 95% CI for the difference in pooled out-of-fold ROC-AUC and
   minority PR-AUC (each patient appears once across the 5 outer test folds,
   so the pooled predictions are a valid dataset-wide OOF set).

With only 5 folds the paired tests are underpowered; the pooled-OOF bootstrap
(n = 287) is the more informative comparison and is reported as primary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

PROPOSED = "det_fusion"
METRICS = ["f1_macro", "f1_non_urgent", "roc_auc", "pr_auc_non_urgent", "mcc", "balanced_accuracy"]
RNG = np.random.default_rng(C.SEED)


def _per_fold(model: str, metric: str) -> np.ndarray:
    out = []
    for k in range(1, C.OUTER_FOLDS + 1):
        d = json.loads((C.RUN_DIR / model / f"fold_{k}" / "done.json").read_text())
        out.append(d["test_metrics"][metric])
    return np.array(out, float)


def _pooled(model: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(C.RUN_DIR / model / f"fold_{k}" / "predictions.csv")
         for k in range(1, C.OUTER_FOLDS + 1)],
        ignore_index=True,
    ).sort_values("patient_id").reset_index(drop=True)


def _bootstrap_delta(y, s_a, s_b, scorer, n=5000):
    """CI for scorer(model_a) - scorer(model_b) on paired pooled OOF predictions."""
    base = scorer(y, s_a) - scorer(y, s_b)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        deltas.append(scorer(y[b], s_a[b]) - scorer(y[b], s_b[b]))
    deltas = np.array(deltas)
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return dict(delta=float(base), ci95=[float(np.percentile(deltas, 2.5)),
                                         float(np.percentile(deltas, 97.5))],
               p_bootstrap=float(p))


def run() -> dict:
    baselines = [m for m in C.ALL_MODELS if m != PROPOSED]
    result = {"proposed": PROPOSED, "per_fold_tests": {}, "pooled_oof_bootstrap": {}}

    for b in baselines:
        pf = {}
        for metric in METRICS:
            a = _per_fold(PROPOSED, metric)
            c = _per_fold(b, metric)
            diff = a - c
            t_p = stats.ttest_rel(a, c).pvalue
            try:
                w_p = stats.wilcoxon(a, c).pvalue
            except ValueError:
                w_p = float("nan")
            pf[metric] = dict(
                proposed_mean=float(a.mean()), baseline_mean=float(c.mean()),
                mean_diff=float(diff.mean()), sd_diff=float(diff.std(ddof=1)),
                per_fold_diff=[float(x) for x in diff],
                paired_t_p=float(t_p), wilcoxon_p=float(w_p),
            )
        result["per_fold_tests"][b] = pf

        pa, pb = _pooled(PROPOSED), _pooled(b)
        assert (pa.patient_id.values == pb.patient_id.values).all()
        y = (pa.True_Label.values == C.URGENCY_TO_INT["non_urgent"]).astype(int)
        sa, sb = pa.Prob_non_urgent.values, pb.Prob_non_urgent.values
        result["pooled_oof_bootstrap"][b] = dict(
            roc_auc=_bootstrap_delta(y, sa, sb, roc_auc_score),
            pr_auc_non_urgent=_bootstrap_delta(y, sa, sb, average_precision_score),
        )

    (C.RESULT_DIR / "stats_comparison.json").write_text(json.dumps(result, indent=2))
    _md(result)
    return result


def _md(r: dict) -> None:
    L = ["# Statistical comparison — proposed (YOLO-TL) vs baselines", "",
         "**Per-fold paired tests (n = 5 folds — underpowered, directional only)**", ""]
    for b, pf in r["per_fold_tests"].items():
        L.append(f"### vs {C.MODEL_DISPLAY[b]}")
        L.append("")
        L.append("| metric | proposed | baseline | Δ (mean±sd) | paired t p | Wilcoxon p |")
        L.append("|---|---|---|---|---|---|")
        for m, d in pf.items():
            L.append(f"| {m} | {d['proposed_mean']*100:.1f} | {d['baseline_mean']*100:.1f} "
                     f"| {d['mean_diff']*100:+.1f} ± {d['sd_diff']*100:.1f} "
                     f"| {d['paired_t_p']:.3f} | {d['wilcoxon_p']:.3f} |")
        L.append("")
    L.append("**Pooled out-of-fold bootstrap (n = 287 patients, 5000 resamples) — primary**")
    L.append("")
    for b, d in r["pooled_oof_bootstrap"].items():
        L.append(f"### vs {C.MODEL_DISPLAY[b]}")
        L.append("")
        L.append("| metric | Δ (proposed − baseline) | 95% CI | bootstrap p |")
        L.append("|---|---|---|---|")
        for m, dd in d.items():
            L.append(f"| {m} | {dd['delta']*100:+.1f} | "
                     f"[{dd['ci95'][0]*100:+.1f}, {dd['ci95'][1]*100:+.1f}] | {dd['p_bootstrap']:.3f} |")
        L.append("")
    (C.RESULT_DIR / "stats_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    run()
