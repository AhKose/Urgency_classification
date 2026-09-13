"""Rigorous statistical comparison that respects the cross-validation structure.

Problem: the 5 outer folds share ~3/4 of their training data, so per-fold metrics
are correlated and the naive `sd/sqrt(5)` / paired-t under-estimate the variance
(Bengio & Grandvalet 2004; Nadeau & Bengio 2003). The pooled per-patient
predictions, however, are one-per-patient (the outer test folds are a partition).

This module therefore reports:

1. **GLMM** (mixed-effects logistic regression, via R lme4):
       correct ~ model + (1 | patient_id) + (1 | fold)
   `(1|patient_id)` absorbs the same-patient-scored-by-every-model correlation,
   `(1|fold)` absorbs fold-level clustering. The `model` fixed effect gives the
   proposed-vs-baseline contrast (odds ratio) with a valid SE.

2. **Cluster bootstrap** for the pooled-OOF ROC-AUC / minority PR-AUC difference:
   resample folds with replacement, then patients within each resampled fold,
   recompute the delta. Wider (honest) CIs than a flat patient bootstrap.

3. **Nadeau-Bengio corrected** SD for the headline per-fold macro-F1.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

PROPOSED = "det_fusion"
RNG = np.random.default_rng(C.SEED)
RSCRIPT = "Rscript"
GLMM_R = Path(__file__).parent / "rscripts" / "glmm.R"


# --------------------------------------------------------------------------- #
def _completed_models() -> list[str]:
    out = []
    for m in C.ALL_MODELS:
        if all((C.RUN_DIR / m / f"fold_{k}" / "done.json").exists() for k in range(1, C.OUTER_FOLDS + 1)):
            out.append(m)
    return out


def _long_table(models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        for k in range(1, C.OUTER_FOLDS + 1):
            df = pd.read_csv(C.RUN_DIR / m / f"fold_{k}" / "predictions.csv")
            df["model"] = m
            df["fold"] = k
            df["correct"] = (df.True_Label == df.Pred).astype(int)
            df["true_non_urgent"] = (df.True_Label == C.URGENCY_TO_INT["non_urgent"]).astype(int)
            rows.append(df[["patient_id", "fold", "model", "correct",
                            "true_non_urgent", "Prob_non_urgent"]])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
def _glmm(long: pd.DataFrame) -> dict:
    csv_in = C.RESULT_DIR / "_glmm_input.csv"
    csv_out = C.RESULT_DIR / "glmm_coefficients.csv"
    long.to_csv(csv_in, index=False)
    try:
        p = subprocess.run([RSCRIPT, str(GLMM_R), str(csv_in), str(csv_out), PROPOSED],
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            return {"error": (p.stderr or p.stdout)[-2000:]}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    co = pd.read_csv(csv_out)
    vc = pd.read_csv(str(csv_out).replace(".csv", "_varcomp.csv"))
    out = {"reference": PROPOSED, "variance_components": vc.to_dict("records"), "contrasts": {}}
    for _, r in co.iterrows():
        if not str(r["term"]).startswith("model"):
            continue
        baseline = str(r["term"])[len("model"):]
        or_ = float(np.exp(r["estimate"]))
        out["contrasts"][baseline] = dict(
            log_odds=float(r["estimate"]), odds_ratio=or_,
            or_ci95=[float(np.exp(r["ci_low"])), float(np.exp(r["ci_high"]))],
            z=float(r["z"]), p_value=float(r["p_value"]),
            direction="proposed better" if r["estimate"] < 0 else "baseline better",
        )
    return out


# --------------------------------------------------------------------------- #
def _cluster_bootstrap(long: pd.DataFrame, baseline: str, scorer, n: int = 4000) -> dict:
    a = long[long.model == PROPOSED].set_index("patient_id")
    b = long[long.model == baseline].set_index("patient_id")
    a = a.loc[b.index]
    folds = sorted(long.fold.unique())
    by_fold = {f: b.index[b.fold == f].tolist() for f in folds}

    def score(idx):
        y = b.loc[idx, "true_non_urgent"].to_numpy()
        if len(np.unique(y)) < 2:
            return None
        return scorer(y, a.loc[idx, "Prob_non_urgent"].to_numpy()) - \
               scorer(y, b.loc[idx, "Prob_non_urgent"].to_numpy())

    base = score(list(b.index))
    deltas = []
    for _ in range(n):
        chosen_folds = RNG.choice(folds, len(folds), replace=True)
        idx = []
        for f in chosen_folds:
            pool = by_fold[f]
            idx += list(RNG.choice(pool, len(pool), replace=True))
        d = score(idx)
        if d is not None:
            deltas.append(d)
    deltas = np.array(deltas)
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return dict(delta=float(base),
               ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
               p_cluster_bootstrap=float(p))


# --------------------------------------------------------------------------- #
def _nadeau_bengio(models: list[str]) -> dict:
    """Corrected SD of k-fold macro-F1: multiply naive var by (1/k + test/train)."""
    k = C.OUTER_FOLDS
    n_test_over_train = (1 / (k - 1))          # |test| / |train| for k-fold
    corr = 1 / k + n_test_over_train
    out = {}
    for m in models:
        vals = np.array([json.loads((C.RUN_DIR / m / f"fold_{i}" / "done.json").read_text())
                         ["test_metrics"]["f1_macro"] for i in range(1, k + 1)])
        naive_sd = vals.std(ddof=1)
        corrected_sd = np.sqrt(vals.var(ddof=1) * corr * k)   # corrected SE*sqrt(k)~spread
        out[m] = dict(mean=float(vals.mean()), naive_sd=float(naive_sd),
                      nadeau_bengio_sd=float(corrected_sd))
    return out


# --------------------------------------------------------------------------- #
def run() -> dict:
    models = _completed_models()
    baselines = [m for m in models if m != PROPOSED]
    long = _long_table(models)

    result = {
        "models": models,
        "glmm": _glmm(long),
        "cluster_bootstrap": {
            b: dict(roc_auc=_cluster_bootstrap(long, b, roc_auc_score),
                    pr_auc_non_urgent=_cluster_bootstrap(long, b, average_precision_score))
            for b in baselines
        },
        "nadeau_bengio_macro_f1": _nadeau_bengio(models),
    }
    (C.RESULT_DIR / "mixed_models.json").write_text(json.dumps(result, indent=2))
    _md(result)
    return result


def _md(r: dict) -> None:
    L = ["# CV-aware statistical comparison", "",
         "Reference = **proposed (YOLO-TL caries-adapted)**. Folds share ~3/4 of "
         "training data, so per-fold t-tests under-estimate variance; the analyses "
         "below respect the CV structure.", ""]

    L.append("## 1. GLMM — `correct ~ model + (1|patient) + (1|fold)`")
    L.append("")
    g = r["glmm"]
    if "error" in g:
        L.append(f"_GLMM failed: {g['error'].splitlines()[-1] if g['error'] else 'unknown'}_")
    else:
        L.append("Odds ratio < 1 ⇒ that baseline is **less** likely to be correct than the proposed model.")
        L.append("")
        L.append("| baseline | odds ratio (95% CI) | z | p |")
        L.append("|---|---|---|---|")
        for b, d in g["contrasts"].items():
            L.append(f"| {C.MODEL_DISPLAY.get(b, b)} | {d['odds_ratio']:.2f} "
                     f"[{d['or_ci95'][0]:.2f}, {d['or_ci95'][1]:.2f}] | {d['z']:.2f} | {d['p_value']:.3f} |")
        L.append("")
        L.append("Variance components: "
                 + ", ".join(f"{v['grp']}={v['vcov']:.3f}" for v in g["variance_components"]))
    L.append("")

    L.append("## 2. Cluster bootstrap — pooled-OOF AUC difference (resample folds, then patients)")
    L.append("")
    L.append("| baseline | Δ ROC-AUC (95% CI) | p | Δ PR-AUC non-urgent (95% CI) | p |")
    L.append("|---|---|---|---|---|")
    for b, d in r["cluster_bootstrap"].items():
        ra, pa = d["roc_auc"], d["pr_auc_non_urgent"]
        L.append(f"| {C.MODEL_DISPLAY.get(b, b)} "
                 f"| {ra['delta']*100:+.1f} [{ra['ci95'][0]*100:+.1f}, {ra['ci95'][1]*100:+.1f}] | {ra['p_cluster_bootstrap']:.3f} "
                 f"| {pa['delta']*100:+.1f} [{pa['ci95'][0]*100:+.1f}, {pa['ci95'][1]*100:+.1f}] | {pa['p_cluster_bootstrap']:.3f} |")
    L.append("")

    L.append("## 3. Nadeau–Bengio corrected spread — per-fold macro-F1")
    L.append("")
    L.append("| model | mean | naive sd | corrected sd |")
    L.append("|---|---|---|---|")
    for m, d in r["nadeau_bengio_macro_f1"].items():
        L.append(f"| {C.MODEL_DISPLAY.get(m, m)} | {d['mean']*100:.1f} "
                 f"| {d['naive_sd']*100:.1f} | {d['nadeau_bengio_sd']*100:.1f} |")
    (C.RESULT_DIR / "mixed_models.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    run()
