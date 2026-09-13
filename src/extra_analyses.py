"""Supplementary analyses expected by a methodology-focused reviewer.

All post-hoc on the saved out-of-fold predictions (no retraining):

1. Bayesian correlated t-test (Benavoli et al. 2017) on per-fold differences
   -> P(proposed better) / P(practically equivalent) / P(baseline better).
   A Bayesian framing does NOT create power that the data lack; it replaces a
   dichotomous p-value with an interpretable posterior and an explicit ROPE.
2. Post-hoc power / sample-size for the paired comparison.
3. Probability calibration: Brier score + Expected Calibration Error + reliability
   diagram (a decision-support tool must output trustworthy confidences).
4. McNemar exact test on pooled per-patient correctness.
5. Subgroup robustness by source dataset (A vs B).
6. Decision-curve analysis (net benefit) for the "flag urgent" use case.
7. Trivial baselines (always-urgent / random / stratified) for context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score, average_precision_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

PROPOSED = "det_fusion"
ROPE = 0.01          # +/- 1 point of macro-F1 == practically equivalent
RHO = 1.0 / C.OUTER_FOLDS   # test-fold fraction -> fold correlation heuristic


# --------------------------------------------------------------------------- #
def _done_models():
    return [m for m in C.ALL_MODELS
            if all((C.RUN_DIR / m / f"fold_{k}" / "done.json").exists()
                   for k in range(1, C.OUTER_FOLDS + 1))]


def _per_fold(model, metric):
    return np.array([json.loads((C.RUN_DIR / model / f"fold_{k}" / "done.json").read_text())
                     ["test_metrics"][metric] for k in range(1, C.OUTER_FOLDS + 1)], float)


def _pooled(model):
    df = pd.concat([pd.read_csv(C.RUN_DIR / model / f"fold_{k}" / "predictions.csv")
                    for k in range(1, C.OUTER_FOLDS + 1)], ignore_index=True)
    return df.sort_values("patient_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def bayesian_correlated_t(diffs: np.ndarray, rope: float = ROPE) -> dict:
    """Corani & Benavoli (2015) correlated-t posterior over the mean difference."""
    n = len(diffs)
    mean = float(diffs.mean())
    var = float(diffs.var(ddof=1))
    scale = np.sqrt(var * (1.0 / n + RHO / (1.0 - RHO)))
    df = n - 1
    if scale == 0:
        p_pos = float(mean > rope)
        return dict(mean=mean, p_proposed_better=p_pos, p_rope=float(abs(mean) <= rope),
                    p_baseline_better=float(mean < -rope))
    t = stats.t(df, loc=mean, scale=scale)
    return dict(
        mean=mean, scale=float(scale),
        p_proposed_better=float(1 - t.cdf(rope)),
        p_rope=float(t.cdf(rope) - t.cdf(-rope)),
        p_baseline_better=float(t.cdf(-rope)),
        hdi95=[float(t.ppf(0.025)), float(t.ppf(0.975))],
    )


def posthoc_power(diffs: np.ndarray, alpha=0.05) -> dict:
    n = len(diffs)
    d = diffs.mean(); sd = diffs.std(ddof=1)
    if sd == 0:
        return dict(effect_size_dz=float("inf"), power_at_n=1.0, n_for_80pct_power=n)
    dz = d / sd
    from scipy.stats import nct, t as tdist
    tcrit = tdist.ppf(1 - alpha / 2, n - 1)
    ncp = dz * np.sqrt(n)
    power = float(1 - nct.cdf(tcrit, n - 1, ncp) + nct.cdf(-tcrit, n - 1, ncp))
    n_need = n
    while n_need < 500:
        ncp2 = dz * np.sqrt(n_need)
        tc2 = tdist.ppf(1 - alpha / 2, n_need - 1)
        pw = 1 - nct.cdf(tc2, n_need - 1, ncp2) + nct.cdf(-tc2, n_need - 1, ncp2)
        if pw >= 0.80:
            break
        n_need += 1
    return dict(effect_size_dz=float(dz), power_at_n=power, n_for_80pct_power=int(n_need))


def calibration(model: str, n_bins=10) -> dict:
    pp = _pooled(model)
    y = (pp.True_Label.values == C.URGENCY_TO_INT["non_urgent"]).astype(int)   # positive = non-urgent
    p = pp.Prob_non_urgent.values
    brier = float(brier_score_loss(y, p))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    curve = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y[m].mean()
        ece += (m.sum() / len(y)) * abs(acc - conf)
        curve.append(dict(bin_conf=float(conf), bin_acc=float(acc), n=int(m.sum())))
    return dict(brier=brier, ece=float(ece), reliability=curve)


def mcnemar_vs(baseline: str) -> dict:
    a = _pooled(PROPOSED); b = _pooled(baseline)
    ca = (a.True_Label == a.Pred).values
    cb = (b.True_Label == b.Pred).values
    n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
    res = mcnemar([[int((ca & cb).sum()), n10], [n01, int((~ca & ~cb).sum())]], exact=True)
    return dict(proposed_only_correct=n10, baseline_only_correct=n01,
                statistic=float(res.statistic), p_value=float(res.pvalue))


def subgroup_by_source(model: str) -> dict:
    man = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")["source"]
    pp = _pooled(model)
    pp["source"] = pp.patient_id.map(man)
    out = {}
    for src, g in pp.groupby("source"):
        y = (g.True_Label.values == C.URGENCY_TO_INT["non_urgent"]).astype(int)
        yhat_non = (g.Pred.values == C.URGENCY_TO_INT["non_urgent"]).astype(int)
        out[str(src)] = dict(
            n=int(len(g)), n_non_urgent=int(y.sum()),
            f1_macro=float(f1_score(g.True_Label, g.Pred, average="macro", zero_division=0)),
            roc_auc=float(roc_auc_score(y, g.Prob_non_urgent)) if len(np.unique(y)) > 1 else None,
        )
    return out


def decision_curve(models: list[str], thresholds=np.linspace(0.05, 0.6, 23)) -> dict:
    """Net benefit for the 'flag as urgent -> act' decision. positive = urgent."""
    out = {"thresholds": [float(t) for t in thresholds], "models": {}}
    ref = _pooled(models[0])
    y_urg = (ref.True_Label.values == C.URGENCY_TO_INT["urgent"]).astype(int)
    n = len(y_urg); prev = y_urg.mean()
    out["treat_all"] = [float(prev - (1 - prev) * (t / (1 - t))) for t in thresholds]
    out["treat_none"] = [0.0 for _ in thresholds]
    for m in models:
        pp = _pooled(m)
        p_urg = pp.Prob_urgent.values
        nb = []
        for t in thresholds:
            flag = p_urg >= t
            tp = int((flag & (y_urg == 1)).sum()); fp = int((flag & (y_urg == 0)).sum())
            nb.append(float(tp / n - (fp / n) * (t / (1 - t))))
        out["models"][m] = nb
    return out


def trivial_baselines() -> dict:
    ref = _pooled(PROPOSED)
    y = ref.True_Label.values                      # 0 urgent, 1 non-urgent
    rng = np.random.default_rng(C.SEED)
    prev_non = (y == 1).mean()
    strat = rng.random(len(y)) < prev_non
    out = {}
    for name, pred in [("always_urgent", np.zeros_like(y)),
                       ("always_non_urgent", np.ones_like(y)),
                       ("random_50_50", (rng.random(len(y)) < 0.5).astype(int)),
                       ("stratified_prior", strat.astype(int))]:
        out[name] = dict(
            f1_macro=float(f1_score(y, pred, average="macro", zero_division=0)),
            f1_non_urgent=float(f1_score(y, pred, pos_label=1, zero_division=0)),
            accuracy=float((y == pred).mean()),
        )
    return out


# --------------------------------------------------------------------------- #
def run() -> dict:
    models = _done_models()
    baselines = [m for m in models if m != PROPOSED]
    METR = ["f1_macro", "f1_non_urgent", "roc_auc", "pr_auc_non_urgent"]

    res = {
        "models": models,
        "bayesian_correlated_t": {}, "posthoc_power": {},
        "calibration": {m: calibration(m) for m in models},
        "mcnemar": {b: mcnemar_vs(b) for b in baselines},
        "subgroup_by_source": {m: subgroup_by_source(m) for m in models},
        "decision_curve": decision_curve(models),
        "trivial_baselines": trivial_baselines(),
        "rope": ROPE, "rho": RHO,
    }
    for b in baselines:
        res["bayesian_correlated_t"][b] = {
            k: bayesian_correlated_t(_per_fold(PROPOSED, k) - _per_fold(b, k)) for k in METR
        }
        res["posthoc_power"][b] = {
            k: posthoc_power(_per_fold(PROPOSED, k) - _per_fold(b, k)) for k in METR
        }
    (C.RESULT_DIR / "extra_analyses.json").write_text(json.dumps(res, indent=2))
    _plots(res, models)
    _md(res, models, baselines)
    return res


def _plots(res, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in models:
        c = res["calibration"][m]["reliability"]
        if c:
            ax[0].plot([x["bin_conf"] for x in c], [x["bin_acc"] for x in c], "o-",
                       label=f"{C.MODEL_DISPLAY[m]} (ECE={res['calibration'][m]['ece']:.02f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("predicted P(non-urgent)"); ax[0].set_ylabel("observed frequency")
    ax[0].set_title("Reliability (pooled OOF)"); ax[0].legend(fontsize=7)

    dc = res["decision_curve"]; t = dc["thresholds"]
    ax[1].plot(t, dc["treat_all"], "k:", label="flag all")
    ax[1].plot(t, dc["treat_none"], color="grey", ls=":", label="flag none")
    for m in models:
        ax[1].plot(t, dc["models"][m], label=C.MODEL_DISPLAY[m])
    ax[1].set_ylim(bottom=min(0, min(dc["treat_all"])))
    ax[1].set_xlabel("threshold probability"); ax[1].set_ylabel("net benefit")
    ax[1].set_title("Decision curve — flag urgent (pooled OOF)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.RESULT_DIR / "extra_analyses.png", dpi=120); plt.close(fig)


def _md(res, models, baselines):
    L = ["# Supplementary analyses", ""]
    L.append(f"ROPE = ±{res['rope']*100:.0f} macro-F1 points · fold-correlation ρ = {res['rho']:.2f}")
    L.append("")
    L.append("## 1. Bayesian correlated t-test (per-fold differences, proposed − baseline)")
    L.append("")
    for b in baselines:
        L.append(f"### vs {C.MODEL_DISPLAY[b]}")
        L.append("")
        L.append("| metric | mean Δ | P(proposed better) | P(equivalent) | P(baseline better) |")
        L.append("|---|---|---|---|---|")
        for k, d in res["bayesian_correlated_t"][b].items():
            L.append(f"| {k} | {d['mean']*100:+.1f} | {d['p_proposed_better']:.2f} "
                     f"| {d['p_rope']:.2f} | {d['p_baseline_better']:.2f} |")
        L.append("")
    L.append("## 2. Post-hoc power (paired, α=0.05)")
    L.append("")
    L.append("| comparison | metric | effect dz | power @ n=5 | n for 80% power |")
    L.append("|---|---|---|---|---|")
    for b in baselines:
        for k, d in res["posthoc_power"][b].items():
            L.append(f"| vs {C.MODEL_DISPLAY[b]} | {k} | {d['effect_size_dz']:.2f} "
                     f"| {d['power_at_n']:.2f} | {d['n_for_80pct_power']} |")
    L.append("")
    L.append("## 3. Calibration (pooled OOF)")
    L.append("")
    L.append("| model | Brier ↓ | ECE ↓ |")
    L.append("|---|---|---|")
    for m in models:
        c = res["calibration"][m]
        L.append(f"| {C.MODEL_DISPLAY[m]} | {c['brier']:.3f} | {c['ece']:.3f} |")
    L.append("")
    L.append("## 4. McNemar (pooled per-patient correctness)")
    L.append("")
    L.append("| baseline | proposed-only correct | baseline-only correct | p (exact) |")
    L.append("|---|---|---|---|")
    for b, d in res["mcnemar"].items():
        L.append(f"| {C.MODEL_DISPLAY[b]} | {d['proposed_only_correct']} | {d['baseline_only_correct']} | {d['p_value']:.3f} |")
    L.append("")
    L.append("## 5. Subgroup robustness by source dataset")
    L.append("")
    L.append("| model | source | n | macro-F1 | ROC-AUC |")
    L.append("|---|---|---|---|---|")
    for m in models:
        for s, d in res["subgroup_by_source"][m].items():
            ra = f"{d['roc_auc']:.3f}" if d["roc_auc"] is not None else "—"
            L.append(f"| {C.MODEL_DISPLAY[m]} | {s} | {d['n']} | {d['f1_macro']:.3f} | {ra} |")
    L.append("")
    L.append("## 6. Trivial baselines (context)")
    L.append("")
    L.append("| strategy | macro-F1 | F1 non-urgent | accuracy |")
    L.append("|---|---|---|---|")
    for n, d in res["trivial_baselines"].items():
        L.append(f"| {n} | {d['f1_macro']*100:.1f} | {d['f1_non_urgent']*100:.1f} | {d['accuracy']*100:.1f} |")
    L.append("")
    L.append("![](extra_analyses.png)  — reliability diagram + decision curve")
    (C.RESULT_DIR / "extra_analyses.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    run()
