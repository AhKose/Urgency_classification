"""Worklist-prioritisation analysis: cumulative gain, lift, and number-needed-to-
review when a clinician processes radiographs in model-predicted-urgency order
instead of an arbitrary order.

This is the triage-relevant framing: it measures how much *sooner* urgent cases
are seen, not classification accuracy at a threshold. Computed on the pooled
out-of-fold predictions (each patient once).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

URGENT = C.URGENCY_TO_INT["urgent"]
DECILES = (0.10, 0.20, 0.30, 0.50)


def _pooled(model):
    df = pd.concat([pd.read_csv(C.RUN_DIR / model / f"fold_{k}" / "predictions.csv")
                    for k in range(1, C.OUTER_FOLDS + 1)], ignore_index=True)
    return df


def _curve(is_urgent_sorted: np.ndarray) -> dict:
    n = len(is_urgent_sorted)
    total_urg = int(is_urgent_sorted.sum())
    cum = np.cumsum(is_urgent_sorted)
    frac_reviewed = np.arange(1, n + 1) / n
    frac_urgent_found = cum / total_urg
    lift = frac_urgent_found / frac_reviewed
    at = {}
    for d in DECILES:
        i = max(0, int(round(d * n)) - 1)
        at[f"top_{int(d*100)}pct"] = dict(
            urgent_found=float(frac_urgent_found[i]),
            lift=float(lift[i]),
        )
    # number needed to review to find 50 / 80 / 95 % of urgent cases
    nnr = {}
    for target in (0.5, 0.8, 0.95):
        idx = np.argmax(frac_urgent_found >= target)
        nnr[f"to_find_{int(target*100)}pct_urgent"] = dict(
            n_reviewed=int(idx + 1), pct_reviewed=float((idx + 1) / n))
    # baseline: arbitrary order needs target * n_urgent picked from n -> ~ target * n
    return dict(frac_reviewed=frac_reviewed.tolist(),
               frac_urgent_found=frac_urgent_found.tolist(),
               at_deciles=at, number_needed_to_review=nnr,
               total=n, total_urgent=total_urg)


def run() -> dict:
    models = [m for m in C.ALL_MODELS
              if all((C.RUN_DIR / m / f"fold_{k}" / "done.json").exists()
                     for k in range(1, C.OUTER_FOLDS + 1))]
    out = {}
    for m in models:
        pp = _pooled(m).sort_values("Prob_urgent", ascending=False)
        is_urg = (pp.True_Label.values == URGENT).astype(int)
        out[m] = _curve(is_urg)
    (C.RESULT_DIR / "gain_lift.json").write_text(json.dumps(out, indent=2))
    _plot(out, models)
    _md(out, models)
    return out


def _plot(out, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in models:
        c = out[m]
        ax[0].plot(c["frac_reviewed"], c["frac_urgent_found"], label=C.MODEL_DISPLAY[m])
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="arbitrary order")
    ax[0].set_xlabel("fraction of worklist reviewed (model-ordered)")
    ax[0].set_ylabel("fraction of urgent cases found")
    ax[0].set_title("Cumulative gain — pooled OOF"); ax[0].legend(fontsize=7)

    for m in models:
        c = out[m]
        fr = np.array(c["frac_reviewed"]); fu = np.array(c["frac_urgent_found"])
        lift = np.divide(fu, fr, out=np.ones_like(fu), where=fr > 0)
        ax[1].plot(fr, lift, label=C.MODEL_DISPLAY[m])
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="arbitrary order")
    ax[1].set_xlabel("fraction of worklist reviewed"); ax[1].set_ylabel("lift")
    ax[1].set_xlim(0.02, 1.0); ax[1].set_title("Lift — pooled OOF"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.RESULT_DIR / "gain_lift.png", dpi=120); plt.close(fig)


def _md(out, models):
    L = ["# Worklist-prioritisation: cumulative gain & lift", "",
         "If radiographs are reviewed in model-predicted-urgency order (pooled OOF, "
         f"n={out[models[0]]['total']}, {out[models[0]]['total_urgent']} urgent):", ""]
    L.append("| model | urgent found in top 10% (lift) | top 20% (lift) | top 30% (lift) | top 50% (lift) |")
    L.append("|---|---|---|---|---|")
    for m in models:
        a = out[m]["at_deciles"]
        L.append(f"| {C.MODEL_DISPLAY[m]} "
                 + " ".join(f"| {a[f'top_{d}pct']['urgent_found']:.0%} ({a[f'top_{d}pct']['lift']:.2f}×)"
                            for d in (10, 20, 30, 50)) + " |")
    L.append("")
    L.append("## Number needed to review (model-ordered) to find …")
    L.append("")
    L.append("| model | 50% of urgent | 80% of urgent | 95% of urgent |")
    L.append("|---|---|---|---|")
    for m in models:
        n = out[m]["number_needed_to_review"]
        L.append(f"| {C.MODEL_DISPLAY[m]} "
                 f"| {n['to_find_50pct_urgent']['n_reviewed']} ({n['to_find_50pct_urgent']['pct_reviewed']:.0%}) "
                 f"| {n['to_find_80pct_urgent']['n_reviewed']} ({n['to_find_80pct_urgent']['pct_reviewed']:.0%}) "
                 f"| {n['to_find_95pct_urgent']['n_reviewed']} ({n['to_find_95pct_urgent']['pct_reviewed']:.0%}) |")
    L.append("")
    L.append(f"Arbitrary order finds ~X% of urgent cases after reviewing X% of the "
             f"worklist (lift = 1.0). Base urgent rate = "
             f"{out[models[0]]['total_urgent']/out[models[0]]['total']:.0%}.")
    L.append("")
    L.append("![](gain_lift.png)")
    (C.RESULT_DIR / "gain_lift.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
