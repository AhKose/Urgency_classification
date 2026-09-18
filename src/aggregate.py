"""Collect per-fold results into aggregate tables, plots and REPORT.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.evaluate import aggregate as agg_folds

_HEADLINE = [
    ("f1_macro", "macro-F1 (primary)"),
    ("f1_non_urgent", "F1 non-urgent (minority)"),
    ("f1_urgent", "F1 urgent"),
    ("balanced_accuracy", "balanced acc"),
    ("mcc", "MCC"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc_non_urgent", "PR-AUC non-urgent"),
    ("accuracy", "accuracy"),
    ("precision_non_urgent", "precision non-urgent"),
    ("recall_non_urgent", "recall non-urgent"),
    ("precision_urgent", "precision urgent"),
    ("recall_urgent", "recall urgent"),
]


def _collect() -> dict:
    out = {}
    for model in C.ALL_MODELS:
        folds = []
        for k in range(1, C.OUTER_FOLDS + 1):
            done = C.RUN_DIR / model / f"fold_{k}" / "done.json"
            if done.exists():
                folds.append(json.loads(done.read_text())["test_metrics"])
        if folds:
            out[model] = dict(per_fold=folds, aggregate=agg_folds(folds))
    return out


def _pooled_predictions(model: str) -> pd.DataFrame | None:
    parts = []
    for k in range(1, C.OUTER_FOLDS + 1):
        p = C.RUN_DIR / model / f"fold_{k}" / "predictions.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else None


def _fmt(cell: dict) -> str:
    return f"{cell['mean']*100:.2f} ± {cell['sd']*100:.2f}"


def _comparison_table(data: dict) -> pd.DataFrame:
    rows = []
    for model in C.ALL_MODELS:
        if model not in data:
            continue
        a = data[model]["aggregate"]
        row = {"model": C.MODEL_DISPLAY[model]}
        for key, label in _HEADLINE:
            row[label] = _fmt(a[key]) if key in a else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for model in C.ALL_MODELS:
        pp = _pooled_predictions(model)
        if pp is None:
            continue
        y_non = (pp.True_Label.to_numpy() == C.URGENCY_TO_INT["non_urgent"]).astype(int)
        score_non = pp.Prob_non_urgent.to_numpy()
        prec, rec, _ = precision_recall_curve(y_non, score_non)
        ap = auc(rec, prec)
        axes[0].plot(rec, prec, label=f"{C.MODEL_DISPLAY[model]} (AP={ap:.3f})")
        fpr, tpr, _ = roc_curve(y_non, score_non)
        axes[1].plot(fpr, tpr, label=f"{C.MODEL_DISPLAY[model]} (AUC={auc(fpr, tpr):.3f})")
    axes[0].set_title("Precision-Recall (minority = non-urgent), pooled OOF")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_title("ROC (non-urgent), pooled OOF")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.RESULT_DIR / "pr_roc_curves.png", dpi=120); plt.close(fig)


def _plot_confusions(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in C.ALL_MODELS if m in data]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        cm = np.array(data[model]["aggregate"]["confusion_matrix_sum"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["urgent", "non-urg"]); ax.set_yticklabels(["urgent", "non-urg"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(C.MODEL_DISPLAY[model])
    fig.suptitle("Normalised confusion matrices (summed over 5 outer folds)")
    fig.tight_layout(); fig.savefig(C.RESULT_DIR / "confusion_matrices.png", dpi=120); plt.close(fig)


def _report(data: dict, table: pd.DataFrame) -> None:
    L = ["# Results summary", ""]
    L.append("## Protocol")
    L += [
        "- **287 unique patients** (211 urgent / 76 non-urgent, 2.78:1) after removing 5 exact-duplicate images.",
        f"- **Nested CV**: {C.OUTER_FOLDS} patient-level outer folds; hyper-parameters tuned by Optuna "
        f"({C.N_OPTUNA_TRIALS} trials, macro-F1) on an inner 80/20 hold-out of each outer fold's training data only.",
        "- Same protocol and Optuna budget applied to every model.",
        "- Mirror/flip augmentation applied to training images only; validation and test use original images.",
        "- Each outer test fold is evaluated **exactly once**.",
        "",
        "## Headline comparison (mean ± sd over 5 folds, %)",
        "",
        table.to_markdown(index=False),
        "",
    ]
    for model in C.ALL_MODELS:
        if model not in data:
            continue
        a = data[model]["aggregate"]
        L.append(f"### {C.MODEL_DISPLAY[model]} — per-fold macro-F1")
        vals = a["f1_macro"]["values"]
        ci = a["f1_macro"]["ci95"]
        L.append(f"- folds: {['%.3f' % v for v in vals]}")
        L.append(f"- mean {a['f1_macro']['mean']:.3f} ± {a['f1_macro']['sd']:.3f}  "
                 f"(95% CI {ci[0]:.3f}–{ci[1]:.3f})")
        L.append(f"- confusion (summed): {a['confusion_matrix_sum']}  [rows true urgent/non-urgent]")
        L.append("")
    L.append("![](pr_roc_curves.png)")
    L.append("")
    L.append("![](confusion_matrices.png)")
    (C.RESULT_DIR / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    data = _collect()
    if not data:
        print("no completed folds yet.")
        return
    (C.RESULT_DIR / "aggregated.json").write_text(json.dumps(data, indent=2))
    table = _comparison_table(data)
    table.to_csv(C.RESULT_DIR / "comparison_table.csv", index=False)
    _plot_curves(data)
    _plot_confusions(data)
    _report(data, table)
    try:
        from src.operating_point import run as _op_run
        _op_run()
    except Exception as e:                     # non-fatal: needs model.pt checkpoints
        print(f"[operating_point] skipped: {e}")


if __name__ == "__main__":
    main()
