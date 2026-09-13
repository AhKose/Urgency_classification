"""Results figures.

fig A (main): model comparison, macro-F1 and ROC-AUC, mean +/- sd over the 5
             outer folds, one horizontal bar per model.
fig B (opt) : pooled out-of-fold ROC curves and non-urgent precision-recall
             curves, computed from the raw per-patient scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

ORDER = ["det_fusion", "det_count", "ens_cnn", "yolo_tl",
         "yolo_coco", "efficientnet_b3", "resnet50v2", "yolo_scratch"]
LABEL = {
    "det_fusion": "Detect-then-count + CNN\n(late fusion, proposed)",
    "det_count": "Detect-then-count",
    "ens_cnn": "CNN ensemble",
    "yolo_tl": "YOLO-TL (CNN transfer)",
    "yolo_coco": "COCO-only backbone",
    "efficientnet_b3": "EfficientNet-B3 (ImageNet)",
    "resnet50v2": "ResNet50V2 (ImageNet)",
    "yolo_scratch": "YOLOv8-L from scratch",
}
PROPOSED_C = "#1f4e79"
OTHER_C = "#9db8d2"


def _pooled(model: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(C.RUN_DIR / model / f"fold_{k}" / "predictions.csv") for k in range(1, 6)],
        ignore_index=True)


def fig_compare(out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or (C.RESULT_DIR / "fig_model_comparison.png")
    agg = json.loads((C.RESULT_DIR / "aggregated.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, key, title in ((axes[0], "f1_macro", "(a) Macro-F1"),
                           (axes[1], "roc_auc", "(b) ROC-AUC")):
        means = np.array([agg[m]["aggregate"][key]["mean"] * 100 for m in ORDER])
        sds = np.array([agg[m]["aggregate"][key]["sd"] * 100 for m in ORDER])
        colors = [PROPOSED_C if m == "det_fusion" else OTHER_C for m in ORDER]
        ax.barh(y, means, xerr=sds, color=colors, height=0.62,
                error_kw=dict(ecolor="#444", capsize=3, lw=1))
        for yi, mu, sd in zip(y, means, sds):
            ax.text(mu + sd + 1.2, yi, f"{mu:.1f}", va="center", fontsize=9)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"{title.split(') ')[1]} (%, mean ± SD over 5 folds)", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if True:  # stratified-prior reference on the macro-F1 panel
        axes[0].axvline(52.3, ls="--", lw=1, color="#b02418")
        axes[0].text(52.3, len(ORDER) - 0.3, " stratified-prior baseline (52.3)",
                     color="#b02418", fontsize=8, va="top")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[m] for m in ORDER], fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


def fig_curves(out: Path | None = None,
               models=("det_fusion", "det_count", "ens_cnn", "yolo_tl",
                       "yolo_coco", "efficientnet_b3", "resnet50v2")) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (auc, average_precision_score,
                                 precision_recall_curve, roc_curve, roc_auc_score)

    out = out or (C.RESULT_DIR / "fig_roc_pr.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(models)))

    for m, col in zip(models, cmap):
        df = _pooled(m)
        y_urgent = (df.True_Label == 0).astype(int).to_numpy()   # 1 = urgent
        s_urgent = df.Prob_urgent.to_numpy()
        lw = 2.4 if m == "det_fusion" else 1.4
        # ROC for detecting the urgent class
        fpr, tpr, _ = roc_curve(y_urgent, s_urgent)
        axes[0].plot(fpr, tpr, color=col, lw=lw,
                     label=f"{LABEL[m].splitlines()[0]} (AUC {roc_auc_score(y_urgent, s_urgent):.3f})")
        # PR for the non-urgent minority class
        y_non = 1 - y_urgent
        s_non = df.Prob_non_urgent.to_numpy()
        prec, rec, _ = precision_recall_curve(y_non, s_non)
        ap = average_precision_score(y_non, s_non)
        axes[1].plot(rec, prec, color=col, lw=lw,
                     label=f"{LABEL[m].splitlines()[0]} (AP {ap:.3f})")

    axes[0].plot([0, 1], [0, 1], ls=":", color="#888", lw=1)
    axes[0].set_title("(a) ROC, urgent-case detection (pooled out-of-fold)", fontsize=12, loc="left")
    axes[0].set_xlabel("1 - specificity"); axes[0].set_ylabel("sensitivity")
    axes[1].axhline(76 / 287, ls=":", color="#888", lw=1)
    axes[1].set_title("(b) Precision-recall, non-urgent (minority) class", fontsize=12, loc="left")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7.5, loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    fig_compare()
    fig_curves()
