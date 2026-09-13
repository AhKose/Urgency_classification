"""Figure 2 - proposed model architecture.

Frozen domain-adapted caries detector feeding two parallel branches
(detect-then-count tabular classifier + CNN feature-transfer head), combined
by equal-weight late fusion. Draft for the manuscript; refine the vector output
(results/fig_model.svg) in Inkscape / Illustrator.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def run(out_stem: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out_stem = out_stem or (C.RESULT_DIR / "fig_model")

    # palette
    C_PRE = "#E8DFF0"   # pretraining
    C_DET = "#DCE9F5"   # detect-then-count branch
    C_CNN = "#E3F0DD"   # cnn branch
    C_FUSE = "#F6E3CE"  # fusion
    C_OUT = "#F5D9D9"   # output
    EDGE = "#4A4A4A"

    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, *, fs=9.2, bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
            linewidth=1.3, edgecolor=EDGE, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)

    def arrow(x1, y1, x2, y2, *, style="-|>", lw=1.5, ls="-"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
            linewidth=lw, color=EDGE, linestyle=ls,
            connectionstyle="arc3,rad=0.0", shrinkA=2, shrinkB=2))

    # ---- pretraining ----------------------------------------------------
    box(2, 84, 30, 12,
        "Adult panoramic dataset\n(tooth-level caries annotations,\nDENTEX; independent public data)", C_PRE, fs=8.6)
    box(38, 84, 24, 12, "YOLOv8-L\n(COCO-initialized)\nfine-tuning", C_PRE, fs=8.8)
    arrow(32, 90, 38, 90)

    box(20, 66, 40, 11,
        "Frozen caries-adapted backbone\n(no training on study images,\nno urgency labels seen)", C_PRE, fs=8.8, bold=True)
    arrow(50, 84, 44, 77)

    # split point
    arrow(34, 66, 24, 55)     # to detect-then-count
    arrow(46, 66, 60, 55)     # to cnn

    # ---- branch A : detect-then-count --------------------------------
    ax.text(20, 58.5, "Branch A  ·  detect-then-count", ha="center",
            fontsize=9.5, fontweight="bold", color="#2A5D8F")
    box(2, 44, 36, 10,
        "Run detector on each radiograph\n->  bounding boxes + confidences", C_DET, fs=8.6)
    box(2, 30, 36, 11,
        "12 case-level features:\ncount, confidence, box size,\nseverity, spatial spread", C_DET, fs=8.6)
    box(2, 17, 36, 9,
        "Tuned tabular classifier\n(logistic regression)", C_DET, fs=8.8)
    arrow(20, 44, 20, 41)
    arrow(20, 30, 20, 26)
    arrow(20, 17, 20, 12.5)
    box(4, 5, 32, 7, "P1  =  P(urgent | detections)", C_DET, fs=8.8, bold=True)

    # ---- branch B : cnn feature transfer ----------------------------
    ax.text(80, 58.5, "Branch B  ·  CNN feature transfer", ha="center",
            fontsize=9.5, fontweight="bold", color="#3E7A32")
    box(62, 44, 36, 10,
        "Multi-scale backbone features\n(256 + 512 + 512 channels)", C_CNN, fs=8.6)
    box(62, 30, 36, 11,
        "Global average pool + concat\n->  1280-d vector\n->  2-layer MLP head", C_CNN, fs=8.6)
    box(62, 17, 36, 9,
        "3-stage progressive\nunfreezing", C_CNN, fs=8.8)
    arrow(80, 44, 80, 41)
    arrow(80, 30, 80, 26)
    arrow(80, 17, 80, 12.5)
    box(64, 5, 32, 7, "P2  =  P(urgent | image)", C_CNN, fs=8.8, bold=True)

    # ---- fusion -------------------------------------------------------
    box(34, -6, 32, 8,
        "Late fusion\nP = 0.5 P1 + 0.5 P2", C_FUSE, fs=9.2, bold=True)
    ax.set_ylim(-8, 100)
    arrow(20, 5, 36, -2)
    arrow(80, 5, 64, -2)

    box(38, -18, 24, 7, "Urgent  /  Non-urgent", C_OUT, fs=9.4, bold=True)
    ax.set_ylim(-20, 100)
    arrow(50, -6, 50, -11)

    ax.text(50, 99, "Proposed model: frozen caries detector, two branches, equal-weight late fusion",
            ha="center", va="top", fontsize=10.5, fontweight="bold")

    fig.tight_layout()
    png = out_stem.with_suffix(".png")
    svg = out_stem.with_suffix(".svg")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, "and", svg)
    return png


if __name__ == "__main__":
    run()
