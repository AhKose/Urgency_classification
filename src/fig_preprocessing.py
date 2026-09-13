"""Before/after preprocessing figure for the manuscript.

For a few example patients, show the original panoramic radiograph next to the
output of the standardized preprocessing pipeline (central crop -> resize to
1200x800 -> CLAHE -> Non-Local Means denoising).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import imread_gray

# one urgent, one non-urgent; both good quality
EXAMPLES = ["bad_5", "normal_5"]


def run(examples=EXAMPLES, out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or (C.RESULT_DIR / "preprocessing_examples.png")
    man = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.1 * n))
    axes = np.atleast_2d(axes)

    for i, pid in enumerate(examples):
        row = axes[i]
        r = man.loc[pid]
        orig = imread_gray(r["hepsi_path"])
        prep = imread_gray(r["preproc_path"])
        for ax, img, col in zip(row, (orig, prep), ("Original", "Preprocessed")):
            ax.imshow(img, cmap="gray", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("black"); s.set_linewidth(1.0)
            if i == 0:
                ax.set_title(col, fontsize=13, pad=8)
        row[0].set_ylabel(f"({r['urgency'].replace('_', '-')})",
                          fontsize=11, rotation=90, labelpad=8)

    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # also emit individual panels, useful if the journal wants separate files
    panel_dir = C.RESULT_DIR / "preprocessing_panels"
    panel_dir.mkdir(exist_ok=True)
    for pid in examples:
        r = man.loc[pid]
        for tag, key in (("original", "hepsi_path"), ("preprocessed", "preproc_path")):
            g = imread_gray(r[key])
            f, a = plt.subplots(figsize=(6, 3.5))
            a.imshow(g, cmap="gray"); a.axis("off")
            f.savefig(panel_dir / f"{pid}_{tag}.png", dpi=300,
                      bbox_inches="tight", pad_inches=0, facecolor="white")
            plt.close(f)

    print("wrote", out)
    print("panels in", panel_dir)
    for pid in examples:
        r = man.loc[pid]
        print(f"\n{pid} ({r['urgency']}, source {r['source']})")
        print("  original     :", r["hepsi_path"])
        print("  central crop :", r["crop_path"])
        print("  preprocessed :", r["preproc_path"])
    return out


if __name__ == "__main__":
    run()
