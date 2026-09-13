"""Regional interpretability figure: for example cases, overlay the caries
detector's boxes on the radiograph and show the per-quadrant detection count /
confidence. The spatial features did not improve classification (see
results/comparison), but this decomposition is a clinically readable view of
what drove the model's call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import imread_gray
from src.case_ids import case_id_map

CONF_LOW = 0.15


def run(n_per_group: int = 3, out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ultralytics import YOLO

    out = out or (C.RESULT_DIR / "spatial_map.png")
    man = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")
    # pooled OOF predictions of the proposed model, to pick correct/typical cases
    pp = pd.concat([pd.read_csv(C.RUN_DIR / "det_fusion" / f"fold_{k}" / "predictions.csv")
                    for k in range(1, 6)], ignore_index=True).set_index("patient_id")
    det = YOLO(str(C.DETECTOR_WEIGHTS))
    cid_map = case_id_map()

    urg = pp[(pp.True_Label == 0) & pp.Correct].sort_values("Prob_urgent", ascending=False).head(n_per_group)
    non = pp[(pp.True_Label == 1) & pp.Correct].sort_values("Prob_urgent").head(n_per_group)
    picks = pd.concat([urg, non])

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 3.0 * len(picks)),
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, (pid, pr) in zip(np.atleast_2d(axes), picks.iterrows()):
        g = imread_gray(man.loc[pid, "preproc_path"])
        H, W = g.shape
        img3 = np.repeat(g[:, :, None], 3, axis=2)
        res = det.predict(img3, imgsz=1024, conf=CONF_LOW, verbose=False, device=C.DEVICE)[0]
        xywhn = res.boxes.xywhn.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))

        ax = row[0]
        ax.imshow(g, cmap="gray")
        for (cx, cy, w, h), c in zip(xywhn, conf):
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            ax.add_patch(Rectangle((x0, y0), w * W, h * H, fill=False,
                                   edgecolor=plt.cm.autumn(1 - min(c, 1)), lw=1.2))
        ax.axvline(W / 2, color="cyan", lw=0.7, ls="--")
        ax.axhline(H / 2, color="cyan", lw=0.7, ls="--")
        ax.set_title(f"{cid_map[pid]}  true={'urgent' if pr.True_Label == 0 else 'non-urgent'}  "
                     f"model P(urgent)={pr.Prob_urgent:.2f}", fontsize=9)
        ax.axis("off")

        # per-quadrant counts / confidence-sum
        q = np.zeros((2, 2)); qc = np.zeros((2, 2))
        for (cx, cy, _, _), c in zip(xywhn, conf):
            i, j = int(cy >= 0.5), int(cx >= 0.5)
            q[i, j] += 1; qc[i, j] += c
        axq = row[1]
        axq.imshow(qc, cmap="Reds", vmin=0)
        for i in range(2):
            for j in range(2):
                axq.text(j, i, f"n={int(q[i, j])}\nΣconf={qc[i, j]:.1f}",
                         ha="center", va="center", fontsize=8)
        axq.set_xticks([0, 1]); axq.set_yticks([0, 1])
        axq.set_xticklabels(["right", "left"], fontsize=7)
        axq.set_yticklabels(["upper", "lower"], fontsize=7)
        axq.set_title("per-quadrant caries detections", fontsize=8)
    fig.suptitle("Regional decomposition of the caries-detector output (example cases)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    run()
