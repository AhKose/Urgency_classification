"""EXPLORATORY Grad-CAM for the proposed YOLO-TL model.

Not a quantitative result -- produced only to sanity-check that the network
attends to dental structures. Uses the last tapped backbone block (P5/SPPF).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import imread_gray
from src.datasets import _to_tensor as _norm
from src.models import build_model


def _cam_for(model, x, target_layer):
    acts = {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    x = x.clone().requires_grad_(True)                 # force graph through frozen backbone
    model.zero_grad(set_to_none=True)
    logit = model(x)
    cls = int(logit.argmax(1))
    a = acts["v"]
    a.retain_grad()
    logit[0, cls].backward()
    g = a.grad
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1)).squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (C.MODEL_INPUT, C.MODEL_INPUT))
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    h1.remove()
    return cam, cls, torch.softmax(logit, 1)[0].detach().cpu().numpy()


def run(fold: int = 1, n_per_group: int = 3, out: Path | None = None) -> Path:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or (C.RESULT_DIR / "gradcam_exploratory.png")
    manifest = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")
    run_dir = C.RUN_DIR / "yolo_tl" / f"fold_{fold}"
    cfg = __import__("json").loads((run_dir / "done.json").read_text())["config"]
    device = torch.device("cuda" if (C.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    model = build_model("yolo_tl", dropout=cfg["dropout_rate"], hidden=cfg.get("hidden_size", 256))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.to(device).eval()
    target_layer = model.backbone[9]

    preds = pd.read_csv(run_dir / "predictions.csv")
    picks = pd.concat([
        preds[(preds.True_Label == 0) & preds.Correct].head(n_per_group),
        preds[(preds.True_Label == 1)].head(n_per_group),
        preds[~preds.Correct].head(2),
    ]).drop_duplicates("patient_id")

    fig, axes = plt.subplots(len(picks), 2, figsize=(6, 2.6 * len(picks)))
    for row, (_, pr) in zip(np.atleast_2d(axes), picks.iterrows()):
        p = manifest.loc[pr.patient_id, "preproc_path"]
        g = cv2.resize(imread_gray(p), (C.MODEL_INPUT, C.MODEL_INPUT))
        x = _norm(g).unsqueeze(0).to(device)
        cam, cls, prob = _cam_for(model, x, target_layer)
        row[0].imshow(g, cmap="gray"); row[0].axis("off")
        row[0].set_title(f"{pr.patient_id}  true={'urgent' if pr.True_Label==0 else 'non-urg'}", fontsize=8)
        row[1].imshow(g, cmap="gray"); row[1].imshow(cam, cmap="jet", alpha=0.45); row[1].axis("off")
        row[1].set_title(f"pred={'urgent' if cls==0 else 'non-urg'}  p(urgent)={prob[0]:.2f}", fontsize=8)
    fig.suptitle("Grad-CAM (exploratory) — proposed YOLO-TL, fold %d" % fold)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    run()
