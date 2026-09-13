"""Resolution sensitivity check.

Re-fit + evaluate the proposed model and the two ImageNet baselines at a larger
input size, **reusing the per-fold hyper-parameters already selected by the
512 px nested search** (no re-tuning). This is a robustness / sensitivity
analysis, not a fresh nested-CV result, and is labelled as such.

Rationale: the domain-adapted detector backbone was pretrained at 1024 px, so
512 px is a 2x down-scale from the resolution its features were learned on;
640 px (the value used in the manuscript) is a closer match. ImageNet baselines
were pretrained at ~224-300 px and are not expected to benefit the same way.

    python -m src.resolution_check --input 640
    python -m src.resolution_check --input 640 --models yolo_tl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def run(models: list[str], input_size: int) -> dict:
    C.MODEL_INPUT = input_size
    out_root = C.PROJECT / f"runs_res{input_size}"
    out_root.mkdir(exist_ok=True)

    # datasets caches decoded images keyed by (path, size); safe as-is, but clear
    from src import datasets
    datasets._load_resized.cache_clear()

    from src.train import run_training, _device, _evaluate
    from src.models import build_model
    from src.datasets import UrgencyDataset
    import torch

    manifest = pd.read_csv(C.MANIFEST_CSV)
    folds = {f["fold"]: f for f in json.loads(C.SPLIT_JSON.read_text())["folds"]}
    results = {}

    for model in models:
        results[model] = []
        for k in range(1, C.OUTER_FOLDS + 1):
            src_done = C.RUN_DIR / model / f"fold_{k}" / "done.json"
            cfg = json.loads(src_done.read_text())["config"]
            od = out_root / model / f"fold_{k}"
            od.mkdir(parents=True, exist_ok=True)
            if (od / "done.json").exists():
                results[model].append(json.loads((od / "done.json").read_text()))
                continue

            t0 = time.time()
            seed = C.SEED + k
            refit = run_training(model, cfg, manifest, folds[k]["final_train"],
                                 folds[k]["final_val"], seed=seed,
                                 max_epochs=C.MAX_EPOCHS_PER_FOLD)

            dev = _device()
            net = build_model(model, dropout=cfg["dropout_rate"], hidden=cfg.get("hidden_size", 256)).to(dev)
            net.load_state_dict(refit["state_dict"])
            ds = UrgencyDataset(manifest, folds[k]["test"], train=False)
            dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)
            tm, tp = _evaluate(net, dl, dev)

            rec = dict(model=model, fold=k, input_size=input_size, config=cfg,
                       test_metrics=tm, minutes=round((time.time() - t0) / 60, 1))
            (od / "done.json").write_text(json.dumps(rec, indent=2))
            pd.DataFrame({"patient_id": tp["patient_id"], "True_Label": tp["y_true"],
                         "Pred": tp["y_pred"], "Prob_urgent": tp["prob_urgent"]}).to_csv(
                od / "predictions.csv", index=False)
            results[model].append(rec)
            print(f"[{model} f{k} @{input_size}] macroF1={tm['f1_macro']:.3f} "
                  f"roc={tm['roc_auc']:.3f} ({rec['minutes']}m)")

    _compare(results, input_size)
    return results


def _compare(results: dict, input_size: int) -> None:
    keys = ["f1_macro", "f1_non_urgent", "roc_auc", "pr_auc_non_urgent", "mcc"]
    L = [f"# Resolution sensitivity: 512 px (nested-CV) vs {input_size} px (configs reused)", "",
         "The 512 px column is the primary nested-CV result. The "
         f"{input_size} px column reuses each fold's selected hyper-parameters "
         "and only re-fits + re-evaluates.", ""]
    for model in results:
        L.append(f"## {C.MODEL_DISPLAY[model]}")
        L.append("")
        L.append(f"| metric | 512 px | {input_size} px | Δ |")
        L.append("|---|---|---|---|")
        base = [json.loads((C.RUN_DIR / model / f"fold_{k}" / "done.json").read_text())["test_metrics"]
                for k in range(1, C.OUTER_FOLDS + 1)]
        new = results[model]
        for key in keys:
            b = np.array([m[key] for m in base], float)
            n = np.array([r["test_metrics"][key] for r in new], float)
            L.append(f"| {key} | {b.mean()*100:.1f} ± {b.std(ddof=1)*100:.1f} "
                     f"| {n.mean()*100:.1f} ± {n.std(ddof=1)*100:.1f} "
                     f"| {(n.mean()-b.mean())*100:+.1f} |")
        L.append("")
    (C.RESULT_DIR / f"resolution_check_{input_size}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=int, default=640)
    ap.add_argument("--models", nargs="+", default=["yolo_tl", "resnet50v2", "efficientnet_b3"])
    a = ap.parse_args()
    run(a.models, a.input)
