"""Run one (model, outer fold): inner Optuna search -> refit -> evaluate on the
held-out outer test set exactly once. Resumable via runs/<model>/fold_<k>/done.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.evaluate import compute_metrics
from src.train import _device, _evaluate, run_training
from src.datasets import UrgencyDataset
from src.models import build_model
from src.tune import tune_fold


def _plot_confusion(cm, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["urgent", "non-urgent"]); ax.set_yticklabels(["urgent", "non-urgent"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


@torch.no_grad()
def _predict_test(model_name, cfg, state_dict, manifest, test_ids, seed):
    device = _device()
    model = build_model(model_name, dropout=cfg["dropout_rate"],
                        hidden=cfg.get("hidden_size", 256)).to(device)
    model.load_state_dict(state_dict)
    ds = UrgencyDataset(manifest, test_ids, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                                     num_workers=C.NUM_WORKERS, pin_memory=True)
    metrics, preds = _evaluate(model, dl, device)
    return metrics, preds


def run_fold(model_name: str, fold: dict, manifest: pd.DataFrame, *, force: bool = False) -> dict:
    k = fold["fold"]
    out_dir = C.RUN_DIR / model_name / f"fold_{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "done.json"
    if done.exists() and not force:
        return json.loads(done.read_text())

    t0 = time.time()
    seed = C.SEED + k

    # 1. inner hyper-parameter search (inner_train / inner_val only)
    tuned = tune_fold(model_name, manifest, fold["inner_train"], fold["inner_val"],
                      n_trials=C.N_OPTUNA_TRIALS, seed=seed, out_dir=out_dir)
    cfg = tuned["best_params"]

    # 2. refit on final_train, early-stop on final_val
    refit = run_training(model_name, cfg, manifest, fold["final_train"], fold["final_val"],
                         seed=seed, max_epochs=C.MAX_EPOCHS_PER_FOLD, verbose=True)
    pd.DataFrame(refit["history"]).to_csv(out_dir / "history.csv", index=False)

    # 3. evaluate ONCE on the outer test fold
    test_metrics, test_preds = _predict_test(
        model_name, cfg, refit["state_dict"], manifest, fold["test"], seed)

    man = manifest.set_index("patient_id")
    pred_df = pd.DataFrame({
        "patient_id": test_preds["patient_id"],
        "source": man.loc[test_preds["patient_id"], "source"].to_numpy(),
        "True_Label": test_preds["y_true"],
        "Pred": test_preds["y_pred"],
        "Prob_urgent": test_preds["prob_urgent"],
        "Prob_non_urgent": [1 - p for p in test_preds["prob_urgent"]],
    })
    pred_df["Correct"] = pred_df.True_Label == pred_df.Pred
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    _plot_confusion(test_metrics["confusion_matrix"], out_dir / "confusion_matrix.png",
                    f"{C.MODEL_DISPLAY[model_name]} - fold {k}")
    torch.save(refit["state_dict"], out_dir / "model.pt")

    result = dict(
        model=model_name, fold=k, config=cfg,
        inner=dict(f1_macro=tuned["best_inner_f1_macro"],
                   f1_non_urgent=tuned["best_inner_f1_non_urgent"]),
        refit_val_f1_macro=refit["best_val_metrics"]["f1_macro"],
        test_metrics=test_metrics,
        class_count_final_train=refit["class_count"],
        epochs_done=refit["epochs_done"],
        minutes=round((time.time() - t0) / 60, 1),
    )
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    done.write_text(json.dumps(result, indent=2))
    print(f"[{model_name} fold {k}] test macro-F1={test_metrics['f1_macro']:.3f} "
          f"non-urgent-F1={test_metrics['f1_non_urgent']:.3f} "
          f"({result['minutes']} min)")
    return result
