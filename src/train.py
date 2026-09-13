"""One training run: (model, config, train_ids, val_ids) -> trained model + val metrics.

Progressive 3-stage unfreezing, capped at config['unfreeze_level']:
    stage 0  head only              (always)
    stage 1  + late backbone blocks (if unfreeze_level >= 1)
    stage 2  + mid+late blocks      (if unfreeze_level >= 2)
A fresh optimizer/scheduler is created at every stage. Best checkpoint is chosen
by validation macro-F1. AMP + gradient clipping (L2 = 1.0).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.datasets import loaders
from src.evaluate import compute_metrics
from src.models import build_model


def _device() -> torch.device:
    if C.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _make_loss(cfg, class_count, device):
    inv = 1.0 / np.clip(class_count, 1, None)
    weight = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32, device=device)
    if cfg.get("use_focal_loss"):
        return FocalLoss(alpha=weight, gamma=2.0, label_smoothing=cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.get("label_smoothing", 0.0))


@torch.no_grad()
def _evaluate(model, dl, device) -> tuple[dict, dict]:
    model.eval()
    ids, ys, probs = [], [], []
    for x, y, pid in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=(device.type == "cuda" and C.AMP)):
            logit = model(x)
        p = torch.softmax(logit.float(), dim=1)[:, 0]        # P(urgent)
        probs.append(p.cpu().numpy()); ys.append(y.numpy()); ids += list(pid)
    prob_urgent = np.concatenate(probs)
    y_true = np.concatenate(ys)
    y_pred = (prob_urgent < 0.5).astype(int)                  # >=0.5 -> urgent(0)
    metrics = compute_metrics(y_true, y_pred, prob_urgent)
    preds = dict(patient_id=ids, y_true=y_true.tolist(),
                 y_pred=y_pred.tolist(), prob_urgent=prob_urgent.tolist())
    return metrics, preds


def _stage_epochs(cfg) -> list[int]:
    base = list(cfg.get("stage_epochs", C.STAGE_EPOCHS))
    return base[: cfg["unfreeze_level"] + 1]


def run_training(model_name: str, cfg: dict, manifest, train_ids, val_ids, *,
                 seed: int, max_epochs: int | None = None, verbose: bool = False) -> dict:
    from src.common import set_seed
    set_seed(seed)
    device = _device()

    dl_tr, dl_val, class_count = loaders(
        manifest, train_ids, val_ids, batch_size=cfg["batch_size"], seed=seed
    )
    model = build_model(model_name, dropout=cfg["dropout_rate"],
                        hidden=cfg.get("hidden_size", 256)).to(device)
    criterion = _make_loss(cfg, class_count, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and C.AMP))

    stage_epochs = _stage_epochs(cfg)
    budget = max_epochs or C.MAX_EPOCHS_PER_FOLD
    history = []
    best = {"f1_macro": -1.0}
    best_state = copy.deepcopy(model.state_dict())
    epochs_done, no_improve = 0, 0

    for stage, n_ep in enumerate(stage_epochs):
        model.set_unfreeze(stage)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        n_ep = min(n_ep, max(1, budget - epochs_done))
        if cfg.get("scheduler", "onecycle") == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=cfg["lr"], steps_per_epoch=max(1, len(dl_tr)), epochs=n_ep, pct_start=0.3)
            step_each_batch = True
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
            step_each_batch = False

        for ep in range(n_ep):
            model.train()
            dl_tr.dataset.set_epoch(epochs_done)
            run_loss = 0.0
            for x, y, _ in dl_tr:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=(device.type == "cuda" and C.AMP)):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, C.GRAD_CLIP_L2)
                scaler.step(opt); scaler.update()
                if step_each_batch:
                    sched.step()
                run_loss += loss.item() * len(y)
            epochs_done += 1

            val_metrics, val_preds = _evaluate(model, dl_val, device)
            if not step_each_batch:
                sched.step(val_metrics["f1_macro"])
            history.append(dict(stage=stage, epoch=epochs_done,
                                train_loss=run_loss / len(dl_tr.dataset),
                                val_f1_macro=val_metrics["f1_macro"],
                                val_f1_non_urgent=val_metrics["f1_non_urgent"],
                                val_acc=val_metrics["accuracy"]))
            if verbose:
                h = history[-1]
                print(f"  s{stage} e{epochs_done:02d} loss={h['train_loss']:.3f} "
                      f"val_f1M={h['val_f1_macro']:.3f} val_f1min={h['val_f1_non_urgent']:.3f}")
            if val_metrics["f1_macro"] > best["f1_macro"] + 1e-4:
                best = val_metrics; best_preds = val_preds
                best_state = copy.deepcopy(model.state_dict()); no_improve = 0
            else:
                no_improve += 1
            if no_improve >= C.EARLY_STOP_PATIENCE or epochs_done >= budget:
                break
        if no_improve >= C.EARLY_STOP_PATIENCE or epochs_done >= budget:
            break

    if best["f1_macro"] < 0:                       # never improved -> last eval
        best, best_preds = _evaluate(model, dl_val, device)
    return dict(config=cfg, best_val_metrics=best, val_predictions=best_preds,
               history=history, state_dict=best_state, epochs_done=epochs_done,
               class_count=class_count.tolist())
