"""Triage-relevant operating-point analysis (post-hoc, no retraining).

For every (model, outer fold):
  1. reload the refit checkpoint, score the fold's `final_val` patients
  2. pick the threshold t* = highest P(urgent) cut-off that still reaches a
     target urgent-sensitivity on final_val  (targets: 0.90, 0.95)
  3. apply t* to that fold's held-out TEST set and record
     sensitivity / specificity / NPV / PPV / cleared-fraction / missed-urgent

NPV here = P(truly non-urgent | model says "can wait") -- the safety metric for
automatic deferral. Results are aggregated as mean +/- sd over folds and written
to results/operating_points.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.datasets import UrgencyDataset
from src.models import build_model

TARGETS = (0.90, 0.95)
URGENT = 0


@torch.no_grad()
def _score(model_name, cfg, state_path, manifest, ids):
    dev = torch.device("cuda" if (C.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    net = build_model(model_name, dropout=cfg["dropout_rate"], hidden=cfg.get("hidden_size", 256)).to(dev)
    net.load_state_dict(torch.load(state_path, map_location=dev))
    net.eval()
    ds = UrgencyDataset(manifest, ids, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)
    ys, ps = [], []
    for x, y, _ in dl:
        with torch.autocast("cuda", enabled=(dev.type == "cuda" and C.AMP)):
            logit = net(x.to(dev))
        ps.append(torch.softmax(logit.float(), 1)[:, URGENT].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _op(y, p_urg, thr):
    pred_urg = p_urg >= thr
    is_urg = y == URGENT
    TP = int((pred_urg & is_urg).sum()); FN = int((~pred_urg & is_urg).sum())
    FP = int((pred_urg & ~is_urg).sum()); TN = int((~pred_urg & ~is_urg).sum())
    return dict(
        thr=float(thr),
        sensitivity=TP / (TP + FN) if TP + FN else np.nan,
        specificity=TN / (TN + FP) if TN + FP else np.nan,
        npv=TN / (TN + FN) if TN + FN else np.nan,
        ppv=TP / (TP + FP) if TP + FP else np.nan,
        cleared_fraction=(~pred_urg).mean(),
        missed_urgent=FN, n_urgent=TP + FN,
    )


def _pick_threshold(y_val, p_val, target: float) -> float:
    grid = np.linspace(0.01, 0.99, 197)
    ok = [t for t in grid if _op(y_val, p_val, t)["sensitivity"] >= target]
    return max(ok) if ok else 0.01


def run() -> dict:
    manifest = pd.read_csv(C.MANIFEST_CSV)
    folds = {f["fold"]: f for f in json.loads(C.SPLIT_JSON.read_text())["folds"]}
    out = {}
    for model in C.ALL_MODELS:
        per_target = {f"sens>={int(t*100)}": [] for t in TARGETS}
        per_target["default_0.5"] = []
        skip = False
        for k in range(1, C.OUTER_FOLDS + 1):
            rd = C.RUN_DIR / model / f"fold_{k}"
            if not (rd / "done.json").exists():
                break
            if (rd / "val_predictions.csv").exists():          # tabular / fusion models
                vp = pd.read_csv(rd / "val_predictions.csv")
                tp = pd.read_csv(rd / "predictions.csv")
                y_val, p_val = vp.True_Label.to_numpy(), vp.Prob_urgent.to_numpy()
                y_te, p_te = tp.True_Label.to_numpy(), tp.Prob_urgent.to_numpy()
            elif (rd / "model.pt").exists():                    # CNN models
                cfg = json.loads((rd / "done.json").read_text())["config"]
                y_val, p_val = _score(model, cfg, rd / "model.pt", manifest, folds[k]["final_val"])
                y_te, p_te = _score(model, cfg, rd / "model.pt", manifest, folds[k]["test"])
            else:
                skip = True
                break
            per_target["default_0.5"].append(_op(y_te, p_te, 0.5))
            for t in TARGETS:
                t_star = _pick_threshold(y_val, p_val, t)
                rec = _op(y_te, p_te, t_star)
                rec["threshold_on_val"] = t_star
                per_target[f"sens>={int(t*100)}"].append(rec)
        if skip or not per_target["default_0.5"]:
            continue
        out[model] = {}
        for key, recs in per_target.items():
            agg = {}
            for mk in ("sensitivity", "specificity", "npv", "ppv", "cleared_fraction"):
                v = np.array([r[mk] for r in recs], float)
                agg[mk] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)
            agg["missed_urgent_total"] = int(sum(r["missed_urgent"] for r in recs))
            agg["urgent_total"] = int(sum(r["n_urgent"] for r in recs))
            agg["per_fold"] = recs
            out[model][key] = agg

    (C.RESULT_DIR / "operating_points.json").write_text(json.dumps(out, indent=2))
    _md(out)
    _prevalence_md(out)
    return out


def _prevalence_md(out: dict, prevalences=(0.74, 0.30, 0.15)) -> None:
    """Project PPV/NPV to other urgent-prevalence settings (spectrum bias).

    Sensitivity/specificity are treated as transportable; PPV/NPV re-derived by
    Bayes' rule at each assumed prevalence. The observed cohort (~0.74 urgent)
    reflects referral populations; a general screening population is much lower.
    """
    L = ["# Prevalence-adjusted PPV / NPV (spectrum-bias projection)", "",
         "The evaluated cohort is referral-skewed (~74% urgent). Sensitivity and "
         "specificity at the `sens>=95` operating point are held fixed and PPV/NPV "
         "are recomputed by Bayes' rule at lower urgent prevalences representative "
         "of an unscreened / general pediatric population.", ""]
    for model in C.ALL_MODELS:
        if model not in out or "sens>=95" not in out[model]:
            continue
        a = out[model]["sens>=95"]
        se, sp = a["sensitivity"]["mean"], a["specificity"]["mean"]
        L.append(f"## {C.MODEL_DISPLAY[model]}  (Se={se:.2f}, Sp={sp:.2f} @ sens>=95)")
        L.append("")
        L.append("| urgent prevalence | PPV (flag=urgent) | NPV (defer safety) |")
        L.append("|---|---|---|")
        for pi in prevalences:
            ppv = se * pi / (se * pi + (1 - sp) * (1 - pi) + 1e-9)
            npv = sp * (1 - pi) / (sp * (1 - pi) + (1 - se) * pi + 1e-9)
            tag = " (observed)" if abs(pi - 0.74) < 0.02 else ""
            L.append(f"| {pi:.0%}{tag} | {ppv:.2f} | {npv:.2f} |")
        L.append("")
    (C.RESULT_DIR / "prevalence_adjusted.md").write_text("\n".join(L), encoding="utf-8")


def _md(out: dict) -> None:
    L = ["# Operating-point analysis (triage framing)", "",
         "Threshold chosen on each fold's `final_val` to reach the target urgent-sensitivity, "
         "then applied once to that fold's held-out test set. Mean ± sd over 5 folds.",
         "",
         "`NPV` = P(truly non-urgent | model says \"can wait\") — the safety metric for automatic deferral.",
         "`cleared` = fraction of the caseload the tool would defer.", ""]
    for model in C.ALL_MODELS:
        if model not in out:
            continue
        L.append(f"## {C.MODEL_DISPLAY[model]}")
        L.append("")
        L.append("| operating point | urgent sens | specificity | NPV (defer safety) | PPV | cleared | missed urgent |")
        L.append("|---|---|---|---|---|---|---|")
        for key in ("default_0.5", *[f"sens>={int(t*100)}" for t in TARGETS]):
            a = out[model][key]
            L.append(
                f"| {key} | {a['sensitivity']['mean']:.2f} ± {a['sensitivity']['sd']:.2f} "
                f"| {a['specificity']['mean']:.2f} ± {a['specificity']['sd']:.2f} "
                f"| {a['npv']['mean']:.2f} ± {a['npv']['sd']:.2f} "
                f"| {a['ppv']['mean']:.2f} ± {a['ppv']['sd']:.2f} "
                f"| {a['cleared_fraction']['mean']:.0%} "
                f"| {a['missed_urgent_total']}/{a['urgent_total']} |")
        L.append("")
    (C.RESULT_DIR / "operating_points.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
