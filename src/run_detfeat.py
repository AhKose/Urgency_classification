"""Nested-CV runner for the tabular 'detect-then-count' model.

Same protocol as the CNN models: 5 patient-level outer folds (from
splits/nested_cv.json), hyper-parameters tuned by Optuna (15 trials, macro-F1)
on the inner 80/20 hold-out of each outer fold's training data, then one refit
and one test evaluation per fold.

Model  `det_count`  : ~12 features engineered from the caries detector's outputs
                      -> tuned classifier (logreg / MLP / hist-grad-boosting).

Outputs runs/det_count/fold_<k>/{done.json, predictions.csv, best_config.json}
in the same format as the CNN runs, so aggregate / mixed_models / extra_analyses
/ gain_lift pick it up automatically.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.detection_features import FEATURE_COLS, ALL_FEATURE_COLS
from src.evaluate import compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
import os
FEATSET = os.environ.get("LFR_FEATSET", "base")
MODEL = "det_count" if FEATSET == "base" else "det_count_sp"
FEATS = FEATURE_COLS if FEATSET == "base" else ALL_FEATURE_COLS
N_TRIALS = C.N_OPTUNA_TRIALS
URGENT = 0


def _xy(feat: pd.DataFrame, man: pd.DataFrame, ids):
    """y uses the project-wide convention: 0 = urgent, 1 = non-urgent."""
    sub = feat.set_index("patient_id").loc[list(ids)]
    y = man.set_index("patient_id").loc[list(ids), "urgency_int"].to_numpy().astype(int)
    return sub[FEATS].to_numpy(dtype=float), y, list(ids)


def _make_clf(cfg):
    if cfg["clf"] == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=cfg["C"], class_weight="balanced",
                                                max_iter=2000, solver="liblinear"))
    if cfg["clf"] == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=cfg["hidden"],
                                           alpha=cfg["alpha"], learning_rate_init=cfg["lr"],
                                           max_iter=800, early_stopping=True, random_state=C.SEED))
    return HistGradientBoostingClassifier(
        max_depth=cfg["max_depth"], learning_rate=cfg["hgb_lr"], max_iter=cfg["max_iter"],
        l2_regularization=cfg["l2"], class_weight="balanced", random_state=C.SEED)


def _prob_urgent(clf, X):
    """P(urgent) = P(class 0), regardless of sklearn's internal class ordering."""
    cls = list(clf.classes_)
    return clf.predict_proba(X)[:, cls.index(0)]


def _suggest(t: optuna.Trial) -> dict:
    clf = t.suggest_categorical("clf", ["logreg", "mlp", "hgb"])
    cfg = {"clf": clf}
    if clf == "logreg":
        cfg["C"] = t.suggest_float("C", 1e-3, 1e2, log=True)
    elif clf == "mlp":
        cfg["hidden"] = t.suggest_categorical("hidden", [(16,), (32,), (32, 16), (64, 16)])
        cfg["alpha"] = t.suggest_float("alpha", 1e-5, 1e-1, log=True)
        cfg["lr"] = t.suggest_float("lr", 1e-4, 1e-2, log=True)
    else:
        cfg["max_depth"] = t.suggest_int("max_depth", 2, 5)
        cfg["hgb_lr"] = t.suggest_float("hgb_lr", 0.01, 0.3, log=True)
        cfg["max_iter"] = t.suggest_int("max_iter", 60, 400)
        cfg["l2"] = t.suggest_float("l2", 1e-6, 10.0, log=True)
    return cfg


def run_fold(fold: dict, feat: pd.DataFrame, man: pd.DataFrame) -> dict:
    k = fold["fold"]
    out = C.RUN_DIR / MODEL / f"fold_{k}"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    t0 = time.time()

    Xtr, ytr, _ = _xy(feat, man, fold["inner_train"])
    Xiv, yiv, _ = _xy(feat, man, fold["inner_val"])

    def objective(tr):
        cfg = _suggest(tr)
        clf = _make_clf(cfg)
        clf.fit(Xtr, ytr)
        pu = _prob_urgent(clf, Xiv)
        yp = (pu < 0.5).astype(int)
        return compute_metrics(yiv, yp, pu)["f1_macro"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED + k, multivariate=True))
    study.optimize(objective, n_trials=N_TRIALS)
    best = _suggest_from_params(study.best_params)
    (out / "optuna_trials.csv").write_text(study.trials_dataframe().to_csv(index=False))
    (out / "best_config.json").write_text(json.dumps(dict(model=MODEL, best_params=best,
                                                          best_inner_f1_macro=study.best_value), indent=2))

    # refit on final_train, evaluate once on test
    Xft, yft, _ = _xy(feat, man, fold["final_train"])
    clf = _make_clf(best)
    clf.fit(Xft, yft)
    Xte, yte, ids = _xy(feat, man, fold["test"])
    pu = _prob_urgent(clf, Xte)
    yp = (pu < 0.5).astype(int)
    metrics = compute_metrics(yte, yp, pu)

    # val predictions (for downstream operating-point threshold selection)
    Xv, yv, vids = _xy(feat, man, fold["final_val"])
    pv = _prob_urgent(clf, Xv)
    pd.DataFrame({"patient_id": vids, "True_Label": yv,
                  "Prob_urgent": pv, "Prob_non_urgent": 1 - pv}).to_csv(
        out / "val_predictions.csv", index=False)

    src = man.set_index("patient_id")
    pd.DataFrame({
        "patient_id": ids, "source": src.loc[ids, "source"].to_numpy(),
        "True_Label": yte, "Pred": yp, "Prob_urgent": pu, "Prob_non_urgent": 1 - pu,
        "Correct": yte == yp,
    }).to_csv(out / "predictions.csv", index=False)

    res = dict(model=MODEL, fold=k, config=best,
               inner=dict(f1_macro=study.best_value),
               test_metrics=metrics, minutes=round((time.time() - t0) / 60, 2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    done.write_text(json.dumps(res, indent=2))
    print(f"[{MODEL} fold {k}] test macro-F1={metrics['f1_macro']:.3f} "
          f"ROC-AUC={metrics['roc_auc']:.3f} non-urgent-F1={metrics['f1_non_urgent']:.3f} "
          f"({res['minutes']}m)  best={best['clf']}")
    return res


def _suggest_from_params(p: dict) -> dict:
    cfg = {"clf": p["clf"]}
    for k in ("C", "hidden", "alpha", "lr", "max_depth", "hgb_lr", "max_iter", "l2"):
        if k in p:
            cfg[k] = p[k]
    return cfg


def main():
    from src.common import set_seed
    set_seed(C.SEED)
    feat = pd.read_csv(C.DATA_DIR / "detection_features.csv")
    man = pd.read_csv(C.MANIFEST_CSV)
    folds = json.loads(C.SPLIT_JSON.read_text())["folds"]
    for f in folds:
        run_fold(f, feat, man)


if __name__ == "__main__":
    main()
