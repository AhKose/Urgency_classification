"""Optuna hyper-parameter search -- runs strictly inside one outer fold's
training data (inner_train to fit, inner_val to score). Identical budget and
search space for every model (fix #3: fair baseline tuning).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.train import run_training

optuna.logging.set_verbosity(optuna.logging.WARNING)

# cheaper schedule during search; the winning config is refit with full epochs
_SEARCH_STAGE_EPOCHS = (4, 6, 8)
_SEARCH_MAX_EPOCHS = 16
_HIDDEN_SIZE = 256


def _suggest(trial: optuna.Trial) -> dict:
    s = C.SEARCH_SPACE
    return {
        "dropout_rate":    trial.suggest_categorical("dropout_rate", s["dropout_rate"]),
        "unfreeze_level":  trial.suggest_categorical("unfreeze_level", s["unfreeze_level"]),
        "lr":              trial.suggest_categorical("lr", s["lr"]),
        "weight_decay":    trial.suggest_categorical("weight_decay", s["weight_decay"]),
        "label_smoothing": trial.suggest_categorical("label_smoothing", s["label_smoothing"]),
        "use_focal_loss":  trial.suggest_categorical("use_focal_loss", s["use_focal_loss"]),
        "scheduler":       trial.suggest_categorical("scheduler", s["scheduler"]),
        "batch_size":      trial.suggest_categorical("batch_size", s["batch_size"]),
        "hidden_size":     _HIDDEN_SIZE,
        "stage_epochs":    list(_SEARCH_STAGE_EPOCHS),
    }


def tune_fold(model_name: str, manifest: pd.DataFrame, inner_train, inner_val, *,
              n_trials: int, seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "best_config.json"
    if cache.exists():
        return json.loads(cache.read_text())

    def objective(trial: optuna.Trial) -> float:
        cfg = _suggest(trial)
        res = run_training(model_name, cfg, manifest, inner_train, inner_val,
                           seed=seed + trial.number, max_epochs=_SEARCH_MAX_EPOCHS)
        m = res["best_val_metrics"]
        trial.set_user_attr("f1_non_urgent", m["f1_non_urgent"])
        trial.set_user_attr("accuracy", m["accuracy"])
        return m[C.OPTUNA_OBJECTIVE]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        study_name=f"{model_name}_f{out_dir.name}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)

    best = dict(study.best_params)
    best["hidden_size"] = _HIDDEN_SIZE
    best["stage_epochs"] = list(C.STAGE_EPOCHS)          # full schedule for the refit
    payload = dict(
        model=model_name, best_params=best,
        best_inner_f1_macro=study.best_value,
        best_inner_f1_non_urgent=study.best_trial.user_attrs.get("f1_non_urgent"),
        n_trials=n_trials, objective=C.OPTUNA_OBJECTIVE,
    )
    cache.write_text(json.dumps(payload, indent=2))
    return payload
