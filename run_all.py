"""Orchestrator. Resumable: skips any (model, fold) whose done.json exists.

    python run_all.py                 # all models, all folds, then aggregate
    python run_all.py --models yolo_tl
    python run_all.py --folds 1 2
    python run_all.py --smoke          # 1 model, 1 fold, tiny budget (sanity check)
    python run_all.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from src.common import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(C.MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(1, C.OUTER_FOLDS + 1)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    set_seed(C.SEED)

    if args.smoke:
        C.N_OPTUNA_TRIALS = 2
        C.STAGE_EPOCHS = (2, 2, 2)
        C.MAX_EPOCHS_PER_FOLD = 4
        C.EARLY_STOP_PATIENCE = 3
        args.models = args.models[:1]
        args.folds = args.folds[:1]
        print(">> SMOKE MODE: trials=2, epochs<=4")

    from src.run_fold import run_fold
    from src import tune
    if args.smoke:
        tune._SEARCH_STAGE_EPOCHS = (2, 2, 2)
        tune._SEARCH_MAX_EPOCHS = 4

    manifest = pd.read_csv(C.MANIFEST_CSV)
    folds = {f["fold"]: f for f in json.loads(C.SPLIT_JSON.read_text())["folds"]}

    if not args.aggregate_only:
        for model in args.models:
            for k in args.folds:
                print(f"\n===== {model} | fold {k} =====")
                run_fold(model, folds[k], manifest, force=args.force)

    from src.aggregate import main as aggregate_main
    aggregate_main()


if __name__ == "__main__":
    main()
