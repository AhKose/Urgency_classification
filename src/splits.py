"""ADIM 4 - step 2: patient-level nested cross-validation splits.

Structure (leak-free):
  outer: StratifiedKFold(5) over the 287 unique patients, stratified by urgency.
    - outer_test  : the held-out fold, touched exactly once (final evaluation)
    - outer_train : remaining 4 folds
        - inner_train / inner_val : single stratified 80/20 hold-out
                                    -> ALL Optuna hyper-parameter search happens here
        - final_train / final_val : single stratified 80/20 hold-out (different seed)
                                    -> refit with the winning config, early-stopping on final_val

Guarantees asserted below:
  * every patient appears in exactly one outer_test
  * outer_train and outer_test are disjoint for every fold
  * inner_val / final_val never intersect outer_test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import set_seed


def _counts(df: pd.DataFrame, ids: list[str]) -> dict:
    sub = df[df.patient_id.isin(ids)]
    return {"n": len(sub),
            "urgent": int((sub.urgency == "urgent").sum()),
            "non_urgent": int((sub.urgency == "non_urgent").sum())}


def build() -> dict:
    set_seed(C.SEED)
    df = pd.read_csv(C.MANIFEST_CSV)
    df = df.sort_values("patient_id").reset_index(drop=True)
    pids = df.patient_id.to_numpy()
    y = df.urgency_int.to_numpy()

    skf = StratifiedKFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=C.SEED)
    outer = []
    for k, (tr_idx, te_idx) in enumerate(skf.split(pids, y), start=1):
        train_ids = sorted(pids[tr_idx].tolist())
        test_ids = sorted(pids[te_idx].tolist())
        y_tr = df.set_index("patient_id").loc[train_ids, "urgency_int"].to_numpy()

        sss_i = StratifiedShuffleSplit(1, test_size=C.INNER_VAL_FRACTION, random_state=C.SEED + k)
        it_idx, iv_idx = next(sss_i.split(train_ids, y_tr))
        inner_train = sorted(np.array(train_ids)[it_idx].tolist())
        inner_val = sorted(np.array(train_ids)[iv_idx].tolist())

        sss_f = StratifiedShuffleSplit(1, test_size=C.FINAL_VAL_FRACTION, random_state=C.SEED + 100 + k)
        ft_idx, fv_idx = next(sss_f.split(train_ids, y_tr))
        final_train = sorted(np.array(train_ids)[ft_idx].tolist())
        final_val = sorted(np.array(train_ids)[fv_idx].tolist())

        outer.append(dict(
            fold=k,
            train=train_ids, test=test_ids,
            inner_train=inner_train, inner_val=inner_val,
            final_train=final_train, final_val=final_val,
            counts=dict(
                train=_counts(df, train_ids), test=_counts(df, test_ids),
                inner_train=_counts(df, inner_train), inner_val=_counts(df, inner_val),
                final_train=_counts(df, final_train), final_val=_counts(df, final_val),
            ),
        ))

    payload = dict(
        seed=C.SEED, outer_folds=C.OUTER_FOLDS,
        inner_val_fraction=C.INNER_VAL_FRACTION, final_val_fraction=C.FINAL_VAL_FRACTION,
        n_patients=len(df), n_urgent=int((y == 0).sum()), n_non_urgent=int((y == 1).sum()),
        folds=outer,
    )
    _validate(payload, set(pids.tolist()))
    C.SPLIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _report(payload)
    return payload


def _validate(payload: dict, all_ids: set[str]) -> None:
    seen_test: list[str] = []
    for f in payload["folds"]:
        tr, te = set(f["train"]), set(f["test"])
        assert tr.isdisjoint(te), f"fold {f['fold']}: train/test overlap"
        assert tr | te == all_ids, f"fold {f['fold']}: train+test != all patients"
        assert set(f["inner_train"]).isdisjoint(f["inner_val"])
        assert set(f["inner_train"]) | set(f["inner_val"]) == tr
        assert set(f["final_train"]).isdisjoint(f["final_val"])
        assert set(f["final_train"]) | set(f["final_val"]) == tr
        assert te.isdisjoint(f["inner_val"]) and te.isdisjoint(f["final_val"])
        seen_test += f["test"]
    assert sorted(seen_test) == sorted(all_ids), "outer test folds are not a partition"
    assert len(seen_test) == len(set(seen_test)), "a patient appears in >1 outer test fold"
    print("leakage checks: PASSED "
          f"({len(all_ids)} patients partitioned into {payload['outer_folds']} disjoint outer test folds)")


def _report(payload: dict) -> None:
    rows = []
    for f in payload["folds"]:
        c = f["counts"]
        rows.append(dict(
            fold=f["fold"],
            test_n=c["test"]["n"], test_urg=c["test"]["urgent"], test_non=c["test"]["non_urgent"],
            train_n=c["train"]["n"],
            inner_tr=c["inner_train"]["n"], inner_val=c["inner_val"]["n"],
            final_tr=c["final_train"]["n"], final_val=c["final_val"]["n"],
        ))
    tbl = pd.DataFrame(rows)
    md = ["# Nested CV splits", "",
          f"- patients: {payload['n_patients']}  "
          f"({payload['n_urgent']} urgent / {payload['n_non_urgent']} non-urgent)",
          f"- outer folds: {payload['outer_folds']}  | inner val: {payload['inner_val_fraction']}"
          f"  | final val: {payload['final_val_fraction']}",
          f"- Optuna trials per (model, outer fold): {C.N_OPTUNA_TRIALS}",
          "", tbl.to_markdown(index=False), "",
          "Mirror/flip is applied to TRAIN images only; inner_val, final_val and test "
          "use original images exclusively (no test-set mirror inflation)."]
    (C.REPORT_DIR / "splits_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    build()
