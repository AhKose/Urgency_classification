"""Switch the pipeline to the 3-rater consensus labels.

- backs up the rater-1 manifest + splits + runs + results (kept for a
  robustness comparison, not shown in the paper),
- flips the 7 consensus label changes in data/manifest.csv,
- moves the 7 preprocessed images to the correct urgency folder,
- regenerates the nested-CV splits from the new labels.

Run once, then `python run_all.py`.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# consensus (2 of 3) labels that differ from rater 1
FLIPS = {
    "good_17": "urgent", "good_22": "urgent", "normal_24": "urgent",
    "bad_20": "non_urgent", "bad_28": "non_urgent", "bad_95": "non_urgent", "bad_206": "non_urgent",
}
INT = {"urgent": 0, "non_urgent": 1}


def _archive() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name in ("runs", "results", "runs_res640"):
        src = C.PROJECT / name
        if src.exists():
            dst = C.PROJECT / f"{name}_rater1_{stamp}"
            shutil.move(str(src), str(dst))
            print(f"archived {name} -> {dst.name}")
    for f in (C.MANIFEST_CSV, C.SPLIT_JSON):
        if f.exists():
            shutil.copy2(f, f.with_suffix(f.suffix + f".rater1_{stamp}"))
            print(f"backed up {f.name}")


def _relabel() -> None:
    df = pd.read_csv(C.MANIFEST_CSV)
    changed = 0
    for _, row in df.iterrows():
        new = FLIPS.get(row.patient_id)
        if new is None or new == row.urgency:
            continue
        old = row.urgency
        # move the preprocessed image
        old_p = Path(row.preproc_path)
        new_p = C.PREPROC_DIR / new / old_p.name
        new_p.parent.mkdir(parents=True, exist_ok=True)
        if old_p.exists():
            shutil.move(str(old_p), str(new_p))
        df.loc[df.patient_id == row.patient_id, ["urgency", "urgency_int", "preproc_path"]] = [
            new, INT[new], str(new_p)]
        changed += 1
        print(f"  {row.patient_id}: {old} -> {new}")
    df.to_csv(C.MANIFEST_CSV, index=False)
    n_u = int((df.urgency == "urgent").sum())
    n_n = int((df.urgency == "non_urgent").sum())
    print(f"relabelled {changed} patients | now {n_u} urgent / {n_n} non-urgent "
          f"({n_u / n_n:.2f}:1)")


if __name__ == "__main__":
    _archive()
    _relabel()
    from src.splits import build as build_splits
    build_splits()
    print("\nconsensus labels active. Next: python run_all.py")
