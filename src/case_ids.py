"""Single source of truth for the public, neutral case identifier.

Our internal `patient_id` (e.g. "bad_5", "normal_13") is derived from the
source-repository class folder a case was originally organized under, and is
used throughout the working pipeline (splits, runs, detection features).
That naming must never appear in anything released publicly -- it exposes an
informal severity category tied to an identifiable original image, which for
7 of 287 cases even disagrees with the final consensus label. Everything we
publish (labels/, and the release copies of splits/runs/results/data) uses
`case_id` instead: a plain sequential "case_001".."case_287", carrying no
information beyond grouping rows that refer to the same case.

This mapping is deterministic (sorted patient_id -> case_001, case_002, ...)
so it only needs to be computed once, from data/manifest.csv, and is shared
by build_label_release.py and anonymize_release.py so both always agree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def case_id_map() -> dict[str, str]:
    man = pd.read_csv(C.MANIFEST_CSV)
    uniq_patients = sorted(man.patient_id.unique())
    return {pid: f"case_{i + 1:03d}" for i, pid in enumerate(uniq_patients)}
