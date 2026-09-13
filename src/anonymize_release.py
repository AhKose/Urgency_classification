"""Replace every internal patient_id (bad_5, good_12, normal_3, ...) with the
public, neutral case_id (case_005, case_012, ...) across a release folder's
CSV / JSON / Markdown files.

Run this as the LAST step before publishing a copy of runs/, results/,
splits/, or data/ anywhere public. It never touches the working project's
own files -- only the given --target directory (a separate release copy).

Usage:
    python -m src.anonymize_release --target ../github_repo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.case_ids import case_id_map

_PID_RE_CACHE: re.Pattern | None = None
_SLASH_RE = re.compile(r"\b(bad|good|normal)/(\d+)\b")  # e.g. data/duplicates.csv


def _pid_pattern(mapping: dict[str, str]) -> re.Pattern:
    global _PID_RE_CACHE
    if _PID_RE_CACHE is None:
        alts = "|".join(re.escape(k) for k in mapping)
        _PID_RE_CACHE = re.compile(rf"\b(?:{alts})\b")
    return _PID_RE_CACHE


def _sub_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    n = 0

    def repl_slash(m: re.Match) -> str:
        nonlocal n
        pid = f"{m.group(1)}_{m.group(2)}"
        if pid in mapping:
            n += 1
            return mapping[pid]
        return m.group(0)

    text = _SLASH_RE.sub(repl_slash, text)

    pat = _pid_pattern(mapping)

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return mapping[m.group(0)]

    text = pat.sub(repl, text)
    return text, n


def _fix_csv(path: Path, mapping: dict[str, str]) -> int:
    df = pd.read_csv(path)
    total = 0
    for col in df.columns:
        if df[col].dtype != object:
            continue
        as_str = df[col].astype(str)
        new_vals, counts = [], 0
        for v in as_str:
            nv, n = _sub_text(v, mapping)
            new_vals.append(nv)
            counts += n
        if counts:
            df[col] = new_vals
            total += counts
    if total:
        df.to_csv(path, index=False)
    return total


def _fix_json(path: Path, mapping: dict[str, str]) -> int:
    total = 0

    def walk(obj):
        nonlocal total
        if isinstance(obj, str):
            new, n = _sub_text(obj, mapping)
            total += n
            return new
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        return obj

    data = json.loads(path.read_text(encoding="utf-8"))
    new_data = walk(data)
    if total:
        path.write_text(json.dumps(new_data, indent=1), encoding="utf-8")
    return total


def _fix_text(path: Path, mapping: dict[str, str]) -> int:
    text = path.read_text(encoding="utf-8")
    new_text, n = _sub_text(text, mapping)
    if n:
        path.write_text(new_text, encoding="utf-8")
    return n


def run(target: Path) -> None:
    mapping = case_id_map()
    _pid_pattern(mapping)  # build once
    subdirs = ["splits", "runs", "results", "data"]
    total_files, total_repl = 0, 0
    for sub in subdirs:
        root = target / sub
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                if path.suffix == ".csv":
                    n = _fix_csv(path, mapping)
                elif path.suffix == ".json":
                    n = _fix_json(path, mapping)
                elif path.suffix == ".md":
                    n = _fix_text(path, mapping)
                else:
                    continue
            except Exception as e:
                print(f"  SKIP {path} ({e})")
                continue
            if n:
                total_files += 1
                total_repl += n
                print(f"  {path.relative_to(target)}: {n} replacements")
    print(f"\nanonymize_release: {total_repl} patient_id occurrences replaced "
         f"across {total_files} files under {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path,
                    help="release folder to anonymize in place (e.g. ../github_repo)")
    args = ap.parse_args()
    run(args.target.resolve())
