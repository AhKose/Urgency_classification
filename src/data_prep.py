"""ADIM 4 - step 1: build the deduplicated patient-level manifest and run the
manuscript preprocessing pipeline (crop -> resize 1200x800 -> CLAHE -> NLM).

Outputs
-------
data/manifest.csv          one row per UNIQUE patient (radiograph)
data/preprocessed/<urg>/<patient_id>.png   grayscale preprocessed images
reports/data_prep_summary.md
reports/data_prep_samples.png
data/duplicates.csv        the exact-duplicate groups that were removed
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import imread_gray, imwrite, list_pngs, md5_file, set_seed


# --------------------------------------------------------------------------- #
# source identity (A = "ilk datalar", B = "yeni datalar")
# --------------------------------------------------------------------------- #
_SRC_SUBDIRS = {
    "A": {"bad": "kötü 1", "good": "iyi 1", "normal": "orta 1"},
    "B": {"bad": "kötü 2", "good": "iyi 2", "normal": "orta 2"},
}


def _source_hash_lookup() -> dict[str, str]:
    """md5 -> 'A'/'B' for every image in the two pre-merge source folders."""
    lut: dict[str, str] = {}
    for src, base in (("A", C.SRC_A), ("B", C.SRC_B)):
        for cls3, sub in _SRC_SUBDIRS[src].items():
            d = base / sub
            for name in list_pngs(d):
                lut[md5_file(d / name)] = src
    return lut


# --------------------------------------------------------------------------- #
# preprocessing pipeline
# --------------------------------------------------------------------------- #
def preprocess(crop_path: Path) -> np.ndarray:
    """crop -> resize (bicubic) -> CLAHE -> Non-Local-Means denoise. Returns uint8 gray."""
    img = imread_gray(crop_path)
    img = cv2.resize(img, C.RESIZE_WH, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_GRID)
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, None, C.NLM_H, C.NLM_TEMPLATE, C.NLM_SEARCH)
    return img


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    set_seed(C.SEED)
    src_lut = _source_hash_lookup()

    # 1. enumerate the merged 292-image set, deterministic order
    records = []
    for cls3 in ("Bad", "Good", "Normal"):
        for name in list_pngs(C.HEPSI_DIR / cls3):
            p = C.HEPSI_DIR / cls3 / name
            records.append(
                dict(
                    cls3=cls3.lower(),
                    stem=Path(name).stem,
                    hepsi_path=str(p),
                    md5=md5_file(p),
                )
            )
    df = pd.DataFrame(records)
    n_raw = len(df)

    # 2. de-duplicate by exact content (md5); keep first in deterministic order
    dup_groups = [g for g in df.groupby("md5").groups.values() if len(g) > 1]
    dup_rows = []
    for g in dup_groups:
        members = df.loc[g].sort_values(["cls3", "stem"])
        keep = members.iloc[0]
        for _, drop in members.iloc[1:].iterrows():
            dup_rows.append(
                dict(kept=f"{keep.cls3}/{keep.stem}", removed=f"{drop.cls3}/{drop.stem}", md5=drop.md5)
            )
    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(C.DATA_DIR / "duplicates.csv", index=False)

    df = df.sort_values(["cls3", "stem"]).drop_duplicates("md5", keep="first").reset_index(drop=True)

    # 3. enrich: patient_id, urgency, source, crop path
    df["patient_id"] = df["cls3"] + "_" + df["stem"]
    df["urgency"] = df["cls3"].map(C.CLASS3_TO_URGENCY)
    df["urgency_int"] = df["urgency"].map(C.URGENCY_TO_INT)
    df["source"] = df["md5"].map(src_lut).fillna("?")
    df["crop_path"] = df.apply(
        lambda r: str(C.CROP_DIR / r.cls3 / f"{r.stem}.png"), axis=1
    )
    missing_crop = df.loc[~df["crop_path"].apply(lambda p: Path(p).exists()), "patient_id"].tolist()
    if missing_crop:
        raise RuntimeError(f"missing crop files for: {missing_crop}")

    # 4. run preprocessing, write images
    for sub in ("urgent", "non_urgent"):
        (C.PREPROC_DIR / sub).mkdir(parents=True, exist_ok=True)
    proc_paths, whs = [], []
    for _, r in df.iterrows():
        out = C.PREPROC_DIR / r.urgency / f"{r.patient_id}.png"
        arr = preprocess(Path(r.crop_path))
        imwrite(out, arr)
        proc_paths.append(str(out))
        oh, ow = imread_gray(r.crop_path).shape
        whs.append((ow, oh))
    df["preproc_path"] = proc_paths
    df["crop_w"] = [w for w, _ in whs]
    df["crop_h"] = [h for _, h in whs]

    df = df[
        ["patient_id", "cls3", "urgency", "urgency_int", "source",
         "hepsi_path", "crop_path", "preproc_path", "crop_w", "crop_h", "md5"]
    ]
    df.to_csv(C.MANIFEST_CSV, index=False)

    _write_summary(df, n_raw, dup_df)
    _write_samples(df)
    return df


def _write_summary(df: pd.DataFrame, n_raw: int, dup_df: pd.DataFrame) -> None:
    lines = ["# Data preparation summary", ""]
    lines.append(f"- Raw merged files (Cocuk/Hepsi): **{n_raw}**")
    lines.append(f"- Exact-duplicate images removed: **{len(dup_df)}** "
                 f"({dup_df['removed'].tolist() if len(dup_df) else '—'})")
    lines.append(f"- Unique patients (radiographs) retained: **{len(df)}**")
    lines.append("")
    ct = pd.crosstab(df["urgency"], df["source"], margins=True)
    lines.append("## Class x source")
    lines.append("")
    lines.append(ct.to_markdown())
    lines.append("")
    ct3 = pd.crosstab(df["cls3"], df["source"], margins=True)
    lines.append("## 3-class label x source")
    lines.append("")
    lines.append(ct3.to_markdown())
    lines.append("")
    n_u = int((df.urgency == "urgent").sum())
    n_n = int((df.urgency == "non_urgent").sum())
    lines.append(f"- Imbalance: {n_u} urgent : {n_n} non-urgent  = **{n_u / n_n:.2f}:1**")
    lines.append(f"- Minority class = **non_urgent** (n={n_n})")
    lines.append("")
    lines.append("## Preprocessing (manuscript pipeline)")
    lines.append(f"- input: dento-alveolar central crop (Croped_All)")
    lines.append(f"- resize {C.RESIZE_WH} bicubic -> CLAHE(clip={C.CLAHE_CLIP}, grid={C.CLAHE_GRID}) "
                 f"-> NLM(h={C.NLM_H}, t={C.NLM_TEMPLATE}, s={C.NLM_SEARCH})")
    lines.append(f"- crop size range: {df.crop_w.min()}x{df.crop_h.min()} .. {df.crop_w.max()}x{df.crop_h.max()}")
    (C.REPORT_DIR / "data_prep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _write_samples(df: pd.DataFrame, n: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(C.SEED)
    picks = pd.concat([
        df[df.urgency == "urgent"].sample(n, random_state=C.SEED),
        df[df.urgency == "non_urgent"].sample(n, random_state=C.SEED),
    ])
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 2.4 * len(picks)))
    for ax_row, (_, r) in zip(axes, picks.iterrows()):
        raw = imread_gray(r.hepsi_path)
        crop = imread_gray(r.crop_path)
        proc = imread_gray(r.preproc_path)
        for ax, im, t in zip(
            ax_row, (raw, crop, proc),
            (f"raw {r.patient_id}", f"crop [{r.source}]", f"preprocessed ({r.urgency})"),
        ):
            ax.imshow(im, cmap="gray")
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR / "data_prep_samples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    build()
