"""Build data/manifest.csv and the preprocessed image set directly from the
two raw public collections + labels/pediatric_urgency_labels.csv -- no manual
sorting into class folders and no manual cropping step.

Every path and every label comes from the release files, so patient_id in
the resulting manifest.csv IS the public case_id (case_001..case_287); there
is no separate internal identifier anymore.

The central crop (Methods 3.2, step 1) is applied automatically and exactly:
the crop box for each of the 14 distinct original resolutions found across
the two collections was recovered once by matching our archived (pre-release)
crops against the uncropped originals, and reproduces all 287 images used in
the paper byte-for-byte (0 pixel difference, see data/crop_lookup.csv). A
resolution not in that table (e.g. if a collection is ever re-exported at a
different size) falls back to the fixed-fraction crop used to derive the
table -- 21% left/right margin, 20% top / 10% bottom margin -- with a warning,
since that case was not verified pixel-for-pixel.

Usage:
    python -m src.build_dataset_from_release \\
        --source-a /path/to/extracted/childrens-teeth-supplement \\
        --source-b /path/to/extracted/childrens-dental-panoramic-radiographs-dataset

`--source-a` must be the folder that directly contains `img/` (i.e. the
extracted "Children's teeth - supplement" .7z). `--source-b` must be the
folder that directly contains
"Children's dental caries segmentation dataset/" (i.e. the extracted
"Children's Dental Panoramic Radiographs Dataset" .zip).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import imread_gray, imwrite, md5_file

FALLBACK_FRAC = dict(x0=0.21, y0=0.20, w=0.58, h=0.70)


def _labels_csv() -> Path:
    for cand in (C.ROOT / "labels" / "pediatric_urgency_labels.csv",
                C.REPORT_DIR / "pediatric_urgency_labels.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError("pediatric_urgency_labels.csv not found under labels/ or reports/")


def _crop_lookup() -> pd.DataFrame:
    return pd.read_csv(C.DATA_DIR / "crop_lookup.csv")


def _crop_box(img: np.ndarray, lut: pd.DataFrame) -> tuple[int, int, int, int]:
    H, W = img.shape
    row = lut[(lut.orig_h == H) & (lut.orig_w == W)]
    if len(row):
        r = row.iloc[0]
        return int(r.crop_x0), int(r.crop_y0), int(r.crop_w), int(r.crop_h)
    print(f"  WARNING: no verified crop box for resolution {W}x{H}; "
         f"using the unverified fallback fraction crop")
    x0 = round(FALLBACK_FRAC["x0"] * W)
    y0 = round(FALLBACK_FRAC["y0"] * H)
    cw = round(FALLBACK_FRAC["w"] * W)
    ch = round(FALLBACK_FRAC["h"] * H)
    return x0, y0, cw, ch


def _preprocess(cropped: np.ndarray) -> np.ndarray:
    """resize (bicubic) -> CLAHE -> Non-Local-Means denoise. Same as Methods 3.2."""
    img = cv2.resize(cropped, C.RESIZE_WH, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_GRID)
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, None, C.NLM_H, C.NLM_TEMPLATE, C.NLM_SEARCH)
    return img


def build(source_a: Path, source_b: Path, save_intermediate_crops: bool = True) -> pd.DataFrame:
    labels = pd.read_csv(_labels_csv())
    lut = _crop_lookup()
    cropped_dir = C.RAW_DATA / "cropped"
    if save_intermediate_crops:
        cropped_dir.mkdir(parents=True, exist_ok=True)
    for urg in ("urgent", "non_urgent"):
        (C.PREPROC_DIR / urg).mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for case_id, grp in labels.groupby("case_id"):
        r = grp.iloc[0]  # duplicate-content pairs: build the case once, from either original file
        is_a = r.figshare_dataset_title.startswith("Children's teeth")
        source_root = source_a if is_a else source_b
        raw_path = source_root / r.original_subfolder / r.original_filename
        if not raw_path.exists():
            missing.append((case_id, str(raw_path)))
            continue

        img = imread_gray(raw_path)
        x0, y0, cw, ch = _crop_box(img, lut)
        cropped = img[y0:y0 + ch, x0:x0 + cw]
        if save_intermediate_crops:
            imwrite(cropped_dir / f"{case_id}.png", cropped)

        pre = _preprocess(cropped)
        preproc_path = C.PREPROC_DIR / r.consensus_urgency_label / f"{case_id}.png"
        imwrite(preproc_path, pre)

        rows.append(dict(
            patient_id=case_id,                 # case_id IS the patient_id from here on
            urgency=r.consensus_urgency_label,
            urgency_int=C.URGENCY_TO_INT[r.consensus_urgency_label],
            source="A" if is_a else "B",
            raw_path=str(raw_path),
            crop_path=str(cropped_dir / f"{case_id}.png") if save_intermediate_crops else "",
            preproc_path=str(preproc_path),
            crop_w=cw, crop_h=ch,
            md5=md5_file(raw_path),
        ))

    if missing:
        print(f"\n{len(missing)} cases could not be located under --source-a/--source-b:")
        for cid, p in missing[:20]:
            print("   ", cid, "->", p)
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
        print("Check that --source-a / --source-b point at the correct extracted "
             "archive roots (see the docstring at the top of this file).")

    man = pd.DataFrame(rows).sort_values("patient_id")
    man.to_csv(C.MANIFEST_CSV, index=False)
    print(f"\nwrote {C.MANIFEST_CSV}  ({len(man)}/{len(labels.case_id.unique())} cases)")
    print(man.urgency.value_counts().to_string())
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-a", required=True, type=Path,
                    help="extracted 'Children's teeth - supplement' root (contains img/)")
    ap.add_argument("--source-b", required=True, type=Path,
                    help="extracted 'Children's Dental Panoramic Radiographs Dataset' root "
                         "(contains \"Children's dental caries segmentation dataset/\")")
    ap.add_argument("--no-save-crops", action="store_true",
                    help="do not persist the intermediate cropped images, only the final preprocessed ones")
    args = ap.parse_args()
    build(args.source_a.resolve(), args.source_b.resolve(),
         save_intermediate_crops=not args.no_save_crops)
