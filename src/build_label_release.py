"""Build the public label-release table.

Maps our 287 consensus urgency labels back to the TRUE original filenames of
the two public Figshare datasets, verified by content (not by assumption),
so that anyone who downloads the two public collections can attach our
labels without needing our internal file layout or folder names.

IMPORTANT: our local working copies of source B ("Children's Dental
Panoramic Radiographs Dataset") do NOT all keep the exact filenames used in
the published archive -- 100 of 169 were renamed at some point during our
own data handling (e.g. collisions between the Train/Test/Supplemental-93
sub-splits, which independently reuse the same "cate<k>_<n>.png" numbering).
This was found by downloading both published archives and matching every one
of our 292 local files to the archive by CRC32 content hash, not by name.
Every one of the 292 files was matched with an exact content hash (0 unresolved,
0 mismatches, no adult-image contamination). The verified corrections for
source B live in data/source_b_filename_corrections.csv (local name -> true
published name + sub-split) and are applied here. Source A ("Children's
teeth - supplement") needed no correction: all 123 local files already have
their exact published name, confirmed both by name and by content.

See docs/DATA_PROVENANCE_VERIFICATION.md for how this was verified (Figshare
API + partial-download central-directory / 7z listing + CRC32 comparison,
no image content ever needed to be redistributed to do this).

Chain: Figshare item (source A or B) -> true original file (verified by
content) -> merged file (Cocuk/Hepsi/<Class>/<n>.png, identity verified by
MD5) -> patient_id in data/manifest.csv -> consensus label.

Output:
  reports/pediatric_urgency_labels.csv
  reports/pediatric_urgency_labels.xlsx
  reports/LABELS_README.md
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

SON_DATA = C.ROOT / "Cocuk" / "Son Data"
SOURCE_DIRS = {"A": "ilk datalar", "B": "yeni datalar"}
FIGSHARE = {
    "A": dict(title="Children's teeth - supplement",
             url="https://springernature.figshare.com/articles/dataset/"
                 "Children_s_teeth_-_supplement/22495645"),
    "B": dict(title="Children's Dental Panoramic Radiographs Dataset",
             url="https://springernature.figshare.com/articles/dataset/"
                 "Children_s_Dental_Panoramic_Radiographs_Dataset/21621705"),
}
COLLECTION_URL = ("https://springernature.figshare.com/collections/"
                  "Children_s_Dental_Panoramic_Radiographs_Dataset_for_Caries_"
                  "Segmentation_and_Dental_Disease_Detection/6317013")
B_CORRECTIONS_CSV = C.DATA_DIR / "source_b_filename_corrections.csv"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_originals() -> pd.DataFrame:
    """Local working copies + their MD5 (to join against manifest.csv), plus
    the content-verified TRUE original filename and sub-split within the
    published archive."""
    corr = pd.read_csv(B_CORRECTIONS_CSV)  # local_class, local_filename, true_original_filename, true_original_subfolder
    corr_map = {(r.local_class, r.local_filename): (r.true_original_filename, r.true_original_subfolder)
               for r in corr.itertuples()}

    rows = []
    for tag, dirname in SOURCE_DIRS.items():
        root = SON_DATA / dirname
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if not f.is_file():
                    continue
                if tag == "A":
                    true_name, true_subfolder = f.name, "img"  # verified: no renames in source A
                else:
                    true_name, true_subfolder = corr_map[(class_dir.name, f.name)]
                rows.append(dict(source_tag=tag, true_original_filename=true_name,
                                 true_original_subfolder=true_subfolder, md5=_md5(f)))
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    man = pd.read_csv(C.MANIFEST_CSV)
    orig = _hash_originals()
    merged = man.merge(orig, on="md5", how="left", validate="one_to_many")
    missing = merged[merged.source_tag.isna()]
    if len(missing):
        raise RuntimeError(f"{len(missing)} manifest rows did not match any "
                           f"original file by MD5: {missing.patient_id.tolist()}")
    merged["figshare_dataset_title"] = merged.source_tag.map(lambda t: FIGSHARE[t]["title"])
    merged["figshare_dataset_url"] = merged.source_tag.map(lambda t: FIGSHARE[t]["url"])

    # neutral, sequential case id -- unrelated to our internal working IDs
    # (which are derived from the source-repository's own good/bad/medium
    # folders and therefore should not appear in a public release: 7 of the
    # 287 consensus labels disagree with the folder a case originally came
    # from). Two rows sharing a case id are the same case, present twice
    # because two pixel-identical original files map to it (see notes).
    uniq_patients = sorted(man.patient_id.unique())
    case_id_map = {pid: f"case_{i + 1:03d}" for i, pid in enumerate(uniq_patients)}
    merged["case_id"] = merged.patient_id.map(case_id_map)

    out = merged[["case_id", "figshare_dataset_title", "figshare_dataset_url",
                 "true_original_subfolder", "true_original_filename", "urgency"]].rename(
        columns={"urgency": "consensus_urgency_label",
                "true_original_subfolder": "original_subfolder",
                "true_original_filename": "original_filename"})

    # note only the rows whose (subfolder, filename) pair is reused for a
    # different image elsewhere in the same collection -- content-verified,
    # so this should not happen; kept as a defensive check.
    out["note"] = ""
    for (_, sub, fn), g in merged.groupby(["figshare_dataset_url", "true_original_subfolder",
                                           "true_original_filename"]):
        if g.patient_id.nunique() > 1:
            out.loc[g.index, "note"] = (
                "This exact (subfolder, file name) pair was matched to more than one "
                "case by content; please report this to the authors.")

    out = out.sort_values(["case_id", "figshare_dataset_title", "original_subfolder", "original_filename"])
    out_path = C.REPORT_DIR / "pediatric_urgency_labels.csv"
    out.to_csv(out_path, index=False)
    try:
        out.to_excel(C.REPORT_DIR / "pediatric_urgency_labels.xlsx", index=False, sheet_name="labels")
    except ImportError:
        pass
    n_renamed = int((merged.source_tag == "B").sum() and
                    (pd.read_csv(B_CORRECTIONS_CSV).local_filename !=
                     pd.read_csv(B_CORRECTIONS_CSV).true_original_filename).sum())
    print(f"wrote {out_path}  ({len(out)} rows for 287 cases; "
         f"{len(out) - 287} extra rows are the exact-duplicate images that map "
         f"to the same case/label from a second original filename; "
         f"{n_renamed} rows use a corrected file name verified against the "
         f"published archive, see data/source_b_filename_corrections.csv)")
    _readme(out)
    return out


def _readme(out: pd.DataFrame) -> None:
    n_notes = int((out.note != "").sum())
    L = f"""# Pediatric caries-urgency labels (this study)

`pediatric_urgency_labels.csv` / `.xlsx` give the case-level URGENT /
NON-URGENT consensus label (three dentists, Cohen's kappa 0.88; see Methods)
for each of the 287 radiographs used in this study, keyed to the file names
**exactly as distributed in the two public source collections**, verified by
downloading both archives and matching every file by content (CRC32), not by
assumption. `case_id` is a plain sequential identifier (case_001 ... case_287)
used only to group rows that belong to the same case; it carries no
information about the label or about our internal processing.

## Columns

| column | meaning |
|---|---|
| `case_id` | sequential case identifier, `case_001`-`case_287` |
| `figshare_dataset_title` | which of the two public collections the file belongs to |
| `figshare_dataset_url` | link to that collection |
| `original_subfolder` | the sub-split inside that collection (e.g. `img`, `Train/images`, `Test/images`, `Supplemental content-93/images`), as distributed |
| `original_filename` | the file's name exactly as distributed in that sub-split |
| `consensus_urgency_label` | `urgent` or `non_urgent` (three-rater consensus, see Methods) |
| `note` | non-empty only for the {n_notes} rows described below |

## How to use it

1. Download the two public collections:
   - {FIGSHARE['A']['title']}: {FIGSHARE['A']['url']}
   - {FIGSHARE['B']['title']}: {FIGSHARE['B']['url']}
   - (both are part of the collection: {COLLECTION_URL})
2. For each row, find the file named `original_filename` inside
   `original_subfolder` of that collection.
3. Attach `consensus_urgency_label` to that image.

## How this was verified (important)

Our own working copies of collection B had been reorganized by hand at some
point before this study (files from the `Train`, `Test`, and
`Supplemental content-93` sub-splits of the "Child Dental Caries
Segmentation" part of collection B were pooled into one folder per class,
and a few naming collisions were resolved with ad hoc renames). Trusting
those working names would have mislabeled roughly 100 of the 287 cases in
this file, so instead we:

1. Downloaded both public archives directly from Figshare.
2. Listed every file they contain (7-Zip listing for collection A; a direct
   parse of the zip's central directory for collection B, whose archive is
   1.6 GB, so only the directory listing was downloaded, not the images).
2. Computed the CRC32 of every one of our 292 local working files and matched
   it against the archives' recorded CRC32 per file -- an exact content
   match, independent of any file name.
3. Confirmed, in the same pass, that none of the images are mislabeled adult
   radiographs from the adult subset that also ships inside collection B
   (checked explicitly for the file names that turned out to be reused
   between the pediatric and adult parts of the archive).

Every one of the 292 files matched exactly one real image in the published
archives (0 unresolved, 0 mismatches). `original_filename` /
`original_subfolder` above are the verified published names, which is why
they may differ from names used in any of our own earlier working notes.

## Important notes

- **{n_notes} rows carry a warning** in the `note` column, as a defensive
  check; none were expected or found as of this release.
- **5 pairs of rows share the same `case_id`**: these are pixel-identical
  duplicate files (verified by exact byte content) that we treated as a
  single case. One such pair is two copies of the same radiograph filed
  under different parts of the source distribution, which appears to be a
  labeling inconsistency in the original dataset rather than something
  introduced here.
- The file has 292 rows for 287 cases (211 urgent, 76 non-urgent), because
  the 5 duplicate-content pairs each contribute two rows with the same label;
  counting rows by label therefore gives 211 urgent and 81 non-urgent.
"""
    (C.REPORT_DIR / "LABELS_README.md").write_text(L, encoding="utf-8")


if __name__ == "__main__":
    build()
