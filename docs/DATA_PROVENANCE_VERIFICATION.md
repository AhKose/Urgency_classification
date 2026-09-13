# Data provenance verification

Before releasing `pediatric_urgency_labels.csv`, we checked whether the file
names we had been using internally for the 287 study images actually match
the file names in the two published Figshare collections, rather than
assuming our own working copies (`Cocuk/Son Data/ilk datalar`,
`Cocuk/Son Data/yeni datalar`) preserved the published names exactly.

## Method

1. Queried the Figshare API for both collections to get the exact archive
   file and its size:
   - "Children's teeth - supplement" (22495645): a single 62,658,104-byte
     `.7z` archive.
   - "Children's Dental Panoramic Radiographs Dataset" (21621705): a single
     1,648,927,506-byte `.zip` archive.
2. For the `.7z` archive (small enough to download in full, 62 MB), listed
   its contents directly with `py7zr`, including each entry's CRC32.
3. For the 1.6 GB `.zip` archive, we did **not** download the images. A
   standard zip file's directory listing (central directory) sits at the
   very end of the file, so we downloaded only the **last 2 MB** via an
   HTTP range request and parsed the ZIP central-directory records from it
   directly (file name, CRC32, size for all 6,415 entries). No image
   content was downloaded for this step.
4. Computed the CRC32 of every one of our 292 local working files (123 from
   source A, 169 from source B) and matched it against the CRC32 values
   recorded in the two archive listings above -- an exact byte-content
   match, independent of any file name.

## Findings

- **Source A ("Children's teeth - supplement"): fully consistent.** All 123
  local files matched an archive entry both by name and by content, under
  the archive's single `img/` folder. No corrections needed.
- **Source B ("Children's Dental Panoramic Radiographs Dataset"): 100 of 169
  local files had been renamed** at some point before this study. The
  relevant part of this collection ("Children's dental caries segmentation
  dataset") is itself split into `Train/images`, `Test/images`, and
  `Supplemental content-93/images`, each independently numbered
  (`cate<k>_<n>.png`), so the same short name is reused across splits for
  different images, and a flat, hand-merged working copy is prone to
  collisions and ad hoc renames (we found file names such as
  `cate1_0033333.png` that are clearly a renamed duplicate of
  `cate1_0033.png`). Every one of the 169 local files was nonetheless
  matched to exactly one real archive entry by CRC32 (0 unresolved).
- **No adult-image contamination.** A subset of the renamed local files had
  plain numeric names (`8.png`, `9.png`, `10.png`, ...) that coincidentally
  also exist in the *adult* tooth-segmentation part of the same archive. We
  explicitly checked the CRC32 of our local files against those specific
  adult-dataset entries: **none matched**, ruling out any adult radiograph
  being mislabeled as pediatric.
- **No new duplicate patients.** Several of the renamed files turned out to
  be exact byte-for-byte duplicates of another file in our own working set;
  in every case this reproduced one of the 5 duplicate pairs already known
  from `data/duplicates.csv` and already excluded from the 287-patient
  cohort (cross-checked by MD5). No 6th duplicate was found.

## Result

`pediatric_urgency_labels.csv` reports `original_subfolder` /
`original_filename` using the **verified published names**, resolved via
`data/source_b_filename_corrections.csv` for source B (identity mapping for
source A). This does not change any label, split, or result in the study;
it only corrects which exact published file name each label should be
attached to when someone re-downloads the public collections.
