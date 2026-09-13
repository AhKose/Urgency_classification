# Pediatric caries-urgency labels (this study)

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
| `note` | non-empty only for the 0 rows described below |

## How to use it

1. Download the two public collections:
   - Children's teeth - supplement: https://springernature.figshare.com/articles/dataset/Children_s_teeth_-_supplement/22495645
   - Children's Dental Panoramic Radiographs Dataset: https://springernature.figshare.com/articles/dataset/Children_s_Dental_Panoramic_Radiographs_Dataset/21621705
   - (both are part of the collection: https://springernature.figshare.com/collections/Children_s_Dental_Panoramic_Radiographs_Dataset_for_Caries_Segmentation_and_Dental_Disease_Detection/6317013)
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

- **0 rows carry a warning** in the `note` column, as a defensive
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
