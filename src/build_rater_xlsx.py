"""Build the 3-sheet dentist-labels + inter-rater Excel (for the repo)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# rater 1 = original folder-derived labels, saved before the consensus relabel
# overwrote data/manifest.csv
_r1_bak = sorted(C.DATA_DIR.glob("manifest.csv.rater1_*"))
if not _r1_bak:
    raise FileNotFoundError("rater-1 backup manifest not found")
man = pd.read_csv(_r1_bak[-1])

RATER2_FLIPS = {
    "good_17": "urgent", "good_22": "urgent",
    "normal_13": "urgent", "normal_14": "urgent", "normal_17": "urgent", "normal_23": "urgent",
    "normal_24": "urgent", "normal_36": "urgent", "normal_40": "urgent",
    "bad_20": "non_urgent", "bad_28": "non_urgent", "bad_95": "non_urgent", "bad_206": "non_urgent",
}
RATER3 = {
    "good_17": "urgent", "good_22": "urgent", "normal_24": "urgent",
    "bad_20": "non_urgent", "bad_28": "non_urgent", "bad_95": "non_urgent", "bad_206": "non_urgent",
    "normal_13": "non_urgent", "normal_14": "non_urgent", "normal_17": "non_urgent",
    "normal_23": "non_urgent", "normal_36": "non_urgent", "normal_40": "non_urgent",
}
PRETTY = {"urgent": "urgent", "non_urgent": "non-urgent"}

man["rater1"] = man["urgency"]
man["rater2"] = man.apply(lambda r: RATER2_FLIPS.get(r.patient_id, r.rater1), axis=1)
man["_c"] = man.patient_id.str.split("_").str[0]
man["_n"] = man.patient_id.str.split("_").str[1].astype(int)
man = man.sort_values(["_c", "_n"], key=lambda s: s.map({"bad": 0, "good": 1, "normal": 2}) if s.name == "_c" else s)
man = man.reset_index(drop=True)


def consensus(pid, r1, r2):
    r3 = RATER3.get(pid, r1)
    v = [r1, r2, r3]
    return max(set(v), key=v.count)


A = lambda **k: Font(name="Arial", **k)
HF = PatternFill("solid", fgColor="1F4E79")
CHG = PatternFill("solid", fgColor="FFE699")
URG = PatternFill("solid", fgColor="FCE4D6")
NON = PatternFill("solid", fgColor="E2EFDA")
_t = Side("thin", color="BFBFBF")
BD = Border(_t, _t, _t, _t)


def header(ws, cols, row=1):
    for j, cval in enumerate(cols, 1):
        x = ws.cell(row, j, cval)
        x.font = A(bold=True, color="FFFFFF")
        x.fill = HF
        x.alignment = Alignment(wrap_text=True, vertical="center")
        x.border = BD


wb = Workbook()

# ---- Sheet 1: Rater 1 & Rater 2, all 287 ----
ws = wb.active
ws.title = "Rater 1 and 2 (all images)"
header(ws, ["Image", "Source dataset", "Rater 1 label", "Rater 2 label", "Agree?"])
r = 2
for _, row in man.iterrows():
    r1, r2 = PRETTY[row.rater1], PRETTY[row.rater2]
    ag = "yes" if r1 == r2 else "NO"
    for j, v in enumerate([row.patient_id, row.source, r1, r2, ag], 1):
        c = ws.cell(r, j, v)
        c.font = A(size=10)
        c.border = BD
        c.alignment = Alignment(vertical="center", horizontal="left" if j == 1 else "center")
    if ag == "NO":
        for j in range(1, 6):
            ws.cell(r, j).fill = CHG
    r += 1
ws.freeze_panes = "A2"
ws.auto_filter.ref = f"A1:E{r - 1}"
for j, w in enumerate([16, 16, 15, 15, 9], 1):
    ws.column_dimensions[get_column_letter(j)].width = w
ws.cell(r + 1, 1, "287 unique patients (5 exact-duplicate images removed). Binary urgency label used in the study "
        "(3-class origin: Bad -> urgent; Good/Normal -> non-urgent). Rater 1 & Rater 2 = two dentists, "
        "independent and blind labelling.").font = A(size=8, italic=True)

# ---- Sheet 2: Rater 3 + consensus (13 disagreements) ----
ws2 = wb.create_sheet("Rater 3 and consensus")
header(ws2, ["Image", "Source", "Rater 1", "Rater 2", "Rater 3 (adjudication)",
             "Final consensus (2 of 3)", "Changed vs Rater 1", "Rater 3 sided with"])
dis = man[man.rater1 != man.rater2]
r = 2
for _, row in dis.iterrows():
    r1, r2 = row.rater1, row.rater2
    r3 = RATER3[row.patient_id]
    cons = consensus(row.patient_id, r1, r2)
    vals = [row.patient_id, row.source, PRETTY[r1], PRETTY[r2], PRETTY[r3], PRETTY[cons],
            "YES" if cons != r1 else "no", "Rater 2" if r3 == r2 else "Rater 1"]
    for j, v in enumerate(vals, 1):
        c = ws2.cell(r, j, v)
        c.font = A(size=10, bold=(j == 6))
        c.border = BD
        c.alignment = Alignment(vertical="center", horizontal="left" if j == 1 else "center")
    ws2.cell(r, 6).fill = URG if cons == "urgent" else NON
    if cons != r1:
        ws2.cell(r, 7).fill = CHG
    r += 1
ws2.freeze_panes = "A2"
ws2.auto_filter.ref = f"A1:H{r - 1}"
for j, w in enumerate([14, 9, 13, 13, 21, 24, 18, 17], 1):
    ws2.column_dimensions[get_column_letter(j)].width = w

# ---- Sheet 3: inter-rater analysis ----
ws3 = wb.create_sheet("Inter-rater analysis")
a = (man.rater1 == "urgent").to_numpy().astype(int)
b = (man.rater2 == "urgent").to_numpy().astype(int)
n = len(a)
po = float((a == b).mean())
m1, m2 = a.mean(), b.mean()
pe = m1 * m2 + (1 - m1) * (1 - m2)
kappa = (po - pe) / (1 - pe)
pabak = 2 * po - 1
pi = (m1 + m2) / 2
ac1 = (po - 2 * pi * (1 - pi)) / (1 - 2 * pi * (1 - pi))
rng = np.random.default_rng(42)
ks = []
for _ in range(5000):
    s = rng.choice(n, n, replace=True)
    aa, bb = a[s], b[s]
    p0 = (aa == bb).mean()
    q1, q2 = aa.mean(), bb.mean()
    pe_ = q1 * q2 + (1 - q1) * (1 - q2)
    ks.append((p0 - pe_) / (1 - pe_) if pe_ < 1 else 1.0)
klo, khi = np.percentile(ks, [2.5, 97.5])
TP = int(((a == 1) & (b == 1)).sum())
FN = int(((a == 1) & (b == 0)).sum())
FP = int(((a == 0) & (b == 1)).sum())
TN = int(((a == 0) & (b == 0)).sum())

cons_series = man.apply(lambda r: consensus(r.patient_id, r.rater1, r.rater2), axis=1)
o = man.rater1.value_counts()
cc = cons_series.value_counts()


def put(row, label, val, bold=False):
    ws3.cell(row, 1, label).font = A(bold=bold, size=10)
    ws3.cell(row, 2, val).font = A(bold=bold, size=10)


ws3.cell(1, 1, "Inter-rater agreement - Rater 1 vs Rater 2 (binary urgent / non-urgent, all 287 patients)").font = A(bold=True, size=11)
for i, (k, v) in enumerate([
    ("Patients labelled (both raters, independent and blind)", n),
    ("Agreements", TP + TN),
    ("Disagreements", FN + FP),
    ("Raw percent agreement", f"{po * 100:.1f}%"),
    ("Cohen's kappa", round(kappa, 3)),
    ("Cohen's kappa 95% CI (bootstrap, 5000 resamples)", f"{klo:.3f} - {khi:.3f}"),
    ("PABAK (prevalence-adjusted, bias-adjusted kappa)", round(pabak, 3)),
    ("Gwet's AC1 (imbalance-robust)", round(ac1, 3)),
    ("Interpretation (Landis and Koch)", "almost perfect (kappa >= 0.81)"),
], 3):
    put(i, k, v)

ws3.cell(13, 1, "Confusion matrix (Rater 1 rows, Rater 2 columns)").font = A(bold=True, size=10)
ws3.cell(14, 2, "R2: urgent").font = A(bold=True, size=10)
ws3.cell(14, 3, "R2: non-urgent").font = A(bold=True, size=10)
ws3.cell(15, 1, "R1: urgent").font = A(bold=True, size=10)
ws3.cell(15, 2, TP).font = A(size=10)
ws3.cell(15, 3, FN).font = A(size=10)
ws3.cell(16, 1, "R1: non-urgent").font = A(bold=True, size=10)
ws3.cell(16, 2, FP).font = A(size=10)
ws3.cell(16, 3, TN).font = A(size=10)

ws3.cell(18, 1, "Adjudication (Rater 3) and final consensus").font = A(bold=True, size=11)
for i, (k, v) in enumerate([
    ("Cases adjudicated by Rater 3 (Rater 1 vs Rater 2 disagreements)", 13),
    ("Rater 3 agreed with Rater 2", 7),
    ("Rater 3 agreed with Rater 1", 6),
    ("Final label rule", "majority vote (2 of 3)"),
    ("Consensus labels differing from Rater 1 (used in current analysis)", "7 / 287 (2.4%)"),
    ("   non-urgent -> urgent", "good_17, good_22, normal_24"),
    ("   urgent -> non-urgent", "bad_20, bad_28, bad_95, bad_206"),
    ("Class counts - Rater 1 (original)", f"urgent {o['urgent']} / non-urgent {o['non_urgent']}  (2.83:1)"),
    ("Class counts - consensus (2 of 3)", f"urgent {cc['urgent']} / non-urgent {cc['non_urgent']}  ({cc['urgent'] / cc['non_urgent']:.2f}:1)"),
], 20):
    put(i, k, v)

ws3.cell(31, 1, "Notes").font = A(bold=True, size=10)
for i, tnote in enumerate([
    "Rater 3 labelled only the 13 disagreement cases (adjudication design); a full three-rater kappa is not defined.",
    "The two-rater kappa above is the inter-rater reliability statistic for the study.",
    "Final study labels = consensus (2 of 3); see sheet 'Rater 3 and consensus'.",
    "Bootstrap seed = 42.",
], 32):
    ws3.cell(i, 1, "- " + tnote).font = A(size=9, italic=True)
ws3.column_dimensions["A"].width = 64
ws3.column_dimensions["B"].width = 42
ws3.column_dimensions["C"].width = 16

out = C.REPORT_DIR / "dentist_labels_interrater.xlsx"
wb.save(out)
print("saved", out)
print("sheets:", wb.sheetnames)
print(f"kappa={kappa:.3f}  CI {klo:.3f}-{khi:.3f}  agreement {po*100:.1f}%  CM TP{TP} FN{FN} FP{FP} TN{TN}")
