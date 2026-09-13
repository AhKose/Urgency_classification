"""Inter-rater agreement (rater 1 = original annotator, rater 2 = second dentist),
computed on the binary urgent / non-urgent label over all 287 patients, BEFORE
adjudication. Rater 3 only breaks ties and is not part of this calculation.

Rater 2's disagreements were supplied as a list of patient_ids that rater 2
flipped relative to rater 1; every other case matches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# rater 2 disagreements (patient_id -> rater 2's label)
RATER2_FLIPS = {
    "case_221": "urgent", "case_227": "urgent",
    "case_254": "urgent", "case_255": "urgent", "case_258": "urgent",
    "case_265": "urgent", "case_266": "urgent", "case_278": "urgent", "case_283": "urgent",
    "case_113": "non_urgent", "case_134": "non_urgent", "case_208": "non_urgent", "case_120": "non_urgent",
}


def _rater1_csv() -> Path:
    """Rater 1 = the original folder-derived annotation, saved before the
    consensus relabel overwrote data/manifest.csv."""
    cands = sorted(C.DATA_DIR.glob("manifest.csv.rater1_*"))
    if not cands:
        raise FileNotFoundError("rater-1 backup manifest not found; "
                                "data/manifest.csv now holds consensus labels")
    return cands[-1]


def _labels() -> pd.DataFrame:
    m = pd.read_csv(_rater1_csv())[["patient_id", "urgency", "source"]].rename(
        columns={"urgency": "rater1"})
    m["rater2"] = m.apply(lambda r: RATER2_FLIPS.get(r.patient_id, r.rater1), axis=1)
    cons = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")["urgency"]
    m["consensus"] = m.patient_id.map(cons)
    return m


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b are 0/1 arrays (1 = urgent). Cohen kappa, PABAK, Gwet AC1, % agree."""
    n = len(a)
    po = float((a == b).mean())
    # marginals
    pa1, pb1 = a.mean(), b.mean()
    pa0, pb0 = 1 - pa1, 1 - pb1
    pe_cohen = pa1 * pb1 + pa0 * pb0
    kappa = (po - pe_cohen) / (1 - pe_cohen) if pe_cohen < 1 else 1.0
    pabak = 2 * po - 1
    # Gwet AC1
    pi = (pa1 + pb1) / 2
    pe_gwet = 2 * pi * (1 - pi)
    ac1 = (po - pe_gwet) / (1 - pe_gwet) if pe_gwet < 1 else 1.0
    return dict(n=n, percent_agreement=po, cohen_kappa=float(kappa),
                pabak=float(pabak), gwet_ac1=float(ac1))


def _bootstrap_ci(a, b, fn, n=5000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    vals = {k: [] for k in fn(a, b)}
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        r = fn(a[s], b[s])
        for k, v in r.items():
            vals[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in vals.items() if k != "n"}


def run() -> dict:
    df = _labels()
    a = (df.rater1 == "urgent").to_numpy().astype(int)
    b = (df.rater2 == "urgent").to_numpy().astype(int)

    s = _stats(a, b)
    ci = _bootstrap_ci(a, b, _stats)

    # 2x2 confusion (rater1 rows, rater2 cols), order [urgent, non_urgent]
    cm = [[int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())],
          [int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())]]

    dis = df[df.rater1 != df.rater2].copy()
    dis_list = dis.assign(direction=dis.rater1 + " -> " + dis.rater2)[
        ["patient_id", "source", "rater1", "rater2", "consensus"]].to_dict("records")
    n_changed = int((df.rater1 != df.consensus).sum())

    result = dict(
        n=s["n"], n_disagreements=len(dis), n_consensus_changed_vs_rater1=n_changed,
        percent_agreement=dict(value=s["percent_agreement"], ci95=ci["percent_agreement"]),
        cohen_kappa=dict(value=s["cohen_kappa"], ci95=ci["cohen_kappa"]),
        pabak=dict(value=s["pabak"], ci95=ci["pabak"]),
        gwet_ac1=dict(value=s["gwet_ac1"], ci95=ci["gwet_ac1"]),
        confusion_rater1_rows_rater2_cols=cm,
        disagreements=dis_list,
        rater1_urgent=int(a.sum()), rater2_urgent=int(b.sum()),
    )
    (C.RESULT_DIR / "inter_rater.json").write_text(json.dumps(result, indent=2))
    _md(result)
    return result


def _md(r: dict) -> None:
    L = ["# Inter-rater agreement (rater 1 vs rater 2, binary urgent / non-urgent)", "",
         f"All **{r['n']}** patients, before adjudication. Rater 3 breaks ties only.", "",
         f"- **Raw agreement**: {r['percent_agreement']['value']*100:.1f}% "
         f"(95% CI {r['percent_agreement']['ci95'][0]*100:.1f}–{r['percent_agreement']['ci95'][1]*100:.1f})",
         f"- **Cohen's κ**: {r['cohen_kappa']['value']:.3f} "
         f"(95% CI {r['cohen_kappa']['ci95'][0]:.3f}–{r['cohen_kappa']['ci95'][1]:.3f})  — "
         f"\"almost perfect\" (Landis & Koch)" if r['cohen_kappa']['value'] >= 0.81 else "",
         f"- **PABAK**: {r['pabak']['value']:.3f} "
         f"(95% CI {r['pabak']['ci95'][0]:.3f}–{r['pabak']['ci95'][1]:.3f})  — "
         f"prevalence-adjusted (imbalance-robust)",
         f"- **Gwet's AC1**: {r['gwet_ac1']['value']:.3f} "
         f"(95% CI {r['gwet_ac1']['ci95'][0]:.3f}–{r['gwet_ac1']['ci95'][1]:.3f})",
         f"- **Disagreements**: {r['n_disagreements']} / {r['n']} "
         f"({r['n_disagreements']/r['n']*100:.1f}%) — the cases rater 3 adjudicated",
         "",
         "## Confusion matrix (rater 1 rows, rater 2 columns)",
         "",
         "|  | rater 2 urgent | rater 2 non-urgent |",
         "|---|---|---|",
         f"| **rater 1 urgent** | {r['confusion_rater1_rows_rater2_cols'][0][0]} | {r['confusion_rater1_rows_rater2_cols'][0][1]} |",
         f"| **rater 1 non-urgent** | {r['confusion_rater1_rows_rater2_cols'][1][0]} | {r['confusion_rater1_rows_rater2_cols'][1][1]} |",
         "",
         "## Disagreement cases (→ adjudicated by rater 3)",
         "",
         "| patient_id | source | rater 1 | rater 2 | consensus (rater 3) |",
         "|---|---|---|---|---|",
     ]
    for d in r["disagreements"]:
        L.append(f"| {d['patient_id']} | {d['source']} | {d['rater1']} | {d['rater2']} | {d['consensus']} |")
    L.append("")
    L.append(f"_Final consensus label = majority of three. Of the {r['n_disagreements']} "
             f"disagreements, rater 3's adjudication changed {r['n_consensus_changed_vs_rater1']} "
             f"of rater 1's original labels; the remainder were resolved in rater 1's favour._")
    (C.RESULT_DIR / "inter_rater.md").write_text("\n".join(x for x in L if x is not None),
                                                 encoding="utf-8")
    print("\n".join(x for x in L if x))


if __name__ == "__main__":
    run()
