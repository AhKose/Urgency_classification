"""Late-fusion models, computed post-hoc from saved per-fold test predictions
(no retraining, no tuned weights).

  det_fusion : equal-weight mean of P_urgent(det_count) and P_urgent(YOLO-TL).
               A fixed 0.5 weight is used deliberately -- with ~46 validation
               patients per fold a tuned weight overfits and performs worse than
               the fixed prior; 0.5 needs no justification and is leak-free.
  ens_cnn    : equal-weight mean of the three CNN models' P_urgent (context row).

Outputs runs/<name>/fold_<k>/{done.json, predictions.csv} in the standard format.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.evaluate import compute_metrics


def _load(model, k):
    return pd.read_csv(C.RUN_DIR / model / f"fold_{k}" / "predictions.csv").set_index("patient_id")


def _val_probs_cnn(model, k, ids):
    """re-score a CNN checkpoint on `ids` -> P(urgent)."""
    import torch
    from src.datasets import UrgencyDataset
    from src.models import build_model
    from src.train import _device, _evaluate

    rd = C.RUN_DIR / model / f"fold_{k}"
    cfg = json.loads((rd / "done.json").read_text())["config"]
    dev = _device()
    net = build_model(model, dropout=cfg["dropout_rate"], hidden=cfg.get("hidden_size", 256)).to(dev)
    net.load_state_dict(torch.load(rd / "model.pt", map_location=dev))
    man = pd.read_csv(C.MANIFEST_CSV)
    ds = UrgencyDataset(man, ids, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)
    _, preds = _evaluate(net, dl, dev)
    return dict(zip(preds["patient_id"], preds["prob_urgent"]))


def _val_probs_detcount(k, ids):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.detection_features import FEATURE_COLS
    from src.run_detfeat import _make_clf

    feat = pd.read_csv(C.DATA_DIR / "detection_features.csv").set_index("patient_id")
    man = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")
    fold = next(f for f in json.loads(C.SPLIT_JSON.read_text())["folds"] if f["fold"] == k)
    cfg = json.loads((C.RUN_DIR / "det_count" / f"fold_{k}" / "done.json").read_text())["config"]
    Xtr = feat.loc[fold["final_train"], FEATURE_COLS].to_numpy(float)
    ytr = man.loc[fold["final_train"], "urgency_int"].to_numpy(int)
    clf = _make_clf(cfg); clf.fit(Xtr, ytr)
    cls = list(clf.classes_)
    Xv = feat.loc[list(ids), FEATURE_COLS].to_numpy(float)
    return dict(zip(ids, clf.predict_proba(Xv)[:, cls.index(0)]))


def _save(name, k, ids, y_true, prob_urgent, extra=None):
    out = C.RUN_DIR / name / f"fold_{k}"
    out.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(prob_urgent) < 0.5).astype(int)
    m = compute_metrics(y_true, y_pred, prob_urgent)
    src = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")
    pd.DataFrame({"patient_id": ids, "source": src.loc[ids, "source"].to_numpy(),
                  "True_Label": y_true, "Pred": y_pred,
                  "Prob_urgent": prob_urgent, "Prob_non_urgent": 1 - np.asarray(prob_urgent),
                  "Correct": np.asarray(y_true) == y_pred}).to_csv(out / "predictions.csv", index=False)
    res = dict(model=name, fold=k, test_metrics=m, config=extra or {})
    (out / "done.json").write_text(json.dumps(res, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return m


def main():
    man = pd.read_csv(C.MANIFEST_CSV).set_index("patient_id")
    folds = json.loads(C.SPLIT_JSON.read_text())["folds"]

    for f in folds:
        k = f["fold"]
        val_ids = f["final_val"]
        det = _load("det_count", k); yolo = _load("yolo_tl", k)
        ids = list(det.index)
        yt = man.loc[ids, "urgency_int"].to_numpy(int)
        pf = 0.5 * det.loc[ids, "Prob_urgent"].to_numpy() + 0.5 * yolo.loc[ids, "Prob_urgent"].to_numpy()
        mf = _save("det_fusion", k, ids, yt, pf, extra=dict(weight_detcount=0.5))

        # equal-weight CNN ensemble (context row)
        probs = [_load(m, k).loc[ids, "Prob_urgent"].to_numpy()
                 for m in ("yolo_tl", "resnet50v2", "efficientnet_b3")]
        pe = np.mean(probs, axis=0)
        me = _save("ens_cnn", k, ids, yt, pe)

        # val predictions for det_fusion (operating-point threshold selection)
        dv = _val_probs_detcount(k, val_ids)
        cv = _val_probs_cnn("yolo_tl", k, val_ids)
        yv = man.loc[val_ids, "urgency_int"].to_numpy(int)
        pfv = np.array([0.5 * dv[i] + 0.5 * cv[i] for i in val_ids])
        pd.DataFrame({"patient_id": val_ids, "True_Label": yv,
                      "Prob_urgent": pfv, "Prob_non_urgent": 1 - pfv}).to_csv(
            C.RUN_DIR / "det_fusion" / f"fold_{k}" / "val_predictions.csv", index=False)
        print(f"fold {k}: det_fusion macro-F1={mf['f1_macro']:.3f}  "
              f"ens_cnn macro-F1={me['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
