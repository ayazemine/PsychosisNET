"""Post-hoc calibration of a trained PsychosisNET run, fitted on the validation part of each fold.

    python -m psikoz.calibrate --run v3_spd_skip_szhc

Calibration uses the validation and test probabilities saved by psikoz.train (val_predictions.csv,
test_predictions.csv); runs of the model versions before v11 can be re-scored with archive/v10/calibrate.py.

  Platt scaling   logistic regression on the validation logits -> calibrated test probabilities,
                  so that folds share one probability scale and can be pooled
  threshold       decision threshold that maximises balanced accuracy on the validation part

The test part is never used for fitting. AUC within a fold is unchanged by construction.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve

from .config import load_config


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    cli = ap.parse_args()
    cfg = load_config()
    run = cfg["paths"]["data"] / "runs" / cli.run
    val, test = (pd.read_csv(run / f"{part}_predictions.csv") for part in ("val", "test"))
    folds = [(v["p1"].to_numpy(), v["y"].to_numpy(), t["p1"].to_numpy(), t["y"].to_numpy(), t["subject"].tolist(), t["site"].to_numpy())
             for (_, v), (_, t) in zip(val.groupby("fold"), test.groupby("fold"))]
    rows, preds = [], []
    for fold, (pv, yv, pt, yt, sub, site) in enumerate(folds):
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv)[:, None], yv)
        cv, ct = (platt.predict_proba(logit(p)[:, None])[:, 1] for p in (pv, pt))
        thr = best_threshold(yv, cv)
        rows.append(dict(fold=fold, auc=roc_auc_score(yt, pt), bacc_raw=balanced_accuracy_score(yt, pt > 0.5),
                         bacc_platt=balanced_accuracy_score(yt, ct > 0.5), threshold=thr,
                         bacc_threshold=balanced_accuracy_score(yt, ct > thr),
                         sensitivity=float((ct[yt == 1] > thr).mean()), specificity=float((ct[yt == 0] <= thr).mean()),
                         brier_raw=brier_score_loss(yt, pt), brier_platt=brier_score_loss(yt, ct)))
        preds.append(pd.DataFrame(dict(fold=fold, subject=sub, site=site, y=yt, p_raw=pt, p_calibrated=ct,
                                       decision=(ct > thr).astype(int))))
    report(cli.run, run, rows, preds)


def report(run_name, run, rows, preds):
    res, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    res.to_csv(run / "calibration.csv", index=False)
    preds.to_csv(run / "test_predictions_calibrated.csv", index=False)
    print(run_name, "\n", res.round(3).to_string(index=False))
    print("\nmean +- sd over folds\n", res.drop(columns="fold").agg(["mean", "std"]).T.round(3).to_string())
    print("\npooled over folds")
    for name, col in (("raw", "p_raw"), ("calibrated", "p_calibrated")):
        print(f"  AUC {name:10s} {roc_auc_score(preds.y, preds[col]):.3f}   per site",
              {int(s): round(roc_auc_score(g.y, g[col]), 3) for s, g in preds.groupby("site") if g.y.nunique() > 1})
    print(f"  balanced accuracy, raw 0.5 threshold      {balanced_accuracy_score(preds.y, preds.p_raw > 0.5):.3f}")
    print(f"  balanced accuracy, validation thresholds  {balanced_accuracy_score(preds.y, preds.decision):.3f}"
          f"   sensitivity {preds.decision[preds.y == 1].mean():.3f}   specificity {1 - preds.decision[preds.y == 0].mean():.3f}")


if __name__ == "__main__":
    main()
