"""Fixed-weight combination of PsychosisNET with the linear multi-view models of psikoz.stack.

    python -m psikoz.combine --net v9_pre_lr1e-3 --stack stack_psychosis

Both sources were scored on the same outer test folds. The linear models were refitted on the whole
development part (training + validation), so their validation scores are in-sample and cannot train
a combiner; the combination therefore has no fitted weights: the logits are averaged with equal weight.
Nothing is estimated on the test part. Decisions use the 0.5 threshold (all members are class-weighted).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score as _auc

from .config import load_config


def roc_auc_score(y, p):
    """AUC, or NaN when the test part holds one class only (a held-out patient-only site)."""
    return _auc(y, p) if len(np.unique(y)) > 1 else float("nan")


def balanced_accuracy(y, d):
    return balanced_accuracy_score(y, d) if len(np.unique(y)) > 1 else float((np.asarray(d) == np.asarray(y)).mean())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="v9_pre_lr1e-3")
    ap.add_argument("--stack", default="stack_psychosis")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    runs = load_config()["paths"]["data"] / "runs"
    net = pd.read_csv(runs / args.net / "test_predictions.csv")[["fold", "subject", "y", "p1"]].rename(columns={"p1": "net"})
    st = pd.read_csv(runs / args.stack / "test_predictions.csv")
    st = st.pivot_table(index=["fold", "subject", "dataset"], columns="how", values="p1").reset_index()
    d = net.merge(st, on=["fold", "subject"], how="inner")
    print(f"subjects in both: {len(d)} (net {len(net)}, stack {st.shape[0]})")
    ref = "reference (schaefer100s_gsr tangent C0.001)"
    members = {"PsychosisNET": ["net"], "tangent reference": [ref], "mean of all views": ["mean of all views"],
               "stacking": ["stacking"], "PsychosisNET + tangent": ["net", ref],
               "PsychosisNET + mean of views": ["net", "mean of all views"], "PsychosisNET + stacking": ["net", "stacking"]}
    rows, preds = [], []
    for how, cols in members.items():
        z = np.mean([logit(d[c].to_numpy()) for c in cols], 0)
        p = 1 / (1 + np.exp(-z))
        for k, g in d.assign(p=p).groupby("fold"):
            y, q = g["y"].to_numpy(), g["p"].to_numpy()
            rows.append(dict(how=how, fold=k, auc=roc_auc_score(y, q), bacc=balanced_accuracy(y, q > 0.5),
                             sensitivity=float((q[y == 1] > 0.5).mean()), specificity=float((q[y == 0] <= 0.5).mean())))
        preds.append(d[["fold", "subject", "dataset", "y"]].assign(how=how, p1=p))
    res, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    out = runs / (args.name or f"combine_{args.net}")
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    print(res.groupby("how", sort=False)[["auc", "bacc", "sensitivity", "specificity"]].agg(["mean", "std"]).round(3).to_string())
    for how, g in preds.groupby("how", sort=False):
        site = {s: round(roc_auc_score(h["y"], h["p1"]), 3) for s, h in g.groupby("dataset") if h["y"].nunique() > 1}
        print(f"{how}: pooled AUC {roc_auc_score(g['y'], g['p1']):.3f} per site {site}")
    print("results ->", out)


if __name__ == "__main__":
    main()
