"""Summary of the leave-site-out evaluation (run_lso.sh): one row per held-out site and method.

    python -m psikoz.lso

Methods: linear models of psikoz.stack, PsychosisNET with and without label-free pretraining, and the
equal-weight logit average of pretrained PsychosisNET and the mean of the 32 linear views.
Pooled scores concatenate the eight held-out sites. ds005073 has no healthy controls: its AUC separates
psychotic from non-psychotic bipolar patients (23 vs 7) and is reported, but the mean site AUC is also
given without it.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .config import load_config

SITES = ["cobre", "ds000030", "ds001461", "ds005073", "srpbs-KTT", "srpbs-KUT", "srpbs-SWA", "srpbs-UTO"]
LINEAR = {"tangent reference": "reference (schaefer100s_gsr tangent C0.001)",
          "mean of 32 views": "mean of all views", "stacking": "stacking"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    runs = load_config()["paths"]["data"] / "runs"
    preds = []
    for site in SITES:
        f = runs / f"lso_stack_{site}" / "test_predictions.csv"
        if f.exists():
            st = pd.read_csv(f)
            for name, how in LINEAR.items():
                preds.append(st[st["how"] == how][["subject", "y", "p1"]].assign(method=name, site=site))
        for tag, name in (("lso_net", "PsychosisNET (pretrained)"), ("lso_net0", "PsychosisNET (no pretraining)")):
            f = runs / f"{tag}_{site}" / "test_predictions.csv"
            if f.exists():
                preds.append(pd.read_csv(f)[["subject", "y", "p1"]].assign(method=name, site=site))
    d = pd.concat(preds, ignore_index=True)
    a = d[d["method"] == "PsychosisNET (pretrained)"].merge(d[d["method"] == "mean of 32 views"], on=["subject", "site", "y"])
    if len(a):
        z = (logit(a["p1_x"].to_numpy()) + logit(a["p1_y"].to_numpy())) / 2
        d = pd.concat([d, a[["subject", "y", "site"]].assign(method="PsychosisNET + mean of views", p1=1 / (1 + np.exp(-z)))])
    rows = []
    for (method, site), g in d.groupby(["method", "site"]):
        y, p = g["y"].to_numpy(), g["p1"].to_numpy()
        rows.append(dict(method=method, site=site, n=len(y), patients=int(y.sum()),
                         auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                         sensitivity=float((p[y == 1] > 0.5).mean()) if y.any() else np.nan,
                         specificity=float((p[y == 0] <= 0.5).mean()) if (y == 0).any() else np.nan))
    res = pd.DataFrame(rows)
    out = runs / "lso_summary"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "per_site.csv", index=False)
    d.to_csv(out / "test_predictions.csv", index=False)
    print(res.pivot(index="site", columns="method", values="auc").round(3).to_string())
    print("\nds005073 (patients only) sensitivity:",
          res[res["site"] == "ds005073"].set_index("method")["sensitivity"].round(3).to_dict())
    print("\npooled over held-out sites")
    for method, g in d.groupby("method"):
        sites = g["site"].nunique()
        both = res[(res["method"] == method) & res["auc"].notna()]["auc"]
        print(f"  {method:32s} sites {sites}  pooled AUC {roc_auc_score(g['y'], g['p1']):.3f}"
              f"  mean site AUC {both.mean():.3f} +- {both.std():.3f}"
              f"  (without ds005073 {res[(res['method'] == method) & (res['site'] != 'ds005073')]['auc'].mean():.3f})"
              f"  bacc {balanced_accuracy_score(g['y'], g['p1'] > 0.5):.3f}"
              f"  sens {(g['p1'][g['y'] == 1] > 0.5).mean():.3f}  spec {(g['p1'][g['y'] == 0] <= 0.5).mean():.3f}")
    print("results ->", out)


if __name__ == "__main__":
    main()
