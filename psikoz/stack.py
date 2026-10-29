"""Multi-view feature extraction with stacking (after EMPaSchiz, Kalmady et al. 2019, and the
connectome benchmarks of Dadi et al. 2019 and Pervaiz et al. 2020).

    python -m psikoz.stack --name stack_psychosis

Views, computed for every feature set (parcellation x denoising):

  tangent    tangent-space covariance, Ledoit-Wolf shrinkage (reference: training part)
  partial    partial correlation from the Ledoit-Wolf precision matrix
  corr       Fisher-z correlation
  slow5      tangent-space covariance of the 0.010-0.027 Hz band
  slow4      tangent-space covariance of the 0.027-0.073 Hz band
  dfc        temporal standard deviation of every connection over sliding windows
  lag1       lag-1 cross-correlation (asymmetric, directed interactions)
  falff      fractional amplitude of low-frequency fluctuations per region

One class-weighted logistic regression per view and regularisation strength. Their out-of-fold
probabilities on the training part (inner cross-validation) train a second-level logistic
regression; the test part of each outer fold is scored once.

Label-free quantities (tangent reference, feature scaling) are fitted on the whole training part
of the outer fold and reused inside the inner folds; labels of held-out subjects are never used.
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score as _auc
from sklearn.model_selection import StratifiedKFold

from .benchmark import load
from .config import load_config

CS = (1e-3, 1e-2)
BANDS = {"slow5": (0.010, 0.027), "slow4": (0.027, 0.073)}
WINDOW, STEP = 30, 5


def subject_views(file, cfg):
    """Views that need no other subject."""
    z = np.load(cfg["root"] / file)
    ts, good = z["timeseries"].astype(float), z["frame_mask"]
    x = ts[good]
    n = x.shape[1]
    iu = np.triu_indices(n, 1)
    out = {}
    prec = LedoitWolf().fit(x).precision_
    dg = np.sqrt(np.diag(prec))
    out["partial"] = np.arctanh(np.clip(-prec / np.outer(dg, dg), -0.999, 0.999))[iu]
    wins = []
    for s in range(0, len(x) - WINDOW + 1, STEP):
        w = x[s:s + WINDOW]
        sd = w.std(0)
        c = np.zeros((n, n))
        ok = sd > 0
        c[np.ix_(ok, ok)] = np.corrcoef(w[:, ok].T)
        wins.append(np.arctanh(np.clip(c[iu], -0.999, 0.999)))
    out["dfc"] = np.std(wins, 0) if len(wins) > 1 else np.zeros(len(iu[0]))
    pair = good[1:] & good[:-1]
    a, b = ts[:-1][pair], ts[1:][pair]
    a, b = ((v - v.mean(0)) / np.maximum(v.std(0), 1e-8) for v in (a, b))
    out["lag1"] = (a.T @ b / max(len(a), 1)).ravel()
    bands = {}
    for name, band in BANDS.items():
        sos = signal.butter(2, band, btype="band", fs=1 / float(z["tr"]), output="sos")
        bands[name] = signal.sosfiltfilt(sos, ts, axis=0)[good]
    return out, bands


def rotate(xtr, xte):
    """Standardise and rotate to the row space of the training matrix (lossless for L2 linear models)."""
    mu, sd = xtr.mean(0), xtr.std(0)
    keep = sd > 1e-8
    xtr, xte = (xtr[:, keep] - mu[keep]) / sd[keep], (xte[:, keep] - mu[keep]) / sd[keep]
    u, s, vt = np.linalg.svd(xtr, full_matrices=False)
    return u * s, xte @ vt.T


def run_set(cfg, name, args, folds):
    from nilearn.connectome import ConnectivityMeasure

    d = load(cfg, name, args)
    keep = d["index"]["subject"].isin(args.subjects).to_numpy()
    idx = np.flatnonzero(keep)
    subjects = d["index"]["subject"].to_numpy()[idx]
    y = d["y"][idx]
    per = [subject_views(f, cfg) for f in d["index"]["file"].to_numpy()[idx]]
    fixed = {"corr": d["corr"][idx], "falff": d["falff"][idx],
             **{k: np.stack([p[0][k] for p in per]) for k in ("partial", "dfc", "lag1")}}
    series = {"tangent": [d["ts"][i] for i in idx], **{b: [p[1][b] for p in per] for b in BANDS}}
    pos = pd.Series(np.arange(len(subjects)), index=subjects)
    strata_all = pd.factorize(d["index"]["dataset"].to_numpy()[idx])[0] * 10 + y
    rows = []
    for k, parts in enumerate(folds):
        dev = pos.reindex(parts["train"] + parts["val"]).dropna().astype(int).to_numpy()
        test = pos.reindex(parts["test"]).dropna().astype(int).to_numpy()
        inner = list(StratifiedKFold(args.inner, shuffle=True, random_state=k).split(dev, strata_all[dev]))
        for view in list(series) + list(fixed):
            if view in series:
                cm = ConnectivityMeasure(kind="tangent", vectorize=True).fit([series[view][i] for i in dev])
                xtr, xte = (cm.transform([series[view][i] for i in part]) for part in (dev, test))
            else:
                xtr, xte = fixed[view][dev], fixed[view][test]
            rtr, rte = rotate(xtr, xte)
            for c in CS:
                oof = np.zeros(len(dev))
                for tr, te in inner:
                    oof[te] = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(rtr[tr], y[dev[tr]]).predict_proba(rtr[te])[:, 1]
                p = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(rtr, y[dev]).predict_proba(rte)[:, 1]
                rows.append(dict(fold=k, model=f"{name}|{view}|C{c:g}", dev=subjects[dev], test=subjects[test], oof=oof, prob=p))
        print(f"{name} fold {k} done", flush=True)
    return rows


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
    ap.add_argument("--features", nargs="*", default=["schaefer100s_gsr", "schaefer200s_gsr", "schaefer100s", "schaefer200s"])
    ap.add_argument("--split", default="pooled_cv")
    ap.add_argument("--label", default="psychosis")
    ap.add_argument("--diagnoses", nargs="*", default=["HC", "SZ", "BP"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--name", default="stack_psychosis")
    args = ap.parse_args()
    cfg = load_config()
    folds = json.loads((cfg["paths"]["splits"] / f"{args.split}.json").read_text())["folds"]
    # Subjects that pass quality control in every feature set, so that all views describe the same people.
    common = None
    for name in args.features:
        s = set(load_index(cfg, name, args)["subject"])
        common = s if common is None else common & s
    args.subjects = sorted(common)
    ref = load_index(cfg, args.features[0], args).set_index("subject").loc[args.subjects]
    print(f"subjects: {len(ref)}\n{pd.crosstab(ref['dataset'], ref[args.label].astype(int)).to_string()}", flush=True)

    rows = [r for part in Parallel(len(args.features))(delayed(run_set)(cfg, n, args, folds) for n in args.features) for r in part]
    models = sorted({r["model"] for r in rows})
    res, preds, weights = [], [], []
    for k in range(len(folds)):
        fr = {r["model"]: r for r in rows if r["fold"] == k}
        dev, test = fr[models[0]]["dev"], fr[models[0]]["test"]
        ydev, yte = (ref.loc[s, args.label].astype(int).to_numpy() for s in (dev, test))
        zdev = np.column_stack([logit(fr[m]["oof"]) for m in models])
        zte = np.column_stack([logit(fr[m]["prob"]) for m in models])
        inner_auc = np.array([roc_auc_score(ydev, zdev[:, j]) for j in range(len(models))])
        best = int(inner_auc.argmax())
        meta = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(zdev, ydev)
        out = {"reference (schaefer100s_gsr tangent C0.001)": 1 / (1 + np.exp(-zte[:, models.index("schaefer100s_gsr|tangent|C0.001")]))
               if "schaefer100s_gsr|tangent|C0.001" in models else None,
               "best single view (inner AUC)": 1 / (1 + np.exp(-zte[:, best])),
               "mean of all views": 1 / (1 + np.exp(-zte.mean(1))),
               "stacking": meta.predict_proba(zte)[:, 1]}
        weights.append(pd.Series(meta.coef_[0], index=models))
        for how, p in out.items():
            if p is None:
                continue
            res.append(dict(fold=k, how=how, auc=roc_auc_score(yte, p), bacc=balanced_accuracy(yte, p > 0.5),
                            sensitivity=float((p[yte == 1] > 0.5).mean()), specificity=float((p[yte == 0] <= 0.5).mean()),
                            chosen=models[best] if how.startswith("best") else ""))
            preds.append(pd.DataFrame(dict(fold=k, how=how, subject=test, dataset=ref.loc[test, "dataset"].to_numpy(), y=yte, p1=p)))
        for j, m in enumerate(models):
            res.append(dict(fold=k, how="view: " + m, auc=roc_auc_score(yte, zte[:, j]), bacc=balanced_accuracy(yte, zte[:, j] > 0),
                            inner_auc=inner_auc[j]))
    res, preds = pd.DataFrame(res), pd.concat(preds, ignore_index=True)
    out = cfg["paths"]["data"] / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    pd.concat(weights, axis=1).mean(1).sort_values(ascending=False).to_csv(out / "stacking_weights.csv", header=["weight"])
    views = res[res["how"].str.startswith("view")].copy()
    views[["set", "view", "C"]] = views["how"].str[6:].str.split("|", expand=True)
    by = views.groupby(["set", "view", "C"])["auc"].mean().reset_index()
    print("\ntest AUC per view (best C per cell)\n", by.groupby(["view", "set"])["auc"].max().unstack().round(3).to_string())
    top = res[~res["how"].str.startswith("view")]
    print("\n", top.groupby("how", sort=False)[["auc", "bacc", "sensitivity", "specificity"]].agg(["mean", "std"]).round(3).to_string())
    print(top[top["chosen"] != ""][["fold", "chosen"]].to_string(index=False))
    for how, g in preds.groupby("how", sort=False):
        site = {s: round(roc_auc_score(h["y"], h["p1"]), 3) for s, h in g.groupby("dataset") if h["y"].nunique() > 1}
        print(f"{how}: pooled AUC {roc_auc_score(g['y'], g['p1']):.3f} per site {site}")
    print("results ->", out)


def load_index(cfg, name, args):
    index = pd.read_csv(cfg["paths"]["features"] / name / "index.csv")
    index = index[(index["task"] == "rest") & (index["qc_pass"] == 1) & index[args.label].notna()]
    index = index[index["diagnosis"].isin(args.diagnoses)]
    index = index[index["role"].fillna("main") == "main"] if "role" in index.columns else index
    if args.datasets:
        from .config import in_datasets
        index = index[in_datasets(index["dataset"], args.datasets)]
    return index.drop_duplicates("subject")


if __name__ == "__main__":
    main()
