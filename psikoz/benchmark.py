"""Nested cross-validation over feature sets, connectivity measures and linear classifiers.

    python -m psikoz.benchmark --split pooled_cv --label psychosis --diagnoses HC SZ BP

Outer loop: the test parts of the split file. Inner loop: stratified k-fold on train + val of the
same outer fold. Every configuration (feature set x measure x ComBat x classifier) gets an inner
score; the configuration with the best inner score is applied to the outer test part ("selected"),
as is the mean probability of the five best ("top5"). Only these two rows are unbiased estimates;
the per-configuration test scores are descriptive, and picking the best of them would be optimistic.
Everything that is fitted (ComBat, tangent reference, scaler, classifier) sees training data only.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import in_datasets, load_config, main_rows
from .features import feature_sets
from .harmonize import ComBat

MEASURES = ("corr", "tangent", "corr+falff")
MODELS = {f"{m}_C{c:g}": (m, c) for m in ("logreg", "svm") for c in (1e-3, 1e-2, 1e-1)}


def load(cfg, name, args):
    index = pd.read_csv(cfg["paths"]["features"] / name / "index.csv")
    index = index[(index["task"] == "rest") & (index["qc_pass"] == 1) & index[args.label].notna()]
    index = main_rows(index[index["diagnosis"].isin(args.diagnoses)])
    if args.datasets:
        index = index[in_datasets(index["dataset"], args.datasets)]
    index = index.drop_duplicates("subject").sort_values("subject").reset_index(drop=True)
    files = [np.load(cfg["root"] / f) for f in index["file"]]
    iu = np.triu_indices(files[0]["fc"].shape[0], 1)
    cov = np.column_stack([index["age"].fillna(index["age"].mean()), index["sex"] == "M"]).astype(float)
    return dict(index=index, y=index[args.label].astype(int).to_numpy(), site=pd.factorize(index["dataset"])[0],
                cov=cov, corr=np.stack([z["fc"][iu] for z in files]).astype(float),
                falff=np.stack([z["falff"] for z in files]).astype(float),
                ts=[z["timeseries"][z["frame_mask"]].astype(float) for z in files])


def design(d, measure, combat, tr, te):
    """Feature matrices for one train / test partition."""
    if measure == "tangent":
        from nilearn.connectome import ConnectivityMeasure
        cm = ConnectivityMeasure(kind="tangent", vectorize=True).fit([d["ts"][i] for i in tr])
        xtr, xte = (cm.transform([d["ts"][i] for i in part]) for part in (tr, te))
    else:
        x = d["corr"] if measure == "corr" else np.hstack([d["corr"], d["falff"]])
        xtr, xte = x[tr], x[te]
    if combat:
        keep = xtr.std(0) > 1e-8
        cb = ComBat().fit(xtr[:, keep], d["site"][tr], d["cov"][tr], np.eye(d["y"].max() + 1)[d["y"][tr]][:, 1:])
        xtr, xte = (cb.transform(a[:, keep], d["site"][p], d["cov"][p]) for a, p in ((xtr, tr), (xte, te)))
    sc = StandardScaler().fit(xtr)
    return sc.transform(xtr), sc.transform(xte)


def predict(xtr, ytr, xte):
    """{model: P(class 1)}. Both classifiers are linear, so they work on the Gram matrix (n << p)."""
    ktr, kte = xtr @ xtr.T / xtr.shape[1], xte @ xtr.T / xtr.shape[1]
    w, v = np.linalg.eigh(ktr)
    root = v * np.sqrt(w.clip(0))                                  # ktr = root root'
    rte = kte @ v / np.sqrt(w.clip(1e-10)) * (w > 1e-10)
    out = {}
    for name, (kind, c) in MODELS.items():
        c = c * xtr.shape[1]                                        # same scale as C on the raw features
        if kind == "logreg":
            m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(root, ytr)
            out[name] = m.predict_proba(rte)[:, 1]
        else:
            m = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(ktr, ytr)
            s = m.decision_function(kte)
            out[name] = 1 / (1 + np.exp(-s))
    return out


def run_fold(cfg, name, args, fold, parts):
    d = load(cfg, name, args)
    pos = pd.Series(np.arange(len(d["y"])), index=d["index"]["subject"])
    dev = pos.reindex(parts["train"] + parts["val"]).dropna().astype(int).to_numpy()
    test = pos.reindex(parts["test"]).dropna().astype(int).to_numpy()
    strata = d["site"][dev] * 10 + d["y"][dev]
    inner = list(StratifiedKFold(args.inner, shuffle=True, random_state=args.seed + fold).split(dev, strata))
    rows = []
    for measure, combat in itertools.product(MEASURES, (False, True)):
        oof = {m: np.zeros(len(dev)) for m in MODELS}
        for tr, te in inner:
            xtr, xte = design(d, measure, combat, dev[tr], dev[te])
            for m, p in predict(xtr, d["y"][dev[tr]], xte).items():
                oof[m][te] = p
        xtr, xte = design(d, measure, combat, dev, test)
        for m, p in predict(xtr, d["y"][dev], xte).items():
            rows.append(dict(fold=fold, features=name, measure=measure, combat=combat, model=m,
                             inner_auc=roc_auc_score(d["y"][dev], oof[m]), prob=p,
                             y=d["y"][test], subject=d["index"]["subject"].to_numpy()[test],
                             dataset=d["index"]["dataset"].to_numpy()[test]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="pooled_cv")
    ap.add_argument("--label", default="psychosis")
    ap.add_argument("--diagnoses", nargs="*", default=["HC", "SZ", "BP"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--features", nargs="*", default=None, help="feature sets (default: all in the config)")
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = load_config()
    folds = json.loads((cfg["paths"]["splits"] / f"{args.split}.json").read_text())["folds"]
    names = args.features or list(feature_sets(cfg))
    jobs = [delayed(run_fold)(cfg, n, args, k, parts) for n in names for k, parts in enumerate(folds)]
    rows = [r for part in Parallel(args.jobs, verbose=5)(jobs) for r in part]
    res = pd.DataFrame(rows)
    res["config"] = res["features"] + "|" + res["measure"] + "|" + np.where(res["combat"], "combat", "raw") + "|" + res["model"]

    def score(g):
        y, p = np.concatenate(g["y"].tolist()), np.concatenate(g["prob"].tolist())
        return roc_auc_score(y, p), balanced_accuracy_score(y, p > 0.5)

    table = res.groupby("config").apply(lambda g: pd.Series(dict(
        inner_auc=g["inner_auc"].mean(), test_auc_mean=np.mean([roc_auc_score(y, p) for y, p in zip(g["y"], g["prob"])]),
        test_auc_sd=np.std([roc_auc_score(y, p) for y, p in zip(g["y"], g["prob"])], ddof=1),
        test_auc_pooled=score(g)[0], test_bacc_pooled=score(g)[1]))).sort_values("inner_auc", ascending=False)

    nested, preds = [], []
    for k, g in res.groupby("fold"):
        g = g.sort_values("inner_auc", ascending=False)
        ref = g.iloc[0]
        order = pd.Series(np.arange(len(ref["subject"])), index=ref["subject"])
        top = np.mean([r["prob"][pd.Series(np.arange(len(r["subject"])), index=r["subject"]).reindex(order.index).to_numpy()]
                       for _, r in g.head(5).iterrows()], 0)
        for how, p in (("selected", ref["prob"]), ("top5", top)):
            nested.append(dict(fold=k, how=how, config=ref["config"] if how == "selected" else "mean of 5 best inner",
                               auc=roc_auc_score(ref["y"], p), bacc=balanced_accuracy_score(ref["y"], p > 0.5)))
            preds.append(pd.DataFrame(dict(fold=k, how=how, subject=ref["subject"], dataset=ref["dataset"], y=ref["y"], p1=p)))
    nested, preds = pd.DataFrame(nested), pd.concat(preds, ignore_index=True)

    out = cfg["paths"]["data"] / "runs" / (args.name or f"benchmark_{args.split}_{args.label}")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "configs.csv")
    nested.to_csv(out / "nested.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(f"\nsubjects: {len(preds) // 2}   configurations: {len(table)}")
    print("\n15 best configurations by INNER score (test columns are descriptive)\n", table.head(15).round(3).to_string())
    print("\nby factor: mean pooled test AUC over configurations")
    parts = table.index.to_series().str.split("|", expand=True).set_axis(["features", "measure", "combat", "model"], axis=1)
    for c in parts:
        print(table["test_auc_pooled"].groupby(parts[c]).mean().round(3).to_string(), "\n")
    print("nested estimate (unbiased)\n", nested.to_string(index=False))
    summary = nested.groupby("how")[["auc", "bacc"]].agg(["mean", "std"]).round(3)
    print("\n", summary.to_string())
    for how, g in preds.groupby("how"):
        site = {s: round(roc_auc_score(h["y"], h["p1"]), 3) for s, h in g.groupby("dataset") if h["y"].nunique() > 1}
        print(how, "pooled AUC", round(roc_auc_score(g["y"], g["p1"]), 3), "per site", site)
    print("results ->", out)


if __name__ == "__main__":
    main()
