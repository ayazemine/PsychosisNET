"""Feature engineering on top of the extracted ROI signals, and its evaluation.

    python -m psikoz.engineer build --features schaefer100s_gsr      # -> data/features/<set>/engineered.csv
    python -m psikoz.engineer eval  --features schaefer100s_gsr

Every engineered feature is computed from one subject's own run, so nothing is fitted across
subjects at this stage. Blocks:

  netfc      mean Fisher-z connectivity within / between the functional networks
  strength   per region: sum of positive and of negative correlations
  degree     per region: degree in the binary graph that keeps the strongest 15 % of connections
  cluster    per region: clustering coefficient in that graph
  particip   per region: participation coefficient with respect to the functional networks
  global     global efficiency, mean clustering, modularity of the network partition
  falff      per region: fractional amplitude of low-frequency fluctuations
  ac1        per region: lag-1 autocorrelation of the signal (clean consecutive frames)
  dynfc      temporal standard deviation of the network connectivity over sliding windows

`eval` scores each block, all blocks together and their combination with tangent-space
connectivity on the folds of a split. Scaling, regularisation strength (inner cross-validation)
and the tangent reference are fitted on the training part of each fold only.
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import shortest_path

from .config import load_config

DENSITY, WINDOW, STEP = 0.15, 30, 5


def network_means(z, groups):
    out = {}
    for a in range(len(groups)):
        for b in range(a, len(groups)):
            block = z[np.ix_(groups[a][1], groups[b][1])]
            if a == b:
                block = block[np.triu_indices(len(block), 1)]
            out[f"{groups[a][0]}-{groups[b][0]}"] = float(block.mean()) if block.size else 0.0
    return out


def subject_features(file, names, net):
    z = np.load(file)
    fc, ts, good = z["fc"].astype(float), z["timeseries"].astype(float), z["frame_mask"]
    r = np.tanh(fc)
    groups = [(g, np.flatnonzero(net == g)) for g in pd.unique(net)]
    f = {f"netfc__{k}": v for k, v in network_means(fc, groups).items()}

    f.update({f"strength__pos_{n}": v for n, v in zip(names, np.clip(r, 0, None).sum(0))})
    f.update({f"strength__neg_{n}": v for n, v in zip(names, np.clip(r, None, 0).sum(0))})

    iu = np.triu_indices(len(r), 1)
    adj = (r >= np.quantile(r[iu], 1 - DENSITY)).astype(float)
    np.fill_diagonal(adj, 0)
    k = adj.sum(0)
    clust = np.divide(np.einsum("ij,jk,ki->i", adj, adj, adj), k * (k - 1), out=np.zeros_like(k), where=k > 1)
    k_net = np.stack([adj[:, idx].sum(1) for _, idx in groups], 1)
    part = 1 - np.divide((k_net ** 2).sum(1), k ** 2, out=np.ones_like(k), where=k > 0)
    for block, values in (("degree", k), ("cluster", clust), ("particip", part)):
        f.update({f"{block}__{n}": v for n, v in zip(names, values)})
    dist = shortest_path(adj, unweighted=True)
    same = net[:, None] == net[None, :]
    f["global__efficiency"] = float((1 / dist[iu]).mean())
    f["global__clustering"] = float(clust.mean())
    f["global__modularity"] = float(((adj - np.outer(k, k) / max(k.sum(), 1)) * same).sum() / max(k.sum(), 1))

    f.update({f"falff__{n}": v for n, v in zip(names, z["falff"])})

    pair = good[1:] & good[:-1]
    x0, x1 = ts[:-1][pair], ts[1:][pair]
    ac1 = (x0 * x1).mean(0) / np.maximum(np.sqrt((x0 ** 2).mean(0) * (x1 ** 2).mean(0)), 1e-12)
    f.update({f"ac1__{n}": v for n, v in zip(names, ac1)})

    windows = []
    for s in range(0, len(ts) - WINDOW + 1, STEP):
        m = good[s:s + WINDOW]
        if m.sum() >= 20:
            w = ts[s:s + WINDOW][m]
            sd = w.std(0)
            c = np.corrcoef(w[:, sd > 0].T)
            full = np.zeros_like(r)
            full[np.ix_(sd > 0, sd > 0)] = np.arctanh(np.clip(c, -0.999, 0.999))
            np.fill_diagonal(full, 0)
            windows.append(network_means(full, groups))
    dyn = pd.DataFrame(windows).std(0) if len(windows) > 1 else pd.Series(0.0, index=network_means(fc, groups).keys())
    f.update({f"dynfc__{k}": v for k, v in dyn.items()})
    return f


def build(cfg, name, jobs):
    feat = cfg["paths"]["features"] / name
    index = pd.read_csv(feat / "index.csv")
    labels = pd.read_csv(feat / "labels.csv")
    rows = Parallel(jobs)(delayed(subject_features)(cfg["root"] / f, labels["name"].to_numpy(), labels["network"].to_numpy())
                          for f in index["file"])
    table = pd.DataFrame(rows).fillna(0.0)
    table.insert(0, "file", index["file"].to_numpy())
    table.to_csv(feat / "engineered.csv", index=False)
    blocks = pd.Series([c.split("__")[0] for c in table.columns[1:]]).value_counts(sort=False)
    print(f"{name}: {len(table)} runs, {table.shape[1] - 1} features\n{blocks.to_string()}")


# ------------------------------------------------------------------ evaluation
def fit_predict(xtr, ytr, xte, seed):
    """Standardise, pick C by inner cross-validation, return (P(class 1) test, P train, chosen C)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = xtr.std(0) > 1e-10
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, scoring="roc_auc",
                                                                max_iter=3000, class_weight="balanced", random_state=seed))
    clf.fit(xtr[:, keep], ytr)
    return clf.predict_proba(xte[:, keep])[:, 1], clf.predict_proba(xtr[:, keep])[:, 1], float(clf[-1].C_[0])


def eval_fold(cfg, name, args, fold, parts):
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    from .benchmark import design, load

    d = load(cfg, name, args)
    eng = pd.read_csv(cfg["paths"]["features"] / name / "engineered.csv").set_index("file").loc[d["index"]["file"]]
    block_of = np.array([c.split("__")[0] for c in eng.columns])
    x = eng.to_numpy(float)
    pos = pd.Series(np.arange(len(d["y"])), index=d["index"]["subject"])
    tr = pos.reindex(parts["train"] + parts["val"]).dropna().astype(int).to_numpy()
    te = pos.reindex(parts["test"]).dropna().astype(int).to_numpy()
    ytr, yte = d["y"][tr], d["y"][te]
    rows, prob = [], {}

    def add(label, n_feat, p_te, p_tr, extra=""):
        prob[label] = p_te
        rows.append(dict(fold=fold, features=label, n_features=n_feat, train_auc=roc_auc_score(ytr, p_tr),
                         test_auc=roc_auc_score(yte, p_te), note=extra))

    for block in list(pd.unique(block_of)) + ["all engineered"]:
        cols = np.ones(len(block_of), bool) if block == "all engineered" else block_of == block
        p_te, p_tr, c = fit_predict(x[tr][:, cols], ytr, x[te][:, cols], fold)
        add(block, int(cols.sum()), p_te, p_tr, f"C={c:g}")
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                        random_state=fold, n_jobs=2).fit(x[tr], ytr)
    add("all engineered (XGBoost)", x.shape[1], xgb.predict_proba(x[te])[:, 1], xgb.predict_proba(x[tr])[:, 1])

    ttr, tte = design(d, "tangent", False, tr, te)
    p_te, p_tr, c = fit_predict(ttr, ytr, tte, fold)
    add("tangent", ttr.shape[1], p_te, p_tr, f"C={c:g}")
    p_te, p_tr, c = fit_predict(np.hstack([ttr, x[tr]]), ytr, np.hstack([tte, x[te]]), fold)
    add("tangent + engineered (one model)", ttr.shape[1] + x.shape[1], p_te, p_tr, f"C={c:g}")
    rows.append(dict(fold=fold, features="tangent + engineered (mean of two models)", n_features=ttr.shape[1] + x.shape[1],
                     train_auc=np.nan, test_auc=roc_auc_score(yte, (prob["tangent"] + prob["all engineered"]) / 2), note=""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "eval"])
    ap.add_argument("--features", nargs="*", default=["schaefer100s_gsr"])
    ap.add_argument("--split", default="pooled_cv")
    ap.add_argument("--label", default="psychosis")
    ap.add_argument("--diagnoses", nargs="*", default=["HC", "SZ", "BP"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    for name in args.features:
        if args.command == "build":
            build(cfg, name, args.jobs)
            continue
        folds = json.loads((cfg["paths"]["splits"] / f"{args.split}.json").read_text())["folds"]
        rows = Parallel(min(args.jobs, len(folds)))(delayed(eval_fold)(cfg, name, args, k, p) for k, p in enumerate(folds))
        res = pd.DataFrame([r for part in rows for r in part])
        out = cfg["paths"]["data"] / "runs" / f"engineered_{args.split}_{args.label}_{name}.csv"
        res.to_csv(out, index=False)
        table = res.groupby("features", sort=False).agg(n_features=("n_features", "first"), train_auc=("train_auc", "mean"),
                                                        test_auc=("test_auc", "mean"), test_sd=("test_auc", "std"))
        print(f"\n{name}  ({args.label}, {' '.join(args.diagnoses)})\n{table.round(3).to_string()}\nresults -> {out}")


if __name__ == "__main__":
    main()
