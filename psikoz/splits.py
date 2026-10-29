"""Stage 4: subject-level splits, stratified by dataset x diagnosis.

Splits list every participant; QC filtering happens when a Dataset is built, so
changing QC thresholds never moves a subject between train and test.

Single-site splits (site_cv_<site>, written by single_site(); `python -m psikoz.splits --single-site`):
5-fold CV within one site, for every classification site that has both classes (ds005073 has no
participant without psychosis other than 7 bipolar patients and is left out). Existing split files are
never rewritten by this function.

External sites (config "external") are left out of every split except their own external_<site>
split, in which all other participants are training (80 %) and validation (20 %) data.
"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def kfold(part, n_folds, seed):
    strata = part["dataset"] + "_" + part["diagnosis"]
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    chunks = [part["subject"].iloc[idx].tolist() for _, idx in skf.split(part, strata)]
    folds = []
    for k in range(n_folds):
        val = (k + 1) % n_folds
        train = [s for j, c in enumerate(chunks) if j not in (k, val) for s in c]
        folds.append(dict(train=train, val=chunks[val], test=chunks[k]))
    return folds


def run(cfg):
    out = cfg["paths"]["splits"]
    out.mkdir(parents=True, exist_ok=True)
    part = pd.read_csv(cfg["paths"]["manifest"] / "participants.csv")
    external = part["dataset"].isin(cfg.get("external", []))
    part, ext = part[~external].reset_index(drop=True), part[external]
    n, seed = cfg["splits"]["n_folds"], cfg["splits"]["seed"]
    by_site = {d: g for d, g in part.groupby("dataset")}
    cobre, ds = by_site["cobre"], by_site["ds005073"]

    def holdout(train_part, test_part):
        tr, va = train_test_split(train_part["subject"], test_size=0.2, random_state=seed,
                                  stratify=train_part["dataset"] + "_" + train_part["diagnosis"])
        return [dict(train=tr.tolist(), val=va.tolist(), test=test_part["subject"].tolist())]

    splits = {
        "cobre_cv": kfold(cobre, n, seed),          # SZ vs HC within one site
        "pooled_cv": kfold(part, n, seed),          # all sites mixed
        "cobre_to_ds005073": holdout(cobre, ds),
    }
    if "ds000030" in by_site:
        splits["ds000030_cv"] = kfold(by_site["ds000030"], n, seed)   # HC / SZ / BP / ADHD within one site
        for site, g in by_site.items():             # train on the other sites, test on this one
            splits[f"leave_out_{site}"] = holdout(part[part["dataset"] != site], g)
    for site, g in ext.groupby("dataset"):
        splits[f"external_{site}"] = holdout(part, g)
    for name, folds in splits.items():
        (out / f"{name}.json").write_text(json.dumps(dict(name=name, seed=seed, folds=folds), indent=1))
        print(f"{name}: {len(folds)} fold(s), sizes train/val/test = "
              f"{len(folds[0]['train'])}/{len(folds[0]['val'])}/{len(folds[0]['test'])}")


SINGLE_SITES = ["cobre", "ds000030", "ds001461", "srpbs-KTT", "srpbs-KUT", "srpbs-SWA", "srpbs-UTO"]


def single_site(cfg):
    """site_cv_<site>: the kfold() of psikoz.splits restricted to one site (same seed and stratification by
    diagnosis; for COBRE identical to cobre_cv). Only missing files are written."""
    out = cfg["paths"]["splits"]
    part = pd.read_csv(cfg["paths"]["manifest"] / "participants.csv")
    n, seed = cfg["splits"]["n_folds"], cfg["splits"]["seed"]
    for site in SINGLE_SITES:
        f = out / f"site_cv_{site}.json"
        if f.exists():
            print(f"{f.name}: exists, kept")
            continue
        folds = kfold(part[part["dataset"] == site].reset_index(drop=True), n, seed)
        f.write_text(json.dumps(dict(name=f.stem, seed=seed, folds=folds), indent=1))
        print(f"{f.name}: {len(folds)} folds, train/val/test = "
              f"{len(folds[0]['train'])}/{len(folds[0]['val'])}/{len(folds[0]['test'])}")


if __name__ == "__main__":
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single-site", action="store_true", help="write the missing site_cv_<site> splits only")
    if ap.parse_args().single_site:
        single_site(load_config())
