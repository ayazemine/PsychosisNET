"""Does synthetic minority data help? Class-imbalance remedies compared on the folds of a split.

    python -m psikoz.augment --features schaefer100s_gsr --datasets cobre ds000030 ds001461 srpbs

Features: tangent-space connectivity (reference fitted on the training part), linear classifier.
Every remedy sees the training part of the fold only; the test part stays real and untouched.

  none       plain logistic regression
  weights    class-weighted loss (what the benchmark and PsychosisNET already use)
  smote      synthetic patients on the line between a patient and one of its 5 nearest patients
  gan        conditional WGAN-GP (Gulrajani et al. 2017) trained on the leading principal
             components of the training part; synthetic patients are drawn until classes are equal
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .benchmark import design, load
from .config import load_config


def smote(x, n, rng, k=5):
    d = ((x[:, None] - x[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argsort(d, 1)[:, :k]
    a = rng.integers(0, len(x), n)
    b = nn_idx[a, rng.integers(0, k, n)]
    return x[a] + rng.random((n, 1)) * (x[b] - x[a])


def gan(x, y, n, seed, dims=64, steps=3000, device="cuda"):
    """Train on (x, y), return n synthetic rows of class 1. x: principal-component scores."""
    torch.manual_seed(seed)
    scale = x[:, :dims].std(0)
    real = torch.tensor(x[:, :dims] / scale, dtype=torch.float32, device=device)
    lab = torch.tensor(y, dtype=torch.float32, device=device)[:, None]
    z_dim = 32
    g = nn.Sequential(nn.Linear(z_dim + 1, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, dims)).to(device)
    c = nn.Sequential(nn.Linear(dims + 1, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1)).to(device)
    og = torch.optim.Adam(g.parameters(), 1e-4, betas=(0.5, 0.9))
    oc = torch.optim.Adam(c.parameters(), 1e-4, betas=(0.5, 0.9))
    w = torch.where(lab[:, 0] > 0, 1 / lab.mean(), 1 / (1 - lab.mean()))        # both classes equally often
    for step in range(steps):
        for _ in range(5):
            i = torch.multinomial(w, 64, replacement=True)
            fake = g(torch.cat([torch.randn(64, z_dim, device=device), lab[i]], 1)).detach()
            eps = torch.rand(64, 1, device=device)
            mix = (eps * real[i] + (1 - eps) * fake).requires_grad_(True)
            grad = torch.autograd.grad(c(torch.cat([mix, lab[i]], 1)).sum(), mix, create_graph=True)[0]
            loss = c(torch.cat([fake, lab[i]], 1)).mean() - c(torch.cat([real[i], lab[i]], 1)).mean() \
                + 10 * ((grad.norm(dim=1) - 1) ** 2).mean()
            oc.zero_grad(); loss.backward(); oc.step()
        i = torch.multinomial(w, 64, replacement=True)
        loss = -c(torch.cat([g(torch.cat([torch.randn(64, z_dim, device=device), lab[i]], 1)), lab[i]], 1)).mean()
        og.zero_grad(); loss.backward(); og.step()
    with torch.no_grad():
        out = g(torch.cat([torch.randn(n, z_dim, device=device), torch.ones(n, 1, device=device)], 1)).cpu().numpy() * scale
    full = np.zeros((n, x.shape[1]))
    full[:, :dims] = out
    return full, real[lab[:, 0] > 0].cpu().numpy() * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="schaefer100s_gsr")
    ap.add_argument("--split", default="pooled_cv")
    ap.add_argument("--label", default="psychosis")
    ap.add_argument("--diagnoses", nargs="*", default=["HC", "SZ"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--name", default="augment_szhc")
    args = ap.parse_args()
    cfg = load_config()
    d = load(cfg, args.features, args)
    pos = pd.Series(np.arange(len(d["y"])), index=d["index"]["subject"])
    folds = json.loads((cfg["paths"]["splits"] / f"{args.split}.json").read_text())["folds"]
    rows = []
    for k, parts in enumerate(folds):
        tr = pos.reindex(parts["train"] + parts["val"]).dropna().astype(int).to_numpy()
        te = pos.reindex(parts["test"]).dropna().astype(int).to_numpy()
        xtr, xte = design(d, "tangent", False, tr, te)
        ytr, yte = d["y"][tr], d["y"][te]
        # Rotation to principal components: lossless for an L2-penalised linear model, and all synthetic rows live in it.
        u, s, vt = np.linalg.svd(xtr, full_matrices=False)
        ptr, pte = u * s, xte @ vt.T
        need = int((ytr == 0).sum() - (ytr == 1).sum())
        rng = np.random.default_rng(k)
        fake, real = gan(ptr, ytr, need, k)
        sets = {"none": (ptr, ytr, None), "weights": (ptr, ytr, "balanced"),
                "smote": (np.vstack([ptr, smote(ptr[ytr == 1], need, rng)]), np.r_[ytr, np.ones(need, int)], None),
                "gan": (np.vstack([ptr, fake]), np.r_[ytr, np.ones(need, int)], None)}
        for name, (x, y, cw) in sets.items():
            p = LogisticRegression(C=args.C, max_iter=5000, class_weight=cw).fit(x, y).predict_proba(pte)[:, 1]
            rows.append(dict(fold=k, method=name, n_train=len(y), n_synthetic=len(y) - len(ytr), auc=roc_auc_score(yte, p),
                             bacc=balanced_accuracy_score(yte, p > 0.5), sensitivity=float((p[yte == 1] > 0.5).mean()),
                             specificity=float((p[yte == 0] <= 0.5).mean())))
        # How close are the synthetic patients to real ones? (first 64 components)
        f = fake[:, :real.shape[1]]
        print(f"fold {k}: train HC {int((ytr == 0).sum())} SZ {int((ytr == 1).sum())}, synthetic SZ {need};"
              f" per-component sd ratio fake/real {np.median(f.std(0) / real.std(0)):.2f},"
              f" mean shift {np.abs(f.mean(0) - real.mean(0)).mean() / real.std(0).mean():.2f} sd", flush=True)
    res = pd.DataFrame(rows)
    out = cfg["paths"]["data"] / "runs" / f"{args.name}.csv"
    res.to_csv(out, index=False)
    print(res.groupby("method", sort=False)[["auc", "bacc", "sensitivity", "specificity"]].agg(["mean", "std"]).round(3).to_string())
    print("results ->", out)


if __name__ == "__main__":
    main()
