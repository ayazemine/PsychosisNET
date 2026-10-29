"""Comparison models on the same folds as PsychosisNET.

    python -m psikoz.baselines --model svm --name cmp_svm
    python -m psikoz.baselines --model brainnetcnn --lr 1e-3 --name cmp_brainnetcnn_lr1e-3

Models
  svm          linear SVM on tangent-space connectivity (Varoquaux et al. 2010; Dadi et al. 2019)
  xgboost      gradient-boosted trees on tangent-space connectivity (Chen & Guestrin 2016)
  mlp          fully connected network on the vectorised correlation matrix (cf. Zeng et al. 2018)
  brainnetcnn  edge-to-edge / edge-to-node / node-to-graph CNN (Kawahara et al. 2017)
  bnt          Brain Network Transformer with orthonormal clustering readout (Kan et al. 2022)

Classical models use the nilearn tangent embedding (Ledoit-Wolf covariance, reference = training part
of the fold, features standardised on the training part); their hyperparameter (SVM C, tree depth) is
chosen per fold on the validation part. Deep models take the correlation matrix (unit diagonal) and
are trained with the PsychosisNET recipe: class-weighted cross-entropy with label smoothing 0.1 and
mixup (alpha 0.4), AdamW (weight decay 0.05), one-cycle schedule, batch 32, at most 60 epochs, early
stopping on validation AUC (patience 15), 5 seeds per fold. Their learning rate is a command-line
setting; run_compare.sh tries two and keeps the one with the higher mean validation AUC.

Outputs are those of psikoz.train (folds.csv, test_ / val_predictions.csv, summary.json, and for the
deep models history.csv and fold<k>_seed<s>.pt), so the same reporting tools apply. Classical models
are saved as fold<k>_seed<s>.joblib (fitted tangent reference and scale, feature scaler, classifier and
the chosen setting), so that psikoz.evaluate can rescore every model without refitting anything.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from .config import load_config
from .harmonize import correlation_loaders, tangent_loaders
from .torch_data import make_loaders
from .train import cross_entropy, predict, scores


# ----------------------------------------------------------------------------- deep models
class MLP(nn.Module):
    def __init__(self, n_roi, n_classes=2, hidden=(256, 64), drop=0.5):
        super().__init__()
        self.register_buffer("triu", torch.triu_indices(n_roi, n_roi, 1), persistent=False)
        layers, d = [], self.triu.shape[1]
        for h in hidden:
            layers += [nn.Dropout(drop), nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers, nn.Dropout(drop), nn.Linear(d, n_classes))

    def forward(self, c):
        return self.net(c[:, self.triu[0], self.triu[1]])


class E2E(nn.Module):
    """Edge-to-edge filter: cross-shaped (row + column) convolution (Kawahara et al. 2017)."""

    def __init__(self, cin, cout, n_roi):
        super().__init__()
        self.row = nn.Conv2d(cin, cout, (1, n_roi))
        self.col = nn.Conv2d(cin, cout, (n_roi, 1))

    def forward(self, x):
        return self.row(x) + self.col(x)          # (B, C, R, 1) + (B, C, 1, R) -> (B, C, R, R)


class BrainNetCNN(nn.Module):
    """E2E(32) -> E2E(32) -> E2N(64) -> N2G(256) -> FC 128 -> FC 30 -> FC 2, leaky ReLU 0.33, dropout 0.5."""

    def __init__(self, n_roi, n_classes=2, drop=0.5):
        super().__init__()
        self.e2e1, self.e2e2 = E2E(1, 32, n_roi), E2E(32, 32, n_roi)
        self.e2n = nn.Conv2d(32, 64, (1, n_roi))
        self.n2g = nn.Conv2d(64, 256, (n_roi, 1))
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(256, 128), nn.LeakyReLU(0.33),
                                nn.Dropout(drop), nn.Linear(128, 30), nn.LeakyReLU(0.33), nn.Linear(30, n_classes))

    def forward(self, c):
        act = lambda x: F.leaky_relu(x, 0.33)
        x = act(self.e2e1(c[:, None]))
        x = act(self.e2e2(x))
        x = act(self.e2n(x))
        x = act(self.n2g(x))
        return self.fc(x.flatten(1))


class BNT(nn.Module):
    """Brain Network Transformer: node features = rows of the connectivity matrix, 2 transformer
    encoder layers (4 heads, feed-forward 1024), orthonormal clustering readout with 10 clusters,
    per-cluster dimension reduction to 8, MLP 80 -> 256 -> 32 -> 2 (Kan et al. 2022)."""

    def __init__(self, n_roi, n_classes=2, layers=2, heads=4, ffn=1024, clusters=10, reduce=8, drop=0.1):
        super().__init__()
        d = n_roi + (-n_roi) % heads                 # model width divisible by the number of heads
        self.pad = d - n_roi
        enc = nn.TransformerEncoderLayer(d, heads, ffn, drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.centres = nn.Parameter(torch.linalg.qr(torch.randn(d, clusters))[0].T.contiguous())
        self.reduce = nn.Sequential(nn.Linear(d, reduce), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(clusters * reduce, 256), nn.LeakyReLU(), nn.Linear(256, 32),
                                nn.LeakyReLU(), nn.Linear(32, n_classes))

    def forward(self, c):
        x = F.pad(c, (0, self.pad))
        x = self.encoder(x)                                            # (B, R, d)
        p = torch.softmax(x @ self.centres.T, dim=-1)                  # soft assignment to clusters
        z = p.transpose(1, 2) @ x                                      # (B, K, d)
        return self.fc(self.reduce(z).flatten(1))


DEEP = {"mlp": MLP, "brainnetcnn": BrainNetCNN, "bnt": BNT}


def train_deep(loaders, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = loaders["train"].dataset
    n_roi, n_classes = train[0]["x"].shape[-1], int(train.y.max()) + 1
    model = DEEP[args.model](n_roi, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loaders["train"]),
                                                pct_start=0.1)
    weight = train.class_weights().to(device)
    train_eval = DataLoader(train, batch_size=64, shuffle=False)
    history, best, best_state, wait = [], -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        objective = []
        for b in loaders["train"]:
            x, y = b["x"].to(device), b["y"].to(device)
            lam = torch.distributions.Beta(args.mixup, args.mixup).sample().item() if args.mixup > 0 else 1.0
            perm = torch.randperm(len(x), device=device)
            x = lam * x + (1 - lam) * x[perm]
            y = lam * F.one_hot(y, n_classes).float() + (1 - lam) * F.one_hot(y[perm], n_classes).float()
            y = y * (1 - args.label_smoothing) + args.label_smoothing / n_classes
            loss = -(weight * y * F.log_softmax(model(x), -1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            objective.append(loss.item())
        pv, yv = predict(model, loaders["val"], device)[:2]
        val = scores(pv, yv)["auc"]
        pt, yt = predict(model, train_eval, device)[:2]
        history.append(dict(epoch=epoch + 1, objective=float(np.mean(objective)), train_ce=cross_entropy(pt, yt),
                            train_auc=scores(pt, yt)["auc"], val_ce=cross_entropy(pv, yv), val_auc=val,
                            lr=sched.get_last_lr()[0]))
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    best_epoch = epoch + 1 - wait
    for h in history:
        h["selected"] = int(h["epoch"] == best_epoch)
    return model, best_epoch, history


# ----------------------------------------------------------------------------- classical models
def vectors(ds):
    iu = np.triu_indices(ds.items[0][0].shape[-1], 1)
    return np.stack([it[0].numpy()[iu] for it in ds.items]), ds.y.numpy(), ds.index["subject"].tolist(), ds.site.numpy()


def classical(loaders, args, seed):
    """Fit on the training part for every candidate setting, keep the best on the validation part.
    Returns {part: (prob (N, 2), y, subjects, sites)}, the chosen setting and the fitted scaler and model."""
    from xgboost import XGBClassifier

    data = {part: vectors(loaders[part].dataset) for part in ("train", "val", "test")}
    scaler = StandardScaler().fit(data["train"][0])
    x = {part: scaler.transform(d[0]) for part, d in data.items()}
    ytr = data["train"][1]
    if args.model == "svm":
        grid = [dict(C=c) for c in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
        make = lambda g: LinearSVC(C=g["C"], class_weight="balanced", max_iter=20000, random_state=seed)
    else:
        grid = [dict(max_depth=d) for d in (2, 3, 4)]
        ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        make = lambda g: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=g["max_depth"], subsample=0.8,
                                       colsample_bytree=0.3, scale_pos_weight=ratio, random_state=seed, n_jobs=args.threads,
                                       eval_metric="logloss")
    best = None
    for g in grid:
        m = make(g).fit(x["train"], ytr)
        auc = scores(np.c_[1 - prob(m, x["val"]), prob(m, x["val"])], data["val"][1])["auc"]
        if best is None or auc > best[0]:
            best = (auc, g, m)
    _, g, m = best
    out = {}
    for part, d in data.items():
        p = prob(m, x[part])
        out[part] = (np.c_[1 - p, p], d[1], d[2], d[3])
    return out, g, scaler, m


def prob(m, x):
    """Psychosis score of a classical model: SVM margin through a sigmoid (monotone; ranking and Platt
    calibration only), XGBoost class probability."""
    if isinstance(m, LinearSVC):
        return 1 / (1 + np.exp(-m.decision_function(x)))
    return m.predict_proba(x)[:, 1]


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["svm", "xgboost", *DEEP])
    ap.add_argument("--split", default="pooled_cv")
    ap.add_argument("--atlas", default="schaefer100s_gsr")
    ap.add_argument("--label", default="psychosis")
    ap.add_argument("--diagnoses", nargs="*", default=["HC", "SZ", "BP"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="deep models and XGBoost: models per fold, probabilities averaged")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = len(json.loads((cfg["paths"]["splits"] / f"{args.split}.json").read_text())["folds"])
    out = cfg["paths"]["data"] / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    deep = args.model in DEEP
    n_seeds = 1 if args.model == "svm" else args.seeds

    rows, preds, vals, hist = [], [], [], []
    for fold in range(n_folds):
        loaders = make_loaders(split=args.split, fold=fold, atlas=args.atlas, label=args.label, input="fc",
                               diagnoses=args.diagnoses, batch_size=args.batch_size, cfg=cfg)
        tangent = None if deep else tangent_loaders(loaders)
        if deep:
            correlation_loaders(loaders)
        runs, extra = [], {}
        for s in range(n_seeds):
            seed = args.seed + fold + 1000 * s
            if deep:
                model, best_epoch, h = train_deep(loaders, args, seed, device)
                hist += [dict(fold=fold, seed=s, **r) for r in h]
                torch.save(model.state_dict(), out / f"fold{fold}_seed{s}.pt")
                runs.append({part: predict(model, loaders[part], device) for part in ("train", "val", "test")})
                extra.setdefault("best_epoch", []).append(best_epoch)
            else:
                r, g, scaler, clf = classical(loaders, args, seed)
                joblib.dump(dict(tangent=tangent[0], scale=tangent[1], scaler=scaler, model=clf, setting=g),
                            out / f"fold{fold}_seed{s}.joblib")
                runs.append(r)
                extra.setdefault("setting", []).append(json.dumps(g))
        fit = {}
        for part in runs[0]:
            order = [{s: i for i, s in enumerate(r[part][2])} for r in runs]
            fit[part] = (np.mean([r[part][0][[o[s] for s in runs[0][part][2]]] for r, o in zip(runs, order)], 0),) \
                + tuple(runs[0][part][1:])
        row = dict(fold=fold, n_train=len(fit["train"][1]), n_test=len(fit["test"][1]),
                   **{k: (float(np.mean(v)) if k == "best_epoch" else v[0]) for k, v in extra.items()})
        for part, (p, y, _, _) in fit.items():
            row.update({f"{part}_{k}": v for k, v in scores(p, y).items()})
        rows.append(row)
        for part, store in (("test", preds), ("val", vals)):
            p, y, sub, site = fit[part]
            store.append(pd.DataFrame(dict(fold=fold, subject=sub, site=site, y=y, **{f"p{c}": p[:, c] for c in range(p.shape[1])})))
        if hist:
            pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "folds.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    pd.concat(vals, ignore_index=True).to_csv(out / "val_predictions.csv", index=False)
    summary = {c: [float(res[c].mean()), float(res[c].std())] for c in res.columns if c.endswith(("auc", "bacc"))}
    if deep:
        summary["n_parameters"] = sum(p.numel() for p in model.parameters())
    (out / "summary.json").write_text(json.dumps(dict(args=vars(args), summary=summary), indent=2))
    print("\nmean +- sd over folds")
    for k, v in summary.items():
        if isinstance(v, list):
            print(f"  {k:14s} {v[0]:.3f} +- {v[1]:.3f}")
    print("  results ->", out)


if __name__ == "__main__":
    main()
