"""Cross-validated training of PsychosisNET.

    python -m psikoz.train --atlas schaefer100s_gsr --pretrained data/runs/pretrain_schaefer100s_gsr/encoder.pt --name v13

Model selection uses the validation part only (early stopping on validation AUC);
the test part of each fold is scored once with the selected weights.

Loss  class-weighted cross-entropy (label smoothing, mixup) + aux * (age MSE + sex BCE), --aux
Optimiser  AdamW: L2 regularisation as decoupled weight decay (Loshchilov & Hutter 2019), --weight-decay

Outputs (data/runs/<name>/): folds.csv, test_ / val_predictions.csv, summary.json, history.csv (per fold,
seed and epoch: training objective, eval-mode train and validation cross-entropy and AUC, learning rate),
fold<k>_seed<s>.pt (every seed's selected weights) and fold<k>.pt (last seed, kept for older tools).
The extra per-epoch evaluation of the training part restores the random number generator state, so it
does not change training.

Options against overfitting
  --depth / --dim / --head-hidden   smaller network (needs an encoder pretrained with the same size)
  --freeze-spd                      SPD layer keeps the classical tangent map (D = 0), only the shrinkage is learned
  --ema d                           exponential moving average of the weights (decay d per step); the
                                    averaged weights are validated, early-stopped and tested
  --crop f                          every training step recomputes each subject's correlation matrix from a
                                    random contiguous segment covering a fraction U(f, 1) of the run (clean frames)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .config import load_config
from .harmonize import correlation_loaders
from .model import PsychosisNET
from .torch_data import make_loaders


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    p, y, sub, site = [], [], [], []
    for b in loader:
        p.append(model(b["x"].to(device)).softmax(-1).cpu())
        y.append(b["y"]); site.append(b["site"]); sub += b["subject"]
    return torch.cat(p).numpy(), torch.cat(y).numpy(), sub, torch.cat(site).numpy()


def scores(prob, y):
    if len(np.unique(y)) < 2:   # a held-out site with one class only (ds005073: patients only)
        return dict(auc=float("nan"), bacc=float((prob.argmax(1) == y).mean()))
    if prob.shape[1] == 2:
        auc = roc_auc_score(y, prob[:, 1])
    else:
        auc = roc_auc_score(y, prob, multi_class="ovr", labels=range(prob.shape[1]))
    return dict(auc=float(auc), bacc=float(balanced_accuracy_score(y, prob.argmax(1))))


def load_timeseries(ds, cfg):
    """ROI time series and clean-frame masks of every item of a dataset (for --crop)."""
    out = []
    for f in ds.index["file"]:
        z = np.load(cfg["root"] / f)
        out.append((torch.from_numpy(z["timeseries"]), torch.from_numpy(z["frame_mask"])))
    return out


def crop_correlation(series, idx, frac, device):
    """Correlation matrices (unit diagonal) of random contiguous segments, as in features.connectivity:
    Pearson over the clean frames of the segment; regions without signal get zero correlations."""
    out = []
    for i in idx:
        ts, good = series[i]
        t = len(ts)
        w = max(int(round(t * float(torch.empty(()).uniform_(frac, 1.0)))), 3)
        s = int(torch.randint(0, t - w + 1, ()))
        x = ts[s:s + w][good[s:s + w]].to(device, torch.float64)
        if len(x) < 3:                                                     # too few clean frames: whole run
            x = ts[good].to(device, torch.float64)
        x = x - x.mean(0)
        sd = x.pow(2).sum(0).sqrt()
        r = (x.T @ x) / (sd[:, None] * sd[None, :]).clamp(min=1e-12)
        r = r.clamp(-0.999999, 0.999999)
        r[(sd < 1e-8)] = 0
        r[:, (sd < 1e-8)] = 0
        out.append(r.fill_diagonal_(1.0).float())
    return torch.stack(out)


def build_model(args, n_roi, n_classes, networks):
    return PsychosisNET(n_roi, n_classes, networks, dim=args.dim, depth=args.depth, drop=args.drop,
                        drop_path=args.drop_path, edge_drop=args.edge_drop, head_hidden=args.head_hidden,
                        skip_drop=args.skip_drop, aux=args.aux > 0)


def train_fold(loaders, networks, args, seed, device, series=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = loaders["train"].dataset
    n_roi, n_classes = train[0]["x"].shape[-1], int(train.y.max()) + 1
    model = build_model(args, n_roi, n_classes, networks).to(device)
    model.spd.init_reference(torch.stack([it[0] for it in train.items]).to(device))
    if args.pretrained:   # encoder weights from psikoz.pretrain; the classification head starts fresh
        # The SPD reference belongs to the training fold.
        state = {k: v for k, v in torch.load(args.pretrained, map_location=device).items()
                 if not k.startswith(("head.", "spd.", "skip.", "aux_head."))}
        missing = model.load_state_dict(state, strict=False).missing_keys
        assert all(k.startswith(("head.", "skip.", "spd.", "aux_head.")) for k in missing), missing
    if args.freeze_spd:
        model.spd.delta.requires_grad_(False)
    # --ema: the averaged copy is what gets validated, early-stopped and returned.
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    scored = ema.module if ema is not None else model
    # The wide path gets its own learning rate and joins later (--skip-delay), so that it does not
    # fit the training set before the encoder has learned anything.
    wide = [p for n, p in model.named_parameters() if n.startswith("skip.")]
    deep = [p for n, p in model.named_parameters() if not n.startswith("skip.") and p.requires_grad]
    groups = [dict(params=deep, lr=args.lr), dict(params=wide, lr=args.lr * args.skip_lr_scale)]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = args.epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    weight = train.class_weights().to(device)
    age_mean, age_sd = float(train.age.nanmean()), float(train.age[~train.age.isnan()].std())

    # Unshuffled copy of the training part for the per-epoch learning curves.
    train_eval = DataLoader(train, batch_size=64, shuffle=False)
    history = []
    best, best_state, wait = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        objective = []
        for p in wide:
            p.requires_grad_(epoch >= args.skip_delay)
        for b in loaders["train"]:
            x, y = b["x"].to(device), b["y"].to(device)
            if args.crop:
                x = crop_correlation(series, b["idx"].tolist(), args.crop, device)
            lam = torch.distributions.Beta(args.mixup, args.mixup).sample().item() if args.mixup > 0 else 1.0
            perm = torch.randperm(len(x), device=device)
            x = lam * x + (1 - lam) * x[perm]
            y = lam * F.one_hot(y, n_classes).float() + (1 - lam) * F.one_hot(y[perm], n_classes).float()
            logits, pooled = model(x, return_pooled=True)
            y = y * (1 - args.label_smoothing) + args.label_smoothing / n_classes
            loss = -(weight * y * F.log_softmax(logits, -1)).sum(1).mean()
            if args.aux:
                out = model.aux_head(pooled)
                age = ((b["age"] - age_mean) / age_sd).to(device)
                age = lam * age + (1 - lam) * age[perm]
                ok = ~age.isnan()
                sex = b["sex"].float().to(device)
                sex = lam * sex + (1 - lam) * sex[perm]
                loss = loss + args.aux * (F.mse_loss(out[ok, 0], age[ok]) + F.binary_cross_entropy_with_logits(out[:, 1], sex))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            objective.append(loss.item())
            if ema is not None:
                ema.update_parameters(model)
        pv, yv = predict(scored, loaders["val"], device)[:2]
        val = scores(pv, yv)["auc"]
        rng = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        pt, yt = predict(scored, train_eval, device)[:2]
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        history.append(dict(epoch=epoch + 1, objective=float(np.mean(objective)),
                            train_ce=cross_entropy(pt, yt), train_auc=scores(pt, yt)["auc"],
                            val_ce=cross_entropy(pv, yv), val_auc=val, lr=sched.get_last_lr()[0]))
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in scored.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    scored.load_state_dict(best_state)
    best_epoch = epoch + 1 - wait
    for h in history:
        h["selected"] = int(h["epoch"] == best_epoch)
    return scored, best_epoch, history


def cross_entropy(prob, y):
    """Mean negative log-likelihood of the true class (unweighted, no smoothing)."""
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-7, 1)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="pooled_cv")
    ap.add_argument("--atlas", default="schaefer100s_gsr")
    ap.add_argument("--label", default="psychosis")
    ap.add_argument("--diagnoses", nargs="*", default=["HC", "SZ", "BP"])
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05, help="AdamW decoupled weight decay")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--head-hidden", type=int, default=64, help="hidden units of the MLP head (0 = linear head)")
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--drop-path", type=float, default=0.2)
    ap.add_argument("--edge-drop", type=float, default=0.2)
    ap.add_argument("--skip-drop", type=float, default=0.5)
    ap.add_argument("--skip-lr-scale", type=float, default=0.1, help="learning rate of the wide path relative to --lr")
    ap.add_argument("--skip-delay", type=int, default=10, help="epochs during which the wide path stays frozen at zero")
    ap.add_argument("--aux", type=float, default=0.1, help="weight of the auxiliary age / sex loss (0 = no aux head)")
    ap.add_argument("--freeze-spd", action="store_true", help="keep the SPD congruence at the classical tangent map")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights per step (0 = off)")
    ap.add_argument("--crop", type=float, default=0.0, help="random-segment correlation augmentation, minimum fraction of the run (0 = off)")
    ap.add_argument("--pretrained", default=None, help="encoder checkpoint written by psikoz.pretrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="models per fold; their probabilities are averaged")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (the machine is shared)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = pd.read_csv(cfg["paths"]["features"] / args.atlas / "labels.csv")
    networks = pd.factorize(labels["network"])[0]
    n_folds = len(json.loads((cfg["paths"]["splits"] / f"{args.split}.json").read_text())["folds"])
    out = cfg["paths"]["data"] / "runs" / (args.name or f"{args.split}_{args.label}_{args.atlas}")
    out.mkdir(parents=True, exist_ok=True)

    rows, preds, vals, hist = [], [], [], []
    for fold in range(n_folds):
        loaders = make_loaders(split=args.split, fold=fold, atlas=args.atlas, label=args.label, input="fc",
                               diagnoses=args.diagnoses, datasets=args.datasets, batch_size=args.batch_size, cfg=cfg)
        correlation_loaders(loaders)
        series = load_timeseries(loaders["train"].dataset, cfg) if args.crop else None
        runs = []
        for s in range(args.seeds):
            model, best_epoch, h = train_fold(loaders, networks, args, args.seed + fold + 1000 * s, device, series)
            hist += [dict(fold=fold, seed=s, **r) for r in h]
            torch.save(model.state_dict(), out / f"fold{fold}_seed{s}.pt")
            # Training-set score in eval mode (no dropout, no mixup) to expose the train/test gap.
            runs.append({part: predict(model, loaders[part], device) for part in ("train", "val", "test")})
        # The training loader is shuffled, so every seed returns its own order: align on subject before averaging.
        fit = {}
        for part in runs[0]:
            order = [{s: i for i, s in enumerate(r[part][2])} for r in runs]
            fit[part] = (np.mean([r[part][0][[o[s] for s in runs[0][part][2]]] for r, o in zip(runs, order)], 0),) \
                + tuple(runs[0][part][1:])
        row = dict(fold=fold, best_epoch=best_epoch, n_train=len(fit["train"][1]), n_test=len(fit["test"][1]))
        for part, (p, y, _, _) in fit.items():
            row.update({f"{part}_{k}": v for k, v in scores(p, y).items()})
        rows.append(row)
        p, y, sub, site = fit["test"]
        preds.append(pd.DataFrame(dict(fold=fold, subject=sub, site=site, y=y, **{f"p{c}": p[:, c] for c in range(p.shape[1])})))
        p, y, sub, site = fit["val"]
        vals.append(pd.DataFrame(dict(fold=fold, subject=sub, site=site, y=y, **{f"p{c}": p[:, c] for c in range(p.shape[1])})))
        torch.save(model.state_dict(), out / f"fold{fold}.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

    res, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    res.to_csv(out / "folds.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    pd.concat(vals, ignore_index=True).to_csv(out / "val_predictions.csv", index=False)
    summary = {c: [float(res[c].mean()), float(res[c].std())] for c in res.columns if c.endswith(("auc", "bacc"))}
    summary["n_parameters"] = sum(p.numel() for p in model.parameters())
    (out / "summary.json").write_text(json.dumps(dict(args=vars(args), summary=summary), indent=2))
    print("\nmean +- sd over folds")
    for k, (m, s) in ((k, v) for k, v in summary.items() if isinstance(v, list)):
        print(f"  {k:14s} {m:.3f} +- {s:.3f}")
    print("  parameters    ", summary["n_parameters"], "\n  results ->", out)


if __name__ == "__main__":
    main()
