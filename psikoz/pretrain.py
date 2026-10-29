"""Pretraining of the PsychosisNET encoder on unlabeled runs.

    python -m psikoz.pretrain --atlas schaefer100s_gsr

Runs  every quality-controlled resting run of the feature index that can never enter the
      psychosis task: participants with role "pretrain" (SRPBS sites / diagnoses outside the
      classification sample) and participants without a psychosis label (ADHD, bipolar without
      psychosis information), plus data/features/<atlas>/pretrain_index.csv (ABIDE) when present.

Objective
  masked connectivity modelling  a random subset of regions is hidden (rows and columns of the
                                 window connectivity zeroed, token replaced by a mask token) and
                                 the network reconstructs their connectivity profiles
  auxiliary demographics         age (regression) and sex (classification) from the pooled token
  regularisation                 AdamW decoupled weight decay

No diagnosis label is used and no participant of the psychosis task (train, validation or test
part of any fold) is seen, so one encoder serves all folds without leakage.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .model import PsychosisNET, window_connectivity


class WindowDataset(Dataset):
    """Random fixed-length windows of ROI time series (padded and masked when the run is short)."""

    def __init__(self, index, cfg, window, train):
        self.window, self.train = window, train
        self.items = []
        for f in index["file"]:
            z = np.load(cfg["root"] / f)
            self.items.append((torch.from_numpy(z["timeseries"]), torch.from_numpy(z["frame_mask"])))
        age = np.log(index["age"].to_numpy(dtype=np.float32))
        self.age = torch.tensor((age - 2.8) / 0.5)                       # roughly standardised log-years
        self.sex = torch.tensor((index["sex"] == "M").to_numpy(dtype=np.int64))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ts, mask = self.items[i]
        t, w = len(ts), self.window
        if t >= w:
            s = int(torch.randint(0, t - w + 1, ())) if self.train else (t - w) // 2
            ts, mask = ts[s:s + w], mask[s:s + w]
        else:
            ts = torch.cat([ts, ts.new_zeros(w - t, ts.shape[1])])
            mask = torch.cat([mask, mask.new_zeros(w - t)])
        return ts, mask, self.age[i], self.sex[i]


def correlation(ts, mask):
    """Correlation matrix with unit diagonal of a window (input of the SPD layer)."""
    r = torch.tanh(window_connectivity(ts, mask))
    return r + torch.eye(r.shape[-1], device=r.device)


def step(model, heads, batch, mask_ratio, aux_weight, device):
    ts, mask, age, sex = (b.to(device) for b in batch)
    with torch.no_grad():               # same input representation as in fine-tuning; the SPD layer stays frozen here
        fc = model.spd(correlation(ts, mask))
    hidden = torch.rand(fc.shape[:2], device=device) < mask_ratio
    tokens, _ = model.encode(fc, hidden=hidden)
    recon = F.mse_loss(heads["decoder"](tokens)[hidden], fc[hidden])
    pooled, _ = model.pool(tokens)
    demo = heads["demo"](pooled)
    aux = F.mse_loss(demo[:, 0], age) + F.cross_entropy(demo[:, 1:], sex)
    return recon + aux_weight * aux, recon.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", default="schaefer100")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--max-fd", type=float, default=0.3, help="drop runs with higher mean framewise displacement")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = cfg["paths"]["features"] / args.atlas
    index = pd.read_csv(feat / "index.csv")
    role = index["role"].fillna("main") if "role" in index.columns else "main"
    index = index[(index["task"] == "rest") & (index["qc_pass"] == 1) & ((role == "pretrain") | index["psychosis"].isna())
                  & ~index["dataset"].isin(cfg.get("external", []))]     # external test sites stay unseen
    if (feat / "pretrain_index.csv").exists():
        index = pd.concat([index, pd.read_csv(feat / "pretrain_index.csv")], ignore_index=True)
    index = index[(index["fd_mean"] <= args.max_fd) & index["age"].notna()].reset_index(drop=True)
    print(f"pretraining runs: {len(index)}", index["dataset"].value_counts().to_dict(), flush=True)
    val_idx = index.sample(frac=0.1, random_state=args.seed).index
    loaders = {
        "train": DataLoader(WindowDataset(index.drop(val_idx), cfg, args.window, True), args.batch_size, shuffle=True, drop_last=True),
        "val": DataLoader(WindowDataset(index.loc[val_idx], cfg, args.window, False), args.batch_size)}
    networks = pd.factorize(pd.read_csv(feat / "labels.csv")["network"])[0]
    n_roi = len(networks)

    # Pretraining uses masking as its regulariser; dropout stays lower than in fine-tuning.
    model = PsychosisNET(n_roi, 2, networks, dim=args.dim, depth=args.depth, head_hidden=0, drop=0.1, attn_drop=0.1,
                         drop_path=0.1, edge_drop=0.0, token_drop=0.0).to(device)
    train = loaders["train"].dataset
    model.spd.init_reference(torch.cat([correlation(ts[None].to(device), m[None].to(device)) for ts, m in train.items]))
    model.spd.requires_grad_(False)           # a learnable target could be made trivially predictable
    heads = nn.ModuleDict(dict(decoder=nn.Linear(args.dim, n_roi), demo=nn.Linear(args.dim, 3))).to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loaders["train"]), pct_start=0.05)

    out = cfg["paths"]["data"] / "runs" / (args.name or f"pretrain_{args.atlas}")
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train(); heads.train()
        tr = []
        for batch in loaders["train"]:
            loss, recon = step(model, heads, batch, args.mask_ratio, args.aux_weight, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); tr.append(recon)
        model.eval(); heads.eval()
        with torch.no_grad():
            torch.manual_seed(1234)                                       # same hidden regions every epoch
            va = [step(model, heads, b, args.mask_ratio, args.aux_weight, device)[1] for b in loaders["val"]]
            torch.manual_seed(args.seed + epoch + 1)
        history.append(dict(epoch=epoch, train_recon=float(np.mean(tr)), val_recon=float(np.mean(va))))
        if history[-1]["val_recon"] < best:
            best = history[-1]["val_recon"]
            torch.save(model.state_dict(), out / "encoder.pt")
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(history[-1], flush=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "summary.json").write_text(json.dumps(dict(args=vars(args), n_train=len(loaders["train"].dataset),
                                                      n_val=len(loaders["val"].dataset), best_val_recon=best), indent=2))
    print(f"runs: {len(index)}  best val reconstruction MSE: {best:.4f}  -> {out / 'encoder.pt'}")


if __name__ == "__main__":
    main()
