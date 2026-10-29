"""Interpretation of a trained PsychosisNET run: Grad-CAM region maps, Integrated Gradients on the
connections and the attention-pooling weights, computed on the test part of every fold.

    python -m psikoz.interpret --run final_psychosisnet

Grad-CAM (Selvaraju et al. 2017) adapted to region tokens: A (114 x 64) are the region tokens after
the last transformer block and its LayerNorm (the input of the attention pooling), y the psychosis
log-odds (logit_1 - logit_0) of the deep path,
    alpha_k = mean_i dy/dA_ik,     cam_i = ReLU(sum_k alpha_k A_ik).
Each subject's map is divided by its maximum, then averaged over the seeds of the fold. The wide
(linear) path does not pass through the tokens and is covered by Integrated Gradients instead.

Integrated Gradients (Sundararajan et al. 2017) of the full model's log-odds with respect to the input
correlation matrix, baseline = mean correlation matrix of the fold's training part, 32 steps; edge
attribution = A_ij + A_ji. Summarised as the mean signed attribution per network pair.

Sanity check (Adebayo et al. 2018): Grad-CAM of the same subjects after re-initialising every weight;
the rank correlation with the trained map should be low. Fold consistency: rank correlation of the
group maps of the five folds.

Outputs in data/runs/<run>/interpret/: gradcam_subjects.csv, attention_subjects.csv, regions.csv
(group maps per class), network_ig.csv (8 x 8), edge_ig_mean.npy (114 x 114), consistency.csv,
sanity.csv, gradcam_psychosis.nii.gz and gradcam_control.nii.gz (group maps painted on the atlas),
embeddings.csv (pooled 64-d subject vector of the test subjects, averaged over seeds; for t-SNE).
"""
import argparse
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .config import load_config
from .features import load_atlas
from .harmonize import correlation_loaders
from .torch_data import make_loaders
from .train import build_model


def log_odds(model, c, return_tokens=False):
    """Psychosis log-odds; optionally the deep-path log-odds and the region tokens (grad retained)."""
    fc = model.spd(c)
    tokens, _ = model.encode(fc)
    pooled, weights = model.pool(tokens)
    deep = model.head(pooled)
    full = deep + model.skip(fc[:, model.triu[0], model.triu[1]])
    if return_tokens:
        return full[:, 1] - full[:, 0], deep[:, 1] - deep[:, 0], tokens, weights
    return full[:, 1] - full[:, 0]


def gradcam(model, c):
    model.zero_grad()
    _, deep, tokens, weights = log_odds(model, c, return_tokens=True)
    grad, = torch.autograd.grad(deep.sum(), tokens)
    alpha = grad.mean(1, keepdim=True)                                 # (B, 1, D)
    cam = torch.relu((alpha * tokens).sum(-1))                         # (B, R)
    cam = cam / cam.amax(1, keepdim=True).clamp(min=1e-12)
    return cam.detach().cpu().numpy(), weights.detach().cpu().numpy()


def integrated_gradients(model, c, baseline, steps=32):
    total = torch.zeros_like(c)
    for a in (torch.arange(steps, device=c.device) + 0.5) / steps:
        x = (baseline + a * (c - baseline)).requires_grad_(True)
        g, = torch.autograd.grad(log_odds(model, x).sum(), x)
        total += g
    attr = (c - baseline) * total / steps
    return (attr + attr.transpose(1, 2)).detach().cpu().numpy()


def reinitialise(model):
    for m in model.modules():
        if hasattr(m, "reset_parameters") and m is not model:
            m.reset_parameters()
    for name, p in model.named_parameters():                           # bare parameters (embeddings, gains, ...)
        if not any(name.endswith(s) for s in (".weight", ".bias")):
            torch.nn.init.normal_(p, std=0.02)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ig-steps", type=int, default=32)
    cli = ap.parse_args()
    cfg = load_config()
    run = cfg["paths"]["data"] / "runs" / cli.run
    a = SimpleNamespace(**json.loads((run / "summary.json").read_text())["args"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = pd.read_csv(cfg["paths"]["features"] / a.atlas / "labels.csv")
    networks = pd.factorize(labels["network"])[0]
    out = run / "interpret"
    out.mkdir(exist_ok=True)
    n_folds = len(json.loads((cfg["paths"]["splits"] / f"{a.split}.json").read_text())["folds"])

    cams, atts, igs, sanity, embs = [], [], [], [], []
    for fold in range(n_folds):
        loaders = make_loaders(split=a.split, fold=fold, atlas=a.atlas, label=a.label, input="fc",
                               diagnoses=a.diagnoses, datasets=getattr(a, "datasets", None), cfg=cfg)
        correlation_loaders(loaders)
        test, train = loaders["test"].dataset, loaders["train"].dataset
        c = torch.stack([it[0] for it in test.items]).to(device)
        baseline = torch.stack([it[0] for it in train.items]).mean(0, keepdim=True).to(device)
        seeds = sorted(run.glob(f"fold{fold}_seed*.pt")) or [run / f"fold{fold}.pt"]
        cam_f, att_f, ig_f, p_f, z_f = [], [], [], [], []
        for f in seeds:
            model = build_model(a, c.shape[-1], 2, networks).to(device)
            model.load_state_dict(torch.load(f, map_location=device))
            model.eval()
            cam, att = gradcam(model, c)
            cam_f.append(cam); att_f.append(att)
            with torch.no_grad():
                p_f.append(torch.sigmoid(log_odds(model, c)).cpu().numpy())
                z_f.append(model(c, return_pooled=True)[1].cpu().numpy())
            ig_f.append(np.concatenate([integrated_gradients(model, c[i:i + 16], baseline, cli.ig_steps)
                                        for i in range(0, len(c), 16)]))
        rand = reinitialise(build_model(a, c.shape[-1], 2, networks).to(device))
        rand.spd.init_reference(torch.stack([it[0] for it in train.items]).to(device))
        rand.eval()
        rcam, _ = gradcam(rand, c)
        cam, att, ig, prob = (np.mean(v, 0) for v in (cam_f, att_f, ig_f, p_f))
        sanity.append(dict(fold=fold, spearman_trained_vs_random=float(spearmanr(cam.mean(0), rcam.mean(0))[0])))
        meta = dict(fold=fold, subject=test.index["subject"].tolist(), site=test.site.numpy(), y=test.y.numpy(), p=prob)
        cams.append(pd.DataFrame(cam, columns=labels["name"]).assign(**meta))
        atts.append(pd.DataFrame(att, columns=labels["name"]).assign(**meta))
        embs.append(pd.DataFrame(np.mean(z_f, 0), columns=[f"z{k}" for k in range(np.shape(z_f)[-1])]).assign(**meta))
        igs.append((ig, test.y.numpy()))
        print(f"fold {fold}: {len(c)} test subjects, {len(seeds)} seed(s)", flush=True)

    cam = pd.concat(cams, ignore_index=True); att = pd.concat(atts, ignore_index=True)
    cam.to_csv(out / "gradcam_subjects.csv", index=False)
    att.to_csv(out / "attention_subjects.csv", index=False)
    pd.concat(embs, ignore_index=True).to_csv(out / "embeddings.csv", index=False)
    names = labels["name"].tolist()
    regions = labels.copy()
    regions["gradcam_psychosis"] = cam.loc[cam.y == 1, names].mean().values
    regions["gradcam_control"] = cam.loc[cam.y == 0, names].mean().values
    regions["gradcam_correct_psychosis"] = cam.loc[(cam.y == 1) & (cam.p > 0.5), names].mean().values
    regions["attention_mean"] = att[names].mean().values
    regions.to_csv(out / "regions.csv", index=False)

    ig_all = np.concatenate([g for g, _ in igs]); y_all = np.concatenate([y for _, y in igs])
    edge = ig_all[y_all == 1].mean(0)
    np.save(out / "edge_ig_mean.npy", edge)
    nets = labels["network"].tolist(); order = list(dict.fromkeys(nets))
    idx = {n: [i for i, m in enumerate(nets) if m == n] for n in order}
    net = pd.DataFrame([[edge[np.ix_(idx[r], idx[s])].mean() for s in order] for r in order], index=order, columns=order)
    net.to_csv(out / "network_ig.csv")

    group = [cam[cam.fold == f].query("y == 1")[names].mean().values for f in sorted(cam.fold.unique())]
    pairs = [dict(fold_a=i, fold_b=j, spearman=float(spearmanr(group[i], group[j])[0]))
             for i in range(len(group)) for j in range(i + 1, len(group))]
    pd.DataFrame(pairs).to_csv(out / "consistency.csv", index=False)
    pd.DataFrame(sanity).to_csv(out / "sanity.csv", index=False)

    img, _ = load_atlas(a.atlas.replace("_gsr", ""))
    lab = np.asarray(img.dataobj).astype(int)
    for col, fname in (("gradcam_psychosis", "gradcam_psychosis"), ("gradcam_control", "gradcam_control")):
        vol = np.zeros(lab.shape, np.float32)
        vol[lab > 0] = regions[col].values[lab[lab > 0] - 1]
        nib.save(nib.Nifti1Image(vol, img.affine), out / f"{fname}.nii.gz")
    print("fold consistency (Spearman, psychosis maps): mean %.2f" % np.mean([p["spearman"] for p in pairs]))
    print("sanity (trained vs re-initialised): mean %.2f" % np.mean([s["spearman_trained_vs_random"] for s in sanity]))
    print("top regions (psychosis):", regions.nlargest(10, "gradcam_psychosis")["name"].tolist())
    print("->", out)


if __name__ == "__main__":
    main()
