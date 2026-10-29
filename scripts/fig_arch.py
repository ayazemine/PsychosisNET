"""Figure 2 (pictorial): PsychosisNET layer by layer, drawn with the real tensors of one participant.

Called by scripts/make_figures.py. Weights: fold-0 seed-0 model of the final run when given, otherwise the
pretrained encoder (head and wide path then untrained; only the encoder tensors are meaningful). The SPD
reference is the mean correlation matrix of 60 UCLA CNP participants.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fig_story import YEO  # noqa: E402

LANE = {"in": "#2f5d8a", "spd": "#2a7a66", "tok": "#4f7a28", "att": "#b4621b", "pool": "#4f7a28",
        "head": "#b4621b", "wide": "#4d4d4d", "out": "#2f5d8a"}


def arrow(ax, p, q, color="#555555", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def tensors(final_run=None):
    from psikoz.model import PsychosisNET

    feat = ROOT / "data" / "features" / "schaefer100s_gsr"
    labels = pd.read_csv(feat / "labels.csv")
    networks = pd.factorize(labels["network"])[0]
    index = pd.read_csv(feat / "index.csv", low_memory=False)
    ref = index[(index.dataset == "ds000030") & (index.qc_pass == 1) & (index.task == "rest")].head(60)
    corr = lambda f: np.fill_diagonal(c := np.tanh(np.load(ROOT / f)["fc"]), 1) or c
    ref_c = torch.tensor(np.stack([corr(f) for f in ref.file]), dtype=torch.float32)
    c = torch.tensor(corr("data/features/schaefer100s_gsr/ds000030/sub-10159_task-rest_run-1.npz"), dtype=torch.float32)[None]
    model = PsychosisNET(len(labels), 2, networks, aux=True)
    if final_run and (ROOT / "data" / "runs" / final_run / "fold0_seed0.pt").exists():
        model.load_state_dict(torch.load(ROOT / "data" / "runs" / final_run / "fold0_seed0.pt", map_location="cpu"))
    else:
        state = torch.load(ROOT / "data" / "runs" / "pretrain_schaefer100s_gsr" / "encoder.pt", map_location="cpu")
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith(("head.", "spd.", "skip."))}, strict=False)
        model.spd.init_reference(ref_c)
    model.eval()
    with torch.no_grad():
        t = model.spd(c)
        tokens, attns = model.encode(t)
        pooled, w = model.pool(tokens)
        deep = model.head(pooled); wide = model.skip(t[:, model.triu[0], model.triu[1]])
        prob = torch.softmax(deep + wide, -1)[0, 1].item()
    return dict(c=c[0].numpy(), t=t[0].numpy(), tokens=tokens[0].numpy(), attns=[a[0].mean(0).numpy() for a in attns],
                w=w[0].numpy(), z=pooled[0].numpy(), prob=prob, labels=labels)


def figure2_pictorial(final_run=None):
    d = tensors(final_run)
    labels = d["labels"]
    nets = list(YEO)
    order = np.argsort([nets.index(n) for n in labels.network], kind="stable")
    W, H = 7.2, 3.1
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")

    def panel(x, y, w, h):
        ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#777777")
        return ax

    def title(x, y, text, sub, colour):
        bg.text(x, y, text, ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=colour)
        bg.text(x, y - 0.03, sub, ha="center", va="top", fontsize=5.4, color="#444444", linespacing=1.15)

    def strip(ax, n, horizontal=True):
        bounds = np.cumsum([0] + [int((labels.network == k).sum()) for k in nets])
        for i, k in enumerate(nets):
            if horizontal:
                ax.add_patch(plt.Rectangle((bounds[i] - 0.5, -6), bounds[i + 1] - bounds[i], 4.5, color=YEO[k], clip_on=False, lw=0))
            ax.add_patch(plt.Rectangle((-6, bounds[i] - 0.5), 4.5, bounds[i + 1] - bounds[i], color=YEO[k], clip_on=False, lw=0))

    yt = 1.62                                       # bottom of the main row of tensors
    s = 0.92                                        # tensor panel size (inches)
    # 1 input
    x = 0.12
    ax = panel(x, yt, s, s)
    ax.imshow(d["c"][np.ix_(order, order)], cmap="RdBu_r", vmin=-0.8, vmax=0.8, interpolation="nearest"); strip(ax, 114)
    ax.set_xlim(-6.5, 113.5); ax.set_ylim(113.5, -6.5); ax.axis("off")
    title(x + s / 2, yt + s + 0.32, "Input", "correlation matrix\n" + r"$C$: 114 × 114", LANE["in"])
    # 2 SPD tangent layer
    x2 = 1.38
    ax = panel(x2, yt, s, s)
    tt = d["t"][np.ix_(order, order)]; lim = np.percentile(np.abs(tt), 98)
    ax.imshow(tt, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest"); strip(ax, 114)
    ax.set_xlim(-6.5, 113.5); ax.set_ylim(113.5, -6.5); ax.axis("off")
    title(x2 + s / 2, yt + s + 0.32, "SPD tangent layer", r"$T=\mathrm{logm}(W^{\top}C_sW)$" + "\n114 × 114", LANE["spd"])
    arrow(bg, (x + s + 0.06, yt + s / 2), (x2 - 0.06, yt + s / 2))
    # 3 region tokens (stacked look)
    x3 = 2.64
    for k in (2, 1):
        ax = panel(x3 + 0.05 * k, yt + 0.05 * k, s * 0.62, s); ax.set_facecolor("#e6f0d8")
    ax = panel(x3, yt, s * 0.62, s)
    tk = d["tokens"][order]; lim = np.percentile(np.abs(tk), 98)
    ax.imshow(tk, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    title(x3 + 0.3, yt + s + 0.32, "Region tokens", "row of T → Linear 114→64\n+ region & network emb.", LANE["tok"])
    bg.text(x3 - 0.05, yt + s / 2, "114 regions", rotation=90, ha="right", va="center", fontsize=4.8, color="#555555")
    bg.text(x3 + s * 0.31, yt - 0.06, "64 features", ha="center", va="top", fontsize=4.8, color="#555555")
    arrow(bg, (x2 + s + 0.06, yt + s / 2), (x3 - 0.13, yt + s / 2))
    # 4 four transformer blocks: real attention maps (mean over heads)
    x4 = 3.75
    for b in range(4):
        ax = panel(x4 + 0.12 * b, yt - 0.04 * b + 0.1, s * 0.78, s * 0.78)
        a = d["attns"][b][np.ix_(order, order)]
        ax.imshow(a, cmap="magma", vmin=0, vmax=np.percentile(a, 99.5), interpolation="nearest")
        ax.text(0.03, 0.97, f"B{b + 1}", transform=ax.transAxes, ha="left", va="top", fontsize=4.6, color="white")
    title(x4 + 0.55, yt + s + 0.32, "Transformer blocks × 4", "connectivity-biased attention\n"
          + r"softmax$(QK^{\top}/\sqrt{d}+g\,T+o\,|T|)$", LANE["att"])
    arrow(bg, (x3 + s * 0.62 + 0.16, yt + s / 2), (x4 - 0.06, yt + s / 2))
    # 5 attention pooling weights over regions
    x5 = 5.27
    ax = panel(x5, yt + 0.06, 0.42, s - 0.06)
    w = d["w"][order]
    ax.barh(np.arange(len(w)), w, color=[YEO[labels.network[i]] for i in order], height=1.0)
    ax.set_ylim(len(w) - 0.5, -0.5); ax.set_xlim(0, w.max() * 1.05)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    title(x5 + 0.21, yt + s + 0.32, "Attention pooling", "weights over regions\n(region importance)", LANE["pool"])
    arrow(bg, (x4 + 0.36 + s * 0.78 + 0.08, yt + s / 2), (x5 - 0.06, yt + s / 2))
    # 6 subject vector z
    x6 = 5.9
    ax = panel(x6, yt + 0.06, 0.12, s - 0.06)
    ax.imshow(d["z"][:, None], cmap="PuOr_r", aspect="auto", vmin=-np.abs(d["z"]).max(), vmax=np.abs(d["z"]).max())
    bg.text(x6 + 0.06, yt - 0.02, r"$z$" + "\n64", ha="center", va="top", fontsize=5, color="#444444")
    arrow(bg, (x5 + 0.42 + 0.04, yt + s / 2), (x6 - 0.04, yt + s / 2))
    # 7 classification head (neural net icon)
    x7 = 6.2
    hx = [x7, x7 + 0.26, x7 + 0.52]
    hn = [6, 6, 2]
    pts = []
    for xx, n in zip(hx, hn):
        ys = yt + s / 2 + (np.arange(n) - (n - 1) / 2) * 0.12
        pts.append([(xx, yy) for yy in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                bg.plot([p[0], q[0]], [p[1], q[1]], color="#d9b38c", lw=0.25, zorder=1)
    for layer in pts:
        for p in layer:
            bg.add_patch(Circle(p, 0.035, fc="#fbe8d6", ec=LANE["head"], lw=0.5, zorder=2))
    title(x7 + 0.26, yt + s + 0.32, "Head", "Linear 64→64, GELU,\nLN, Linear 64→2", LANE["head"])
    arrow(bg, (x6 + 0.12 + 0.03, yt + s / 2), (x7 - 0.06, yt + s / 2))

    # wide path (bottom lane)
    yw = 0.42
    ax = panel(x2 + 0.05, yw, s - 0.1, 0.14)
    iu = np.triu_indices(114, 1)
    vec = d["t"][iu]; lim = np.percentile(np.abs(vec), 98)
    ax.imshow(vec[None, :], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    bg.text(x2 + s / 2, yw - 0.06, "upper triangle of T: 6,441 connections", ha="center", va="top", fontsize=5.2, color="#444444")
    arrow(bg, (x2 + s / 2, yt - 0.08), (x2 + s / 2, yw + 0.2), color=LANE["wide"])
    bx = 3.2
    bg.add_patch(FancyBboxPatch((bx, yw - 0.06), 2.25, 0.26, boxstyle="round,pad=0,rounding_size=0.04", fc="#ececec",
                                ec=LANE["wide"], lw=0.6))
    bg.text(bx + 1.125, yw + 0.07, "Wide path: Dropout 0.5 → Linear 6,441 → 2  (zero-initialised)", ha="center",
            va="center", fontsize=5.6, color="#222222")
    arrow(bg, (x2 + s - 0.02, yw + 0.07), (bx - 0.04, yw + 0.07), color=LANE["wide"])
    # sum and output
    sx, sy = 6.46, 0.98
    bg.add_patch(Circle((sx, sy), 0.1, fc="white", ec="#333333", lw=0.7))
    bg.text(sx, sy - 0.005, "+", ha="center", va="center", fontsize=8)
    arrow(bg, (hx[2], yt + s / 2 - 0.15), (sx + 0.03, sy + 0.1), rad=-0.2)
    arrow(bg, (bx + 2.25 + 0.03, yw + 0.07), (sx - 0.07, sy - 0.07), color=LANE["wide"], rad=0.25)
    bg.text(sx + 0.16, sy + 0.05, "logits", fontsize=5, color="#555555", va="center")
    ox = 6.62
    ax = panel(ox, 0.3, 0.13, 0.5)
    ax.bar([0], [d["prob"]], color="#D55E00", width=1); ax.set_ylim(0, 1); ax.set_xlim(-0.5, 0.5)
    arrow(bg, (sx + 0.03, sy - 0.1), (ox + 0.065, 0.83), rad=0.0)
    bg.text(ox + 0.065, 0.24, "softmax\n" + r"$P$(psychosis)", ha="center", va="top", fontsize=5.2, color=LANE["out"])

    # auxiliary head note
    bg.text(x6 + 0.06, yt - 0.42, "aux. head:\nage, sex\n(training only)", ha="center", va="top", fontsize=4.6,
            color="#888888")
    # parameter count / regularisation footnote
    bg.text(0.12, 0.08, "184,573 parameters.  Regularisation: edge dropout 0.2, region dropout 0.15, dropout 0.3, attention "
            "dropout 0.2, stochastic depth 0.2, LayerScale, AdamW weight decay 0.05, mixup, label smoothing.",
            fontsize=5.2, color="#555555", ha="left", va="bottom")
    # network legend
    bg.text(0.14, 1.36, "Networks (colour strips):", fontsize=4.8, color="#555555", va="center")
    for i, k in enumerate(nets):
        xx, yy = 0.14 + (i % 2) * 0.55, 1.2 - (i // 2) * 0.13
        bg.add_patch(plt.Rectangle((xx, yy - 0.035), 0.07, 0.07, color=YEO[k], lw=0))
        bg.text(xx + 0.09, yy, k, fontsize=4.6, va="center")
    return fig
