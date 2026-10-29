"""Figure 1 (pictorial): the study as a left-to-right story built from real images of the pipeline.

Called by scripts/make_figures.py. Example data: participant sub-10159 (UCLA CNP, ds000030); the mean
BOLD images of five sites; the ROC and Grad-CAM thumbnails of the final run (fallback: v9_pre_lr1e-3).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YEO = {"Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E", "SalVentAttn": "#C43AFA",
       "Limbic": "#B8CF6E", "Cont": "#E69422", "Default": "#CD3E4E", "Subcortex": "#8C8C8C"}
STAGE = {1: "#2f5d8a", 2: "#2f5d8a", 3: "#2a7a66", 4: "#4f7a28", 5: "#4f7a28", 6: "#4f7a28",
         7: "#7a3f86", 8: "#b4621b", 9: "#4d4d4d"}


def arrow(ax, p, q, color="#666666", lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color, shrinkA=0, shrinkB=0))


def axial(path, z_mm):
    img = nib.load(str(path))
    k = int(round(np.linalg.solve(img.affine, [0, 0, z_mm, 1])[2]))
    return np.rot90(np.asarray(img.dataobj[:, :, k], dtype=float))


def norm(a, lo=2, hi=99.5):
    v = a[a > 0]
    if v.size == 0:
        return a
    l, h = np.percentile(v, [lo, hi])
    return np.clip((a - l) / (h - l), 0, 1)


def inset(fig, W, H, x, y, w, h, frame=False):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame); s.set_linewidth(0.5); s.set_color("#999999")
    return ax


def figure1_story(final_run=None):
    from nilearn import plotting
    from sklearn.metrics import roc_auc_score, roc_curve
    from psikoz.features import load_atlas

    W, H = 7.2, 4.35
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
    pw, ph = 1.3, 1.25
    xs = [0.1, 1.53, 2.96, 4.39, 5.82]
    xb = [0.6, 2.34, 4.08, 5.82]                           # second row: 9, 8, 7, 6
    y1, y2 = 2.58, 0.45

    def caption(x, y, n, title, text):
        bg.add_patch(plt.Circle((x + 0.09, y + ph + 0.33), 0.1, fc=STAGE[n], ec="none"))
        bg.text(x + 0.09, y + ph + 0.326, str(n), ha="center", va="center", color="white", fontsize=6.6, fontweight="bold")
        bg.text(x + 0.23, y + ph + 0.33, title, ha="left", va="center", fontsize=6.9, fontweight="bold", color=STAGE[n])
        bg.text(x + pw / 2, y - 0.05, text, ha="center", va="top", fontsize=5.5, color="#333333", linespacing=1.18)

    feat = ROOT / "data" / "features" / "schaefer100s_gsr"
    prep = ROOT / "data" / "preproc"
    sub = "sub-10159"

    # 1 multi-site cohorts: stacked mean BOLD images of five sites
    caption(xs[0], y1, 1, "Multi-site cohorts", "8 sites in 3 countries, 965 participants\n(318 psychosis) + external site (TCP)")
    sites = [("ds000030", "sub-10159"), ("ds001461", "sub-101"), ("srpbs-KUT", "sub-0670"),
             ("srpbs-UTO", "sub-1023"), ("ds005073", "sub-B06")]
    for i, (d, s) in enumerate(sites[::-1]):
        sl = axial(prep / d / s / "func" / f"{s}_task-rest_run-1_space-MNI152NLin2009aSym_boldref.nii.gz", 20)
        off = (len(sites) - 1 - i) * 0.1
        ax = inset(fig, W, H, xs[0] + off, y1 + off * 0.72, 0.86, 0.9, frame=True)
        ax.set_facecolor("black"); ax.imshow(norm(sl), cmap="gray", interpolation="bilinear")

    # 2 raw rs-fMRI time stack + T1w
    caption(xs[1], y1, 2, "Raw rs-fMRI and T1w", "BOLD time series (TR 0.8–2.7 s)\nand anatomical image per participant")
    img = nib.load(str(ROOT / "datasets" / "ds000030" / sub / "func" / f"{sub}_task-rest_bold.nii.gz"))
    k = img.shape[2] // 2
    for i, t in enumerate([120, 90, 60, 30, 0]):
        off = (4 - i) * 0.09
        ax = inset(fig, W, H, xs[1] + 0.02 + off, y1 + 0.3 + off * 0.6, 0.6, 0.6, frame=True)
        ax.imshow(norm(np.rot90(np.asarray(img.dataobj[:, :, k, t], dtype=float))), cmap="gray")
    arrow(bg, (xs[1] + 0.1, y1 + 0.2), (xs[1] + 0.48, y1 + 0.45), color="#888888", lw=0.6)
    bg.text(xs[1] + 0.06, y1 + 0.12, "time", fontsize=4.8, style="italic", color="#555555")
    t1img = nib.load(str(ROOT / "datasets" / "ds000030" / sub / "anat" / f"{sub}_T1w.nii.gz"))
    ax = inset(fig, W, H, xs[1] + 0.82, y1 + 0.02, 0.48, 0.62, frame=True)
    ax.imshow(norm(np.rot90(np.asarray(t1img.dataobj[t1img.shape[0] // 2], dtype=float))), cmap="gray", aspect="auto")
    bg.text(xs[1] + 1.06, y1 + 0.7, "T1w", fontsize=4.8, ha="center", color="#555555")

    # 3 preprocessing: BOLD in MNI space with the template outline
    caption(xs[2], y1, 3, "Preprocessing", "motion correction, T1w → MNI (SyN),\nBOLD → T1w, 3 mm grid, QC")
    ax = inset(fig, W, H, xs[2] + 0.02, y1 - 0.01, pw - 0.04, ph + 0.02)
    disp = plotting.plot_anat(str(prep / "ds000030" / sub / "func" / f"{sub}_task-rest_run-1_space-MNI152NLin2009aSym_boldref.nii.gz"),
                              display_mode="z", cut_coords=[20], axes=ax, annotate=False, draw_cross=False, black_bg=True, dim=-0.3,
                              colorbar=False)
    disp.add_edges(str(prep / "template" / "tpl-MNI152NLin2009aSym_T1w.nii.gz"), color="#ff5050")

    # 4 parcellation coloured by network
    caption(xs[3], y1, 4, "Parcellation", "114 regions: Schaefer-100 (7 networks)\n+ 14 Harvard–Oxford subcortical")
    atlas, table = load_atlas("schaefer100s")
    nets = list(YEO)
    lut = np.zeros(len(table) + 1, int); lut[1:] = [nets.index(n) + 1 for n in table.network]
    lab = np.asarray(atlas.dataobj).astype(int)
    for z, xo in ((46, 0.0), (6, 0.62)):
        kk = int(round(np.linalg.solve(atlas.affine, [0, 0, z, 1])[2]))
        sl = np.rot90(lut[lab[:, :, kk]]).astype(float); sl[sl == 0] = np.nan
        ax = inset(fig, W, H, xs[3] + xo, y1 + 0.3, 0.68, 0.8)
        ax.imshow(sl, cmap=ListedColormap([YEO[n] for n in nets]), vmin=0.5, vmax=len(nets) + 0.5, interpolation="nearest")
        ax.axis("off")
    for i, n in enumerate(nets):
        x0, yy = xs[3] + 0.04 + (i % 4) * 0.32, y1 + 0.17 - (i // 4) * 0.12
        bg.add_patch(plt.Rectangle((x0, yy - 0.035), 0.07, 0.07, color=YEO[n], lw=0))
        bg.text(x0 + 0.09, yy, n.replace("SalVentAttn", "SalVA").replace("Subcortex", "Subc."), fontsize=4.3, va="center")

    # 5 denoised regional time series
    caption(xs[4], y1, 5, "Denoising", "censoring (FD), band-pass 0.01–0.1 Hz,\nmotion + WM + CSF + global regression")
    z = np.load(feat / "ds000030" / f"{sub}_task-rest_run-1.npz")
    ts, ok = z["timeseries"], z["frame_mask"]
    pick = [table.index[table.network == n][3] for n in ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Default", "Subcortex")]
    ax = inset(fig, W, H, xs[4] + 0.02, y1 + 0.03, pw - 0.04, ph - 0.05)
    tt = np.arange(len(ts)) * 2.0
    for j, r in enumerate(pick):
        ax.plot(tt, ts[:, r] * 0.28 - j, lw=0.5, color=YEO[table.network[r]])
    for b in np.flatnonzero(~ok):
        ax.axvspan(tt[b] - 1, tt[b] + 1, color="#dddddd", lw=0, zorder=0)
    ax.set_xlim(0, tt[-1]); ax.axis("off")

    # 6 connectivity matrix (row 2, right to left)
    caption(xb[3], y2, 6, "Connectivity", "Pearson correlation over clean frames\n" + r"$C\in\mathbb{R}^{114\times114}$ per participant")
    order = np.argsort([nets.index(n) for n in table.network], kind="stable")
    c = np.tanh(z["fc"]); np.fill_diagonal(c, 1)
    ax = inset(fig, W, H, xb[3] + 0.12, y2 + 0.02, ph - 0.05, ph - 0.05)
    ax.imshow(c[np.ix_(order, order)], cmap="RdBu_r", vmin=-0.8, vmax=0.8, interpolation="nearest")
    bounds = np.cumsum([0] + [int((table.network == n).sum()) for n in nets])
    for i, n in enumerate(nets):
        ax.add_patch(plt.Rectangle((-7, bounds[i] - 0.5), 5, bounds[i + 1] - bounds[i], color=YEO[n], clip_on=False, lw=0))
        ax.add_patch(plt.Rectangle((bounds[i] - 0.5, -7), bounds[i + 1] - bounds[i], 5, color=YEO[n], clip_on=False, lw=0))
    ax.set_xlim(-7.5, 113.5); ax.set_ylim(113.5, -7.5); ax.axis("off")

    # 7 partitioning
    caption(xb[2], y2, 7, "Partitioning", "subject-level 5-fold CV (train / val / test),\nleave-one-site-out, external test")
    ax = inset(fig, W, H, xb[2] + 0.02, y2 + 0.02, pw - 0.04, ph - 0.04)
    col = {"train": "#9ecae1", "val": "#fdd49e", "test": "#D55E00"}
    for f in range(5):
        for b in range(5):
            role = "test" if b == f else "val" if b == (f + 1) % 5 else "train"
            ax.add_patch(plt.Rectangle((b * 0.15, 0.93 - f * 0.11), 0.14, 0.09, color=col[role], lw=0))
        ax.text(-0.03, 0.975 - f * 0.11, f"Fold {f + 1}", ha="right", va="center", fontsize=4.3)
    for b in range(8):
        ax.add_patch(plt.Rectangle((b * 0.093, 0.24), 0.086, 0.09, color=col["test"] if b == 2 else col["train"], lw=0))
    ax.text(-0.03, 0.285, "LOSO", ha="right", va="center", fontsize=4.3)
    ax.add_patch(plt.Rectangle((0, 0.04), 0.6, 0.09, color=col["train"], lw=0))
    ax.add_patch(plt.Rectangle((0.63, 0.04), 0.11, 0.09, color="#4d4d4d", lw=0))
    ax.text(-0.03, 0.085, "External", ha="right", va="center", fontsize=4.3)
    for i, (key, name) in enumerate((("train", "train"), ("val", "validation"), ("test", "test"))):
        ax.add_patch(plt.Rectangle((0.8, 0.86 - i * 0.12), 0.05, 0.07, color=col[key], lw=0))
        ax.text(0.87, 0.895 - i * 0.12, name, fontsize=4.3, va="center")
    ax.add_patch(plt.Rectangle((0.8, 0.5), 0.05, 0.07, color="#4d4d4d", lw=0))
    ax.text(0.87, 0.535, "TCP", fontsize=4.3, va="center")
    ax.set_xlim(-0.2, 1.15); ax.set_ylim(0, 1.04); ax.axis("off")

    # 8 learning
    caption(xb[1], y2, 8, "PsychosisNET", "self-supervised pretraining (770 runs),\nfine-tuning, 5-seed ensemble (Fig. 2)")
    ax = inset(fig, W, H, xb[1] + 0.02, y2 + 0.02, pw - 0.04, ph - 0.04)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [("SPD", "#dcefe9", "#2a7a66"), ("Tokens", "#e6f0d8", "#4f7a28"), ("Attn\n×4", "#fbe8d6", "#b4621b"),
              ("Pool", "#e6f0d8", "#4f7a28")]
    for i, (t, fc_, ec) in enumerate(stages):
        x0 = 0.02 + i * 0.25
        ax.add_patch(FancyBboxPatch((x0, 0.4), 0.2, 0.36, boxstyle="round,pad=0,rounding_size=0.03", fc=fc_, ec=ec, lw=0.6))
        ax.text(x0 + 0.1, 0.58, t, ha="center", va="center", fontsize=4.8, color=ec, fontweight="bold")
        if i:
            arrow(ax, (x0 - 0.045, 0.58), (x0 - 0.005, 0.58), lw=0.5)
    ax.add_patch(FancyBboxPatch((0.02, 0.07), 0.95, 0.2, boxstyle="round,pad=0,rounding_size=0.03", fc="#ececec", ec="#4d4d4d", lw=0.6))
    ax.text(0.5, 0.17, "+ wide linear path (6,441 connections)", ha="center", va="center", fontsize=4.6)
    ax.text(0.5, 0.93, "masked-region reconstruction", ha="center", va="center", fontsize=4.5, color="#b4621b")
    arrow(ax, (0.5, 0.87), (0.5, 0.79), lw=0.5, color="#b4621b")

    # 9 evaluation and explanation
    caption(xb[0], y2, 9, "Evaluation", "per-fold ROC / AUC, confusion matrices,\nlearning curves, Grad-CAM maps")
    run = ROOT / "data" / "runs" / (final_run or "v9_pre_lr1e-3")
    pr = pd.read_csv(run / "test_predictions.csv")
    ax = inset(fig, W, H, xb[0] + 0.08, y2 + 0.6, 0.5, 0.5, frame=True)
    fpr, tpr, _ = roc_curve(pr.y, pr.p1)
    ax.plot(fpr, tpr, color="#D55E00", lw=0.9); ax.plot([0, 1], [0, 1], ls="--", lw=0.4, color="#999999")
    ax.text(0.95, 0.07, f"AUC {roc_auc_score(pr.y, pr.p1):.2f}", ha="right", fontsize=4.5, transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax = inset(fig, W, H, xb[0] + 0.72, y2 + 0.66, 0.42, 0.42, frame=True)
    ax.imshow([[0.8, 0.2], [0.25, 0.75]], cmap="Blues", vmin=0, vmax=1)
    for (i, j), t in {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=4.4, color="white" if i == j else "#1a1a1a")
    ax = inset(fig, W, H, xb[0] + 0.05, y2 + 0.0, 1.22, 0.56)
    cam = run / "interpret" / "gradcam_psychosis.nii.gz"
    if cam.exists():
        plotting.plot_glass_brain(str(cam), display_mode="lzr", axes=ax, annotate=False, cmap="hot", colorbar=False,
                                  threshold=0.5 * float(np.nanmax(nib.load(str(cam)).get_fdata())))
    else:
        plotting.plot_glass_brain(None, display_mode="lzr", axes=ax, annotate=False)

    # story arrows
    m1, m2 = y1 + ph / 2, y2 + ph / 2
    for i in range(4):
        arrow(bg, (xs[i] + pw + 0.02, m1), (xs[i + 1] - 0.03, m1))
    arrow(bg, (xs[4] + pw / 2, y1 - 0.4), (xb[3] + pw / 2, y2 + ph + 0.47))
    for i in range(3, 0, -1):
        arrow(bg, (xb[i] - 0.03, m2), (xb[i - 1] + pw + 0.04, m2))
    return fig
