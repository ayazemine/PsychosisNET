"""Manuscript figures: pictorial methodology (Figure 1, scripts/fig_story.py), detailed methodology
(Supplementary Figure S1), PsychosisNET layer by layer with real tensors (Figure 2, scripts/fig_arch.py)
and the architecture with its equations (Supplementary Figure S2).

    python scripts/make_figures.py [final_run]

Writes figures/<name>.{pdf,svg,png} (PNG at 300 dpi).
Sample sizes are the classification sample of the schaefer100s_gsr feature set (METHODS.md, section 4).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

C = dict(data=("#dbe8f5", "#2f5d8a"), prep=("#dcefe9", "#2a7a66"), feat=("#e6f0d8", "#4f7a28"),
         split=("#f3e6f5", "#7a3f86"), learn=("#fbe8d6", "#b4621b"), eval=("#ececec", "#4d4d4d"),
         white=("#ffffff", "#7f7f7f"))


def box(ax, x, y, w, h, kind, title=None, lines=(), size=6.3, title_size=7.2, align="left", lw=0.8, pad=0.07):
    face, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=face, ec=edge, lw=lw))
    ty = y + h - pad
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=edge)
        ty -= 0.2
    tx = x + pad if align == "left" else x + w / 2
    for line in lines:
        heading = line.startswith("**")
        text = line.strip("*") if heading else ("• " + line[2:]) if line.startswith("- ") else line
        ax.text(tx, ty, text, ha="left" if align == "left" else "center", va="top", fontsize=size,
                color=edge if heading else "#1a1a1a", fontweight="bold" if heading else "normal")
        ty -= 0.135 * size / 6.3
    return x, y, w, h


def label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="bottom", fontsize=9, fontweight="bold")


def arrow(ax, p, q, color="#555555", lw=1.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))



def figure1():
    W, H = 7.2, 4.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    pw, ph, gap = 2.22, 2.05, 0.17
    xs = [0.12, 0.12 + pw + gap, 0.12 + 2 * (pw + gap)]
    top, bottom = H - 0.18 - ph, 0.08

    panels = [
        ("A", xs[0], top, "data", "Data acquisition", [
            "**Classification cohorts (rs-fMRI)**",
            "- COBRE, USA (n = 98)",
            "- UCLA CNP, ds000030, USA (n = 180)",
            "- SZ-Smoking, ds001461, China (n = 81)",
            "- VSC SZ/BP, ds005073, USA (n = 30)",
            "- SRPBS KTT / KUT / SWA / UTO, Japan",
            "   (n = 122 / 202 / 120 / 132)",
            "  8 sites, n = 965 (318 psychosis,",
            "  647 no psychosis: HC, non-psychotic BP)",
            "**Label-free pool**",
            "- 770 runs never in the psychosis task",
            "**Independent external test**",
            "- TCP, ds005237, 2 US sites (n = 241)",
        ]),
        ("B", xs[1], top, "prep", "Image preprocessing (ANTs)", [
            "- Raw BIDS / per-volume archives",
            "   (COBRE: released NIAK-preprocessed)",
            "- Rigid motion correction (6 dof)",
            "- T1w → MNI152 2009a sym. (SyN)",
            "- BOLD → T1w (affine, 12 dof);",
            "   no T1w: BOLD → template (SyN)",
            "- Single resampling to 3 mm MNI",
            "- Nuisance signals: 6 motion,",
            "   WM, CSF, global signal, FD",
            "- QC: registration overlay and",
            "   brain-mask Dice for every run",
        ]),
        ("C", xs[2], top, "feat", "Denoising and connectivity", [
            "- Discard initial volumes (8–10 s)",
            "- Censor frames, FD > 0.5 mm",
            "   (multiband: low-passed FD > 0.3 mm)",
            "- Interpolate censored frames, detrend,",
            "   band-pass 0.01–0.1 Hz",
            "- Regress 6 motion + WM + CSF + GS",
            "- Resample to TR = 2 s",
            "- 114 ROIs: Schaefer-100 (7 networks)",
            "   + 14 Harvard–Oxford subcortical",
            "- Pearson correlation, clean frames",
            "   → " + r"$C\in\mathbb{R}^{114\times114}$",
            "- Inclusion: ≥ 120 s clean data,",
            "   ≥ 90 % of ROIs covered",
        ]),
        ("D", xs[2], bottom, "split", "Partitioning and leakage control", [
            "- Subject-level splits",
            "- Pooled 5-fold CV stratified by",
            "   site × diagnosis (train/val/test 3:1:1)",
            "- Leave-one-site-out (8 folds)",
            "- External site: never used for training,",
            "   pretraining or model selection",
            "- Fitted only on the training part:",
            "   SPD reference, scaling, weights",
            "- Validation part only: early stopping,",
            "   Platt scaling, decision threshold",
            "- Test part scored once",
        ]),
        ("E", xs[1], bottom, "learn", "Model learning", [
            "**Self-supervised pretraining**",
            "- Masked-ROI connectivity reconstruction",
            "   + age / sex (770 unlabelled runs)",
            "**Supervised fine-tuning: PsychosisNET**",
            "- Class-weighted cross-entropy, mixup,",
            "   label smoothing, aux. age / sex loss",
            "- AdamW (weight decay 0.05), one-cycle LR",
            "- Early stopping on validation AUC",
            "- 5-seed ensemble per fold",
            "- Platt calibration + threshold",
            "   (validation part)",
        ]),
        ("F", xs[0], bottom, "eval", "Evaluation", [
            "- Pooled 5-fold CV (mean ± SD)",
            "- Leave-one-site-out generalisation",
            "- External test on TCP: psychosis vs HC;",
            "   non-psychotic psychiatric patients",
            "   (specificity of the psychosis signal)",
            "- AUC, balanced accuracy,",
            "   sensitivity, specificity, 95 % CI",
            "- Comparators: linear models on",
            "   tangent connectivity, multi-view",
            "   stacking, ablations",
        ]),
    ]
    for tag, x, y, kind, title, lines in panels:
        box(ax, x, y, pw, ph, kind, title, lines)
        label(ax, x - 0.02, y + ph + 0.005, tag)
    mid = top + ph / 2
    arrow(ax, (xs[0] + pw, mid), (xs[1], mid))
    arrow(ax, (xs[1] + pw, mid), (xs[2], mid))
    arrow(ax, (xs[2] + pw / 2, top), (xs[2] + pw / 2, bottom + ph))
    midb = bottom + ph / 2
    arrow(ax, (xs[2], midb), (xs[1] + pw, midb))
    arrow(ax, (xs[1], midb), (xs[0] + pw, midb))
    return fig


def figure2():
    W, H = 3.5, 6.4
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    L, R = 0.1, 3.4                      # full width
    dl, dr = 0.1, 2.5                    # deep path
    wl, wr = 2.64, 3.4                   # wide path
    LS, GAP = 0.165, 0.15                # line spacing, vertical gap between boxes

    def stack(top, x0, x1, kind, title, lines, size=6.0, title_size=6.6, ls=LS):
        h = 0.08 + (0.2 if title else 0) + len(lines) * ls + 0.05
        face, edge = C[kind]
        ax.add_patch(FancyBboxPatch((x0, top - h), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=face, ec=edge, lw=0.8))
        y = top - 0.07
        if title:
            ax.text((x0 + x1) / 2, y, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=edge)
            y -= 0.2
        for line in lines:
            ax.text((x0 + x1) / 2, y - ls / 2 + 0.01, line, ha="center", va="center", fontsize=size, color="#1a1a1a")
            y -= ls
        return top - h

    def down(x, y0, y1):
        arrow(ax, (x, y0), (x, y1))

    dc = (dl + dr) / 2
    y = H - 0.08
    y = stack(y, L, R, "data", "Input: one resting-state run",
              [r"correlation matrix $C\in\mathbb{R}^{114\times114}$  (100 cortical + 14 subcortical ROIs)"])
    down((L + R) / 2, y, y - GAP); y -= GAP
    y = stack(y, L, R, "prep", "Learnable SPD tangent layer", [
        r"$C_s=(1-\alpha)\,C+\alpha I$  (learned shrinkage $\alpha$)",
        r"$Y=W^{\top}C_s\,W,\quad W=W_0\,(I+D)$,  $W_0=\bar{C}_{\mathrm{train}}^{-1/2}$",
        r"$T=\mathrm{logm}(Y)\in\mathbb{R}^{114\times114}$"], ls=0.19)
    split = y - GAP
    down(dc, y, split); down((wl + wr) / 2, y, split)
    # deep path
    y = stack(split, dl, dr, "feat", "Region tokens", [
        r"$x_i=\mathrm{LN}(W_e\,t_i)+e^{\mathrm{ROI}}_i+e^{\mathrm{net}}_{n(i)}$",
        r"$t_i$ = row $i$ of $T$,   $X\in\mathbb{R}^{114\times64}$"], ls=0.19)
    down(dc, y, y - GAP); y -= GAP
    blk_top = y
    face, edge = C["learn"]
    y_in = blk_top - 0.27
    ax.text(dl + 0.06, blk_top - 0.06, u"Transformer block  \u00d7 4", ha="left", va="top", fontsize=6.6,
            fontweight="bold", color=edge)
    y_in = stack(y_in, dl + 0.08, dr - 0.08, "learn", "Connectivity-biased multi-head attention", [
        r"$\mathrm{softmax}\left(QK^{\top}/\sqrt{d}+g_h\,T+o_h\,|T|\right)V$",
        "4 heads; pre-LayerNorm, LayerScale, DropPath"], size=5.8, title_size=6.2, ls=0.19)
    down(dc, y_in, y_in - 0.1); y_in -= 0.1
    y_in = stack(y_in, dl + 0.08, dr - 0.08, "learn", "Feed-forward", [
        u"LN \u2192 Linear 64\u2192128 \u2192 GELU \u2192 Linear 128\u219264"], size=5.8, title_size=6.2)
    blk_bottom = y_in - 0.08
    ax.add_patch(FancyBboxPatch((dl, blk_bottom), dr - dl, blk_top - blk_bottom,
                                boxstyle="round,pad=0,rounding_size=0.05", fc="none", ec=edge, lw=0.8, ls="--"))
    y = blk_bottom
    down(dc, y, y - GAP); y -= GAP
    y = stack(y, dl, dr, "feat", "Attention pooling", [
        r"learned query over regions $\rightarrow$ $z\in\mathbb{R}^{64}$",
        "weights give a region-importance map"])
    pool_bottom = y
    hx = dl + 1.45
    down(dl + 0.72, y, y - GAP); down((hx + 0.08 + dr) / 2, y, y - GAP)
    y -= GAP
    y_head = stack(y, dl, hx, "learn", "Classification head", [
        u"Linear 64\u219264, GELU, LN", u"Linear 64\u21922"], size=5.8, title_size=6.2)
    stack(y, hx + 0.08, dr, "white", "Auxiliary head", ["age, sex", "(training loss only)"],
          size=5.6, title_size=6.0)
    # wide path
    face, edge = C["eval"]
    ax.add_patch(FancyBboxPatch((wl, y_head), wr - wl, split - y_head, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=face, ec=edge, lw=0.8))
    wc = (wl + wr) / 2
    ax.text(wc, split - 0.07, "Wide path", ha="center", va="top", fontsize=6.6, fontweight="bold", color=edge)
    items = ["upper triangle", "of T", r"$\in\mathbb{R}^{6441}$", u"\u2193", "Dropout 0.5", u"\u2193",
             "Linear", u"6441\u21922", "", "zero-initialised,", "frozen for the", "first 10 epochs,", "own learning",
             "rate (\u00d70.1)"]
    yy = (split + y_head) / 2 + len(items) * 0.15 / 2
    for t in items:
        ax.text(wc, yy, t, ha="center", va="center", fontsize=5.7, color="#1a1a1a")
        yy -= 0.15
    # sum and output
    sx, sy = 1.75, y_head - 0.3
    ax.add_patch(plt.Circle((sx, sy), 0.09, fc="white", ec="#333333", lw=0.8))
    ax.text(sx, sy - 0.005, "+", ha="center", va="center", fontsize=8)
    arrow(ax, ((dl + hx) / 2, y_head), (sx - 0.08, sy + 0.04))
    arrow(ax, (wc, y_head), (sx + 0.08, sy + 0.04))
    ax.text(sx + 0.15, sy, "logits", ha="left", va="center", fontsize=5.6, color="#555555")
    down(sx, sy - 0.09, sy - 0.09 - GAP)
    y_end = stack(sy - 0.09 - GAP, L, R, "data", None, [r"softmax  $\rightarrow$  $P(\mathrm{psychosis})$"], size=6.4)
    ax.set_ylim(y_end - 0.06, H)
    fig.set_size_inches(W, H - (y_end - 0.06))
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fig_arch import figure2_pictorial
    from fig_story import figure1_story

    final = sys.argv[1] if len(sys.argv) > 1 else None      # run folder of the final model (thumbnails)
    OUT.mkdir(exist_ok=True)
    for name, make in (("fig1_methodology", lambda: figure1_story(final)), ("figS1_methodology_detail", figure1),
                       ("fig2_architecture", lambda: figure2_pictorial(final)), ("figS2_architecture_detail", figure2)):
        fig = make()
        for ext in ("pdf", "svg"):
            fig.savefig(OUT / f"{name}.{ext}")
        fig.savefig(OUT / f"{name}.png", dpi=300)
        plt.close(fig)
        print("->", OUT / f"{name}.pdf")
