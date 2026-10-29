"""Result figures and tables for the manuscript (SVG + PDF figures, CSV + Markdown tables).

    python -m psikoz.report --final final_psychosisnet \
        --compare "Linear SVM=cmp_svm" "XGBoost=cmp_xgboost" "MLP=cmp_mlp_lr1e-3" \
                  "BrainNetCNN=cmp_brainnetcnn_lr1e-3" "BNT=cmp_bnt_lr1e-4" --out results

Every model is evaluated the same way. In each fold the test probabilities are Platt-calibrated on the
validation part and dichotomised at the validation threshold that maximises balanced accuracy
(Youden), so threshold-based metrics never use the test part. Pooled metrics concatenate the five test
folds; 95 % confidence intervals are stratified bootstrap percentiles (2 000 resamples); AUCs are
compared with the proposed model by the paired DeLong test (Sun & Xu 2014), Holm-corrected.

Tables  T_performance (all models: mean +- SD over folds and pooled with 95 % CI), T_folds (proposed
        model: every metric per fold), T_folds_all (every model, every metric, every fold),
        T_confusion (per fold and pooled confusion matrices of every model), T_confusion_cv (confusion
        matrix cells and rates as mean +- SD over folds, every model), T_delong
Figures F_roc_folds (proposed model), F_roc_folds_all (every model, per fold, mean +- SD), F_roc_compare
        (all models, pooled test folds), F_roc_compare_cv (all models, fold-averaged ROC +- SD),
        F_confusion (proposed model per fold), F_confusion_cmp<i>_<model>, F_confusion_all (every model:
        pooled counts, fold mean +- SD rates), F_cv_metrics (per-fold metrics of every model, mean +- SD),
        F_learning_curves (proposed model), F_learning_curves_<model> (deep comparison models),
        F_learning_curves_compare (all deep models), F_predictions, F_calibration, F_metrics_compare and,
        when psikoz.interpret has been run, F_gradcam (glass brain), F_gradcam_groups (psychosis, control),
        F_gradcam_surface, F_gradcam_heatmap (subject x region and network x fold heat maps),
        F_top_regions, F_network_ig, F_embedding; F_confounds (log-odds against motion, age and sex);
        T_sites_pooled_cv (multi-site CV scored per site); with --single the single-site CV runs:
        T_single_vs_multi and F_single_vs_multi (single-site CV vs multi-site CV vs leave-site-out);
        with --lso the leave-one-site-out forest plot F_lso_forest and table T_lso (per-site AUC, bootstrap
        95 % CI)

Nothing is trained here: the inputs are the saved predictions of each run (or, with --predictions
rescored, those recomputed from the saved models by psikoz.evaluate) and the saved Grad-CAM maps.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)

from .config import load_config

# Okabe-Ito colour-blind safe palette; the proposed model is always the first (vermillion).
PALETTE = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
CLASS_COL = {0: "#0072B2", 1: "#D55E00"}
# Yeo et al. (2011) 7-network colours + grey for the subcortical regions (as in scripts/fig_story.py)
NET_COL = {"Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E", "SalVentAttn": "#C43AFA",
           "Limbic": "#B8CF6E", "Cont": "#E69422", "Default": "#CD3E4E", "Subcortex": "#8C8C8C"}
# Sites: a qualitative palette that does not reuse the model colours
SITE_COL = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666", "#1F78B4"]
MODEL_COL = {}                                                  # model name -> colour, fixed in main()


def net_colour(n):
    return NET_COL.get(str(n).replace("7Networks_", ""), "#999999")
CLASS_NAME = {0: "No psychosis", 1: "Psychosis"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none",
    "pdf.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ----------------------------------------------------------------------------- data
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


PREDICTIONS = ""                                              # "" = run folder, "rescored" = psikoz.evaluate


def load_run(run_dir):
    """Test predictions with calibrated probability and validation-threshold decision per fold."""
    val, test = (pd.read_csv(run_dir / PREDICTIONS / f"{part}_predictions.csv") for part in ("val", "test"))
    out = []
    for fold, t in test.groupby("fold"):
        v = val[val.fold == fold]
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(v.p1.to_numpy())[:, None], v.y)
        cal = lambda p: platt.predict_proba(logit(p)[:, None])[:, 1]
        pv = cal(v.p1.to_numpy())
        fpr, tpr, thr = roc_curve(v.y, pv)
        threshold = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
        out.append(t.assign(p_cal=cal(t.p1.to_numpy()), threshold=threshold,
                            decision=(cal(t.p1.to_numpy()) > threshold).astype(int)))
    return pd.concat(out, ignore_index=True)


def metrics(y, p, d):
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(AUC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, BAcc=balanced_accuracy_score(y, d),
                Acc=(tp + tn) / len(y), Sens=sens, Spec=spec, PPV=tp / max(tp + fp, 1), NPV=tn / max(tn + fn, 1),
                F1=f1_score(y, d, zero_division=0), MCC=matthews_corrcoef(y, d), Brier=brier_score_loss(y, p),
                TP=tp, FN=fn, FP=fp, TN=tn, n=len(y))


def bootstrap_ci(y, p, d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics(y[i], p[i], d[i])
        draws.append({k: m[k] for k in ("AUC", "BAcc", "Acc", "Sens", "Spec", "PPV", "NPV", "F1", "MCC", "Brier")})
    draws = pd.DataFrame(draws)
    return {k: (draws[k].quantile(0.025), draws[k].quantile(0.975)) for k in draws}


def delong(y, p1, p2):
    """Paired DeLong test for two correlated AUCs (fast version, Sun & Xu 2014). Returns (diff, z, p)."""
    pos, neg = p1[y == 1], p1[y == 0]

    def components(s):
        sp, sn = s[y == 1], s[y == 0]
        cmp = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
        return cmp.mean(), cmp.mean(1), cmp.mean(0)

    a1, v10_1, v01_1 = components(p1)
    a2, v10_2, v01_2 = components(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2])); s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(pos) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(neg)
    z = (a1 - a2) / np.sqrt(max(var, 1e-12))
    return a1 - a2, z, 2 * stats.norm.sf(abs(z))


def markdown(df):
    """GitHub-style Markdown table (avoids the optional tabulate dependency)."""
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join([lines[0], "|" + "|".join("---" for _ in df.columns) + "|", *lines[1:]]) + "\n"


def save(fig, out, name):
    for ext in ("svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def legend_box(leg):
    leg.set_frame_on(True)
    leg.get_frame().set(facecolor="white", edgecolor="#cccccc", linewidth=0.5, alpha=0.92)


def fmt(m, s=None, ci=None, digits=3):
    t = f"{m:.{digits}f}"
    if s is not None:
        t += f" ± {s:.{digits}f}"
    if ci is not None:
        t += f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
    return t


# ----------------------------------------------------------------------------- tables
KEYS = ["AUC", "BAcc", "Acc", "Sens", "Spec", "PPV", "NPV", "F1", "MCC", "Brier"]


def tables(models, out):
    perf, folds_rows, conf_rows, pooled_store, all_folds, conf_cv = [], [], [], {}, [], []
    for name, df in models.items():
        per_fold = pd.DataFrame([dict(fold=f, **metrics(g.y.to_numpy(), g.p_cal.to_numpy(), g.decision.to_numpy()))
                                 for f, g in df.groupby("fold")])
        all_folds.append(per_fold.assign(Model=name))
        rates = per_fold.assign(TPR=per_fold.TP / (per_fold.TP + per_fold.FN), FNR=per_fold.FN / (per_fold.TP + per_fold.FN),
                                FPR=per_fold.FP / (per_fold.FP + per_fold.TN), TNR=per_fold.TN / (per_fold.FP + per_fold.TN))
        conf_cv.append({"Model": name, **{k: fmt(rates[k].mean(), rates[k].std(), digits=1 if k in ("TP", "FN", "FP", "TN") else 3)
                                          for k in ("TP", "FN", "FP", "TN", "TPR", "FNR", "FPR", "TNR")}})
        y, p, d = df.y.to_numpy(), df.p_cal.to_numpy(), df.decision.to_numpy()
        pooled = metrics(y, p, d)
        ci = bootstrap_ci(y, p, d)
        pooled_store[name] = (pooled, ci)
        row = {"Model": name}
        for k in KEYS:
            row[f"{k} (fold mean ± SD)"] = fmt(per_fold[k].mean(), per_fold[k].std())
            row[f"{k} (pooled, 95% CI)"] = fmt(pooled[k], ci=ci[k])
        perf.append(row)
        for _, r in per_fold.iterrows():
            conf_rows.append(dict(Model=name, Fold=int(r.fold) + 1, TP=int(r.TP), FN=int(r.FN), FP=int(r.FP), TN=int(r.TN),
                                  Sens=r.Sens, Spec=r.Spec, BAcc=r.BAcc))
        conf_rows.append(dict(Model=name, Fold="Pooled", TP=pooled["TP"], FN=pooled["FN"], FP=pooled["FP"], TN=pooled["TN"],
                              Sens=pooled["Sens"], Spec=pooled["Spec"], BAcc=pooled["BAcc"]))
        if not folds_rows:                                  # first model = proposed model: full per-fold table
            for _, r in per_fold.iterrows():
                folds_rows.append({"Fold": int(r.fold) + 1, "n": int(r.n), **{k: round(r[k], 3) for k in KEYS}})
            folds_rows.append({"Fold": "Mean ± SD", "n": int(per_fold.n.sum()),
                               **{k: fmt(per_fold[k].mean(), per_fold[k].std()) for k in KEYS}})
            folds_rows.append({"Fold": "Pooled [95% CI]", "n": int(pooled["n"]), **{k: fmt(pooled[k], ci=ci[k]) for k in KEYS}})

    names = list(models)
    ref = models[names[0]]
    tests = []
    for name in names[1:]:
        m = ref.merge(models[name], on=["subject", "fold"], suffixes=("_a", "_b"))
        diff, z, p = delong(m.y_a.to_numpy(), m.p_cal_a.to_numpy(), m.p_cal_b.to_numpy())
        tests.append(dict(Comparison=f"{names[0]} vs {name}", n=len(m), **{"ΔAUC": round(diff, 3), "z": round(z, 2), "p": p}))
    if tests:
        t = pd.DataFrame(tests).sort_values("p")
        k = len(t)
        t["p (Holm)"] = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * t["p"].to_numpy()))
        t["p"] = t["p"].map(lambda v: f"{v:.2g}"); t["p (Holm)"] = t["p (Holm)"].map(lambda v: f"{v:.2g}")
        tests = t
    fold_table = pd.concat(all_folds, ignore_index=True)
    fold_table = fold_table.assign(Fold=fold_table.fold + 1)[["Model", "Fold", "n", *KEYS, "TP", "FN", "FP", "TN"]].round(3)
    for fname, rows in (("T_performance", perf), ("T_folds", folds_rows), ("T_confusion", conf_rows),
                        ("T_folds_all", fold_table), ("T_confusion_cv", conf_cv)):
        df = pd.DataFrame(rows)
        if fname == "T_confusion":
            df = df.round(3)
        df.to_csv(out / f"{fname}.csv", index=False)
        (out / f"{fname}.md").write_text(markdown(df), encoding="utf-8")
    if len(tests):
        tests.to_csv(out / "T_delong.csv", index=False)
        (out / "T_delong.md").write_text(markdown(tests), encoding="utf-8")
    return pooled_store, fold_table


# ----------------------------------------------------------------------------- figures
def fig_roc_folds(df, out, name):
    fig, ax = plt.subplots(figsize=(3.0, 2.9))
    grid = np.linspace(0, 1, 201)
    curves, aucs = [], []
    for i, (f, g) in enumerate(df.groupby("fold")):
        fpr, tpr, _ = roc_curve(g.y, g.p_cal)
        a = roc_auc_score(g.y, g.p_cal)
        ax.plot(fpr, tpr, lw=0.8, alpha=0.75, color=PALETTE[1 + i % 6], label=f"Fold {f + 1} (AUC = {a:.3f})")
        curves.append(np.interp(grid, fpr, tpr)); aucs.append(a)
    curves = np.array(curves); curves[:, 0] = 0
    mean, sd = curves.mean(0), curves.std(0)
    ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color="#999999", alpha=0.25, lw=0, label="± 1 SD")
    ax.plot(grid, mean, color=PALETTE[0], lw=1.8, label=f"Mean (AUC = {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#888888")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)",
           title=f"{name}: ROC per test fold")
    ax.set_aspect("equal"); legend_box(ax.legend(loc="lower right"))
    save(fig, out, "F_roc_folds")


def fig_roc_compare(models, pooled, out):
    fig, ax = plt.subplots(figsize=(3.0, 2.9))
    for i, (name, df) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(df.y, df.p_cal)
        a, ci = pooled[name][0]["AUC"], pooled[name][1]["AUC"]
        ax.plot(fpr, tpr, color=PALETTE[i], lw=1.8 if i == 0 else 1.0,
                label=f"{name}  {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", zorder=10 - i)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#888888")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)",
           title="Pooled test folds: AUC [95 % CI]")
    ax.set_aspect("equal"); legend_box(ax.legend(loc="lower right", handlelength=1.4))
    save(fig, out, "F_roc_compare")


def fig_confusion(df, out, name, fname="F_confusion"):
    groups = [(f"Fold {f + 1}", g) for f, g in df.groupby("fold")] + [("Pooled", df)]
    fig, axes = plt.subplots(1, len(groups), figsize=(1.25 * len(groups) + 0.3, 1.55))
    for ax, (title, g) in zip(axes, groups):
        cm = confusion_matrix(g.y, g.decision, labels=[0, 1])
        rate = cm / cm.sum(1, keepdims=True)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{rate[i, j]:.0%}", ha="center", va="center", fontsize=6.3,
                        color="white" if rate[i, j] > 0.55 else "#1a1a1a")
        ax.set_xticks([0, 1], ["No", "Yes"]); ax.set_yticks([0, 1], ["No", "Yes"] if ax is axes[0] else ["", ""])
        ax.set_title(title, pad=3)
        ax.spines[:].set_visible(True)
        if ax is axes[0]:
            ax.set_ylabel("True: psychosis")
        ax.set_xlabel("Predicted")
    fig.suptitle(f"{name}: confusion matrices (validation-selected threshold)", y=1.06, fontsize=7.5)
    fig.subplots_adjust(wspace=0.25)
    save(fig, out, fname)


def fig_learning_curves(hist, out, name, fname="F_learning_curves"):
    folds = sorted(hist.fold.unique())
    fig, axes = plt.subplots(2, len(folds), figsize=(1.45 * len(folds) + 0.4, 2.9), sharex=True)
    for j, f in enumerate(folds):
        h = hist[hist.fold == f]
        for row, (cols, ylabel) in enumerate(((("train_ce", "val_ce"), "Cross-entropy"), (("train_auc", "val_auc"), "AUC"))):
            ax = axes[row, j]
            for col, colour, lab in zip(cols, (CLASS_COL[0], CLASS_COL[1]), ("Training", "Validation")):
                piv = h.pivot_table(index="epoch", columns="seed", values=col)
                for s in piv.columns:
                    ax.plot(piv.index, piv[s], color=colour, lw=0.4, alpha=0.3)
                ax.plot(piv.index, piv.mean(1), color=colour, lw=1.2, label=lab)
            for e in h[h.selected == 1].epoch:
                ax.axvline(e, color="#666666", lw=0.4, ls=":")
            if row == 0:
                ax.set_title(f"Fold {f + 1}")
            if j == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Epoch"); ax.set_ylim(0.45, 1.01)
    axes[0, -1].legend(loc="upper right")
    fig.suptitle(f"{name}: learning curves (thin: seeds, thick: mean, dotted: selected epoch)", fontsize=7.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    save(fig, out, fname)


def fig_predictions(df, out, name):
    folds = sorted(df.fold.unique())
    fig, axes = plt.subplots(1, len(folds), figsize=(1.35 * len(folds) + 0.4, 2.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes, folds):
        g = df[df.fold == f]
        for k in (0, 1):
            v = g.loc[g.y == k, "p_cal"].to_numpy()
            parts = ax.violinplot(v, positions=[k], widths=0.75, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(CLASS_COL[k]); b.set_alpha(0.25); b.set_edgecolor("none")
            ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=2.5, color=CLASS_COL[k], alpha=0.75, lw=0)
            ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color="#1a1a1a", lw=1)
        ax.axhline(g.threshold.iloc[0], color="#444444", lw=0.6, ls="--")
        ax.set_xticks([0, 1], ["No\npsychosis", "Psychosis"])
        ax.set_title(f"Fold {f + 1}  (AUC {roc_auc_score(g.y, g.p_cal):.3f})")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Predicted probability of psychosis\n(calibrated)")
    fig.suptitle(f"{name}: test-set predictions per fold (dashed: decision threshold, bar: median)", y=1.04, fontsize=7.5)
    save(fig, out, "F_predictions")


def fig_calibration(models, out, bins=10):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(2.9, 3.3), gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
    edges = np.linspace(0, 1, bins + 1)
    for i, (name, df) in enumerate(models.items()):
        idx = np.clip(np.digitize(df.p_cal, edges) - 1, 0, bins - 1)
        frac = [df.y[idx == b].mean() if (idx == b).sum() >= 5 else np.nan for b in range(bins)]
        mid = [df.p_cal[idx == b].mean() if (idx == b).sum() >= 5 else np.nan for b in range(bins)]
        ax.plot(mid, frac, marker="o", ms=2.5, color=PALETTE[i], lw=1.6 if i == 0 else 0.9,
                label=f"{name} (Brier {brier_score_loss(df.y, df.p_cal):.3f})")
        if i == 0:
            axh.hist(df.p_cal[df.y == 0], bins=edges, color=CLASS_COL[0], alpha=0.6, label=CLASS_NAME[0])
            axh.hist(df.p_cal[df.y == 1], bins=edges, color=CLASS_COL[1], alpha=0.6, label=CLASS_NAME[1])
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#888888")
    ax.set(ylabel="Observed fraction with psychosis", xlim=(0, 1), ylim=(0, 1), title="Calibration (pooled test folds)")
    ax.legend(loc="upper left", fontsize=5.8)
    axh.set(xlabel="Predicted probability (calibrated)", ylabel="Count")
    axh.legend(fontsize=5.8)
    save(fig, out, "F_calibration")


def fig_metrics_compare(pooled, out):
    keys = ["AUC", "BAcc", "Sens", "Spec", "F1", "MCC"]
    names = list(pooled)
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        m, ci = pooled[name]
        x = np.arange(len(keys)) + (i - (len(names) - 1) / 2) * w
        vals = [m[k] for k in keys]
        err = np.array([[m[k] - ci[k][0], ci[k][1] - m[k]] for k in keys]).T
        ax.bar(x, vals, w * 0.92, color=PALETTE[i], label=name, yerr=err, error_kw=dict(lw=0.5, capsize=1.2))
    ax.set_xticks(range(len(keys)), ["AUC", "Balanced acc.", "Sensitivity", "Specificity", "F1", "MCC"])
    ax.set(ylim=(0, 1), ylabel="Pooled test score (95 % CI)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, out, "F_metrics_compare")


def fig_gradcam_heatmap(d, regions, out):
    """Grad-CAM heat maps: (a) every test participant x region, regions grouped by network, participants
    grouped by class and sorted by predicted probability; (b) network x fold mean for each class and their
    difference."""
    cam = pd.read_csv(d / "gradcam_subjects.csv")
    order = regions.reset_index().sort_values(["network", "index"]).index.to_numpy()
    order = regions.iloc[order]
    names, nets = order["name"].tolist(), order["network"].tolist()
    net_list = list(dict.fromkeys(nets))
    ncol = {n: net_colour(n) for n in net_list}
    cam = cam.sort_values(["y", "p"], ascending=[False, False]).reset_index(drop=True)

    folds = sorted(cam.fold.unique())
    rows = []
    for k in (1, 0):
        g = cam[cam.y == k]
        m = pd.DataFrame({f"F{f + 1}": g[g.fold == f][names].mean() for f in folds})
        m["Mean"] = g[names].mean()
        rows.append(m.groupby(pd.Series(nets, index=names)).mean().reindex(net_list))
    diff = rows[0][["Mean"]] - rows[1][["Mean"]]
    short = [n.replace("7Networks_", "") for n in net_list]

    fig = plt.figure(figsize=(7.1, 4.3))
    gs = fig.add_gridspec(2, 8, height_ratios=[0.035, 1], width_ratios=[0.03, 1.9, 0.04, 0.62, 0.62, 0.62, 0.15, 0.03],
                          hspace=0.02, wspace=0.12, left=0.05, right=0.92, top=0.9, bottom=0.14)
    ax = fig.add_subplot(gs[1, 1]); top = fig.add_subplot(gs[0, 1], sharex=ax); side = fig.add_subplot(gs[1, 0], sharey=ax)
    im = ax.imshow(cam[names].to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    top.imshow([[net_list.index(n) for n in nets]], aspect="auto", interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap([ncol[n] for n in net_list]))
    side.imshow(cam.y.to_numpy()[:, None], aspect="auto", interpolation="nearest",
                cmap=matplotlib.colors.ListedColormap([CLASS_COL[0], CLASS_COL[1]]), vmin=0, vmax=1)
    top.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    side.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for a in (top, side):
        a.spines[:].set_visible(False)
    bounds = np.flatnonzero(np.r_[True, np.array(nets[1:]) != np.array(nets[:-1])])
    ax.set_xticks([(b + e) / 2 - 0.5 for b, e in zip(bounds, list(bounds[1:]) + [len(nets)])], short,
                  rotation=45, ha="right", fontsize=5.8)
    for b in bounds[1:]:
        ax.axvline(b - 0.5, color="white", lw=0.4)
    n1 = int((cam.y == 1).sum())
    ax.axhline(n1 - 0.5, color="white", lw=0.8)
    ax.tick_params(left=False, labelleft=False)
    side.set_ylabel(f"Test participants: psychosis (n = {n1}, top), no psychosis (n = {len(cam) - n1}, bottom)", fontsize=6)
    ax.set_xlabel("Regions grouped by network")
    top.set_title("a  Grad-CAM per participant and region", loc="left", fontsize=7.2, fontweight="bold")
    cax = fig.add_subplot(gs[1, 2]); fig.colorbar(im, cax=cax); cax.set_ylabel("Grad-CAM (per-participant max = 1)", fontsize=6)

    vmax = float(max(rows[0].values.max(), rows[1].values.max()))
    lim = float(np.abs(diff.values).max()) or 1.0
    for c, (r, title) in enumerate(((rows[0], "Psychosis"), (rows[1], "No psychosis"))):
        a = fig.add_subplot(gs[1, 4 + c])
        a.imshow(r.values, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        a.set_xticks(range(r.shape[1]), r.columns, rotation=90, fontsize=5.6)
        a.set_yticks(range(len(short)), short if c == 0 else [], fontsize=6)
        a.set_title(title, fontsize=6.4)
        a.axvline(r.shape[1] - 1.5, color="white", lw=0.8)
        if c == 0:
            a.text(-0.1, 1.06, "b  Network mean per fold", transform=a.transAxes, ha="right", fontsize=7.2,
                   fontweight="bold")
    a = fig.add_subplot(gs[1, 6])
    imd = a.imshow(diff.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(len(short)):
        a.text(0, i, f"{diff.values[i, 0]:+.3f}", ha="center", va="center", fontsize=4.8, rotation=90)
    a.set_xticks([0], ["Δ"], fontsize=6); a.set_yticks([])
    a.set_title("Diff.", fontsize=6.4)
    cax = fig.add_subplot(gs[1, 7]); fig.colorbar(imd, cax=cax); cax.set_ylabel("Psychosis − no psychosis", fontsize=6)
    save(fig, out, "F_gradcam_heatmap")
    t = pd.concat({"psychosis": rows[0], "no_psychosis": rows[1], "difference": diff}, axis=1).round(4)
    t.index.name = "network"
    t.to_csv(out / "T_gradcam_networks.csv")
    flat = t.copy(); flat.columns = [f"{a} {b}" for a, b in flat.columns]
    (out / "T_gradcam_networks.md").write_text(markdown(flat.reset_index()), encoding="utf-8")


def fig_gradcam_surface(img, vmax, out):
    """Grad-CAM painted on the inflated fsaverage5 surface: lateral and medial views of both hemispheres."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from nilearn import datasets, plotting, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    fig = plt.figure(figsize=(5.2, 3.5))
    pos = {("left", "lateral"): (0, 0), ("right", "lateral"): (0, 1), ("left", "medial"): (1, 0), ("right", "medial"): (1, 1)}
    for (hemi, view), (r, c) in pos.items():
        tex = surface.vol_to_surf(img, fs[f"pial_{hemi}"], interpolation="nearest_most_frequent", radius=3.0)
        ax = fig.add_axes([0.02 + 0.44 * c, 0.52 - 0.46 * r, 0.44, 0.44], projection="3d")
        plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], tex, hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                    cmap="hot", vmin=0, vmax=vmax, threshold=0.3 * vmax, colorbar=False,
                                    axes=ax, figure=fig, bg_on_data=True)
        ax.set_title(f"{hemi.capitalize()} hemisphere, {view}", fontsize=6.5, y=0.92)
    cax = fig.add_axes([0.91, 0.25, 0.02, 0.5])
    cb = fig.colorbar(ScalarMappable(Normalize(0, vmax), "hot"), cax=cax)
    cb.set_label("Mean Grad-CAM (psychosis)", fontsize=6.5)
    cb.ax.axhline(0.3 * vmax, color="#333333", lw=0.6)
    fig.text(0.5, 0.995, "Grad-CAM on the inflated cortical surface (regions above 30 % of the maximum)",
             ha="center", va="top", fontsize=7.5)
    save(fig, out, "F_gradcam_surface")


def fig_interpret(run_dir, out, cfg):
    from nilearn import plotting

    d = run_dir / "interpret"
    if not d.exists():
        return
    regions = pd.read_csv(d / "regions.csv")
    vmax = float(regions["gradcam_psychosis"].max())
    fig = plt.figure(figsize=(7.0, 2.2))
    display = plotting.plot_glass_brain(str(d / "gradcam_psychosis.nii.gz"), display_mode="lyrz", colorbar=True,
                                        cmap="hot", vmin=0, vmax=vmax, threshold=0.5 * vmax, plot_abs=False, figure=fig)
    fig.suptitle("Grad-CAM region importance for psychosis (mean over test folds; regions above 50 % of the maximum)",
                 fontsize=7.5, y=1.0)
    save(fig, out, "F_gradcam")
    fig = plt.figure(figsize=(7.0, 3.6))
    vmax2 = float(regions[["gradcam_psychosis", "gradcam_control"]].values.max())
    for k, (fname, title) in enumerate((("gradcam_psychosis", "Participants with psychosis"),
                                        ("gradcam_control", "Participants without psychosis"))):
        ax = fig.add_axes([0.0, 0.5 - 0.5 * k, 1.0, 0.45])
        plotting.plot_glass_brain(str(d / f"{fname}.nii.gz"), display_mode="lyrz", colorbar=True, cmap="hot",
                                  vmin=0, vmax=vmax2, threshold=0.5 * vmax2, plot_abs=False, axes=ax, title=None)
        fig.text(0.01, 0.97 - 0.5 * k, f"{'ab'[k]}  {title}", fontsize=7.5, fontweight="bold", va="top")
    save(fig, out, "F_gradcam_groups")
    fig_gradcam_heatmap(d, regions, out)
    try:
        fig_gradcam_surface(d / "gradcam_psychosis.nii.gz", vmax, out)
    except Exception as e:                                   # surface plotting is optional
        print("surface plot skipped:", e)
        plt.close("all")

    top = regions.nlargest(15, "gradcam_psychosis")[::-1]
    from matplotlib.patches import Patch
    nets = [n for n in NET_COL if n in set(regions.network.str.replace("7Networks_", ""))]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.barh(range(len(top)), top.gradcam_psychosis, color=[net_colour(n) for n in top.network])
    ax.set_yticks(range(len(top)), [n.replace("7Networks_", "").replace("_", " ") for n in top.name], fontsize=5.8)
    ax.set_xlabel("Mean Grad-CAM (psychosis, test folds)")
    ax.legend(handles=[Patch(color=NET_COL[n], label=n) for n in nets], fontsize=5.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), title="Network", title_fontsize=5.8)
    ax.set_title("Top 15 regions")
    save(fig, out, "F_top_regions")

    net = pd.read_csv(d / "network_ig.csv", index_col=0)
    lim = float(np.abs(net.values).max())
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    im = ax.imshow(net.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(net)), net.columns, rotation=45, ha="right"); ax.set_yticks(range(len(net)), net.index)
    ax.spines[:].set_visible(True)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean IG attribution\n(+: towards psychosis)")
    ax.set_title("Integrated Gradients by network pair")
    save(fig, out, "F_network_ig")

    emb = pd.read_csv(d / "embeddings.csv")
    from sklearn.manifold import TSNE
    z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(emb.filter(regex=r"^z\d+$").to_numpy())
    site_names = {v: k for k, v in __import__("psikoz.torch_data", fromlist=["site_codes"]).site_codes(cfg).items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6))
    for k in (0, 1):
        a1.scatter(*z[emb.y == k].T, s=3, color=CLASS_COL[k], label=CLASS_NAME[k], lw=0, alpha=0.8)
    for i, s in enumerate(sorted(emb.site.unique())):
        name = site_names.get(s, s)
        a2.scatter(*z[emb.site == s].T, s=3, color=SITE_COL[i % len(SITE_COL)], label=SITE_LABEL.get(name, name).rstrip("*"),
                   lw=0, alpha=0.8)
    for k, (a, t) in enumerate(((a1, "Diagnosis"), (a2, "Site"))):
        a.set_xticks([]); a.set_yticks([]); a.spines[:].set_visible(True)
        a.set_title(f"{'ab'[k]}  t-SNE of the subject embedding: {t.lower()}", loc="left", fontsize=7.2, fontweight="bold")
        a.set_xlabel("t-SNE 1"); a.set_ylabel("t-SNE 2")
        a.legend(fontsize=5.5, markerscale=2.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), title=t, title_fontsize=5.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "F_embedding")


def fig_confounds(df, out, cfg, atlas):
    index = pd.read_csv(cfg["paths"]["features"] / atlas / "index.csv", low_memory=False)
    m = df.merge(index[["subject", "fd_mean", "age", "sex"]].drop_duplicates("subject"), on="subject", how="left")
    m["log_odds"] = logit(m.p_cal.to_numpy())
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0))
    rows = []
    for ax, (col, lab) in zip(axes, (("fd_mean", "Mean framewise displacement (mm)"), ("age", "Age (years)"))):
        for k in (0, 1):
            g = m[m.y == k].dropna(subset=[col])
            ax.scatter(g[col], g.log_odds, s=3, color=CLASS_COL[k], alpha=0.6, lw=0, label=CLASS_NAME[k])
            r, p = stats.spearmanr(g[col], g.log_odds)
            rows.append(dict(confound=col, group=CLASS_NAME[k], n=len(g), spearman_r=round(r, 3), p=f"{p:.2g}"))
        ax.set(xlabel=lab, ylabel="Model log-odds")
    ax = axes[2]
    for i, (k, s) in enumerate([(k, s) for k in (0, 1) for s in ("F", "M")]):
        v = m[(m.y == k) & (m.sex == s)].log_odds
        ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=CLASS_COL[k], alpha=0.35, lw=0.6), medianprops=dict(color="#1a1a1a"))
    for k in (0, 1):
        g = m[m.y == k]
        u = stats.mannwhitneyu(g[g.sex == "F"].log_odds, g[g.sex == "M"].log_odds)
        rows.append(dict(confound="sex (F vs M)", group=CLASS_NAME[k], n=len(g), spearman_r=np.nan, p=f"{u.pvalue:.2g}"))
    ax.set_xticks(range(4), ["F", "M", "F", "M"]); ax.set_ylabel("Model log-odds")
    ax.set_xlabel("No psychosis        Psychosis")
    axes[0].legend(fontsize=5.8)
    fig.suptitle("Model output against potential confounds, within each diagnostic group", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "F_confounds")
    t = pd.DataFrame(rows)
    t.to_csv(out / "T_confounds.csv", index=False)
    (out / "T_confounds.md").write_text(markdown(t), encoding="utf-8")


def mean_roc(df, grid=np.linspace(0, 1, 201)):
    """Fold-averaged ROC (vertical averaging) and per-fold AUCs."""
    curves, aucs = [], []
    for _, g in df.groupby("fold"):
        fpr, tpr, _ = roc_curve(g.y, g.p_cal)
        c = np.interp(grid, fpr, tpr); c[0] = 0
        curves.append(c); aucs.append(roc_auc_score(g.y, g.p_cal))
    return grid, np.array(curves), np.array(aucs)


def fig_roc_folds_all(models, out):
    n = len(models)
    ncol = 3 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.35 * nrow), squeeze=False)
    for k, (ax, (name, df)) in enumerate(zip(axes.flat, models.items())):
        for i, (f, g) in enumerate(df.groupby("fold")):
            fpr, tpr, _ = roc_curve(g.y, g.p_cal)
            ax.plot(fpr, tpr, lw=0.6, alpha=0.6, color=PALETTE[1 + i % 6], label=f"Fold {f + 1}")
        grid, curves, aucs = mean_roc(df)
        m, sd = curves.mean(0), curves.std(0)
        ax.fill_between(grid, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color="#999999", alpha=0.25, lw=0)
        ax.plot(grid, m, color=PALETTE[k], lw=1.6, label="Mean")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.5, color="#888888")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), title=name)
        ax.set_aspect("equal")
        ax.text(0.97, 0.05, f"AUC {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}", transform=ax.transAxes, ha="right",
                fontsize=6.5, color=PALETTE[k], fontweight="bold")
        if k == 0:
            legend_box(ax.legend(loc="upper left", fontsize=5.4, handlelength=1.2, ncol=2))
        ax.text(-0.02, 1.08, "abcdefghij"[k], transform=ax.transAxes, fontsize=9, fontweight="bold")
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("False positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("True positive rate")
    fig.suptitle("ROC curves per cross-validation test fold (band: ± 1 SD of the fold-averaged curve)", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "F_roc_folds_all")


def fig_roc_compare_cv(models, out):
    fig, ax = plt.subplots(figsize=(3.0, 2.9))
    for i, (name, df) in enumerate(models.items()):
        grid, curves, aucs = mean_roc(df)
        m, sd = curves.mean(0), curves.std(0)
        if i == 0:
            ax.fill_between(grid, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color=PALETTE[0], alpha=0.15, lw=0)
        ax.plot(grid, m, color=PALETTE[i], lw=1.8 if i == 0 else 1.0, zorder=10 - i,
                label=f"{name}  {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#888888")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (1 − specificity)",
           ylabel="True positive rate (sensitivity)", title="5-fold cross-validation: AUC mean ± SD")
    ax.set_aspect("equal"); legend_box(ax.legend(loc="lower right", handlelength=1.4))
    save(fig, out, "F_roc_compare_cv")


def fig_confusion_all(models, out):
    n = len(models)
    ncol = 3 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.5, 1.95 * nrow), squeeze=False)
    for k, (ax, (name, df)) in enumerate(zip(axes.flat, models.items())):
        cm = confusion_matrix(df.y, df.decision, labels=[0, 1])
        folds = np.array([confusion_matrix(g.y, g.decision, labels=[0, 1]) for _, g in df.groupby("fold")], float)
        rates = folds / folds.sum(2, keepdims=True)
        mean, sd = rates.mean(0), rates.std(0, ddof=1)
        im = ax.imshow(mean, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{mean[i, j]:.1%} ± {sd[i, j]:.1%}", ha="center", va="center", fontsize=5.8,
                        color="white" if mean[i, j] > 0.55 else "#1a1a1a")
        ax.set_xticks([0, 1], ["No psychosis", "Psychosis"])
        ax.set_yticks([0, 1], ["No psychosis", "Psychosis"], rotation=90, va="center")
        ax.set_title(name, pad=3)
        ax.spines[:].set_visible(True)
        ax.set_xlabel("Predicted")
        if k % ncol == 0:
            ax.set_ylabel("True")
        else:
            ax.set_yticklabels([])
        ax.text(-0.12, 1.06, "abcdefghij"[k], transform=ax.transAxes, fontsize=9, fontweight="bold")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.12, hspace=0.45)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Row-normalised rate (fold mean)")
    fig.suptitle("Confusion matrices: pooled counts; rates as mean ± SD over the 5 test folds", fontsize=7.5)
    save(fig, out, "F_confusion_all")


def fig_cv_metrics(fold_table, out):
    keys = [("AUC", "AUC"), ("BAcc", "Balanced acc."), ("Sens", "Sensitivity"), ("Spec", "Specificity"),
            ("F1", "F1"), ("MCC", "MCC")]
    names = list(dict.fromkeys(fold_table.Model))
    fig, axes = plt.subplots(1, len(keys), figsize=(7.1, 2.3), sharey=False)
    rng = np.random.default_rng(0)
    for ax, (k, lab) in zip(axes, keys):
        for i, name in enumerate(names):
            v = fold_table.loc[fold_table.Model == name, k].to_numpy(float)
            ax.bar(i, v.mean(), 0.7, color=PALETTE[i], alpha=0.35, lw=0)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="#1a1a1a", lw=0.7, capsize=1.6)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=PALETTE[i], lw=0, zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=5.8)
        ax.set_title(lab)
        lo = fold_table[k].min()
        ax.set_ylim(min(0, lo - 0.05) if k == "MCC" else max(0, lo - 0.1), 1.0)
    axes[0].set_ylabel("Test score per fold (bar: mean ± SD)")
    fig.tight_layout(w_pad=0.4)
    save(fig, out, "F_cv_metrics")


def fig_learning_curves_compare(histories, out):
    """histories: {model: history.csv}; mean over folds and seeds per epoch (runs that stopped early
    contribute until their last epoch)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for i, (name, h) in enumerate(histories.items()):
        col = MODEL_COL.get(name, PALETTE[i])
        runs = h.groupby("epoch").size()
        h = h[h.epoch.isin(runs.index[runs >= 0.5 * runs.max()])]     # avoid tails averaged over few runs
        g = h.groupby("epoch")
        col_ = col
        for ax, col, ls in ((axes[0], "train_ce", "-"), (axes[0], "val_ce", "--"), (axes[1], "train_auc", "-"),
                            (axes[1], "val_auc", "--")):
            m = g[col].mean()
            ax.plot(m.index, m.values, color=col_, ls=ls, lw=1.4 if i == 0 else 0.9)
        m, sd = g["val_auc"].mean(), g["val_auc"].std()
        axes[2].plot(m.index, m.values, color=col_, lw=1.4 if i == 0 else 0.9, label=name)
        axes[2].fill_between(m.index, m - sd, m + sd, color=col_, alpha=0.12, lw=0)
        sel = h[h.selected == 1].epoch
        axes[2].scatter(sel.mean(), np.interp(sel.mean(), m.index, m.values), color=col_, s=10, marker="v", zorder=5)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss (solid: training, dashed: validation)")
    axes[1].set(xlabel="Epoch", ylabel="AUC", title="AUC (solid: training, dashed: validation)", ylim=(0.45, 1.01))
    axes[2].set(xlabel="Epoch", ylabel="Validation AUC", title="Validation AUC ± SD (▼ mean selected epoch)")
    for k, ax in enumerate(axes):
        ax.text(-0.18, 1.08, "abc"[k], transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.93))
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
    save(fig, out, "F_learning_curves_compare")


LSO_SITES = ["cobre", "ds000030", "ds001461", "ds005073", "srpbs-KTT", "srpbs-KUT", "srpbs-SWA", "srpbs-UTO"]
SITE_LABEL = {"cobre": "COBRE", "ds000030": "UCLA CNP", "ds001461": "SZ-Smoking", "ds005073": "VSC SZ/BP*",
              "srpbs-KTT": "SRPBS-KTT", "srpbs-KUT": "SRPBS-KUT", "srpbs-SWA": "SRPBS-SWA", "srpbs-UTO": "SRPBS-UTO",
              "external_ds005237": "TCP (external)"}


def site_auc_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return np.nan, (np.nan, np.nan)
    b = [roc_auc_score(y[i], p[i]) for i in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                                             for _ in range(n_boot))]
    return roc_auc_score(y, p), (np.percentile(b, 2.5), np.percentile(b, 97.5))


def lso(models, runs, out):
    """models: {name: run-folder prefix}; the run of held-out site s is <prefix>_<s>."""
    rows = []
    for name, prefix in models.items():
        pooled = []
        for site in LSO_SITES + ["external_ds005237"]:
            f = runs / f"{prefix}_{site}" / PREDICTIONS / "test_predictions.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            if "how" in t.columns:                          # psikoz.stack writes several methods per file
                t = t[t["how"] == "stacking"]
            if site in LSO_SITES:                           # the external site is not part of the pooled LOSO score
                pooled.append(t)
            auc, ci = site_auc_ci(t.y.to_numpy(), t.p1.to_numpy())
            rows.append(dict(Model=name, Site=SITE_LABEL[site], n=len(t), psychosis=int(t.y.sum()), AUC=auc, lo=ci[0], hi=ci[1]))
        if pooled:
            t = pd.concat(pooled)
            auc, ci = site_auc_ci(t.y.to_numpy(), t.p1.to_numpy())
            rows.append(dict(Model=name, Site="Pooled (LOSO)", n=len(t), psychosis=int(t.y.sum()), AUC=auc, lo=ci[0], hi=ci[1]))
    if not rows:
        return
    r = pd.DataFrame(rows)
    table = r.assign(**{"AUC [95% CI]": [fmt(a, ci=(l, h)) if not np.isnan(a) else "— (one class)" for a, l, h in zip(r.AUC, r.lo, r.hi)]})
    wide = table.pivot(index="Site", columns="Model", values="AUC [95% CI]").reindex([s for s in [*list(SITE_LABEL.values())[:-1], "Pooled (LOSO)", SITE_LABEL["external_ds005237"]]
                         if s in set(r.Site)])[list(models)]
    wide.insert(0, "n (psychosis)", r.drop_duplicates("Site").set_index("Site").apply(lambda x: f"{x.n} ({x.psychosis})", axis=1))
    wide = wide.reset_index()
    wide.to_csv(out / "T_lso.csv", index=False)
    (out / "T_lso.md").write_text(markdown(wide) + "\n* VSC SZ/BP (ds005073) has no healthy controls (psychotic vs non-psychotic bipolar).\n",
                                  encoding="utf-8")
    sites = [s for s in [*list(SITE_LABEL.values())[:-1], "Pooled (LOSO)", SITE_LABEL["external_ds005237"]] if s in set(r.Site)]
    fig, ax = plt.subplots(figsize=(3.6, 0.25 * len(sites) + 0.9))
    k = len(models)
    for i, name in enumerate(models):
        g = r[r.Model == name].set_index("Site").reindex(sites)
        yy = np.arange(len(sites)) + (i - (k - 1) / 2) * 0.13
        ax.errorbar(g.AUC, yy, xerr=[g.AUC - g.lo, g.hi - g.AUC], fmt="o", ms=2.6 if i else 3.4, color=PALETTE[i],
                    elinewidth=0.6, capsize=0, label=name)
    ax.axvline(0.5, color="#999999", lw=0.5, ls="--")
    for i, s_ in enumerate(sites):
        if s_.startswith("Pooled"):
            ax.axhline(i - 0.5, color="#cccccc", lw=0.5)
            ax.axhline(i + 0.5, color="#cccccc", lw=0.5)
    ax.set_yticks(range(len(sites)), sites); ax.invert_yaxis()
    lo = min(0.2, np.floor(np.nanmin(r.lo) * 10) / 10)
    ax.set(xlabel="AUC on the held-out site (95 % bootstrap CI)", xlim=(lo, 1.0), title="Leave-one-site-out generalisation")
    ax.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    if any(s_.endswith("*") for s_ in sites):
        fig.text(0.01, -0.02, "* No healthy controls: psychotic vs non-psychotic bipolar disorder (n = 30).",
                 fontsize=5.6, va="top")
    save(fig, out, "F_lso_forest")


# ----------------------------------------------------------------------------- single-site vs multi-site
SINGLE_SITES = ["cobre", "ds000030", "ds001461", "srpbs-KTT", "srpbs-KUT", "srpbs-SWA", "srpbs-UTO"]


def site_names(cfg):
    from .torch_data import site_codes
    return {v: k for k, v in site_codes(cfg).items()}


def sites_pooled(models, cfg, out):
    """Multi-site result per site: the pooled 5-fold CV test predictions (one model trained on all sites)
    split by the site of the test participant. Nothing is retrained."""
    names = site_names(cfg)
    rows = []
    for name, df in models.items():
        for code, g in df.groupby("site"):
            site = names.get(code, str(code))
            y, p, d = g.y.to_numpy(), g.p_cal.to_numpy(), g.decision.to_numpy()
            auc, ci = site_auc_ci(y, p)
            m = metrics(y, p, d)
            rows.append(dict(Model=name, Site=SITE_LABEL.get(site, site), site=site, n=len(g), psychosis=int(y.sum()),
                             AUC=auc, lo=ci[0], hi=ci[1], BAcc=m["BAcc"], Sens=m["Sens"], Spec=m["Spec"]))
    r = pd.DataFrame(rows)
    t = r.assign(**{"AUC [95% CI]": [fmt(a, ci=(l, h)) if not np.isnan(a) else "— (one class)"
                                     for a, l, h in zip(r.AUC, r.lo, r.hi)]})
    t = t[["Model", "Site", "n", "psychosis", "AUC [95% CI]", "BAcc", "Sens", "Spec"]].round(3)
    t.to_csv(out / "T_sites_pooled_cv.csv", index=False)
    (out / "T_sites_pooled_cv.md").write_text(markdown(t) + "\n* VSC SZ/BP (ds005073): psychotic vs non-psychotic bipolar only.\n",
                                              encoding="utf-8")
    return r


def single_vs_multi(single, models, pooled_sites, lso_prefix, runs, out):
    """single: {model name: run prefix}; the single-site run of site s is <prefix>_site_<s> (split site_cv_<s>).
    Three settings per site: single-site CV (trained and tested within the site), multi-site pooled CV
    (trained on all sites, scored on that site's test participants), leave-one-site-out (site unseen)."""
    rows = []
    for name, prefix in single.items():
        for site in SINGLE_SITES:
            run = runs / f"{prefix}_site_{site}"
            if not (run / PREDICTIONS / "test_predictions.csv").exists():
                continue
            df = load_run(run)
            per_fold = [roc_auc_score(g.y, g.p_cal) for _, g in df.groupby("fold") if g.y.nunique() == 2]
            auc, ci = site_auc_ci(df.y.to_numpy(), df.p_cal.to_numpy())
            rows.append(dict(Model=name, site=site, Setting="Single-site CV", AUC=auc, lo=ci[0], hi=ci[1],
                             fold_mean=np.mean(per_fold), fold_sd=np.std(per_fold, ddof=1), n=len(df)))
    if not rows:
        return
    for _, r in pooled_sites[pooled_sites.site.isin(SINGLE_SITES)].iterrows():
        if r.Model in single:
            rows.append(dict(Model=r.Model, site=r.site, Setting="Multi-site CV", AUC=r.AUC, lo=r.lo, hi=r.hi, n=r.n))
    if lso_prefix:
        for site in SINGLE_SITES:
            f = runs / f"{lso_prefix}_{site}" / PREDICTIONS / "test_predictions.csv"
            if f.exists():
                t = pd.read_csv(f)
                auc, ci = site_auc_ci(t.y.to_numpy(), t.p1.to_numpy())
                rows.append(dict(Model=list(models)[0], site=site, Setting="Leave-site-out", AUC=auc, lo=ci[0],
                                 hi=ci[1], n=len(t)))
    r = pd.DataFrame(rows)
    for c in ("fold_mean", "fold_sd"):
        if c not in r:
            r[c] = np.nan
    r["Site"] = r.site.map(lambda s: SITE_LABEL.get(s, s))
    cell = [fmt(a, ci=(l, h)) + ("" if pd.isna(m) else f"; folds {m:.3f} ± {s:.3f}")
            for a, l, h, m, s in zip(r.AUC, r.lo, r.hi, r.fold_mean, r.fold_sd)]
    wide = r.assign(cell=cell).pivot_table(index=["Site", "Model"], columns="Setting", values="cell", aggfunc="first")
    wide = wide[[c for c in ("Single-site CV", "Multi-site CV", "Leave-site-out") if c in wide.columns]].reset_index()
    wide.to_csv(out / "T_single_vs_multi.csv", index=False)
    (out / "T_single_vs_multi.md").write_text(markdown(wide), encoding="utf-8")

    settings = [("Single-site CV", "o", "#0072B2"), ("Multi-site CV", "s", "#D55E00"), ("Leave-site-out", "D", "#009E73")]
    final = list(models)[0]
    sites = [s for s in SINGLE_SITES if s in set(r.site)]
    labels = [SITE_LABEL.get(s, s) for s in sites]
    fig = plt.figure(figsize=(7.1, 0.28 * len(sites) + 1.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    for k, (setting, marker, colour) in enumerate(settings):
        g = r[(r.Model == final) & (r.Setting == setting)].set_index("site").reindex(sites)
        if g.AUC.isna().all():
            continue
        yy = np.arange(len(sites)) + (k - 1) * 0.22
        ax.errorbar(g.AUC, yy, xerr=[g.AUC - g.lo, g.hi - g.AUC], fmt=marker, ms=3, color=colour, elinewidth=0.6,
                    capsize=0, label=setting)
    ax.axvline(0.5, color="#999999", lw=0.5, ls="--")
    ax.set_yticks(range(len(sites)), labels); ax.invert_yaxis()
    ax.set(xlabel="AUC (95 % bootstrap CI)", xlim=(0.2, 1.0))
    ax.set_title(f"a  {final}: single-site vs multi-site", loc="left", fontsize=7.2, fontweight="bold")
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.3, columnspacing=1.0)
    ax2 = fig.add_subplot(gs[1])
    names = [n for n in single if n in set(r.Model)]
    a = r[r.Setting == "Single-site CV"].pivot_table(index="site", columns="Model", values="AUC").reindex(index=sites, columns=names)
    b = r[r.Setting == "Multi-site CV"].pivot_table(index="site", columns="Model", values="AUC").reindex(index=sites, columns=names)
    d = b - a
    lim = float(np.nanmax(np.abs(d.values))) if np.isfinite(d.values).any() else 0.1
    im = ax2.imshow(d.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if np.isfinite(d.values[i, j]):
                ax2.text(j, i, f"{d.values[i, j]:+.2f}", ha="center", va="center", fontsize=5.4)
    ax2.set_xticks(range(len(names)), names, rotation=40, ha="right"); ax2.set_yticks(range(len(sites)), labels)
    ax2.spines[:].set_visible(True)
    fig.colorbar(im, ax=ax2, shrink=0.8, label="ΔAUC (multi-site − single-site)")
    ax2.set_title("b  Gain from multi-site training", loc="left", fontsize=7.2, fontweight="bold")
    save(fig, out, "F_single_vs_multi")


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", required=True, help="run folder of the proposed model")
    ap.add_argument("--final-name", default="PsychosisNET")
    ap.add_argument("--compare", nargs="*", default=[], help='"Display name=run folder" for each comparison model')
    ap.add_argument("--lso", nargs="*", default=[], help='"Display name=run prefix" (runs <prefix>_<site>) for the forest plot')
    ap.add_argument("--single", nargs="*", default=[],
                    help='"Display name=run prefix" (runs <prefix>_site_<site>, splits site_cv_<site>) for single-site CV')
    ap.add_argument("--out", default="results")
    ap.add_argument("--predictions", default="", choices=["", "rescored"],
                    help='"rescored" = predictions recomputed from the saved models by psikoz.evaluate')
    cli = ap.parse_args()
    global PREDICTIONS
    PREDICTIONS = cli.predictions
    cfg = load_config()
    runs = cfg["paths"]["data"] / "runs"
    out = cfg["root"] / cli.out
    out.mkdir(parents=True, exist_ok=True)
    folders = {cli.final_name: cli.final, **dict(item.split("=", 1) for item in cli.compare)}
    models = {name: load_run(runs / folder) for name, folder in folders.items()}
    MODEL_COL.update({name: PALETTE[i] for i, name in enumerate(folders)})
    pooled, fold_table = tables(models, out)
    final = models[cli.final_name]
    fig_roc_folds(final, out, cli.final_name)
    fig_roc_compare(models, pooled, out)
    fig_roc_compare_cv(models, out)
    fig_roc_folds_all(models, out)
    fig_confusion_all(models, out)
    fig_cv_metrics(fold_table, out)
    fig_confusion(final, out, cli.final_name)
    for i, (name, df) in enumerate(list(models.items())[1:], 1):
        fig_confusion(df, out, name, fname=f"F_confusion_cmp{i}_{name.replace(' ', '_')}")
    fig_predictions(final, out, cli.final_name)
    fig_calibration(models, out)
    fig_metrics_compare(pooled, out)
    histories = {name: pd.read_csv(runs / folder / "history.csv") for name, folder in folders.items()
                 if (runs / folder / "history.csv").exists()}                    # deep models only
    for i, (name, h) in enumerate(histories.items()):
        fig_learning_curves(h, out, name, "F_learning_curves" if name == cli.final_name
                            else f"F_learning_curves_{name.replace(' ', '_')}")
    if len(histories) > 1:
        fig_learning_curves_compare(histories, out)
    atlas = json.loads((runs / cli.final / "summary.json").read_text())["args"]["atlas"]
    fig_confounds(final, out, cfg, atlas)
    fig_interpret(runs / cli.final, out, cfg)
    if cli.lso:
        lso(dict(item.split("=", 1) for item in cli.lso), runs, out)
    pooled_sites = sites_pooled(models, cfg, out)
    if cli.single:
        lso_prefix = dict(item.split("=", 1) for item in cli.lso).get(cli.final_name) if cli.lso else None
        single_vs_multi(dict(item.split("=", 1) for item in cli.single), models, pooled_sites, lso_prefix, runs, out)
    print((out / "T_performance.md").read_text(encoding="utf-8"))
    print("->", out)


if __name__ == "__main__":
    main()
