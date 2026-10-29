"""Ablation of self-supervised pretraining (PsychosisNET with vs without the pretrained encoder).

Pairs data/runs/final_psychosisnet* with data/runs/nopre_psychosisnet* participant by participant and compares the
AUCs with the paired DeLong test (report.delong); Holm correction over all rows; class-stratified bootstrap
95 % CI of each AUC (report.site_auc_ci, 2000 draws, seed 0). Pooled CV on calibrated probabilities as Tables 2-3. Writes
results/T_ablation_pretraining.csv / .md and the figure results/F_ablation_pretraining.svg / .pdf / .png.

    .venv/Scripts/python scripts/ablation_pretraining.py [runs folder, default data/runs]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from psikoz.report import PALETTE, delong, load_run, markdown, save, site_auc_ci  # noqa: E402

SITES = {"cobre": "COBRE", "ds000030": "UCLA CNP", "ds001461": "SZ-Smoking", "ds005073": "VSC SZ/BP*",
         "srpbs-KTT": "SRPBS-KTT", "srpbs-KUT": "SRPBS-KUT", "srpbs-SWA": "SRPBS-SWA", "srpbs-UTO": "SRPBS-UTO"}  # Table 4 order
IN_PRETRAINING = {"srpbs-UTO", "srpbs-KUT", "srpbs-SWA", "ds000030"}   # other participants of these sites were pretrained on


def pair(runs, suffix, calibrated=False):
    """calibrated=True: per-fold Platt-calibrated probabilities (report.load_run), as in Tables 2 and 3 for the pooled
    CV; single-fold LOSO / TCP runs use the raw probabilities, as in Tables 4 and 5 (same AUC within one fold)."""
    if calibrated:
        pre, nopre = (load_run(runs / f"{prefix}_psychosisnet{suffix}").drop(columns="p1").rename(columns={"p_cal": "p1"})
                      for prefix in ("final", "nopre"))
    else:
        pre, nopre = (pd.read_csv(runs / f"{prefix}_psychosisnet{suffix}" / "test_predictions.csv")
                      for prefix in ("final", "nopre"))
    m = pre.merge(nopre, on="subject", suffixes=("_pre", "_nopre"))
    assert len(m) == len(pre) == len(nopre) and (m.y_pre == m.y_nopre).all()
    return m


def row(name, m, note=""):
    y = m.y_pre.values
    diff, z, p = delong(y, m.p1_pre.values, m.p1_nopre.values)
    a1, (l1, h1) = site_auc_ci(y, m.p1_pre.values)
    a0, (l0, h0) = site_auc_ci(y, m.p1_nopre.values)
    return dict(Evaluation=name, n=len(m), psychosis=int(y.sum()), AUC_pretrained=a1, lo_pretrained=l1, hi_pretrained=h1,
                AUC_no_pretraining=a0, lo_no_pretraining=l0, hi_no_pretraining=h0, dAUC=diff, z=z, p=p, note=note)


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "runs"
    rows = [row("Pooled 5-fold CV", pair(runs, "", calibrated=True))]
    lso = []
    for s, label in SITES.items():
        m = pair(runs, f"_{s}")
        lso.append(m)
        rows.append(row(f"LOSO {label}", m, "site also in pretraining (other participants)" if s in IN_PRETRAINING else ""))
    rows.append(row("LOSO pooled (8 sites)", pd.concat(lso)))
    rows.append(row("TCP external (psychosis vs HC)", pair(runs, "_external_ds005237")))
    df = pd.DataFrame(rows)
    order, adj, running = np.argsort(df.p.values), np.empty(len(df)), 0.0
    for k, i in enumerate(order):
        running = max(running, (len(df) - k) * df.p.values[i])
        adj[i] = min(1.0, running)
    df.insert(df.columns.get_loc("note"), "p_Holm", adj)
    out = ROOT / "results"
    df.to_csv(out / "T_ablation_pretraining.csv", index=False)
    ci = lambda a, lo, hi: f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
    show = pd.DataFrame({
        "Evaluation": df.Evaluation, "n (psychosis)": [f"{n} ({k})" for n, k in zip(df.n, df.psychosis)],
        "Pretrained": [ci(*r) for r in df[["AUC_pretrained", "lo_pretrained", "hi_pretrained"]].values],
        "No pretraining": [ci(*r) for r in df[["AUC_no_pretraining", "lo_no_pretraining", "hi_no_pretraining"]].values],
        "ΔAUC": df.dAUC.map("{:+.3f}".format), "z": df.z.map("{:.2f}".format), "p": df.p.map("{:.2g}".format),
        "p (Holm)": df.p_Holm.map("{:.2g}".format), "Note": df.note})
    (out / "T_ablation_pretraining.md").write_text(
        markdown(show) + "\nAUC [95 % class-stratified bootstrap CI]; ΔAUC = pretrained − no pretraining; paired DeLong "
        "test; Holm correction over all 11 rows.\n* VSC SZ/BP (ds005073) has no healthy controls (psychotic vs "
        "non-psychotic bipolar).\n", encoding="utf-8")
    print(markdown(show))
    figure(df, out)


def figure(df, out):
    labels = [(e if e.startswith("LOSO pooled") else e.replace("LOSO ", "LOSO, ")) + (" †" if n else "")
              for e, n in zip(df.Evaluation, df.note)]
    fig, ax = plt.subplots(figsize=(4.4, 0.25 * len(df) + 0.95))
    y = np.arange(len(df))
    col = PALETTE[0]                                                    # PsychosisNET colour in every figure
    for k, (key, face, lab) in enumerate((("pretrained", col, "PsychosisNET, pretrained encoder (reported model)"),
                                          ("no_pretraining", "white", "PsychosisNET, no pretraining"))):
        a = df[f"AUC_{key}"].values
        ax.errorbar(a, y + (k - 0.5) * 0.26, xerr=[a - df[f"lo_{key}"].values, df[f"hi_{key}"].values - a], fmt="o",
                    ms=3.2, color=col, mfc=face, mec=col, mew=0.8, elinewidth=0.6, capsize=0, label=lab)
    ax.axvline(0.5, color="#999999", lw=0.5, ls="--")
    for i, e in enumerate(df.Evaluation):                               # development | LOSO sites | pooled | external
        if i == 1 or e.startswith(("LOSO pooled", "TCP")):
            ax.axhline(i - 0.5, color="#cccccc", lw=0.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    lo = min(0.2, np.floor(df[["lo_pretrained", "lo_no_pretraining"]].min().min() * 10) / 10)
    ax.set(xlabel="AUC (95 % bootstrap CI)", xlim=(lo, 1.0))
    ax.set_title("Effect of self-supervised pretraining", pad=16)
    tr = ax.get_yaxis_transform()
    ax.text(1.03, -0.9, "ΔAUC", transform=tr, va="center", fontsize=5.8, fontweight="bold")
    ax.text(1.17, -0.9, "p (Holm)", transform=tr, va="center", fontsize=5.8, fontweight="bold")
    for i, (d, p) in enumerate(zip(df.dAUC, df.p_Holm)):
        ax.text(1.03, i, f"{d:+.3f}", transform=tr, va="center", fontsize=5.8)
        ax.text(1.17, i, f"{p:.2g}", transform=tr, va="center", fontsize=5.8)
    ax.legend(fontsize=5.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
    ax.text(-0.38, -0.145, "† Other participants of this site were in the unlabelled pretraining set.  * No healthy controls "
            "(n = 30).\nΔAUC = pretrained − no pretraining; paired DeLong test, Holm-corrected over 11 comparisons.",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.4)
    save(fig, out, "F_ablation_pretraining")


if __name__ == "__main__":
    main()
