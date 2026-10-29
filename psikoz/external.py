"""One-time scoring of the external test site (TCP, ds005237) with a model trained on all other sites.

    python -m psikoz.train <final options> --split external_ds005237 --name final_external
    python -m psikoz.external --run final_external --out results

The run's own test predictions cover the labelled TCP participants (psychosis vs healthy control). This
script additionally scores every TCP participant that passes quality control, including those outside
the classification sample (non-psychotic psychiatric patients and bipolar patients without psychosis
information), with the same seed ensemble, the same Platt calibration and the same decision threshold
(both fitted on the validation part of the external split, never on TCP).

Outputs: external_predictions.csv, T_external.csv/.md (AUC with stratified bootstrap 95 % CI,
sensitivity, specificity; positive rate in every diagnostic group with Wilson 95 % CI) and F_external
(calibrated probabilities per group with the decision threshold, and the ROC curve).
"""
import argparse
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from .config import load_config
from .report import CLASS_COL, PALETTE, legend_box, logit, markdown, save
from .train import build_model


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def group_of(row):
    if row.diagnosis == "HC":
        return "Healthy control"
    if row.psychosis == 1:
        return "Psychosis"
    if row.diagnosis == "BP":
        return "Bipolar, psychosis unknown"
    return "Non-psychotic psychiatric"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="PsychosisNET")
    cli = ap.parse_args()
    cfg = load_config()
    run = cfg["paths"]["data"] / "runs" / cli.run
    out = cfg["root"] / cli.out
    out.mkdir(parents=True, exist_ok=True)
    a = SimpleNamespace(**json.loads((run / "summary.json").read_text())["args"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = pd.read_csv(cfg["paths"]["features"] / a.atlas / "labels.csv")
    networks = pd.factorize(labels["network"])[0]

    index = pd.read_csv(cfg["paths"]["features"] / a.atlas / "index.csv", low_memory=False)
    part = pd.read_csv(cfg["paths"]["manifest"] / "participants.csv", low_memory=False)
    ext = index[index.dataset.isin(cfg.get("external", [])) & (index.task == "rest") & (index.qc_pass == 1)].copy()
    ext = ext.merge(part[["subject", "diagnosis_detail", "active_psychosis"]].rename(columns={"diagnosis_detail": "dx_detail"}),
                    on="subject", how="left")

    def corr(f):
        c = np.tanh(np.load(cfg["root"] / f)["fc"]).astype(np.float32)
        np.fill_diagonal(c, 1.0)
        return c
    x = torch.tensor(np.stack([corr(f) for f in ext.file])).to(device)
    probs = []
    for f in sorted(run.glob("fold0_seed*.pt")) or [run / "fold0.pt"]:
        model = build_model(a, x.shape[-1], 2, networks).to(device)
        model.load_state_dict(torch.load(f, map_location=device))
        model.eval()
        with torch.no_grad():
            probs.append(torch.cat([model(x[i:i + 64]).softmax(-1)[:, 1] for i in range(0, len(x), 64)]).cpu().numpy())
    ext["p1"] = np.mean(probs, 0)

    val = pd.read_csv(run / "val_predictions.csv")
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(val.p1.to_numpy())[:, None], val.y)
    cal = lambda p: platt.predict_proba(logit(p)[:, None])[:, 1]
    fpr, tpr, thr = roc_curve(val.y, cal(val.p1.to_numpy()))
    threshold = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
    ext["p_cal"] = cal(ext.p1.to_numpy())
    ext["decision"] = (ext.p_cal > threshold).astype(int)
    ext["group"] = ext.apply(group_of, axis=1)
    ext.drop(columns=[c for c in ext.columns if c not in {"subject", "dataset", "diagnosis", "dx_detail", "psychosis",
                                                          "active_psychosis", "age", "sex", "fd_mean", "p1", "p_cal",
                                                          "decision", "group"}]).to_csv(out / "external_predictions.csv", index=False)

    rng = np.random.default_rng(0)

    def auc_row(lab, title):
        y, p = (lab.group == "Psychosis").astype(int).to_numpy(), lab.p_cal.to_numpy()
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        boots = [roc_auc_score(y[i], p[i]) for i in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                                                     for _ in range(2000))]
        auc = roc_auc_score(y, p)
        row = dict(Analysis=title, n=f"{len(pos)} / {len(neg)}",
                   Result=f"AUC {auc:.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]",
                   Note=f"sensitivity {lab[lab.group == 'Psychosis'].decision.mean():.2f}, "
                        f"specificity {1 - lab[lab.group == 'Healthy control'].decision.mean():.2f} (threshold {threshold:.2f})")
        return row, y, p, auc, boots

    # Primary: the classification definition of the training sites (HC vs psychotic SZ / BP), i.e. the
    # external run's own test set. Secondary: also participants with another primary diagnosis whose
    # record flags psychosis (e.g. MDD with psychotic features), which the training label never covered.
    both = ext[ext.group.isin(["Healthy control", "Psychosis"])]
    primary, y, p, auc, boots = auc_row(both[both.diagnosis.isin(["HC", "SZ", "BP"])],
                                        "Psychosis (SZ / BP) vs healthy control [primary]")
    secondary = auc_row(both, "Psychosis (any primary diagnosis) vs healthy control [secondary]")[0]
    rows = [primary, secondary]
    for g in ["Healthy control", "Psychosis", "Non-psychotic psychiatric", "Bipolar, psychosis unknown"]:
        sub = ext[ext.group == g]
        k, n = int(sub.decision.sum()), len(sub)
        lo, hi = wilson(k, n)
        rows.append(dict(Analysis=f"Classified as psychosis: {g}", n=str(n),
                         Result=f"{k}/{n} = {k / max(n, 1):.0%} [{lo:.0%}, {hi:.0%}]", Note=""))
    other = ext[ext.group == "Non-psychotic psychiatric"]
    for dx, sub in other.groupby(other.dx_detail.str.replace(r"^(past |Past )", "", regex=True).str.strip()):
        if len(sub) >= 5:
            rows.append(dict(Analysis=f"  of which primary diagnosis {dx}", n=str(len(sub)),
                             Result=f"{int(sub.decision.sum())}/{len(sub)} = {sub.decision.mean():.0%}", Note=""))
    t = pd.DataFrame(rows)
    t.to_csv(out / "T_external.csv", index=False)
    (out / "T_external.md").write_text(markdown(t), encoding="utf-8")
    print(markdown(t))

    order = ["Healthy control", "Non-psychotic psychiatric", "Bipolar, psychosis unknown", "Psychosis"]
    colours = [CLASS_COL[0], "#56B4E9", "#CC79A7", CLASS_COL[1]]
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(5.6, 2.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    for i, (g, c) in enumerate(zip(order, colours)):
        v = ext[ext.group == g].p_cal.to_numpy()
        if not len(v):
            continue
        parts = ax.violinplot(v, positions=[i], widths=0.75, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(c); b.set_alpha(0.25); b.set_edgecolor("none")
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=3, color=c, lw=0, alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#1a1a1a", lw=1)
        ax.text(i, 1.04, f"n = {len(v)}\n{(v > threshold).mean():.0%} positive", ha="center", va="bottom", fontsize=5.4)
    ax.axhline(threshold, color="#444444", ls="--", lw=0.6)
    ax.set_xticks(range(len(order)), ["Healthy\ncontrol", "Non-psychotic\npsychiatric", "Bipolar\n(psychosis\nunknown)",
                                      "Psychosis\n(any\ndiagnosis)"])
    ax.set(ylabel="Predicted probability of psychosis", ylim=(-0.02, 1.0))
    axr.set_title(f"Psychosis (SZ / BP, n = {int(y.sum())}) vs\nhealthy control (n = {int((y == 0).sum())})", fontsize=6.6)
    f, tp_, _ = roc_curve(y, p)
    axr.plot(f, tp_, color=PALETTE[0], lw=1.6, label=f"AUC {auc:.3f}\n[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
    axr.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#888888")
    axr.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    axr.set_aspect("equal"); legend_box(axr.legend(loc="lower right"))
    fig.suptitle(f"{cli.name} on the independent external site (TCP): never used for training or selection",
                 fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "F_external")
    print("->", out)


if __name__ == "__main__":
    main()
