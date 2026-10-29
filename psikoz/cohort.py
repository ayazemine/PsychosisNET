"""Sample description for the manuscript: demographics, acquisition parameters and quality control.

    python -m psikoz.cohort --out results

T_demographics   per site and group (no psychosis / psychosis) of the classification sample: n, age,
                 female, mean FD, clean scan time; group tests per site (Welch t for age, chi-square for
                 sex, Mann-Whitney U for FD and clean time) and over all sites
T_acquisition    per site: scanner, TR, volumes, voxel size, echo time and scan duration of the
                 resting-state run, read from the raw images and BIDS sidecars ("—" when not available;
                 scanner names of COBRE, UCLA CNP and SRPBS from their documentation)
F_qc             brain-mask Dice of the registration and mean FD per site, classification sample
The external test site (config "external") is described in separate rows when it has been processed.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config, main_rows
from .report import CLASS_COL, PALETTE, markdown, save

SCANNER_DOC = {"cobre": "Siemens TIM Trio 3T", "ds000030": "Siemens Trio 3T", "srpbs-KTT": "Siemens Trio 3T", "srpbs-KUT": "Siemens TimTrio 3T",
               "srpbs-SWA": "Siemens Verio 3T", "srpbs-UTO": "GE Discovery MR750w 3T"}
SITE_NAME = {"cobre": "COBRE", "ds000030": "UCLA CNP", "ds001461": "SZ-Smoking", "ds005073": "VSC SZ/BP",
             "srpbs-KTT": "SRPBS-KTT", "srpbs-KUT": "SRPBS-KUT", "srpbs-SWA": "SRPBS-SWA", "srpbs-UTO": "SRPBS-UTO",
             "ds005237": "TCP (external)"}


def sample(cfg, atlas):
    index = pd.read_csv(cfg["paths"]["features"] / atlas / "index.csv", low_memory=False)
    s = main_rows(index[index.task == "rest"])
    return s[(s.qc_pass == 1) & s.diagnosis.isin(["HC", "SZ", "BP"]) & s.psychosis.notna()].copy()


def ms(v):
    v = pd.to_numeric(v, errors="coerce").dropna()
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) else "—"


def demographics(s, external):
    rows = []
    def block(name, g):
        a, b = g[g.psychosis == 0], g[g.psychosis == 1]
        row = {"Site": name}
        for lab, part in (("No psychosis", a), ("Psychosis", b)):
            row[f"n ({lab})"] = len(part)
            row[f"Age ({lab})"] = ms(part.age)
            row[f"Female, n (%) ({lab})"] = f"{(part.sex == 'F').sum()} ({100 * (part.sex == 'F').mean():.0f})" if len(part) else "—"
            row[f"FD, mm ({lab})"] = (f"{part.fd_mean.mean():.2f} ± {part.fd_mean.std():.2f}" if len(part) else "—")
            row[f"Clean time, s ({lab})"] = ms(part.clean_seconds)
        if len(a) > 1 and len(b) > 1:
            row["p age"] = f"{stats.ttest_ind(a.age.dropna(), b.age.dropna(), equal_var=False).pvalue:.2g}"
            tab = pd.crosstab(g.psychosis, g.sex == "F")
            row["p sex"] = f"{stats.chi2_contingency(tab)[1]:.2g}" if tab.shape == (2, 2) else "—"
            row["p FD"] = f"{stats.mannwhitneyu(a.fd_mean, b.fd_mean).pvalue:.2g}"
            row["p clean time"] = f"{stats.mannwhitneyu(a.clean_seconds, b.clean_seconds).pvalue:.2g}"
        else:
            row.update({"p age": "—", "p sex": "—", "p FD": "—", "p clean time": "—"})
        rows.append(row)
    for d, g in s.groupby("dataset", sort=False):
        block(SITE_NAME.get(d, d), g)
    block("All sites", s)
    if external is not None and len(external):
        block(SITE_NAME.get(external.dataset.iloc[0]), external)
    return pd.DataFrame(rows)


def acquisition(cfg, datasets):
    runs = pd.read_csv(cfg["paths"]["manifest"] / "runs.csv")
    rows = []
    for d in datasets:
        r = runs[(runs.dataset == d) & (runs.task == "rest")]
        first = r.iloc[0]
        img = nib.load(str(cfg["root"] / first.bold))
        side = (cfg["root"] / first.bold).with_suffix("").with_suffix(".json")
        meta = json.loads(side.read_text()) if side.exists() else {}
        scanner = " ".join(str(meta[k]) for k in ("Manufacturer", "ManufacturersModelName") if k in meta)
        if "MagneticFieldStrength" in meta:
            scanner += f" {meta['MagneticFieldStrength']:g}T"
        te = meta.get("EchoTime")
        zooms = img.header.get_zooms()[:3]
        tr = float(r.tr.median()); vols = int(r.n_vols.median())
        rows.append({"Site": SITE_NAME.get(d, d), "Scanner": SCANNER_DOC.get(d) or scanner.strip() or "—",
                     "TR (s)": f"{tr:g}", "TE (ms)": f"{te * 1000:.0f}" if te else "—",
                     "Volumes": vols, "Duration (min)": f"{tr * vols / 60:.1f}",
                     "Voxel (mm)": "—" if d == "cobre" else " × ".join(f"{z:.2f}".rstrip("0").rstrip(".") for z in zooms),
                     "Image format": "NIAK-preprocessed (MNI)" if d == "cobre" else "raw"})
    return pd.DataFrame(rows)


def fig_qc(cfg, s, out):
    dice = []
    for d in s.dataset.unique():
        f = cfg["paths"]["preproc"] / d / "qc.csv"
        if f.exists():
            q = pd.read_csv(f)
            q["subject"] = d + "_" + q.participant_id
            dice.append(q[q.subject.isin(s.subject)].assign(dataset=d))
    dice = pd.concat(dice) if dice else pd.DataFrame(columns=["dataset", "brainmask_dice"])
    sites = list(dict.fromkeys(s.dataset))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    rng = np.random.default_rng(0)
    panels = ((dice, "brainmask_dice", "Registration: brain-mask Dice", None),
              (s, "fd_mean", "Mean framewise displacement (mm)", "psychosis"),
              (s, "clean_seconds", "Clean scan time (s)", "psychosis"))
    for ax, (df, col, lab, split) in zip(axes, panels):
        for i, d in enumerate(sites):
            g = df[df.dataset == d]
            if not len(g):
                continue
            if split:
                for k, dx in ((0, -0.18), (1, 0.18)):
                    v = g[g[split] == k][col].dropna()
                    ax.scatter(i + dx + rng.uniform(-0.08, 0.08, len(v)), v, s=1.5, color=CLASS_COL[k], alpha=0.5, lw=0)
                    ax.plot([i + dx - 0.12, i + dx + 0.12], [v.median()] * 2, color="#1a1a1a", lw=0.9)
            else:
                v = g[col].dropna()
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=1.5, color=PALETTE[1], alpha=0.5, lw=0)
                ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="#1a1a1a", lw=0.9)
        ax.set_xticks(range(len(sites)), [SITE_NAME.get(d, d) for d in sites], rotation=45, ha="right")
        ax.set_ylabel(lab)
    axes[1].scatter([], [], s=8, color=CLASS_COL[0], label="No psychosis"); axes[1].scatter([], [], s=8, color=CLASS_COL[1], label="Psychosis")
    axes[1].legend(fontsize=5.5, loc="upper right")
    axes[0].set_title("Registration quality (COBRE: released preprocessed, not shown)", fontsize=6)
    axes[1].set_title("Head motion", fontsize=6); axes[2].set_title("Data retained after censoring", fontsize=6)
    fig.tight_layout()
    save(fig, out, "F_qc")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", default="schaefer100s_gsr")
    ap.add_argument("--out", default="results")
    cli = ap.parse_args()
    cfg = load_config()
    out = cfg["root"] / cli.out
    out.mkdir(parents=True, exist_ok=True)
    s = sample(cfg, cli.atlas)
    ext_sites = cfg.get("external", [])
    main_s, external = s[~s.dataset.isin(ext_sites)], s[s.dataset.isin(ext_sites)]
    for name, df in (("T_demographics", demographics(main_s, external)),
                     ("T_acquisition", acquisition(cfg, list(dict.fromkeys(main_s.dataset)) +
                                                   [d for d in ext_sites if (cfg["paths"]["manifest"] / "runs.csv").exists()
                                                    and d in set(pd.read_csv(cfg["paths"]["manifest"] / "runs.csv").dataset)]))):
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(markdown(df), encoding="utf-8")
        print(markdown(df))
    fig_qc(cfg, main_s, out)
    print("->", out)


if __name__ == "__main__":
    main()
