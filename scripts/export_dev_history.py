"""Development history of PsychosisNET (Supplementary Table S10).

Reads data/runs/<run>/summary.json, folds.csv and test_predictions.csv for every development run
(v2 ... v13) and the final model, and writes
    results/T_dev_history.csv, results/T_dev_history.md      (full precision / readable copy)
    paper/latex/assets/TableS10_dev_history.tex              (supplementary longtable)
Only the "change" column is written by hand; it restates the argument differences between
successive runs recorded in summary.json (see DIFF below). Every number is read from the run files.

    .venv/Scripts/python scripts/export_dev_history.py
"""
import datetime
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
FINAL = "final_psychosisnet"

# run -> (stage, change relative to the preceding run of the same stage, as recorded in summary.json args)
DIFF = {
    "v2_depth4": ("v2", "Transformer on fixed tangent-space input, depth 4"),
    "v2_depth4_skip": ("v2", "+ wide linear path"),
    "v2_depth4_drop05": ("v2", "stronger dropout (0.5 / path 0.3 / edge 0.4), no wide path"),
    "v2_depth6_drop05_skip": ("v2", "depth 6, stronger dropout, wide path"),
    "v3_diff_skip": ("v3", "differential attention, wide path"),
    "v3_spd_skip": ("v3", "learnable SPD tangent layer, wide path"),
    "v3_spd_diff_skip": ("v3", "SPD layer + differential attention, wide path"),
    "v3_spd_diff": ("v3", "SPD layer + differential attention, no wide path"),
    "v4_rank20": ("v4", "low-rank (20) wide path, fixed tangent input"),
    "v4_rank10_drop05": ("v4", "low-rank (10) wide path, stronger dropout"),
    "v4_rank10": ("v4", "low-rank (10) wide path"),
    "v4_spd_rank10": ("v4", "SPD layer + low-rank (10) wide path"),
    "v3_spd_skip_szhc": ("v3", "SPD + wide path; SZ vs HC, COBRE and UCLA CNP only"),
    "v5_spd_skip_szhc_3site": ("v5", "as above + SZ-Smoking (3 datasets)"),
    "v6_spd_skip_szhc_7site": ("v6", "as above + SRPBS sites (SZ vs HC)"),
    "v7_spd_skip_psychosis_8site": ("v7", "psychosis label, HC/SZ/BP, all classification sites"),
    "v8_base": ("v8", "60 epochs, patience 15, one seed"),
    "v8_delay": ("v8", "+ wide path frozen 10 epochs, learning-rate scale 0.1"),
    "v8_siteadv": ("v8", "+ site-adversarial head (0.1)"),
    "v8_views": ("v8", "+ band-limited and lagged connectivity views"),
    "v8_aux": ("v8", "+ auxiliary age/sex head (0.1)"),
    "v8_all": ("v8", "delay + site-adversarial + views + auxiliary head"),
    "v8_ens5": ("v8", "five-seed ensemble"),
    "v9_pre_lr1e-3": ("v9", "+ self-supervised pretrained encoder, lr 1e-3"),
    "v9_pre_lr3e-4": ("v9", "pretrained encoder, lr 3e-4"),
    "v10_e2e": ("v10", "+ edge-to-edge layer (8 filters)"),
    "v10_ocread": ("v10", "+ orthonormal clustering readout (4 clusters)"),
    "v10_e2e_ocread": ("v10", "+ edge-to-edge layer and clustering readout"),
    "v11_l2_1e-5": ("v11", "simplified code; Adam + explicit L2 1e-5, re-pretrained encoder"),
    "v11_l2_1e-4": ("v11", "Adam + explicit L2 1e-4"),
    "v11_l2_1e-3": ("v11", "Adam + explicit L2 1e-3"),
    "v13_base": ("v13", "simplified code, AdamW (weight decay 0.05), pretrained encoder"),
    "v13_small": ("v13", "+ smaller encoder (depth 2, dim 32), own pretraining"),
    "v13_freeze": ("v13", "+ frozen SPD congruence matrix"),
    "v13_ema": ("v13", "+ weight moving average (0.99)"),
    "v13_crop": ("v13", "+ random-segment cropping (0.8--1 of the run)"),
    FINAL: ("final", "v13_crop configuration, final run used for all reported results"),
}


def test_keys(run):
    d = pd.read_csv(RUNS / run / "test_predictions.csv")
    return set(zip(d["fold"], d["subject"]))


def main():
    final_keys = test_keys(FINAL)
    rows = []
    for run, (stage, change) in DIFF.items():
        s = json.loads((RUNS / run / "summary.json").read_text())
        a, m = s["args"], s["summary"]
        folds = pd.read_csv(RUNS / run / "folds.csv")
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(RUNS / run / "summary.json"))
        rows.append(dict(
            stage=stage, run=run, finished=mtime.strftime("%Y-%m-%d"), change=change,
            label="SZ vs HC" if a.get("diagnoses") == ["HC", "SZ"] else "psychosis (HC/SZ/BP)",
            n_test=int(folds["n_test"].sum()), n_folds=len(folds),
            same_folds_as_final=test_keys(run) == final_keys,
            seeds=a.get("seeds", 1) or 1, pretrained=bool(a.get("pretrained")),
            val_auc_mean=m["val_auc"][0], val_auc_sd=m["val_auc"][1],
            test_auc_mean=m["test_auc"][0], test_auc_sd=m["test_auc"][1]))
    t = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    t.to_csv(ROOT / "results" / "T_dev_history.csv", index=False)

    fmt = lambda mu, sd: f"{mu:.3f} $\\pm$ {sd:.3f}"
    md = ["| Stage | Run | Finished | Change | Label | n test | Same folds as final | Seeds | Pretrained | Val AUC | Test AUC |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [r"% Generated by scripts/export_dev_history.py from data/runs/*/summary.json; do not edit by hand.",
           r"\begingroup\renewcommand{\theHtable}{TableS10}",
           r"\setcounter{table}{9}",
           r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.05\textwidth}>{\raggedright\arraybackslash}p{0.33\textwidth}>{\raggedright\arraybackslash}p{0.07\textwidth}>{\raggedright\arraybackslash}p{0.06\textwidth}>{\raggedright\arraybackslash}p{0.05\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}}",
           r"\caption{Development history of PsychosisNET. Each row is one recorded development run on the five development cross-validation folds, in order of completion; the change column summarises the configuration or modification tested in that run. Label: Psy., psychosis versus no psychosis; SZ/HC, schizophrenia versus healthy controls. $n$ is the number of test-part participants summed over folds; ``Same'' indicates that the fold and participant assignment of the test parts is identical to that of the final model. AUCs are means $\pm$ SD over the five folds. Validation and test-part AUCs were both recorded for every run; test-part values were therefore visible during development.}\label{tab:S10_dev_history}\\",
           r"\toprule",
           r"Stage & Change & Label & $n$ & Same & Validation AUC & Test-part AUC\\\midrule\endfirsthead",
           r"\toprule Stage & Change & Label & $n$ & Same & Validation AUC & Test-part AUC\\\midrule\endhead"]
    for r in t.itertuples():
        md.append(f"| {r.stage} | {r.run} | {r.finished} | {r.change} | {r.label} | {r.n_test} | "
                  f"{'yes' if r.same_folds_as_final else 'no'} | {r.seeds} | {'yes' if r.pretrained else 'no'} | "
                  f"{r.val_auc_mean:.3f} ± {r.val_auc_sd:.3f} | {r.test_auc_mean:.3f} ± {r.test_auc_sd:.3f} |")
        change = r.change.replace("_", r"\_")
        if r.seeds > 1:
            change += f"; {r.seeds} seeds"
        label = "SZ/HC" if r.label == "SZ vs HC" else "Psy."
        tex.append(f"{r.stage} & {change} & {label} & {r.n_test} & {'yes' if r.same_folds_as_final else 'no'} & "
                   f"{fmt(r.val_auc_mean, r.val_auc_sd)} & {fmt(r.test_auc_mean, r.test_auc_sd)}\\\\")
    tex += [r"\bottomrule\end{longtable}\endgroup"]
    (ROOT / "results" / "T_dev_history.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "paper" / "latex" / "assets" / "TableS10_dev_history.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(t[["stage", "run", "n_test", "same_folds_as_final", "val_auc_mean", "test_auc_mean"]].to_string())


if __name__ == "__main__":
    main()
