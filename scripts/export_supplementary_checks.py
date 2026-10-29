"""Values referred to in the manuscript that were saved by the pipeline but not exported as tables (29 Oct 2026).

    results/T_lr_choice           mean ± SD validation AUC of the deep comparison models at the two learning rates
                                  of run_final.sh stage 2 (pooled CV) and the rate chosen (higher validation AUC;
                                  test AUC not used), read from data/runs/cmp_<model>_lr<lr>/summary.json
    results/T_gradcam_checks      Grad-CAM randomisation check (Spearman correlation between the mean map over all
                                  test-part participants, trained ensemble vs re-initialised model, per fold) and fold consistency
                                  (Spearman correlations between the five folds' psychosis group maps), read from
                                  data/runs/final_psychosisnet/interpret/{sanity,consistency}.csv (psikoz.interpret)
    results/T_network_ig          mean Integrated Gradients attribution per network pair (7 Yeo networks +
                                  subcortex), read from data/runs/final_psychosisnet/interpret/network_ig.csv

    .venv/Scripts/python scripts/export_supplementary_checks.py [runs folder, default data/runs]
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psikoz.report import markdown  # noqa: E402

MODELS = {"mlp": "MLP", "brainnetcnn": "BrainNetCNN", "bnt": "BNT"}


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "runs"
    out = ROOT / "results"

    rows = []
    for key, name in MODELS.items():
        val = {lr: json.load(open(runs / f"cmp_{key}_lr{lr}" / "summary.json"))["summary"]["val_auc"] for lr in ("1e-3", "1e-4")}
        best = max(val, key=lambda lr: val[lr][0])
        for lr, (m, s) in val.items():
            rows.append(dict(Model=name, learning_rate=lr, val_auc_mean=m, val_auc_sd=s, chosen=lr == best))
    lr = pd.DataFrame(rows)
    lr.to_csv(out / "T_lr_choice.csv", index=False)
    show = lr.assign(**{"Validation AUC (fold mean ± SD)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(lr.val_auc_mean, lr.val_auc_sd)],
                        "Chosen": lr.chosen.map({True: "yes", False: ""})})[["Model", "learning_rate", "Validation AUC (fold mean ± SD)", "Chosen"]]
    (out / "T_lr_choice.md").write_text(markdown(show.rename(columns={"learning_rate": "Learning rate"})) +
                                        "\nPooled 5-fold CV, five seeds per fold; the rate with the higher mean validation AUC "
                                        "was used for all evaluations (test AUC not used).\n", encoding="utf-8")

    interp = runs / "final_psychosisnet" / "interpret"
    sanity = pd.read_csv(interp / "sanity.csv")
    cons = pd.read_csv(interp / "consistency.csv")
    checks = pd.DataFrame([dict(Check="Randomisation check: trained vs re-initialised model, per fold",
                                n=len(sanity), mean=sanity.spearman_trained_vs_random.mean(),
                                min=sanity.spearman_trained_vs_random.min(), max=sanity.spearman_trained_vs_random.max()),
                           dict(Check="Fold consistency: pairs of folds", n=len(cons), mean=cons.spearman.mean(),
                                min=cons.spearman.min(), max=cons.spearman.max())])
    checks.to_csv(out / "T_gradcam_checks.csv", index=False)
    show = checks.copy()
    for c in ("mean", "min", "max"):
        show[c] = show[c].map("{:.2f}".format)
    (out / "T_gradcam_checks.md").write_text(
        markdown(show) + "\nSpearman correlations over the 114 regions. Randomisation check: mean Grad-CAM map over all "
        "test-part participants of a fold, trained five-seed ensemble vs re-initialised model (psikoz/interpret.py). "
        "Fold consistency: psychosis group maps of pairs of folds. Per-fold randomisation values: " + ", ".join(f"{v:.2f}" for v in sanity.spearman_trained_vs_random) + ".\n", encoding="utf-8")

    ig = pd.read_csv(interp / "network_ig.csv", index_col=0)
    ig.to_csv(out / "T_network_ig.csv")
    (out / "T_network_ig.md").write_text(
        markdown(ig.map(lambda v: f"{v:.2e}").reset_index().rename(columns={"index": "Network"})) +
        "\nMean Integrated Gradients attribution of the full model's psychosis log-odds per network pair "
        "(symmetric; development test folds).\n", encoding="utf-8")
    for f in ("T_lr_choice", "T_gradcam_checks", "T_network_ig"):
        print((out / f"{f}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
