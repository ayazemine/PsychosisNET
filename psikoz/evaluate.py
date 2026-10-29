"""Rescoring of a finished run from its saved models only: nothing is trained or refitted.

    python -m psikoz.evaluate --runs final_psychosisnet cmp_svm cmp_xgboost cmp_mlp_lr1e-3 ...

For every fold the saved models are loaded and applied to the train / validation / test parts of the
same split (the split file and the run's summary.json fix the participants):
  PsychosisNET         fold<k>_seed<s>.pt (the SPD reference of the training part is a saved buffer)
  MLP, BrainNetCNN, BNT fold<k>_seed<s>.pt
  SVM, XGBoost         fold<k>_seed<s>.joblib (saved tangent reference and scale, feature scaler, classifier)
Seed probabilities are averaged per participant, as in training.

Outputs in data/runs/<run>/rescored/: test_ / val_ / train_predictions.csv (same columns as the run's own
files) and check.json (largest absolute difference to the stored predictions; ~1e-6 = identical up to
GPU arithmetic). psikoz.report --predictions rescored then builds every figure and table from these.
"""
import argparse
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baselines import DEEP, prob, vectors
from .config import load_config
from .harmonize import apply_tangent, correlation_loaders
from .torch_data import make_loaders
from .train import build_model, predict

PARTS = ("train", "val", "test")


def frame(fold, p, y, sub, site):
    return pd.DataFrame(dict(fold=fold, subject=sub, site=site, y=y, **{f"p{c}": p[:, c] for c in range(p.shape[1])}))


def average(runs):
    """Mean probability per participant over seeds (aligned on the subject id)."""
    ref = runs[0]
    order = [{s: i for i, s in enumerate(r[2])} for r in runs]
    p = np.mean([r[0][[o[s] for s in ref[2]]] for r, o in zip(runs, order)], 0)
    return (p,) + tuple(ref[1:])


def rescore(run, cfg, device):
    a = SimpleNamespace(**json.loads((run / "summary.json").read_text())["args"])
    kind = getattr(a, "model", "psychosisnet")
    n_folds = len(json.loads((cfg["paths"]["splits"] / f"{a.split}.json").read_text())["folds"])
    networks = pd.factorize(pd.read_csv(cfg["paths"]["features"] / a.atlas / "labels.csv")["network"])[0]
    out = {part: [] for part in PARTS}
    for fold in range(n_folds):
        loaders = make_loaders(split=a.split, fold=fold, atlas=a.atlas, label=a.label, input="fc",
                               diagnoses=a.diagnoses, datasets=getattr(a, "datasets", None), cfg=cfg)
        if kind in ("svm", "xgboost"):
            files = sorted(run.glob(f"fold{fold}_seed*.joblib"))
            if not files:
                raise FileNotFoundError(f"{run.name}: no saved classical model for fold {fold} "
                                        "(runs made before model saving was added must be retrained)")
            runs = []
            for f in files:
                saved = joblib.load(f)
                apply_tangent(loaders, saved["tangent"], saved["scale"])   # same reference for every seed
                r = {}
                for part in PARTS:
                    x, y, sub, site = vectors(loaders[part].dataset)
                    p = prob(saved["model"], saved["scaler"].transform(x))
                    r[part] = (np.c_[1 - p, p], y, sub, site)
                runs.append(r)
        else:
            correlation_loaders(loaders)
            files = sorted(run.glob(f"fold{fold}_seed*.pt")) or [run / f"fold{fold}.pt"]
            n_roi = loaders["train"].dataset[0]["x"].shape[-1]
            runs = []
            for f in files:
                model = (build_model(a, n_roi, 2, networks) if kind == "psychosisnet" else DEEP[kind](n_roi, 2)).to(device)
                model.load_state_dict(torch.load(f, map_location=device))
                runs.append({part: predict(model, DataLoader(loaders[part].dataset, batch_size=64, shuffle=False), device)
                             for part in PARTS})
        for part in PARTS:
            out[part].append(frame(fold, *average([r[part] for r in runs])))
        print(f"  {run.name} fold {fold}: {len(files)} saved model(s)", flush=True)
    return {part: pd.concat(v, ignore_index=True) for part, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders in data/runs")
    ap.add_argument("--threads", type=int, default=4)
    cli = ap.parse_args()
    torch.set_num_threads(cli.threads)
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in cli.runs:
        run = cfg["paths"]["data"] / "runs" / name
        res = rescore(run, cfg, device)
        out = run / "rescored"
        out.mkdir(exist_ok=True)
        check = {}
        for part, df in res.items():
            df.to_csv(out / f"{part}_predictions.csv", index=False)
            stored = run / f"{part}_predictions.csv"
            if stored.exists():
                m = pd.read_csv(stored).merge(df, on=["fold", "subject"], suffixes=("_stored", ""))
                check[part] = dict(n=len(m), n_rescored=len(df), max_abs_diff=float((m.p1_stored - m.p1).abs().max()))
        (out / "check.json").write_text(json.dumps(check, indent=2))
        print(f"{name}: {check}  -> {out}")


if __name__ == "__main__":
    main()
