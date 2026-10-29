"""PsychosisNET as a feature extractor: classical classifiers on its pooled subject embedding.

    python -m psikoz.embed_classify --run v11_l2_1e-3

For every fold the saved network (trained on the training part, early-stopped on the validation
part) embeds all subjects; the classifiers are fitted on the train + val embeddings and scored on
the test part. Hyperparameters are fixed in advance, nothing is tuned on the test part.
Works with runs of the current model (v11); earlier runs: archive/v10/embed_classify.py.
"""
import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import load_config
from .harmonize import correlation_loaders
from .torch_data import make_loaders
from .train import build_model, scores


def classifiers(seed):
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    return {
        "logreg": scaled(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        "svm_rbf": scaled(SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed)),
        "svm_linear": scaled(SVC(C=0.1, kernel="linear", class_weight="balanced", probability=True, random_state=seed)),
        "knn": scaled(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "mlp": scaled(MLPClassifier((32,), alpha=1e-2, early_stopping=True, max_iter=1000, random_state=seed)),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=seed, n_jobs=4),
    }


@torch.no_grad()
def embed(model, ds, device):
    x = torch.stack([it[0] for it in ds.items]).to(device)
    tokens, _ = model.encode(model.spd(x))
    return model.pool(tokens)[0].cpu().numpy(), model(x).softmax(-1).cpu().numpy(), ds.y.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="folder in data/runs written by psikoz.train")
    cli = ap.parse_args()
    cfg = load_config()
    run = cfg["paths"]["data"] / "runs" / cli.run
    a = SimpleNamespace(**json.loads((run / "summary.json").read_text())["args"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    networks = pd.factorize(pd.read_csv(cfg["paths"]["features"] / a.atlas / "labels.csv")["network"])[0]

    rows = []
    for fold in range(len(list(run.glob("fold*.pt")))):
        loaders = make_loaders(split=a.split, fold=fold, atlas=a.atlas, label=a.label, input="fc",
                               diagnoses=a.diagnoses, datasets=a.datasets, cfg=cfg)
        correlation_loaders(loaders)
        n_classes = int(loaders["train"].dataset.y.max()) + 1
        model = build_model(a, len(networks), n_classes, networks).to(device)
        model.load_state_dict(torch.load(run / f"fold{fold}.pt", map_location=device))
        model.eval()
        e = {part: embed(model, loaders[part].dataset, device) for part in loaders}
        xtr, ytr = np.vstack([e["train"][0], e["val"][0]]), np.concatenate([e["train"][2], e["val"][2]])
        xte, pnet, yte = e["test"]
        rows.append(dict(fold=fold, classifier="PsychosisNET head", **scores(pnet, yte)))
        for name, clf in classifiers(fold).items():
            rows.append(dict(fold=fold, classifier=name, **scores(clf.fit(xtr, ytr).predict_proba(xte), yte),
                             train_auc=scores(clf.predict_proba(xtr), ytr)["auc"]))
    res = pd.DataFrame(rows)
    res.to_csv(run / "embedding_classifiers.csv", index=False)
    print(cli.run, "| embedding dim", xtr.shape[1])
    print(res.groupby("classifier", sort=False)[["train_auc", "auc", "bacc"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
