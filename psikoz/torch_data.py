"""PyTorch access to the extracted features.

    from psikoz.torch_data import make_loaders
    loaders = make_loaders(split="cobre_cv", fold=0, atlas="schaefer100",
                           input="fc", label="psychosis", batch_size=16)
    for batch in loaders["train"]:
        batch["x"], batch["y"]
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import in_datasets, load_config, main_rows

LABEL_MAPS = {"diagnosis": {"HC": 0, "SZ": 1, "BP": 2, "ADHD": 3, "MDD": 4, "ASD": 5}, "sex": {"F": 0, "M": 1},
              "dataset": {"cobre": 0, "ds005073": 1, "ds000030": 2, "ds001461": 3}}


def site_codes(cfg):
    """Integer code per dataset / site; the first four keep their historical codes. External sites are
    numbered after all others, so that adding one does not renumber the SRPBS sites."""
    codes = dict(LABEL_MAPS["dataset"])
    part = pd.read_csv(cfg["paths"]["manifest"] / "participants.csv", usecols=["dataset"])
    external = set(cfg.get("external", []))
    for d in sorted(set(part["dataset"]) - set(codes), key=lambda d: (d in external, d)):
        codes[d] = len(codes)
    return codes


class FMRIDataset(Dataset):
    """One item per run.

    input="fc"          x: (R, R) Fisher-z matrix, or (R*(R-1)/2,) if vectorize=True
    input="timeseries"  x: (window, R) z-scored ROI signals at TR 2 s, mask: (window,) bool
                        (False = motion-censored or padded frame). Random crop when
                        train=True, centre crop otherwise.
    Each item also carries y, site, age, sex and the subject id.
    """

    def __init__(self, atlas="schaefer100", subjects=None, input="fc", label="psychosis",
                 window=60, train=False, vectorize=False, require_qc=True, clean_labels_only=False,
                 diagnoses=None, datasets=None, tasks=("rest",), cfg=None):
        self.cfg = cfg or load_config()
        index = pd.read_csv(self.cfg["paths"]["features"] / atlas / "index.csv")
        index = main_rows(index[index["task"].isin(tasks)])
        if subjects is not None:
            index = index[index["subject"].isin(subjects)]
        if require_qc:
            index = index[index["qc_pass"] == 1]
        if clean_labels_only:
            index = index[index["label_clean"] == 1]
        if diagnoses is not None:  # e.g. ("HC", "SZ", "BP") to leave ADHD out
            index = index[index["diagnosis"].isin(diagnoses)]
        if datasets is not None:   # e.g. ("cobre", "ds000030") to keep only sites that have both classes
            index = index[in_datasets(index["dataset"], datasets)]
        index = index[index[label].notna()].reset_index(drop=True)
        self.index, self.input, self.window, self.train, self.vectorize = index, input, window, train, vectorize

        y = index[label].map(LABEL_MAPS[label]) if label in LABEL_MAPS else index[label]
        # Categorical labels (and the 0/1 psychosis flag) are class indices; anything else is a regression target.
        categorical = label in LABEL_MAPS or label == "psychosis"
        self.y = torch.tensor(y.to_numpy(), dtype=torch.long if categorical else torch.float32)
        self.site = torch.tensor(index["dataset"].map(site_codes(self.cfg)).to_numpy())
        self.age = torch.tensor(index["age"].to_numpy(), dtype=torch.float32)
        self.sex = torch.tensor(index["sex"].map(LABEL_MAPS["sex"]).to_numpy())

        # A few hundred small files: keep everything in memory.
        self.items = []
        for f in index["file"]:
            z = np.load(self.cfg["root"] / f)
            self.items.append((torch.from_numpy(z["fc"]),) if input == "fc" else
                              (torch.from_numpy(z["timeseries"]), torch.from_numpy(z["frame_mask"])))
        n_roi = self.items[0][0].shape[-1] if self.items else 0
        self.triu = torch.triu_indices(n_roi, n_roi, offset=1)

    def __len__(self):
        return len(self.index)

    def class_weights(self):
        counts = torch.bincount(self.y).float()
        return counts.sum() / (len(counts) * counts.clamp(min=1))

    def __getitem__(self, i):
        item = dict(y=self.y[i], site=self.site[i], age=self.age[i], sex=self.sex[i],
                    subject=self.index["subject"].iat[i], idx=i)
        if self.input == "fc":
            fc = self.items[i][0]
            item["x"] = fc[self.triu[0], self.triu[1]] if self.vectorize else fc
            return item
        ts, mask = self.items[i]
        t, w = len(ts), self.window
        if t >= w:
            start = int(torch.randint(0, t - w + 1, ())) if self.train else (t - w) // 2
            item["x"], item["mask"] = ts[start:start + w], mask[start:start + w]
        else:
            item["x"] = torch.cat([ts, ts.new_zeros(w - t, ts.shape[1])])
            item["mask"] = torch.cat([mask, mask.new_zeros(w - t)])
        return item


def make_loaders(split="cobre_cv", fold=0, batch_size=16, num_workers=0, cfg=None, **dataset_kwargs):
    """Train / val / test DataLoaders for one fold of a split file in data/splits."""
    cfg = cfg or load_config()
    folds = json.loads((cfg["paths"]["splits"] / f"{split}.json").read_text())["folds"]
    loaders = {}
    for part, subjects in folds[fold].items():
        ds = FMRIDataset(subjects=subjects, train=part == "train", cfg=cfg, **dataset_kwargs)
        loaders[part] = DataLoader(ds, batch_size=batch_size, shuffle=part == "train",
                                   num_workers=num_workers, drop_last=False)
    return loaders
