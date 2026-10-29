from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parents[1]
    cfg["root"] = root
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = (root / rel).resolve()
    data = cfg["paths"]["data"]
    cfg["paths"].update(
        cobre_bids=data / "bids" / "cobre",
        manifest=data / "manifest",
        preproc=data / "preproc",
        features=data / "features",
        splits=data / "splits",
    )
    return cfg


def rel(cfg, path):
    """Path relative to the project root, with forward slashes (portable in CSVs)."""
    return Path(path).resolve().relative_to(cfg["root"]).as_posix()


def per_dataset(mapping, dataset, *default):
    """Per-dataset setting; the sites of a multi-site release ("srpbs-KUT") fall back to its family ("srpbs")."""
    for key in (dataset, dataset.split("-")[0]):
        if key in mapping:
            return mapping[key]
    if default:
        return default[0]
    raise KeyError(dataset)


def main_rows(index):
    """Rows of the classification sample; role "pretrain" marks participants processed for label-free use only."""
    return index[index["role"].fillna("main") == "main"] if "role" in index.columns else index


def in_datasets(column, names):
    """Rows whose dataset is listed in names, by full name or by family ("srpbs" selects every srpbs-* site)."""
    return column.isin(names) | column.str.split("-").str[0].isin(names)
