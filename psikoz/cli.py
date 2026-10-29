"""python -m psikoz.cli <stage> [--config configs/default.yaml]"""
import argparse

from .config import load_config

STAGES = ["organize", "preprocess", "features", "splits"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=STAGES + ["reindex", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", nargs="*", help="preprocess: limit to these participant ids")
    ap.add_argument("--datasets", nargs="*", help="features: only these datasets; their index rows are replaced, all others kept")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for stage in STAGES if args.stage == "all" else [args.stage]:
        print(f"=== {stage}")
        if stage == "reindex":  # organize changed labels; refresh feature indexes without recomputing
            from . import features
            features.reindex(cfg)
        if stage == "organize":
            from . import organize
            organize.run(cfg)
        elif stage == "preprocess":
            from . import preprocess
            preprocess.run(cfg, args.subjects)
        elif stage == "features":
            from . import features
            features.run(cfg, args.datasets)
        elif stage == "splits":
            from . import splits
            splits.run(cfg)


if __name__ == "__main__":
    main()
