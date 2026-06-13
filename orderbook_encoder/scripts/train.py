"""Train the order book MLP encoder/classifier.

Usage: python scripts/train.py [--config configs/default.yaml] [--epochs N]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from training.trainer import train


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from the config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    metrics = train(cfg)
    print(f"\nArtifacts written to {metrics['run_dir']}")


if __name__ == "__main__":
    main()
