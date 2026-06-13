"""Evaluate model on test set.

Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pt [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset.klines_dataset import KlinesDataset, make_split
from models.price_transformer import PriceTransformer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    test_files = make_split(data_dir, *cfg["data"]["test_months"])
    print(f"Test files: {len(test_files)}")

    seq_cfg = cfg["sequence"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]

    test_ds = KlinesDataset(
        test_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )
    test_dl = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model = PriceTransformer(
        delta_vocab_size=model_cfg["delta_vocab_size"],
        bucket_vocab_size=model_cfg["bucket_vocab_size"],
        delta_emb_dim=model_cfg["delta_emb_dim"],
        bucket_emb_dim=model_cfg["bucket_emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=0.0,
        num_classes=model_cfg["num_classes"],
        seq_len=seq_cfg["length"],
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    device = cfg["training"]["device"]
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for delta, vol, vb, labels in test_dl:
            delta = delta.to(device)
            vol = vol.to(device)
            vb = vb.to(device)
            logits = model(delta, vol, vb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["DOWN", "FLAT", "UP"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
