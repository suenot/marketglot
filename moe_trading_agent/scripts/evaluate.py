#!/usr/bin/env python3
"""valuation the MoE trading model on test data."""
from __future__ import annotations

import os
import sys

import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.moe_dataset import MoEDataset
from models.moe_model import MoETradingModel
from training.trainer import get_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(
    config_path: str = "configs/default.yaml",
    checkpoint_path: str | None = None,
    data_dir: str | None = None,
) -> None:
    """Main valueation entry point."""
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Find parquet files
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    file_paths = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.endswith(".parquet")
    ]

    if not file_paths:
        print(f"No parquet files found in {data_dir}")
        sys.exit(1)

    # Create dataset
    dataset = MoEDataset(
        file_paths=file_paths,
        seq_len=model_cfg["seq_len"],
        horizon=train_cfg["horizon"],
        threshold=train_cfg["threshold"],
    )

    # Test split
    total = len(dataset)
    train_end = int(total * train_cfg["train_split"])
    val_end = int(total * (train_cfg["train_split"] + train_cfg["val_split"]))
    test_dataset = Subset(dataset, range(val_end, total))

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    print(f"Test set: {len(test_dataset)} samples")

    # Build and load model
    model = MoETradingModel(
        seq_len=model_cfg["seq_len"],
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dim=model_cfg["dim"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "best_model.pt"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (F1: {checkpoint['val_f1']:.4f})")

    # valueate
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            delta_ids, vol_ids, vb_ids, ind_dict, labels = batch
            delta_ids = delta_ids.to(device)
            vol_ids = vol_ids.to(device)
            vb_ids = vb_ids.to(device)
            ind_dict = {k: v.to(device) for k, v in ind_dict.items()}

            logits, _ = model(delta_ids, vol_ids, vb_ids, ind_dict)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    target_names = ["UP", "FLAT", "DOWN"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    data = sys.argv[3] if len(sys.argv) > 3 else None
    main(config_path=config, checkpoint_path=ckpt, data_dir=data)
