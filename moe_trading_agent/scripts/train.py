#!/usr/bin/env python3
"""Train the MoE trading model."""
from __future__ import annotations

import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.moe_dataset import MoEDataset
from models.moe_model import MoETradingModel
from training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/default.yaml", data_dir: str | None = None) -> None:
    """Main training entry point."""
    config = load_config(config_path)

    model_cfg = config["model"]
    train_cfg = config["training"]

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

    print(f"Found {len(file_paths)} parquet files")

    # Create dataset
    dataset = MoEDataset(
        file_paths=file_paths,
        seq_len=model_cfg["seq_len"],
        horizon=train_cfg["horizon"],
        threshold=train_cfg["threshold"],
    )
    print(f"Dataset size: {len(dataset)}")

    # Split dataset
    total = len(dataset)
    train_end = int(total * train_cfg["train_split"])
    val_end = int(total * (train_cfg["train_split"] + train_cfg["val_split"]))

    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build model
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
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints"
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        aux_loss_lambda=train_cfg["aux_loss_lambda"],
        patience=train_cfg["patience"],
        checkpoint_dir=checkpoint_dir,
    )

    trainer.train(epochs=train_cfg["epochs"])


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    data = sys.argv[2] if len(sys.argv) > 2 else None
    main(config_path=config, data_dir=data)
