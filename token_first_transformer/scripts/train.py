"""Train the token-first transformer model.

Usage: python scripts/train.py [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from dataset.klines_dataset import KlinesDataset, make_split
from training.trainer import Trainer, compute_class_weights
from models.price_transformer import PriceTransformer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    train_files = make_split(data_dir, *cfg["data"]["train_months"])
    val_files = make_split(data_dir, *cfg["data"]["val_months"])
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    seq_cfg = cfg["sequence"]
    tok_cfg = cfg["tokenizer"]
    train_ds = KlinesDataset(
        train_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )
    val_ds = KlinesDataset(
        val_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )

    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model_cfg = cfg["model"]
    model = PriceTransformer(
        delta_vocab_size=model_cfg["delta_vocab_size"],
        bucket_vocab_size=model_cfg["bucket_vocab_size"],
        delta_emb_dim=model_cfg["delta_emb_dim"],
        bucket_emb_dim=model_cfg["bucket_emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
        seq_len=seq_cfg["length"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_accum_steps=cfg["training"]["grad_accum_steps"],
        early_stop_patience=cfg["training"]["early_stop_patience"],
        device=cfg["training"]["device"],
        checkpoint_dir=Path(cfg["training"]["checkpoint_dir"]),
    )
    trainer.train()


if __name__ == "__main__":
    main()
