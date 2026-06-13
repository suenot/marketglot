"""Run backtest on test period.

Usage: python scripts/backtest.py --checkpoint checkpoints/best.pt [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from dataset.klines_dataset import KlinesDataset, make_split
from models.price_transformer import PriceTransformer
from backtest.engine import BacktestEngine


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

    dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for delta, vol, vb, _ in dl:
            delta = delta.to(device)
            vol = vol.to(device)
            vb = vb.to(device)
            logits = model(delta, vol, vb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
    predictions = np.array(all_preds)

    import pandas as pd
    frames = [pq.read_table(f).to_pandas() for f in test_files]
    closes = pd.concat([f["close"] for f in frames]).values.astype(np.float32)

    bt_cfg = cfg["backtest"]
    engine = BacktestEngine(
        commission=bt_cfg["commission"], stop_loss=bt_cfg["stop_loss"],
        take_profit=bt_cfg["take_profit"], max_hold=bt_cfg["max_hold"],
    )
    result = engine.run(closes, predictions)

    print(f"\nBacktest Results:")
    print(f"  Total PnL:    {result.total_pnl:+.2%}")
    print(f"  Sharpe:        {result.sharpe:.2f}")
    print(f"  Max Drawdown:  {result.max_drawdown:.2%}")
    print(f"  Win Rate:      {result.win_rate:.2%}")
    print(f"  Trade Count:   {result.trade_count}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Duration:  {result.avg_duration:.1f} candles")


if __name__ == "__main__":
    main()
