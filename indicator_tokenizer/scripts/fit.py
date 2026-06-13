"""Fit indicator tokenizers on training data and save boundaries."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import pyarrow.parquet as pq
from indicators.computer import IndicatorComputer
from indicators.tokenizer import IndicatorTokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    klines_dir = data_dir / cfg["data"]["symbol"] / "klines_1m"
    start, end = cfg["data"]["train_months"]
    files = sorted(klines_dir.glob("*.parquet"))
    files = [f for f in files if start <= f.stem <= end]
    print(f"Loading {len(files)} months of training data...")

    all_indicators = {k: [] for k in [
        "rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"
    ]}
    comp = IndicatorComputer()
    for f in files:
        table = pq.read_table(f)
        ohlcv = {col: np.array([v.as_py() for v in table.column(col)], dtype=np.float32)
                 for col in ["open", "high", "low", "close", "volume"]}
        indicators = comp.compute_all(ohlcv)
        for k in all_indicators:
            all_indicators[k].append(indicators[k])

    combined = {k: np.concatenate(v) for k, v in all_indicators.items()}
    for k, v in combined.items():
        print(f"  {k}: {len(v):,} values")

    tok = IndicatorTokenizer()
    tok.fit(combined)
    boundaries_dir = Path(cfg["boundaries_dir"])
    tok.save(boundaries_dir)
    print(f"\nBoundaries saved to {boundaries_dir}/")
    vs = tok.vocab_sizes()
    for k, v in vs.items():
        print(f"  {k}: vocab_size={v}")


if __name__ == "__main__":
    main()
