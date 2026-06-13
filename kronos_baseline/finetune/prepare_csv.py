"""Convert marketglot parquet klines -> Kronos finetune CSV.

Kronos finetune_csv expects columns: timestamps, open, high, low, close, volume, amount.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = "/Users/suenot/projects/w_trading/w_trender/backtests/data"


def main() -> None:
    ap = argparse.ArgumentParser(description="parquet klines -> Kronos finetune CSV")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--granularity", default="klines_1m")
    ap.add_argument("--days", type=int, default=0, help="keep only the last N days (0 = all history)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data_root}/{a.symbol}/{a.granularity}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {a.data_root}/{a.symbol}/{a.granularity}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="s")
    if a.days > 0:
        cutoff = df["timestamps"].max() - pd.Timedelta(days=a.days)
        df = df[df["timestamps"] >= cutoff].reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = 0.0

    out = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows  {out['timestamps'].min()} .. {out['timestamps'].max()}")


if __name__ == "__main__":
    main()
