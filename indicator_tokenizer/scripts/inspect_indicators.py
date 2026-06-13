"""Print indicator statistics from data."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import pyarrow.parquet as pq
from indicators.computer import IndicatorComputer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--months", default=1, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    klines_dir = data_dir / cfg["data"]["symbol"] / "klines_1m"
    files = sorted(klines_dir.glob("*.parquet"))[:args.months]
    print(f"Inspecting {len(files)} file(s)")

    comp = IndicatorComputer()
    for f in files:
        print(f"\n=== {f.stem} ===")
        table = pq.read_table(f)
        ohlcv = {col: np.array([v.as_py() for v in table.column(col)], dtype=np.float32)
                 for col in ["open", "high", "low", "close", "volume"]}
        indicators = comp.compute_all(ohlcv)
        for name, values in indicators.items():
            v = values[len(values) // 5:]
            print(f"  {name:20s}  min={v.min():10.4f}  max={v.max():10.4f}  mean={v.mean():10.4f}  std={v.std():10.4f}")


if __name__ == "__main__":
    main()
