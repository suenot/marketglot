"""Download raw order book parquet files from the warehouse.

Usage:
    python scripts/download_data.py --config configs/default.yaml \
        --dates 2026-06-01:2026-06-09
    python scripts/download_data.py --dates 2026-06-09
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from warehouse.client import WarehouseClient


def parse_dates(spec: str) -> list[str]:
    """Expand ``YYYY-MM-DD`` or inclusive ``YYYY-MM-DD:YYYY-MM-DD`` to a list."""
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = date.fromisoformat(start_s)
        end = date.fromisoformat(end_s)
        if end < start:
            raise ValueError(f"end {end_s} before start {start_s}")
        days = []
        cur = start
        while cur <= end:
            days.append(cur.isoformat())
            cur += timedelta(days=1)
        return days
    # Validate single date too.
    return [date.fromisoformat(spec).isoformat()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw order book data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dates", required=True, help="YYYY-MM-DD or START:END")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]
    exchange = data_cfg["exchange"]
    raw_dir = Path(data_cfg["raw_dir"])

    client = WarehouseClient(data_cfg["api_base"], data_cfg["s3_base"])
    dates = parse_dates(args.dates)
    print(f"Downloading {symbol}/{exchange} for {len(dates)} day(s) -> {raw_dir}")

    for day in dates:
        paths = client.download_day(symbol, exchange, day, raw_dir)
        print(f"  {day}: {len(paths)} file(s)")


if __name__ == "__main__":
    main()
