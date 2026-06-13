"""Build sampled feature files from raw order book parquet days.

For each day, replays the book and writes a compressed ``.npz`` with keys
``ts, features, mid`` to ``{samples_dir}/{symbol}/{exchange}/{date}.npz``.

Usage:
    python scripts/build_samples.py --config configs/default.yaml \
        --dates 2026-06-01:2026-06-09 [--force]
    python scripts/build_samples.py --dates 2026-06-09
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from book.sampler import sample_day


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
    return [date.fromisoformat(spec).isoformat()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sampled feature npz files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dates", required=True, help="YYYY-MM-DD or START:END")
    parser.add_argument("--force", action="store_true", help="rebuild existing npz")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]
    exchange = data_cfg["exchange"]
    raw_dir = Path(data_cfg["raw_dir"])
    samples_dir = Path(data_cfg["samples_dir"])

    depth = int(cfg["sampling"]["depth"])
    interval_sec = float(cfg["sampling"]["interval_sec"])

    out_dir = samples_dir / symbol / exchange
    out_dir.mkdir(parents=True, exist_ok=True)

    for day in parse_dates(args.dates):
        out_path = out_dir / f"{day}.npz"
        if out_path.exists() and not args.force:
            print(f"  {day}: skip (exists)")
            continue

        day_dir = raw_dir / symbol / exchange / day
        result = sample_day(day_dir, depth=depth, interval_sec=interval_sec)
        if result is None:
            print(f"  {day}: no valid samples")
            continue

        np.savez_compressed(
            out_path,
            ts=result["ts"],
            features=result["features"],
            mid=result["mid"],
        )
        print(f"  {day}: {len(result['ts'])} samples -> {out_path}")


if __name__ == "__main__":
    main()
