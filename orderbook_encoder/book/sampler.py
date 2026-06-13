"""Turn raw snapshot/delta parquet files into a sampled feature time series.

The book is replayed delta-by-delta and sampled on a fixed time grid (by
``event_time``). Each sample is a flat feature vector plus the mid price.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from book.book import LocalBook

# {HH}_snapshot.parquet.zst and {HH}_delta.parquet.zst[.N]
_SNAPSHOT_RE = re.compile(r"^(\d{2})_snapshot\.parquet\.zst$")
_DELTA_RE = re.compile(r"^(\d{2})_delta\.parquet\.zst(?:\.(\d+))?$")


def features_from_book(book: LocalBook, depth: int) -> tuple[np.ndarray, float] | None:
    """Build a feature vector from the current book state.

    Returns ``(vector, mid)`` where ``vector`` has shape ``(4 * depth,)`` and
    layout ``[bid_off x D, bid_qty x D, ask_off x D, ask_qty x D]`` with
    ``off = |price - mid| / mid`` and ``qty = log1p(qty)``. When fewer than
    ``depth`` levels exist on a side, the offset repeats the deepest available
    level and the quantity is zero (padding). Returns ``None`` if the book is
    invalid.
    """
    if not book.is_valid():
        return None
    mid = book.mid()
    if mid is None:
        return None

    bids, asks = book.top_levels(depth)

    bid_off, bid_qty = _side_features(bids, mid, depth)
    ask_off, ask_qty = _side_features(asks, mid, depth)

    vector = np.concatenate([bid_off, bid_qty, ask_off, ask_qty]).astype(np.float32)
    return vector, mid


def _side_features(
    levels: list[tuple[float, float]], mid: float, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(offsets, log_qtys)`` of length ``depth`` for one side."""
    off = np.zeros(depth, dtype=np.float32)
    qty = np.zeros(depth, dtype=np.float32)
    last_off = 0.0
    for i in range(depth):
        if i < len(levels):
            price, q = levels[i]
            last_off = abs(price - mid) / mid
            off[i] = last_off
            qty[i] = np.log1p(q)
        else:
            # Padding: repeat the deepest offset, quantity stays zero.
            off[i] = last_off
            qty[i] = 0.0
    return off, qty


def _scan_hours(day_dir: Path) -> dict[str, dict]:
    """Group files in ``day_dir`` by hour into ``{snapshot, deltas}`` entries."""
    hours: dict[str, dict] = {}
    for path in day_dir.iterdir():
        if not path.is_file():
            continue
        m = _SNAPSHOT_RE.match(path.name)
        if m:
            hours.setdefault(m.group(1), {"snapshot": None, "deltas": []})
            hours[m.group(1)]["snapshot"] = path
            continue
        m = _DELTA_RE.match(path.name)
        if m:
            rotation = int(m.group(2)) if m.group(2) is not None else 0
            hours.setdefault(m.group(1), {"snapshot": None, "deltas": []})
            hours[m.group(1)]["deltas"].append((rotation, path))
    return hours


def _load_deltas(delta_files: list[tuple[int, Path]]) -> pd.DataFrame | None:
    """Concatenate delta rotations and sort by replay order."""
    if not delta_files:
        return None
    frames = [pd.read_parquet(path) for _, path in sorted(delta_files)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["event_time", "final_update_id"], kind="stable")
    return df.reset_index(drop=True)


def sample_day(day_dir: Path, depth: int, interval_sec: float) -> dict | None:
    """Replay a full day and sample features on a fixed time grid.

    Hours are processed in ascending order. For each hour: if a snapshot file
    exists, apply it (resync); then replay the hour's deltas (rotations
    concatenated and sorted by ``event_time, final_update_id``) row by row,
    advancing the per-hour clock by ``event_time``. A sample is emitted on each
    grid tick spaced ``interval_sec`` apart (by event_time), holding the book
    state reached by that tick. Ticks where the book is invalid are skipped.

    Returns ``{'ts': int64 (T,) ms, 'features': float32 (T, 4*depth),
    'mid': float64 (T,)}`` or ``None`` when the day has no valid samples.
    """
    day_dir = Path(day_dir)
    if not day_dir.exists():
        return None

    hours = _scan_hours(day_dir)
    interval_ms = int(round(interval_sec * 1000))

    book = LocalBook()
    ts_out: list[int] = []
    feat_out: list[np.ndarray] = []
    mid_out: list[float] = []

    for hour in sorted(hours):
        entry = hours[hour]
        if entry["snapshot"] is not None:
            snap = pd.read_parquet(entry["snapshot"])
            book.apply_snapshot(
                snap["price"].to_numpy(),
                snap["qty"].to_numpy(),
                snap["side"].astype(str).to_numpy(),
            )

        deltas = _load_deltas(entry["deltas"])
        if deltas is None or len(deltas) == 0:
            continue

        event_time = deltas["event_time"].to_numpy()
        prices = deltas["price"].to_numpy()
        qtys = deltas["qty"].to_numpy()
        sides = deltas["side"].astype(str).to_numpy()

        # Grid anchored at the first delta of the hour.
        next_tick = int(event_time[0])
        for i in range(len(deltas)):
            t = int(event_time[i])
            # Emit one sample per crossed grid tick before applying this row.
            while t >= next_tick:
                out = features_from_book(book, depth)
                if out is not None:
                    vector, mid = out
                    ts_out.append(next_tick)
                    feat_out.append(vector)
                    mid_out.append(mid)
                next_tick += interval_ms
            book.apply_delta((prices[i],), (qtys[i],), (sides[i],))

    if not ts_out:
        return None

    return {
        "ts": np.asarray(ts_out, dtype=np.int64),
        "features": np.stack(feat_out).astype(np.float32),
        "mid": np.asarray(mid_out, dtype=np.float64),
    }
