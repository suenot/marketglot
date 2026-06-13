import numpy as np
import pandas as pd
import pytest

from book.book import LocalBook
from book.sampler import features_from_book, sample_day


# -- features_from_book ----------------------------------------------------


def test_features_shape_and_layout():
    book = LocalBook()
    book.apply_snapshot(
        prices=[100.0, 99.0, 101.0, 102.0],
        qtys=[1.0, 2.0, 3.0, 4.0],
        sides=["bid", "bid", "ask", "ask"],
    )
    depth = 2
    out = features_from_book(book, depth)
    assert out is not None
    vec, mid = out
    assert vec.shape == (4 * depth,)
    assert vec.dtype == np.float32
    assert mid == pytest.approx(100.5)

    bid_off = vec[0:depth]
    bid_qty = vec[depth : 2 * depth]
    ask_off = vec[2 * depth : 3 * depth]
    ask_qty = vec[3 * depth : 4 * depth]

    assert bid_off[0] == pytest.approx(abs(100.0 - mid) / mid)
    assert bid_off[1] == pytest.approx(abs(99.0 - mid) / mid)
    assert bid_qty[0] == pytest.approx(np.log1p(1.0))
    assert bid_qty[1] == pytest.approx(np.log1p(2.0))
    assert ask_off[0] == pytest.approx(abs(101.0 - mid) / mid)
    assert ask_off[1] == pytest.approx(abs(102.0 - mid) / mid)
    assert ask_qty[0] == pytest.approx(np.log1p(3.0))
    assert ask_qty[1] == pytest.approx(np.log1p(4.0))


def test_features_padding_repeats_deepest_offset_zero_qty():
    book = LocalBook()
    # Only one level per side, depth 3 -> two padded slots per side.
    book.apply_snapshot(
        prices=[100.0, 101.0],
        qtys=[1.0, 2.0],
        sides=["bid", "ask"],
    )
    depth = 3
    vec, mid = features_from_book(book, depth)
    bid_off = vec[0:depth]
    bid_qty = vec[depth : 2 * depth]
    deepest = abs(100.0 - mid) / mid
    assert bid_off[0] == pytest.approx(deepest)
    # Padded offsets repeat the deepest available level.
    assert bid_off[1] == pytest.approx(deepest)
    assert bid_off[2] == pytest.approx(deepest)
    # Padded quantities are zero.
    assert bid_qty[1] == 0.0
    assert bid_qty[2] == 0.0


def test_features_invalid_book_returns_none():
    book = LocalBook()
    book.apply_snapshot(prices=[100.0], qtys=[1.0], sides=["bid"])
    assert features_from_book(book, depth=2) is None


# -- sample_day ------------------------------------------------------------


def _write_snapshot(day_dir, hour, prices, qtys, sides, ts=0):
    df = pd.DataFrame(
        {
            "ts": np.full(len(prices), ts, dtype=np.int64),
            "last_update_id": np.arange(len(prices), dtype=np.int64),
            "side": pd.Series(sides, dtype="category"),
            "price": np.asarray(prices, dtype=np.float64),
            "qty": np.asarray(qtys, dtype=np.float64),
        }
    )
    df.to_parquet(day_dir / f"{hour}_snapshot.parquet.zst", compression="zstd")


def _write_delta(day_dir, hour, rows, rotation=None):
    """rows: list of (event_time, side, price, qty)."""
    n = len(rows)
    event_time = np.array([r[0] for r in rows], dtype=np.int64)
    df = pd.DataFrame(
        {
            "event_time": event_time,
            "recv_time": event_time + 1,
            "first_update_id": np.arange(n, dtype=np.int64),
            "final_update_id": np.arange(n, dtype=np.int64),
            "side": pd.Series([r[1] for r in rows], dtype="category"),
            "price": np.array([r[2] for r in rows], dtype=np.float64),
            "qty": np.array([r[3] for r in rows], dtype=np.float64),
        }
    )
    name = f"{hour}_delta.parquet.zst"
    if rotation is not None:
        name += f".{rotation}"
    df.to_parquet(day_dir / name, compression="zstd")


def test_sample_day_missing_dir_returns_none(tmp_path):
    assert sample_day(tmp_path / "nope", depth=2, interval_sec=1.0) is None


def test_sample_day_no_valid_returns_none(tmp_path):
    day = tmp_path / "day"
    day.mkdir()
    # One-sided book is never valid -> no samples.
    _write_snapshot(day, "00", [100.0], [1.0], ["bid"])
    _write_delta(day, "00", [(1000, "bid", 99.0, 1.0)])
    assert sample_day(day, depth=2, interval_sec=1.0) is None


def test_sample_day_basic_grid_and_values(tmp_path):
    day = tmp_path / "day"
    day.mkdir()
    # Valid two-sided snapshot at hour 00.
    _write_snapshot(
        day,
        "00",
        prices=[100.0, 101.0],
        qtys=[1.0, 1.0],
        sides=["bid", "ask"],
    )
    # Deltas at t=0ms, 500ms, 1000ms. interval 1s -> grid ticks at 0 and 1000.
    _write_delta(
        day,
        "00",
        rows=[
            (0, "bid", 100.0, 2.0),   # update bid qty before tick after 0
            (500, "ask", 101.0, 3.0),
            (1000, "bid", 100.0, 4.0),
        ],
    )
    out = sample_day(day, depth=2, interval_sec=1.0)
    assert out is not None
    assert list(out["ts"]) == [0, 1000]
    assert out["features"].shape == (2, 8)
    assert out["mid"][0] == pytest.approx(100.5)
    assert out["mid"][1] == pytest.approx(100.5)

    # At tick 0 the book is the raw snapshot (bid qty 1.0, before delta applied).
    bid_qty0 = out["features"][0][2]
    assert bid_qty0 == pytest.approx(np.log1p(1.0))
    # At tick 1000 the first two deltas have been applied (bid qty 2.0, ask 3.0).
    bid_qty1 = out["features"][1][2]
    ask_qty1 = out["features"][1][6]
    assert bid_qty1 == pytest.approx(np.log1p(2.0))
    assert ask_qty1 == pytest.approx(np.log1p(3.0))


def test_sample_day_delta_rotation_merged(tmp_path):
    day = tmp_path / "day"
    day.mkdir()
    _write_snapshot(
        day, "00", prices=[100.0, 101.0], qtys=[1.0, 1.0], sides=["bid", "ask"]
    )
    # Base delta file and a rotation .1; rotation rows are interleaved by time.
    _write_delta(day, "00", rows=[(0, "bid", 100.0, 2.0), (2000, "bid", 100.0, 9.0)])
    _write_delta(day, "00", rows=[(1000, "ask", 101.0, 5.0)], rotation=1)
    out = sample_day(day, depth=2, interval_sec=1.0)
    assert out is not None
    # Ticks at 0, 1000, 2000 (anchored at first event_time 0).
    assert list(out["ts"]) == [0, 1000, 2000]
    # At tick 1000 the rotation delta (ask qty 5.0 @ 1000) must NOT yet be applied
    # (sample is emitted before the row at t==1000), so ask qty is still 1.0.
    ask_qty_at_1000 = out["features"][1][6]
    assert ask_qty_at_1000 == pytest.approx(np.log1p(1.0))
    # At tick 2000 the rotation row at 1000 has been applied -> ask qty 5.0.
    ask_qty_at_2000 = out["features"][2][6]
    assert ask_qty_at_2000 == pytest.approx(np.log1p(5.0))


def test_sample_day_qty_zero_removes_level(tmp_path):
    day = tmp_path / "day"
    day.mkdir()
    _write_snapshot(
        day,
        "00",
        prices=[100.0, 99.0, 101.0],
        qtys=[1.0, 1.0, 1.0],
        sides=["bid", "bid", "ask"],
    )
    # Remove best bid 100.0 at t=0, then sample at tick 1000.
    _write_delta(
        day,
        "00",
        rows=[(0, "bid", 100.0, 0.0), (1000, "ask", 101.0, 2.0)],
    )
    out = sample_day(day, depth=2, interval_sec=1.0)
    assert out is not None
    # Tick 0 still sees the full snapshot (mid 100.5), tick 1000 after removal
    # has best bid 99.0 -> mid 100.0.
    assert out["mid"][0] == pytest.approx(100.5)
    assert out["mid"][1] == pytest.approx(100.0)
