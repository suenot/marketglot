import numpy as np

from dataset.orderbook_dataset import OrderbookDataset, build_splits, DOWN, FLAT, UP


def _write_npz(path, ts, mid, n_feat=8):
    ts = np.asarray(ts, dtype=np.int64)
    mid = np.asarray(mid, dtype=np.float64)
    features = np.tile(mid[:, None], (1, n_feat)).astype(np.float32)
    np.savez_compressed(path, ts=ts, features=features, mid=mid)


# interval 1s, horizon 2s -> h = 2 rows; threshold 1% on the mid return.
HORIZON = 2.0
INTERVAL = 1.0
THRESH = 1.0  # percent


def test_labels_up_flat_down(tmp_path):
    # mid known: from index 0 (100) to index 2 (102) -> +2% -> UP
    #            from index 1 (100) to index 3 (100) -> 0%  -> FLAT
    #            from index 2 (102) to index 4 (98)  -> -3.9% -> DOWN
    ts = [0, 1000, 2000, 3000, 4000]
    mid = [100.0, 100.0, 102.0, 100.0, 98.0]
    p = tmp_path / "day.npz"
    _write_npz(p, ts, mid)
    ds = OrderbookDataset([p], horizon_sec=HORIZON, threshold_pct=THRESH,
                          interval_sec=INTERVAL)
    # pairs: (0,2),(1,3),(2,4) all valid (3 rows have a partner)
    assert len(ds) == 3
    labels = [int(ds[i][1]) for i in range(len(ds))]
    assert labels[0] == UP
    assert labels[1] == FLAT
    assert labels[2] == DOWN


def test_feature_shape(tmp_path):
    ts = [0, 1000, 2000, 3000]
    mid = [100.0, 100.0, 100.0, 100.0]
    p = tmp_path / "day.npz"
    _write_npz(p, ts, mid, n_feat=80)
    ds = OrderbookDataset([p], horizon_sec=HORIZON, threshold_pct=THRESH,
                          interval_sec=INTERVAL)
    x, y = ds[0]
    assert x.shape == (80,)
    assert x.dtype.is_floating_point
    assert int(y) in (DOWN, FLAT, UP)


def test_hole_in_ts_dropped(tmp_path):
    # gap between index 1 and 2 (10s) makes pairs spanning it invalid.
    # horizon gap expected = 2000ms; tolerance is also 2000ms (drop if |gap-2000|>2000).
    # pair (0,2): ts 0 -> 11000, gap 11000, |11000-2000|=9000 > 2000 -> dropped
    # pair (1,3): ts 1000 -> 12000, gap 11000 -> dropped
    # pair (2,4): ts 11000 -> 13000, gap 2000 -> kept
    ts = [0, 1000, 11000, 12000, 13000]
    mid = [100.0, 100.0, 100.0, 100.0, 100.0]
    p = tmp_path / "day.npz"
    _write_npz(p, ts, mid)
    ds = OrderbookDataset([p], horizon_sec=HORIZON, threshold_pct=THRESH,
                          interval_sec=INTERVAL)
    assert len(ds) == 1


def test_no_cross_file_boundary(tmp_path):
    # Two files each with 4 rows; pairs must stay within a file.
    ts1 = [0, 1000, 2000, 3000]
    mid1 = [100.0, 100.0, 100.0, 100.0]
    ts2 = [0, 1000, 2000, 3000]
    mid2 = [200.0, 200.0, 200.0, 200.0]
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_npz(p1, ts1, mid1)
    _write_npz(p2, ts2, mid2)
    ds = OrderbookDataset([p1, p2], horizon_sec=HORIZON, threshold_pct=THRESH,
                          interval_sec=INTERVAL)
    # Each file yields rows 0,1 with a partner at +2 -> 2 pairs per file = 4.
    assert len(ds) == 4
    # No pair should ever mix mids from different files: returns are all 0 -> FLAT.
    labels = [int(ds[i][1]) for i in range(len(ds))]
    assert all(lbl == FLAT for lbl in labels)


def test_build_splits_missing_dates_skipped(tmp_path, capsys):
    samples = tmp_path / "samples"
    sym_dir = samples / "XRPUSDT" / "bybit"
    sym_dir.mkdir(parents=True)
    ts = [0, 1000, 2000, 3000]
    mid = [100.0, 100.0, 102.0, 100.0]
    _write_npz(sym_dir / "2026-06-01.npz", ts, mid)

    cfg = {
        "data": {"symbol": "XRPUSDT", "exchange": "bybit",
                 "samples_dir": str(samples)},
        "sampling": {"interval_sec": INTERVAL},
        "target": {"horizon_sec": HORIZON, "threshold_pct": THRESH},
        "split": {
            "train_days": ["2026-06-01"],
            "val_days": ["2026-06-02"],   # missing -> skipped
            "test_days": ["2026-06-03"],  # missing -> skipped
        },
    }
    train_ds, val_ds, test_ds = build_splits(cfg)
    assert len(train_ds) > 0
    assert len(val_ds) == 0
    assert len(test_ds) == 0
    out = capsys.readouterr().out
    assert "missing samples file" in out
