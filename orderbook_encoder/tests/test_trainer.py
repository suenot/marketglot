import json

import numpy as np

from training.trainer import train, compute_class_weights


def _write_day(path, n=60, depth=5, seed=0):
    rng = np.random.default_rng(seed)
    ts = (np.arange(n, dtype=np.int64)) * 1000
    # Random walk mid so all three classes appear across the horizon.
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    features = rng.standard_normal((n, 4 * depth)).astype(np.float32)
    np.savez_compressed(path, ts=ts, features=features, mid=mid.astype(np.float64))


def _make_cfg(tmp_path, depth=5):
    samples = tmp_path / "samples"
    sym_dir = samples / "XRPUSDT" / "bybit"
    sym_dir.mkdir(parents=True)
    for i, date in enumerate(["2026-06-01", "2026-06-02", "2026-06-03"]):
        _write_day(sym_dir / f"{date}.npz", n=80, depth=depth, seed=i)

    return {
        "data": {"symbol": "XRPUSDT", "exchange": "bybit",
                 "samples_dir": str(samples)},
        "sampling": {"interval_sec": 1.0, "depth": depth},
        "target": {"horizon_sec": 2.0, "threshold_pct": 0.05},
        "split": {
            "train_days": ["2026-06-01"],
            "val_days": ["2026-06-02"],
            "test_days": ["2026-06-03"],
        },
        "model": {
            "input_dim": 4 * depth,
            "hidden_dims": [16, 8],
            "embedding_dim": 8,
            "num_classes": 3,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.01,
            "epochs": 2,
            "early_stop_patience": 3,
            "device": "cpu",
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    }


def test_compute_class_weights():
    labels = [0, 0, 1, 1, 1, 1, 2, 2]
    weights = compute_class_weights(labels, num_classes=3)
    assert len(weights) == 3
    assert weights[1] < weights[0]
    assert weights[1] < weights[2]


def test_train_creates_artifacts(tmp_path):
    cfg = _make_cfg(tmp_path)
    metrics = train(cfg)

    run_dir = tmp_path / "artifacts"
    runs = list(run_dir.glob("run_*"))
    assert len(runs) == 1
    run = runs[0]

    assert (run / "best.pt").exists()
    assert (run / "config.json").exists()
    assert (run / "test_metrics.json").exists()

    saved = json.loads((run / "test_metrics.json").read_text())
    assert "report" in saved
    assert "confusion_matrix" in saved
    cm = np.array(saved["confusion_matrix"])
    assert cm.shape == (3, 3)
    # Report must contain the three named classes.
    report = saved["report"]
    for name in ("DOWN", "FLAT", "UP"):
        assert name in report

    assert metrics["run_dir"] == str(run)


def test_train_respects_epochs(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg["training"]["epochs"] = 1
    metrics = train(cfg)
    assert "report" in metrics
