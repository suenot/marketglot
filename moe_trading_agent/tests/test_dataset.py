"""Tests for MoEDataset."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch


def _create_mock_parquet(tmpdir, n_rows=500, base_price=30000.0):
    """Create mock parquet with realistic-looking BTC price data."""
    np.random.seed(42)
    returns = np.random.randn(n_rows) * 0.001  # Small random returns
    close = base_price * np.exp(np.cumsum(returns)).astype(np.float32)
    high = (close * (1 + np.abs(np.random.randn(n_rows)) * 0.001)).astype(np.float32)
    low = (close * (1 - np.abs(np.random.randn(n_rows)) * 0.001)).astype(np.float32)
    open_prices = ((close + np.random.randn(n_rows) * 5).astype(np.float32))
    volume = (np.abs(np.random.randn(n_rows)) * 100 + 50).astype(np.float32)
    timestamp = np.arange(n_rows, dtype=np.int32)

    df = pd.DataFrame({
        "timestamp": timestamp,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    parquet_path = os.path.join(tmpdir, "mock_data.parquet")
    df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a MoEDataset with mock parquet data."""
    parquet_path = _create_mock_parquet(str(tmp_path))
    from dataset.moe_dataset import MoEDataset
    return MoEDataset(
        file_paths=[parquet_path],
        seq_len=128,
        horizon=60,
        threshold=0.0015,
    )


def test_length_positive(mock_dataset):
    """Dataset length is positive with mock data."""
    assert len(mock_dataset) > 0, f"Dataset length should be positive, got {len(mock_dataset)}"


def test_item_shapes(mock_dataset):
    """Verify tensor shapes from __getitem__."""
    delta_ids, vol_ids, vb_ids, ind_dict, label = mock_dataset[0]

    assert delta_ids.shape == (128,), f"delta_ids shape: {delta_ids.shape}"
    assert vol_ids.shape == (128,), f"vol_ids shape: {vol_ids.shape}"
    assert vb_ids.shape == (128,), f"vb_ids shape: {vb_ids.shape}"

    expected_keys = {"rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"}
    assert set(ind_dict.keys()) == expected_keys, f"Keys: {ind_dict.keys()}"
    for key in expected_keys:
        assert ind_dict[key].shape == (128,), f"{key} shape: {ind_dict[key].shape}"

    assert label.dim() == 0, f"label should be scalar, got shape {label.shape}"


def test_labels_valid(mock_dataset):
    """All labels are in {0, 1, 2}."""
    labels = set()
    for i in range(min(len(mock_dataset), 50)):
        _, _, _, _, label = mock_dataset[i]
        labels.add(label.item())

    assert labels.issubset({0, 1, 2}), f"Invalid labels found: {labels}"
