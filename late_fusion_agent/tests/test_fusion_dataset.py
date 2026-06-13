import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataset.fusion_dataset import FusionDataset


@pytest.fixture
def sample_parquet(tmp_path):
    n = 500
    rng = np.random.default_rng(42)
    close = (30000 + np.cumsum(rng.standard_normal(n) * 10)).astype(np.float32)
    table = pa.table({
        "timestamp": np.arange(n, dtype=np.int32),
        "open": (close - rng.standard_normal(n) * 2).astype(np.float32),
        "high": (close + np.abs(rng.standard_normal(n)) * 5).astype(np.float32),
        "low": (close - np.abs(rng.standard_normal(n)) * 5).astype(np.float32),
        "close": close.astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 10).astype(np.float32),
    })
    path = tmp_path / "2023-02.parquet"
    pq.write_table(table, path)
    return path


def test_dataset_length(sample_parquet):
    ds = FusionDataset([sample_parquet], seq_len=64, target_horizon=10)
    assert len(ds) == 500 - 64 - 10


def test_item_shapes(sample_parquet):
    ds = FusionDataset([sample_parquet], seq_len=64, target_horizon=10)
    delta, vol, vb, ind_tokens, label = ds[0]
    assert delta.shape == (64,)
    assert vol.shape == (64,)
    assert vb.shape == (64,)
    assert isinstance(ind_tokens, dict)
    assert len(ind_tokens) == 6
    for arr in ind_tokens.values():
        assert arr.shape == (64,)
    assert label in (0, 1, 2)


def test_cls_at_zero(sample_parquet):
    ds = FusionDataset([sample_parquet], seq_len=64, target_horizon=10)
    delta, _, _, _, _ = ds[0]
    assert delta[0] == 1


def test_labels_valid(sample_parquet):
    ds = FusionDataset([sample_parquet], seq_len=64, target_horizon=10)
    for i in range(min(10, len(ds))):
        _, _, _, _, label = ds[i]
        assert label in (0, 1, 2)
