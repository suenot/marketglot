import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataset.klines_dataset import KlinesDataset, fit_tokenizers, make_split


@pytest.fixture
def sample_parquet(tmp_path):
    n = 500
    rng = np.random.default_rng(42)
    base_price = 30000.0
    close = base_price + np.cumsum(rng.standard_normal(n) * 10).astype(np.float32)
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
    seq_len = 64
    horizon = 10
    ds = KlinesDataset(
        file_paths=[sample_parquet],
        seq_len=seq_len,
        target_horizon=horizon,
    )
    assert len(ds) == 500 - seq_len - horizon


def test_dataset_item_shape(sample_parquet):
    ds = KlinesDataset(file_paths=[sample_parquet], seq_len=64, target_horizon=10)
    delta_ids, vol_ids, vb_ids, label = ds[0]
    assert delta_ids.shape == (64,)
    assert vol_ids.shape == (64,)
    assert vb_ids.shape == (64,)
    assert label in (0, 1, 2)


def test_dataset_cls_at_position_zero(sample_parquet):
    ds = KlinesDataset(file_paths=[sample_parquet], seq_len=64, target_horizon=10)
    delta_ids, _, _, _ = ds[0]
    assert delta_ids[0] == 1


def test_dataset_no_lookahead(sample_parquet):
    ds = KlinesDataset(file_paths=[sample_parquet], seq_len=64, target_horizon=10)
    for i in range(min(10, len(ds))):
        _, _, _, label = ds[i]
        assert label in (0, 1, 2)


def test_fit_tokenizers(sample_parquet):
    delta_tok, vol_tok, vb_tok = fit_tokenizers([sample_parquet])
    assert delta_tok.vocab_size == 122
    assert vol_tok.boundaries is not None
    assert vb_tok.boundaries is not None


def test_make_split(sample_parquet, tmp_path):
    klines_dir = tmp_path / "BTCUSDT" / "klines_1m"
    klines_dir.mkdir(parents=True)
    import shutil
    shutil.copy(sample_parquet, klines_dir / "2023-02.parquet")
    data_dir = tmp_path
    split = make_split(data_dir, "2023-02", "2023-02")
    assert len(split) == 1
    assert split[0].name == "2023-02.parquet"
