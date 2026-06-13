"""End-to-end smoke test: data loading -> tokenization -> model forward -> backtest."""
import numpy as np
import torch
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from dataset.klines_dataset import KlinesDataset
from models.price_transformer import PriceTransformer
from backtest.engine import BacktestEngine
from training.trainer import Trainer
from torch.utils.data import DataLoader


@pytest.fixture
def mock_data_dir(tmp_path):
    symbol_dir = tmp_path / "BTCUSDT" / "klines_1m"
    symbol_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    n = 500
    for month in ["2025-01", "2025-02", "2025-03"]:
        base = 30000.0 + rng.standard_normal() * 100
        close = base + np.cumsum(rng.standard_normal(n) * 5).astype(np.float32)
        table = pa.table({
            "timestamp": np.arange(n, dtype=np.int32),
            "open": (close - rng.standard_normal(n) * 2).astype(np.float32),
            "high": (close + np.abs(rng.standard_normal(n)) * 5).astype(np.float32),
            "low": (close - np.abs(rng.standard_normal(n)) * 5).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": (np.abs(rng.standard_normal(n)) * 100 + 10).astype(np.float32),
        })
        pq.write_table(table, symbol_dir / f"{month}.parquet")
    return tmp_path


def test_full_pipeline(mock_data_dir):
    files = sorted((mock_data_dir / "BTCUSDT" / "klines_1m").glob("*.parquet"))
    seq_len = 32
    horizon = 10

    ds = KlinesDataset(files, seq_len=seq_len, target_horizon=horizon, target_threshold=0.001)
    assert len(ds) > 0

    delta, vol, vb, label = ds[0]
    assert delta.shape == (seq_len,)

    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=16, bucket_emb_dim=8,
        hidden_dim=32, num_layers=1, num_heads=2,
        ffn_dim=64, dropout=0.0, num_classes=3, seq_len=seq_len,
    )

    delta_t = torch.tensor(delta).unsqueeze(0)
    vol_t = torch.tensor(vol).unsqueeze(0)
    vb_t = torch.tensor(vb).unsqueeze(0)
    logits = model(delta_t, vol_t, vb_t)
    assert logits.shape == (1, 3)

    dl = DataLoader(ds, batch_size=8, shuffle=False)
    trainer = Trainer(
        model=model, train_loader=dl, val_loader=dl,
        epochs=1, lr=1e-3, device="cpu",
        checkpoint_dir=mock_data_dir / "ckpts",
    )
    metrics = trainer.train()
    assert len(metrics) == 1

    closes = ds.closes
    predictions = np.random.default_rng(42).integers(0, 3, len(closes))
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 0
