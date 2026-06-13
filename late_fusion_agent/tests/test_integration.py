import numpy as np
import torch
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataset.fusion_dataset import FusionDataset
from models.indicator_model import IndicatorModel
from models.meta_model import MetaModel
from torch.utils.data import DataLoader


@pytest.fixture
def mock_parquet(tmp_path):
    n = 300
    rng = np.random.default_rng(42)
    close = (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32)
    table = pa.table({
        "timestamp": np.arange(n, dtype=np.int32),
        "open": (close - rng.standard_normal(n)).astype(np.float32),
        "high": (close + np.abs(rng.standard_normal(n)) * 3).astype(np.float32),
        "low": (close - np.abs(rng.standard_normal(n)) * 3).astype(np.float32),
        "close": close.astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 50 + 10).astype(np.float32),
    })
    path = tmp_path / "2025-01.parquet"
    pq.write_table(table, path)
    return path


INDICATOR_KEYS = ["rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"]


def _collate(batch):
    delta = torch.stack([torch.tensor(b[0]) for b in batch])
    vol = torch.stack([torch.tensor(b[1]) for b in batch])
    vb = torch.stack([torch.tensor(b[2]) for b in batch])
    keys = list(batch[0][3].keys())
    ind = [torch.stack([torch.tensor(b[3][k]) for b in batch]) for k in keys]
    labels = torch.tensor([b[4] for b in batch])
    return delta, vol, vb, ind, labels


def test_full_pipeline(mock_parquet, tmp_path):
    seq_len = 32
    ds = FusionDataset([mock_parquet], seq_len=seq_len, target_horizon=10, target_threshold=0.001)
    assert len(ds) > 0

    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=_collate)
    batch = next(iter(dl))
    delta, vol, vb, ind, labels = batch

    # Model A (PriceTransformer) — import from sibling project
    import sys, importlib
    tft_path = str(Path(__file__).resolve().parent.parent.parent / "token_first_transformer")
    if tft_path not in sys.path:
        sys.path.append(tft_path)
    # Import PriceTransformer directly from the token_first_transformer models package
    spec = importlib.util.spec_from_file_location(
        "tft_models_price_transformer",
        str(Path(tft_path) / "models" / "price_transformer.py"),
    )
    tft_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tft_models)
    PriceTransformer = tft_models.PriceTransformer
    model_a = PriceTransformer(delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=16, bucket_emb_dim=8, hidden_dim=32, num_layers=1, num_heads=2,
        ffn_dim=64, dropout=0.0, num_classes=3, seq_len=seq_len)
    logits_a = model_a(delta, vol, vb)
    assert logits_a.shape == (delta.shape[0], 3)

    # Model B
    model_b = IndicatorModel(vocab_sizes=[7,9,7,8,7,7], emb_dim=8, hidden_dim=32,
        num_layers=1, num_heads=2, ffn_dim=64, dropout=0.0, num_classes=3, seq_len=seq_len)
    logits_b = model_b(ind)
    assert logits_b.shape == (delta.shape[0], 3)

    # Meta
    meta = MetaModel(input_dim=6, hidden_dim=16, num_classes=3)
    fused = meta(logits_a.detach(), logits_b.detach())
    assert fused.shape == (delta.shape[0], 3)
