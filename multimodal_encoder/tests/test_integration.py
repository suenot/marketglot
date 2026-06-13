import numpy as np
import torch
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataset.multimodal_dataset import MultimodalDataset
from models.multimodal_model import MultimodalEncoder
from torch.utils.data import DataLoader


@pytest.fixture
def mock_parquet(tmp_path):
    n = 300; rng = np.random.default_rng(42)
    c = (30000 + np.cumsum(rng.standard_normal(n)*5)).astype(np.float32)
    t = pa.table({"timestamp": np.arange(n,dtype=np.int32), "open": (c-rng.standard_normal(n)).astype(np.float32),
        "high": (c+np.abs(rng.standard_normal(n))*3).astype(np.float32), "low": (c-np.abs(rng.standard_normal(n))*3).astype(np.float32),
        "close": c.astype(np.float32), "volume": (np.abs(rng.standard_normal(n))*50+10).astype(np.float32)})
    pq.write_table(t, tmp_path / "2025-01.parquet")
    return tmp_path / "2025-01.parquet"


def collate(batch):
    delta = torch.stack([torch.tensor(b[0]) for b in batch])
    vol = torch.stack([torch.tensor(b[1]) for b in batch])
    vb = torch.stack([torch.tensor(b[2]) for b in batch])
    keys = list(batch[0][3].keys())
    ind = [torch.stack([torch.tensor(b[3][k]) for b in batch]) for k in keys]
    y = torch.tensor([b[4] for b in batch])
    return delta, vol, vb, ind, y


def test_pipeline(mock_parquet, tmp_path):
    ds = MultimodalDataset([mock_parquet], seq_len=32, target_horizon=10, target_threshold=0.001)
    assert len(ds) > 0
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    delta, vol, vb, ind, y = next(iter(dl))

    model = MultimodalEncoder(delta_vocab_size=122, bucket_vocab_size=10, delta_emb_dim=16, bucket_emb_dim=8,
        candle_proj_dim=32, ind_emb_dim=8, ind_proj_dim=32, hidden_dim=64, num_layers=1, num_heads=2,
        ffn_dim=128, dropout=0.0, num_classes=3, seq_len=32)
    logits = model(delta, vol, vb, ind)
    assert logits.shape == (delta.shape[0], 3)

    # Quick train step
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    assert loss.item() > 0
