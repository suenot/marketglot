import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from dataset.multimodal_dataset import MultimodalDataset

@pytest.fixture
def parquet(tmp_path):
    n = 500; rng = np.random.default_rng(42)
    c = (30000 + np.cumsum(rng.standard_normal(n)*10)).astype(np.float32)
    t = pa.table({"timestamp": np.arange(n,dtype=np.int32), "open": (c-rng.standard_normal(n)*2).astype(np.float32),
        "high": (c+np.abs(rng.standard_normal(n))*5).astype(np.float32), "low": (c-np.abs(rng.standard_normal(n))*5).astype(np.float32),
        "close": c.astype(np.float32), "volume": (np.abs(rng.standard_normal(n))*100+10).astype(np.float32)})
    pq.write_table(t, tmp_path / "2023-02.parquet")
    return tmp_path / "2023-02.parquet"

def test_length(parquet):
    assert len(MultimodalDataset([parquet], seq_len=64, target_horizon=10)) == 500-64-10

def test_shapes(parquet):
    ds = MultimodalDataset([parquet], seq_len=64, target_horizon=10)
    delta, vol, vb, ind, label = ds[0]
    assert delta.shape == (64,) and vol.shape == (64,) and vb.shape == (64,)
    assert isinstance(ind, dict) and len(ind) == 6
    for a in ind.values(): assert a.shape == (64,)
    assert label in (0,1,2)

def test_cls(parquet):
    delta = MultimodalDataset([parquet], seq_len=64, target_horizon=10)[0][0]
    assert delta[0] == 1
