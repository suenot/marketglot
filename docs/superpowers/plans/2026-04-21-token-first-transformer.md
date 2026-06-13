# Token-First Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a token-based transformer classifier that predicts BTCUSDT 1m price direction (UP/FLAT/DOWN) over 60 candles.

**Architecture:** Price deltas discretized into tokens, combined with volatility/volume bucket tokens, fed through a 4-layer transformer encoder. CLS token pooling produces a 3-class prediction. Streaming dataset reads monthly parquet files without loading all into RAM.

**Tech Stack:** Python 3.11+, PyTorch >= 2.1, PyArrow, Polars, NumPy, PyYAML, scikit-learn

**Spec:** `docs/superpowers/specs/2026-04-21-token-transformer-trading-agent-design.md`

---

## File Structure

```
w_training/token_first_transformer/
├── pyproject.toml
├── configs/
│   └── default.yaml
├── tokenizer/
│   ├── __init__.py
│   ├── delta_tokenizer.py
│   └── bucket_tokenizer.py
├── dataset/
│   ├── __init__.py
│   └── klines_dataset.py
├── models/
│   ├── __init__.py
│   └── price_transformer.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── backtest/
│   ├── __init__.py
│   └── engine.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── backtest.py
├── checkpoints/           (gitignored)
├── logs/                  (gitignored)
└── tests/
    ├── __init__.py
    ├── test_delta_tokenizer.py
    ├── test_bucket_tokenizer.py
    ├── test_dataset.py
    ├── test_model.py
    ├── test_trainer.py
    ├── test_backtest.py
    └── test_integration.py
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `token_first_transformer/pyproject.toml`
- Create: `token_first_transformer/configs/default.yaml`
- Create: `token_first_transformer/tokenizer/__init__.py`
- Create: `token_first_transformer/dataset/__init__.py`
- Create: `token_first_transformer/models/__init__.py`
- Create: `token_first_transformer/training/__init__.py`
- Create: `token_first_transformer/backtest/__init__.py`
- Create: `token_first_transformer/tests/__init__.py`
- Create: `token_first_transformer/checkpoints/.gitkeep`
- Create: `token_first_transformer/logs/.gitkeep`
- Create: `token_first_transformer/.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/suenot/projects/w_trading/w_training
mkdir -p token_first_transformer/{configs,tokenizer,dataset,models,training,backtest,scripts,tests,checkpoints,logs}
touch token_first_transformer/{tokenizer,dataset,models,training,backtest,tests}/__init__.py
touch token_first_transformer/checkpoints/.gitkeep token_first_transformer/logs/.gitkeep
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "token-first-transformer"
version = "0.1.0"
description = "Token-based transformer trading agent"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "pyarrow>=15.0",
    "polars>=0.20",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "scikit-learn>=1.3",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write configs/default.yaml**

```yaml
data:
  symbol: BTCUSDT
  timeframe: 1m
  data_dir: ../../../w_trender/backtests/data
  train_months: ["2023-02", "2025-06"]
  val_months: ["2025-07", "2025-10"]
  test_months: ["2025-11", "2026-02"]

tokenizer:
  delta:
    range_pct: 3.0
    step_pct: 0.05
  bucket:
    n_bins: 8

sequence:
  length: 128
  target_horizon: 60
  target_threshold: 0.0015  # 0.15%

model:
  delta_vocab_size: 122   # 120 bins + PAD(0) + CLS(1)
  bucket_vocab_size: 10   # 8 bins + PAD(0) + special(9)
  delta_emb_dim: 64
  bucket_emb_dim: 16
  hidden_dim: 256
  num_layers: 4
  num_heads: 8
  ffn_dim: 1024
  dropout: 0.1
  num_classes: 3  # DOWN=0, FLAT=1, UP=2

training:
  batch_size: 64
  grad_accum_steps: 2
  learning_rate: 3.0e-4
  weight_decay: 0.01
  epochs: 10
  early_stop_patience: 3
  device: auto  # "mps", "cpu", or "auto"
  seed: 42
  checkpoint_dir: checkpoints
  log_dir: logs

backtest:
  commission: 0.0004  # 0.04% per side
  stop_loss: -0.005   # -0.5%
  take_profit: 0.01   # +1.0%
  max_hold: 60
```

- [ ] **Step 4: Write .gitignore**

```
checkpoints/*.pt
logs/
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 5: Verify structure and commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
find token_first_transformer -type f | sort
git add token_first_transformer/
git commit -m "feat: scaffold token_first_transformer project structure"
```

---

### Task 2: DeltaTokenizer

**Files:**
- Create: `token_first_transformer/tokenizer/delta_tokenizer.py`
- Create: `token_first_transformer/tests/test_delta_tokenizer.py`

- [ ] **Step 1: Write test_delta_tokenizer.py**

```python
import numpy as np
import pytest
from tokenizer.delta_tokenizer import DeltaTokenizer


def test_vocab_size():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    assert tok.vocab_size == 122


def test_pad_and_cls_ids():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    assert tok.pad_id == 0
    assert tok.cls_id == 1


def test_zero_delta_maps_to_middle_bin():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    token_id = tok.encode_single(0.0)
    assert token_id == 1 + 60


def test_positive_delta_encodes():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    token_id = tok.encode_single(0.0005)
    assert token_id == 62


def test_negative_delta_encodes():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    token_id = tok.encode_single(-0.0005)
    assert token_id == 60


def test_out_of_range_clips():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    high = tok.encode_single(0.05)
    assert high == tok.vocab_size - 1
    low = tok.encode_single(-0.05)
    assert low == 2


def test_encode_batch():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    deltas = np.array([0.0, 0.0005, -0.0005, 0.05, -0.05], dtype=np.float32)
    ids = tok.encode_batch(deltas)
    assert ids.shape == (5,)
    assert ids[0] == 1 + 60
    assert ids[1] == 62
    assert ids[2] == 60
    assert ids[3] == tok.vocab_size - 1
    assert ids[4] == 2


def test_from_closes():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    closes = np.array([100.0, 101.0, 99.0, 100.5, 100.5], dtype=np.float32)
    ids = tok.from_closes(closes)
    assert ids[0] == tok.pad_id
    assert len(ids) == 5
    assert ids[4] == 1 + 60
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_delta_tokenizer.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write tokenizer/delta_tokenizer.py**

```python
from __future__ import annotations

import numpy as np


class DeltaTokenizer:
    """Tokenizes percentage price deltas into discrete bins.

    Vocabulary layout:
        0 = PAD
        1 = CLS
        2 .. vocab_size-1 = quantized delta bins
    """

    def __init__(self, range_pct: float = 3.0, step_pct: float = 0.05) -> None:
        self.range_pct = range_pct
        self.step_pct = step_pct
        self.n_bins = int(2 * range_pct / step_pct)
        self.pad_id = 0
        self.cls_id = 1
        self._offset = 2
        self.vocab_size = self._offset + self.n_bins

    def encode_single(self, delta: float) -> int:
        delta_pct = delta * 100
        bin_idx = round(delta_pct / self.step_pct) + self.n_bins // 2
        bin_idx = max(0, min(self.n_bins - 1, bin_idx))
        return self._offset + bin_idx

    def encode_batch(self, deltas: np.ndarray) -> np.ndarray:
        deltas_pct = deltas * 100
        bin_idx = np.round(deltas_pct / self.step_pct).astype(np.int32) + self.n_bins // 2
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
        return (bin_idx + self._offset).astype(np.int32)

    def from_closes(self, closes: np.ndarray) -> np.ndarray:
        n = len(closes)
        ids = np.full(n, self.pad_id, dtype=np.int32)
        if n < 2:
            return ids
        deltas = (closes[1:] - closes[:-1]) / closes[:-1]
        ids[1:] = self.encode_batch(deltas)
        return ids
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_delta_tokenizer.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/tokenizer/delta_tokenizer.py token_first_transformer/tests/test_delta_tokenizer.py
git commit -m "feat: add DeltaTokenizer with tests"
```

---

### Task 3: BucketTokenizer

**Files:**
- Create: `token_first_transformer/tokenizer/bucket_tokenizer.py`
- Create: `token_first_transformer/tests/test_bucket_tokenizer.py`

- [ ] **Step 1: Write test_bucket_tokenizer.py**

```python
import numpy as np
import pytest
from tokenizer.bucket_tokenizer import BucketTokenizer


def test_default_vocab_size():
    tok = BucketTokenizer(n_bins=8)
    assert tok.vocab_size == 10


def test_pad_id():
    tok = BucketTokenizer(n_bins=8)
    assert tok.pad_id == 0


def test_fit_creates_boundaries():
    tok = BucketTokenizer(n_bins=4)
    data = np.arange(100, dtype=np.float32)
    tok.fit(data)
    assert tok.boundaries is not None
    assert len(tok.boundaries) == 3


def test_encode_after_fit():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    tok.fit(data)
    ids = tok.encode_batch(data)
    assert ids.min() >= 2
    assert ids.max() <= 5
    assert ids[0] == 2
    assert ids[-1] == 5


def test_encode_single():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tok.fit(data)
    token = tok.encode_single(5.0)
    assert 2 <= token <= 5


def test_encode_below_min():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tok.fit(data)
    assert tok.encode_single(5.0) == 2


def test_encode_above_max():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tok.fit(data)
    assert tok.encode_single(50.0) == 5


def test_save_load(tmp_path):
    tok = BucketTokenizer(n_bins=4)
    data = np.arange(100, dtype=np.float32)
    tok.fit(data)
    path = tmp_path / "bounds.npy"
    tok.save(path)
    tok2 = BucketTokenizer(n_bins=4)
    tok2.load(path)
    np.testing.assert_array_equal(tok.boundaries, tok2.boundaries)
    sample = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    np.testing.assert_array_equal(tok.encode_batch(sample), tok2.encode_batch(sample))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_bucket_tokenizer.py -v
```

Expected: FAIL

- [ ] **Step 3: Write tokenizer/bucket_tokenizer.py**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np


class BucketTokenizer:
    """Quantile-based bucket tokenizer for volatility / volume features.

    Vocabulary layout:
        0 = PAD
        1 = CLS (reserved)
        2 .. n_bins+1 = quantile bins
    """

    def __init__(self, n_bins: int = 8) -> None:
        self.n_bins = n_bins
        self.pad_id = 0
        self._offset = 2
        self.vocab_size = self._offset + n_bins
        self.boundaries: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.boundaries = np.percentile(values, quantiles).astype(np.float32)

    def encode_single(self, value: float) -> int:
        assert self.boundaries is not None, "Call fit() first"
        return int(np.searchsorted(self.boundaries, value, side="right")) + self._offset

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None, "Call fit() first"
        bin_idx = np.searchsorted(self.boundaries, values, side="right")
        return (bin_idx + self._offset).astype(np.int32)

    def save(self, path: Path) -> None:
        assert self.boundaries is not None
        np.save(path, self.boundaries)

    def load(self, path: Path) -> None:
        self.boundaries = np.load(path)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_bucket_tokenizer.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/tokenizer/bucket_tokenizer.py token_first_transformer/tests/test_bucket_tokenizer.py
git commit -m "feat: add BucketTokenizer with quantile fitting and save/load"
```

---

### Task 4: KlinesDataset

**Files:**
- Create: `token_first_transformer/dataset/klines_dataset.py`
- Create: `token_first_transformer/tests/test_dataset.py`

- [ ] **Step 1: Write test_dataset.py**

```python
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
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
    data_dir = tmp_path
    split = make_split(data_dir, "2023-02", "2023-02")
    assert len(split) == 1
    assert split[0].name == "2023-02.parquet"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_dataset.py -v
```

Expected: FAIL

- [ ] **Step 3: Write dataset/klines_dataset.py**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tokenizer.delta_tokenizer import DeltaTokenizer
from tokenizer.bucket_tokenizer import BucketTokenizer


def _load_month(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {col: table[col].to_numpy() for col in table.column_names}


def fit_tokenizers(
    file_paths: list[Path],
    range_pct: float = 3.0,
    step_pct: float = 0.05,
    n_bins: int = 8,
) -> tuple[DeltaTokenizer, BucketTokenizer, BucketTokenizer]:
    delta_tok = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    all_range_pct = []
    all_log_vol = []
    for p in file_paths:
        d = _load_month(p)
        closes = d["close"]
        if len(closes) < 2:
            continue
        highs, lows = d["high"], d["low"]
        ranges = (highs[1:] - lows[1:]) / closes[1:]
        all_range_pct.append(ranges)
        log_vols = np.log1p(d["volume"][1:])
        all_log_vol.append(log_vols)
    range_arr = np.concatenate(all_range_pct)
    vol_arr = np.concatenate(all_log_vol)
    vol_tok = BucketTokenizer(n_bins=n_bins)
    vol_tok.fit(range_arr)
    vb_tok = BucketTokenizer(n_bins=n_bins)
    vb_tok.fit(vol_arr)
    return delta_tok, vol_tok, vb_tok


def make_split(data_dir: Path, start_month: str, end_month: str) -> list[Path]:
    klines_dir = data_dir / "BTCUSDT" / "klines_1m"
    if not klines_dir.exists():
        raise FileNotFoundError(f"No klines_1m directory at {klines_dir}")
    files = sorted(klines_dir.glob("*.parquet"))
    return [f for f in files if start_month <= f.stem <= end_month]


class KlinesDataset:
    def __init__(
        self,
        file_paths: list[Path],
        seq_len: int = 128,
        target_horizon: int = 60,
        target_threshold: float = 0.0015,
        range_pct: float = 3.0,
        step_pct: float = 0.05,
        n_bins: int = 8,
    ) -> None:
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold

        self.delta_tok, self.vol_tok, self.vb_tok = fit_tokenizers(
            file_paths, range_pct, step_pct, n_bins
        )
        self._load_data(file_paths)

    def _load_data(self, file_paths: list[Path]) -> None:
        frames = [_load_month(p) for p in file_paths]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        n = len(self.closes)
        self._len = max(0, n - self.seq_len - self.target_horizon)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        start = idx
        end = start + self.seq_len
        closes = self.closes[start:end]
        highs = self.highs[start:end]
        lows = self.lows[start:end]
        vols = self.volumes[start:end]

        delta_ids = self.delta_tok.from_closes(closes)
        delta_ids[0] = self.delta_tok.cls_id

        range_pct = np.zeros(self.seq_len, dtype=np.float32)
        range_pct[1:] = (highs[1:] - lows[1:]) / closes[1:]
        vol_ids = self.vol_tok.encode_batch(range_pct)
        vol_ids[0] = self.vol_tok.pad_id

        log_vol = np.log1p(vols)
        vb_ids = self.vb_tok.encode_batch(log_vol)
        vb_ids[0] = self.vb_tok.pad_id

        target_close = self.closes[end + self.target_horizon - 1]
        current_close = self.closes[end - 1]
        delta = (target_close - current_close) / current_close

        if delta > self.target_threshold:
            label = 2
        elif delta < -self.target_threshold:
            label = 0
        else:
            label = 1

        return delta_ids, vol_ids, vb_ids, label
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_dataset.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/dataset/klines_dataset.py token_first_transformer/tests/test_dataset.py
git commit -m "feat: add KlinesDataset with streaming parquet and tokenizer fitting"
```

---

### Task 5: PriceTransformer Model

**Files:**
- Create: `token_first_transformer/models/price_transformer.py`
- Create: `token_first_transformer/tests/test_model.py`

- [ ] **Step 1: Write test_model.py**

```python
import torch
import pytest
from models.price_transformer import PriceTransformer


def test_output_shape():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=64, bucket_emb_dim=16,
        hidden_dim=256, num_layers=4, num_heads=8,
        ffn_dim=1024, dropout=0.1, num_classes=3, seq_len=128,
    )
    batch = 4
    delta = torch.randint(2, 121, (batch, 128))
    vol = torch.randint(2, 9, (batch, 128))
    vb = torch.randint(2, 9, (batch, 128))
    logits = model(delta, vol, vb)
    assert logits.shape == (batch, 3)


def test_single_input():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=64, bucket_emb_dim=16,
        hidden_dim=256, num_layers=2, num_heads=4,
        ffn_dim=512, dropout=0.0, num_classes=3, seq_len=64,
    )
    delta = torch.randint(2, 121, (1, 64))
    vol = torch.randint(2, 9, (1, 64))
    vb = torch.randint(2, 9, (1, 64))
    logits = model(delta, vol, vb)
    assert logits.shape == (1, 3)


def test_gradients_flow():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=32, bucket_emb_dim=8,
        hidden_dim=64, num_layers=1, num_heads=2,
        ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16,
    )
    delta = torch.randint(2, 121, (2, 16))
    vol = torch.randint(2, 9, (2, 16))
    vb = torch.randint(2, 9, (2, 16))
    logits = model(delta, vol, vb)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_cls_position_output_finite():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=32, bucket_emb_dim=8,
        hidden_dim=64, num_layers=1, num_heads=2,
        ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16,
    )
    model.eval()
    with torch.no_grad():
        delta = torch.randint(2, 121, (1, 16))
        vol = torch.randint(2, 9, (1, 16))
        vb = torch.randint(2, 9, (1, 16))
        logits = model(delta, vol, vb)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_model.py -v
```

Expected: FAIL

- [ ] **Step 3: Write models/price_transformer.py**

```python
from __future__ import annotations

import torch
import torch.nn as nn


class PriceTransformer(nn.Module):
    def __init__(
        self,
        delta_vocab_size: int = 122,
        bucket_vocab_size: int = 10,
        delta_emb_dim: int = 64,
        bucket_emb_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.delta_emb = nn.Embedding(delta_vocab_size, delta_emb_dim)
        self.vol_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        self.vb_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)

        concat_dim = delta_emb_dim + bucket_emb_dim * 2
        self.proj = nn.Linear(concat_dim, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self, delta_ids: torch.Tensor, vol_ids: torch.Tensor, vb_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T = delta_ids.shape
        d = self.delta_emb(delta_ids)
        v = self.vol_emb(vol_ids)
        vb = self.vb_emb(vb_ids)
        x = torch.cat([d, v, vb], dim=-1)
        x = self.proj(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)
        x = self.transformer(x)
        x = self.layer_norm(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_model.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/models/price_transformer.py token_first_transformer/tests/test_model.py
git commit -m "feat: add PriceTransformer model with CLS pooling"
```

---

### Task 6: Trainer

**Files:**
- Create: `token_first_transformer/training/trainer.py`
- Create: `token_first_transformer/tests/test_trainer.py`

- [ ] **Step 1: Write test_trainer.py**

```python
import torch
import pytest
from training.trainer import Trainer, compute_class_weights
from models.price_transformer import PriceTransformer


def _make_model():
    return PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=16, bucket_emb_dim=8,
        hidden_dim=32, num_layers=1, num_heads=2,
        ffn_dim=64, dropout=0.0, num_classes=3, seq_len=16,
    )


def _make_dataloader(n=32, seq_len=16):
    delta = torch.randint(2, 121, (n, seq_len))
    vol = torch.randint(2, 9, (n, seq_len))
    vb = torch.randint(2, 9, (n, seq_len))
    labels = torch.randint(0, 3, (n,))
    return torch.utils.data.DataLoader(
        list(zip(delta, vol, vb, labels)), batch_size=8,
    )


def test_compute_class_weights():
    labels = [0, 0, 1, 1, 1, 1, 2, 2]
    weights = compute_class_weights(labels, num_classes=3)
    assert len(weights) == 3
    assert weights[1] < weights[0]
    assert weights[1] < weights[2]


def test_trainer_one_epoch(tmp_path):
    model = _make_model()
    train_dl = _make_dataloader()
    val_dl = _make_dataloader(n=16)
    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=1, lr=1e-3, device="cpu", checkpoint_dir=tmp_path,
    )
    metrics = trainer.train()
    assert len(metrics) == 1
    assert "train_loss" in metrics[0]
    assert "val_loss" in metrics[0]
    assert "val_f1" in metrics[0]


def test_trainer_saves_checkpoint(tmp_path):
    model = _make_model()
    train_dl = _make_dataloader()
    val_dl = _make_dataloader(n=8)
    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=1, lr=1e-3, device="cpu", checkpoint_dir=tmp_path,
    )
    trainer.train()
    ckpts = list(tmp_path.glob("*.pt"))
    assert len(ckpts) >= 1


def test_trainer_early_stop(tmp_path):
    model = _make_model()
    train_dl = _make_dataloader()
    val_dl = _make_dataloader(n=8)
    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=50, lr=1e-3, device="cpu", checkpoint_dir=tmp_path,
        early_stop_patience=2,
    )
    metrics = trainer.train()
    assert len(metrics) <= 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_trainer.py -v
```

Expected: FAIL

- [ ] **Step 3: Write training/trainer.py**

```python
from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import f1_score

from models.price_transformer import PriceTransformer


def compute_class_weights(labels: list[int], num_classes: int = 3) -> list[float]:
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return weights


class Trainer:
    def __init__(
        self,
        model: PriceTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: list[float] | None = None,
        epochs: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 1,
        early_stop_patience: int = 3,
        device: str = "auto",
        checkpoint_dir: Path = Path("checkpoints"),
    ) -> None:
        if device == "auto":
            self.device = self._auto_device()
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.early_stop_patience = early_stop_patience
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weights_tensor = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

    @staticmethod
    def _auto_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def train(self) -> list[dict]:
        best_f1 = -1.0
        patience_counter = 0
        all_metrics = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_f1 = self._val_epoch()

            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_f1, 4),
            }
            all_metrics.append(metrics)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self._save_checkpoint("best.pt", epoch)
            else:
                patience_counter += 1

            self._save_checkpoint(f"epoch_{epoch}.pt", epoch)

            if patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            self.scheduler.step()

        return all_metrics

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for i, (delta, vol, vb, labels) in enumerate(self.train_loader):
            delta = delta.to(self.device)
            vol = vol.to(self.device)
            vb = vb.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(delta, vol, vb)
            loss = self.criterion(logits, labels) / self.grad_accum_steps
            loss.backward()

            if (i + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for delta, vol, vb, labels in self.val_loader:
                delta = delta.to(self.device)
                vol = vol.to(self.device)
                vb = vb.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(delta, vol, vb)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, float(f1)

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        path = self.checkpoint_dir / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_trainer.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/training/trainer.py token_first_transformer/tests/test_trainer.py
git commit -m "feat: add Trainer with early stopping, checkpoints, class weights"
```

---

### Task 7: Backtest Engine

**Files:**
- Create: `token_first_transformer/backtest/engine.py`
- Create: `token_first_transformer/tests/test_backtest.py`

- [ ] **Step 1: Write test_backtest.py**

```python
import numpy as np
import pytest
from backtest.engine import BacktestEngine, BacktestResult


def test_long_win():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    closes = np.array([100.0, 100.1, 100.3, 100.5, 100.8, 101.0, 101.2, 101.3, 101.4, 101.5], dtype=np.float32)
    predictions = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 1
    assert result.win_rate > 0
    assert result.total_pnl > 0


def test_short_loss():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    closes = np.array([100.0, 100.2, 100.4, 100.6, 100.8, 101.0], dtype=np.float32)
    predictions = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 1
    assert result.total_pnl < 0


def test_flat_no_trade():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    closes = np.array([100.0, 100.1, 100.2, 100.1, 100.0, 99.9], dtype=np.float32)
    predictions = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count == 0
    assert result.total_pnl == 0.0


def test_max_hold_exit():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.1, take_profit=0.1, max_hold=3)
    closes = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6], dtype=np.float32)
    predictions = np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 1


def test_result_metrics():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    rng = np.random.default_rng(42)
    n = 200
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.2).astype(np.float32)
    predictions = rng.integers(0, 3, n).astype(np.int32)
    result = engine.run(closes, predictions)
    assert isinstance(result.total_pnl, float)
    assert isinstance(result.sharpe, float)
    assert isinstance(result.max_drawdown, float)
    assert 0.0 <= result.win_rate <= 1.0
    assert isinstance(result.trade_count, int)
    assert isinstance(result.profit_factor, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_backtest.py -v
```

Expected: FAIL

- [ ] **Step 3: Write backtest/engine.py**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    pnl: float


@dataclass
class BacktestResult:
    total_pnl: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    profit_factor: float
    avg_duration: float
    trades: list[Trade]


class BacktestEngine:
    def __init__(
        self,
        commission: float = 0.0004,
        stop_loss: float = -0.005,
        take_profit: float = 0.01,
        max_hold: int = 60,
    ) -> None:
        self.commission = commission
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold = max_hold

    def run(self, closes: np.ndarray, predictions: np.ndarray) -> BacktestResult:
        trades: list[Trade] = []
        in_position = False
        direction = 0
        entry_idx = 0
        entry_price = 0.0

        for i in range(len(predictions)):
            if in_position:
                pnl_pct = direction * (closes[i] - entry_price) / entry_price
                hold_duration = i - entry_idx
                exit = False
                if pnl_pct <= self.stop_loss:
                    exit = True
                elif pnl_pct >= self.take_profit:
                    exit = True
                elif hold_duration >= self.max_hold:
                    exit = True

                if exit:
                    trade_pnl = pnl_pct - 2 * self.commission
                    trades.append(Trade(
                        entry_idx=entry_idx, exit_idx=i,
                        direction=direction, entry_price=entry_price,
                        exit_price=float(closes[i]), pnl=trade_pnl,
                    ))
                    in_position = False

            if not in_position and i + 1 < len(closes):
                pred = predictions[i]
                if pred == 2:
                    direction = 1
                elif pred == 0:
                    direction = -1
                else:
                    continue
                in_position = True
                entry_idx = i + 1
                entry_price = float(closes[i + 1])

        return self._compute_metrics(trades)

    def _compute_metrics(self, trades: list[Trade]) -> BacktestResult:
        if not trades:
            return BacktestResult(
                total_pnl=0.0, sharpe=0.0, max_drawdown=0.0,
                win_rate=0.0, trade_count=0, profit_factor=0.0,
                avg_duration=0.0, trades=trades,
            )

        pnls = [t.pnl for t in trades]
        total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(trades)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        sharpe = 0.0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 1440))

        avg_duration = float(np.mean([t.exit_idx - t.entry_idx for t in trades]))

        return BacktestResult(
            total_pnl=total_pnl, sharpe=sharpe, max_drawdown=max_drawdown,
            win_rate=win_rate, trade_count=len(trades),
            profit_factor=profit_factor, avg_duration=avg_duration, trades=trades,
        )
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_backtest.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/backtest/engine.py token_first_transformer/tests/test_backtest.py
git commit -m "feat: add BacktestEngine with SL/TP/max-hold and metrics"
```

---

### Task 8: CLI Scripts

**Files:**
- Create: `token_first_transformer/scripts/train.py`
- Create: `token_first_transformer/scripts/evaluate.py`
- Create: `token_first_transformer/scripts/backtest.py`

- [ ] **Step 1: Write scripts/train.py**

```python
"""Train the token-first transformer model.

Usage: python scripts/train.py [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from dataset.klines_dataset import KlinesDataset, make_split
from training.trainer import Trainer, compute_class_weights
from models.price_transformer import PriceTransformer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    train_files = make_split(data_dir, *cfg["data"]["train_months"])
    val_files = make_split(data_dir, *cfg["data"]["val_months"])
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    seq_cfg = cfg["sequence"]
    tok_cfg = cfg["tokenizer"]
    train_ds = KlinesDataset(
        train_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )
    val_ds = KlinesDataset(
        val_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )

    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model_cfg = cfg["model"]
    model = PriceTransformer(
        delta_vocab_size=model_cfg["delta_vocab_size"],
        bucket_vocab_size=model_cfg["bucket_vocab_size"],
        delta_emb_dim=model_cfg["delta_emb_dim"],
        bucket_emb_dim=model_cfg["bucket_emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
        seq_len=seq_cfg["length"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_accum_steps=cfg["training"]["grad_accum_steps"],
        early_stop_patience=cfg["training"]["early_stop_patience"],
        device=cfg["training"]["device"],
        checkpoint_dir=Path(cfg["training"]["checkpoint_dir"]),
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write scripts/evaluate.py**

```python
"""Evaluate model on test set.

Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pt [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset.klines_dataset import KlinesDataset, make_split
from models.price_transformer import PriceTransformer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    test_files = make_split(data_dir, *cfg["data"]["test_months"])
    print(f"Test files: {len(test_files)}")

    seq_cfg = cfg["sequence"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]

    test_ds = KlinesDataset(
        test_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )
    test_dl = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model = PriceTransformer(
        delta_vocab_size=model_cfg["delta_vocab_size"],
        bucket_vocab_size=model_cfg["bucket_vocab_size"],
        delta_emb_dim=model_cfg["delta_emb_dim"],
        bucket_emb_dim=model_cfg["bucket_emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=0.0,
        num_classes=model_cfg["num_classes"],
        seq_len=seq_cfg["length"],
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    device = cfg["training"]["device"]
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for delta, vol, vb, labels in test_dl:
            delta = delta.to(device)
            vol = vol.to(device)
            vb = vb.to(device)
            logits = model(delta, vol, vb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["DOWN", "FLAT", "UP"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write scripts/backtest.py**

```python
"""Run backtest on test period.

Usage: python scripts/backtest.py --checkpoint checkpoints/best.pt [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from dataset.klines_dataset import KlinesDataset, make_split
from models.price_transformer import PriceTransformer
from backtest.engine import BacktestEngine


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    test_files = make_split(data_dir, *cfg["data"]["test_months"])
    seq_cfg = cfg["sequence"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]

    test_ds = KlinesDataset(
        test_files,
        seq_len=seq_cfg["length"],
        target_horizon=seq_cfg["target_horizon"],
        target_threshold=seq_cfg["target_threshold"],
        range_pct=tok_cfg["delta"]["range_pct"],
        step_pct=tok_cfg["delta"]["step_pct"],
        n_bins=tok_cfg["bucket"]["n_bins"],
    )

    model = PriceTransformer(
        delta_vocab_size=model_cfg["delta_vocab_size"],
        bucket_vocab_size=model_cfg["bucket_vocab_size"],
        delta_emb_dim=model_cfg["delta_emb_dim"],
        bucket_emb_dim=model_cfg["bucket_emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=0.0,
        num_classes=model_cfg["num_classes"],
        seq_len=seq_cfg["length"],
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    device = cfg["training"]["device"]
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for delta, vol, vb, _ in dl:
            delta = delta.to(device)
            vol = vol.to(device)
            vb = vb.to(device)
            logits = model(delta, vol, vb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
    predictions = np.array(all_preds)

    import pandas as pd
    frames = [pq.read_table(f).to_pandas() for f in test_files]
    closes = pd.concat([f["close"] for f in frames]).values.astype(np.float32)

    bt_cfg = cfg["backtest"]
    engine = BacktestEngine(
        commission=bt_cfg["commission"], stop_loss=bt_cfg["stop_loss"],
        take_profit=bt_cfg["take_profit"], max_hold=bt_cfg["max_hold"],
    )
    result = engine.run(closes, predictions)

    print(f"\nBacktest Results:")
    print(f"  Total PnL:    {result.total_pnl:+.2%}")
    print(f"  Sharpe:        {result.sharpe:.2f}")
    print(f"  Max Drawdown:  {result.max_drawdown:.2%}")
    print(f"  Win Rate:      {result.win_rate:.2%}")
    print(f"  Trade Count:   {result.trade_count}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Duration:  {result.avg_duration:.1f} candles")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify scripts parse without error**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/train.py', 'scripts/evaluate.py', 'scripts/backtest.py']]"
echo "All scripts parse OK"
```

Expected: "All scripts parse OK"

- [ ] **Step 5: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/scripts/train.py token_first_transformer/scripts/evaluate.py token_first_transformer/scripts/backtest.py
git commit -m "feat: add CLI scripts for train, evaluate, backtest"
```

---

### Task 9: Integration Smoke Test

**Files:**
- Create: `token_first_transformer/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
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
```

- [ ] **Step 2: Run integration test**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/test_integration.py -v --tb=short
```

Expected: PASS

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/suenot/projects/w_trading/w_training/token_first_transformer
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add token_first_transformer/tests/test_integration.py
git commit -m "feat: add integration smoke test for full pipeline"
```
