# Indicator Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reusable library that computes technical indicators from OHLCV data and bucketizes them into discrete tokens. Fitted boundaries saved to disk for reuse by downstream projects (4, 5).

**Architecture:** Two-layer design — `IndicatorComputer` computes raw indicator values from numpy arrays, `IndicatorTokenizer` fits quantile boundaries on training data and encodes indicator values to token IDs. Each indicator has its own vocabulary.

**Tech Stack:** Python 3.11+, NumPy, PyArrow

**Indicators implemented:**

| Indicator | Computation | Bins | Rationale |
|-----------|-------------|------|-----------|
| RSI(14) | Wilder's smoothing of gains/losses | 5: `<20, 20-30, 30-70, 70-80, >80` | Standard overbought/oversold zones |
| MACD Histogram | EMA(12)-EMA(26), signal=EMA(9), hist=MACD-signal | 7 quantile bins | Symmetric distribution, quantile bins capture magnitude |
| Bollinger %B(20,2) | (close-lower)/(upper-lower) | 5: `<0, 0-0.25, 0.25-0.75, 0.75-1, >1` | Outside/inside band zones |
| ATR(14) | EMA of True Range | 6 quantile bins | Volatility regime detection |
| Volume Ratio | volume / SMA(volume, 20) | 5 quantile bins | Relative activity level |
| Price vs SMA(20) | (close - sma20) / sma20 | 5 quantile bins | Trend direction proxy |

Total: 6 indicators × 5-7 bins each = 6 vocabularies.

---

## File Structure

```
w_training/indicator_tokenizer/
├── pyproject.toml
├── configs/
│   └── default.yaml
├── indicators/
│   ├── __init__.py
│   ├── computer.py          # compute raw indicator values
│   └── tokenizer.py         # quantile bucketizer per indicator
├── scripts/
│   ├── fit.py               # fit on training data, save boundaries
│   └── inspect.py           # print indicator stats
├── boundaries/              # saved .npy files (gitignored)
└── tests/
    ├── __init__.py
    ├── test_computer.py
    └── test_tokenizer.py
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `indicator_tokenizer/pyproject.toml`
- Create: `indicator_tokenizer/configs/default.yaml`
- Create: `indicator_tokenizer/indicators/__init__.py`
- Create: `indicator_tokenizer/scripts/` (empty)
- Create: `indicator_tokenizer/boundaries/.gitkeep`
- Create: `indicator_tokenizer/tests/__init__.py`
- Create: `indicator_tokenizer/.gitignore`

- [ ] **Step 1: Create directory structure and files**

```bash
cd /Users/suenot/projects/w_trading/w_training
mkdir -p indicator_tokenizer/{configs,indicators,scripts,boundaries,tests}
touch indicator_tokenizer/indicators/__init__.py
touch indicator_tokenizer/tests/__init__.py
touch indicator_tokenizer/boundaries/.gitkeep
```

**pyproject.toml:**
```toml
[project]
name = "indicator-tokenizer"
version = "0.1.0"
description = "Technical indicator computation and tokenization for trading models"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pyarrow>=15.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**configs/default.yaml:**
```yaml
data:
  symbol: BTCUSDT
  data_dir: ../../../w_trender/backtests/data
  train_months: ["2023-02", "2025-06"]

indicators:
  rsi:
    period: 14
    bins: [20, 30, 70, 80]       # 5 bins: <20, 20-30, 30-70, 70-80, >80
  macd_hist:
    fast: 12
    slow: 26
    signal: 9
    n_quantile_bins: 7
  bollinger_pctb:
    period: 20
    num_std: 2
    bins: [0.0, 0.25, 0.75, 1.0] # 5 bins: <0, 0-0.25, 0.25-0.75, 0.75-1, >1
  atr:
    period: 14
    n_quantile_bins: 6
  volume_ratio:
    period: 20
    n_quantile_bins: 5
  price_vs_sma:
    period: 20
    n_quantile_bins: 5

boundaries_dir: boundaries
```

**.gitignore:**
```
boundaries/*.npy
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 2: Commit**

```bash
cd /Users/suenot/projects/w_trading/w_training
git add indicator_tokenizer/
git commit -m "feat: scaffold indicator_tokenizer project"
```

---

### Task 2: IndicatorComputer

**Files:**
- Create: `indicator_tokenizer/indicators/computer.py`
- Create: `indicator_tokenizer/tests/test_computer.py`

- [ ] **Step 1: Write tests/test_computer.py**

```python
import numpy as np
import pytest
from indicators.computer import IndicatorComputer


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(42)
    n = 200
    close = 30000.0 + np.cumsum(rng.standard_normal(n) * 5).astype(np.float32)
    return {
        "open": close - rng.standard_normal(n).astype(np.float32),
        "high": (close + np.abs(rng.standard_normal(n)) * 3).astype(np.float32),
        "low": (close - np.abs(rng.standard_normal(n)) * 3).astype(np.float32),
        "close": close.astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }


def test_rsi_shape_and_range(ohlcv):
    comp = IndicatorComputer()
    rsi = comp.rsi(ohlcv["close"], period=14)
    assert len(rsi) == len(ohlcv["close"])
    # After warmup, RSI should be in [0, 100]
    valid = rsi[14:]
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_rsi_known_values():
    # Monotonically increasing closes -> RSI should approach 100
    closes = np.arange(100, 130, dtype=np.float32)
    comp = IndicatorComputer()
    rsi = comp.rsi(closes, period=14)
    assert rsi[-1] > 90


def test_macd_hist_shape(ohlcv):
    comp = IndicatorComputer()
    hist = comp.macd_hist(ohlcv["close"], fast=12, slow=26, signal=9)
    assert len(hist) == len(ohlcv["close"])


def test_bollinger_pctb_shape_and_range(ohlcv):
    comp = IndicatorComputer()
    pctb = comp.bollinger_pctb(ohlcv["close"], period=20, num_std=2)
    assert len(pctb) == len(ohlcv["close"])
    # %B can go outside [0,1] but mostly stays near it
    assert pctb[20:].mean() > -1
    assert pctb[20:].mean() < 2


def test_atr_shape(ohlcv):
    comp = IndicatorComputer()
    atr = comp.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
    assert len(atr) == len(ohlcv["close"])
    # ATR should be positive after warmup
    assert atr[14:].min() >= 0


def test_volume_ratio_shape(ohlcv):
    comp = IndicatorComputer()
    vr = comp.volume_ratio(ohlcv["close"], ohlcv["volume"], period=20)
    assert len(vr) == len(ohlcv["close"])


def test_price_vs_sma_shape(ohlcv):
    comp = IndicatorComputer()
    pvs = comp.price_vs_sma(ohlcv["close"], period=20)
    assert len(pvs) == len(ohlcv["close"])


def test_compute_all(ohlcv):
    comp = IndicatorComputer()
    result = comp.compute_all(ohlcv)
    assert "rsi" in result
    assert "macd_hist" in result
    assert "bollinger_pctb" in result
    assert "atr" in result
    assert "volume_ratio" in result
    assert "price_vs_sma" in result
    for key, arr in result.items():
        assert len(arr) == len(ohlcv["close"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/indicator_tokenizer
python -m pytest tests/test_computer.py -v
```

- [ ] **Step 3: Write indicators/computer.py**

```python
from __future__ import annotations

import numpy as np


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.zeros(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


class IndicatorComputer:
    """Computes technical indicators from OHLCV numpy arrays."""

    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close.astype(np.float64), prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = _ema(gain, period)
        avg_loss = _ema(loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        return (100 - 100 / (1 + rs)).astype(np.float32)

    def macd_hist(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9,
    ) -> np.ndarray:
        c = close.astype(np.float64)
        macd_line = _ema(c, fast) - _ema(c, slow)
        signal_line = _ema(macd_line, signal)
        return (macd_line - signal_line).astype(np.float32)

    def bollinger_pctb(
        self, close: np.ndarray, period: int = 20, num_std: float = 2.0,
    ) -> np.ndarray:
        c = close.astype(np.float64)
        out = np.zeros(len(c), dtype=np.float32)
        for i in range(period - 1, len(c)):
            window = c[i - period + 1 : i + 1]
            sma = window.mean()
            std = window.std()
            upper = sma + num_std * std
            lower = sma - num_std * std
            out[i] = (c[i] - lower) / (upper - lower + 1e-10)
        return out

    def atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14,
    ) -> np.ndarray:
        tr = np.zeros(len(close), dtype=np.float64)
        tr[0] = high[0] - low[0]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:].astype(np.float64) - close[:-1].astype(np.float64)),
                np.abs(low[1:].astype(np.float64) - close[:-1].astype(np.float64)),
            ),
        )
        return _ema(tr, period).astype(np.float32)

    def volume_ratio(
        self, close: np.ndarray, volume: np.ndarray, period: int = 20,
    ) -> np.ndarray:
        v = volume.astype(np.float64)
        sma = np.zeros(len(v), dtype=np.float64)
        out = np.zeros(len(v), dtype=np.float32)
        for i in range(period - 1, len(v)):
            sma[i] = v[i - period + 1 : i + 1].mean()
            out[i] = v[i] / (sma[i] + 1e-10)
        return out

    def price_vs_sma(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        c = close.astype(np.float64)
        out = np.zeros(len(c), dtype=np.float32)
        for i in range(period - 1, len(c)):
            sma = c[i - period + 1 : i + 1].mean()
            out[i] = (c[i] - sma) / (sma + 1e-10)
        return out

    def compute_all(self, ohlcv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "rsi": self.rsi(ohlcv["close"]),
            "macd_hist": self.macd_hist(ohlcv["close"]),
            "bollinger_pctb": self.bollinger_pctb(ohlcv["close"]),
            "atr": self.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]),
            "volume_ratio": self.volume_ratio(ohlcv["close"], ohlcv["volume"]),
            "price_vs_sma": self.price_vs_sma(ohlcv["close"]),
        }
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/indicator_tokenizer
python -m pytest tests/test_computer.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add indicator_tokenizer/indicators/computer.py indicator_tokenizer/tests/test_computer.py
git commit -m "feat: add IndicatorComputer with RSI, MACD, Bollinger, ATR, volume ratio, price vs SMA"
```

---

### Task 3: IndicatorTokenizer

**Files:**
- Create: `indicator_tokenizer/indicators/tokenizer.py`
- Create: `indicator_tokenizer/tests/test_tokenizer.py`

- [ ] **Step 1: Write tests/test_tokenizer.py**

```python
import numpy as np
import pytest
from pathlib import Path
from indicators.tokenizer import IndicatorTokenizer, FixedBoundaries, QuantileBoundaries


def test_fixed_boundaries_encode():
    fb = FixedBoundaries(bins=[20, 30, 70, 80], offset=2)
    assert fb.vocab_size == 7  # 5 bins + PAD(0) + special(1)
    assert fb.encode(0.0) == 2      # <20 -> bin 0
    assert fb.encode(25.0) == 3     # 20-30 -> bin 1
    assert fb.encode(50.0) == 4     # 30-70 -> bin 2
    assert fb.encode(75.0) == 5     # 70-80 -> bin 3
    assert fb.encode(90.0) == 6     # >80 -> bin 4


def test_fixed_boundaries_batch():
    fb = FixedBoundaries(bins=[20, 30, 70, 80], offset=2)
    vals = np.array([10.0, 25.0, 50.0, 75.0, 90.0], dtype=np.float32)
    ids = fb.encode_batch(vals)
    assert list(ids) == [2, 3, 4, 5, 6]


def test_quantile_boundaries_fit():
    qb = QuantileBoundaries(n_bins=5, offset=2)
    data = np.arange(100, dtype=np.float32)
    qb.fit(data)
    assert qb.vocab_size == 7  # 5 bins + PAD + special
    assert qb.boundaries is not None
    assert len(qb.boundaries) == 4  # n_bins - 1


def test_quantile_boundaries_encode():
    qb = QuantileBoundaries(n_bins=4, offset=2)
    data = np.arange(100, dtype=np.float32)
    qb.fit(data)
    ids = qb.encode_batch(data)
    assert ids.min() >= 2
    assert ids.max() <= 5


def test_indicator_tokenizer_initial_state():
    tok = IndicatorTokenizer()
    assert tok.rsi is not None
    assert tok.macd_hist is not None
    assert tok.bollinger_pctb is not None
    assert tok.atr is not None
    assert tok.volume_ratio is not None
    assert tok.price_vs_sma is not None


def test_indicator_tokenizer_fit_and_encode():
    rng = np.random.default_rng(42)
    n = 500
    ohlcv = {
        "close": (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32),
        "high": (30000 + np.cumsum(rng.standard_normal(n) * 5) + 5).astype(np.float32),
        "low": (30000 + np.cumsum(rng.standard_normal(n) * 5) - 5).astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }
    from indicators.computer import IndicatorComputer
    comp = IndicatorComputer()
    indicators = comp.compute_all(ohlcv)

    tok = IndicatorTokenizer()
    tok.fit(indicators)
    encoded = tok.encode(indicators)
    assert "rsi" in encoded
    assert "macd_hist" in encoded
    for key, ids in encoded.items():
        assert len(ids) == n
        assert ids.dtype == np.int32


def test_indicator_tokenizer_save_load(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    ohlcv = {
        "close": (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32),
        "high": (30000 + np.cumsum(rng.standard_normal(n) * 5) + 5).astype(np.float32),
        "low": (30000 + np.cumsum(rng.standard_normal(n) * 5) - 5).astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }
    from indicators.computer import IndicatorComputer
    comp = IndicatorComputer()
    indicators = comp.compute_all(ohlcv)

    tok = IndicatorTokenizer()
    tok.fit(indicators)
    tok.save(tmp_path)

    tok2 = IndicatorTokenizer()
    tok2.load(tmp_path)
    encoded1 = tok.encode(indicators)
    encoded2 = tok2.encode(indicators)
    for key in encoded1:
        np.testing.assert_array_equal(encoded1[key], encoded2[key])


def test_indicator_tokenizer_vocab_sizes():
    tok = IndicatorTokenizer()
    rng = np.random.default_rng(42)
    n = 200
    ohlcv = {
        "close": (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32),
        "high": (30000 + np.cumsum(rng.standard_normal(n) * 5) + 5).astype(np.float32),
        "low": (30000 + np.cumsum(rng.standard_normal(n) * 5) - 5).astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }
    from indicators.computer import IndicatorComputer
    comp = IndicatorComputer()
    indicators = comp.compute_all(ohlcv)
    tok.fit(indicators)
    vs = tok.vocab_sizes()
    assert vs["rsi"] == 7     # 5 bins + PAD + special
    assert vs["macd_hist"] == 9  # 7 bins + PAD + special
    assert vs["atr"] == 8     # 6 bins + PAD + special
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/suenot/projects/w_trading/w_training/indicator_tokenizer
python -m pytest tests/test_tokenizer.py -v
```

- [ ] **Step 3: Write indicators/tokenizer.py**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np


class FixedBoundaries:
    """Encodes values into bins defined by fixed threshold array."""

    def __init__(self, bins: list[float], offset: int = 2) -> None:
        self.bins = np.array(bins, dtype=np.float32)
        self.offset = offset
        self.vocab_size = len(bins) + 1 + offset  # n_bins+1 regions + offset

    def encode(self, value: float) -> int:
        return int(np.searchsorted(self.bins, value, side="right")) + self.offset

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        return (np.searchsorted(self.bins, values, side="right") + self.offset).astype(np.int32)

    def save(self, path: Path) -> None:
        np.save(path, self.bins)

    def load(self, path: Path) -> None:
        self.bins = np.load(path)


class QuantileBoundaries:
    """Encodes values into quantile-based bins (fitted on data)."""

    def __init__(self, n_bins: int, offset: int = 2) -> None:
        self.n_bins = n_bins
        self.offset = offset
        self.vocab_size = n_bins + offset
        self.boundaries: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.boundaries = np.percentile(values, quantiles).astype(np.float32)

    def encode(self, value: float) -> int:
        assert self.boundaries is not None
        return int(np.searchsorted(self.boundaries, value, side="right")) + self.offset

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None
        return (np.searchsorted(self.boundaries, values, side="right") + self.offset).astype(np.int32)

    def save(self, path: Path) -> None:
        assert self.boundaries is not None
        np.save(path, self.boundaries)

    def load(self, path: Path) -> None:
        self.boundaries = np.load(path)


class IndicatorTokenizer:
    """Tokenizes all indicators using per-indicator boundary definitions."""

    PAD_ID = 0
    SPECIAL_ID = 1  # reserved

    def __init__(self) -> None:
        # RSI: fixed bins [20, 30, 70, 80] -> 5 regions
        self.rsi = FixedBoundaries(bins=[20, 30, 70, 80])
        # MACD hist: 7 quantile bins
        self.macd_hist = QuantileBoundaries(n_bins=7)
        # Bollinger %B: fixed bins [0, 0.25, 0.75, 1.0] -> 5 regions
        self.bollinger_pctb = FixedBoundaries(bins=[0.0, 0.25, 0.75, 1.0])
        # ATR: 6 quantile bins
        self.atr = QuantileBoundaries(n_bins=6)
        # Volume ratio: 5 quantile bins
        self.volume_ratio = QuantileBoundaries(n_bins=5)
        # Price vs SMA: 5 quantile bins
        self.price_vs_sma = QuantileBoundaries(n_bins=5)

        self._quantile_fields = ["macd_hist", "atr", "volume_ratio", "price_vs_sma"]
        self._all_fields = ["rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"]

    def fit(self, indicators: dict[str, np.ndarray]) -> None:
        for field in self._quantile_fields:
            getattr(self, field).fit(indicators[field])

    def encode(self, indicators: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}
        for field in self._all_fields:
            result[field] = getattr(self, field).encode_batch(indicators[field])
        return result

    def vocab_sizes(self) -> dict[str, int]:
        return {field: getattr(self, field).vocab_size for field in self._all_fields}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for field in self._all_fields:
            getattr(self, field).save(directory / f"{field}.npy")

    def load(self, directory: Path) -> None:
        for field in self._all_fields:
            getattr(self, field).load(directory / f"{field}.npy")
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/suenot/projects/w_trading/w_training/indicator_tokenizer
python -m pytest tests/test_tokenizer.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add indicator_tokenizer/indicators/tokenizer.py indicator_tokenizer/tests/test_tokenizer.py
git commit -m "feat: add IndicatorTokenizer with fixed and quantile boundaries"
```

---

### Task 4: Fit and Inspect Scripts

**Files:**
- Create: `indicator_tokenizer/scripts/fit.py`
- Create: `indicator_tokenizer/scripts/inspect.py`

- [ ] **Step 1: Write scripts/fit.py**

```python
"""Fit indicator tokenizers on training data and save boundaries.

Usage: python scripts/fit.py [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import pyarrow.parquet as pq

from indicators.computer import IndicatorComputer
from indicators.tokenizer import IndicatorTokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    klines_dir = data_dir / cfg["data"]["symbol"] / "klines_1m"
    start, end = cfg["data"]["train_months"]

    files = sorted(klines_dir.glob("*.parquet"))
    files = [f for f in files if start <= f.stem <= end]
    print(f"Loading {len(files)} months of training data...")

    all_indicators = {k: [] for k in [
        "rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"
    ]}
    comp = IndicatorComputer()
    for f in files:
        table = pq.read_table(f)
        ohlcv = {col: np.array([v.as_py() for v in table.column(col)], dtype=np.float32)
                 for col in ["open", "high", "low", "close", "volume"]}
        indicators = comp.compute_all(ohlcv)
        for k in all_indicators:
            all_indicators[k].append(indicators[k])

    combined = {k: np.concatenate(v) for k, v in all_indicators.items()}
    for k, v in combined.items():
        print(f"  {k}: {len(v):,} values")

    tok = IndicatorTokenizer()
    tok.fit(combined)

    boundaries_dir = Path(cfg["boundaries_dir"])
    tok.save(boundaries_dir)
    print(f"\nBoundaries saved to {boundaries_dir}/")
    vs = tok.vocab_sizes()
    for k, v in vs.items():
        print(f"  {k}: vocab_size={v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write scripts/inspect.py**

```python
"""Print indicator statistics from data.

Usage: python scripts/inspect.py [--config configs/default.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import pyarrow.parquet as pq

from indicators.computer import IndicatorComputer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--months", default=1, type=int, help="Number of months to inspect")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    klines_dir = data_dir / cfg["data"]["symbol"] / "klines_1m"
    files = sorted(klines_dir.glob("*.parquet"))
    files = files[:args.months]
    print(f"Inspecting {len(files)} file(s)")

    comp = IndicatorComputer()
    for f in files:
        print(f"\n=== {f.stem} ===")
        table = pq.read_table(f)
        ohlcv = {col: np.array([v.as_py() for v in table.column(col)], dtype=np.float32)
                 for col in ["open", "high", "low", "close", "volume"]}
        indicators = comp.compute_all(ohlcv)

        for name, values in indicators.items():
            v = values[len(values) // 5:]  # skip warmup
            print(f"  {name:20s}  min={v.min():10.4f}  max={v.max():10.4f}  "
                  f"mean={v.mean():10.4f}  std={v.std():10.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify scripts parse**

```bash
cd /Users/suenot/projects/w_trading/w_training/indicator_tokenizer
python -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/fit.py', 'scripts/inspect.py']]; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add indicator_tokenizer/scripts/fit.py indicator_tokenizer/scripts/inspect.py
git commit -m "feat: add fit and inspect scripts for indicator boundaries"
```

---

### Task 5: Run fit.py on real data

This is a verification step, not code. Run the fit script to generate boundary files and verify everything works end-to-end.

- [ ] **Step 1: Run fit.py**

```bash
cd /Users/suenot/projects/w_trading/w_training/indicator_tokenizer
pip install pyyaml pyarrow  # if needed
python scripts/fit.py
```

Expected: loads 28 months of BTCUSDT data, computes indicators, fits quantile boundaries, saves to `boundaries/`.

- [ ] **Step 2: Verify boundary files exist**

```bash
ls -la boundaries/
```

Expected: 6 `.npy` files (rsi, macd_hist, bollinger_pctb, atr, volume_ratio, price_vs_sma).
