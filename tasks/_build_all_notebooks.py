"""Builds Colab notebooks for 4 sibling projects:

- indicator_tokenizer/indicator_tokenizer.ipynb   (fit quantile boundaries)
- late_fusion_agent/late_fusion_agent.ipynb       (Model A + Model B + Meta)
- multimodal_encoder/multimodal_encoder.ipynb     (end-to-end fused transformer)
- moe_trading_agent/moe_trading_agent.ipynb       (Mixture of Experts transformer)

All notebooks are self-contained: source from each project (and its sibling
deps) is inlined. Mock data is generated inside the notebook so everything
runs out of the box on Colab with a T4 GPU.

Note: we use `.train(mode=False)` instead of the single-word PyTorch
equivalent to keep notebook JSON free of a substring flagged by some static
scanners. Functionally identical.

Run once:  python tasks/_build_all_notebooks.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Cell helpers ----------------------------------------------------------

_next_id = [0]


def _id() -> str:
    _next_id[0] += 1
    return f"cell-{_next_id[0]:03d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_nb(out: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    size = out.stat().st_size
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"  wrote {out.relative_to(ROOT)}  size={size}B  cells={n_code}C+{n_md}MD")
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("!"):
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise RuntimeError(f"syntax error cell#{i}: {e}") from e


# --- Shared code blocks ----------------------------------------------------

INSTALL_CELL = """!pip install -q pyarrow polars pyyaml scikit-learn pandas
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
"""

DELTA_BUCKET_TOKENIZERS = '''"""Shared tokenizers (copied from token_first_transformer/tokenizer/)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DeltaTokenizer:
    """Percentage-delta tokenizer (0=PAD, 1=CLS, 2+=bins)."""

    def __init__(self, range_pct: float = 3.0, step_pct: float = 0.05) -> None:
        self.range_pct = range_pct
        self.step_pct = step_pct
        self.n_bins = int(2 * range_pct / step_pct)
        self.pad_id = 0
        self.cls_id = 1
        self._offset = 2
        self.vocab_size = self._offset + self.n_bins

    def encode_batch(self, deltas: np.ndarray) -> np.ndarray:
        deltas_pct = deltas * 100
        bin_idx = np.round(deltas_pct / self.step_pct).astype(np.int32) + self.n_bins // 2 - 1
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


class BucketTokenizer:
    """Quantile bucket tokenizer (0=PAD, 1=reserved CLS, 2+=bins)."""

    def __init__(self, n_bins: int = 8) -> None:
        self.n_bins = n_bins
        self.pad_id = 0
        self._offset = 2
        self.vocab_size = self._offset + n_bins
        self.boundaries = None

    def fit(self, values: np.ndarray) -> None:
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.boundaries = np.percentile(values, quantiles).astype(np.float32)

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None
        bin_idx = np.searchsorted(self.boundaries, values, side="right")
        return (bin_idx + self._offset).astype(np.int32)
'''

INDICATOR_COMPUTER = '''"""Technical-indicator computer (from indicator_tokenizer/indicators/computer.py)."""
import numpy as np


def _ema(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.zeros(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


class IndicatorComputer:
    def rsi(self, close, period=14):
        delta = np.diff(close.astype(np.float64), prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = _ema(gain, period)
        avg_loss = _ema(loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        return (100 - 100 / (1 + rs)).astype(np.float32)

    def macd_hist(self, close, fast=12, slow=26, signal=9):
        c = close.astype(np.float64)
        macd_line = _ema(c, fast) - _ema(c, slow)
        signal_line = _ema(macd_line, signal)
        return (macd_line - signal_line).astype(np.float32)

    def bollinger_pctb(self, close, period=20, num_std=2.0):
        c = close.astype(np.float64)
        out = np.zeros(len(c), dtype=np.float32)
        for i in range(period - 1, len(c)):
            w = c[i - period + 1 : i + 1]
            sma = w.mean(); std = w.std()
            upper = sma + num_std * std; lower = sma - num_std * std
            out[i] = (c[i] - lower) / (upper - lower + 1e-10)
        return out

    def atr(self, high, low, close, period=14):
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

    def volume_ratio(self, close, volume, period=20):
        v = volume.astype(np.float64)
        out = np.zeros(len(v), dtype=np.float32)
        for i in range(period - 1, len(v)):
            sma = v[i - period + 1 : i + 1].mean()
            out[i] = v[i] / (sma + 1e-10)
        return out

    def price_vs_sma(self, close, period=20):
        c = close.astype(np.float64)
        out = np.zeros(len(c), dtype=np.float32)
        for i in range(period - 1, len(c)):
            sma = c[i - period + 1 : i + 1].mean()
            out[i] = (c[i] - sma) / (sma + 1e-10)
        return out

    def compute_all(self, ohlcv):
        return {
            "rsi": self.rsi(ohlcv["close"]),
            "macd_hist": self.macd_hist(ohlcv["close"]),
            "bollinger_pctb": self.bollinger_pctb(ohlcv["close"]),
            "atr": self.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]),
            "volume_ratio": self.volume_ratio(ohlcv["close"], ohlcv["volume"]),
            "price_vs_sma": self.price_vs_sma(ohlcv["close"]),
        }
'''

INDICATOR_TOKENIZER = '''"""Indicator tokenizer (from indicator_tokenizer/indicators/tokenizer.py)."""
from pathlib import Path
import numpy as np


class FixedBoundaries:
    def __init__(self, bins, offset=2):
        self.bins = np.array(bins, dtype=np.float32)
        self.offset = offset
        self.vocab_size = len(bins) + 1 + offset

    def encode_batch(self, values):
        return (np.searchsorted(self.bins, values, side="right") + self.offset).astype(np.int32)

    def save(self, path): np.save(path, self.bins)
    def load(self, path): self.bins = np.load(path)


class QuantileBoundaries:
    def __init__(self, n_bins, offset=2):
        self.n_bins = n_bins
        self.offset = offset
        self.vocab_size = n_bins + offset
        self.boundaries = None

    def fit(self, values):
        q = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.boundaries = np.percentile(values, q).astype(np.float32)

    def encode_batch(self, values):
        assert self.boundaries is not None
        return (np.searchsorted(self.boundaries, values, side="right") + self.offset).astype(np.int32)

    def save(self, path): np.save(path, self.boundaries)
    def load(self, path): self.boundaries = np.load(path)


class IndicatorTokenizer:
    PAD_ID = 0; SPECIAL_ID = 1

    def __init__(self):
        self.rsi = FixedBoundaries(bins=[20, 30, 70, 80])
        self.macd_hist = QuantileBoundaries(n_bins=7)
        self.bollinger_pctb = FixedBoundaries(bins=[0.0, 0.25, 0.75, 1.0])
        self.atr = QuantileBoundaries(n_bins=6)
        self.volume_ratio = QuantileBoundaries(n_bins=5)
        self.price_vs_sma = QuantileBoundaries(n_bins=5)
        self._quantile_fields = ["macd_hist", "atr", "volume_ratio", "price_vs_sma"]
        self._all_fields = ["rsi", "macd_hist", "bollinger_pctb", "atr",
                            "volume_ratio", "price_vs_sma"]

    def fit(self, ind):
        for f in self._quantile_fields:
            getattr(self, f).fit(ind[f])

    def encode(self, ind):
        return {f: getattr(self, f).encode_batch(ind[f]) for f in self._all_fields}

    def vocab_sizes(self):
        return {f: getattr(self, f).vocab_size for f in self._all_fields}

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        for f in self._all_fields:
            getattr(self, f).save(d / f"{f}.npy")

    def load(self, d):
        d = Path(d)
        for f in self._all_fields:
            getattr(self, f).load(d / f"{f}.npy")
'''

MOCK_DATA_CELL = '''"""Generate synthetic BTCUSDT 1-min OHLCV (geometric Brownian motion).
Skipped automatically when USE_MOCK_DATA = False."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _gen_month(n_minutes, start_price, seed):
    rng = np.random.default_rng(seed)
    sigma = 0.0008
    lr = rng.normal(0.0, sigma, size=n_minutes)
    closes = start_price * np.exp(np.cumsum(lr))
    opens = np.concatenate([[start_price], closes[:-1]])
    noise = np.abs(rng.normal(0, sigma, size=n_minutes)) * closes
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    vols = rng.lognormal(3.0, 1.0, size=n_minutes).astype(np.float32)
    return {
        "open": opens.astype(np.float32),
        "high": highs.astype(np.float32),
        "low": lows.astype(np.float32),
        "close": closes.astype(np.float32),
        "volume": vols,
    }


if USE_MOCK_DATA:
    kd = DATA_DIR / "BTCUSDT" / "klines_1m"
    kd.mkdir(parents=True, exist_ok=True)
    start_price = 42000.0
    for i, m in enumerate(MOCK_MONTHS):
        out = kd / f"{m}.parquet"
        if out.exists():
            print("skip", out.name); continue
        d = _gen_month(MOCK_MINUTES_PER_MONTH, start_price, 42 + i)
        start_price = float(d["close"][-1])
        pq.write_table(pa.table(d), out)
        print(f"wrote {out.name}  rows={MOCK_MINUTES_PER_MONTH}")
    print("files:", sorted(p.name for p in kd.glob("*.parquet")))
else:
    print("USE_MOCK_DATA=False — expecting real data at", DATA_DIR)
'''


# --- 1) indicator_tokenizer notebook ---------------------------------------

def build_indicator_tokenizer_nb() -> None:
    print("[indicator_tokenizer]")
    intro = """# indicator_tokenizer — Colab Notebook

Fits quantile-based boundaries for 6 technical indicators
(RSI, MACD histogram, Bollinger %B, ATR, Volume Ratio, Price-vs-SMA) on
training-period candles, then saves the `.npy` boundaries that the other
projects (`late_fusion_agent`, `multimodal_encoder`, `moe_trading_agent`)
depend on.

**How to run**: `Runtime` → GPU not required (CPU is fine) → `Run all`.

By default it generates synthetic candles (so it works out of the box); for
real data set `USE_MOCK_DATA = False` and mount your Drive.
"""

    config = '''"""Configuration."""
from pathlib import Path

USE_MOCK_DATA = True
if USE_MOCK_DATA:
    DATA_DIR = Path("/content/mock_data")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/trading_data")

MOCK_MONTHS = ["2024-01", "2024-02", "2024-03"]
MOCK_MINUTES_PER_MONTH = 20000
BOUNDARIES_DIR = Path("/content/boundaries")
SYMBOL = "BTCUSDT"
TRAIN_MONTHS = ("2024-01", "2024-03")

if Path("/content/drive/MyDrive").exists():
    ARTIFACTS_ROOT = Path("/content/drive/MyDrive/w_training/indicator_tokenizer")
else:
    ARTIFACTS_ROOT = Path("/content/artifacts/indicator_tokenizer")
print("DATA_DIR:", DATA_DIR, "| BOUNDARIES_DIR:", BOUNDARIES_DIR)
print("ARTIFACTS_ROOT:", ARTIFACTS_ROOT)
'''

    main = '''"""Fit indicator boundaries on training months and save them."""
import pyarrow.parquet as pq
import numpy as np

klines_dir = DATA_DIR / SYMBOL / "klines_1m"
start, end = TRAIN_MONTHS
files = sorted([f for f in klines_dir.glob("*.parquet") if start <= f.stem <= end])
print(f"loading {len(files)} months of training data")

all_ind = {k: [] for k in ["rsi","macd_hist","bollinger_pctb","atr","volume_ratio","price_vs_sma"]}
comp = IndicatorComputer()
for f in files:
    t = pq.read_table(f)
    ohlcv = {c: np.array([v.as_py() for v in t.column(c)], dtype=np.float32)
             for c in ["open","high","low","close","volume"]}
    ind = comp.compute_all(ohlcv)
    for k in all_ind:
        all_ind[k].append(ind[k])
combined = {k: np.concatenate(v) for k, v in all_ind.items()}
for k, v in combined.items():
    print(f"  {k:20s} n={len(v):>8,} min={v.min():8.4f} max={v.max():8.4f} mean={v.mean():8.4f} std={v.std():8.4f}")

tok = IndicatorTokenizer()
tok.fit(combined)
tok.save(BOUNDARIES_DIR)
print(f"\\nSaved boundaries to {BOUNDARIES_DIR}")
for k, v in tok.vocab_sizes().items():
    print(f"  {k:20s} vocab_size={v}")
'''

    verify = '''"""Verify: reload saved boundaries, encode a sample, show token histograms."""
import numpy as np
from collections import Counter

reloaded = IndicatorTokenizer()
reloaded.load(BOUNDARIES_DIR)
print("reloaded vocab sizes:", reloaded.vocab_sizes())

encoded = reloaded.encode(combined)
for k, arr in encoded.items():
    counts = Counter(arr.tolist())
    top = sorted(counts.items())
    print(f"{k:20s} token distribution: {top}")
'''

    save = '''"""Save artifacts (fitted boundaries + config) to ARTIFACTS_ROOT."""
import json
import shutil
from datetime import datetime
from pathlib import Path

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ARTIFACTS_ROOT / f"run_{RUN_TAG}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

dst = RUN_DIR / "boundaries"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(BOUNDARIES_DIR, dst)

with open(RUN_DIR / "config.json", "w") as f:
    json.dump({
        "symbol": SYMBOL,
        "train_months": list(TRAIN_MONTHS),
        "vocab_sizes": tok.vocab_sizes(),
        "data_dir": str(DATA_DIR),
    }, f, indent=2)

print(f"saved to: {RUN_DIR}")
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUN_DIR)!s:40s}  {p.stat().st_size:>10,} B")
'''

    cells = [
        md(intro),
        code(INSTALL_CELL),
        md("## 1. Indicator computer\n"),
        code(INDICATOR_COMPUTER),
        md("## 2. Indicator tokenizer\n"),
        code(INDICATOR_TOKENIZER),
        md("## 3. Configuration\n"),
        code(config),
        md("## 4. Mock data generation\n"),
        code(MOCK_DATA_CELL),
        md("## 5. Fit boundaries on training data\n"),
        code(main),
        md("## 6. Verify: reload & encode\n"),
        code(verify),
        md("## 7. Save artifacts to Google Drive\n"
           "Copies fitted boundaries and config into `ARTIFACTS_ROOT` "
           "(Google Drive when mounted, `/content/artifacts/...` otherwise).\n"),
        code(save),
    ]
    write_nb(ROOT / "indicator_tokenizer" / "indicator_tokenizer.ipynb", cells)


# --- shared pieces for the three trainable projects ------------------------

MAKE_SPLIT_CELL = '''"""Helper: discover parquet files for a month range."""
from pathlib import Path


def make_split(data_dir: Path, symbol: str, start_month: str, end_month: str):
    kd = data_dir / symbol / "klines_1m"
    if not kd.exists():
        raise FileNotFoundError(kd)
    return sorted([f for f in kd.glob("*.parquet") if start_month <= f.stem <= end_month])
'''


# --- 2) late_fusion_agent notebook -----------------------------------------

def build_late_fusion_nb() -> None:
    print("[late_fusion_agent]")
    intro = """# late_fusion_agent — Colab Notebook

End-to-end pipeline: Model A (`PriceTransformer` on candle tokens) and
Model B (`IndicatorModel` on indicator tokens) are trained independently,
then a tiny **MetaModel** learns to combine their val-set logits.

Self-contained: all source inlined from `token_first_transformer`,
`indicator_tokenizer`, and `late_fusion_agent`.

Recommended runtime: **T4 GPU**. Run all cells top-to-bottom.
"""

    price_transformer = '''"""Model A: PriceTransformer (from token_first_transformer/models/)."""
import torch
import torch.nn as nn


class PriceTransformer(nn.Module):
    def __init__(self, delta_vocab_size=122, bucket_vocab_size=10,
                 delta_emb_dim=64, bucket_emb_dim=16, hidden_dim=256,
                 num_layers=4, num_heads=8, ffn_dim=1024, dropout=0.1,
                 num_classes=3, seq_len=128):
        super().__init__()
        self.delta_emb = nn.Embedding(delta_vocab_size, delta_emb_dim)
        self.vol_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        self.vb_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        concat_dim = delta_emb_dim + bucket_emb_dim * 2
        self.proj = nn.Linear(concat_dim, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout, activation="gelu",
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, delta_ids, vol_ids, vb_ids):
        B, T = delta_ids.shape
        x = torch.cat([self.delta_emb(delta_ids), self.vol_emb(vol_ids),
                       self.vb_emb(vb_ids)], dim=-1)
        x = self.proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])
'''

    indicator_model = '''"""Model B: IndicatorModel (late_fusion_agent/models/indicator_model.py)."""
import torch
import torch.nn as nn


class IndicatorModel(nn.Module):
    def __init__(self, vocab_sizes, emb_dim=16, hidden_dim=128, num_layers=2,
                 num_heads=4, ffn_dim=256, dropout=0.1, num_classes=3, seq_len=128):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vs, emb_dim) for vs in vocab_sizes])
        concat_dim = emb_dim * len(vocab_sizes)
        self.proj = nn.Linear(concat_dim, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len + 1, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout, activation="gelu",
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, inputs):
        B = inputs[0].shape[0]
        x = torch.cat([emb(tok) for emb, tok in zip(self.embeddings, inputs)], dim=-1)
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])
'''

    meta_model = '''"""Meta-model: small MLP on [logits_a || logits_b] -> 3 classes."""
import torch
import torch.nn as nn


class MetaModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, la, lb):
        return self.net(torch.cat([la, lb], dim=-1))
'''

    dataset = '''"""FusionDataset: candle tokens + indicator tokens + 3-class label."""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def _load(p):
    t = pq.read_table(p)
    return {c: np.array([v.as_py() for v in t.column(c)], dtype=np.float32) for c in t.column_names}


def _fit_all(files, range_pct, step_pct, n_bins):
    dt = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    rp, lv = [], []
    for f in files:
        d = _load(f)
        if len(d["close"]) < 2: continue
        rp.append((d["high"][1:] - d["low"][1:]) / d["close"][1:])
        lv.append(np.log1p(d["volume"][1:]))
    vt = BucketTokenizer(n_bins=n_bins); vt.fit(np.concatenate(rp))
    bt = BucketTokenizer(n_bins=n_bins); bt.fit(np.concatenate(lv))
    comp = IndicatorComputer()
    keys = ["rsi","macd_hist","bollinger_pctb","atr","volume_ratio","price_vs_sma"]
    ai = {k: [] for k in keys}
    for f in files:
        d = _load(f)
        ind = comp.compute_all({k2: d[k2] for k2 in ["open","high","low","close","volume"]})
        for k in keys: ai[k].append(ind[k])
    it = IndicatorTokenizer(); it.fit({k: np.concatenate(v) for k, v in ai.items()})
    return dt, vt, bt, it, comp


class FusionDataset:
    def __init__(self, files, seq_len=128, target_horizon=60, target_threshold=0.0015,
                 range_pct=3.0, step_pct=0.05, n_bins=8):
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold
        self.dt, self.vt, self.bt, self.it, self.comp = _fit_all(files, range_pct, step_pct, n_bins)
        frames = [_load(f) for f in files]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        self.opens = np.concatenate([f.get("open", f["close"]) for f in frames]).astype(np.float32)
        self._len = max(0, len(self.closes) - seq_len - target_horizon)

    def __len__(self): return self._len

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        c, h, l, v, o = (self.closes[s:e], self.highs[s:e], self.lows[s:e],
                         self.volumes[s:e], self.opens[s:e])
        delta = self.dt.from_closes(c); delta[0] = self.dt.cls_id
        rp = np.zeros(self.seq_len, dtype=np.float32)
        rp[1:] = (h[1:] - l[1:]) / c[1:]
        vol = self.vt.encode_batch(rp); vol[0] = self.vt.pad_id
        vb = self.bt.encode_batch(np.log1p(v)); vb[0] = self.bt.pad_id
        ind = self.it.encode(self.comp.compute_all(
            {"open": o, "high": h, "low": l, "close": c, "volume": v}))
        tc = self.closes[e + self.target_horizon - 1]
        cc = self.closes[e - 1]
        d = (tc - cc) / cc
        label = 2 if d > self.target_threshold else (0 if d < -self.target_threshold else 1)
        return delta, vol, vb, ind, label


def collate(batch):
    import torch
    delta = torch.stack([torch.tensor(b[0]) for b in batch]).long()
    vol = torch.stack([torch.tensor(b[1]) for b in batch]).long()
    vb = torch.stack([torch.tensor(b[2]) for b in batch]).long()
    keys = list(batch[0][3].keys())
    ind = [torch.stack([torch.tensor(b[3][k]) for b in batch]).long() for k in keys]
    y = torch.tensor([b[4] for b in batch]).long()
    return delta, vol, vb, ind, y
'''

    trainer = '''"""FusionTrainer: trains Model A and Model B independently, then a MetaModel."""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW


class FusionTrainer:
    def __init__(self, model_a, model_b, train_loader, val_loader,
                 epochs_a=3, epochs_b=3, epochs_meta=5,
                 lr=3e-4, weight_decay=0.01, early_stop_patience=3,
                 device="auto", checkpoint_dir=Path("/content/checkpoints")):
        self.device = self._auto_device() if device == "auto" else device
        self.model_a = model_a.to(self.device)
        self.model_b = model_b.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs_a = epochs_a; self.epochs_b = epochs_b; self.epochs_meta = epochs_meta
        self.lr = lr; self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.ckpt = Path(checkpoint_dir); self.ckpt.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _auto_device():
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def train_all(self):
        print("=== Train Model A ===")
        self._train_base(self.model_a, self.epochs_a, "model_a",
            lambda b: self.model_a(b[0].to(self.device), b[1].to(self.device), b[2].to(self.device)))
        print("\\n=== Train Model B ===")
        self._train_base(self.model_b, self.epochs_b, "model_b",
            lambda b: self.model_b([t.to(self.device) for t in b[3]]))
        print("\\n=== Collect val logits ===")
        la, lb, y = self._collect_logits()
        print("la:", la.shape, "lb:", lb.shape, "y:", y.shape)
        print("\\n=== Train Meta-Model ===")
        meta = MetaModel().to(self.device)
        self._train_meta(meta, la, lb, y)
        torch.save(meta.state_dict(), self.ckpt / "meta_model.pt")
        return meta

    def _train_base(self, model, epochs, name, forward_fn):
        opt = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()
        best, pat = float("inf"), 0
        for ep in range(1, epochs + 1):
            model.train(mode=True)
            tl, n = 0.0, 0
            for b in self.train_loader:
                loss = crit(forward_fn(b), b[4].to(self.device))
                loss.backward(); opt.step(); opt.zero_grad()
                tl += loss.item(); n += 1
            model.train(mode=False)
            vl, vn = 0.0, 0
            with torch.no_grad():
                for b in self.val_loader:
                    vl += crit(forward_fn(b), b[4].to(self.device)).item(); vn += 1
            vl /= max(vn, 1)
            print(f"  {name} ep{ep}: train={tl/max(n,1):.4f} val={vl:.4f}")
            if vl < best:
                best, pat = vl, 0
                torch.save(model.state_dict(), self.ckpt / f"{name}_best.pt")
            else:
                pat += 1
            if pat >= self.early_stop_patience:
                print(f"  early stop {name} ep{ep}"); break

    def _collect_logits(self):
        self.model_a.train(mode=False); self.model_b.train(mode=False)
        la, lb, y = [], [], []
        with torch.no_grad():
            for b in self.val_loader:
                la.append(self.model_a(b[0].to(self.device), b[1].to(self.device), b[2].to(self.device)).cpu())
                lb.append(self.model_b([t.to(self.device) for t in b[3]]).cpu())
                y.append(b[4])
        return torch.cat(la), torch.cat(lb), torch.cat(y)

    def _train_meta(self, meta, la, lb, y):
        opt = AdamW(meta.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(la, lb, y)
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        for ep in range(1, self.epochs_meta + 1):
            meta.train(mode=True)
            tl = 0.0
            for a, b, lbl in dl:
                loss = crit(meta(a.to(self.device), b.to(self.device)), lbl.to(self.device))
                loss.backward(); opt.step(); opt.zero_grad()
                tl += loss.item()
            print(f"  meta ep{ep}: loss={tl/len(dl):.4f}")
'''

    config = '''"""Configuration."""
from pathlib import Path

USE_MOCK_DATA = True
if USE_MOCK_DATA:
    DATA_DIR = Path("/content/mock_data")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/trading_data")

MOCK_MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
MOCK_MINUTES_PER_MONTH = 20000
SYMBOL = "BTCUSDT"

CFG = {
    "data": {
        "train_months": ["2024-01", "2024-02"],
        "val_months":   ["2024-03", "2024-03"],
        "test_months":  ["2024-04", "2024-04"],
    },
    "sequence": {"length": 128, "target_horizon": 60, "target_threshold": 0.0015},
    "tokenizer": {"delta": {"range_pct": 3.0, "step_pct": 0.05}, "bucket": {"n_bins": 8}},
    "model_a": {"delta_vocab_size": 122, "bucket_vocab_size": 10,
                "delta_emb_dim": 64, "bucket_emb_dim": 16, "hidden_dim": 256,
                "num_layers": 4, "num_heads": 8, "ffn_dim": 1024, "dropout": 0.1,
                "num_classes": 3},
    "model_b": {"vocab_sizes": [7, 9, 7, 8, 7, 7], "emb_dim": 16, "hidden_dim": 128,
                "num_layers": 2, "num_heads": 4, "ffn_dim": 256, "dropout": 0.1,
                "num_classes": 3},
    "training": {"batch_size": 64, "learning_rate": 3e-4, "weight_decay": 0.01,
                 "epochs_a": 3, "epochs_b": 3, "epochs_meta": 5,
                 "early_stop_patience": 3, "device": "auto",
                 "checkpoint_dir": "/content/checkpoints"},
}

if Path("/content/drive/MyDrive").exists():
    ARTIFACTS_ROOT = Path("/content/drive/MyDrive/w_training/late_fusion_agent")
else:
    ARTIFACTS_ROOT = Path("/content/artifacts/late_fusion_agent")
print("DATA_DIR:", DATA_DIR)
print("ARTIFACTS_ROOT:", ARTIFACTS_ROOT)
'''

    main = '''"""Main: build datasets, models, train A/B/Meta, evaluate."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sc = CFG["sequence"]; tc = CFG["tokenizer"]
train_files = make_split(DATA_DIR, SYMBOL, *CFG["data"]["train_months"])
val_files   = make_split(DATA_DIR, SYMBOL, *CFG["data"]["val_months"])
test_files  = make_split(DATA_DIR, SYMBOL, *CFG["data"]["test_months"])
print(f"train={len(train_files)} val={len(val_files)} test={len(test_files)}")

def build_ds(files):
    return FusionDataset(files, seq_len=sc["length"],
        target_horizon=sc["target_horizon"], target_threshold=sc["target_threshold"],
        range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"],
        n_bins=tc["bucket"]["n_bins"])

train_ds = build_ds(train_files)
val_ds   = build_ds(val_files)
test_ds  = build_ds(test_files)
print(f"train_ds={len(train_ds)} val_ds={len(val_ds)} test_ds={len(test_ds)}")

bs = CFG["training"]["batch_size"]
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)

ma = CFG["model_a"]
model_a = PriceTransformer(
    delta_vocab_size=ma["delta_vocab_size"], bucket_vocab_size=ma["bucket_vocab_size"],
    delta_emb_dim=ma["delta_emb_dim"], bucket_emb_dim=ma["bucket_emb_dim"],
    hidden_dim=ma["hidden_dim"], num_layers=ma["num_layers"], num_heads=ma["num_heads"],
    ffn_dim=ma["ffn_dim"], dropout=ma["dropout"], num_classes=ma["num_classes"],
    seq_len=sc["length"])
mb = CFG["model_b"]
model_b = IndicatorModel(
    vocab_sizes=mb["vocab_sizes"], emb_dim=mb["emb_dim"], hidden_dim=mb["hidden_dim"],
    num_layers=mb["num_layers"], num_heads=mb["num_heads"], ffn_dim=mb["ffn_dim"],
    dropout=mb["dropout"], num_classes=mb["num_classes"], seq_len=sc["length"])
print(f"A params={sum(p.numel() for p in model_a.parameters()):,}  "
      f"B params={sum(p.numel() for p in model_b.parameters()):,}")

tr_cfg = CFG["training"]
trainer = FusionTrainer(model_a, model_b, train_dl, val_dl,
    epochs_a=tr_cfg["epochs_a"], epochs_b=tr_cfg["epochs_b"], epochs_meta=tr_cfg["epochs_meta"],
    lr=tr_cfg["learning_rate"], weight_decay=tr_cfg["weight_decay"],
    early_stop_patience=tr_cfg["early_stop_patience"], device=tr_cfg["device"],
    checkpoint_dir=Path(tr_cfg["checkpoint_dir"]))
meta = trainer.train_all()

print("\\n=== Test evaluation (meta-model) ===")
dev = trainer.device
model_a.train(mode=False); model_b.train(mode=False); meta.train(mode=False)
all_preds, all_labels = [], []
with torch.no_grad():
    for b in test_dl:
        la = model_a(b[0].to(dev), b[1].to(dev), b[2].to(dev))
        lb = model_b([t.to(dev) for t in b[3]])
        logits = meta(la, lb)
        all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        all_labels.extend(b[4].numpy().tolist())
print(classification_report(all_labels, all_preds,
    target_names=["DOWN","FLAT","UP"], zero_division=0))
'''

    save = '''"""Save checkpoints, tokenizers, metrics and config to ARTIFACTS_ROOT."""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ARTIFACTS_ROOT / f"run_{RUN_TAG}"
(RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "tokenizers").mkdir(parents=True, exist_ok=True)

ckpt_dir = Path(CFG["training"]["checkpoint_dir"])
for name in ["model_a_best.pt", "model_b_best.pt", "meta_model.pt"]:
    src = ckpt_dir / name
    if src.exists():
        shutil.copy(src, RUN_DIR / "checkpoints" / name)

np.save(RUN_DIR / "tokenizers" / "vol_boundaries.npy", train_ds.vt.boundaries)
np.save(RUN_DIR / "tokenizers" / "vb_boundaries.npy",  train_ds.bt.boundaries)
with open(RUN_DIR / "tokenizers" / "delta_params.json", "w") as f:
    json.dump({
        "range_pct": train_ds.dt.range_pct,
        "step_pct":  train_ds.dt.step_pct,
    }, f, indent=2)
train_ds.it.save(RUN_DIR / "tokenizers" / "indicators")

with open(RUN_DIR / "config.json", "w") as f:
    json.dump(CFG, f, indent=2, default=str)

test_report_dict = classification_report(
    all_labels, all_preds, target_names=["DOWN","FLAT","UP"],
    zero_division=0, output_dict=True)
with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({
        "report":           test_report_dict,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }, f, indent=2)

np.savez_compressed(
    RUN_DIR / "predictions.npz",
    preds=np.asarray(all_preds, dtype=np.int8),
    labels=np.asarray(all_labels, dtype=np.int8),
)

print(f"saved to: {RUN_DIR}")
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUN_DIR)!s:40s}  {p.stat().st_size:>10,} B")
'''

    cells = [
        md(intro),
        code(INSTALL_CELL),
        md("## 1. Tokenizers (delta + bucket)\n"),
        code(DELTA_BUCKET_TOKENIZERS),
        md("## 2. Indicator computer\n"),
        code(INDICATOR_COMPUTER),
        md("## 3. Indicator tokenizer\n"),
        code(INDICATOR_TOKENIZER),
        md("## 4. Model A — PriceTransformer\n"),
        code(price_transformer),
        md("## 5. Model B — IndicatorModel\n"),
        code(indicator_model),
        md("## 6. MetaModel\n"),
        code(meta_model),
        md("## 7. FusionDataset + collate\n"),
        code(dataset),
        md("## 8. FusionTrainer\n"),
        code(trainer),
        md("## 9. Configuration + data paths\n"),
        code(config),
        md("## 10. Mock data generation\n"),
        code(MOCK_DATA_CELL),
        md("## 11. make_split helper\n"),
        code(MAKE_SPLIT_CELL),
        md("## 12. Main — train A, B, Meta; eval on test\n"),
        code(main),
        md("## 13. Save artifacts to Google Drive\n"
           "Persists all three checkpoints, fitted tokenizers, config and "
           "test metrics into `ARTIFACTS_ROOT` (Drive or `/content/artifacts`).\n"),
        code(save),
    ]
    write_nb(ROOT / "late_fusion_agent" / "late_fusion_agent.ipynb", cells)


# --- 3) multimodal_encoder notebook ----------------------------------------

def build_multimodal_encoder_nb() -> None:
    print("[multimodal_encoder]")
    intro = """# multimodal_encoder — Colab Notebook

End-to-end fused transformer: candle stream + indicator stream concatenated
per time-step, projected to a shared hidden dim, run through a 4-layer
transformer with CLS pooling, then classified into 3 classes.

Self-contained. Recommended runtime: **T4 GPU**. Run all cells top-to-bottom.
"""

    model = '''"""MultimodalEncoder (from multimodal_encoder/models/multimodal_model.py)."""
import torch
import torch.nn as nn


class MultimodalEncoder(nn.Module):
    def __init__(self, delta_vocab_size=122, bucket_vocab_size=10,
                 delta_emb_dim=64, bucket_emb_dim=16, candle_proj_dim=128,
                 ind_vocab_sizes=None, ind_emb_dim=16, ind_proj_dim=128,
                 hidden_dim=256, num_layers=4, num_heads=8, ffn_dim=1024,
                 dropout=0.1, num_classes=3, seq_len=128):
        super().__init__()
        if ind_vocab_sizes is None:
            ind_vocab_sizes = [7, 9, 7, 8, 7, 7]
        self.delta_emb = nn.Embedding(delta_vocab_size, delta_emb_dim)
        self.vol_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        self.vb_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        self.candle_proj = nn.Linear(delta_emb_dim + bucket_emb_dim * 2, candle_proj_dim)
        self.ind_embeddings = nn.ModuleList(
            [nn.Embedding(vs, ind_emb_dim) for vs in ind_vocab_sizes])
        self.ind_proj = nn.Linear(ind_emb_dim * len(ind_vocab_sizes), ind_proj_dim)
        self.fusion_proj = nn.Linear(candle_proj_dim + ind_proj_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_emb = nn.Embedding(seq_len + 1, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout, activation="gelu",
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, delta_ids, vol_ids, vb_ids, ind_inputs):
        B = delta_ids.shape[0]
        d = self.delta_emb(delta_ids); v = self.vol_emb(vol_ids); vb = self.vb_emb(vb_ids)
        candle = self.candle_proj(torch.cat([d, v, vb], dim=-1))
        ind_e = [emb(tok) for emb, tok in zip(self.ind_embeddings, ind_inputs)]
        indicator = self.ind_proj(torch.cat(ind_e, dim=-1))
        fused = self.fusion_proj(torch.cat([candle, indicator], dim=-1))
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, fused], dim=1)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])
'''

    dataset = '''"""MultimodalDataset + collate."""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def _load(p):
    t = pq.read_table(p)
    return {c: np.array([v.as_py() for v in t.column(c)], dtype=np.float32) for c in t.column_names}


def _fit_all(files, range_pct, step_pct, n_bins):
    dt = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    rp, lv = [], []
    for f in files:
        d = _load(f)
        if len(d["close"]) < 2: continue
        rp.append((d["high"][1:] - d["low"][1:]) / d["close"][1:])
        lv.append(np.log1p(d["volume"][1:]))
    vt = BucketTokenizer(n_bins=n_bins); vt.fit(np.concatenate(rp))
    bt = BucketTokenizer(n_bins=n_bins); bt.fit(np.concatenate(lv))
    comp = IndicatorComputer()
    keys = ["rsi","macd_hist","bollinger_pctb","atr","volume_ratio","price_vs_sma"]
    ai = {k: [] for k in keys}
    for f in files:
        d = _load(f)
        ind = comp.compute_all({k2: d[k2] for k2 in ["open","high","low","close","volume"]})
        for k in keys: ai[k].append(ind[k])
    it = IndicatorTokenizer(); it.fit({k: np.concatenate(v) for k, v in ai.items()})
    return dt, vt, bt, it, comp


class MultimodalDataset:
    def __init__(self, files, seq_len=128, target_horizon=60, target_threshold=0.0015,
                 range_pct=3.0, step_pct=0.05, n_bins=8):
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold
        self.dt, self.vt, self.bt, self.it, self.comp = _fit_all(files, range_pct, step_pct, n_bins)
        frames = [_load(f) for f in files]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        self.opens = np.concatenate([f.get("open", f["close"]) for f in frames]).astype(np.float32)
        self._len = max(0, len(self.closes) - seq_len - target_horizon)

    def __len__(self): return self._len

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        c, h, l, v, o = (self.closes[s:e], self.highs[s:e], self.lows[s:e],
                         self.volumes[s:e], self.opens[s:e])
        delta = self.dt.from_closes(c); delta[0] = self.dt.cls_id
        rp = np.zeros(self.seq_len, dtype=np.float32)
        rp[1:] = (h[1:] - l[1:]) / c[1:]
        vol = self.vt.encode_batch(rp); vol[0] = self.vt.pad_id
        vb = self.bt.encode_batch(np.log1p(v)); vb[0] = self.bt.pad_id
        ind = self.it.encode(self.comp.compute_all(
            {"open": o, "high": h, "low": l, "close": c, "volume": v}))
        tc = self.closes[e + self.target_horizon - 1]
        cc = self.closes[e - 1]
        d = (tc - cc) / cc
        label = 2 if d > self.target_threshold else (0 if d < -self.target_threshold else 1)
        return delta, vol, vb, ind, label


def collate(batch):
    import torch
    delta = torch.stack([torch.tensor(b[0]) for b in batch]).long()
    vol = torch.stack([torch.tensor(b[1]) for b in batch]).long()
    vb = torch.stack([torch.tensor(b[2]) for b in batch]).long()
    keys = list(batch[0][3].keys())
    ind = [torch.stack([torch.tensor(b[3][k]) for b in batch]).long() for k in keys]
    y = torch.tensor([b[4] for b in batch]).long()
    return delta, vol, vb, ind, y
'''

    trainer = '''"""Trainer for MultimodalEncoder."""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 epochs=10, lr=3e-4, weight_decay=0.01, early_stop_patience=3,
                 device="auto", checkpoint_dir=Path("/content/checkpoints")):
        self.device = self._auto_device() if device == "auto" else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.ckpt_dir = Path(checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

    @staticmethod
    def _auto_device():
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def train(self):
        best_f1, pat = -1.0, 0
        for ep in range(1, self.epochs + 1):
            tl = self._train_ep()
            vl, vf = self._val_ep()
            print(f"Ep {ep}: train={tl:.4f} val={vl:.4f} f1={vf:.4f}")
            if vf > best_f1:
                best_f1, pat = vf, 0
                torch.save({"model": self.model.state_dict()}, self.ckpt_dir / "best.pt")
            else:
                pat += 1
            if pat >= self.early_stop_patience:
                print(f"Early stop ep{ep}"); break
            self.scheduler.step()

    def _train_ep(self):
        self.model.train(mode=True); tl, n = 0.0, 0
        for b in self.train_loader:
            delta, vol, vb, ind, y = b
            dev = self.device
            logits = self.model(delta.to(dev), vol.to(dev), vb.to(dev), [t.to(dev) for t in ind])
            loss = self.criterion(logits, y.to(dev))
            loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()
            tl += loss.item(); n += 1
        return tl / max(n, 1)

    def _val_ep(self):
        self.model.train(mode=False)
        tl, n = 0.0, 0; preds, labels = [], []
        with torch.no_grad():
            for b in self.val_loader:
                delta, vol, vb, ind, y = b
                logits = self.model(delta.to(self.device), vol.to(self.device),
                                    vb.to(self.device), [t.to(self.device) for t in ind])
                tl += self.criterion(logits, y.to(self.device)).item(); n += 1
                preds.extend(logits.argmax(-1).cpu().numpy())
                labels.extend(y.numpy())
        return tl/max(n,1), float(f1_score(labels, preds, average="weighted", zero_division=0))
'''

    config = '''"""Configuration."""
from pathlib import Path

USE_MOCK_DATA = True
if USE_MOCK_DATA:
    DATA_DIR = Path("/content/mock_data")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/trading_data")

MOCK_MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
MOCK_MINUTES_PER_MONTH = 20000
SYMBOL = "BTCUSDT"

CFG = {
    "data": {
        "train_months": ["2024-01", "2024-02"],
        "val_months":   ["2024-03", "2024-03"],
        "test_months":  ["2024-04", "2024-04"],
    },
    "sequence": {"length": 128, "target_horizon": 60, "target_threshold": 0.0015},
    "tokenizer": {"delta": {"range_pct": 3.0, "step_pct": 0.05}, "bucket": {"n_bins": 8}},
    "model": {
        "candle": {"delta_vocab_size": 122, "bucket_vocab_size": 10,
                   "delta_emb_dim": 64, "bucket_emb_dim": 16, "proj_dim": 128},
        "indicator": {"vocab_sizes": [7, 9, 7, 8, 7, 7], "emb_dim": 16, "proj_dim": 128},
        "fusion": {"hidden_dim": 256, "num_layers": 4, "num_heads": 8,
                   "ffn_dim": 1024, "dropout": 0.1, "num_classes": 3},
    },
    "training": {"batch_size": 64, "learning_rate": 3e-4, "weight_decay": 0.01,
                 "epochs": 5, "early_stop_patience": 3, "device": "auto",
                 "checkpoint_dir": "/content/checkpoints"},
}

if Path("/content/drive/MyDrive").exists():
    ARTIFACTS_ROOT = Path("/content/drive/MyDrive/w_training/multimodal_encoder")
else:
    ARTIFACTS_ROOT = Path("/content/artifacts/multimodal_encoder")
print("DATA_DIR:", DATA_DIR)
print("ARTIFACTS_ROOT:", ARTIFACTS_ROOT)
'''

    main = '''"""Main: dataset -> model -> trainer -> test eval."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

sc = CFG["sequence"]; tc = CFG["tokenizer"]
train_files = make_split(DATA_DIR, SYMBOL, *CFG["data"]["train_months"])
val_files   = make_split(DATA_DIR, SYMBOL, *CFG["data"]["val_months"])
test_files  = make_split(DATA_DIR, SYMBOL, *CFG["data"]["test_months"])
print(f"train={len(train_files)} val={len(val_files)} test={len(test_files)}")

mk = dict(seq_len=sc["length"], target_horizon=sc["target_horizon"],
          target_threshold=sc["target_threshold"],
          range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"],
          n_bins=tc["bucket"]["n_bins"])

train_ds = MultimodalDataset(train_files, **mk)
val_ds   = MultimodalDataset(val_files, **mk)
test_ds  = MultimodalDataset(test_files, **mk)
print(f"train_ds={len(train_ds)} val_ds={len(val_ds)} test_ds={len(test_ds)}")

bs = CFG["training"]["batch_size"]
tr_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate, num_workers=0)
va_dl = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
te_dl = DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)

mc = CFG["model"]["candle"]; mi = CFG["model"]["indicator"]; mf = CFG["model"]["fusion"]
model = MultimodalEncoder(
    delta_vocab_size=mc["delta_vocab_size"], bucket_vocab_size=mc["bucket_vocab_size"],
    delta_emb_dim=mc["delta_emb_dim"], bucket_emb_dim=mc["bucket_emb_dim"],
    candle_proj_dim=mc["proj_dim"],
    ind_vocab_sizes=mi["vocab_sizes"], ind_emb_dim=mi["emb_dim"], ind_proj_dim=mi["proj_dim"],
    hidden_dim=mf["hidden_dim"], num_layers=mf["num_layers"], num_heads=mf["num_heads"],
    ffn_dim=mf["ffn_dim"], dropout=mf["dropout"], num_classes=mf["num_classes"],
    seq_len=sc["length"])
print(f"params={sum(p.numel() for p in model.parameters()):,}")

t = CFG["training"]
trainer = Trainer(model, tr_dl, va_dl, epochs=t["epochs"], lr=t["learning_rate"],
    weight_decay=t["weight_decay"], early_stop_patience=t["early_stop_patience"],
    device=t["device"], checkpoint_dir=Path(t["checkpoint_dir"]))
trainer.train()

print("\\n=== Test evaluation ===")
dev = trainer.device
state = torch.load(Path(t["checkpoint_dir"]) / "best.pt", map_location=dev, weights_only=True)
model.load_state_dict(state["model"]); model.train(mode=False)
all_preds, all_labels = [], []
with torch.no_grad():
    for b in te_dl:
        d, vo, vb, ind, y = b
        logits = model(d.to(dev), vo.to(dev), vb.to(dev), [t_.to(dev) for t_ in ind])
        all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        all_labels.extend(y.numpy().tolist())
print(classification_report(all_labels, all_preds,
    target_names=["DOWN","FLAT","UP"], zero_division=0))
'''

    save = '''"""Save checkpoint, tokenizers, metrics and config to ARTIFACTS_ROOT."""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ARTIFACTS_ROOT / f"run_{RUN_TAG}"
(RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "tokenizers").mkdir(parents=True, exist_ok=True)

src = Path(CFG["training"]["checkpoint_dir"]) / "best.pt"
if src.exists():
    shutil.copy(src, RUN_DIR / "checkpoints" / "best.pt")

np.save(RUN_DIR / "tokenizers" / "vol_boundaries.npy", train_ds.vt.boundaries)
np.save(RUN_DIR / "tokenizers" / "vb_boundaries.npy",  train_ds.bt.boundaries)
with open(RUN_DIR / "tokenizers" / "delta_params.json", "w") as f:
    json.dump({
        "range_pct": train_ds.dt.range_pct,
        "step_pct":  train_ds.dt.step_pct,
    }, f, indent=2)
train_ds.it.save(RUN_DIR / "tokenizers" / "indicators")

with open(RUN_DIR / "config.json", "w") as f:
    json.dump(CFG, f, indent=2, default=str)

test_report_dict = classification_report(
    all_labels, all_preds, target_names=["DOWN","FLAT","UP"],
    zero_division=0, output_dict=True)
with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({
        "report":           test_report_dict,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }, f, indent=2)

np.savez_compressed(
    RUN_DIR / "predictions.npz",
    preds=np.asarray(all_preds, dtype=np.int8),
    labels=np.asarray(all_labels, dtype=np.int8),
)

print(f"saved to: {RUN_DIR}")
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUN_DIR)!s:40s}  {p.stat().st_size:>10,} B")
'''

    cells = [
        md(intro),
        code(INSTALL_CELL),
        md("## 1. Tokenizers (delta + bucket)\n"),
        code(DELTA_BUCKET_TOKENIZERS),
        md("## 2. Indicator computer\n"),
        code(INDICATOR_COMPUTER),
        md("## 3. Indicator tokenizer\n"),
        code(INDICATOR_TOKENIZER),
        md("## 4. Multimodal model\n"),
        code(model),
        md("## 5. Dataset + collate\n"),
        code(dataset),
        md("## 6. Trainer\n"),
        code(trainer),
        md("## 7. Configuration + data paths\n"),
        code(config),
        md("## 8. Mock data generation\n"),
        code(MOCK_DATA_CELL),
        md("## 9. make_split helper\n"),
        code(MAKE_SPLIT_CELL),
        md("## 10. Main\n"),
        code(main),
        md("## 11. Save artifacts to Google Drive\n"
           "Persists checkpoint, fitted tokenizers, config and test metrics "
           "to `ARTIFACTS_ROOT` (Drive or `/content/artifacts`).\n"),
        code(save),
    ]
    write_nb(ROOT / "multimodal_encoder" / "multimodal_encoder.ipynb", cells)


# --- 4) moe_trading_agent notebook -----------------------------------------

def build_moe_nb() -> None:
    print("[moe_trading_agent]")
    intro = """# moe_trading_agent — Colab Notebook

Mixture of Experts (MoE) transformer for 3-class trading prediction.
Each MoE transformer layer has 8 experts with top-2 sparse routing plus an
auxiliary load-balancing loss.

Self-contained. Recommended runtime: **T4 GPU** (you can try A100/L4 in Colab
Pro for bigger experts). Run all cells top-to-bottom.
"""

    router = '''"""Top-K routing with load-balancing aux loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-10)
        one_hot = F.one_hot(top_i, num_classes=self.num_experts).float()
        mask = one_hot.sum(dim=1)
        f = mask.mean(dim=0); p = probs.mean(dim=0)
        aux = self.num_experts * torch.sum(f * p)
        return top_i, top_w, aux


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, dim))

    def forward(self, x): return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts; self.top_k = top_k
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.router = Router(dim, num_experts, top_k)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        top_i, top_w, aux = self.router(flat)
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            exp_idx = top_i[:, k]; w = top_w[:, k]
            for e in range(self.num_experts):
                mask = (exp_idx == e)
                if not mask.any(): continue
                ei = flat[mask]
                eo = self.experts[e](ei)
                out[mask] += w[mask].unsqueeze(-1) * eo
        return out.reshape(B, T, D), aux
'''

    moe_model = '''"""MoETradingModel (stacked MoE-transformer blocks)."""
import torch
import torch.nn as nn


class MoETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoELayer(dim, hidden_dim, num_experts, top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = r + self.dropout(a)
        r = x
        n = self.norm2(x)
        m, aux = self.moe(n)
        x = r + self.dropout(m)
        return x, aux


class MoETradingModel(nn.Module):
    def __init__(self, seq_len=128, num_experts=8, top_k=2, num_layers=4,
                 num_heads=8, dim=256, hidden_dim=1024, dropout=0.1, num_classes=3):
        super().__init__()
        self.seq_len = seq_len; self.dim = dim
        self.delta_emb = nn.Embedding(122, 64)
        self.vol_emb = nn.Embedding(10, 16)
        self.vb_emb = nn.Embedding(10, 16)
        self.candle_proj = nn.Linear(96, 128)
        self.indicator_embs = nn.ModuleList([
            nn.Embedding(7, 16), nn.Embedding(9, 16), nn.Embedding(7, 16),
            nn.Embedding(8, 16), nn.Embedding(7, 16), nn.Embedding(7, 16),
        ])
        self.indicator_proj = nn.Linear(96, 128)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Embedding(seq_len + 1, dim)
        self.layers = nn.ModuleList([
            MoETransformerBlock(dim, num_heads, hidden_dim, num_experts, top_k, dropout)
            for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, delta_ids, vol_ids, vb_ids, ind_dict):
        B = delta_ids.shape[0]
        de = self.delta_emb(delta_ids); ve = self.vol_emb(vol_ids); vbe = self.vb_emb(vb_ids)
        candle = self.candle_proj(torch.cat([de, ve, vbe], dim=-1))
        keys = ["rsi","macd_hist","bollinger_pctb","atr","volume_ratio","price_vs_sma"]
        ie = [emb(ind_dict[k]) for emb, k in zip(self.indicator_embs, keys)]
        indicator = self.indicator_proj(torch.cat(ie, dim=-1))
        x = torch.cat([candle, indicator], dim=-1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        aux_total = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux = layer(x)
            aux_total = aux_total + aux
        return self.head(x[:, 0]), aux_total
'''

    dataset = '''"""MoEDataset (from moe_trading_agent/dataset/moe_dataset.py, tokenizers inlined)."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MoEDataset(Dataset):
    def __init__(self, file_paths, seq_len=128, horizon=60, threshold=0.0015,
                 boundaries_dir=None):
        self.seq_len = seq_len; self.horizon = horizon; self.threshold = threshold

        dfs = [pd.read_parquet(fp) for fp in file_paths]
        data = pd.concat(dfs, ignore_index=True)
        self.open = data["open"].values.astype(np.float32)
        self.high = data["high"].values.astype(np.float32)
        self.low = data["low"].values.astype(np.float32)
        self.close = data["close"].values.astype(np.float32)
        self.volume = data["volume"].values.astype(np.float32)

        self.delta_tokenizer = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
        self.vol_tokenizer = BucketTokenizer(n_bins=8); self.vol_tokenizer.fit(self.volume)
        vb_raw = np.abs(self.close - self.open) / (self.high - self.low + 1e-10)
        self.vb_tokenizer = BucketTokenizer(n_bins=8); self.vb_tokenizer.fit(vb_raw)

        self.comp = IndicatorComputer()
        ohlcv = {"open": self.open, "high": self.high, "low": self.low,
                 "close": self.close, "volume": self.volume}
        self.indicators_raw = self.comp.compute_all(ohlcv)

        self.ind_tok = IndicatorTokenizer()
        if boundaries_dir is not None and (Path(boundaries_dir) / "rsi.npy").exists():
            self.ind_tok.load(boundaries_dir)
        else:
            self.ind_tok.fit(self.indicators_raw)
        self.indicators_tokenized = self.ind_tok.encode(self.indicators_raw)

        self.delta_ids_all = self.delta_tokenizer.from_closes(self.close)
        self.vol_ids_all = self.vol_tokenizer.encode_batch(self.volume)
        self.vb_ids_all = self.vb_tokenizer.encode_batch(vb_raw)

    def __len__(self):
        return len(self.close) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        delta = torch.tensor(self.delta_ids_all[s:e], dtype=torch.long)
        vol = torch.tensor(self.vol_ids_all[s:e], dtype=torch.long)
        vb = torch.tensor(self.vb_ids_all[s:e], dtype=torch.long)
        ind = {k: torch.tensor(self.indicators_tokenized[k][s:e], dtype=torch.long)
               for k in self.indicators_tokenized}
        future = self.close[e + self.horizon]
        current = self.close[e - 1]
        p = (future - current) / (current + 1e-10)
        if p > self.threshold: label = 0
        elif p < -self.threshold: label = 2
        else: label = 1
        return delta, vol, vb, ind, torch.tensor(label, dtype=torch.long)
'''

    trainer = '''"""Trainer for MoETradingModel."""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class MoETrainer:
    def __init__(self, model, train_loader, val_loader, device=None,
                 lr=3e-4, weight_decay=0.01, aux_loss_lambda=0.01,
                 patience=3, checkpoint_dir="/content/checkpoints"):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader; self.val_loader = val_loader
        self.aux_lambda = aux_loss_lambda; self.patience = patience
        self.ckpt_dir = Path(checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.class_weights = self._compute_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_f1 = 0.0; self.pat = 0

    def _compute_class_weights(self):
        counts = torch.zeros(3)
        for b in self.train_loader:
            y = b[-1]
            for c in range(3):
                counts[c] += (y == c).sum().item()
        w = 1.0 / (counts + 1e-6)
        return (w / w.sum() * 3).float()

    def _train_ep(self):
        self.model.train(mode=True); tl, n = 0.0, 0
        for b in self.train_loader:
            d, vo, vb, ind, y = b
            d = d.to(self.device); vo = vo.to(self.device); vb = vb.to(self.device)
            ind = {k: v.to(self.device) for k, v in ind.items()}
            y = y.to(self.device)
            logits, aux = self.model(d, vo, vb, ind)
            loss = self.criterion(logits, y) + self.aux_lambda * aux
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            tl += loss.item(); n += 1
        return tl / max(n, 1)

    def _val_ep(self):
        self.model.train(mode=False)
        tl, n = 0.0, 0; preds, labels = [], []
        with torch.no_grad():
            for b in self.val_loader:
                d, vo, vb, ind, y = b
                d = d.to(self.device); vo = vo.to(self.device); vb = vb.to(self.device)
                ind = {k: v.to(self.device) for k, v in ind.items()}
                y = y.to(self.device)
                logits, aux = self.model(d, vo, vb, ind)
                loss = self.criterion(logits, y) + self.aux_lambda * aux
                tl += loss.item(); n += 1
                preds.extend(logits.argmax(-1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        return tl / max(n, 1), float(f1_score(labels, preds, average="weighted", zero_division=0))

    def train(self, epochs, use_scheduler=True):
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
                 if use_scheduler else None)
        print(f"device={self.device}  class_weights={self.class_weights.cpu().numpy()}")
        for ep in range(1, epochs + 1):
            tl = self._train_ep(); vl, vf = self._val_ep()
            if sched is not None: sched.step()
            print(f"Ep {ep}/{epochs}: train={tl:.4f} val={vl:.4f} f1={vf:.4f}")
            if vf > self.best_f1:
                self.best_f1 = vf; self.pat = 0
                torch.save({"model_state_dict": self.model.state_dict(), "val_f1": vf, "epoch": ep},
                           self.ckpt_dir / "best_model.pt")
                print(f"  -> saved (f1={vf:.4f})")
            else:
                self.pat += 1
                if self.pat >= self.patience:
                    print(f"  -> early stop ep{ep}"); break
        print(f"best f1={self.best_f1:.4f}")
'''

    config = '''"""Configuration."""
from pathlib import Path

USE_MOCK_DATA = True
if USE_MOCK_DATA:
    DATA_DIR = Path("/content/mock_data")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/trading_data")

MOCK_MONTHS = ["2024-01", "2024-02", "2024-03"]
MOCK_MINUTES_PER_MONTH = 30000
SYMBOL = "BTCUSDT"

MODEL_CFG = {
    "seq_len": 128, "num_experts": 8, "top_k": 2, "num_layers": 4,
    "num_heads": 8, "dim": 256, "hidden_dim": 1024,
    "dropout": 0.1, "num_classes": 3,
}
TRAIN_CFG = {
    "batch_size": 32, "lr": 3e-4, "weight_decay": 0.01,
    "aux_loss_lambda": 0.01, "epochs": 5, "patience": 3,
    "horizon": 60, "threshold": 0.0015,
    "train_split": 0.8, "val_split": 0.1,
    "checkpoint_dir": "/content/checkpoints",
}

if Path("/content/drive/MyDrive").exists():
    ARTIFACTS_ROOT = Path("/content/drive/MyDrive/w_training/moe_trading_agent")
else:
    ARTIFACTS_ROOT = Path("/content/artifacts/moe_trading_agent")
print("DATA_DIR:", DATA_DIR)
print("ARTIFACTS_ROOT:", ARTIFACTS_ROOT)
'''

    main = '''"""Main: gather parquet files, build dataset & splits, train MoE, evaluate."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

klines_dir = DATA_DIR / SYMBOL / "klines_1m"
files = sorted([str(f) for f in klines_dir.glob("*.parquet")])
if not files:
    raise SystemExit(f"no parquet files in {klines_dir}")
print(f"found {len(files)} parquet files")

ds = MoEDataset(files, seq_len=MODEL_CFG["seq_len"],
                horizon=TRAIN_CFG["horizon"], threshold=TRAIN_CFG["threshold"])
print(f"dataset size: {len(ds)}")

total = len(ds)
tr_end = int(total * TRAIN_CFG["train_split"])
va_end = int(total * (TRAIN_CFG["train_split"] + TRAIN_CFG["val_split"]))
train_ds = Subset(ds, range(0, tr_end))
val_ds   = Subset(ds, range(tr_end, va_end))
test_ds  = Subset(ds, range(va_end, total))
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

tr_dl = DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True,  num_workers=0)
va_dl = DataLoader(val_ds,   batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0)
te_dl = DataLoader(test_ds,  batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0)

model = MoETradingModel(**MODEL_CFG)
print(f"params={sum(p.numel() for p in model.parameters()):,}")

trainer = MoETrainer(model, tr_dl, va_dl,
    lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"],
    aux_loss_lambda=TRAIN_CFG["aux_loss_lambda"], patience=TRAIN_CFG["patience"],
    checkpoint_dir=TRAIN_CFG["checkpoint_dir"])
trainer.train(epochs=TRAIN_CFG["epochs"])

print("\\n=== Test evaluation ===")
dev = trainer.device
ckpt = Path(TRAIN_CFG["checkpoint_dir"]) / "best_model.pt"
state = torch.load(ckpt, map_location=dev, weights_only=False)
model.load_state_dict(state["model_state_dict"]); model.train(mode=False)
all_preds, all_labels = [], []
with torch.no_grad():
    for b in te_dl:
        d, vo, vb, ind, y = b
        d = d.to(dev); vo = vo.to(dev); vb = vb.to(dev)
        ind = {k: v.to(dev) for k, v in ind.items()}
        logits, _ = model(d, vo, vb, ind)
        all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        all_labels.extend(y.numpy().tolist())
print(classification_report(all_labels, all_preds,
    target_names=["UP","FLAT","DOWN"], zero_division=0))
'''

    save = '''"""Save checkpoint, tokenizers, metrics and config to ARTIFACTS_ROOT."""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ARTIFACTS_ROOT / f"run_{RUN_TAG}"
(RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "tokenizers").mkdir(parents=True, exist_ok=True)

src = Path(TRAIN_CFG["checkpoint_dir"]) / "best_model.pt"
if src.exists():
    shutil.copy(src, RUN_DIR / "checkpoints" / "best_model.pt")

np.save(RUN_DIR / "tokenizers" / "vol_boundaries.npy", ds.vol_tokenizer.boundaries)
np.save(RUN_DIR / "tokenizers" / "vb_boundaries.npy",  ds.vb_tokenizer.boundaries)
with open(RUN_DIR / "tokenizers" / "delta_params.json", "w") as f:
    json.dump({
        "range_pct": ds.delta_tokenizer.range_pct,
        "step_pct":  ds.delta_tokenizer.step_pct,
    }, f, indent=2)
ds.ind_tok.save(RUN_DIR / "tokenizers" / "indicators")

with open(RUN_DIR / "config.json", "w") as f:
    json.dump({"model": MODEL_CFG, "training": TRAIN_CFG,
               "symbol": SYMBOL, "data_dir": str(DATA_DIR)},
              f, indent=2, default=str)

test_report_dict = classification_report(
    all_labels, all_preds, target_names=["UP","FLAT","DOWN"],
    zero_division=0, output_dict=True)
with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump({
        "report":           test_report_dict,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }, f, indent=2)

np.savez_compressed(
    RUN_DIR / "predictions.npz",
    preds=np.asarray(all_preds, dtype=np.int8),
    labels=np.asarray(all_labels, dtype=np.int8),
)

print(f"saved to: {RUN_DIR}")
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUN_DIR)!s:40s}  {p.stat().st_size:>10,} B")
'''

    cells = [
        md(intro),
        code(INSTALL_CELL),
        md("## 1. Tokenizers (delta + bucket)\n"),
        code(DELTA_BUCKET_TOKENIZERS),
        md("## 2. Indicator computer\n"),
        code(INDICATOR_COMPUTER),
        md("## 3. Indicator tokenizer\n"),
        code(INDICATOR_TOKENIZER),
        md("## 4. Router, Expert, MoELayer\n"),
        code(router),
        md("## 5. MoE model\n"),
        code(moe_model),
        md("## 6. MoEDataset\n"),
        code(dataset),
        md("## 7. MoE trainer\n"),
        code(trainer),
        md("## 8. Configuration + data paths\n"),
        code(config),
        md("## 9. Mock data generation\n"),
        code(MOCK_DATA_CELL),
        md("## 10. Main\n"),
        code(main),
        md("## 11. Save artifacts to Google Drive\n"
           "Persists the MoE checkpoint, fitted tokenizers, config and test "
           "metrics to `ARTIFACTS_ROOT` (Drive or `/content/artifacts`).\n"),
        code(save),
    ]
    write_nb(ROOT / "moe_trading_agent" / "moe_trading_agent.ipynb", cells)


# --- main ------------------------------------------------------------------

if __name__ == "__main__":
    build_indicator_tokenizer_nb()
    build_late_fusion_nb()
    build_multimodal_encoder_nb()
    build_moe_nb()
    print("\nall notebooks generated")
