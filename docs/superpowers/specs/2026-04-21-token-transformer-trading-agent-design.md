# Token-First Transformer Trading Agent — Design Spec

**Date:** 2026-04-21
**Status:** Approved
**Target hardware:** MacBook M2 Max (32 GB RAM, MPS)

## 1. Goal

Build an autonomous trading agent that tokenizes market data, trains a small Transformer classifier, and validates via backtesting. MVP scope: BTCUSDT 1m candles, 3-class prediction (UP/FLAT/DOWN) over 60-candle horizon.

## 2. Data Source

- **Path:** `w_trender/backtests/data/BTCUSDT/klines_1m/*.parquet`
- **Size:** 37 monthly files (Feb 2023 — Feb 2026), ~1.5M rows total
- **Columns:** `timestamp(int32), open(float32), high(float32), low(float32), close(float32), volume(float32)`
- **Never load all into RAM.** Stream per-month.

### Feature extraction (per candle)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `delta_close` | `(close[i] - close[i-1]) / close[i-1]` | Main signal |
| `range_pct` | `(high - low) / close` | Volatility |
| `body_pct` | `(close - open) / close` | Direction strength |
| `log_volume` | `log(1 + volume)` | Activity level |

### Tokenization

**Price delta tokenizer:**
- Range: [-3%, +3%]
- Step: 0.05%
- Bins: ~120 + PAD(0) + CLS(1) → vocab_size = 122
- Out-of-range values clipped to nearest bin

**Volatility bucket tokenizer:**
- 8 bins by quantile thresholds (fitted on train set)
- vocab_size = 10 (8 + PAD + special)

**Volume bucket tokenizer:**
- 8 bins by quantile thresholds (fitted on train set)
- vocab_size = 10 (8 + PAD + special)

### Sequence

- Sequence length: 128 candles (~2h 8min context)
- Prepend CLS token (position 0) → used for pooling

### Target

Close price delta after 60 candles (1 hour):
- **UP** (class 2): delta > +0.15%
- **DOWN** (class 0): delta < -0.15%
- **FLAT** (class 1): in between

Thresholds tunable. ~0.15% chosen to approximate median absolute move over 60 candles.

### Train/Val/Test Split

| Split | Period | Months |
|-------|--------|--------|
| Train | 2023-02 — 2025-06 | 28 |
| Val | 2025-07 — 2025-10 | 4 |
| Test | 2025-11 — 2026-02 | 4 |

Strict chronological split. No shuffling across boundaries.

## 3. Model Architecture

```
Input: 3 token streams × 128 positions
         |
[Embedding layers] (separate per stream)
  - delta_emb: vocab=122, dim=64
  - vol_emb: vocab=10, dim=16
  - vbucket_emb: vocab=10, dim=16
         |
[Concat] -> dim=96
         |
[Linear projection] -> dim=256
         |
[+ Learned positional embeddings] dim=256
         |
[Transformer Encoder]
  - 4 layers
  - 8 attention heads
  - dim=256, FFN dim=1024
  - dropout=0.1
         |
[CLS pooling] (output at position 0)
         |
[MLP Head]
  - Linear(256, 128) + ReLU + Dropout(0.1)
  - Linear(128, 3)
         |
Action logits: [DOWN, FLAT, UP]
```

**Total parameters:** ~5-7M (comfortable for M2 Max MPS training)

**Weight initialization:** Default PyTorch (Xavier for linear, truncated normal for embeddings).

## 4. Training

| Parameter | Value |
|-----------|-------|
| Loss | CrossEntropy with class weights |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealing |
| Batch size | 64 |
| Gradient accumulation | 2 (effective batch=128 if needed) |
| Epochs | 5-10 |
| Early stopping | Patience=3 on val loss |
| Device | MPS (Metal Performance Shaders) |
| Precision | float32 (MPS unstable with float16) |

### Class weights

Computed from train set label distribution. Expected: FLAT dominant (~50-60%), UP and DOWN each ~20-25%. Weights = inverse frequency, normalized.

### Metrics (tracked per epoch)

- Accuracy per class
- Confusion matrix
- Precision / Recall / F1 for UP and DOWN
- Weighted F1

### Checkpoints

Save best model (by val weighted F1) + last model per epoch. Path: `w_training/checkpoints/`.

## 5. Backtest Engine

Simple sequential backtest on test period.

### Rules

- Model predicts UP → enter long
- Model predicts DOWN → enter short
- Model predicts FLAT → no position
- Max 1 position at a time
- Entry: next candle open after prediction
- Exit conditions (first triggered):
  - Stop-loss: -0.5% from entry
  - Take-profit: +1.0% from entry
  - Max hold: 60 candles (forced exit at current price)

### Costs

- Commission: 0.04% per trade (Binance taker fee) × 2 (entry + exit)

### Metrics

| Metric | Description |
|--------|-------------|
| Total PnL | Cumulative % return |
| Sharpe ratio | Annualized, risk-free=0 |
| Max drawdown | Worst peak-to-trough |
| Win rate | % profitable trades |
| Trade count | Total trades executed |
| Profit factor | Gross profit / gross loss |
| Avg trade duration | In candles |

## 6. Project Structure

```
w_training/token_first_transformer/
├── pyproject.toml              # torch, pyarrow, polars, pyyaml, numpy
├── configs/
│   └── default.yaml            # hyperparams, paths, tokenizer bins
├── tokenizer/
│   ├── __init__.py
│   ├── delta_tokenizer.py      # price delta -> token IDs
│   └── bucket_tokenizer.py     # vol/volume quantile buckets
├── dataset/
│   ├── __init__.py
│   └── klines_dataset.py       # streaming PyTorch Dataset
├── models/
│   ├── __init__.py
│   └── price_transformer.py    # Transformer model
├── training/
│   ├── __init__.py
│   └── trainer.py              # training loop, validation, checkpoints
├── backtest/
│   ├── __init__.py
│   └── engine.py               # backtest engine
├── scripts/
│   ├── train.py                # CLI: train model
│   ├── evaluate.py             # CLI: eval on test set
│   └── backtest.py             # CLI: run backtest
├── checkpoints/                # saved models (gitignored)
├── logs/                       # training logs (gitignored)
└── tests/
    ├── test_tokenizer.py
    ├── test_dataset.py
    └── test_model.py
```

### Dependencies

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "pyarrow>=15.0",
    "polars>=0.20",
    "pyyaml>=6.0",
    "numpy>=1.26",
]
```

Data path configured in `configs/default.yaml`, points to `../../../w_trender/backtests/data/BTCUSDT`.

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Data leakage | Strict chronological split. Features computed only from past data. Target uses future close but never fed as input. |
| Class imbalance | Weighted CE loss. Monitor per-class metrics, not just accuracy. |
| Overfitting | Dropout, weight decay, early stopping. Small model (5-7M params) relative to 1.5M samples. |
| Unrealistic backtest | Include commission, stop-loss, max hold period. No lookahead in entry timing. |
| MPS instability | float32 only. Fallback to CPU if MPS errors occur. Catch MPS OOM → reduce batch size. |

## 8. Out of Scope (Future Phases)

- Multi-asset training
- RL (policy gradient)
- Order book data
- 1s / 100ms candles
- Hyperparameter search (manual tuning in V1)
- Real-time inference
