# token_first_transformer

Token-based Transformer classifier that treats price action as a language. It
discretizes BTCUSDT 1m candles into discrete tokens and trains a small
Transformer encoder for 3-class direction prediction (DOWN / FLAT / UP).

## Idea

Instead of feeding raw floats to the model, each candle is converted into a
small set of discrete tokens — a "market language". Percentage price deltas are
quantized into fixed bins, while per-candle volatility (high-low range) and
volume are quantized into quantile buckets fitted on the training data. The
Transformer then learns over sequences of these tokens, the same way a language
model learns over word tokens.

## Architecture

- 4-layer `TransformerEncoder`, 8 attention heads, hidden dim 256, GELU FFN of
  width 1024, dropout 0.1 (~3.26M params).
- Three embedding tables, concatenated then projected to the hidden dim:
  - **delta** — price-delta token, vocab 122 (`±3.0%` range, `0.05%` step → 120
    bins + PAD + CLS), embedding dim 64.
  - **volatility** — high-low range bucket, 8 quantile bins (vocab 10), dim 16.
  - **volume** — `log1p(volume)` bucket, 8 quantile bins (vocab 10), dim 16.
- Learned positional embeddings over a 128-token context.
- A `[CLS]` token at position 0; its final hidden state feeds a 2-layer MLP head
  producing 3 logits.

## Input / Output

- **Input:** three aligned token streams (delta, volatility, volume) over a
  128-candle window.
- **Output:** 3 class logits.
- **Target:** sign of the return over a 60-candle horizon, with a `±0.15%`
  threshold → DOWN (0) / FLAT (1) / UP (2).

## Layout

```
token_first_transformer/
├── tokenizer/      # DeltaTokenizer (binned deltas) + BucketTokenizer (quantile buckets)
├── dataset/        # parquet loading, tokenizer fitting, windowing, labels (KlinesDataset)
├── models/         # PriceTransformer encoder + classification head
├── training/       # Trainer: AdamW + CosineAnnealing, class weights, early stopping
├── backtest/       # sequential BacktestEngine with SL/TP/max-hold and commission
├── scripts/        # train / evaluate / backtest CLI entry points
├── configs/        # default.yaml (data splits, tokenizer, model, training, backtest)
└── tests/          # unit + integration tests
```

## Quickstart

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv venv && uv pip install -e ".[dev]"
```

or with venv + pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest -q
```

Train, evaluate, and backtest (data paths and hyperparameters live in
`configs/default.yaml`):

```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
python scripts/backtest.py --checkpoint checkpoints/best.pt --config configs/default.yaml
```

Training writes `latest.pt` atomically every 100 optimizer steps by default and
after each epoch. It contains the model, optimizer, scheduler, random state,
metrics, and the next batch position. The training-fitted bucket boundaries are
saved beside it. To resume with the same config and data:

```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pt
```

Keep the complete checkpoint directory on durable storage when training on an
ephemeral machine. `best.pt` is for evaluation; `latest.pt` is for continuation.
Set `training.checkpoint_every_steps` for a different save interval.

Training uses AdamW with cosine annealing over up to 10 epochs (early stopping
on weighted F1), device `auto` (MPS / CUDA / CPU). The backtest runs sequentially
with `-0.5%` stop-loss, `+1.0%` take-profit, 60-candle max hold, and `0.04%`
commission per side. `configs/default.yaml` points `data.data_dir` at the
server1 BTCUSDT parquet root (`/mnt/second/trender/backtests/data`); override
that path in a separate config when running on a Mac or another machine.

Set `training.class_weighting: balanced` to use inverse-frequency loss weights
from the training split's valid windows. The default `none` keeps unweighted
cross-entropy. Keep the same setting when resuming a checkpoint.

## Status

Code complete; 48 tests pass. A BTCUSDT 1m run on 2026-09-25 used the
chronological splits in `configs/default.yaml`. Its run manifest and durable
checkpoints are on server1 at
`/mnt/third/projects/trading/training/checkpoints/clore-btc1m-20260925T0840Z/`.
The full-history model reached weighted F1 0.4206 on validation and 0.4212 on
the held-out test. The test backtest returned -87.67% with 0.04% commission per
fill, or -99.10% when adding an assumed 0.05% slippage per fill. The current
classifier is not a viable trading strategy; F1 is not a substitute for
cost-adjusted validation. A separate recent-history experiment is in progress.

---

Part of the [marketglot](../README.md) monorepo.
