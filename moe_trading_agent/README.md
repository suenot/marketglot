# moe_trading_agent

A Mixture-of-Experts (MoE) transformer for 3-class trading prediction (UP / FLAT / DOWN) on BTCUSDT 1-minute candles.

## Idea

Different market regimes (trending, ranging, volatile, quiet) reward different behaviours.
Instead of forcing one dense feed-forward network to handle all of them, each MoE layer
holds a pool of specialist **experts** and a lightweight **router** that sends every token
to only its top-`k` experts. Sparse activation lets experts specialize per regime while
keeping the active compute per token small.

## Architecture

Each transformer block replaces the usual FFN with an MoE block
(`LN -> MultiheadAttention -> +residual`, then `LN -> MoE -> +residual`).
Routing inside the MoE layer (per token):

```
token x ──► router (Linear -> softmax over 8 experts)
                │  top-2 selection, renormalized weights
                ▼
        ┌── expert e_a ──┐
   x ──►│                ├─► w_a·y_a + w_b·y_b ─► out
        └── expert e_b ──┘
        (other 6 experts skipped for this token)
```

Defaults (`configs/default.yaml`):

- **8 experts** per MoE layer, **top_k = 2**; **4 MoE transformer layers**
- `dim = 256`, `num_heads = 8`, expert `hidden_dim = 1024`; each expert is `Linear -> ReLU -> Linear`
- Router: `Linear` gate -> softmax -> top-k, with an **auxiliary load-balancing loss**
  (`aux = num_experts * sum(f_i * p_i)`) to keep expert usage balanced
- `seq_len = 128` (+ a CLS token for classification), 3 output classes
- ~10-15M parameters total, only top_k experts active per token

Total loss: `CrossEntropy(class-weighted) + 0.01 * aux_loss`. Training uses AdamW,
cosine LR schedule, gradient clipping, inverse-frequency class weights, and early
stopping on weighted validation F1.

### Input pipeline

Shares the same candle + indicator token scheme as `multimodal_encoder`:

- **Candle tokens:** delta (close-to-close move), volume bucket, and body-ratio bucket
  embeddings, fused to a 128-d candle representation.
- **Indicator tokens:** 6 tokenized indicators (RSI, MACD histogram, Bollinger %b, ATR,
  volume ratio, price-vs-SMA), fused to a 128-d indicator representation.

Both reps are concatenated to the `dim = 256` model width. Tokenizers are imported from
sibling projects (`token_first_transformer`, `indicator_tokenizer`).

## Layout

```
moe_trading_agent/
├── configs/      # default.yaml (model + training hyperparameters)
├── dataset/      # MoEDataset: parquet loading, tokenization, 3-class labeling
├── models/       # expert, router, moe_layer, moe_model
├── training/     # Trainer (loss, class weights, early stopping, checkpoints)
├── scripts/      # train.py, evaluate.py
└── tests/        # 19 unit + integration tests
```

## Quickstart

```bash
# install (uv) ...
uv sync --extra dev
# ... or venv + pip
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# run tests
pytest -q

# train (expects .parquet candle files in ./data; override with a 2nd arg)
python scripts/train.py configs/default.yaml [data_dir]
python scripts/evaluate.py   # after a checkpoint lands in ./checkpoints
```

Device is auto-selected (MPS > CUDA > CPU).

## Status

Code complete, **19 tests pass**, **not trained** — no published checkpoint or
performance numbers yet.

Part of the [marketglot](../README.md) monorepo.
