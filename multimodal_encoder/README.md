# multimodal_encoder

A single end-to-end transformer that jointly encodes candle tokens and indicator tokens, fusing them at the feature level to classify short-term price direction (DOWN / FLAT / UP).

## Idea

This project implements **early (feature) fusion**: candle and indicator token
streams are projected into a shared space, concatenated **per time step**, and
processed by **one** transformer that is trained **end-to-end**.

This contrasts with `late_fusion_agent`, which trains separate models and fuses
only at the **logit level**. Here the two modalities can attend to each other
inside the transformer, and all parameters update from a single loss.

Conceptually it reuses two upstream building blocks:
- `token_first_transformer` — candle delta/bucket tokenizers and the
  `PriceTransformer` embedding pattern.
- `indicator_tokenizer` — indicator computation and per-indicator tokenization.

## Architecture

Dims below are taken from `configs/default.yaml`.

- **Candle encoder**: `delta_emb (64) + vol_emb (16) + vb_emb (16)` -> concat (96)
  -> linear proj -> **128**.
- **Indicator encoder**: 6 indicator embeddings (`emb_dim 16` each) ->
  concat (96) -> linear proj -> **128**. Indicators: RSI, MACD histogram,
  Bollinger %B, ATR, volume ratio, price-vs-SMA.
- **Fusion**: concat candle + indicator (128 + 128 = 256) -> linear -> **256**
  hidden, + CLS token + learned positional embeddings.
- **Transformer**: 4 layers, 8 heads, FFN 1024, GELU, dropout 0.1.
- **Head**: CLS pooling -> LayerNorm -> MLP (256 -> 128 -> 3 logits).

```
candle tokens ─► [delta+vol+vb emb] ─► proj ─► 128 ┐
                                                    ├─ concat(256) ─► fusion proj(256)
indicator tokens ─► [6 emb] ─► proj ─────────► 128 ┘                       │
                                                                           ▼
                                              [CLS] + pos emb ─► Transformer (4L, 8H)
                                                                           │
                                                              CLS pooling ─► MLP ─► 3 logits
```

Sequence length 128; labels from a 60-step horizon with a 0.0015 threshold.

## Layout

```
multimodal_encoder/
├── configs/      # default.yaml — data, tokenizer, model, training
├── dataset/      # MultimodalDataset: load parquet, tokenize, build labels
├── models/       # MultimodalEncoder (candle + indicator + fusion transformer)
├── training/     # Trainer: AdamW, cosine schedule, weighted-F1 early stopping
├── scripts/      # train.py, evaluate.py
└── tests/        # dataset, model, integration tests
```

## Quickstart

Install (using [uv](https://github.com/astral-sh/uv)):

```bash
uv venv && uv pip install -e ".[dev]"
```

Or with plain venv + pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Train and evaluate (expects kline parquet data under the `data_dir` configured
in `configs/default.yaml`, and sibling projects `token_first_transformer` /
`indicator_tokenizer` available on the path):

```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## Status

Code complete, 8 tests pass. **Not trained** — no checkpoints or reported
metrics yet.

Part of the [marketglot](../README.md) monorepo.
