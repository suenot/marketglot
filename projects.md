# marketglot — Neural Trading Projects

**Документация (Colab, структура репо, порядок экспериментов):** [docs/README.md](docs/README.md)

Workspace for training neural network models on market data.
Data source: `w_trender/backtests/data/` (43 GB, 262 symbols).

Each project lives in its own subdirectory: `w_training/<project_name>/`.

---

## Project 1: `token_first_transformer` — MVP

**Status:** Code complete. 36 tests pass. Not trained.

Token-based transformer classifier. Discretizes price deltas into tokens — "market language" — trains small Transformer on BTCUSDT 1m candles for 3-class prediction (UP/FLAT/DOWN).

- Input: 3 token streams (price delta, volatility bucket, volume bucket), 128 candles context
- Model: 4-layer transformer, 8 heads, dim=256, ~5-7M params
- Target: 60-candle horizon, thresholds ±0.15%
- Train: AdamW, CosineAnnealing, 5-10 epochs, MPS float32
- Backtest: sequential, SL/TP/max-hold, 0.04% commission

**Spec:** `docs/superpowers/specs/2026-04-21-token-transformer-trading-agent-design.md`

---

## Project 2: `indicator_tokenizer` — Indicator Encoding

**Status:** Code complete. 15 tests pass. Boundaries fitted on 29 months of data.

Bucketize technical indicators (RSI, MACD, Bollinger %B, ATR, Volume Ratio, Price vs SMA) into discrete tokens. Each indicator gets its own vocabulary.

- Vocab sizes: RSI=7, MACD=9, BB=7, ATR=8, VR=7, PVS=7
- Boundaries saved to `indicator_tokenizer/boundaries/*.npy` (fitted on 1.26M values per indicator)

**Output:** `indicator_tokenizer/` module with fitted quantile boundaries saved to disk.

---

## Project 3: `late_fusion_agent` — Separate Models + Meta-Model

**Status:** Code complete. 13 tests pass. Not trained.

Each data source trains its own independent model. A lightweight meta-model combines their predictions.

- Model A: PriceTransformer (candle tokens) → 3 logits
- Model B: IndicatorModel (6 indicator streams → small transformer) → 3 logits
- Meta-model: MLP on 6 concatenated logits → 3 classes
- Training: train A and B independently → collect val logits → train meta-model

**Dependencies:** projects 1 and 2 (complete).

---

## Project 4: `orderbook_encoder` — Order Book MLP Embeddings

**Status:** Code complete. 46 tests pass. Smoke-trained on 1 real day (XRPUSDT/bybit).

L2 order book data is served by the prod warehouse API (`warehouse.marketmaker.cc`, anonymous S3) — see `docs/data_sources.md`. Collector live since 2026-06-01, history grows daily; deep history available from CryptoHFTData (cryptohftdata.com).

- Pipeline: `scripts/download_data.py` (S3 hourly snapshot+delta parquet) → book reconstruction (`book/`) → 1 Hz sampling, top-20 levels per side → npz
- Input: 80-dim vector per tick `[bid_off×20, bid_qty×20, ask_off×20, ask_qty×20]`
- Architecture: deep MLP → 64-dim embedding (`OrderbookEncoder`) + linear head (`OrderbookClassifier`)
- Training objective: predict mid-price move over 60 s horizon (DOWN/FLAT/UP, ±0.05%)
- Spec: `orderbook_encoder/SPEC.md`

---

## Project 5: `multimodal_encoder` — Multi-Encoder Architecture

**Status:** Code complete. 8 tests pass. Not trained.

Single end-to-end model that processes candle tokens AND indicator tokens through separate encoders, then fuses via concatenation through shared transformer layers.

- Candle encoder: PriceTransformer embedding pattern (delta_emb + vol_emb + vb_emb → proj → 128-dim)
- Indicator encoder: 6 indicator embeddings → concat → proj → 128-dim
- Fusion: concat → 256-dim → 4-layer transformer → CLS pooling → MLP → 3 logits
- End-to-end training (unlike late_fusion which trains separately)

**Dependencies:** projects 1 and 2 (complete).

---

## Project 6: `moe_trading_agent` — Mixture of Experts

**Status:** Code complete. 19 tests pass. Not trained.

Full MoE architecture where different "experts" specialize in different market regimes. Sparse activation (top-K routing).

- Router network: top-K gating with auxiliary load-balancing loss
- 8 experts per MoE layer, top_k=2, 4 MoE transformer layers
- Same input pipeline as multimodal_encoder (candle + indicator tokens)
- Loss = CrossEntropy + 0.01 * aux_loss
- ~10-15M params total, sparse activation

**Dependencies:** projects 1 and 4 (complete).

---

## Project 7: `diffusion_orderbook` — Diffusion Model for Microstructure

**Status:** Not started. Data blocker lifted — order book data available via warehouse API (see `docs/data_sources.md`); reuse the `orderbook_encoder` data pipeline.

- Forward process: add Gaussian noise to order book state
- Reverse process: learn to denoise, recovering distribution
- Conditioning: on current price action context (from candle encoder)

**Dependencies:** order book data (blocked).

---

## Project 8: `transformer_diffusion_fusion` — Combined Architecture

**Status:** Not started — depends on projects 4 (orderbook_encoder, complete) and 7 (diffusion_orderbook, not started).

Most complex project.

- Transformer encoder: candle token sequences → context vector
- Diffusion decoder: conditioned on context, generates order book distribution
- Decision head: uses both context and distribution features for prediction

**Dependencies:** projects 1, 4, 8 (blocked chain).

---

## Dependency Graph

```
token_first_transformer ──┐
                          ├── multimodal_encoder ── moe_trading_agent
indicator_tokenizer ──────┘
                          ├── late_fusion_agent

orderbook_encoder ──────── diffusion_orderbook ── transformer_diffusion_fusion
                                                        │
token_first_transformer ────────────────────────────────┘
```

## Build Status

| # | Project | Tests | Status |
|---|---------|-------|--------|
| 1 | token_first_transformer | 36 | Code complete |
| 2 | indicator_tokenizer | 15 | Code complete, boundaries fitted |
| 3 | late_fusion_agent | 13 | Code complete |
| 4 | orderbook_encoder | 46 | Code complete, smoke-trained on real data |
| 5 | multimodal_encoder | 8 | Code complete |
| 6 | moe_trading_agent | 19 | Code complete |
| 7 | diffusion_orderbook | — | Not started (data available) |
| 8 | transformer_diffusion_fusion | — | Not started (depends on 4, 7) |

**Data sources:** OHLCV klines locally in `w_trender/backtests/data/`; L2 order book via the prod warehouse API / anonymous S3 (collector live since 2026-06-01) or CryptoHFTData for deep history — see `docs/data_sources.md`.

---

## Shared Conventions

- **Data path:** configured per-project in YAML, points to `w_trender/backtests/data/`
- **Device:** MPS (Metal Performance Shaders), float32
- **Framework:** PyTorch >= 2.1
- **Format:** parquet files, monthly `YYYY-MM.parquet`
- **Python:** >= 3.11
- **Git:** single consolidated monorepo (see [README.md](README.md)); most projects are code-complete and unit-tested but not meaningfully trained yet
