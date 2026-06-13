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

**Status:** Design stage — see [`diffusion_orderbook/SPEC.md`](diffusion_orderbook/SPEC.md). Order book data available via warehouse API (see `docs/data_sources.md`); reuses the `orderbook_encoder` data pipeline.

Two tracks (see [`docs/research/diffusion-llms.md`](docs/research/diffusion-llms.md)):

- **Track A (primary):** continuous DDPM/DDIM over order-book state vectors or a windowed "LOB-image"; denoiser conditioned on candle context (from `token_first_transformer`) via FiLM/cross-attention. Forecasting via **inpainting** the future window → mid path → 3 classes; plus a linear probe and a "microstructure surprise" score for repo-comparable evaluation.
- **Track B (v2):** discrete/masked token diffusion (LLaDA/MaskGIT style) over the market-token language for fast parallel generation — the Gemini-Diffusion direction.

**Dependencies:** projects 1 (candle context) and 4 (order book data pipeline).

---

## Project 8: `transformer_diffusion_fusion` — Combined Architecture

**Status:** Design stage — see [`transformer_diffusion_fusion/SPEC.md`](transformer_diffusion_fusion/SPEC.md). Most complex project.

- Transformer encoder: candle (+ optional indicator/footprint) tokens → context vector `c`
- Diffusion decoder (from project 7): conditioned on `c`, yields distribution features — denoiser hidden state, statistics of K sampled futures, and/or conditional "surprise" at the observed book
- Decision head: MLP over `[c ⊕ distribution-features ⊕ current-book-embedding]` → 3 logits
- Training: two-stage (recommended) or joint multi-task (`CE + λ·diffusion-MSE`)

**Dependencies:** projects 1, 4, 7 (+ optional 5, 9).

---

## Project 9: `footprint_encoder` — Footprint / Cluster-Chart Modality

**Status:** Design stage — see [`footprint_encoder/SPEC.md`](footprint_encoder/SPEC.md). A new 4th market modality.

Encodes the **footprint** (volume-at-price per bar, split by aggressive BUY vs SELL) into a per-bar embedding → transformer over bars → 3-class head.

- **Side attribution (pluggable):** trades carry no aggressor side and are sparse, so the primary source is **order-flow inferred from L2 deltas** (ask-side consumption ≈ aggressive buys, bid-side ≈ sells — an approximation); quote-rule (trades vs reconstructed best bid/ask) and tick-rule as cross-checks; synthetic for notebooks. See `docs/data_sources.md`.
- **Per-bar features:** buy/sell/delta per price bin, cumulative delta, POC, imbalance stacks, value area, unfinished-auction flags.
- **Two representations:** footprint-as-image (CNN/MLP) and tokenized "footprint language" (feeds late-fusion / multimodal / MoE and the discrete-diffusion track).

**Dependencies:** project 4 (book reconstruction for side attribution); feeds projects 3, 5, 6, 8.

---

## Baselines

### `kronos_baseline` — external Kronos → 3-class signal

**Status:** Implemented & run locally. Wrapper (not one of the nine architectures) around the third-party MIT model **Kronos** (candles-as-language foundation model). Loads Kronos once, forecasts `horizon` candles with Monte-Carlo sampling, and reduces to a marketglot 3-class signal (DOWN/FLAT/UP, `[DOWN=0, FLAT=1, UP=2]`) + class probabilities.

- Kronos is **not vendored** — clone it and set `KRONOS_PATH` (auto-detects sibling `../Kronos`).
- CLI runs on our klines; library API is `KronosSignal.predict_signal(...) -> SignalResult`.
- Background & analysis: [`docs/research/kronos.md`](docs/research/kronos.md).

---

## Dependency Graph

```
token_first_transformer ──┐
                          ├── multimodal_encoder ── moe_trading_agent
indicator_tokenizer ──────┤
                          ├── late_fusion_agent
footprint_encoder ────────┘   (new 4th modality)

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
| 7 | diffusion_orderbook | — | Design stage ([SPEC](diffusion_orderbook/SPEC.md)) |
| 8 | transformer_diffusion_fusion | — | Design stage ([SPEC](transformer_diffusion_fusion/SPEC.md)) |
| 9 | footprint_encoder | — | Design stage ([SPEC](footprint_encoder/SPEC.md)) |

**Research:** diffusion language models & diffusion for markets — see [`docs/research/diffusion-llms.md`](docs/research/diffusion-llms.md); analysis of **Kronos** (candles-as-language foundation model) and what to adopt — see [`docs/research/kronos.md`](docs/research/kronos.md).

**Data sources:** OHLCV klines locally in `w_trender/backtests/data/`; L2 order book via the prod warehouse API / anonymous S3 (collector live since 2026-06-01) or CryptoHFTData for deep history — see `docs/data_sources.md`.

---

## Shared Conventions

- **Data path:** configured per-project in YAML, points to `w_trender/backtests/data/`
- **Device:** MPS (Metal Performance Shaders), float32
- **Framework:** PyTorch >= 2.1
- **Format:** parquet files, monthly `YYYY-MM.parquet`
- **Python:** >= 3.11
- **Git:** single consolidated monorepo (see [README.md](README.md)); most projects are code-complete and unit-tested but not meaningfully trained yet
