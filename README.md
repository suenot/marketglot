# marketglot — Polyglot Neural Models for the Market

> A collection of small, self-contained neural-network models that learn to read
> the market through different **modalities** — candlestick token streams,
> technical-indicator tokens, and L2 order-book microstructure — and predict
> short-horizon price direction (**UP / FLAT / DOWN**).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-ee4c2c.svg)](https://pytorch.org/)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)](#-project-status)

---

## What is this?

**marketglot** is a **research monorepo**. Each subdirectory is an independent,
self-contained PyTorch project that explores one architectural idea for turning
raw market data into a directional trading signal. They share conventions (data
format, device, 3-class target) but can be developed, tested, and trained in
isolation.

The guiding idea: treat the market as a **language**. Discretize price deltas,
volatility, volume, and indicators into tokens — a "market vocabulary" — and let
sequence models (transformers, MoE, late/early fusion) learn the grammar. A
parallel track learns directly from the **L2 order book** to capture
microstructure that candles throw away.

> ⚠️ **This is exploratory research code, not financial advice and not a
> production trading system.** Most models are code-complete and unit-tested but
> **not meaningfully trained** — see [Project status](#-project-status).

---

## Architecture at a glance

```
            ┌─────────────────────────┐
 candles ──▶│ token_first_transformer  │──┐
            └─────────────────────────┘  │
                                          ├─▶ multimodal_encoder ─▶ moe_trading_agent
            ┌─────────────────────────┐  │
indicators▶ │   indicator_tokenizer    │──┤
            └─────────────────────────┘  │
                                          └─▶ late_fusion_agent

            ┌─────────────────────────┐
order book▶ │    orderbook_encoder     │──▶ diffusion_orderbook ─▶ transformer_diffusion_fusion
            └─────────────────────────┘                                    ▲
                                          token_first_transformer ─────────┘
```

---

## The projects

| # | Project | Idea | Tests | Status |
|---|---------|------|:-----:|--------|
| 1 | [`token_first_transformer`](token_first_transformer/) | Discretize price deltas / vol / volume into tokens, train a small transformer classifier | 36 | ✅ Code complete |
| 2 | [`indicator_tokenizer`](indicator_tokenizer/) | Quantile-bucketize technical indicators (RSI, MACD, BB %B, ATR, VR, PVS) into per-indicator vocabularies | 15 | ✅ Code complete · boundaries fitted |
| 3 | [`late_fusion_agent`](late_fusion_agent/) | Train one model per modality, fuse their logits with a lightweight meta-model | 13 | ✅ Code complete |
| 4 | [`orderbook_encoder`](orderbook_encoder/) | Reconstruct the L2 book from warehouse snapshots+deltas, encode top-20 levels with a deep MLP → 64-dim embedding | 46 | ✅ Code complete · smoke-trained on real data |
| 5 | [`multimodal_encoder`](multimodal_encoder/) | One end-to-end transformer over candle **and** indicator tokens via separate encoders + fusion | 8 | ✅ Code complete |
| 6 | [`moe_trading_agent`](moe_trading_agent/) | Mixture-of-Experts transformer with sparse top-k routing and load-balancing loss | 19 | ✅ Code complete |
| 7 | `diffusion_orderbook` | Diffusion model over order-book microstructure, conditioned on price-action context | — | 🔲 Not started |
| 8 | `transformer_diffusion_fusion` | Transformer context + diffusion decoder + decision head (most complex) | — | 🔲 Not started |

Full per-project descriptions and the dependency graph live in
[`projects.md`](projects.md).

---

## Repository layout

```
marketglot/                    # (local workspace dir: w_training/)
├── README.md                  ← you are here
├── projects.md                ← detailed catalog of all 8 projects + status table
├── LICENSE                    ← MIT
├── docs/                      ← deeper documentation (RU): data sources, Colab, workflow
│   ├── README.md
│   ├── data_sources.md
│   ├── repository.md
│   ├── colab.md
│   ├── training_workflow.md
│   └── superpowers/           ← historical design specs & plans
├── kaggle_notebooks/          ← Kaggle kernel metadata + generated notebooks
├── tasks/                     ← notebook generators & smoke-test scripts
│
├── token_first_transformer/   ┐
├── indicator_tokenizer/       │
├── late_fusion_agent/         │ one self-contained PyTorch project per directory
├── orderbook_encoder/         │ (code · configs · tests · notebook)
├── multimodal_encoder/        │
└── moe_trading_agent/         ┘
```

Each project directory follows the same shape:

```
<project>/
├── pyproject.toml             ← dependencies (installable with uv / pip)
├── README.md                  ← what it does, how to run
├── configs/                   ← YAML hyperparameters
├── dataset/  models/  training/ (or scripts/)   ← source modules
├── tests/                     ← pytest suite
└── <project>.ipynb            ← self-contained Colab/Kaggle notebook (where present)
```

---

## Data sources

| Modality | Source | Notes |
|----------|--------|-------|
| OHLCV klines (1m) | Local `w_trender/backtests/data/` | `YYYY-MM.parquet` per symbol, 262 symbols (~43 GB). |
| L2 order book | Prod warehouse API — `warehouse.marketmaker.cc` (anonymous S3) | Hourly snapshot + delta parquet; live collector since **2026-06-01**. Deep history via [CryptoHFTData](https://cryptohftdata.com). |

Raw data, downloaded datasets, checkpoints, and training artifacts are **not
committed** (see [`.gitignore`](.gitignore)) — every project re-fetches or
regenerates them through its own scripts. The notebooks fall back to **synthetic
data** so they run out-of-the-box without any external dependency. See
[`docs/data_sources.md`](docs/data_sources.md) for details.

---

## Quickstart

Each project is independent. Pick one, install its dependencies, run its tests:

```bash
cd orderbook_encoder

# with uv (recommended)
uv venv && uv pip install -e ".[dev]"

# …or with stdlib venv + pip
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run the test suite
pytest -q
```

Training is driven by per-project configs and entry points, e.g.:

```bash
python scripts/download_data.py   # fetch real data (orderbook_encoder)
python scripts/build_samples.py   # reconstruct book → npz samples
python scripts/train.py --config configs/smoke.yaml
```

Conventions shared across projects: **PyTorch ≥ 2.1**, **Python ≥ 3.11**,
device **MPS / CUDA / CPU** (auto), float32, parquet inputs.

### Notebooks

Self-contained notebooks (code inlined, synthetic-data fallback) are provided for
the token/indicator/fusion/MoE projects and can be run on **Google Colab** or
**Kaggle** with zero local setup. Rebuild them from source with:

```bash
python3 tasks/_build_all_notebooks.py
```

See [`docs/colab.md`](docs/colab.md) and [`kaggle_notebooks/`](kaggle_notebooks/).

---

## 🧪 Project status

This is **active research**, and honesty matters more than a green badge:

- **Code & tests:** all six implemented projects are code-complete with passing
  unit tests (137 tests total).
- **Training:** mostly **not done**. `orderbook_encoder` has been *smoke-trained*
  on a single real day (XRPUSDT/bybit) and `late_fusion_agent` ran a smoke-test
  on Kaggle (synthetic data, CPU) — neither produced meaningful metrics.
- **Not production:** no live trading, no validated edge, no risk management.

Treat results and architectures here as **experiments to learn from**, not
strategies to deploy.

---

## Documentation

- [`projects.md`](projects.md) — detailed catalog of all eight projects
- [`docs/data_sources.md`](docs/data_sources.md) — where the data comes from
- [`docs/repository.md`](docs/repository.md) — repository map (RU)
- [`docs/training_workflow.md`](docs/training_workflow.md) — recommended experiment order (RU)
- [`docs/colab.md`](docs/colab.md) — running the notebooks (RU)
- [`docs/superpowers/`](docs/superpowers/) — historical design specs & plans

---

## License

[MIT](LICENSE) © 2026 Eugen Soloviov
