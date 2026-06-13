# orderbook_encoder

An L2 order book encoder: a deep MLP that compresses limit-order-book microstructure into a 64-dim embedding, trained end-to-end to predict short-horizon mid-price movement.

## Idea

Candlestick data (OHLCV) throws away almost everything about the limit order book — the queue sizes, the shape of the bid/ask ladder, the imbalance between sides. This project learns directly from the raw L2 book: it reconstructs the book from exchange snapshot + delta streams, turns each tick into a fixed-size feature vector, and trains an MLP encoder whose embedding is supervised by a 3-class price-movement target. The embedding is the real artifact; the classifier head is just a training signal.

## Data pipeline

The full flow goes from the production warehouse straight to training-ready `.npz` files:

```
warehouse (anonymous S3 + REST) -> download_data.py -> raw parquet
        -> book reconstruction (book/) -> 1 Hz sampling, top-20 levels/side
        -> build_samples.py -> data/samples/.../{date}.npz
```

- **Source.** The warehouse market-data service at `warehouse.marketmaker.cc` — a REST API (no auth) for metadata and an anonymous S3 store for files. Hourly `{HH}_snapshot.parquet.zst` and `{HH}_delta.parquet.zst` (zstd parquet, read directly with pandas).
- **Reconstruction.** For each hour: apply the snapshot (resync), then replay deltas in `(event_time, final_update_id)` order; `qty=0` removes a level. Collector-reconnect rotations (`.1`, `.2`, …) are concatenated and re-sorted.
- **Sampling.** Walk the `event_time` clock on a 1 Hz grid; at each tick emit the current top-20 levels per side. Invalid books (empty side / crossed) are skipped.
- **Data is not committed.** Nothing under `data/` is checked in — it is fully re-downloadable from the warehouse via the scripts below.

## Input representation

Each tick becomes an **80-dim `float32`** vector (`4 × depth`, depth = 20), laid out as four contiguous blocks:

```
[ bid_off ×20 | bid_qty ×20 | ask_off ×20 | ask_qty ×20 ]
```

where `off = |price − mid| / mid` and `qty = log1p(qty)`. When fewer than 20 levels exist on a side, `off` repeats the deepest available level and `qty = 0` (padding).

## Architecture

Defined in `models/orderbook_mlp.py`, dims from `configs/default.yaml`:

- **`OrderbookEncoder`** — deep MLP, `80 → 256 → 128 → 64`. Each hidden block is `Linear → LayerNorm → GELU → Dropout(0.1)`, with a final linear projection to the **64-dim embedding**.
- **`OrderbookClassifier`** — wraps the encoder with a linear head to 3 logits. Exposes `encode()` to get the embedding without the head.

## Objective

Predict the mid-price move over a **60 s horizon**, as 3 classes with a ±0.05% band:

- `ret = mid[i+h] / mid[i] − 1`, where `h = round(horizon_sec / interval_sec)`
- `DOWN=0` if `ret < −0.05%`, `UP=2` if `ret > +0.05%`, else `FLAT=1`

Training uses AdamW + class-weighted cross-entropy (weights inverse to train frequencies) and early stopping on validation loss. Windows never cross an `.npz` boundary, and samples spanning a data gap are dropped.

## Layout

```
orderbook_encoder/
  warehouse/client.py        # stdlib HTTP client: metadata + atomic day download
  book/book.py               # LocalBook: snapshot/delta replay, top_levels, mid
  book/sampler.py            # features_from_book + sample_day (1 Hz feature extraction)
  dataset/orderbook_dataset.py   # OrderbookDataset + build_splits (labelling)
  models/orderbook_mlp.py    # OrderbookEncoder + OrderbookClassifier
  training/trainer.py        # train loop, metrics, artifact writing
  scripts/                   # download_data.py, build_samples.py, train.py
  configs/                   # default.yaml, smoke.yaml
  tests/                     # 46 tests, no network, synthetic data
```

## Quickstart

Install (with [uv](https://github.com/astral-sh/uv), or a plain venv):

```bash
uv sync                       # or: python -m venv .venv && pip install -e ".[dev]"
pytest -q                     # 46 tests, offline, synthetic data
```

End-to-end run (the smoke config trains on a single day):

```bash
python scripts/download_data.py --config configs/smoke.yaml --dates 2026-06-09
python scripts/build_samples.py --config configs/smoke.yaml --dates 2026-06-09
python scripts/train.py        --config configs/smoke.yaml
```

`--dates` accepts a single `YYYY-MM-DD` or an inclusive `START:END` range. `build_samples.py` skips existing `.npz` unless `--force`; `train.py` accepts `--epochs N` to override the config. Artifacts (`best.pt`, `config.json`, `test_metrics.json`) land in `artifacts/run_<timestamp>/`.

## Status

Code complete; **46 tests pass**. Smoke-trained on **1 real day** (XRPUSDT / bybit) to validate the full pipeline. This is a working scaffold, **not a validated trading model** — no performance claims beyond "it trains end-to-end".

See [SPEC.md](SPEC.md) for the full design (module contracts, data schemas, reconstruction details). A runnable Kaggle notebook lives at [`../kaggle_notebooks/orderbook_encoder/`](../kaggle_notebooks/orderbook_encoder/).

Part of the [marketglot](../README.md) monorepo.
