# indicator_tokenizer

Compute classic technical indicators from OHLCV and bucketize each one into its own discrete vocabulary of "indicator tokens" for downstream sequence models.

## Idea

Continuous indicator values are hard to feed directly into token-based models. This project maps each indicator into a small set of discrete bins, so a bar becomes a tuple of integer tokens (one per indicator). Two bucketization strategies are used:

- **Fixed thresholds** for indicators with well-known semantic levels (RSI overbought/oversold, Bollinger %B 0/1 bands).
- **Quantile cut points** fitted on training data for indicators without natural levels (MACD histogram, ATR, volume ratio, price-vs-SMA).

Each indicator keeps a reserved `PAD` (0) and `SPECIAL` (1) id via an offset of 2, so vocab size = number of bins + 2.

## Indicators & vocab sizes

| Indicator        | Strategy | Bins | Vocab size | Boundary file                 |
| ---------------- | -------- | ---- | ---------- | ----------------------------- |
| RSI              | fixed    | 5    | 7          | `boundaries/rsi.npy`          |
| MACD histogram   | quantile | 7    | 9          | `boundaries/macd_hist.npy`    |
| Bollinger %B     | fixed    | 5    | 7          | `boundaries/bollinger_pctb.npy` |
| ATR              | quantile | 6    | 8          | `boundaries/atr.npy`          |
| Volume ratio     | quantile | 5    | 7          | `boundaries/volume_ratio.npy` |
| Price vs SMA     | quantile | 5    | 7          | `boundaries/price_vs_sma.npy` |

Vocab size includes the 2 reserved ids (`PAD`, `SPECIAL`).

## Layout

```
indicators/
  computer.py     # IndicatorComputer: RSI, MACD hist, BB %B, ATR, volume ratio, price vs SMA from OHLCV
  tokenizer.py    # FixedBoundaries / QuantileBoundaries / IndicatorTokenizer (fit, encode, save, load)
scripts/
  fit.py                 # fit quantile boundaries on training data, save boundaries/*.npy
  inspect_indicators.py  # print per-indicator min/max/mean/std for sanity checks
configs/
  default.yaml    # symbol, data dir, train month range, per-indicator params
boundaries/       # committed fitted artifacts (one .npy per indicator)
tests/            # unit tests for computer and tokenizer
```

## Boundaries are committed

`boundaries/*.npy` hold the fitted quantile cut points (and fixed thresholds) for each indicator. They are the only data files intentionally kept in git — raw `*.parquet` / `*.csv` klines are gitignored. This lets the tokenizer be loaded and applied without re-running the fit.

## Quickstart

Install (uv):

```bash
uv venv && uv pip install -e ".[dev]"
```

Or venv + pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Fit quantile boundaries on training klines, then load and encode in code:

```bash
python scripts/fit.py --config configs/default.yaml   # writes boundaries/*.npy
```

```python
from pathlib import Path
from indicators.computer import IndicatorComputer
from indicators.tokenizer import IndicatorTokenizer

indicators = IndicatorComputer().compute_all(ohlcv)   # ohlcv: dict of open/high/low/close/volume arrays
tok = IndicatorTokenizer()
tok.load(Path("boundaries"))                          # load fitted boundaries
tokens = tok.encode(indicators)                       # dict[str, np.ndarray] of int tokens
```

`scripts/fit.py` expects 1-minute klines parquet files under `<data_dir>/<symbol>/klines_1m/` (see `configs/default.yaml`); paths point at a sibling `w_trender` data directory and may need adjusting for your setup.

## Status

Code complete, 15 tests pass, boundaries fitted on ~29 months of data (~1.26M values per indicator).

Part of the [marketglot](../README.md) monorepo.
