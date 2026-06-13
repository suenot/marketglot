# kronos_baseline

Reusable wrapper that turns the external **Kronos** foundation model
([candles-as-language](../docs/research/kronos.md)) into a marketglot-style
**3-class signal** (DOWN/FLAT/UP) on our own klines.

It loads Kronos once, forecasts `horizon` candles with Monte-Carlo sampling, and
reduces the result to a label + class probabilities using marketglot conventions
(class order `[DOWN=0, FLAT=1, UP=2]`, forward-return deadband `±threshold%`).

> This is a **baseline** built on a third-party MIT model — not one of the nine
> marketglot architectures. Kronos is **not vendored**.

## Setup

Kronos ships as a repo (not a PyPI package), so clone it and point `KRONOS_PATH`
at it (the wrapper also auto-detects a sibling `../Kronos`):

```bash
git clone https://github.com/shiyu-coder/Kronos
export KRONOS_PATH=/abs/path/to/Kronos

# deps (torch, pandas, pyarrow, einops, huggingface_hub, safetensors, tqdm)
uv venv && uv pip install -e ".[dev]"
```

You can also just reuse the Kronos checkout's own venv — it already has everything.

## Use

CLI on our klines (auto-downloads Kronos-small + tokenizer from HF on first run):

```bash
KRONOS_PATH=../Kronos python kronos_baseline/scripts/run.py \
    --symbol BTCUSDT --start 20000 --lookback 400 --horizon 60 --n-paths 8
```

Library:

```python
from kronos_baseline.kronos_signal import KronosSignal

sig = KronosSignal()                                  # Kronos-small + tokenizer-base
res = sig.predict_signal(df, x_ts, y_ts, horizon=60, threshold_pct=0.15, n_paths=10)
print(res.label_name, res.class_probs, res.pred_return)
```

`predict_signal` returns a `SignalResult` (label, point forecast, horizon return,
MC class probabilities, mean predicted path).

## Evaluation

`scripts/evaluate.py` benchmarks the signal on many (symbol, window) pairs
(batched via `predict_batch`) and reports 3-class / directional accuracy and
forecast MAE vs naive baselines:

```bash
KRONOS_PATH=../Kronos python kronos_baseline/scripts/evaluate.py \
    --n-symbols 12 --windows-per-symbol 40 --sample-count 5 --batch-size 16 \
    --out kronos_baseline/artifacts/eval.json
```

Zero-shot result on our crypto 1m data (480 windows × 12 symbols, horizon 60,
±0.15%): only a marginal edge — directional ≈0.58, 3-class 0.415 vs 0.379
baseline, MAE worse than persistence. Details & interpretation in
[`docs/research/kronos.md`](../docs/research/kronos.md#evaluation-on-our-data-zero-shot-2026-06-13).

## Tests

Network-free unit tests cover the labeling logic and path resolution:

```bash
pytest -q          # tests/test_classify.py
```

(The forecast itself needs the Kronos checkout + weights and is exercised via the
CLI, not in CI.)

Part of the [marketglot](../README.md) monorepo. Background: [`docs/research/kronos.md`](../docs/research/kronos.md).
