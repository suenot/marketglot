# Kronos finetune on Apple GPU (MPS)

Finetune [Kronos](../../docs/research/kronos.md) on our klines **using the Mac GPU
(MPS)**. Kronos's own `finetune_csv` pipeline only selects **CUDA-or-CPU**, so on
a Mac it silently runs on CPU — pointless when an Apple GPU is available. This
folder adds a thin launcher that reuses Kronos's `SequentialTrainer` unchanged but
**forces `device = mps`** (its train loops take `device` as an argument and contain
no CUDA-only ops — no AMP/GradScaler/`.cuda()`, and DDP stays off for a single
process).

## Steps

```bash
# 0) clone Kronos next to the repo (or set KRONOS_PATH), deps already in its venv
#    + pyyaml (used by the config loader): uv pip install pyyaml

# 1) export our klines to Kronos CSV format (timestamps,open,high,low,close,volume,amount)
python kronos_baseline/finetune/prepare_csv.py --symbol BTCUSDT --days 20 \
    --out kronos_baseline/finetune/data/BTCUSDT_1m_20d.csv

# 2) finetune the predictor on MPS (tokenizer kept pretrained; --finetune-tokenizer to also train it)
KRONOS_PATH=../Kronos python kronos_baseline/finetune/run_finetune_mps.py \
    --csv kronos_baseline/finetune/data/BTCUSDT_1m_20d.csv \
    --out-dir kronos_baseline/finetune/runs/btc \
    --epochs 1 --batch-size 16 --lookback 256 --predict 60
```

The finetuned predictor lands in `runs/<...>/mps_finetune/basemodel/best_model/`;
load it with `Kronos.from_pretrained(<that dir>)` and plug into `KronosSignal`.

## Incremental training (~30-min sessions)

The upstream loop only checkpoints **at the end of an epoch** (validation + save run
after the full pass), so killing a 3-hour epoch mid-way loses everything. To train
in short sessions, run **short epochs and let each finish**, with **auto-resume**:
each run continues from the `best_model` already in `--out-dir` (unless `--no-resume`).

Rule of thumb on MPS: ~0.5 min of training per day of 1m data, so ~55 days ≈ one
~30-min epoch.

```bash
# one-time: a ~55-day slice (~30-min epoch)
python kronos_baseline/finetune/prepare_csv.py --symbol BTCUSDT --days 55 \
    --out kronos_baseline/finetune/data/BTCUSDT_1m_55d.csv

# run this whenever you have ~30 min; it auto-resumes from the last checkpoint
KRONOS_PATH=../Kronos python kronos_baseline/finetune/run_finetune_mps.py \
    --csv kronos_baseline/finetune/data/BTCUSDT_1m_55d.csv \
    --out-dir kronos_baseline/finetune/runs/btc_incr \
    --epochs 1 --batch-size 32 --lookback 256 --predict 60 --lr 2e-5
```

Each session finishes on its own (no manual kill needed) and saves; the next picks
up where it left off. `--no-resume` starts fresh from pretrained Kronos.

## Notes

- **Device**: auto MPS → CUDA → CPU; override with `--device`. `PYTORCH_ENABLE_MPS_FALLBACK=1`
  is set so any op MPS lacks falls back to CPU instead of crashing.
- **Predictor-only by default**: the tokenizer's BSQ entropy uses `scatter_reduce`/
  `multinomial`, which are riskier on MPS, so tokenizer finetune is opt-in
  (`--finetune-tokenizer`). The pretrained tokenizer is reused from the HF cache.
- **Weights** are resolved from the local HF cache (no re-download); the launcher
  generates the YAML config itself.
- **`amount`** isn't in our klines, so it's written as 0 (Kronos tolerates this).
- `data/` and `runs/` are gitignored (CSVs, checkpoints, logs stay local).

## Verified

Smoke run on BTCUSDT 1m (20 days, Kronos-small, batch 16, lookback 256, horizon 60)
ran on **`Device: mps`** at **~2.5 steps/s** with the loss decreasing
(2.39 → ~2.25 over the first ~120 steps) — i.e. training really uses the Apple GPU.

Throughput on MPS ≈ **42 samples/s** (Kronos-small): batch 16 ≈ 2.5 steps/s,
batch 32 ≈ 1.3 steps/s. So a 12-month BTCUSDT finetune (~525k bars) is ~3 h/epoch.
A full run (12 months, 1 epoch, batch 32) example:

```bash
python kronos_baseline/finetune/prepare_csv.py --symbol BTCUSDT --days 365 \
    --out kronos_baseline/finetune/data/BTCUSDT_1m_12mo.csv
KRONOS_PATH=../Kronos python kronos_baseline/finetune/run_finetune_mps.py \
    --csv kronos_baseline/finetune/data/BTCUSDT_1m_12mo.csv \
    --out-dir kronos_baseline/finetune/runs/btc_full \
    --epochs 1 --batch-size 32 --lookback 256 --predict 60 --lr 2e-5
```

Part of the [marketglot](../../README.md) monorepo. Background: [`docs/research/kronos.md`](../../docs/research/kronos.md).
