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

For a paired hourly evaluation of a locally served Kev teacher against the
Transformer and simple baselines, use the same checkpoint and split data:

```bash
python scripts/evaluate_teacher.py --config configs/default.yaml \
  --data-dir data/teacher_eval --checkpoint checkpoints/teacher_eval/best.pt \
  --teacher-id 'kev-0.8b:adapter=9a45d25eb2ab761841196625383fa1dff0e56c1e:base=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68:code=3d9973b80b34d187f9fc8ce81de940b5767eb624' \
  --split val --endpoint http://127.0.0.1:8008/v1/systemone
```

Run `--split test` only after the validation protocol is fixed. Supply the
exact served checkpoint and revision in `--teacher-id`; it separates caches,
but cannot attest third-party server weights. Check the pinned local snapshot
and the server's `/health` or `/v1/models` response before resuming a run. The script
selects UTC hourly closes with complete 128-candle history and 60-candle
future, sends compact causal price/volatility/volume features without an
absolute timestamp, and uses the next open for all backtest entries. It writes
a fsynced, resumable JSONL decision cache, experiment manifest, and paired summary under ignored
`runs/teacher_eval/`. `--max-signals N` checks the first N signals and labels
its summary as partial. Each decision stores the full DOWN/FLAT/UP probability
distribution or a FLAT abstention status if the local service fails. Repeated
service failures stop the run and leave the cache for resumption.
Each strategy is reported with 4bp commission per side, then with 5bp
slippage per side, and under a doubled-cost 8bp/10bp stress scenario.

To measure sensitivity to answer order efficiently, add `--all-orders` to
query the three cyclic orders consecutively for each market state. It preserves
three separate fsynced caches with the same experiment IDs as individual runs,
so it can also resume them. Alternatively run the evaluation three times with
`--criteria-order DOWN,FLAT,UP` (the default), `FLAT,UP,DOWN`, and
`UP,DOWN,FLAT`. After all three runs finish, combine them
offline with the same checkpoint, data, teacher ID, split, endpoint and optional
`--max-signals` arguments:

```bash
python scripts/evaluate_teacher.py --config configs/default.yaml \
  --data-dir data/teacher_eval --checkpoint checkpoints/teacher_eval/best.pt \
  --teacher-id 'kev-0.8b:adapter=9a45d25eb2ab761841196625383fa1dff0e56c1e:base=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68:code=3d9973b80b34d187f9fc8ce81de940b5767eb624' \
  --split val --endpoint http://127.0.0.1:8008/v1/systemone \
  --ensemble-from NORMAL_RUN_DIR FLAT_FIRST_RUN_DIR UP_FIRST_RUN_DIR
```

Use the three run directories printed with the individual caches, in the
order shown. The ensemble requires matching manifests, complete successful
probability rows and matching baseline summaries. It averages DOWN/FLAT/UP
probabilities per hourly signal, chooses the largest mean, and reports the
same cost scenarios and baselines without model or service calls. For runs
without service failures, it also reports multiclass Brier score and log loss;
lower is better, and uniform probabilities score 2/3 and log(3) respectively.

Laya can be served separately after installing `laya==0.3.20` and its runtime
in a separate environment. For the pinned local snapshot, start
`python scripts/serve_laya.py --model runs/teacher_eval/model-cache/laya/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --device mps --port 8009`,
then use `--teacher-id convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
and `--endpoint http://127.0.0.1:8009/v1/systemone` with the evaluator.

Kev-4B also fits on a MacBook M2 Max with 32 GB RAM. With the [Kev source](https://github.com/jaredpalmer/kev)
at commit `3d9973b80b34d187f9fc8ce81de940b5767eb624` installed in a separate
environment and its pinned adapter/base downloaded, serve it from this directory:

```bash
HF_HOME="$PWD/runs/teacher_eval/model-cache/kev4b" \
  python -m kev.serve \
  --run jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --port 8010
```

Use `--endpoint http://127.0.0.1:8010/v1/systemone`, a `--teacher-id` that
includes the adapter and Qwen base revisions, and `--all-orders` with the
evaluator. Query `/v1/models` to confirm the served adapter before starting.

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

Code complete; 61 tests pass. A BTCUSDT 1m run on 2026-09-25 used the
chronological splits in `configs/default.yaml`. Its run manifest and durable
checkpoints are on server1 at
`/mnt/third/projects/trading/training/checkpoints/clore-btc1m-20260925T0840Z/`.
The full-history model reached weighted F1 0.4206 on validation and 0.4212 on
the held-out test. The test backtest returned -87.67% with 0.04% commission per
fill, or -99.10% when adding an assumed 0.05% slippage per fill. The current
classifier is not a viable trading strategy; F1 is not a substitute for
cost-adjusted validation. A separate six-month-history run early-stopped at
epoch 5; it reached test weighted F1 0.3661 and returned -93.61% in the
commission-only backtest. See the [experiment audit](../docs/research/btcusdt-1m-2026-09-25.md)
for the assumptions and next steps.
The [open-teacher validation](../docs/research/open-teacher-btc1m-2026-09-26.md)
found no profitable or calibrated teacher among Laya, Kev-0.8B, and Kev-4B.

---

Part of the [marketglot](../README.md) monorepo.
