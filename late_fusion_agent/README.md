# late_fusion_agent

Late-fusion trading agent: two independent per-modality models predict price
direction, and a lightweight meta-model combines their logits into a final
3-class decision (DOWN / FLAT / UP).

## Idea

Instead of feeding every input through one big model (early fusion), each data
modality trains its own specialist:

- **Model A** sees tokenized candles.
- **Model B** sees tokenized indicators.

Each emits 3 class logits. A small **meta-model** learns how to weigh and
combine those logits. This keeps the per-modality models simple, lets them be
trained (and debugged) independently, and makes the fusion step cheap.

## Architecture

```
candle tokens ---> Model A (PriceTransformer) --> 3 logits --\
                                                              +--> Meta-MLP --> 3 classes
indicator tokens -> Model B (IndicatorModel)   --> 3 logits --/
```

- **Model A — PriceTransformer** (reused from `token_first_transformer`):
  delta + range-bucket + volume-bucket token streams, `hidden_dim=256`,
  4 layers, 8 heads, FFN 1024, CLS-token pooling -> 3 logits.
- **Model B — IndicatorModel** (`models/indicator_model.py`): 6 indicator
  token streams (`rsi`, `macd_hist`, `bollinger_pctb`, `atr`, `volume_ratio`,
  `price_vs_sma`), per-stream embeddings (`emb_dim=16`) concatenated and
  projected to `hidden_dim=128`, 2-layer / 4-head transformer encoder with a
  CLS token -> 3 logits.
- **Meta-model — MetaModel** (`models/meta_model.py`): MLP over the
  concatenated logits (`input_dim=6` -> `hidden_dim=16` -> 3 classes).

Sequence length is 128; the label is the sign of the forward return over a
60-step horizon with a 0.15% flat threshold (see `configs/default.yaml`).

## Training pipeline

Three stages, driven by `training/fusion_trainer.py`:

1. **Train Model A** independently on candle tokens (cross-entropy, early
   stopping on val loss) -> `model_a_best.pt`.
2. **Train Model B** independently on indicator tokens -> `model_b_best.pt`.
3. **Collect val logits** from the frozen A and B, then **train the
   meta-model** on those logits -> `meta_model.pt`.

Evaluation runs the full A + B + meta chain on the held-out test months and
prints a `classification_report` and confusion matrix.

## Layout

```
late_fusion_agent/
├── configs/        # default.yaml: data split, tokenizer, model & training dims
├── dataset/        # FusionDataset: builds candle + indicator tokens and labels
├── models/         # IndicatorModel (B), MetaModel (meta)  [A comes from sibling]
├── training/       # FusionTrainer: 3-stage train_all()
├── scripts/        # train.py, evaluate.py
├── tests/          # 13 tests (dataset, model B, meta, integration)
└── late_fusion_agent.ipynb
```

Model A's `PriceTransformer` is imported at runtime from the sibling
`token_first_transformer` project; indicator computation/tokenization comes
from `indicator_tokenizer`. Both must sit alongside this project in the
monorepo.

## Quickstart

```bash
# install (uv)
uv venv && uv pip install -e ".[dev]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# run tests
pytest -q

# train full pipeline (A -> B -> meta), then evaluate on test months
python scripts/train.py    --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --checkpoint-dir checkpoints
```

Training/eval read 1-minute klines parquet from `data.data_dir` in the config
(default points at the `w_trender` backtest data); the tests use synthetic
parquet and need no external data.

## Status

Code complete; **13 tests pass**. Only a single Kaggle smoke-test on synthetic
data (CPU) has been run to confirm the pipeline executes end-to-end — the
models have **not** been meaningfully trained, so there are no real metrics yet.

Part of the [marketglot](../README.md) monorepo.
