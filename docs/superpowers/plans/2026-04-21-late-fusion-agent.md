# Late Fusion Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Two independent base models (candle transformer + indicator model) feed logits into a meta-model. 3-class UP/FLAT/DOWN on BTCUSDT 1m.

**Architecture:** Model A = PriceTransformer from token_first_transformer. Model B = new IndicatorModel (6 indicator token streams -> small transformer). Meta-model = MLP on 6 logits (3+3).

**Training:** Train A and B independently -> collect val logits -> train meta-model.

**Tech Stack:** PyTorch, NumPy, PyArrow, scikit-learn, PyYAML. Reuses code from token_first_transformer/ and indicator_tokenizer/.

**Plan file:** docs/superpowers/plans/2026-04-21-late-fusion-agent.md

---

## File Structure

```
w_training/late_fusion_agent/
├── pyproject.toml
├── configs/default.yaml
├── models/
│   ├── __init__.py
│   ├── indicator_model.py
│   └── meta_model.py
├── dataset/
│   ├── __init__.py
│   └── fusion_dataset.py
├── training/
│   ├── __init__.py
│   └── fusion_trainer.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── checkpoints/.gitkeep
├── logs/.gitkeep
└── tests/
    ├── __init__.py
    ├── test_indicator_model.py
    ├── test_meta_model.py
    ├── test_fusion_dataset.py
    └── test_integration.py
```

---

### Task 1: Project Scaffold + IndicatorModel + MetaModel

Single subagent creates scaffold, both models, and their tests.

**IndicatorModel:** 6 indicator embedding layers -> concat -> linear projection -> learned CLS token prepended -> 2-layer transformer -> CLS pooling -> MLP head -> 3 logits. Vocab sizes: [7,9,7,8,7,7], emb_dim=16, hidden_dim=128.

**MetaModel:** Linear(6,16) -> ReLU -> Linear(16,3). Takes concatenated logits from A and B.

**Tests:** output shape, single input, gradients, finite output — 4 per model.

---

### Task 2: FusionDataset

Loads klines parquet, computes both candle tokens (via DeltaTokenizer/BucketTokenizer from token_first_transformer) and indicator tokens (via IndicatorComputer/IndicatorTokenizer from indicator_tokenizer). Returns (delta_ids, vol_ids, vb_ids, ind_dict, label) per sample.

Adds sibling projects to sys.path for imports. Fits all tokenizers on provided file_paths.

Tests: length, item shapes (6 indicator streams each seq_len), CLS at pos 0, labels valid.

---

### Task 3: FusionTrainer + Scripts + Integration Test

**FusionTrainer:** orchestrates training of Model A, Model B (separate loops with early stopping), then collects val logits from both, trains MetaModel on logits.

**collate_fn:** for DataLoader — stacks delta/vol/vb tensors, stacks 6 indicator tensors separately, stacks labels.

**scripts/train.py:** loads config, builds datasets, models, trainer, calls train_all().

**scripts/evaluate.py:** loads checkpoints for A+B+meta, runs on test set, prints classification report.

**Integration test:** mock parquet -> FusionDataset -> both models forward -> meta forward -> verify shapes.
