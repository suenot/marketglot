# Multimodal Encoder Implementation Plan

> Subagent-driven execution.

**Goal:** Single model that processes candle tokens AND indicator tokens through separate encoders, then fuses them via cross-attention or late concat for 3-class prediction.

**Architecture:**
- Candle encoder: PriceTransformer layers (shared weights with project 1)
- Indicator encoder: small transformer on 6 indicator streams (shared with project 3's Model B)
- Fusion: concatenate encoder outputs -> projection -> transformer layers -> CLS pooling -> MLP head
- End-to-end training (unlike late_fusion which trains separately)

**Key difference from late_fusion_agent:** Single model trained end-to-end, not 3 separate models. Encoders can learn to cooperate.

**Tech Stack:** PyTorch, NumPy, PyArrow, scikit-learn, PyYAML

---

## File Structure

```
w_training/multimodal_encoder/
├── pyproject.toml
├── configs/default.yaml
├── models/
│   ├── __init__.py
│   └── multimodal_model.py
├── dataset/
│   ├── __init__.py
│   └── multimodal_dataset.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── checkpoints/.gitkeep
├── logs/.gitkeep
└── tests/
    ├── __init__.py
    ├── test_model.py
    ├── test_dataset.py
    └── test_integration.py
```

---

### Task 1: All-in-one build

Single subagent creates everything: scaffold, model, dataset, trainer, scripts, tests.

**MultimodalModel architecture:**
```
Candle stream: delta_emb(122,64) + vol_emb(10,16) + vb_emb(10,16) -> concat -> proj -> 128-dim
Indicator stream: 6x emb(vocab,16) -> concat -> proj -> 128-dim
Fuse: concat [candle_repr, indicator_repr] -> 256-dim
Positional embeddings -> 4-layer transformer -> CLS pooling -> MLP -> 3 logits
```

The model uses separate embedding+projection for each stream, then concatenates for shared transformer layers. This is simpler than cross-attention but effective.

**MultimodalDataset:** Same as FusionDataset from late_fusion_agent — loads klines, tokenizes candles AND indicators, returns both sets of tokens + label.

**Trainer:** Standard training loop with early stopping. Single model, single loss.

**Scripts:** train.py (loads config, trains), evaluate.py (loads checkpoint, evaluates on test set).
