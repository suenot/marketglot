# MoE Trading Agent Implementation Plan

> Subagent-driven execution.

**Goal:** Mixture of Experts transformer that routes token streams to specialized expert FFN layers. 3-class UP/FLAT/DOWN on BTCUSDT 1m.

**Architecture:** Base = multimodal encoder pattern (candle + indicator streams fused). Replace monolithic FFN layers with MoE layers: router selects top-K experts per token. Sparse activation, more capacity without proportional compute.

**Key difference from multimodal_encoder:** Same input pipeline, but transformer FFN replaced by MoE FFN. Router learns regime specialization (trending up, trending down, ranging, volatile).

**Tech Stack:** PyTorch, NumPy, PyArrow, scikit-learn, PyYAML

---

## File Structure

```
w_training/moe_trading_agent/
├── pyproject.toml
├── configs/default.yaml
├── models/
│   ├── __init__.py
│   ├── expert.py          # single expert FFN
│   ├── router.py          # top-K gating router
│   ├── moe_layer.py       # MoE layer: router + experts
│   └── moe_model.py       # full model with MoE transformer
├── dataset/
│   ├── __init__.py
│   └── moe_dataset.py     # reuses multimodal dataset pattern
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
    ├── test_expert.py
    ├── test_router.py
    ├── test_moe_layer.py
    ├── test_moe_model.py
    ├── test_dataset.py
    └── test_integration.py
```

---

### Task 1: All-in-one build

Single subagent creates everything: scaffold, models, dataset, trainer, scripts, tests.

**Expert module:**
```
Expert(dim, hidden_dim): Linear(dim, hidden_dim) -> ReLU -> Linear(hidden_dim, dim)
```
Standard FFN, same as transformer FFN layer. Each expert is independent.

**Router module:**
```
Router(dim, num_experts): Linear(dim, num_experts) -> softmax -> top-K selection
Returns: gate_logits, top_k_indices, top_k_weights (normalized)
Also returns auxiliary loss (load balancing): num_experts * sum(p_i^2) where p_i = fraction of tokens routed to expert i
```

**MoELayer:**
```
MoELayer(dim, hidden_dim, num_experts, top_k):
  - experts = ModuleList of Expert modules
  - router = Router
  - forward(x): 
    1. Flatten batch+seq -> (tokens, dim)
    2. Route: get top-K experts per token
    3. Dispatch tokens to experts (sparse)
    4. Weight expert outputs by gate weights
    5. Combine -> reshape back to (batch, seq, dim)
    6. Return output + aux_loss
```

**MoEModel:**
```
Candle stream: delta_emb(122,64) + vol_emb(10,16) + vb_emb(10,16) -> concat -> proj -> 128-dim
Indicator stream: 6x emb(vocab,16) -> concat -> proj -> 128-dim
Fusion: concat [candle_repr, indicator_repr] -> 256-dim
CLS token prepended at position 0
Positional embeddings(seq_len+1, 256)
4 MoE Transformer layers:
  - MultiheadAttention(8 heads, dim=256)
  - MoELayer(dim=256, hidden_dim=1024, num_experts=8, top_k=2)
  - LayerNorm + residual connections
CLS pooling -> MLP(256->128->3) -> 3 logits
Total ~10-15M params (8 experts * 4 layers, but sparse activation)
```

**MoEDataset:** Same as FusionDataset/MultimodalDataset — loads klines, tokenizes candles AND indicators, returns both sets of tokens + label. Reuses sibling tokenizers via sys.path.

**Trainer:** Standard training loop. Loss = CE + aux_loss * lambda (lambda=0.01). Early stopping on val F1.

**Scripts:** train.py (loads config, trains), evaluate.py (loads checkpoint, evaluates on test set).

**Tests:**
- test_expert.py: output shape, gradients flow, finite output
- test_router.py: output shapes, top-k selection valid, aux_loss positive, load balance loss works
- test_moe_layer.py: output shape matches input, aux_loss returned, sparse routing (only top-k experts active)
- test_moe_model.py: output shape (batch, 3), logits finite, gradients flow through entire model
- test_dataset.py: length > 0, item shapes correct, labels valid {0,1,2}
- test_integration.py: mock data -> dataset -> model forward -> loss backward -> shapes ok
