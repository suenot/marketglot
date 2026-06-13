# transformer_diffusion_fusion

Project 8 of marketglot — transformer **context** + conditional **diffusion**
order-book decoder + a **decision head** (UP / FLAT / DOWN). The most complex
project; it composes existing pieces rather than reinventing them.

## Idea

A `ContextEncoder` turns candle tokens (optionally + indicators + footprint) into
a context vector `c`. The conditional diffusion model from project 7
`diffusion_orderbook` yields **distribution features** — a denoiser hidden state,
statistics of K sampled futures, and/or the conditional "surprise" at the current
book. A small MLP over `[c ⊕ distribution_features ⊕ current_book_embedding]`
emits 3 logits. This is **Track C (fusion)** of the diffusion research note.

```
 candles ─▶ ContextEncoder ──────────────┐  c
 (+indic.,                                ├─▶ [ c ⊕ df ⊕ book_emb ] ─▶ MLP ─▶ 3 logits
  footprint)   └─▶ candle conditioning ─┐ │                                 (DOWN/FLAT/UP)
                                        ▼ │  df
 order book ─▶ diffusion decoder (proj 7) ┘
            └▶ OrderbookEncoder.encode ──── book_emb
```

Status: design stage — see [SPEC.md](SPEC.md).

Part of the [marketglot](../README.md) monorepo.
