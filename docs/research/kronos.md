# Research: Kronos — and what marketglot can adopt

> **Kronos** (shiyu-coder/Kronos, AAAI 2026, MIT) is the first open-source
> *foundation model* for the "language of financial markets": it tokenizes
> continuous OHLCV candles into **hierarchical discrete tokens** with a learned
> quantizer, then trains a decoder-only autoregressive Transformer to predict the
> next candle — pre-trained on **>12B K-lines from 45 exchanges**.

This is the same thesis as marketglot's `token_first_transformer` ("candles =
language"), executed at foundation-model scale with several ideas worth adopting.

- Paper: arXiv:2508.02739 · Code: <https://github.com/shiyu-coder/Kronos> ·
  Weights: <https://huggingface.co/NeoQuasar>

## ✅ Verified locally (2026-06-13)

Cloned and run on this machine (Apple Silicon, torch 2.12, MPS), outside the
marketglot repo at `../../Kronos`:

- **Regression test passes** — `Kronos-small` + `Kronos-Tokenizer-base` reproduce
  the repo's reference output exactly (max rel-diff ~1e-7, ctx 512 & 256), CPU, ~28 s.
- **Ran on our own data** — BTCUSDT `klines_1m` (`w_trender/.../BTCUSDT/klines_1m`),
  lookback 400 → forecast 60 candles, `sample_count=5` on MPS in ~12 s. Mapped to
  marketglot's objective (60-candle horizon, ±0.15%): on that single window it
  predicted DOWN vs actual UP (close MAE ≈ 436 on ~70k price). One window is a
  smoke test, not an evaluation — but the **pipeline runs on our data end-to-end**.

Reproduce: see [How to run](#how-to-run-locally) below.

---

## Evaluation on our data (zero-shot, 2026-06-13)

Objective benchmark via `kronos_baseline/scripts/evaluate.py` — **480 windows ×
12 symbols** (BTC/ETH/SOL/XRP/BNB/DOGE/ADA/AVAX/LINK/LTC/TRX/DOT), evenly spaced
over 8 months of `klines_1m` (pulled from server1). Kronos-small, lookback 400,
**horizon 60 candles**, ±0.15% deadband, `sample_count=5`. Zero-shot (no finetune).

| Metric | Kronos | Naive baseline | Verdict |
|---|---|---|---|
| 3-class accuracy (DOWN/FLAT/UP) | **0.415** | 0.379 (most-frequent) | lift only **+0.035** |
| Directional acc (non-flat, n=288) | **0.583** | 0.537 (majority UP/DOWN) | lift **+0.046** |
| Forecast MAE, % of price | **0.544** | 0.285 (persistence) | **worse** |

Class balance was DOWN 0.33 / FLAT 0.29 / UP 0.38; per-symbol accuracy ranged
0.30–0.50. The model rarely predicts FLAT correctly (31/141) and leans UP.

**Honest read:** zero-shot Kronos gives only a **marginal edge** on crypto 1m at a
60-minute horizon — directional ~58% (barely above a coin flip) and point-forecast
MAE worse than "price doesn't move" (expected: 60×1m is near-martingale, and Kronos
was pre-trained mostly on non-crypto / coarser bars). Not usable as-is as a signal.
Candidates to improve: **finetune on our symbols**, shorter horizon (5–15 candles),
confidence-gating on the MC class probabilities, or use Kronos embeddings as a
*feature* into our fusion/MoE models rather than a standalone predictor.

---

## How Kronos works (grounded in the code)

Two stages: a **tokenizer** (continuous candle → discrete tokens) and an
**autoregressive LM** (predict next token). Verified hyperparameters of the
released models:

| | d_model | layers | heads | ff | bits (s1/s2) | notes |
|---|---|---|---|---|---|---|
| Tokenizer-base | 256 | 4 enc + 4 dec | 4 | 512 | 10 / 10 | `d_in=6` (OHLC+vol+amount), group_size 4 |
| Kronos-small | 512 | 8 | 8 | 1024 | 10 / 10 | learnable temporal emb |

### 1. BSQ tokenizer (`model/kronos.py`, `model/module.py`)

A Transformer **VQ-VAE** whose quantizer is a **Binary Spherical Quantizer**
(BSQ, arXiv:2406.07548):

1. Encoder Transformer maps the candle vector → `z ∈ R^20`.
2. **Project to the unit sphere**: `z = F.normalize(z)`.
3. **Sign-quantize each dim to ±1** with a straight-through estimator
   (`z + (zhat - z).detach()`). 20 bits ⇒ a code in a space of `2^20 ≈ 1.05M`.
   This is **lookup-free quantization** — the token *index* is computed from the
   bits (`Σ bit·2^k`), there is **no codebook embedding table** to collapse.
4. **Entropy regularization**: per-sample entropy (push codes to be confident) and
   codebook entropy (push global usage up) + a commit loss — keeps the code space
   well-used without a learned codebook.
5. **Hierarchy by bit-split**: the 20 bits split into **s1 (coarse, 10 bits)** and
   **s2 (fine, 10 bits)** → two token ids. The decoder is trained to reconstruct
   the candle from **s1 alone** (`z_pre`) *and* from the full code (`z`), so the
   coarse token already carries a usable approximation.

Net effect: one continuous OHLCV candle → **two discrete tokens** (coarse +
fine), jointly encoding all features (unlike marketglot's *independent*
per-feature quantile buckets).

### 2. Autoregressive LM with hierarchical dual head

- **Vocabulary factorization** (`HierarchicalEmbedding`): instead of one
  `2^20`-row table, two small tables `emb_s1 (1024×d)` + `emb_s2 (1024×d)`,
  concatenated → `fusion_proj`. Two 1024 tables ≪ one 1.05M table.
- **Calendar features** (`TemporalEmbedding`): minute/hour/weekday/day/month
  embeddings added to each token — cheap intraday/weekly seasonality.
- **Decoder-only Transformer**: RMSNorm + **RoPE** + **SwiGLU** FFN, causal SDPA.
- **Hierarchical decoding within each step** (`DualHead` + `DependencyAwareLayer`):
  1. predict **s1** (coarse "scenario": up/down, rough magnitude);
  2. sample s1, embed it, and **cross-attend** (sibling-as-query over hidden
     states) to condition the **s2** head — predict fine detail **given** s1.
  "First the scenario, then the details."

### 3. Inference (`KronosPredictor`, `auto_regressive_inference`)

- **Local-window normalization**: per-series z-score over the lookback window,
  clipped to ±5; de-normalized after — adapts to each instrument's scale.
- **Two-stage sampling** per step: sample s1 (temperature `T`, top-k/top-p), then
  s2 | s1; append to a rolling `max_context` buffer.
- **Monte-Carlo paths**: replicate the context `sample_count` times, generate that
  many stochastic futures, decode to candles, **average** → smoother forecast and
  a usable predictive distribution.

---

## What marketglot can adopt

Prioritized, mapped to specific projects:

| # | Idea from Kronos | Adopt into | Payoff |
|---|---|---|---|
| 1 | **Learned BSQ tokenizer** (joint OHLCV, lookup-free, hierarchical) | `token_first_transformer`, `indicator_tokenizer`, even `footprint`/`orderbook` | Replaces hand-crafted per-feature quantile buckets with a learned joint tokenizer; no codebook collapse |
| 2 | **Pre-trained foundation model + finetune** | new `kronos_adapter` project / baseline | Sidesteps marketglot's data scarcity — most projects are untrained; Kronos is pre-trained on 12B candles, MIT |
| 3 | **Hierarchical coarse→fine dual head** | `token_first_transformer`, `multimodal`, `moe`, diffusion heads | Small factorized vocab + "direction first, magnitude second" matches our UP/FLAT/DOWN then size |
| 4 | **Modern blocks: RoPE + RMSNorm + SwiGLU** | every transformer in the repo | Strictly better than vanilla blocks, ~free |
| 5 | **Calendar/temporal embeddings** | all candle/footprint encoders | Cheap intraday & weekly seasonality we currently ignore |
| 6 | **Local-window normalization + clip** | any continuous-input encoder | Robust per-window scaling (we already do offsets/log1p in orderbook; unify it) |
| 7 | **Generative forecasting + MC sampling** | a generative track; derive 3-class + calibrated probs from sampled paths | Turns point-classification into a distribution; gives quantiles/uncertainty |

### How this ties into the diffusion direction
Kronos is **autoregressive** over its BSQ tokens. The BSQ token stream is exactly
what a **discrete/masked diffusion** decoder (the Gemini-Diffusion direction in
[`diffusion-llms.md`](diffusion-llms.md)) would denoise — so:

> **Kronos BSQ tokenizer + discrete-diffusion decoder = parallel (fast) candle
> forecasting.** Reuse Kronos's learned tokenizer; swap its AR head for a
> masked-diffusion head to forecast a block of future candle tokens in parallel.

This unifies both research threads: adopt Kronos's *tokenizer + hierarchy*, drive
it with a *diffusion* decoder for throughput.

---

## Suggested integration roadmap

1. **Baseline now (no training):** ✅ done — `kronos_baseline/` wraps
   `KronosPredictor` into a marketglot 3-class signal (forecast → UP/FLAT/DOWN with
   our horizon/threshold, MC paths → class probabilities). Run locally on BTCUSDT.
   Zero-shot benchmark on our data done (see [Evaluation](#evaluation-on-our-data-zero-shot-2026-06-13)
   — only a marginal edge). Next: benchmark against `token_first_transformer` once trained.
2. **Borrow components:** add a `marketglot/common/` transformer with
   RoPE+RMSNorm+SwiGLU, calendar embeddings, and local-window norm; retrofit
   `token_first_transformer`.
3. **Learned tokenizer:** prototype a BSQ candle tokenizer (LFQ, s1/s2 split) as a
   drop-in alternative to the quantile tokenizers; evaluate downstream.
4. **Finetune Kronos** on our symbols (the repo ships a finetune pipeline) and/or
   use its embeddings as a modality into `late_fusion` / `multimodal` / `moe`.
5. **Diffusion fusion:** Kronos tokenizer → discrete-diffusion head (Track B) for
   fast parallel forecasting; feed into `transformer_diffusion_fusion`.

Caveats: Kronos is pre-trained mostly on equities/“global exchanges” at coarser
bars; crypto 1m microstructure may need finetuning. It models OHLCV(+amount),
**not** the L2 book or footprints — those remain marketglot-native modalities.

---

## How to run locally

```bash
# outside the marketglot repo
git clone --depth 1 https://github.com/shiyu-coder/Kronos.git
cd Kronos
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pyarrow pytest

# 1) sanity: exact-reproduction regression test (downloads model from HF, CPU)
.venv/bin/python -m pytest tests/test_kronos_regression.py::test_kronos_predictor_regression -s

# 2) forecast on our BTCUSDT klines and map to UP/FLAT/DOWN
#    (scratch script lives in the Kronos clone: run_on_btc.py)
.venv/bin/python run_on_btc.py
```

*Compiled 2026-06-13 from the Kronos source at commit-depth-1 of `main`; numbers
verified on this machine. Re-verify against upstream before relying on them.*
