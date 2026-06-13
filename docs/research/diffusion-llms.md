# Research: Diffusion language models & diffusion for markets

> Why this matters for marketglot: diffusion decoders generate **in parallel**
> (iterative denoising of a whole block) instead of one token at a time. Locally
> that means much higher throughput (tokens/sec) and lower, more predictable
> latency than autoregressive (AR) models — exactly the property we want for a
> trading model that must react fast. And because marketglot already treats the
> market as a **token language**, a *discrete* diffusion model is a natural fit.

This note surveys the relevant work and maps it onto three projects:
`diffusion_orderbook` (7), `transformer_diffusion_fusion` (8), and a discrete
"market-token diffusion" track that the footprint/candle/indicator tokenizers
make possible.

---

## 1. Diffusion language models (text)

### Autoregressive vs diffusion generation

- **Autoregressive (AR)** — GPT-style. Generate token *t+1* conditioned on
  tokens *≤ t*. Strictly sequential: N tokens ⇒ N forward passes. Quality is
  high, but latency grows linearly with output length and errors accumulate
  left-to-right.
- **Diffusion** — start from fully noised/masked output and **iteratively refine
  the entire sequence in parallel** over a *fixed, small* number of steps
  (independent of output length). Supports bidirectional conditioning and
  *error correction* (a token decided early can be revised later).

### Gemini Diffusion (Google DeepMind, 2025)

The headline data point that motivates this direction:

- **Sampling speed: ~1,479 tokens/sec**, with ~**0.84 s** fixed overhead —
  "generates entire blocks of tokens at once," "significantly faster than even
  our fastest model so far."
- Mechanism: "learns to generate outputs by refining noise, step-by-step,"
  enabling parallel generation and **error correction during generation**.
- Quality (vs Gemini 2.0 Flash-Lite): competitive on code/math
  (LiveCodeBench 30.9% vs 28.5%, AIME 23.3% vs 20.0%), weaker on
  knowledge/reasoning (GPQA 40.4% vs 56.5%, BIG-Bench 15.0% vs 21.0%).
- Status: experimental demo.

**Takeaway:** diffusion trades some peak reasoning quality for a large
throughput/latency win. For short-horizon trading inference (small output, tight
latency budget) that trade is attractive.

### Open / commercial diffusion LLMs

- **LLaDA** (*Large Language Diffusion with mAsking*, 2025) — masked **discrete
  diffusion** LM trained from scratch with **full (bidirectional) attention**.
  Forward process progressively masks tokens; a Transformer predicts the masked
  tokens; trained on a **likelihood lower bound** instead of next-token loss.
  An 8B LLaDA is competitive with LLaMA3-8B on in-context learning and even
  beats GPT-4o on a reversal-poem task (no "reversal curse"). Shows core LLM
  abilities do **not** require autoregression. `LLaDA-MoE` adds sparse experts.
- **Mercury** (Inception Labs, 2025) — commercial diffusion LLM for code,
  marketed on very high inference speed at AR-comparable quality.
- **Theory lineage:** **D3PM** (structured discrete diffusion) → **SEDD** (score
  entropy discrete diffusion) → **MDLM** (masked diffusion as a clean,
  well-behaved special case). **MaskGIT** is the image analogue (parallel
  iterative unmasking, confidence-based reveal). **Block diffusion** interpolates
  between AR and diffusion (denoise block-by-block) to recover KV-caching.

### Why diffusion is faster locally — and the catches

- **Faster:** decoding cost ≈ `steps × one_forward_pass`, with `steps`
  small/fixed and **independent of sequence length**; all positions update in
  parallel ⇒ high GPU/Metal utilization and high TPS.
- **Catches:** (1) classic dLLMs use full attention and **can't reuse a KV-cache**
  the way AR does (active research: block diffusion, self-speculative decoding,
  caching for masked dLLMs); (2) quality depends on step count (fewer steps =
  faster but coarser); (3) reasoning/knowledge can lag AR at equal size.
- **The key metric** across recent dLLM papers is **valid tokens/sec (TPS)** under
  a given decoding strategy — parallel decoding + caching is where 2025 work
  concentrates.

---

## 2. Diffusion for markets & limit-order-book data

Diffusion has been applied directly to financial microstructure — relevant to
projects 7 and 8:

- **"Painting the market" (2509.05107, 2025)** — represents the **LOB as an image**
  and uses a diffusion model with **inpainting** to forecast future LOB states.
  Captures spatio-temporal structure, **generates long sequences in parallel**,
  and so **avoids the error accumulation of AR** over long horizons. Reports SOTA
  on LOB-Bench *despite using only Level-2 data*. **Forecasting = inpainting:**
  fix the observed past region of the image, mask the future region, denoise.
- **DiffLOB** — diffusion for **counterfactual** ("what-if") LOB generation.
- **TRADES** — diffusion-based **order-level** market simulator.
- **DiffVolume** — diffusion over high-dimensional **LOB volume snapshots** across
  price levels (close to our footprint volume-at-price grid).
- **Synthetic financial time series via diffusion** — reproduces stylized facts
  (fat tails, volatility clustering, seasonality) better than GAN/VAE baselines;
  cross-attention conditioning enables trend/volatility-constrained generation.

**Design insight:** treating microstructure as a **spatio-temporal image** (price
levels × time, or price bins × {buy,sell}) plays to diffusion's strengths and
sidesteps AR error propagation. This is the same shape as our **footprint** grid
and our order-book feature window.

---

## 3. How this maps onto marketglot

Three concrete tracks, from least to most speculative:

### Track A — Continuous diffusion over order-book microstructure → `diffusion_orderbook` (7)
DDPM/DDIM over the order-book feature vectors we already produce
(`orderbook_encoder` npz: `features (T, 4·depth)`, `mid`), or over a **window**
treated as a LOB-image `(W × 4·depth)`. Condition the denoiser on **price-action
context** from the candle transformer (`token_first_transformer`) via FiLM or
cross-attention. Uses: (a) self-supervised representation for a 3-class probe,
(b) a **"microstructure surprise"** score (conditional reconstruction error),
(c) **forecasting via inpainting** (mask the future window, denoise, read the
predicted mid path). This is the most grounded track — direct analogue of
"Painting the market."

### Track B — Discrete "market-token diffusion" (the Gemini-Diffusion direction)
marketglot already discretizes candles, indicators, and (now) footprints into
**tokens**. A **masked/discrete diffusion** model (LLaDA / MaskGIT style) over
these token streams can **forecast a block of future market tokens in parallel**,
then read them (or their logits) for the UP/FLAT/DOWN decision — fast, with
explicit uncertainty over the masked future. This is where the **TPS advantage**
shows up at inference. Recommended as **v2** for the diffusion projects once the
continuous track is validated; the footprint tokenizer and candle tokenizer are
the enablers.

### Track C — Fusion → `transformer_diffusion_fusion` (8)
Transformer **context** (candle/indicator/footprint encoders) + a **conditional
diffusion decoder** (Track A or B) + a **decision head** over
`[context ⊕ distribution-features ⊕ current-book-embedding]`. Distribution
features can be: a denoiser hidden state at a representative step, statistics of
K sampled futures (mean/var/quantiles of the mid path), or the conditional
score/surprise at the observed book. Two-stage training (pretrain diffusion,
then train head) for stability.

### Practical notes for local inference
- Keep diffusion **steps small** (e.g. DDIM 10–50; masked-diffusion 4–16 reveal
  rounds) and benchmark **TPS / latency**, not just accuracy.
- Prefer **DiT-style** (diffusion transformer) denoisers so the work reuses the
  transformer stack already in the repo.
- Track **block diffusion / caching** for KV-cache-like speedups if/when we move
  to longer token horizons.

---

## References

- Gemini Diffusion — <https://deepmind.google/models/gemini-diffusion/>
- LLaDA — *Large Language Diffusion Models*, arXiv:2502.09992
- LLaDA-MoE — arXiv:2509.24389
- D3PM — *Structured Denoising Diffusion Models in Discrete State-Spaces*, arXiv:2107.03006
- SEDD — *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, arXiv:2310.16834
- MDLM — *Simple and Effective Masked Diffusion Language Models*, arXiv:2406.07524
- MaskGIT — *Masked Generative Image Transformer*, arXiv:2202.04200
- Block Diffusion — *Interpolating Between Autoregressive and Diffusion LMs*, arXiv:2503.09573
- Mercury (Inception Labs) — <https://www.inceptionlabs.ai/>
- Painting the market — *Generative diffusion models for LOB simulation and forecasting*, arXiv:2509.05107
- DiffLOB — *Diffusion Models for Counterfactual Generation in Limit Order Books*, arXiv:2602.03776
- Synthetic financial time series by diffusion — arXiv:2410.18897

*Compiled 2026-06-13. Numbers quoted from vendor pages/abstracts as cited; verify
against the latest sources before relying on them.*
