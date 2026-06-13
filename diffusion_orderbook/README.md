# diffusion_orderbook

Denoising diffusion model over L2 order-book microstructure, conditioned on
price-action context (project 7 of marketglot).

## Idea

Treat the order book as data a diffusion model can denoise — either a single book
state or a window viewed as a spatio-temporal **LOB-image**. Condition the
denoiser on recent candles (via `token_first_transformer`'s `PriceTransformer`)
and **forecast by inpainting**: mask the future region of the LOB-image, denoise
to fill it, then read off the predicted mid path → DOWN/FLAT/UP. Diffusion
generates the whole future block **in parallel**, sidestepping autoregressive
error accumulation and giving faster, more predictable local inference. Same
3-class target and class order `[DOWN=0, FLAT=1, UP=2]` as the rest of the repo.

Two tracks: **Track A** — continuous DDPM/DDIM (primary, v1); **Track B** —
discrete masked-token diffusion (LLaDA / MaskGIT style, v2, the speed direction).

## Status

Design stage — see [SPEC.md](SPEC.md). No implementation yet.

## Dependencies

- Reuses the **`orderbook_encoder`** data pipeline (npz: `ts`, `features`, `mid`).
- Reuses **`token_first_transformer`**'s `PriceTransformer` for context.

## Background

Motivation and references: [../docs/research/diffusion-llms.md](../docs/research/diffusion-llms.md).

Part of the [marketglot](../README.md) monorepo.
