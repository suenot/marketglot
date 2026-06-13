# diffusion_orderbook — spec (module contract)

**Project 7 of marketglot — design stage (no implementation yet).**

A denoising **diffusion model over L2 order-book microstructure**, conditioned on
price-action context. Two tracks: **Track A** (continuous DDPM/DDIM over book
states / LOB-images — *v1, primary*) and **Track B** (discrete masked-token
diffusion over a market-token language — *v2, forward-looking*).

This spec is grounded in the research note
[`../docs/research/diffusion-llms.md`](../docs/research/diffusion-llms.md) (read
it first): Track A is "Track A — Continuous diffusion over order-book
microstructure", and is the direct analogue of the *"Painting the market"*
LOB-image + inpainting approach described there. Track B is the
"market-token diffusion" / Gemini-Diffusion direction (LLaDA / MaskGIT style).

This project **reuses existing repo components** — it does not reinvent the data
pipeline or the context encoder:

- **Project 4 `orderbook_encoder`** — data pipeline. We consume its `npz`
  samples directly (keys `ts`, `features (T, 4*depth)`, `mid`). No new book
  reconstruction is written here. See `../orderbook_encoder/SPEC.md`.
- **Project 1 `token_first_transformer`** — `PriceTransformer` candle encoder.
  We wrap it (frozen or fine-tuned) as the **conditioner** that turns recent
  candles into the conditioning vector `c`.

Output stays comparable to the rest of the repo: a **3-class** DOWN/FLAT/UP
decision, class order `[DOWN=0, FLAT=1, UP=2]`, ±`threshold_pct`.

---

## Tracks at a glance

| | Track A (v1, primary) | Track B (v2) |
|---|---|---|
| State space | continuous `R^d` (book vector or LOB-image) | discrete tokens (market-token language) |
| Forward process | Gaussian DDPM `q(x_t|x_0)` | masking / absorbing-state diffusion |
| Denoiser | ResMLP+FiLM (vector) or DiT (image) | masked-token Transformer (LLaDA/MaskGIT) |
| Objective | epsilon-MSE (v-pred optional) | masked-token cross-entropy (likelihood bound) |
| Sampling | DDPM / **DDIM** (10–50 steps) | parallel masked unmasking (4–16 reveal rounds) |
| Forecasting | **inpainting** the future LOB-image | unmask the future token block |
| Status | this spec specifies in full | this spec sketches; sources the tokenizer |

Both tracks are benchmarked on the same axis the research doc emphasizes:
**steps vs quality vs TPS/latency** — not accuracy alone.

---

## Data (reused from `orderbook_encoder`)

We do **not** download or reconstruct books here. We read the npz that
`orderbook_encoder/scripts/build_samples.py` already writes:

```
{samples_dir}/{symbol}/{exchange}/{date}.npz
  ts:       int64   (T,)            ms
  features: float32 (T, 4*depth)    [bid_off×D, bid_qty×D, ask_off×D, ask_qty×D]
  mid:      float64 (T,)
```

with `depth=20 ⇒ 4*depth = 80`. Feature layout and the `off = |price-mid|/mid`,
`qty = log1p(qty)` conventions are exactly as in `../orderbook_encoder/SPEC.md`.

**Two modeling granularities** (both Track A):

- **(a) single book state** `x0 ∈ R^(4*depth)` — one tick (80-dim at depth 20).
- **(b) window / LOB-image** `x0 ∈ R^(W × 4*depth)` — `W` consecutive ticks
  treated as a spatio-temporal image (time × feature). This is the
  "Painting the market" representation and the one used for **inpainting-based
  forecasting**.

**Standardization.** Per-feature mean/std are **fit on the train split only** and
frozen; all tracks operate in standardized space. `q_sample` adds noise in
standardized space; sampling de-standardizes for read-out (mid path).

**Optional aligned candle context.** For conditioning, each book sample at time
`ts[i]` is aligned to the most recent `ctx_len` 1m candles ending at or before
`ts[i]` (right-aligned, no look-ahead). Candle tokens come from
`token_first_transformer`'s tokenizer. Samples without enough candle history fall
back to the unconditional path (see classifier-free guidance below).

---

## Project structure (mirrors neighbors)

```
diffusion_orderbook/
  pyproject.toml                 # deps: torch, numpy, pyyaml, sklearn (+ pandas for ctx align)
  configs/default.yaml           # full run config
  configs/smoke.yaml             # tiny synthetic/CPU config for a smoke run
  diffusion/schedule.py          # beta schedules, q_sample, NoiseSchedule
  diffusion/denoiser.py          # ResMLPDenoiser (vector), DiTDenoiser (image)
  diffusion/conditioner.py       # wraps PriceTransformer -> conditioning vector c
  diffusion/sampler.py           # ddpm_sample, ddim_sample, inpaint
  dataset/diffusion_dataset.py   # windowed dataset over orderbook_encoder npz (+ ctx)
  models/probe.py                # linear probe / decision head; surprise_score
  training/trainer.py            # diffusion MSE training (+ optional probe head)
  scripts/train.py               # CLI: train diffusion (+ probe)
  scripts/forecast.py            # CLI: inpaint future -> mid path -> 3-class
  tests/                         # synthetic, no network
  # --- Track B (v2), sketched only ---
  discrete/                      # token diffusion (LLaDA/MaskGIT) — see Track B section
```

Only stdlib + deps from `pyproject.toml` — same baseline as `orderbook_encoder`
(`torch`, `numpy`, `pyyaml`, `scikit-learn`, `pandas`), no exotic deps. Intra-project
imports are relative to the project root (`from diffusion.schedule import NoiseSchedule`),
as with the neighbors.

---

## Track A — continuous DDPM (PRIMARY / v1)

### Forward process

Standard Gaussian DDPM. For a clean (standardized) sample `x0` and timestep `t`:

```
q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)
x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) eps,   eps ~ N(0, I)
```

`ᾱ_t = ∏_{s≤t} (1-β_s)`. Beta schedule is linear or cosine (config). Works
identically for the vector case (`x0 ∈ R^d`) and the image case
(`x0 ∈ R^(W×d)`, noise broadcast elementwise).

### `diffusion/schedule.py`

```python
class NoiseSchedule:
    """Precomputed DDPM coefficients for a fixed number of timesteps."""
    def __init__(self, num_steps: int, kind: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 2e-2) -> None: ...
    # tensors of shape (num_steps,): betas, alphas, alphas_cumprod,
    # sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    def to(self, device) -> "NoiseSchedule": ...

def make_betas(num_steps: int, kind: str, beta_start: float,
               beta_end: float) -> torch.Tensor:
    """Linear or cosine (Nichol & Dhariwal) beta schedule, shape (num_steps,)."""

def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             sched: NoiseSchedule) -> torch.Tensor:
    """Forward diffusion: returns x_t with the same shape as x0.
    t: int64 (B,); broadcasts coeffs over feature/window dims."""
```

### Denoiser `eps_theta(x_t, t, c)`

Predicts the noise `eps` (epsilon-objective). `t` enters via a **sinusoidal
timestep embedding** → small MLP. The conditioning vector `c` is injected via
**FiLM** (vector denoiser) or **cross-attention / FiLM** (image denoiser).

### `diffusion/denoiser.py`

```python
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps. t:(B,) -> (B, dim)."""

class FiLM(nn.Module):
    """Per-feature affine modulation from a conditioning vector."""
    def __init__(self, cond_dim: int, feat_dim: int) -> None: ...
    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor: ...

class ResMLPDenoiser(nn.Module):
    """Vector case: x_t in R^d. ResMLP blocks with FiLM(time + cond)."""
    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int,
                 time_emb_dim: int, cond_dim: int, dropout: float = 0.0) -> None: ...
    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None) -> torch.Tensor:
        """x_t:(B,d), t:(B,), cond:(B,cond_dim) or None -> eps_hat:(B,d)."""

class DiTDenoiser(nn.Module):
    """Image/window case: x_t in R^(W x d). DiT-style diffusion transformer:
    patchify time steps as tokens, adaLN/FiLM from (time + cond), optional
    cross-attention to a candle-context token sequence. Preferred — reuses the
    repo transformer stack."""
    def __init__(self, window: int, feat_dim: int, hidden_dim: int,
                 num_layers: int, num_heads: int, time_emb_dim: int,
                 cond_dim: int, cross_attend: bool = False,
                 dropout: float = 0.0) -> None: ...
    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None,
                cond_seq: torch.Tensor | None = None) -> torch.Tensor:
        """x_t:(B,W,d), t:(B,), cond:(B,cond_dim) (pooled context),
        cond_seq:(B,L,cond_dim) (optional token context for cross-attn)
        -> eps_hat:(B,W,d)."""
```

A 1D U-Net over the window axis is an acceptable alternative to `DiTDenoiser`;
DiT is preferred to reuse the repo's transformer stack.

### Conditioning `c` = price-action context

`c` summarizes recent candles, produced by `token_first_transformer`'s
`PriceTransformer` (the encoder body, before its 3-class head). We expose both a
pooled vector (for FiLM) and the token sequence (for cross-attention).

**Classifier-free guidance (CFG).** During training, drop `c` with probability
`p_uncond` (replace with a learned null embedding). At sampling, combine
conditional and unconditional predictions:

```
eps = eps_theta(x_t, t, ∅) + w * (eps_theta(x_t, t, c) - eps_theta(x_t, t, ∅))
```

### `diffusion/conditioner.py`

```python
class CandleConditioner(nn.Module):
    """Wraps token_first_transformer's PriceTransformer as a context encoder.

    Import note: PriceTransformer lives in project 1
    (token_first_transformer/models/price_transformer.py). We import its
    encoder body and ignore its classification head. Can run frozen (use
    pretrained weights) or be fine-tuned end-to-end (config flag).
    """
    def __init__(self, price_transformer: nn.Module, cond_dim: int,
                 p_uncond: float = 0.1, freeze: bool = True) -> None: ...
    def forward(self, ctx_tokens: dict | None, drop_mask: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """ctx_tokens: {'delta','vol','vb'} int64 (B,T) or None.
        drop_mask:(B,) bool — positions forced to the null embedding (CFG).
        Returns (cond_vec:(B,cond_dim), cond_seq:(B,T,cond_dim))."""
    @property
    def null_embedding(self) -> torch.Tensor: ...
```

### Objective

Noise-prediction MSE (epsilon-objective):

```
L = E_{x0, t, eps, c} || eps - eps_theta(x_t, t, c) ||^2
```

with CFG dropout applied to `c`. **v-prediction** (`v = sqrt(ᾱ_t)·eps -
sqrt(1-ᾱ_t)·x0`) is supported as a config option (`objective: eps | v`) for more
stable few-step sampling; the loss and sampler convert accordingly.

### Sampling

### `diffusion/sampler.py`

```python
def ddpm_sample(model: nn.Module, sched: NoiseSchedule, shape: tuple[int, ...],
                cond: torch.Tensor | None = None,
                cond_seq: torch.Tensor | None = None,
                guidance_scale: float = 1.0, device=None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Ancestral DDPM sampling over all `sched.num_steps` steps -> x0_hat."""

def ddim_sample(model: nn.Module, sched: NoiseSchedule, shape: tuple[int, ...],
                num_inference_steps: int = 20, eta: float = 0.0,
                cond: torch.Tensor | None = None,
                cond_seq: torch.Tensor | None = None,
                guidance_scale: float = 1.0, device=None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """DDIM with a strided subsequence of timesteps (eta=0 deterministic).
    Emphasis: small step counts (10-50) for low latency."""

def inpaint(model: nn.Module, sched: NoiseSchedule, x_known: torch.Tensor,
            mask: torch.Tensor, num_inference_steps: int = 20,
            cond: torch.Tensor | None = None,
            cond_seq: torch.Tensor | None = None,
            guidance_scale: float = 1.0, device=None,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Mask-conditioned (RePaint-style) sampling over a LOB-image.
    x_known:(B,W,d) with observed entries set; mask:(B,W,d) bool, True = KNOWN.
    At every reverse step, re-impose the known region via q_sample(x_known,t)
    so observed cells are preserved; returns the filled x0_hat:(B,W,d)."""
```

Sampling is benchmarked across step counts (DDPM full vs DDIM 50/20/10) on the
**steps × quality × TPS/latency** axis, per the research doc's "practical notes".

### Forecasting via inpainting

The headline use of the image granularity (b):

1. Build a LOB-image over `[past | future]` ticks: `W = W_past + W_future`.
2. Set `mask = True` for the `W_past` observed rows, `False` for the future rows.
3. `inpaint(...)` denoises the future region conditioned on the observed past
   (and on candle context `c`).
4. De-standardize; derive the **predicted mid path** from the filled future rows
   (the book's mid is recoverable from the bid/ask offset+qty features, or a
   small read-out maps filled features → mid).
5. Map the mid path to **3 classes**: `ret = mid_future[h]/mid_now - 1`;
   `UP=2` if `ret > threshold_pct/100`, `DOWN=0` if `< -threshold_pct/100`,
   else `FLAT=1` — identical rule and class order to `orderbook_encoder`.

This avoids autoregressive error accumulation (the "Painting the market"
argument) and yields a **distribution** over futures (sample K times → mean/var
of the mid path).

### Repo-comparable evaluation heads — `models/probe.py`

To stay directly comparable to the repo's 3-class metrics without relying solely
on sampling, two cheap read-outs on the **learned conditional features**:

```python
class LinearProbe(nn.Module):
    """Linear/MLP decision head on denoiser conditional features -> 3 logits.

    Features = denoiser hidden state at a representative low timestep
    (e.g. t≈0) for the observed book, concatenated with the pooled context
    cond_vec. Denoiser is frozen while the probe trains (lightweight).
    Class order [DOWN=0, FLAT=1, UP=2]."""
    def __init__(self, feat_dim: int, cond_dim: int, num_classes: int = 3,
                 hidden_dim: int | None = None) -> None: ...
    def forward(self, feats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """-> (B, 3) logits."""

def extract_features(model: nn.Module, x0: torch.Tensor, t_ref: int,
                     cond: torch.Tensor | None, sched: NoiseSchedule
                     ) -> torch.Tensor:
    """Run the denoiser at a fixed reference timestep on the observed book and
    return an internal representation (B, feat_dim) for the probe."""

def surprise_score(model: nn.Module, x0: torch.Tensor,
                   cond: torch.Tensor | None, sched: NoiseSchedule,
                   t_grid: list[int] | None = None) -> torch.Tensor:
    """'Microstructure surprise': conditional reconstruction error / score
    magnitude at the OBSERVED book x0. For each t in t_grid, add noise via
    q_sample then measure || eps - eps_theta(x_t, t, cond) ||^2; average.
    High score = the current book is unlikely under the conditional model
    (a stress / novelty signal). Returns (B,)."""
```

`LinearProbe` gives a head-to-head 3-class number against project 4;
`surprise_score` is an auxiliary stress/novelty feature also consumable by
project 8 (`transformer_diffusion_fusion`).

### Dataset — `dataset/diffusion_dataset.py`

```python
class DiffusionBookDataset(torch.utils.data.Dataset):
    """Windowed dataset over orderbook_encoder npz. Reuses keys ts/features/mid.

    mode='vector' -> __getitem__ returns one book state x0:(4*depth,);
    mode='image'  -> returns a window x0:(W, 4*depth).
    With with_context=True, also returns aligned candle tokens
    {'delta','vol','vb'} (T,) for the conditioner; None-filled if unavailable.
    Windows never cross an npz-file boundary; windows with a ts gap > 2x the
    expected spacing are dropped (same data-hole rule as orderbook_encoder).
    Optionally returns the future-horizon label for probe training/eval.
    """
    def __init__(self, npz_paths: list[Path], mode: str = "image",
                 window: int = 64, w_future: int = 0,
                 with_context: bool = False, ctx_len: int = 128,
                 horizon_sec: float = 60.0, threshold_pct: float = 0.05,
                 interval_sec: float = 1.0,
                 candle_npz: dict | None = None) -> None: ...

def build_splits(cfg: dict) -> tuple["DiffusionBookDataset", ...]:
    """train/val/test from cfg['split']['train_days'|'val_days'|'test_days'],
    same date-list convention as orderbook_encoder. Standardization stats are
    fit on the train split and stored in the returned datasets / config."""
```

### Training — `training/trainer.py` + `scripts/train.py`

```python
def train(cfg: dict) -> dict:
    """Train the diffusion model (epsilon/v MSE) and, optionally, the probe.

    Stage 1: diffusion MSE with CFG dropout on c (AdamW, EMA of weights
    recommended). Stage 2 (optional, cfg['probe']['enable']): freeze denoiser,
    train LinearProbe with class-weighted CrossEntropy (weights inverse to
    train class frequency), early stop on val loss. device auto mps->cuda->cpu.
    Artifacts in artifacts/run_YYYYMMDD_HHMMSS/: best.pt, ema.pt, config.json,
    standardizer.json, and (if probe) test_metrics.json via
    sklearn.metrics.classification_report(output_dict=True) + confusion matrix
    — same artifact format as neighboring projects. Returns final metrics."""
```

`scripts/train.py` — argparse CLI: `--config configs/default.yaml`,
optional `--smoke` (loads `configs/smoke.yaml`).
`scripts/forecast.py` — argparse CLI: load a run, build the past LOB-image for a
given symbol/timestamp, run `inpaint`, print/save the predicted mid path,
K-sample mean/var, and the 3-class decision.

---

## Track B — discrete token diffusion (v2)

Forward-looking variant; the **speed** direction from the research doc (LLaDA /
MaskGIT style — see `../docs/research/diffusion-llms.md` §"Track B" and the
LLaDA/MaskGIT/MDLM references). Specified briefly here; full design deferred
until Track A is validated.

- **Tokenized book / market-token language.** Reuse a **footprint / book
  tokenizer** (project `footprint_encoder`, the same tokenization philosophy as
  the candle/indicator tokenizers) to map a book state or window into a sequence
  of discrete tokens — a volume-at-price / footprint grid is a natural fit
  (cf. *DiffVolume* in the research doc).
- **Masked / absorbing-state diffusion.** Forward process progressively
  **masks** tokens; a bidirectional Transformer predicts the masked tokens
  (likelihood-bound objective), as in LLaDA / MDLM. Image-analogue:
  confidence-based parallel unmasking (MaskGIT).
- **Parallel forecasting.** Append a **masked future block** of market tokens
  and **unmask it in a few reveal rounds (4–16)** — all positions in parallel.
  Read the future tokens (or their logits) for the UP/FLAT/DOWN decision and an
  explicit uncertainty over the masked future. This is where the **TPS/latency
  advantage** shows up at inference.
- **Sketched module layout** (`discrete/`): `discrete/tokenizer.py` (adapter to
  `footprint_encoder`), `discrete/mask_schedule.py`, `discrete/mdlm_denoiser.py`
  (masked-token Transformer), `discrete/sampler.py` (`maskgit_sample`,
  `unmask_future`). KV-cache / block-diffusion speedups (research doc
  "Practical notes") are tracked here for longer token horizons.

---

## Tests

pytest, **no network, synthetic data only** (tiny tensors / fabricated npz in
`tmp_path`). Run from the project root with the shared torch venv (as in
`orderbook_encoder` — do not create a new heavy venv):

`/Users/suenot/projects/w_trading/w_training/token_first_transformer/.venv/bin/python -m pytest tests/ -x -q`

- `test_schedule.py` — `make_betas` shapes & monotonic `ᾱ_t`; `q_sample` output
  shape matches `x0` for both vector and image inputs; at `t=0` returns ≈ `x0`.
- `test_denoiser.py` — `ResMLPDenoiser.forward` shape `(B,d)`; `DiTDenoiser.forward`
  shape `(B,W,d)`; with and without `cond`; **backward** pass produces finite
  grads; `timestep_embedding` shape.
- `test_conditioner.py` — `CandleConditioner` returns `(cond_vec, cond_seq)` of
  expected shapes; CFG `drop_mask` forces the null embedding; `None` tokens →
  unconditional path. (Uses a tiny stub PriceTransformer, no project-1 weights.)
- `test_sampler.py` — `ddim_sample` returns a tensor of the requested `shape`
  with the right step count; `ddpm_sample` runs on a tiny model;
  **`inpaint` respects the mask** (known cells unchanged within tolerance).
- `test_probe.py` — `LinearProbe.forward` → `(B,3)` logits; **probe trains 1
  step** (loss is finite, params update) on a frozen toy denoiser;
  `surprise_score` returns `(B,)` and is higher for out-of-distribution books.
- `test_dataset.py` — windowed sampling over fabricated npz: vector vs image
  shapes, window never crosses file boundary, ts-gap windows dropped, labels
  UP/FLAT/DOWN match a synthetic mid path.

---

## Style

As in neighboring projects: short modules, English docstrings, type hints, no
gratuitous abstraction. Intra-project imports relative to the project root
(`from diffusion.sampler import ddim_sample`). Reuse — never re-implement —
`orderbook_encoder`'s data pipeline and `token_first_transformer`'s
`PriceTransformer`. Keep diffusion **steps small** and report **TPS/latency**
alongside accuracy, per `../docs/research/diffusion-llms.md`.

## Dependencies

- **Project 4 `orderbook_encoder`** (data pipeline; `npz` samples) — required.
- **Project 1 `token_first_transformer`** (`PriceTransformer` context encoder) —
  required for conditioning.
- **Project `footprint_encoder`** (tokenizer) — Track B only (v2).
- Consumed by **Project 8 `transformer_diffusion_fusion`** (context ⊕
  distribution-features ⊕ book-embedding decision head).
