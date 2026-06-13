# transformer_diffusion_fusion — spec (module contract)

**Project 8 of marketglot — design stage (no implementation yet). Depends on
projects 1, 4, 7 (+ optional 5, 9).**

The **most complex** project in the repo: it combines three pieces that already
exist (or are specified) elsewhere into one decision model —

1. a **transformer context encoder** producing a context vector `c` from candle
   tokens (plus, optionally, indicator and footprint modalities),
2. a **conditional diffusion order-book decoder** (imported wholesale from
   project 7 `diffusion_orderbook`) from which we extract **distribution
   features**, and
3. a **decision head** (MLP) over
   `[c ⊕ distribution_features ⊕ current_book_embedding]` → **3 logits**
   (`[DOWN=0, FLAT=1, UP=2]`, ±`threshold_pct`, identical to every other
   project).

This is **Track C — Fusion** from the research note
[`../docs/research/diffusion-llms.md`](../docs/research/diffusion-llms.md) (read
it first; §3 "How this maps onto marketglot"). The diffusion decoder here is
exactly the one specified in
[`../diffusion_orderbook/SPEC.md`](../diffusion_orderbook/SPEC.md) — we **reuse
its denoiser / sampler / conditioner / probe contracts; we never redefine
them.**

> Design rule for this project: **compose, don't reinvent.** Every neural block
> below is either imported from a neighbor (projects 1, 4, 5, 7, 9) or a thin
> wrapper that calls into one. The only genuinely new modules are the
> `DiffusionFeatureExtractor` (a/b/c strategies), the pluggable `ContextEncoder`,
> the `FusionClassifier` head, and the two-regime trainer.

---

## What is reused (imports), at a glance

| Imported symbol | From project | Role here |
|---|---|---|
| `PriceTransformer` | 1 `token_first_transformer/models/price_transformer.py` | candle context encoder body |
| `OrderbookEncoder` (`.encode`) | 4 `orderbook_encoder/models/orderbook_mlp.py` | **current book** embedding |
| npz samples (`ts/features/mid`) | 4 `orderbook_encoder` pipeline | order-book data (no new pipeline) |
| `NoiseSchedule`, `q_sample` | 7 `diffusion_orderbook/diffusion/schedule.py` | forward process for feature strategy (a)/(c) |
| `ResMLPDenoiser` / `DiTDenoiser` | 7 `diffusion/denoiser.py` | the conditional denoiser `eps_theta(x_t,t,c)` |
| `CandleConditioner` | 7 `diffusion/conditioner.py` | wraps `PriceTransformer` → `(cond_vec, cond_seq)` |
| `ddim_sample`, `ddpm_sample`, `inpaint` | 7 `diffusion/sampler.py` | K-sample futures for feature strategy (b) |
| `extract_features`, `surprise_score` | 7 `models/probe.py` | feature strategies (a) and (c) |
| `DiffusionBookDataset` | 7 `dataset/diffusion_dataset.py` | windowed book + aligned candle context |
| indicator encoder embeddings | 5 `multimodal_encoder/models/multimodal_model.py` | optional indicator modality in `ContextEncoder` |
| footprint embedding | 9 `footprint_encoder` | optional footprint modality in `ContextEncoder` |

Optional dependencies (5 indicator, 9 footprint) are **lazily loaded** — the
project runs candle-only if they are absent.

---

## End-to-end shape

```
                candle tokens (B,T)         indicator tokens (B,T,·)   footprint (B,·)
                      │                              │ (opt)              │ (opt)
                      ▼                              ▼                    ▼
            ┌──────────────────────────  ContextEncoder  ──────────────────────────┐
            │  CandleEnc (PriceTransformer body)  [⊕ IndicatorEnc ⊕ FootprintEnc]   │
            └───────────────────────────────────────────────────────────────────────┘
                      │  c : (B, ctx_dim)               │ cond_vec / cond_seq → diffusion
                      │                                 ▼
   current book ──► OrderbookEncoder.encode      DiffusionFeatureExtractor (a|b|c)
   x0 (B, 4·depth)        │                              │  imports project-7 denoiser/sampler/probe
                          │ book_emb (B, emb_dim)        ▼  df : (B, df_dim)
                          ▼                              │
            ┌──────────────  concat  [ c ⊕ df ⊕ book_emb ]  ──────────────┐
            │                       FusionClassifier (MLP)                 │
            └──────────────────────────── → 3 logits ──────────────────────┘
                                       [DOWN=0, FLAT=1, UP=2]
```

---

## Data — aligned (candle window, order-book window, label)

We reuse **project-7's** `DiffusionBookDataset`, which already aligns an
order-book window to its candle context and (optionally) a future-horizon label.
This project wraps it so each item yields **all three** streams plus the label.

- **Order book** comes from `orderbook_encoder` npz (`ts/features/mid`,
  `depth=20 ⇒ 4·depth=80`). No book reconstruction is written here.
- **Candle context** is `ctx_len` 1m candles tokenized by
  `token_first_transformer`'s tokenizer, **right-aligned** to end at or before
  the book sample's `ts[i]` (no look-ahead). Items without enough candle history
  fall back to the conditioner's unconditional (null-embedding) path, exactly as
  in project 7.
- **Current book** `x0` is the single book state at `ts[i]` (the most recent row
  of the book window) — fed to `OrderbookEncoder.encode` and used as the
  observed book for diffusion feature strategies (a)/(c).
- **Label** uses the project-4 / project-7 rule verbatim:
  `h = round(horizon_sec / interval_sec)`; `ret = mid[i+h]/mid[i] - 1`;
  `UP=2` if `ret > threshold_pct/100`, `DOWN=0` if `< -threshold_pct/100`,
  else `FLAT=1`. Windows never cross an npz-file boundary; a `ts` gap > 2× the
  expected spacing drops the sample (same data-hole rule as the neighbors).

**Time alignment summary:** book grid is 1 Hz (`interval_sec=1.0`); candles are
1 m. For book sample at `ts[i]`, take candles with close-time `≤ ts[i]`, newest
`ctx_len` of them. The diffusion window (strategy b) extends `w_future` ticks
**past** `ts[i]` only at *training* time for the diffusion stage; at decision
time only the observed past + the conditioning candles are available (the future
is what the sampler imputes).

### `dataset/fusion_dataset.py`

```python
class FusionDataset(torch.utils.data.Dataset):
    """Aligns a candle window + an order-book window + the current book + label.

    Thin wrapper over project-7's DiffusionBookDataset (image mode, with
    context) that additionally surfaces the *current* book state x0 and the
    3-class label. Reuses ts/features/mid npz from orderbook_encoder and the
    candle tokenizer from token_first_transformer — no new data pipeline.
    """
    def __init__(self, book_dataset: "DiffusionBookDataset") -> None: ...
    def __getitem__(self, i: int) -> dict:
        """Returns:
          'ctx_tokens': {'delta','vol','vb'} int64 (ctx_len,) or None,
          'ind_tokens': list[int64 (ctx_len,)] | None,   # optional (project 5)
          'footprint':  float32 (fp_dim,)  | None,        # optional (project 9)
          'book_window': float32 (W, 4*depth),            # for strategy (b)
          'x0':          float32 (4*depth,),              # current book
          'label':       int64 scalar in {0,1,2}.
        """

def build_splits(cfg: dict) -> tuple["FusionDataset", "FusionDataset", "FusionDataset"]:
    """Delegates to diffusion_orderbook.dataset.build_splits for the underlying
    DiffusionBookDataset (train/val/test by cfg['split']['*_days']), then wraps
    each in FusionDataset. Standardization stats fit on train only (reused from
    project 7's standardizer) and carried in the returned datasets / config."""
```

A `collate_fn` batches the dict, stacking tensors and carrying `None` modalities
through as `None` (so the model takes the unconditional / drop path).

---

## Project structure (mirrors neighbors)

```
transformer_diffusion_fusion/
  pyproject.toml                 # deps: torch, numpy, pyyaml, sklearn, pandas
  configs/default.yaml           # full run config (encoder, diffusion, head, training)
  configs/smoke.yaml             # tiny synthetic/CPU config for a smoke run
  fusion/context_encoder.py      # ContextEncoder: compose 1..N modality encoders -> c
  fusion/diffusion_features.py   # DiffusionFeatureExtractor: strategies a / b / c
  models/fusion_head.py          # FusionClassifier (forward -> 3 logits; .encode())
  dataset/fusion_dataset.py      # FusionDataset (candle ⊕ book window ⊕ x0 ⊕ label)
  training/trainer.py            # two-stage + joint multi-task regimes
  scripts/train.py               # CLI: --config / --smoke / --regime {two_stage,joint}
  scripts/eval.py                # CLI: load a run -> test_metrics.json
  tests/                         # synthetic, no network
```

Only stdlib + deps from `pyproject.toml` — same baseline as `orderbook_encoder`
and `diffusion_orderbook` (`torch`, `numpy`, `pyyaml`, `scikit-learn`,
`pandas`), no exotic deps. Intra-project imports are relative to the project root
(`from fusion.context_encoder import ContextEncoder`); cross-project imports name
the neighbor explicitly (`from diffusion.sampler import ddim_sample`,
`from models.orderbook_mlp import OrderbookEncoder`) and are documented per
module below.

---

## Module contracts

### `fusion/context_encoder.py` — the pluggable context encoder

Produces the context vector `c`. Composes **1..N modality encoders**; candle is
mandatory (primary), indicator (project 5) and footprint (project 9) are
optional plug-ins. Each modality encoder is a small `nn.Module` exposing a common
`forward(inputs) -> (B, modality_dim)`; `ContextEncoder` concatenates their
outputs and projects to `ctx_dim`.

```python
class ModalityEncoder(nn.Module):
    """Common interface for a single modality. Subclasses below."""
    out_dim: int
    def forward(self, inputs) -> torch.Tensor: ...   # (B, out_dim)

class CandleContextEncoder(ModalityEncoder):
    """Wraps token_first_transformer's PriceTransformer ENCODER BODY (the
    transformer + LayerNorm, *not* its 3-class head) and pools to a vector.

    Import note: PriceTransformer lives in project 1
    (token_first_transformer/models/price_transformer.py). We reuse exactly the
    same body project 7's CandleConditioner wraps; in fact this class may *be* a
    CandleConditioner in disguise — see `from_conditioner`. Can run frozen
    (pretrained weights) or be fine-tuned (config flag)."""
    def __init__(self, price_transformer: nn.Module, out_dim: int,
                 freeze: bool = True) -> None: ...
    def forward(self, ctx_tokens: dict | None) -> torch.Tensor:
        """ctx_tokens: {'delta','vol','vb'} int64 (B,T) or None -> (B, out_dim).
        None -> a learned null/zero vector (unconditional)."""

class IndicatorContextEncoder(ModalityEncoder):
    """OPTIONAL (project 5). Reuses multimodal_encoder's indicator embedding
    stack (per-indicator nn.Embedding -> concat -> proj) pooled to a vector.
    Lazily imported; absence is fine (candle-only run)."""
    def __init__(self, ind_vocab_sizes: list[int], ind_emb_dim: int,
                 out_dim: int) -> None: ...
    def forward(self, ind_tokens: list[torch.Tensor] | None) -> torch.Tensor: ...

class FootprintContextEncoder(ModalityEncoder):
    """OPTIONAL (project 9 footprint_encoder). Wraps the footprint embedding
    module; lazily imported. Absence is fine."""
    def __init__(self, footprint_module: nn.Module, out_dim: int) -> None: ...
    def forward(self, footprint: torch.Tensor | None) -> torch.Tensor: ...

class ContextEncoder(nn.Module):
    """Composes 1..N ModalityEncoders -> context vector c.

    Concatenates each present modality's (B, out_dim) output (absent/None
    modalities contribute their learned null vector), then Linear -> LayerNorm
    -> GELU -> ctx_dim. Also exposes the candle conditioning needed by the
    diffusion decoder so we encode the candles ONCE and share them.
    """
    def __init__(self, encoders: dict[str, ModalityEncoder], ctx_dim: int) -> None: ...
    @property
    def ctx_dim(self) -> int: ...
    def forward(self, batch: dict) -> torch.Tensor:
        """batch carries 'ctx_tokens' / 'ind_tokens' / 'footprint' (any may be
        None) -> c:(B, ctx_dim)."""
    def candle_conditioning(self, batch: dict
                            ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (cond_vec:(B,cond_dim), cond_seq:(B,T,cond_dim) | None) for
        the project-7 denoiser/sampler — the SAME candle encoding reused as the
        diffusion conditioner, avoiding a second forward pass."""
```

Composition is config-driven (`context.modalities: [candle, indicator?,
footprint?]`); `candle` is required and must be listed first.

### `fusion/diffusion_features.py` — distribution features (strategies a / b / c)

The heart of the fusion idea: turn the **project-7 conditional diffusion model**
into a fixed-size **distribution feature** vector `df`. THREE strategies are
specified; a config picks one or **combines** several (concatenate, then the head
sees `df_dim = sum of selected widths`). **Recommendation: make it configurable
and default to (a) ⊕ (c)** — both are single-/few-forward-pass and cheap; add (b)
when latency budget allows, since it is the richest but most expensive (K full
sampling rollouts).

All three call **into project 7** and never reimplement diffusion math:

```python
class DiffusionFeatureExtractor(nn.Module):
    """Extracts distribution features from the project-7 conditional diffusion
    model. Strategies (selectable & combinable via `strategies`):

      (a) 'hidden'   — the denoiser's intermediate hidden embedding at a
                       representative timestep t_ref on the OBSERVED book.
                       Uses diffusion_orderbook.models.probe.extract_features.
      (b) 'samples'  — summary statistics (mean, var, quantiles) over K SAMPLED
                       future books / mid-paths via the project-7 sampler
                       (ddim_sample / inpaint). Captures the *forecast
                       distribution*.
      (c) 'surprise' — the conditional score / 'surprise' at the observed book:
                       how (un)expected the current book is given price action.
                       Uses diffusion_orderbook.models.probe.surprise_score.

    Imports (project 7):
      from diffusion.schedule  import NoiseSchedule, q_sample
      from diffusion.sampler   import ddim_sample, inpaint
      from models.probe        import extract_features, surprise_score
    The denoiser (ResMLPDenoiser/DiTDenoiser) and NoiseSchedule are passed in
    already constructed (and usually frozen). This class adds NO new diffusion
    parameters; only small projection/normalization layers per strategy.
    """
    def __init__(self, denoiser: nn.Module, sched: "NoiseSchedule",
                 strategies: list[str],         # subset of {'hidden','samples','surprise'}
                 t_ref: int = 1,                # representative low timestep for (a)
                 t_grid: list[int] | None = None,   # timesteps averaged for (c)
                 k_samples: int = 8,            # K futures for (b)
                 num_inference_steps: int = 10, # DDIM steps for (b); keep small (latency)
                 quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                 guidance_scale: float = 1.0,
                 use_inpaint: bool = True,      # (b): inpaint future window vs sample book
                 out_dim: int | None = None) -> None: ...

    @property
    def df_dim(self) -> int:
        """Total width of the concatenated, selected strategies (after the
        optional projection to out_dim)."""

    def forward(self, x0: torch.Tensor, book_window: torch.Tensor | None,
                cond: torch.Tensor | None,
                cond_seq: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """x0:(B,4*depth) observed book; book_window:(B,W,4*depth) past LOB-image
        (needed for inpainting in strategy b); cond/cond_seq: candle conditioning
        from ContextEncoder.candle_conditioning. Returns df:(B, df_dim)."""

    # --- per-strategy helpers (each returns (B, width_s)) ---
    def _feat_hidden(self, x0, cond) -> torch.Tensor:
        """(a) extract_features(denoiser, x0, t_ref, cond, sched) -> hidden,
        then LayerNorm + (optional) Linear."""
    def _feat_samples(self, book_window, cond, cond_seq, generator) -> torch.Tensor:
        """(b) K rollouts: inpaint the future window (use_inpaint) or
        ddim_sample K book states; de-standardize; derive K mid-paths; return
        [mean, var, quantiles] of the terminal return and of the path —
        a fixed-width statistics vector. NO grad through the sampler by default
        (detach); the K rollouts run under torch.no_grad for speed."""
    def _feat_surprise(self, x0, cond) -> torch.Tensor:
        """(c) surprise_score(denoiser, x0, cond, sched, t_grid) -> (B,),
        expanded/normalized to (B,1) (or a small learned embedding)."""
```

**Differentiability note.** Strategies (a) and (c) are cheap and *can* flow
gradients to the denoiser if joint training is enabled. Strategy (b) involves a
multi-step sampler and is treated as a **frozen feature** (run under
`torch.no_grad`, output detached) — it is only meaningful once the diffusion
model is trained, i.e. in the **two-stage** regime or the late phase of joint
training.

### `models/fusion_head.py` — the decision head

```python
class FusionClassifier(nn.Module):
    """MLP over [c ⊕ distribution_features ⊕ current_book_embedding] -> 3 logits.

    Owns: a ContextEncoder, a DiffusionFeatureExtractor, and an OrderbookEncoder
    (project 4) for the current-book embedding. The order-book encoder is
    imported, not redefined:
      from models.orderbook_mlp import OrderbookEncoder   # project 4
    """
    def __init__(self, context_encoder: "ContextEncoder",
                 feature_extractor: "DiffusionFeatureExtractor",
                 book_encoder: "OrderbookEncoder",
                 mlp_hidden_dims: list[int],
                 num_classes: int = 3, dropout: float = 0.1) -> None: ...

    def encode(self, batch: dict) -> torch.Tensor:
        """Returns the fused representation z:(B, fused_dim) =
        concat[ c, df, book_emb ] BEFORE the classification MLP, where
          c        = context_encoder(batch),                       # (B, ctx_dim)
          cond,cs  = context_encoder.candle_conditioning(batch),
          df       = feature_extractor(batch['x0'], batch['book_window'],
                                       cond, cs),                   # (B, df_dim)
          book_emb = book_encoder.encode(batch['x0']).             # (B, emb_dim)
        fused_dim = ctx_dim + df_dim + emb_dim."""

    def forward(self, batch: dict) -> torch.Tensor:
        """encode -> MLP (Linear/LayerNorm/GELU/Dropout blocks) -> (B, 3) logits.
        Class order [DOWN=0, FLAT=1, UP=2]."""
```

The MLP block matches the repo convention (Linear → LayerNorm → GELU → Dropout),
sizes from `head.mlp_hidden_dims`.

---

## Training — two regimes

Two regimes are specified; **two-stage is recommended for stability** (the
diffusion objective and the classification objective have very different scales
and convergence behavior; decoupling them is the safe default and matches
project 7's own stage-1/stage-2 layout).

### Regime 1 — two-stage (RECOMMENDED)

1. **Stage A — diffusion pretraining.** Train the project-7 conditional
   diffusion model exactly as `diffusion_orderbook` specifies (epsilon/v MSE
   with classifier-free-guidance dropout on `c`, AdamW, EMA recommended). In
   practice this stage is run by **calling project 7's trainer** (or loading a
   project-7 `best.pt` / `ema.pt` checkpoint). The denoiser and its candle
   conditioner are produced here.
2. **Stage B — context + head.** **Freeze** the diffusion denoiser (or low-LR
   finetune via `train.diffusion_lr` ≪ `train.lr`). Train the `ContextEncoder`
   and `FusionClassifier` head with **class-weighted CrossEntropy** (weights
   inverse to train class frequency, as in projects 4/7). Strategy (b) features
   are available because the denoiser is already trained. Early stop on val loss.

### Regime 2 — joint multi-task

Single optimizer; total loss

```
L = CE(logits, label) + lambda * diffusion_MSE(eps, eps_theta(x_t, t, c))
```

with `diffusion_MSE` computed on the same batch via project-7's `q_sample` +
denoiser, and CFG dropout still applied to `c`. `lambda` (config
`train.lambda_diffusion`) is annealed (warm-up the diffusion term first). Caveat
documented in the spec: strategy (b) is unreliable early (the sampler is only
meaningful once the denoiser has converged), so under joint training default
`strategies` to **(a) ⊕ (c)** and enable (b) only after a warm-up step count.

### `training/trainer.py` + `scripts/train.py`

```python
def train(cfg: dict) -> dict:
    """Train per cfg['train']['regime'] in {'two_stage','joint'}.

    two_stage: Stage A delegates to diffusion_orderbook.training.train (or loads
    cfg['diffusion']['checkpoint']); Stage B trains ContextEncoder +
    FusionClassifier with class-weighted CrossEntropy, denoiser frozen or
    low-LR. joint: single loop optimizing CE + lambda * diffusion_MSE with
    CFG dropout and lambda warm-up.

    Common: AdamW (lr, weight_decay), early stop on val loss
    (early_stop_patience), device auto mps->cuda->cpu. Artifacts in
    artifacts/run_YYYYMMDD_HHMMSS/: best.pt, config.json, test_metrics.json
    (sklearn.metrics.classification_report(output_dict=True) + confusion
    matrix) — same artifact format as neighboring projects. Returns final
    metrics. Per the research doc, also logs inference TPS/latency when
    strategy (b) sampling is enabled (it dominates cost)."""
```

`scripts/train.py` — argparse CLI: `--config configs/default.yaml`,
`--smoke` (loads `configs/smoke.yaml`), `--regime {two_stage,joint}` (overrides
config). `scripts/eval.py` — load a run, recompute `test_metrics.json`.

---

## Discrete-diffusion variant (forward-looking — Track B)

> **Forward-looking section** — depends on project 7's **Track B** (discrete
> market-token diffusion, LLaDA / MaskGIT style) which is itself sketched, not
> built. Specify now, implement after Track A fusion is validated.
> Cross-ref: [`../docs/research/diffusion-llms.md`](../docs/research/diffusion-llms.md)
> §"Track B — Discrete market-token diffusion" and
> [`../diffusion_orderbook/SPEC.md`](../diffusion_orderbook/SPEC.md) §"Track B".

If the diffusion decoder is a **discrete market-token diffusion** model
(footprint / candle / indicator tokens, generated **in parallel** via masked
unmasking), the "distribution features" change form:

- Append a **masked future block** of market tokens to the observed token
  sequence and **unmask it in a few reveal rounds (4–16)** — all positions in
  parallel (the **fast-TPS** property; see the Gemini-Diffusion data point in the
  research note).
- The distribution features `df` become **the denoised future market tokens
  and/or their per-position logits** (a soft, explicit distribution over the
  masked future). The `FusionClassifier` head reads those directly instead of
  the continuous strategies (a/b/c).
- Module shape: a `DiscreteDiffusionFeatureExtractor` mirroring
  `DiffusionFeatureExtractor`, importing project-7 `discrete/sampler.py`
  (`unmask_future`) and `discrete/mdlm_denoiser.py`. The decision head, context
  encoder, dataset alignment, and two training regimes are otherwise unchanged.
- This is where the **TPS / latency advantage** of the whole project is realized
  at inference; benchmark on **steps × quality × TPS**, not accuracy alone.

This variant is intentionally specified at the contract level only; no further
detail until project 7 Track B exists.

---

## Tests

pytest, **no network, synthetic data only** (tiny tensors / fabricated npz in
`tmp_path`, tiny **stub** denoiser / PriceTransformer / OrderbookEncoder — no
project-1/4/7 weights). Run from the project root with the shared torch venv
(as in `orderbook_encoder` / `diffusion_orderbook` — do not create a new heavy
venv):

`/Users/suenot/projects/w_trading/w_training/token_first_transformer/.venv/bin/python -m pytest tests/ -x -q`

- `test_context_encoder.py` — `ContextEncoder` with candle-only returns
  `c:(B,ctx_dim)`; with candle⊕indicator⊕footprint stub encoders the dim grows
  as expected; `None` modalities fall back to the null vector;
  `candle_conditioning` returns `(cond_vec, cond_seq)` of expected shapes.
- `test_diffusion_features.py` — for **each** strategy and a tiny stub denoiser +
  `NoiseSchedule`: `_feat_hidden` (a), `_feat_samples` (b), `_feat_surprise` (c)
  each return `(B, width_s)`; the combined `forward` returns `(B, df_dim)` and
  `df_dim` matches `strategies`; strategy (b) runs under `no_grad` and its output
  is detached; (a)/(c) produce finite grads when not frozen.
- `test_fusion_head.py` — `FusionClassifier.encode` returns
  `(B, ctx_dim+df_dim+emb_dim)`; `forward` returns `(B,3)` logits; **backward**
  produces finite grads through the trainable parts (denoiser frozen path leaves
  denoiser grads None).
- `test_dataset.py` — `FusionDataset.__getitem__` yields the documented dict;
  labels UP/FLAT/DOWN match a synthetic mid path; windows never cross an npz
  boundary; ts-gap windows dropped; `None` modalities carried through
  `collate_fn`.
- `test_trainer_smoke.py` — a **1-step two-stage smoke** on fabricated npz in
  `tmp_path`: Stage A loads a stub/loaded denoiser, Stage B runs one
  class-weighted CE step (loss finite, head params update, artifacts dir +
  `config.json` created); a 1-step **joint** step likewise produces a finite
  `CE + lambda*MSE`.

---

## Style

As in neighboring projects: short modules, English docstrings, type hints, no
gratuitous abstraction. Intra-project imports relative to the project root
(`from fusion.diffusion_features import DiffusionFeatureExtractor`); cross-project
imports name the neighbor and are kept thin. **Compose, never re-implement:**
reuse `orderbook_encoder`'s pipeline + `OrderbookEncoder`,
`token_first_transformer`'s `PriceTransformer`, and the **entire**
`diffusion_orderbook` denoiser / sampler / conditioner / probe surface. Keep
diffusion **steps small** (strategy b / Track B) and report **TPS/latency**
alongside accuracy, per
[`../docs/research/diffusion-llms.md`](../docs/research/diffusion-llms.md).

## Dependencies

- **Project 1 `token_first_transformer`** (`PriceTransformer` + candle
  tokenizer) — required (candle context + diffusion conditioning).
- **Project 4 `orderbook_encoder`** (npz pipeline + `OrderbookEncoder.encode`)
  — required (data + current-book embedding).
- **Project 7 `diffusion_orderbook`** (`NoiseSchedule`/`q_sample`, denoiser,
  `CandleConditioner`, sampler, probe `extract_features`/`surprise_score`,
  `DiffusionBookDataset`) — required (the diffusion decoder is imported whole).
- **Project 5 `multimodal_encoder`** (indicator embedding stack) — *optional*
  context modality.
- **Project 9 `footprint_encoder`** (footprint embedding; also enables the
  Track B discrete variant) — *optional* context modality / forward-looking.
