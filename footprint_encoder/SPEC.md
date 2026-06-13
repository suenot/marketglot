# footprint_encoder — specification (module contract)

> **Project 9 of marketglot — design stage (no implementation yet).**

A **4th market modality**: the **footprint / cluster chart** — volume traded at
each price level per time bar, split by aggressor side (aggressive **BUY** vs
aggressive **SELL** volume). Per bar we build a volume-at-price grid plus the
footprint-native features traders actually read (delta, cumulative delta, POC,
imbalance stacks, value area), encode each bar into an embedding, run a
transformer over the bar sequence, and predict short-horizon price direction.

Consistent with the rest of the repo: **3 classes**, order
`[DOWN=0, FLAT=1, UP=2]`, label from forward mid/price move over a horizon with a
`±threshold_pct` deadband — mirroring `orderbook_encoder`'s labelling.

This SPEC is a design contract only. It defines module boundaries, data schemas,
and Python signatures; **no implementation, models, or tests are written yet.**

---

## Critical data reality (verified 2026-06-13 — stated honestly)

Footprints need *aggressor-signed* volume at price. Our raw data does **not**
provide it directly:

- **Trade ticks** exist on the public S3 (`{SYMBOL}/trades/{YYYY-MM-DD}.parquet`)
  and on the server (`/mnt/second/trender/backtests/data/{SYMBOL}/trades/{date}.parquet`,
  reachable via LAN rsync to `root@192.168.28.72 -p 4242`, or public
  `89.179.247.80:22`). **But the trades schema is only**
  `timestamp_ms:int64, price:float32, qty:float32` — there is **NO aggressor
  side**, and trades are **sparse** (~2k–4k rows/day for XRPUSDT, i.e. heavily
  downsampled). **Footprints therefore CANNOT be built from the trade side
  directly.**
- **L2 deltas** (the `orderbook_encoder` source) are rich — ~500k rows/hr,
  available since **2026-06-01** (see `../docs/data_sources.md`). They are the
  best available signal for aggressor-side attribution.

Consequently the side attribution is **pluggable** (`SideAttributor`), and the
**primary strategy reconstructs order flow from L2 deltas** rather than from
trades. Every strategy below documents its assumptions and known biases — none
is ground truth.

---

## Side attribution strategies (primary first)

`footprint/side.py` defines a strategy interface and four implementations.

1. **`L2FlowAttributor` (PRIMARY) — order flow from L2 deltas.**
   Reuse `orderbook_encoder`'s book reconstruction (snapshot + delta replay).
   Walk deltas in `(event_time, final_update_id)` order; maintain best bid/ask.
   - A **qty decrease at/inside the best ask** → aggressive **BUY** volume
     (someone lifted offers): `buy_vol += max(0, prev_qty − new_qty)` for ask
     levels at or below the prevailing best ask.
   - A **qty decrease at/inside the best bid** → aggressive **SELL** volume
     (someone hit bids): `sell_vol += max(0, prev_qty − new_qty)` for bid levels
     at or above the prevailing best bid.
   - **APPROXIMATION (state this clearly):** a level-qty decrease can be a
     *cancellation*, not a fill. We mitigate by (a) only counting decreases at/
     inside the touch, (b) optionally requiring the best price to move through
     the level (a *sweep* heuristic), and (c) optionally cross-checking against
     the sparse trade prints in the same window. This is **inferred** aggressor
     flow, not exchange-reported aggressor flags.

2. **`QuoteRuleAttributor` — quote rule on sparse trades.**
   For each trade tick, compare its price to the reconstructed best bid/ask at
   that `timestamp_ms`: at/above ask ⇒ BUY, at/below bid ⇒ SELL, strictly
   between ⇒ mid-rule (closer side; tie → tick rule). Useful as a **cross-check**
   for strategy 1; limited because trades are sparse (most volume is invisible).

3. **`TickRuleAttributor` (Lee–Ready fallback) — sign by tick.**
   No book required: sign each trade by up-tick (BUY) / down-tick (SELL); carry
   the previous sign on a zero-tick. Coarsest; used only when no book is
   available for a span.

4. **`SyntheticAttributor` — for the notebook / out-of-the-box runs.**
   Generates plausible signed volume-at-price from a random-walk mid plus a
   configurable imbalance process, so the pipeline and notebook run with **no
   network and no real data**.

Selected via config (`footprint.side.strategy`). Strategies 1–3 may be **layered**:
L2 flow as the base, quote/tick rule to reconcile the sparse trades.

---

## Project structure (mirrors neighbours: orderbook_encoder, token_first_transformer)

```
footprint_encoder/
  pyproject.toml                       # deps: torch, pandas, pyarrow, numpy, pyyaml, scikit-learn
  configs/default.yaml                 # full run config
  configs/smoke.yaml                   # tiny config: 1 day, synthetic fallback
  warehouse/client.py                  # reused/adapted from orderbook_encoder (metadata + atomic download)
  book/book.py                         # REUSED from orderbook_encoder (LocalBook) — book reconstruction
  footprint/side.py                    # SideAttributor interface + 4 strategies
  footprint/builder.py                 # build_footprints(...) -> per-bar grids + native features
  footprint/tokenizer.py              # optional tokenized ("footprint language") representation
  dataset/footprint_dataset.py         # windows of L bars -> (tensor, label); orderbook-style labelling
  models/footprint_model.py            # FootprintEncoder, FootprintClassifier (.encode())
  training/trainer.py                  # train loop, metrics, artifacts
  scripts/download_data.py             # rsync / S3 fetch of trades + orderbook
  scripts/build_samples.py             # raw parquet -> per-day footprint .npz
  scripts/train.py                     # CLI training entrypoint
  tests/                               # synthetic, no network
```

Only stdlib + deps from `pyproject.toml` (numpy, pandas, pyarrow, torch, pyyaml,
scikit-learn). Downloads use `urllib.request` / `subprocess` rsync (no `requests`).
Book reconstruction is **reused from `orderbook_encoder`** (`book/book.py`,
`LocalBook`) rather than reimplemented — `download_data.py` fetches the same
`{HH}_snapshot/{HH}_delta.parquet.zst` files plus the daily `trades` parquet.

---

## Footprint construction

Specified in `footprint/builder.py`.

**Bar spec (configurable).** Default **time bars, 1m**. Future options (note as
not-yet-implemented): **volume bars**, **tick bars**, **range bars** — the bar
boundary function is a strategy seam so these can be added without touching the
grid logic.

**Price binning.** Tick-aligned bins relative to a **per-bar reference**,
configurable: `bar_open` (default), `prev_close`, or `bar_vwap`. `B = 2N+1` bins
centred on the reference; bin width = `bin_ticks × tick_size`. Volume falling
outside `±N` bins is clamped into the edge bins (and an `overflow` flag is set).
Bins are stored as **integer offsets from the reference** so the grid is
translation-invariant across bars.

**Per bin.** `buy_vol`, `sell_vol` ⇒ `delta = buy_vol − sell_vol`.

**Per-bar footprint-native features** (what traders actually read):

- `total_volume` — Σ(buy+sell) over bins.
- `delta` — Σ(buy − sell) over bins (bar delta).
- `cum_delta` — **running** delta across bars (cumulative, reset per session/file).
- `poc_offset` — Price-Of-Control: bin of max total volume, stored as **offset
  from reference**.
- `imbalance_stack` — **diagonal** bid/ask imbalance ratio per bin
  (`bid_vol[p] / ask_vol[p−1]` and the reverse), thresholded into a per-bin
  imbalance flag; counts of stacked buy/sell imbalances.
- `value_area` — `(va_low_offset, va_high_offset)`: smallest contiguous price
  range around the POC holding ~`value_area_pct` (default 70%) of `total_volume`.
- `bar_high_offset`, `bar_low_offset` — high/low as offsets from the reference.
- `unfinished_auction` — flags when the bar high/low bin still has both buy and
  sell volume (no single-sided "finish"), top and/or bottom.

All raw volumes are `log1p`-scaled before they enter tensors; offsets stay integer.

---

## Two encoder representations (both offered)

### (i) Footprint-as-image

Per bar a dense tensor `(B_bins, C)` with channels `{buy_vol, sell_vol[, delta]}`
(`log1p`-scaled), fed to a small CNN/MLP → bar embedding. This is the literal
cluster-chart picture. **Cross-ref:** the "LOB-as-image" / **DiffVolume** line in
`../docs/research/diffusion-llms.md` §2 — volume-at-price grids are exactly the
spatio-temporal image shape diffusion models exploit.

### (ii) Tokenized footprint ("footprint language")

Discretize the **scalar** native features — `delta`, `poc_offset`, per-bin/stacked
`imbalance`, `cum_delta`, `value_area` width — into **per-feature token
vocabularies** (the same idea as `indicator_tokenizer`), embed each, sum/concat →
bar embedding. This variant:

- plugs into `late_fusion_agent` / `multimodal_encoder` / `moe_trading_agent` as a
  4th token stream, and
- **enables discrete diffusion** — it is a token language over footprints, the
  enabler for **Track B** in `../docs/research/diffusion-llms.md` §3 (masked /
  discrete diffusion over market tokens).

A run uses one representation (config `footprint.encoder.kind: image | token`);
both produce a bar embedding of the same `embedding_dim` so the transformer is
shared.

---

## Model

Mirrors `token_first_transformer`'s `PriceTransformer`:

per-bar encoder → **bar embedding** (default 128) → positional encoding →
transformer over **L bars** → **CLS pooling** → MLP → **3 logits**.

---

## Contracts

### warehouse/client.py
Adapted from `orderbook_encoder` (same metadata API + anonymous S3, plus the
daily `trades` parquet). See `../docs/data_sources.md`.

```python
class WarehouseClient:
    def __init__(self, api_base: str, s3_base: str): ...
    def get_orderbook_info(self, symbol: str) -> dict          # exchanges -> {days, first, last}
    def list_days(self, symbol: str, exchange: str) -> list[str]
    def list_files(self, symbol: str, exchange: str, date: str) -> list[dict]
    def download_orderbook_day(self, symbol: str, exchange: str, date: str,
                               dest_root: Path, skip_existing: bool = True) -> list[Path]
    def download_trades_day(self, symbol: str, date: str,
                            dest_root: Path, skip_existing: bool = True) -> Path | None
        # {SYMBOL}/trades/{date}.parquet ; None if absent for that day
```

Atomic download (`.part` → `os.replace`), skip non-empty existing files, retry
5xx/timeout 3× (10/30/60 s) — same policy as the orderbook client.

### book/book.py (reused)
`LocalBook` from `orderbook_encoder` is **imported, not reimplemented**:
`apply_snapshot`, `apply_delta`, `top_levels(depth)`, `mid()`, `is_valid()`.
The `L2FlowAttributor` drives it tick-by-tick and reads `mid()` / best levels.

### footprint/side.py
```python
@dataclass
class SignedTape:
    """Aggressor-signed volume-at-price events within one span."""
    ts_ms: np.ndarray        # int64 (K,)
    price: np.ndarray        # float64 (K,)
    buy_vol: np.ndarray      # float64 (K,)  aggressive buy volume at price
    sell_vol: np.ndarray     # float64 (K,)  aggressive sell volume at price

class SideAttributor(Protocol):
    """Strategy: raw inputs -> aggressor-signed volume-at-price tape."""
    def attribute(self, *, deltas: pd.DataFrame | None,
                  snapshots: pd.DataFrame | None,
                  trades: pd.DataFrame | None,
                  span_start_ms: int, span_end_ms: int) -> SignedTape: ...

class L2FlowAttributor:            # PRIMARY — order flow from L2 deltas (approximate)
    def __init__(self, sweep_only: bool = False,
                 reconcile_trades: bool = False, tick_size: float = ...): ...
    def attribute(self, **kw) -> SignedTape: ...

class QuoteRuleAttributor:         # sparse trades vs reconstructed best bid/ask
    def __init__(self, tick_size: float): ...
    def attribute(self, **kw) -> SignedTape: ...

class TickRuleAttributor:          # Lee-Ready fallback, no book
    def attribute(self, **kw) -> SignedTape: ...

class SyntheticAttributor:         # notebook / offline default
    def __init__(self, seed: int = 0, imbalance: float = 0.1): ...
    def attribute(self, **kw) -> SignedTape: ...

def make_attributor(cfg: dict) -> SideAttributor:
    """Build the attributor named by cfg['footprint']['side']['strategy']."""
```

`L2FlowAttributor.attribute`: replay `deltas` over a `LocalBook` (seeded by the
last `snapshot ≤ span_start_ms`); on each level change, compare to previous qty
and assign the decrease to buy/sell per the rules above; emit one `SignedTape`
event per assignment. **Documented bias:** cancellations inflate volume; mitigate
with `sweep_only` and/or `reconcile_trades`.

### footprint/builder.py
```python
@dataclass
class BarSpec:
    kind: str = "time"          # "time" | "volume" | "tick" | "range" (only "time" v1)
    seconds: int = 60           # for kind="time"
    # volume/tick/range params reserved for future bar kinds

@dataclass
class BinSpec:
    reference: str = "bar_open" # "bar_open" | "prev_close" | "bar_vwap"
    n_bins_side: int = 32       # N -> B = 2N+1 bins
    bin_ticks: int = 1          # bin width in ticks
    tick_size: float = ...

@dataclass
class BarFootprint:
    ts_open_ms: int
    ref_price: float
    grid: np.ndarray            # float32 (B_bins, 2): [buy_vol, sell_vol], log1p-scaled
    features: dict              # native scalars (see below)

def build_footprints(tape: SignedTape, *, bar: BarSpec, bins: BinSpec,
                     prev_close: float | None = None) -> list[BarFootprint]:
    """Aggregate a signed tape into per-bar footprints + native features.

    features keys: total_volume, delta, cum_delta, poc_offset,
    n_buy_imbalance, n_sell_imbalance, va_low_offset, va_high_offset,
    bar_high_offset, bar_low_offset, unfinished_top, unfinished_bottom, overflow.
    cum_delta runs across the returned list (reset at file/session boundary).
    """

def footprints_to_arrays(bars: list[BarFootprint]) -> dict:
    """Stack to a day npz: {'ts': int64 (T,), 'grid': float32 (T, B_bins, 2),
    'feats': float32 (T, F), 'ref': float64 (T,), 'mid': float64 (T,)}.
    'mid' = ref-anchored price path used for labelling (close-of-bar reference price)."""
```

### footprint/tokenizer.py (optional — "footprint language")
```python
class FootprintTokenizer:
    """Per-feature quantile vocabularies over native scalars (indicator_tokenizer style)."""
    def __init__(self, vocab: dict[str, int]): ...           # e.g. {"delta": 9, "poc": 7, "imb": 5, "cum_delta": 9, "va_width": 7}
    def fit(self, feats: np.ndarray, feature_names: list[str]) -> None:
        """Fit quantile boundaries per feature; save to boundaries/*.npy."""
    def encode(self, feats: np.ndarray) -> np.ndarray:        # int64 (T, n_token_features)
    @classmethod
    def load(cls, boundaries_dir: Path) -> "FootprintTokenizer": ...
```

### dataset/footprint_dataset.py
```python
class FootprintDataset(torch.utils.data.Dataset):
    def __init__(self, npz_paths: list[Path], *, seq_len: int,
                 horizon_bars: int, threshold_pct: float,
                 representation: str = "image"):  # "image" | "token"
        ...
    # __getitem__ -> (x, label)
    #   image: x float32 (seq_len, B_bins, 2)   [+ feats appended per bar if configured]
    #   token: x int64   (seq_len, n_token_features)
    #   label int64 in {0,1,2}

def build_splits(cfg: dict) -> tuple[FootprintDataset, FootprintDataset, FootprintDataset]:
    """Train/val/test by cfg['split']['train_days'|'val_days'|'test_days']."""
```

**Label rule (mirror `orderbook_encoder`).** Using the per-bar reference/mid path:
`ret = mid[i + horizon_bars] / mid[i] − 1`; `UP=2` if `ret > threshold_pct/100`,
`DOWN=0` if `ret < −threshold_pct/100`, else `FLAT=1`. Windows of `seq_len` bars
**never cross an npz/file boundary**, and any window spanning a **data gap** (bar
timestamps not contiguous at the expected bar interval) is **dropped**; valid
window indices are precomputed in `__init__`.

### models/footprint_model.py
```python
class FootprintEncoder(nn.Module):
    """Per-bar encoder -> bar embedding -> positional encoding -> transformer -> CLS."""
    def __init__(self, *, representation: str, b_bins: int, in_channels: int,
                 n_token_features: int, vocab: dict[str, int],
                 embedding_dim: int = 128, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, max_seq_len: int = 256): ...
    def forward(self, x) -> Tensor      # (B, embedding_dim)  CLS-pooled sequence embedding

class FootprintClassifier(nn.Module):
    def __init__(self, encoder: FootprintEncoder, num_classes: int = 3): ...
    def forward(self, x) -> Tensor      # (B, num_classes) logits
    def encode(self, x) -> Tensor       # sequence embedding without the head
```

Per-bar encoder is `BarImageEncoder` (small CNN/MLP over `(B_bins, C)`) when
`representation="image"`, or `BarTokenEncoder` (per-feature embeddings summed,
like `multimodal_encoder`'s indicator encoder) when `representation="token"`.
The transformer body mirrors `PriceTransformer` (`token_first_transformer`).

### training/trainer.py + scripts/
```python
def train(cfg: dict) -> dict   # returns final metrics
```
AdamW (`lr`, `weight_decay` from config), class-weighted CrossEntropy (weights
inverse to train-set class frequencies), early stop on val loss
(`early_stop_patience`), device auto (mps → cuda → cpu). Artifacts to
`artifacts/run_YYYYMMDD_HHMMSS/`: `best.pt`, `config.json`, `test_metrics.json`
(`sklearn.metrics.classification_report(..., output_dict=True)` + confusion
matrix) — same format as the neighbouring projects.

CLIs (argparse): `--config configs/default.yaml`, `--dates 2026-06-01:2026-06-09`
(inclusive range) or `--dates 2026-06-09` (single day).
- `download_data.py` — rsync from server (`root@192.168.28.72 -p 4242`, fallback
  public `89.179.247.80:22`) or anonymous S3; fetches orderbook snapshot/delta +
  daily trades.
- `build_samples.py` — raw parquet → per-day footprint npz under
  `{samples_dir}/{symbol}/{exchange}/{date}.npz` (keys
  `ts, grid, feats, ref, mid`); skip existing unless `--force`.
- `train.py` — `train(cfg)`; `--epochs N` overrides config.

---

## Tests

pytest, **no network, synthetic data**. Suggested run (reuse the shared venv with
torch, as the orderbook project does — do **not** create a new heavy venv):
`/Users/suenot/projects/w_trading/w_training/token_first_transformer/.venv/bin/python -m pytest tests/ -x -q`.

- `test_side.py` — each `SideAttributor` on synthetic inputs: `L2FlowAttributor`
  attributes an ask-side qty decrease to BUY and a bid-side decrease to SELL and
  ignores far-from-touch cancels under `sweep_only`; `QuoteRuleAttributor` signs
  trades at/above ask as BUY, at/below bid as SELL, mid otherwise;
  `TickRuleAttributor` signs by up/down tick; `SyntheticAttributor` is
  deterministic for a fixed seed.
- `test_builder.py` — synthetic `SignedTape` → expected `grid`, `delta`,
  `cum_delta` accumulation, `poc_offset` at the max-volume bin, `value_area`
  covering ~70%, `unfinished_*` and `overflow` flags; binning relative to each
  `reference` option.
- `test_tokenizer.py` — `fit`/`encode` produce in-range token ids; round-trip via
  `load`; per-feature vocab sizes respected.
- `test_dataset.py` — UP/FLAT/DOWN labels from a synthetic `mid` path; windows do
  not cross file boundaries; windows spanning a bar-timestamp gap are dropped;
  `image` and `token` `__getitem__` shapes.
- `test_model.py` — `forward`/`encode` shapes for both representations; backward
  pass runs.
- `test_trainer.py` — 1 epoch on tiny synthetic npz in `tmp_path`; artifacts
  (`best.pt`, `config.json`, `test_metrics.json`) created.
- `test_client.py` — URL/path construction and `skip_existing` for orderbook and
  trades; download mocked via `monkeypatch` (no network).

---

## Synergies

- **4th modality** for `late_fusion_agent` (project 3), `multimodal_encoder`
  (project 5), and `moe_trading_agent` (project 6): the footprint bar embedding
  (image variant) or footprint token stream (token variant) joins candle and
  indicator streams. `FootprintClassifier.encode()` / the token stream are the
  integration points, matching how those projects already consume per-modality
  encoders.
- **Discrete diffusion:** the **tokenized footprint** is a token language over
  microstructure, feeding **Track B** of `../docs/research/diffusion-llms.md`
  (masked/discrete diffusion forecasting a block of future market tokens). The
  **image** variant aligns with **DiffVolume** / "LOB-as-image" (Track A / §2).

---

## Style

Like the neighbouring projects: short modules, **English docstrings**, type hints,
minimal dependencies, no unnecessary abstraction. Intra-project imports are
relative to the project root (`from footprint.builder import build_footprints`),
and `LocalBook` is imported from the `orderbook_encoder` reconstruction rather
than duplicated. The grid/native-feature math is `numpy`; only the model and
dataset touch `torch`.
