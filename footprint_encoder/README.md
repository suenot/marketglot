# footprint_encoder

A 4th market modality: encode the **footprint / cluster chart** (aggressor-signed
volume-at-price per time bar) into a per-bar embedding, run a transformer over the
bar sequence, and predict short-horizon price direction (DOWN/FLAT/UP).

## Idea

Footprints show *who is aggressing* at each price level — the delta, cumulative
delta, POC, imbalance stacks and value area that order-flow traders actually read,
which candles and even the raw L2 book flatten away. Our trades feed has **no
aggressor side and is sparse**, so the side is **inferred** by a pluggable
`SideAttributor` whose primary strategy reconstructs order flow from rich L2
deltas (an approximation — see SPEC). Two encoder variants are offered: footprint-
as-image (CNN/MLP) and a tokenized "footprint language" that feeds late-fusion /
MoE and the discrete-diffusion track.

Status: design stage — see [SPEC.md](SPEC.md).

Part of the [marketglot](../README.md) monorepo.
