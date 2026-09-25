# Changelog

All notable changes to the marketglot research projects are recorded here.

## Unreleased

## 0.1.1 - 2026-09-25

### Fixed

- Reuse training-fitted tokenizers for validation, evaluation, and backtests.
- Correct chronological MoE splits and the shared DOWN/FLAT/UP label order.
- Align token-first predictions with their source bars and calculate trading
  results from an equity curve with next-open fills and transaction costs.
- Exclude candle windows that cross missing or duplicated minutes.
- Restore the best base-model weights before fitting the late-fusion meta-model.
- Reject overlapping non-smoke order-book splits and labels stretched by gaps.
- Save atomic, resumable token-first checkpoints during an epoch with model,
  optimizer, scheduler, and training cursor state.
