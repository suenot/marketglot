# Changelog

All notable changes to the marketglot research projects are recorded here.

## Unreleased

### Changed

- Document the first resumable BTCUSDT 1m training run and its checkpoint location.

## 0.1.3 - 2026-09-25

### Fixed

- Seed token-first model initialization and random streams for comparable runs.
- Calculate backtest profit factor from compounded monetary trade results.

## 0.1.2 - 2026-09-25

### Fixed

- Use CUDA for token-first evaluation and backtests and multimodal training and evaluation when `device: auto` runs on an NVIDIA machine.

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
