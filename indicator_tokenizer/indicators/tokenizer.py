from __future__ import annotations

from pathlib import Path

import numpy as np


class FixedBoundaries:
    """Encodes values into bins defined by fixed threshold array."""

    def __init__(self, bins: list[float], offset: int = 2) -> None:
        self.bins = np.array(bins, dtype=np.float32)
        self.offset = offset
        self.vocab_size = len(bins) + 1 + offset

    def encode(self, value: float) -> int:
        return int(np.searchsorted(self.bins, value, side="right")) + self.offset

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        return (np.searchsorted(self.bins, values, side="right") + self.offset).astype(np.int32)

    def save(self, path: Path) -> None:
        np.save(path, self.bins)

    def load(self, path: Path) -> None:
        self.bins = np.load(path)


class QuantileBoundaries:
    """Encodes values into quantile-based bins (fitted on data)."""

    def __init__(self, n_bins: int, offset: int = 2) -> None:
        self.n_bins = n_bins
        self.offset = offset
        self.vocab_size = n_bins + offset
        self.boundaries: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.boundaries = np.percentile(values, quantiles).astype(np.float32)

    def encode(self, value: float) -> int:
        assert self.boundaries is not None
        return int(np.searchsorted(self.boundaries, value, side="right")) + self.offset

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None
        return (np.searchsorted(self.boundaries, values, side="right") + self.offset).astype(np.int32)

    def save(self, path: Path) -> None:
        assert self.boundaries is not None
        np.save(path, self.boundaries)

    def load(self, path: Path) -> None:
        self.boundaries = np.load(path)


class IndicatorTokenizer:
    """Tokenizes all indicators using per-indicator boundary definitions."""

    PAD_ID = 0
    SPECIAL_ID = 1

    def __init__(self) -> None:
        self.rsi = FixedBoundaries(bins=[20, 30, 70, 80])
        self.macd_hist = QuantileBoundaries(n_bins=7)
        self.bollinger_pctb = FixedBoundaries(bins=[0.0, 0.25, 0.75, 1.0])
        self.atr = QuantileBoundaries(n_bins=6)
        self.volume_ratio = QuantileBoundaries(n_bins=5)
        self.price_vs_sma = QuantileBoundaries(n_bins=5)

        self._quantile_fields = ["macd_hist", "atr", "volume_ratio", "price_vs_sma"]
        self._all_fields = ["rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"]

    def fit(self, indicators: dict[str, np.ndarray]) -> None:
        for field in self._quantile_fields:
            getattr(self, field).fit(indicators[field])

    def encode(self, indicators: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {field: getattr(self, field).encode_batch(indicators[field]) for field in self._all_fields}

    def vocab_sizes(self) -> dict[str, int]:
        return {field: getattr(self, field).vocab_size for field in self._all_fields}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for field in self._all_fields:
            getattr(self, field).save(directory / f"{field}.npy")

    def load(self, directory: Path) -> None:
        for field in self._all_fields:
            getattr(self, field).load(directory / f"{field}.npy")
