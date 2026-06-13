from __future__ import annotations

from pathlib import Path

import numpy as np


class BucketTokenizer:
    """Quantile-based bucket tokenizer for volatility / volume features.

    Vocabulary layout:
        0 = PAD
        1 = CLS (reserved)
        2 .. n_bins+1 = quantile bins
    """

    def __init__(self, n_bins: int = 8) -> None:
        self.n_bins = n_bins
        self.pad_id = 0
        self._offset = 2
        self.vocab_size = self._offset + n_bins
        self.boundaries: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.boundaries = np.percentile(values, quantiles).astype(np.float32)

    def encode_single(self, value: float) -> int:
        assert self.boundaries is not None, "Call fit() first"
        return int(np.searchsorted(self.boundaries, value, side="right")) + self._offset

    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None, "Call fit() first"
        bin_idx = np.searchsorted(self.boundaries, values, side="right")
        return (bin_idx + self._offset).astype(np.int32)

    def save(self, path: Path) -> None:
        assert self.boundaries is not None
        np.save(path, self.boundaries)

    def load(self, path: Path) -> None:
        self.boundaries = np.load(path)
