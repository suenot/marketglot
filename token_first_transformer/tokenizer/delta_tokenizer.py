from __future__ import annotations

import numpy as np


class DeltaTokenizer:
    """Tokenizes percentage price deltas into discrete bins.

    Vocabulary layout:
        0 = PAD
        1 = CLS
        2 .. vocab_size-1 = quantized delta bins
    """

    def __init__(self, range_pct: float = 3.0, step_pct: float = 0.05) -> None:
        self.range_pct = range_pct
        self.step_pct = step_pct
        self.n_bins = int(2 * range_pct / step_pct)
        self.pad_id = 0
        self.cls_id = 1
        self._offset = 2
        self.vocab_size = self._offset + self.n_bins

    def encode_single(self, delta: float) -> int:
        delta_pct = delta * 100
        bin_idx = round(delta_pct / self.step_pct) + self.n_bins // 2 - 1
        bin_idx = max(0, min(self.n_bins - 1, bin_idx))
        return self._offset + bin_idx

    def encode_batch(self, deltas: np.ndarray) -> np.ndarray:
        deltas_pct = deltas * 100
        bin_idx = np.round(deltas_pct / self.step_pct).astype(np.int32) + self.n_bins // 2 - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
        return (bin_idx + self._offset).astype(np.int32)

    def from_closes(self, closes: np.ndarray) -> np.ndarray:
        n = len(closes)
        ids = np.full(n, self.pad_id, dtype=np.int32)
        if n < 2:
            return ids
        deltas = (closes[1:] - closes[:-1]) / closes[:-1]
        ids[1:] = self.encode_batch(deltas)
        return ids
