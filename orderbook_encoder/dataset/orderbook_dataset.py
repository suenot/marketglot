"""Dataset over sampled order book npz files.

Each npz holds one day of 1 Hz samples with keys:
    ts:       int64 (T,)  millisecond timestamps
    features: float32 (T, 4*depth)
    mid:      float64 (T,)  mid-price

Labels predict the mid-price move over ``horizon_sec``:
    ret = mid[i+h] / mid[i] - 1
    UP(2)   if ret >  threshold_pct/100
    DOWN(0) if ret < -threshold_pct/100
    FLAT(1) otherwise

Windows never cross npz-file boundaries. A pair (i, i+h) whose timestamp gap
deviates from ``horizon_sec*1000`` by more than that amount (a hole in the data)
is dropped. The list of valid (file, i) pairs is built once in ``__init__``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DOWN, FLAT, UP = 0, 1, 2


class OrderbookDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[Path],
        horizon_sec: float,
        threshold_pct: float,
        interval_sec: float,
    ) -> None:
        self.horizon_sec = horizon_sec
        self.threshold_pct = threshold_pct
        self.interval_sec = interval_sec
        self.h = max(1, round(horizon_sec / interval_sec))

        self._features: list[np.ndarray] = []
        self._mid: list[np.ndarray] = []
        # (file_index, i): the row i and its partner i+h are both valid.
        self._pairs: list[tuple[int, int]] = []

        gap_ms = horizon_sec * 1000.0
        for path in npz_paths:
            with np.load(path) as data:
                ts = data["ts"].astype(np.int64)
                feats = data["features"].astype(np.float32)
                mid = data["mid"].astype(np.float64)
            file_idx = len(self._features)
            self._features.append(feats)
            self._mid.append(mid)

            n = len(ts)
            for i in range(n - self.h):
                j = i + self.h
                if abs((ts[j] - ts[i]) - gap_ms) > gap_ms:
                    continue
                self._pairs.append((file_idx, i))

    def __len__(self) -> int:
        return len(self._pairs)

    def _label(self, mid: np.ndarray, i: int, j: int) -> int:
        ret = mid[j] / mid[i] - 1.0
        thr = self.threshold_pct / 100.0
        if ret > thr:
            return UP
        if ret < -thr:
            return DOWN
        return FLAT

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, i = self._pairs[idx]
        j = i + self.h
        feats = self._features[file_idx][i]
        label = self._label(self._mid[file_idx], i, j)
        x = torch.from_numpy(np.ascontiguousarray(feats)).to(torch.float32)
        y = torch.tensor(label, dtype=torch.int64)
        return x, y


def _split_paths(cfg: dict, days: list[str]) -> list[Path]:
    data = cfg["data"]
    samples_dir = Path(data["samples_dir"])
    symbol = data["symbol"]
    exchange = data["exchange"]
    paths: list[Path] = []
    for date in days:
        p = samples_dir / symbol / exchange / f"{date}.npz"
        if p.exists():
            paths.append(p)
        else:
            print(f"Warning: missing samples file, skipping: {p}")
    return paths


def build_splits(
    cfg: dict,
) -> tuple[OrderbookDataset, OrderbookDataset, OrderbookDataset]:
    """Build train/val/test datasets from date lists in ``cfg['split']``."""
    target = cfg["target"]
    interval_sec = cfg["sampling"]["interval_sec"]
    split = cfg["split"]

    def make(days: list[str]) -> OrderbookDataset:
        return OrderbookDataset(
            npz_paths=_split_paths(cfg, days),
            horizon_sec=target["horizon_sec"],
            threshold_pct=target["threshold_pct"],
            interval_sec=interval_sec,
        )

    return make(split["train_days"]), make(split["val_days"]), make(split["test_days"])
