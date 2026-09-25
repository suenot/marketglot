from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Add sibling projects to path for tokenizer imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'token_first_transformer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'indicator_tokenizer'))

from tokenizer.delta_tokenizer import DeltaTokenizer
from tokenizer.bucket_tokenizer import BucketTokenizer
from indicators.computer import IndicatorComputer
from indicators.tokenizer import IndicatorTokenizer


class MoEDataset(Dataset):
    """Mixture of Experts dataset for 3-class trading prediction.

    Loads parquet candle data, tokenizes using sibling project tokenizers,
    computes and tokenizes indicators, and produces 3-class labels.
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        seq_len: int = 128,
        horizon: int = 60,
        threshold: float = 0.0015,
        train_fraction: float = 1.0,
        tokenizer_dir: str | Path | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.horizon = horizon
        self.threshold = threshold

        # Load and concatenate all parquet files
        dfs = []
        for fp in file_paths:
            df = pd.read_parquet(fp)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)

        # Store as numpy arrays
        self.open = data["open"].values.astype(np.float32)
        self.high = data["high"].values.astype(np.float32)
        self.low = data["low"].values.astype(np.float32)
        self.close = data["close"].values.astype(np.float32)
        self.volume = data["volume"].values.astype(np.float32)
        self.data = data  # Keep reference for indicator computation
        self.train_row_end = int(len(data) * train_fraction)
        if not 0 < self.train_row_end <= len(data):
            raise ValueError("train_fraction must leave a nonempty training period")

        # Initialize and fit tokenizers
        self.delta_tokenizer = DeltaTokenizer(range_pct=3.0, step_pct=0.05)

        self.vol_tokenizer = BucketTokenizer(n_bins=8)

        # Volume-body ratio: |close - open| / (high - low + eps)
        vb_raw = np.abs(self.close - self.open) / (self.high - self.low + 1e-10)
        self.vb_tokenizer = BucketTokenizer(n_bins=8)

        # Compute indicators on full data
        self.indicator_computer = IndicatorComputer()
        ohlcv = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        self.indicators_raw = self.indicator_computer.compute_all(ohlcv)

        # Initialize indicator tokenizer
        self.indicator_tokenizer = IndicatorTokenizer()

        if tokenizer_dir is not None:
            boundaries_dir = Path(tokenizer_dir)
            self.vol_tokenizer.load(boundaries_dir / "volume.npy")
            self.vb_tokenizer.load(boundaries_dir / "body_ratio.npy")
            self.indicator_tokenizer.load(boundaries_dir / "indicators")
        else:
            self.vol_tokenizer.fit(self.volume[:self.train_row_end])
            self.vb_tokenizer.fit(vb_raw[:self.train_row_end])
            self.indicator_tokenizer.fit({k: v[:self.train_row_end]
                                          for k, v in self.indicators_raw.items()})

        # Tokenize all indicators at once
        self.indicators_tokenized = self.indicator_tokenizer.encode(self.indicators_raw)

        # Tokenize all candles at once
        self.delta_ids_all = self.delta_tokenizer.from_closes(self.close)
        self.vol_ids_all = self.vol_tokenizer.encode_batch(self.volume)
        self.vb_ids_all = self.vb_tokenizer.encode_batch(vb_raw)

    def __len__(self) -> int:
        return max(0, len(self.close) - self.seq_len - self.horizon + 1)

    def split_ranges(self, val_fraction: float) -> tuple[range, range, range]:
        """Disjoint raw-row periods, including each sample's future target."""
        val_row_end = int(len(self.close) * (self.train_row_end / len(self.close) + val_fraction))
        width = self.seq_len + self.horizon
        if (self.train_row_end < width or val_row_end - self.train_row_end < width
                or len(self.close) - val_row_end < width):
            raise ValueError("train, validation and test periods must each contain a sample")
        return (range(0, self.train_row_end - width + 1),
                range(self.train_row_end, val_row_end - width + 1),
                range(val_row_end, len(self)))

    def save_tokenizers(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vol_tokenizer.save(directory / "volume.npy")
        self.vb_tokenizer.save(directory / "body_ratio.npy")
        self.indicator_tokenizer.save(directory / "indicators")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                               dict[str, torch.Tensor], torch.Tensor]:
        """Returns (delta_ids, vol_ids, vb_ids, ind_dict, label)."""
        start = idx
        end = idx + self.seq_len

        # Candle tokens
        delta_ids = torch.tensor(self.delta_ids_all[start:end], dtype=torch.long)
        vol_ids = torch.tensor(self.vol_ids_all[start:end], dtype=torch.long)
        vb_ids = torch.tensor(self.vb_ids_all[start:end], dtype=torch.long)

        # Indicator tokens
        ind_dict = {
            key: torch.tensor(self.indicators_tokenized[key][start:end], dtype=torch.long)
            for key in self.indicators_tokenized
        }

        # Label: 3-class based on future price movement
        future_close = self.close[end + self.horizon - 1]
        current_close = self.close[end - 1]
        pct_change = (future_close - current_close) / (current_close + 1e-10)

        if pct_change > self.threshold:
            label = 2  # UP
        elif pct_change < -self.threshold:
            label = 0  # DOWN
        else:
            label = 1  # FLAT

        return delta_ids, vol_ids, vb_ids, ind_dict, torch.tensor(label, dtype=torch.long)
