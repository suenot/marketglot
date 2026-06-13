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

        # Initialize and fit tokenizers
        self.delta_tokenizer = DeltaTokenizer(range_pct=3.0, step_pct=0.05)

        self.vol_tokenizer = BucketTokenizer(n_bins=8)
        self.vol_tokenizer.fit(self.volume)

        # Volume-body ratio: |close - open| / (high - low + eps)
        vb_raw = np.abs(self.close - self.open) / (self.high - self.low + 1e-10)
        self.vb_tokenizer = BucketTokenizer(n_bins=8)
        self.vb_tokenizer.fit(vb_raw)

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

        # Try loading boundaries from indicator_tokenizer/boundaries/
        boundaries_dir = Path(os.path.join(
            os.path.dirname(__file__), '..', '..', 'indicator_tokenizer', 'boundaries'
        ))
        if boundaries_dir.exists() and (boundaries_dir / "rsi.npy").exists():
            self.indicator_tokenizer.load(boundaries_dir)
        else:
            self.indicator_tokenizer.fit(self.indicators_raw)

        # Tokenize all indicators at once
        self.indicators_tokenized = self.indicator_tokenizer.encode(self.indicators_raw)

        # Tokenize all candles at once
        self.delta_ids_all = self.delta_tokenizer.from_closes(self.close)
        self.vol_ids_all = self.vol_tokenizer.encode_batch(self.volume)
        self.vb_ids_all = self.vb_tokenizer.encode_batch(vb_raw)

    def __len__(self) -> int:
        return len(self.close) - self.seq_len - self.horizon

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
        future_close = self.close[end + self.horizon]
        current_close = self.close[end - 1]
        pct_change = (future_close - current_close) / (current_close + 1e-10)

        if pct_change > self.threshold:
            label = 0  # UP
        elif pct_change < -self.threshold:
            label = 2  # DOWN
        else:
            label = 1  # FLAT

        return delta_ids, vol_ids, vb_ids, ind_dict, torch.tensor(label, dtype=torch.long)
