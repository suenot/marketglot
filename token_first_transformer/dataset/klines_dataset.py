from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tokenizer.delta_tokenizer import DeltaTokenizer
from tokenizer.bucket_tokenizer import BucketTokenizer


def _load_month(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {col: table[col].to_numpy() for col in table.column_names}


def fit_tokenizers(
    file_paths: list[Path],
    range_pct: float = 3.0,
    step_pct: float = 0.05,
    n_bins: int = 8,
) -> tuple[DeltaTokenizer, BucketTokenizer, BucketTokenizer]:
    delta_tok = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    all_range_pct = []
    all_log_vol = []
    for p in file_paths:
        d = _load_month(p)
        closes = d["close"]
        if len(closes) < 2:
            continue
        highs, lows = d["high"], d["low"]
        ranges = (highs[1:] - lows[1:]) / closes[1:]
        all_range_pct.append(ranges)
        log_vols = np.log1p(d["volume"][1:])
        all_log_vol.append(log_vols)
    range_arr = np.concatenate(all_range_pct)
    vol_arr = np.concatenate(all_log_vol)
    vol_tok = BucketTokenizer(n_bins=n_bins)
    vol_tok.fit(range_arr)
    vb_tok = BucketTokenizer(n_bins=n_bins)
    vb_tok.fit(vol_arr)
    return delta_tok, vol_tok, vb_tok


def make_split(data_dir: Path, start_month: str, end_month: str) -> list[Path]:
    klines_dir = data_dir / "BTCUSDT" / "klines_1m"
    if not klines_dir.exists():
        raise FileNotFoundError(f"No klines_1m directory at {klines_dir}")
    files = sorted(klines_dir.glob("*.parquet"))
    return [f for f in files if start_month <= f.stem <= end_month]


class KlinesDataset:
    def __init__(
        self,
        file_paths: list[Path],
        seq_len: int = 128,
        target_horizon: int = 60,
        target_threshold: float = 0.0015,
        range_pct: float = 3.0,
        step_pct: float = 0.05,
        n_bins: int = 8,
    ) -> None:
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold

        self.delta_tok, self.vol_tok, self.vb_tok = fit_tokenizers(
            file_paths, range_pct, step_pct, n_bins
        )
        self._load_data(file_paths)

    def _load_data(self, file_paths: list[Path]) -> None:
        frames = [_load_month(p) for p in file_paths]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        n = len(self.closes)
        self._len = max(0, n - self.seq_len - self.target_horizon)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        start = idx
        end = start + self.seq_len
        closes = self.closes[start:end]
        highs = self.highs[start:end]
        lows = self.lows[start:end]
        vols = self.volumes[start:end]

        delta_ids = self.delta_tok.from_closes(closes)
        delta_ids[0] = self.delta_tok.cls_id

        range_pct = np.zeros(self.seq_len, dtype=np.float32)
        range_pct[1:] = (highs[1:] - lows[1:]) / closes[1:]
        vol_ids = self.vol_tok.encode_batch(range_pct)
        vol_ids[0] = self.vol_tok.pad_id

        log_vol = np.log1p(vols)
        vb_ids = self.vb_tok.encode_batch(log_vol)
        vb_ids[0] = self.vb_tok.pad_id

        target_close = self.closes[end + self.target_horizon - 1]
        current_close = self.closes[end - 1]
        delta = (target_close - current_close) / current_close

        if delta > self.target_threshold:
            label = 2
        elif delta < -self.target_threshold:
            label = 0
        else:
            label = 1

        return delta_ids, vol_ids, vb_ids, label
