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


def save_tokenizers(tokenizers: tuple[DeltaTokenizer, BucketTokenizer, BucketTokenizer],
                    directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tokenizers[1].save(directory / "volatility.npy")
    tokenizers[2].save(directory / "volume.npy")


def load_tokenizers(directory: Path, range_pct: float = 3.0,
                    step_pct: float = 0.05, n_bins: int = 8
                    ) -> tuple[DeltaTokenizer, BucketTokenizer, BucketTokenizer]:
    delta_tok = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    vol_tok, vb_tok = BucketTokenizer(n_bins=n_bins), BucketTokenizer(n_bins=n_bins)
    vol_tok.load(directory / "volatility.npy")
    vb_tok.load(directory / "volume.npy")
    return delta_tok, vol_tok, vb_tok


def make_split(data_dir: Path, start_month: str, end_month: str,
               symbol: str = "BTCUSDT") -> list[Path]:
    klines_dir = data_dir / symbol / "klines_1m"
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
        tokenizers: tuple[DeltaTokenizer, BucketTokenizer, BucketTokenizer] | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold

        self.delta_tok, self.vol_tok, self.vb_tok = tokenizers or fit_tokenizers(
            file_paths, range_pct, step_pct, n_bins
        )
        self._load_data(file_paths)

    def _load_data(self, file_paths: list[Path]) -> None:
        frames = [_load_month(p) for p in file_paths]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        self.timestamps = np.concatenate([f["timestamp"] for f in frames]).astype(np.int64)
        n = len(self.closes)
        window = self.seq_len + self.target_horizon
        candidate_starts = np.arange(max(0, n - window + 1), dtype=np.int64)
        # A missing or duplicated minute must not turn a 60-minute target into
        # an arbitrary time horizon or bridge disconnected market periods.
        invalid_prefix = np.concatenate(([0], np.cumsum(np.diff(self.timestamps) != 60)))
        valid = invalid_prefix[candidate_starts + window - 1] == invalid_prefix[candidate_starts]
        self.sample_starts = candidate_starts[valid]
        self._len = len(self.sample_starts)

    def __len__(self) -> int:
        return self._len

    def labels(self) -> np.ndarray:
        """Return labels for exactly the timestamp-valid training windows."""
        ends = self.sample_starts + self.seq_len
        current_close = self.closes[ends - 1]
        target_close = self.closes[ends + self.target_horizon - 1]
        delta = (target_close - current_close) / current_close
        return np.where(delta > self.target_threshold, 2,
                        np.where(delta < -self.target_threshold, 0, 1))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        start = int(self.sample_starts[idx])
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
