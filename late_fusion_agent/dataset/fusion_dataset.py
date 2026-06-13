from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT / "token_first_transformer"), str(_PROJECT_ROOT / "indicator_tokenizer")]:
    if _p not in sys.path:
        sys.path.append(_p)

from tokenizer.delta_tokenizer import DeltaTokenizer
from tokenizer.bucket_tokenizer import BucketTokenizer
from indicators.computer import IndicatorComputer
from indicators.tokenizer import IndicatorTokenizer


def _load_month(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {col: np.array([v.as_py() for v in table.column(col)], dtype=np.float32)
            for col in table.column_names}


def _fit_all(file_paths: list[Path], range_pct: float, step_pct: float, n_bins: int):
    delta_tok = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    all_range_pct, all_log_vol = [], []
    for p in file_paths:
        d = _load_month(p)
        if len(d["close"]) < 2:
            continue
        all_range_pct.append((d["high"][1:] - d["low"][1:]) / d["close"][1:])
        all_log_vol.append(np.log1p(d["volume"][1:]))
    vol_tok = BucketTokenizer(n_bins=n_bins)
    vol_tok.fit(np.concatenate(all_range_pct))
    vb_tok = BucketTokenizer(n_bins=n_bins)
    vb_tok.fit(np.concatenate(all_log_vol))

    comp = IndicatorComputer()
    ind_keys = ["rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"]
    all_ind = {k: [] for k in ind_keys}
    for p in file_paths:
        d = _load_month(p)
        ohlcv = {k2: d[k2] for k2 in ["open", "high", "low", "close", "volume"]}
        indicators = comp.compute_all(ohlcv)
        for k in ind_keys:
            all_ind[k].append(indicators[k])
    combined = {k: np.concatenate(v) for k, v in all_ind.items()}
    ind_tok = IndicatorTokenizer()
    ind_tok.fit(combined)

    return delta_tok, vol_tok, vb_tok, ind_tok, comp


class FusionDataset:
    def __init__(self, file_paths: list[Path], seq_len: int = 128,
                 target_horizon: int = 60, target_threshold: float = 0.0015,
                 range_pct: float = 3.0, step_pct: float = 0.05, n_bins: int = 8):
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold
        self.delta_tok, self.vol_tok, self.vb_tok, self.ind_tok, self.comp = _fit_all(
            file_paths, range_pct, step_pct, n_bins
        )
        self._load_data(file_paths)

    def _load_data(self, file_paths):
        frames = [_load_month(p) for p in file_paths]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        self.opens = np.concatenate([f.get("open", f["close"]) for f in frames]).astype(np.float32)
        self._len = max(0, len(self.closes) - self.seq_len - self.target_horizon)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start, end = idx, idx + self.seq_len
        closes = self.closes[start:end]
        highs = self.highs[start:end]
        lows = self.lows[start:end]
        vols = self.volumes[start:end]
        opens = self.opens[start:end]

        # Candle tokens
        delta_ids = self.delta_tok.from_closes(closes)
        delta_ids[0] = self.delta_tok.cls_id
        range_pct = np.zeros(self.seq_len, dtype=np.float32)
        range_pct[1:] = (highs[1:] - lows[1:]) / closes[1:]
        vol_ids = self.vol_tok.encode_batch(range_pct)
        vol_ids[0] = self.vol_tok.pad_id
        vb_ids = self.vb_tok.encode_batch(np.log1p(vols))
        vb_ids[0] = self.vb_tok.pad_id

        # Indicator tokens
        ohlcv = {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}
        raw = self.comp.compute_all(ohlcv)
        ind_tokens = self.ind_tok.encode(raw)

        # Target
        target_close = self.closes[end + self.target_horizon - 1]
        current_close = self.closes[end - 1]
        delta = (target_close - current_close) / current_close
        label = 2 if delta > self.target_threshold else (0 if delta < -self.target_threshold else 1)

        return delta_ids, vol_ids, vb_ids, ind_tokens, label
