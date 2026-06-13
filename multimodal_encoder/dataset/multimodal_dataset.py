from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_ROOT / "token_first_transformer"))
sys.path.append(str(_ROOT / "indicator_tokenizer"))

from tokenizer.delta_tokenizer import DeltaTokenizer
from tokenizer.bucket_tokenizer import BucketTokenizer
from indicators.computer import IndicatorComputer
from indicators.tokenizer import IndicatorTokenizer


def _load(path: Path) -> dict[str, np.ndarray]:
    t = pq.read_table(path)
    return {c: np.array([v.as_py() for v in t.column(c)], dtype=np.float32) for c in t.column_names}


def _fit_all(files, range_pct, step_pct, n_bins):
    dt = DeltaTokenizer(range_pct=range_pct, step_pct=step_pct)
    rp, lv = [], []
    for f in files:
        d = _load(f)
        if len(d["close"]) < 2: continue
        rp.append((d["high"][1:] - d["low"][1:]) / d["close"][1:])
        lv.append(np.log1p(d["volume"][1:]))
    vt = BucketTokenizer(n_bins=n_bins); vt.fit(np.concatenate(rp))
    bt = BucketTokenizer(n_bins=n_bins); bt.fit(np.concatenate(lv))
    comp = IndicatorComputer()
    keys = ["rsi","macd_hist","bollinger_pctb","atr","volume_ratio","price_vs_sma"]
    ai = {k: [] for k in keys}
    for f in files:
        d = _load(f)
        ind = comp.compute_all({k2: d[k2] for k2 in ["open","high","low","close","volume"]})
        for k in keys: ai[k].append(ind[k])
    it = IndicatorTokenizer(); it.fit({k: np.concatenate(v) for k, v in ai.items()})
    return dt, vt, bt, it, comp


class MultimodalDataset:
    def __init__(self, files, seq_len=128, target_horizon=60, target_threshold=0.0015,
                 range_pct=3.0, step_pct=0.05, n_bins=8):
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold
        self.dt, self.vt, self.bt, self.it, self.comp = _fit_all(files, range_pct, step_pct, n_bins)
        frames = [_load(f) for f in files]
        self.closes = np.concatenate([f["close"] for f in frames]).astype(np.float32)
        self.highs = np.concatenate([f["high"] for f in frames]).astype(np.float32)
        self.lows = np.concatenate([f["low"] for f in frames]).astype(np.float32)
        self.volumes = np.concatenate([f["volume"] for f in frames]).astype(np.float32)
        self.opens = np.concatenate([f.get("open", f["close"]) for f in frames]).astype(np.float32)
        self._len = max(0, len(self.closes) - seq_len - target_horizon)

    def __len__(self): return self._len

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        c, h, l, v, o = self.closes[s:e], self.highs[s:e], self.lows[s:e], self.volumes[s:e], self.opens[s:e]

        delta = self.dt.from_closes(c); delta[0] = self.dt.cls_id
        rp = np.zeros(self.seq_len, dtype=np.float32)
        rp[1:] = (h[1:] - l[1:]) / c[1:]
        vol = self.vt.encode_batch(rp); vol[0] = self.vt.pad_id
        vb = self.bt.encode_batch(np.log1p(v)); vb[0] = self.bt.pad_id

        ind = self.it.encode(self.comp.compute_all({"open":o,"high":h,"low":l,"close":c,"volume":v}))

        tc = self.closes[e + self.target_horizon - 1]
        d = (tc - self.closes[e-1]) / self.closes[e-1]
        label = 2 if d > self.target_threshold else (0 if d < -self.target_threshold else 1)
        return delta, vol, vb, ind, label
