from __future__ import annotations

import numpy as np


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.zeros(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


class IndicatorComputer:
    """Computes technical indicators from OHLCV numpy arrays."""

    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close.astype(np.float64), prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = _ema(gain, period)
        avg_loss = _ema(loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        return (100 - 100 / (1 + rs)).astype(np.float32)

    def macd_hist(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9,
    ) -> np.ndarray:
        c = close.astype(np.float64)
        macd_line = _ema(c, fast) - _ema(c, slow)
        signal_line = _ema(macd_line, signal)
        return (macd_line - signal_line).astype(np.float32)

    def bollinger_pctb(
        self, close: np.ndarray, period: int = 20, num_std: float = 2.0,
    ) -> np.ndarray:
        c = close.astype(np.float64)
        out = np.zeros(len(c), dtype=np.float32)
        for i in range(period - 1, len(c)):
            window = c[i - period + 1 : i + 1]
            sma = window.mean()
            std = window.std()
            upper = sma + num_std * std
            lower = sma - num_std * std
            out[i] = (c[i] - lower) / (upper - lower + 1e-10)
        return out

    def atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14,
    ) -> np.ndarray:
        tr = np.zeros(len(close), dtype=np.float64)
        tr[0] = high[0] - low[0]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:].astype(np.float64) - close[:-1].astype(np.float64)),
                np.abs(low[1:].astype(np.float64) - close[:-1].astype(np.float64)),
            ),
        )
        return _ema(tr, period).astype(np.float32)

    def volume_ratio(
        self, close: np.ndarray, volume: np.ndarray, period: int = 20,
    ) -> np.ndarray:
        v = volume.astype(np.float64)
        out = np.zeros(len(v), dtype=np.float32)
        for i in range(period - 1, len(v)):
            sma = v[i - period + 1 : i + 1].mean()
            out[i] = v[i] / (sma + 1e-10)
        return out

    def price_vs_sma(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        c = close.astype(np.float64)
        out = np.zeros(len(c), dtype=np.float32)
        for i in range(period - 1, len(c)):
            sma = c[i - period + 1 : i + 1].mean()
            out[i] = (c[i] - sma) / (sma + 1e-10)
        return out

    def compute_all(self, ohlcv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "rsi": self.rsi(ohlcv["close"]),
            "macd_hist": self.macd_hist(ohlcv["close"]),
            "bollinger_pctb": self.bollinger_pctb(ohlcv["close"]),
            "atr": self.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]),
            "volume_ratio": self.volume_ratio(ohlcv["close"], ohlcv["volume"]),
            "price_vs_sma": self.price_vs_sma(ohlcv["close"]),
        }
