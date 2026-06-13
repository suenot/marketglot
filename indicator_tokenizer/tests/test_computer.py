import numpy as np
import pytest
from indicators.computer import IndicatorComputer


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(42)
    n = 200
    close = 30000.0 + np.cumsum(rng.standard_normal(n) * 5).astype(np.float32)
    return {
        "open": close - rng.standard_normal(n).astype(np.float32),
        "high": (close + np.abs(rng.standard_normal(n)) * 3).astype(np.float32),
        "low": (close - np.abs(rng.standard_normal(n)) * 3).astype(np.float32),
        "close": close.astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }


def test_rsi_shape_and_range(ohlcv):
    comp = IndicatorComputer()
    rsi = comp.rsi(ohlcv["close"], period=14)
    assert len(rsi) == len(ohlcv["close"])
    valid = rsi[14:]
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_rsi_known_values():
    closes = np.arange(100, 130, dtype=np.float32)
    comp = IndicatorComputer()
    rsi = comp.rsi(closes, period=14)
    assert rsi[-1] > 90


def test_macd_hist_shape(ohlcv):
    comp = IndicatorComputer()
    hist = comp.macd_hist(ohlcv["close"])
    assert len(hist) == len(ohlcv["close"])


def test_bollinger_pctb_shape_and_range(ohlcv):
    comp = IndicatorComputer()
    pctb = comp.bollinger_pctb(ohlcv["close"])
    assert len(pctb) == len(ohlcv["close"])
    assert pctb[20:].mean() > -1
    assert pctb[20:].mean() < 2


def test_atr_shape(ohlcv):
    comp = IndicatorComputer()
    atr = comp.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert len(atr) == len(ohlcv["close"])
    assert atr[14:].min() >= 0


def test_volume_ratio_shape(ohlcv):
    comp = IndicatorComputer()
    vr = comp.volume_ratio(ohlcv["close"], ohlcv["volume"])
    assert len(vr) == len(ohlcv["close"])


def test_price_vs_sma_shape(ohlcv):
    comp = IndicatorComputer()
    pvs = comp.price_vs_sma(ohlcv["close"])
    assert len(pvs) == len(ohlcv["close"])


def test_compute_all(ohlcv):
    comp = IndicatorComputer()
    result = comp.compute_all(ohlcv)
    expected = ["rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"]
    for key in expected:
        assert key in result
        assert len(result[key]) == len(ohlcv["close"])
