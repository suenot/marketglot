import numpy as np
import pytest
from pathlib import Path
from indicators.tokenizer import IndicatorTokenizer, FixedBoundaries, QuantileBoundaries


def test_fixed_boundaries_encode():
    fb = FixedBoundaries(bins=[20, 30, 70, 80], offset=2)
    assert fb.vocab_size == 7
    assert fb.encode(0.0) == 2
    assert fb.encode(25.0) == 3
    assert fb.encode(50.0) == 4
    assert fb.encode(75.0) == 5
    assert fb.encode(90.0) == 6


def test_fixed_boundaries_batch():
    fb = FixedBoundaries(bins=[20, 30, 70, 80], offset=2)
    vals = np.array([10.0, 25.0, 50.0, 75.0, 90.0], dtype=np.float32)
    ids = fb.encode_batch(vals)
    assert list(ids) == [2, 3, 4, 5, 6]


def test_quantile_boundaries_fit():
    qb = QuantileBoundaries(n_bins=5, offset=2)
    data = np.arange(100, dtype=np.float32)
    qb.fit(data)
    assert qb.vocab_size == 7
    assert qb.boundaries is not None
    assert len(qb.boundaries) == 4


def test_quantile_boundaries_encode():
    qb = QuantileBoundaries(n_bins=4, offset=2)
    data = np.arange(100, dtype=np.float32)
    qb.fit(data)
    ids = qb.encode_batch(data)
    assert ids.min() >= 2
    assert ids.max() <= 5


def test_indicator_tokenizer_fit_and_encode():
    rng = np.random.default_rng(42)
    n = 500
    ohlcv = {
        "close": (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32),
        "high": (30000 + np.cumsum(rng.standard_normal(n) * 5) + 5).astype(np.float32),
        "low": (30000 + np.cumsum(rng.standard_normal(n) * 5) - 5).astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }
    from indicators.computer import IndicatorComputer
    comp = IndicatorComputer()
    indicators = comp.compute_all(ohlcv)

    tok = IndicatorTokenizer()
    tok.fit(indicators)
    encoded = tok.encode(indicators)
    for key, ids in encoded.items():
        assert len(ids) == n
        assert ids.dtype == np.int32


def test_indicator_tokenizer_save_load(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    ohlcv = {
        "close": (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32),
        "high": (30000 + np.cumsum(rng.standard_normal(n) * 5) + 5).astype(np.float32),
        "low": (30000 + np.cumsum(rng.standard_normal(n) * 5) - 5).astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }
    from indicators.computer import IndicatorComputer
    comp = IndicatorComputer()
    indicators = comp.compute_all(ohlcv)

    tok = IndicatorTokenizer()
    tok.fit(indicators)
    tok.save(tmp_path)

    tok2 = IndicatorTokenizer()
    tok2.load(tmp_path)
    encoded1 = tok.encode(indicators)
    encoded2 = tok2.encode(indicators)
    for key in encoded1:
        np.testing.assert_array_equal(encoded1[key], encoded2[key])


def test_indicator_tokenizer_vocab_sizes():
    tok = IndicatorTokenizer()
    rng = np.random.default_rng(42)
    n = 200
    ohlcv = {
        "close": (30000 + np.cumsum(rng.standard_normal(n) * 5)).astype(np.float32),
        "high": (30000 + np.cumsum(rng.standard_normal(n) * 5) + 5).astype(np.float32),
        "low": (30000 + np.cumsum(rng.standard_normal(n) * 5) - 5).astype(np.float32),
        "volume": (np.abs(rng.standard_normal(n)) * 100 + 50).astype(np.float32),
    }
    from indicators.computer import IndicatorComputer
    comp = IndicatorComputer()
    indicators = comp.compute_all(ohlcv)
    tok.fit(indicators)
    vs = tok.vocab_sizes()
    assert vs["rsi"] == 7
    assert vs["macd_hist"] == 9
    assert vs["atr"] == 8
