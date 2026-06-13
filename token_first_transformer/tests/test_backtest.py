import numpy as np
import pytest
from backtest.engine import BacktestEngine, BacktestResult


def test_long_win():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    closes = np.array([100.0, 100.1, 100.3, 100.5, 100.8, 101.0, 101.2, 101.3, 101.4, 101.5], dtype=np.float32)
    predictions = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 1
    assert result.win_rate > 0
    assert result.total_pnl > 0


def test_short_loss():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    closes = np.array([100.0, 100.2, 100.4, 100.6, 100.8, 101.0], dtype=np.float32)
    predictions = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 1
    assert result.total_pnl < 0


def test_flat_no_trade():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    closes = np.array([100.0, 100.1, 100.2, 100.1, 100.0, 99.9], dtype=np.float32)
    predictions = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count == 0
    assert result.total_pnl == 0.0


def test_max_hold_exit():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.1, take_profit=0.1, max_hold=3)
    closes = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6], dtype=np.float32)
    predictions = np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    result = engine.run(closes, predictions)
    assert result.trade_count >= 1


def test_result_metrics():
    engine = BacktestEngine(commission=0.0004, stop_loss=-0.005, take_profit=0.01, max_hold=60)
    rng = np.random.default_rng(42)
    n = 200
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.2).astype(np.float32)
    predictions = rng.integers(0, 3, n).astype(np.int32)
    result = engine.run(closes, predictions)
    assert isinstance(result.total_pnl, float)
    assert isinstance(result.sharpe, float)
    assert isinstance(result.max_drawdown, float)
    assert 0.0 <= result.win_rate <= 1.0
    assert isinstance(result.trade_count, int)
    assert isinstance(result.profit_factor, float)
