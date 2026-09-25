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


def test_signal_starts_after_input_window_and_fills_next_open():
    engine = BacktestEngine(commission=0, stop_loss=-1, take_profit=1, max_hold=10)
    closes = np.array([100, 100, 100, 100, 110, 120], dtype=float)
    opens = np.array([100, 100, 100, 100, 105, 115], dtype=float)
    # The first signal is formed at the close of bar 3, not bar 0.
    result = engine.run(closes, np.array([2]), opens=opens, signal_start=3)
    assert result.trade_count == 1
    assert result.trades[0].entry_idx == 4
    assert result.trades[0].entry_price == 105
    assert result.trades[0].exit_idx == 5
    assert result.total_pnl == pytest.approx(120 / 105 - 1)
    assert np.all(result.equity_curve[:4] == 1)


def test_stop_uses_next_open_and_applies_both_commissions():
    engine = BacktestEngine(commission=0.01, stop_loss=-0.005,
                            take_profit=1, max_hold=10)
    closes = np.array([100, 100, 99, 99], dtype=float)
    opens = np.array([100, 100, 100, 90], dtype=float)
    result = engine.run(closes, np.array([2]), opens=opens)
    assert result.trades[0].entry_idx == 1
    assert result.trades[0].exit_idx == 3
    assert result.trades[0].exit_price == 90
    assert result.total_pnl == pytest.approx(0.99 * 0.90 * 0.99 - 1)
    assert result.max_drawdown == pytest.approx(result.total_pnl)


def test_compounded_total_return_and_bar_sharpe():
    engine = BacktestEngine(commission=0, stop_loss=-1, take_profit=0.01,
                            max_hold=10)
    closes = np.array([100, 102, 102, 102, 104.04, 104.04], dtype=float)
    opens = np.array([100, 100, 102, 102, 102, 104.04], dtype=float)
    result = engine.run(closes, np.array([2, 1, 1, 2, 1, 1]), opens=opens)
    assert result.trade_count == 2
    assert result.total_pnl == pytest.approx(1.02 ** 2 - 1)
    returns = np.diff(np.r_[1, result.equity_curve]) / np.r_[1, result.equity_curve][:-1]
    assert result.sharpe == pytest.approx(np.mean(returns) / np.std(returns) * np.sqrt(365 * 1440))


def test_profit_factor_uses_compounded_monetary_pnl():
    engine = BacktestEngine(commission=0, stop_loss=-0.09, take_profit=0.09, max_hold=10)
    closes = np.array([100, 110, 110, 110, 99, 99], dtype=float)
    opens = np.array([100, 100, 110, 110, 110, 99], dtype=float)
    result = engine.run(closes, np.array([2, 1, 2, 1, 1, 1]), opens=opens)
    assert [trade.pnl for trade in result.trades] == pytest.approx([0.1, -0.1])
    assert result.total_pnl == pytest.approx(-0.01)
    assert result.profit_factor == pytest.approx(0.1 / 0.11)


def test_misaligned_predictions_rejected():
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="predictions do not fit"):
        engine.run(np.ones(4), np.ones(3), signal_start=2)


def test_sparse_signal_indices_keep_bar_alignment():
    engine = BacktestEngine(commission=0, stop_loss=-1, take_profit=1, max_hold=10)
    closes = np.array([100, 100, 100, 100, 100, 110], dtype=float)
    result = engine.run(closes, np.array([2]), signal_indices=np.array([4]))
    assert result.trades[0].entry_idx == 5
