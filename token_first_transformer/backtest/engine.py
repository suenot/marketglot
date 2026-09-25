from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    pnl: float  # Net return on equity committed to this trade.


@dataclass
class BacktestResult:
    total_pnl: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    profit_factor: float
    avg_duration: float
    trades: list[Trade]
    equity_curve: np.ndarray


class BacktestEngine:
    """One position at a time, with signals at closes and fills at the next open.

    ``signal_start`` is the index of the close that produced predictions[0].
    ``signal_indices`` can instead map sparse predictions to their source bars.
    Sharpe assumes one-minute bars and a 365-day crypto trading year.
    Stops are checked at closes and filled at the next open, including gaps.
    """

    def __init__(
        self,
        commission: float = 0.0004,
        stop_loss: float = -0.005,
        take_profit: float = 0.01,
        max_hold: int = 60,
        slippage: float = 0.0,
    ) -> None:
        if not 0 <= commission < 1 or not 0 <= slippage < 1:
            raise ValueError("commission and slippage must be fractions in [0, 1)")
        if max_hold < 1:
            raise ValueError("max_hold must be positive")
        self.commission = commission
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold = max_hold
        self.slippage = slippage

    def run(
        self,
        closes: np.ndarray,
        predictions: np.ndarray,
        *,
        opens: np.ndarray | None = None,
        signal_start: int = 0,
        signal_indices: np.ndarray | None = None,
    ) -> BacktestResult:
        closes = np.asarray(closes, dtype=np.float64)
        predictions = np.asarray(predictions)
        if opens is None:
            # For callers without OHLC data, use the next close as a fill proxy.
            opens = closes
        else:
            opens = np.asarray(opens, dtype=np.float64)
        if closes.ndim != 1 or opens.shape != closes.shape or predictions.ndim != 1:
            raise ValueError("closes, opens and predictions must be one-dimensional arrays")
        if signal_indices is None:
            signal_indices = np.arange(signal_start, signal_start + len(predictions))
        else:
            signal_indices = np.asarray(signal_indices)
        if (signal_indices.ndim != 1 or len(signal_indices) != len(predictions)
                or np.any(signal_indices < 0) or np.any(signal_indices >= len(closes))
                or np.any(np.diff(signal_indices) <= 0)):
            raise ValueError("predictions do not fit the supplied closes and signal_start")
        if not np.all(np.isfinite(closes)) or not np.all(np.isfinite(opens)) or np.any(closes <= 0) or np.any(opens <= 0):
            raise ValueError("open and close prices must be finite and positive")
        signals_by_bar = np.full(len(closes), -1, dtype=np.int8)
        signals_by_bar[signal_indices] = predictions

        trades: list[Trade] = []
        equity_curve = np.empty(len(closes), dtype=np.float64)
        cash = 1.0
        direction = 0
        pending_entry = 0
        pending_exit = False
        entry_idx = 0
        entry_price = 0.0
        entry_equity = 0.0
        position_equity = 0.0

        for i, (open_price, close_price) in enumerate(zip(opens, closes)):
            if pending_exit:
                fill = open_price * (1 - direction * self.slippage)
                cash = position_equity * (1 + direction * (fill - entry_price) / entry_price) * (1 - self.commission)
                trades.append(Trade(entry_idx, i, direction, entry_price, float(fill), cash / entry_equity - 1))
                direction = 0
                pending_exit = False

            if pending_entry:
                direction = pending_entry
                pending_entry = 0
                entry_idx = i
                entry_price = float(open_price * (1 + direction * self.slippage))
                entry_equity = cash
                position_equity = cash * (1 - self.commission)

            if direction:
                gross_return = direction * (close_price - entry_price) / entry_price
                equity = position_equity * (1 + gross_return)
                if gross_return <= self.stop_loss or gross_return >= self.take_profit or i - entry_idx >= self.max_hold:
                    pending_exit = True
                if i == len(closes) - 1:
                    # End-of-period liquidation is scheduled in advance.
                    fill = close_price * (1 - direction * self.slippage)
                    cash = position_equity * (1 + direction * (fill - entry_price) / entry_price) * (1 - self.commission)
                    trades.append(Trade(entry_idx, i, direction, entry_price, float(fill), cash / entry_equity - 1))
                    equity = cash
                    direction = 0
            else:
                equity = cash
                if i + 1 < len(closes):
                    pred = signals_by_bar[i]
                    if pred == 2:
                        pending_entry = 1
                    elif pred == 0:
                        pending_entry = -1
            equity_curve[i] = equity

        return self._compute_metrics(trades, equity_curve)

    @staticmethod
    def _compute_metrics(trades: list[Trade], equity_curve: np.ndarray) -> BacktestResult:
        total_pnl = float(equity_curve[-1] - 1) if len(equity_curve) else 0.0
        history = np.concatenate(([1.0], equity_curve))
        running_max = np.maximum.accumulate(history)
        max_drawdown = float(np.min(history / running_max - 1))

        returns = np.diff(history) / history[:-1]
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365 * 1440))

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = -sum(losses)
        profit_factor = float(gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit else 0.0)
        return BacktestResult(
            total_pnl=total_pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=len(wins) / len(trades) if trades else 0.0,
            trade_count=len(trades),
            profit_factor=profit_factor,
            avg_duration=float(np.mean([t.exit_idx - t.entry_idx for t in trades])) if trades else 0.0,
            trades=trades,
            equity_curve=equity_curve,
        )
