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
    pnl: float


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


class BacktestEngine:
    def __init__(
        self,
        commission: float = 0.0004,
        stop_loss: float = -0.005,
        take_profit: float = 0.01,
        max_hold: int = 60,
    ) -> None:
        self.commission = commission
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold = max_hold

    def run(self, closes: np.ndarray, predictions: np.ndarray) -> BacktestResult:
        trades: list[Trade] = []
        in_position = False
        direction = 0
        entry_idx = 0
        entry_price = 0.0

        for i in range(len(predictions)):
            if in_position:
                pnl_pct = direction * (closes[i] - entry_price) / entry_price
                hold_duration = i - entry_idx
                exit = False
                if pnl_pct <= self.stop_loss:
                    exit = True
                elif pnl_pct >= self.take_profit:
                    exit = True
                elif hold_duration >= self.max_hold:
                    exit = True

                if exit:
                    trade_pnl = pnl_pct - 2 * self.commission
                    trades.append(Trade(
                        entry_idx=entry_idx, exit_idx=i,
                        direction=direction, entry_price=entry_price,
                        exit_price=float(closes[i]), pnl=trade_pnl,
                    ))
                    in_position = False

            if not in_position and i + 1 < len(closes):
                pred = predictions[i]
                if pred == 2:
                    direction = 1
                elif pred == 0:
                    direction = -1
                else:
                    continue
                in_position = True
                entry_idx = i + 1
                entry_price = float(closes[i + 1])

        return self._compute_metrics(trades)

    def _compute_metrics(self, trades: list[Trade]) -> BacktestResult:
        if not trades:
            return BacktestResult(
                total_pnl=0.0, sharpe=0.0, max_drawdown=0.0,
                win_rate=0.0, trade_count=0, profit_factor=0.0,
                avg_duration=0.0, trades=trades,
            )

        pnls = [t.pnl for t in trades]
        total_pnl = float(sum(pnls))
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(trades)
        gross_profit = float(sum(wins)) if wins else 0.0
        gross_loss = float(abs(sum(losses))) if losses else 1e-10
        profit_factor = float(gross_profit / gross_loss)

        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        sharpe = 0.0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 1440))

        avg_duration = float(np.mean([t.exit_idx - t.entry_idx for t in trades]))

        return BacktestResult(
            total_pnl=total_pnl, sharpe=sharpe, max_drawdown=max_drawdown,
            win_rate=win_rate, trade_count=len(trades),
            profit_factor=profit_factor, avg_duration=avg_duration, trades=trades,
        )
