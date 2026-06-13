"""In-memory L2 order book reconstructed from snapshots and deltas.

The book holds a ``price -> qty`` dict per side. Sorting is deferred to the
read paths (:meth:`LocalBook.top_levels`, :meth:`LocalBook.mid`) because we
only sample at ~1 Hz, so per-update sorting would be wasted work.
"""
from __future__ import annotations

from collections.abc import Iterable

# Side labels as stored in the parquet ``side`` column.
_BID = "bid"
_ASK = "ask"


class LocalBook:
    """A single-symbol L2 order book.

    Internally two dicts map price to quantity. A delta with ``qty == 0``
    removes the corresponding level.
    """

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def _side(self, side: str) -> dict[float, float]:
        return self.bids if side == _BID else self.asks

    def apply_snapshot(
        self,
        prices: Iterable[float],
        qtys: Iterable[float],
        sides: Iterable[str],
    ) -> None:
        """Replace the whole book state with the given levels.

        Levels with ``qty <= 0`` in a snapshot are ignored (not stored).
        """
        self.bids = {}
        self.asks = {}
        for price, qty, side in zip(prices, qtys, sides):
            if qty <= 0:
                continue
            self._side(side)[float(price)] = float(qty)

    def apply_delta(
        self,
        prices: Iterable[float],
        qtys: Iterable[float],
        sides: Iterable[str],
    ) -> None:
        """Apply incremental updates; ``qty == 0`` deletes the level."""
        for price, qty, side in zip(prices, qtys, sides):
            book = self._side(side)
            price = float(price)
            qty = float(qty)
            if qty == 0:
                book.pop(price, None)
            else:
                book[price] = qty

    def top_levels(
        self, depth: int
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return ``(bids, asks)``, each up to ``depth`` ``(price, qty)`` tuples.

        Bids are sorted by descending price (best bid first), asks by
        ascending price (best ask first).
        """
        bids = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:depth]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:depth]
        return bids, asks

    def mid(self) -> float | None:
        """Mid price ``(best_bid + best_ask) / 2`` or ``None`` if a side is empty."""
        if not self.bids or not self.asks:
            return None
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        return (best_bid + best_ask) / 2.0

    def is_valid(self) -> bool:
        """True when both sides are non-empty and ``best_bid < best_ask``."""
        if not self.bids or not self.asks:
            return False
        return max(self.bids) < min(self.asks)
