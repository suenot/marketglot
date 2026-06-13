"""Kronos baseline wrapper for marketglot.

Wraps the external Kronos foundation model ("candles-as-language") into a reusable
3-class trading signal (DOWN/FLAT/UP) consistent with marketglot conventions
(class order [DOWN=0, FLAT=1, UP=2], forward-return label with a +-threshold
deadband).

Kronos is NOT vendored into this repo. Clone it separately and point the wrapper
at it (env var KRONOS_PATH, a constructor arg, or a sibling ``../Kronos`` dir)::

    git clone https://github.com/shiyu-coder/Kronos
    export KRONOS_PATH=/path/to/Kronos

See ../docs/research/kronos.md for the analysis behind this baseline.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Class order shared across all marketglot projects.
DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}


def classify_return(ret: float, threshold_pct: float) -> int:
    """Map a forward return to a 3-class label with a +-threshold deadband.

    Args:
        ret: forward return as a fraction (e.g. 0.002 == +0.2%).
        threshold_pct: deadband half-width in *percent* (e.g. 0.15 == +-0.15%).
    """
    thr = threshold_pct / 100.0
    if ret > thr:
        return UP
    if ret < -thr:
        return DOWN
    return FLAT


def _resolve_kronos_path(kronos_path: str | None = None) -> Path:
    """Locate the Kronos checkout (arg > env KRONOS_PATH > sibling ``../Kronos``)."""
    cand = kronos_path or os.environ.get("KRONOS_PATH")
    if cand is None:
        # default: a sibling of the marketglot repo root
        cand = Path(__file__).resolve().parents[2] / "Kronos"
    p = Path(cand).expanduser().resolve()
    if not (p / "model").is_dir():
        raise FileNotFoundError(
            f"Kronos checkout not found at '{p}'. Clone "
            "https://github.com/shiyu-coder/Kronos and set KRONOS_PATH "
            "(or pass kronos_path=...)."
        )
    return p


@dataclass
class SignalResult:
    """A 3-class Kronos forecast for one window."""

    label: int
    label_name: str
    horizon: int
    threshold_pct: float
    last_close: float
    pred_close: float                 # point forecast: mean of MC paths at horizon
    pred_return: float
    class_probs: dict                 # {"DOWN":p, "FLAT":p, "UP":p} from MC paths
    n_paths: int
    pred_path: np.ndarray = field(repr=False)   # mean predicted close path (horizon,)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "label_name": self.label_name,
            "horizon": self.horizon,
            "threshold_pct": self.threshold_pct,
            "last_close": self.last_close,
            "pred_close": self.pred_close,
            "pred_return": self.pred_return,
            "class_probs": self.class_probs,
            "n_paths": self.n_paths,
        }


class KronosSignal:
    """Reusable wrapper: load Kronos once, emit marketglot 3-class signals.

    Example::

        sig = KronosSignal()                       # loads Kronos-small + tokenizer
        res = sig.predict_signal(df, x_ts, y_ts, horizon=60, threshold_pct=0.15)
        print(res.label_name, res.class_probs)
    """

    PRICE_COLS = ["open", "high", "low", "close"]

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str | None = None,
        max_context: int = 512,
        kronos_path: str | None = None,
    ):
        kp = _resolve_kronos_path(kronos_path)
        if str(kp) not in sys.path:
            sys.path.insert(0, str(kp))
        # Imported lazily so that importing this module (e.g. for classify_return
        # in tests) does not require torch / a Kronos checkout.
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        tokenizer.eval()
        model.eval()
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
        self.device = self.predictor.device
        self.max_context = max_context

    def predict_signal(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        *,
        horizon: int = 60,
        threshold_pct: float = 0.15,
        n_paths: int = 10,
        T: float = 1.0,
        top_p: float = 0.9,
        verbose: bool = False,
    ) -> SignalResult:
        """Forecast ``horizon`` candles and reduce to a 3-class signal.

        Draws ``n_paths`` independent stochastic forecasts (each ``sample_count=1``);
        the point forecast is their mean path, the class probabilities are the
        fraction of paths whose horizon return falls in each class.

        Args:
            df: history with columns ['open','high','low','close'] (+ optional
                'volume','amount'); the last row is the decision point.
            x_timestamp: timestamps for ``df`` (datetime Series).
            y_timestamp: timestamps for the ``horizon`` future candles.
            horizon: forecast length in candles (marketglot default 60).
            threshold_pct: deadband half-width in percent (default 0.15).
            n_paths: number of Monte-Carlo forecast paths.
        """
        if len(y_timestamp) != horizon:
            raise ValueError(f"y_timestamp length {len(y_timestamp)} != horizon {horizon}")
        if "close" not in df.columns:
            raise ValueError("df must contain a 'close' column")

        cols = self.PRICE_COLS + [c for c in ("volume", "amount") if c in df.columns]
        x_df = df[cols].reset_index(drop=True)
        x_ts = pd.Series(pd.to_datetime(x_timestamp)).reset_index(drop=True)
        y_ts = pd.Series(pd.to_datetime(y_timestamp)).reset_index(drop=True)

        paths = []
        for _ in range(n_paths):
            pred = self.predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=horizon, T=T, top_p=top_p, sample_count=1, verbose=verbose,
            )
            paths.append(pred["close"].to_numpy(dtype=np.float64))
        paths = np.stack(paths)  # (n_paths, horizon)

        last_close = float(df["close"].iloc[-1])
        path_returns = paths[:, -1] / last_close - 1.0
        path_labels = [classify_return(r, threshold_pct) for r in path_returns]
        class_probs = {
            CLASS_NAMES[c]: float(np.mean([lbl == c for lbl in path_labels]))
            for c in (DOWN, FLAT, UP)
        }

        mean_path = paths.mean(axis=0)
        pred_close = float(mean_path[-1])
        pred_return = pred_close / last_close - 1.0
        label = classify_return(pred_return, threshold_pct)

        return SignalResult(
            label=label, label_name=CLASS_NAMES[label], horizon=horizon,
            threshold_pct=threshold_pct, last_close=last_close, pred_close=pred_close,
            pred_return=pred_return, class_probs=class_probs, n_paths=n_paths,
            pred_path=mean_path,
        )
