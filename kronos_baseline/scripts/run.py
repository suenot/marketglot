"""Run the Kronos baseline on marketglot klines and print a 3-class signal.

Example:
    KRONOS_PATH=../Kronos python kronos_baseline/scripts/run.py \
        --symbol BTCUSDT --start 20000 --lookback 400 --horizon 60 --n-paths 8

If ground-truth candles exist after the window, the script also reports the
realized class and directional match (a one-window sanity check, not an eval).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from kronos_baseline.kronos_signal import KronosSignal, classify_return, CLASS_NAMES

DEFAULT_DATA_ROOT = "/Users/suenot/projects/w_trading/w_trender/backtests/data"


def load_klines(data_root: str, symbol: str, granularity: str = "klines_1m") -> pd.DataFrame:
    files = sorted(glob.glob(f"{data_root}/{symbol}/{granularity}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {data_root}/{symbol}/{granularity}")
    df = pd.concat([pd.read_parquet(f) for f in files[-3:]], ignore_index=True)
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Kronos baseline 3-class signal on marketglot klines")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--granularity", default="klines_1m")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.15, help="deadband half-width, percent")
    ap.add_argument("--n-paths", type=int, default=10)
    ap.add_argument("--start", default="-1",
                    help="row index where the lookback window ends; '-1' = use the latest complete window")
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    df = load_klines(args.data_root, args.symbol, args.granularity)
    n = len(df)
    end = (n - args.horizon) if args.start == "-1" else int(args.start)
    start = end - args.lookback
    if start < 0:
        raise ValueError(f"not enough history: need {args.lookback}, window start={start}")

    ctx = df.iloc[start:end].reset_index(drop=True)
    have_future = end + args.horizon <= n
    if have_future:
        fut = df.iloc[end:end + args.horizon].reset_index(drop=True)
        y_ts = fut["timestamps"]
    else:  # synthesize future timestamps from the median bar spacing
        step = ctx["timestamps"].diff().median()
        y_ts = pd.Series([ctx["timestamps"].iloc[-1] + step * (i + 1) for i in range(args.horizon)])

    print(f"[{args.symbol}] window rows {start}:{end}  lookback={args.lookback} "
          f"horizon={args.horizon} n_paths={args.n_paths} thr=+-{args.threshold}%")
    sig = KronosSignal(model_name=args.model, tokenizer_name=args.tokenizer, device=args.device)
    print(f"device={sig.device}  (loading + {args.n_paths} forecast paths)")

    res = sig.predict_signal(
        df=ctx[["open", "high", "low", "close", "volume"]],
        x_timestamp=ctx["timestamps"], y_timestamp=y_ts,
        horizon=args.horizon, threshold_pct=args.threshold, n_paths=args.n_paths,
        verbose=False,
    )

    print("\n=== Kronos signal ===")
    print(f"last close      : {res.last_close:,.2f}")
    print(f"pred close (+{res.horizon}): {res.pred_close:,.2f}   ret={res.pred_return*100:+.3f}%")
    print(f"signal          : {res.label_name}")
    print(f"class probs      : " + "  ".join(f"{k}={v:.2f}" for k, v in res.class_probs.items()))

    if have_future:
        act_close = float(fut["close"].iloc[-1])
        act_ret = act_close / res.last_close - 1.0
        act_label = classify_return(act_ret, args.threshold)
        mae = float(np.mean(np.abs(res.pred_path - fut["close"].to_numpy())))
        print("\n=== vs ground truth (single-window sanity) ===")
        print(f"actual close    : {act_close:,.2f}   ret={act_ret*100:+.3f}%  -> {CLASS_NAMES[act_label]}")
        print(f"directional match: {res.label == act_label}    close MAE over horizon: {mae:,.2f}")


if __name__ == "__main__":
    main()
