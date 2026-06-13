"""Objective evaluation of the Kronos baseline on marketglot klines.

Samples many (symbol, window) pairs, forecasts each with Kronos (batched via
predict_batch for speed), and reports 3-class accuracy (DOWN/FLAT/UP), binary
directional accuracy, confusion matrix, and forecast MAE — each against naive
baselines (most-frequent class; persistence = last close held flat).

Example:
    KRONOS_PATH=../Kronos python kronos_baseline/scripts/evaluate.py \
        --windows-per-symbol 30 --sample-count 5 --batch-size 16 \
        --out kronos_baseline/artifacts/eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from kronos_baseline.kronos_signal import (
    KronosSignal, classify_return, CLASS_NAMES, DOWN, FLAT, UP,
)

DEFAULT_DATA_ROOT = "/Users/suenot/projects/w_trading/w_trender/backtests/data"
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT",
          "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT"]


def pick_symbols(data_root: str, n: int, seed: int) -> list[str]:
    have = {p.split("/")[-2] for p in glob.glob(f"{data_root}/*/klines_1m")}
    chosen = [s for s in MAJORS if s in have]
    rng = np.random.default_rng(seed)
    rest = sorted(have - set(chosen))
    rng.shuffle(rest)
    for s in rest:
        if len(chosen) >= n:
            break
        chosen.append(s)
    return chosen[:n]


def load_series(data_root: str, symbol: str, max_files: int) -> pd.DataFrame:
    files = sorted(glob.glob(f"{data_root}/{symbol}/klines_1m/*.parquet"))[-max_files:]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def build_windows(df, symbol, lookback, horizon, n_windows, rng):
    """Evenly-spaced windows with available ground truth."""
    lo, hi = lookback, len(df) - horizon
    if hi <= lo:
        return []
    idxs = np.linspace(lo, hi - 1, num=min(n_windows, hi - lo), dtype=int)
    out = []
    for end in idxs:
        ctx = df.iloc[end - lookback:end]
        fut = df.iloc[end:end + horizon]
        out.append({
            "symbol": symbol,
            "x_df": ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
            "x_ts": ctx["timestamps"].reset_index(drop=True),
            "y_ts": fut["timestamps"].reset_index(drop=True),
            "last_close": float(ctx["close"].iloc[-1]),
            "actual_close": fut["close"].to_numpy(dtype=np.float64),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Kronos baseline on marketglot klines")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--n-symbols", type=int, default=12)
    ap.add_argument("--symbols", nargs="*", default=None, help="explicit symbols (overrides --n-symbols)")
    ap.add_argument("--windows-per-symbol", type=int, default=30)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--sample-count", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-files", type=int, default=6, help="how many recent monthly files to load")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    symbols = args.symbols or pick_symbols(args.data_root, args.n_symbols, args.seed)
    print(f"symbols ({len(symbols)}): {symbols}")

    windows = []
    for s in symbols:
        try:
            df = load_series(args.data_root, s, args.max_files)
        except Exception as e:
            print(f"  skip {s}: {e}")
            continue
        windows += build_windows(df, s, args.lookback, args.horizon, args.windows_per_symbol, rng)
    print(f"total windows: {len(windows)}")

    sig = KronosSignal(model_name=args.model, tokenizer_name=args.tokenizer, device=args.device,
                       max_context=max(512, args.lookback))
    print(f"device={sig.device}  batch_size={args.batch_size}  sample_count={args.sample_count}")

    # batched forecasting
    preds = [None] * len(windows)
    t0 = time.time()
    for b in range(0, len(windows), args.batch_size):
        chunk = windows[b:b + args.batch_size]
        pred_dfs = sig.predictor.predict_batch(
            df_list=[w["x_df"] for w in chunk],
            x_timestamp_list=[w["x_ts"] for w in chunk],
            y_timestamp_list=[w["y_ts"] for w in chunk],
            pred_len=args.horizon, T=1.0, top_p=0.9, sample_count=args.sample_count, verbose=False,
        )
        for j, pd_df in enumerate(pred_dfs):
            preds[b + j] = pd_df["close"].to_numpy(dtype=np.float64)
        done = min(b + args.batch_size, len(windows))
        el = time.time() - t0
        print(f"  {done}/{len(windows)} windows  ({el:.0f}s, {el/done:.2f}s/win)", flush=True)

    # metrics
    rows = []
    for w, pc in zip(windows, preds):
        pred_ret = pc[-1] / w["last_close"] - 1.0
        act_ret = w["actual_close"][-1] / w["last_close"] - 1.0
        mae_pct = float(np.mean(np.abs(pc - w["actual_close"])) / w["last_close"] * 100)
        persist_mae_pct = float(np.mean(np.abs(w["last_close"] - w["actual_close"])) / w["last_close"] * 100)
        rows.append({
            "symbol": w["symbol"],
            "pred_label": classify_return(pred_ret, args.threshold),
            "act_label": classify_return(act_ret, args.threshold),
            "pred_ret": pred_ret, "act_ret": act_ret,
            "mae_pct": mae_pct, "persist_mae_pct": persist_mae_pct,
        })
    R = pd.DataFrame(rows)

    acc3 = float((R.pred_label == R.act_label).mean())
    # most-frequent-class baseline accuracy
    mfc = R.act_label.value_counts(normalize=True)
    mfc_acc = float(mfc.max())
    # binary directional (exclude FLAT in both pred and actual)
    nz = R[(R.pred_label != FLAT) & (R.act_label != FLAT)]
    dir_acc = float((nz.pred_label == nz.act_label).mean()) if len(nz) else float("nan")
    # up/down-only directional baseline = majority among non-flat actuals
    nz_act = R[R.act_label != FLAT]
    dir_base = float(max((nz_act.act_label == UP).mean(), (nz_act.act_label == DOWN).mean())) if len(nz_act) else float("nan")
    conf = pd.crosstab(R.act_label.map(CLASS_NAMES), R.pred_label.map(CLASS_NAMES),
                       rownames=["actual"], colnames=["pred"], dropna=False)

    print("\n================ RESULTS ================")
    print(f"windows={len(R)}  symbols={R.symbol.nunique()}  horizon={args.horizon}  thr=+-{args.threshold}%")
    print(f"\nactual class balance: " +
          "  ".join(f"{CLASS_NAMES[c]}={(R.act_label==c).mean():.2f}" for c in (DOWN, FLAT, UP)))
    print(f"\n3-class accuracy : {acc3:.3f}   (most-frequent baseline {mfc_acc:.3f}, "
          f"lift {acc3-mfc_acc:+.3f})")
    print(f"directional acc  : {dir_acc:.3f}   (non-flat both; UP/DOWN baseline {dir_base:.3f}, "
          f"lift {dir_acc-dir_base:+.3f})  n={len(nz)}")
    print(f"forecast MAE %   : {R.mae_pct.mean():.3f}   (persistence {R.persist_mae_pct.mean():.3f}, "
          f"{'BETTER' if R.mae_pct.mean()<R.persist_mae_pct.mean() else 'WORSE'})")
    print("\nconfusion (rows=actual, cols=pred):")
    print(conf.to_string())
    print("\nper-symbol 3-class accuracy:")
    per = R.groupby("symbol").apply(lambda g: (g.pred_label == g.act_label).mean(), include_groups=False)
    print(per.round(3).to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": vars(args), "n_windows": len(R), "n_symbols": int(R.symbol.nunique()),
            "acc3": acc3, "mfc_baseline": mfc_acc, "dir_acc": dir_acc, "dir_baseline": dir_base,
            "mae_pct": float(R.mae_pct.mean()), "persist_mae_pct": float(R.persist_mae_pct.mean()),
            "class_balance": {CLASS_NAMES[c]: float((R.act_label == c).mean()) for c in (DOWN, FLAT, UP)},
            "per_symbol_acc": {k: float(v) for k, v in per.items()},
        }
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
