"""Paired, hourly BTCUSDT evaluation of a local System One-compatible teacher.

Run from token_first_transformer/ with an existing Transformer checkpoint::

    python scripts/evaluate_teacher.py --checkpoint checkpoints/best.pt \
        --data-dir data/teacher_eval --split val

The first-N --max-signals mode is a partial smoke check, not a split result.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import socket
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from backtest.engine import BacktestEngine
from dataset.klines_dataset import KlinesDataset, load_tokenizers, make_split
from models.price_transformer import PriceTransformer


LABELS = ("DOWN", "FLAT", "UP")
CRITERIA_ORDERS = (LABELS, ("FLAT", "UP", "DOWN"), ("UP", "DOWN", "FLAT"))
PROMPT_VERSION = "btc1m-hourly-features-v1"
MODEL_ID = "kev-latest"
QUESTION = {
    "type": "choice",
    "instructions": (
        "At the signal close, classify BTCUSDT's close-to-close return over the next "
        "60 one-minute candles. Use only the supplied features from 128 completed candles. "
        "Choose the most likely class; do not account for trading costs."
    ),
    "criteria": {
        "DOWN": "The close 60 minutes after the signal is more than 0.15% below the signal close.",
        "FLAT": "The close 60 minutes after the signal is within +/-0.15% of the signal close, inclusive.",
        "UP": "The close 60 minutes after the signal is more than 0.15% above the signal close.",
    },
}
STATE_SCHEMA = (
    "JSON string with BTCUSDT 1m features from the 128 completed candles only: "
    "close_return_pct at 1/5/15/30/60/120 minutes; high_low_range_pct over "
    "5/15/60/128 candles divided by the signal close; realized_vol_pct of "
    "1m log returns over 15/60/120 minutes; current candle body/range pct; "
    "current and last-5 mean volume divided by last-60 mean volume. "
    "No timestamp or absolute price is supplied. All values are rounded to 5 decimals."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def ordered_question(order: tuple[str, ...]) -> dict:
    return {**QUESTION, "criteria": {label: QUESTION["criteria"][label] for label in order}}


def request_bytes(state: str, order: tuple[str, ...]) -> bytes:
    # Match the original canonical request byte order in every field except
    # criteria. Only the option order may differ across the three runs.
    question = ordered_question(order)
    ordered = {"criteria": question["criteria"],
               "instructions": question["instructions"], "type": question["type"]}
    return json.dumps({"model": MODEL_ID, "questions": {"decision": ordered}, "state": state},
                      separators=(",", ":"), allow_nan=False).encode()


def hourly_sample_positions(ds: KlinesDataset) -> np.ndarray:
    """Indices into the dataset whose *completed* signal candle ends on UTC hour."""
    signal_indices = ds.sample_starts + ds.seq_len - 1
    return np.flatnonzero((ds.timestamps[signal_indices] + 60) % 3600 == 0)


def make_state(ds: KlinesDataset, signal_index: int, opens: np.ndarray) -> str:
    start = signal_index - ds.seq_len + 1
    if start < 0:
        raise ValueError("incomplete history")
    close = float(ds.closes[signal_index])
    def pct(value: float) -> float:
        return round(100 * float(value), 5)

    returns = {str(h): pct(close / ds.closes[signal_index - h] - 1)
               for h in (1, 5, 15, 30, 60, 120)}
    ranges = {str(h): pct((np.max(ds.highs[signal_index - h + 1:signal_index + 1])
                           - np.min(ds.lows[signal_index - h + 1:signal_index + 1])) / close)
              for h in (5, 15, 60, 128)}
    volatility = {str(h): pct(np.std(np.diff(np.log(
        ds.closes[signal_index - h:signal_index + 1].astype(np.float64)
    )))) for h in (15, 60, 120)}
    mean_volume = float(np.mean(ds.volumes[signal_index - 59:signal_index + 1]))
    return json.dumps({
        "symbol": "BTCUSDT", "timeframe": "1m",
        "close_return_pct": returns,
        "high_low_range_pct": ranges,
        "realized_vol_pct": volatility,
        "last_candle_body_pct": pct(close / opens[signal_index] - 1),
        "last_candle_range_pct": pct((ds.highs[signal_index] - ds.lows[signal_index]) / close),
        "volume_ratio_current_to_60": round(float(ds.volumes[signal_index]) / mean_volume, 5) if mean_volume else 0.0,
        "volume_ratio_5_to_60": round(float(np.mean(ds.volumes[signal_index - 4:signal_index + 1])) / mean_volume, 5) if mean_volume else 0.0,
    }, separators=(",", ":"), allow_nan=False)


def parse_choice(response: object) -> tuple[int, dict[str, float]]:
    if not isinstance(response, dict):
        raise ValueError("response is not an object")
    answers = response.get("answers")
    answer = answers.get("decision") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("missing choice answer")
    choice, raw = answer.get("choice"), answer.get("probabilities")
    if choice not in LABELS or not isinstance(raw, dict) or set(raw) != set(LABELS):
        raise ValueError("invalid choice or probabilities")
    if any(type(raw[key]) not in (int, float) for key in LABELS):
        raise ValueError("non-numeric probability")
    probabilities = {key: float(raw[key]) for key in LABELS}
    values = np.array(list(probabilities.values()))
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1) or abs(values.sum() - 1) > 0.02:
        raise ValueError("invalid probability distribution")
    return LABELS.index(choice), probabilities


def query_teacher(endpoint: str, state: str, timeout: float,
                  order: tuple[str, ...] = LABELS) -> tuple[int, dict[str, float] | None, str]:
    body = request_bytes(state, order)
    http_request = request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = json.load(response)
        prediction, probabilities = parse_choice(payload)
        return prediction, probabilities, "ok"
    except (TimeoutError, socket.timeout):
        return 1, None, "timeout"
    except error.HTTPError:
        return 1, None, "http_error"
    except error.URLError as exc:
        return 1, None, "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "connection_error"
    except (ValueError, TypeError, UnicodeDecodeError, http.client.HTTPException):
        return 1, None, "invalid_response"
    except OSError:
        return 1, None, "connection_error"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_jsonl(path: Path, row: dict) -> None:
    """Fsync one record; read_cache discards an interrupted trailing write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        data = canonical_bytes(row) + b"\n"
        while data:
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)


def read_cache(path: Path, experiment_id: str, signals: np.ndarray, timestamps: np.ndarray) -> dict[int, dict]:
    if not path.exists():
        return {}
    expected = {int(i): int(timestamps[i]) for i in signals}
    cached = {}
    with path.open("rb+") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.seek(0)
                content = stream.read()
                complete = content.rfind(b"\n") + 1
                stream.truncate(complete)
                stream.flush()
                os.fsync(stream.fileno())
        stream.seek(0)
        for line in stream:
            row = json.loads(line)
            index = row.get("signal_index")
            if (row.get("experiment_id") != experiment_id or index not in expected
                    or row.get("signal_timestamp") != expected[index]
                    or row.get("prediction") not in (0, 1, 2)):
                raise ValueError(f"cache does not match experiment: {path}")
            cached[index] = row
    return cached


def expected_months(start: str, end: str) -> list[str]:
    first_y, first_m = map(int, start.split("-"))
    last_y, last_m = map(int, end.split("-"))
    first, last = first_y * 12 + first_m - 1, last_y * 12 + last_m - 1
    if first > last or not (1 <= first_m <= 12 and 1 <= last_m <= 12):
        raise ValueError("invalid split months")
    return [f"{month // 12:04d}-{month % 12 + 1:02d}" for month in range(first, last + 1)]


def validate_split_coverage(timestamps: np.ndarray, months: list[str]) -> None:
    """Require full UTC monthly coverage, not merely continuous available rows."""
    first_year, first_month = map(int, months[0].split("-"))
    last_year, last_month = map(int, months[-1].split("-"))
    next_year, next_month = (last_year + 1, 1) if last_month == 12 else (last_year, last_month + 1)
    expected_first = int(datetime(first_year, first_month, 1, tzinfo=timezone.utc).timestamp())
    expected_end = int(datetime(next_year, next_month, 1, tzinfo=timezone.utc).timestamp())
    if (len(timestamps) != (expected_end - expected_first) // 60
            or int(timestamps[0]) != expected_first
            or int(timestamps[-1]) != expected_end - 60
            or np.any(np.diff(timestamps) != 60)):
        raise ValueError("split must contain every UTC minute of the requested months")


def json_number(value: float) -> float | None:
    """JSON has no infinity; an all-winning profit factor is undefined numerically."""
    return float(value) if np.isfinite(value) else None


def transformer_predictions(ds: KlinesDataset, positions: np.ndarray, cfg: dict, checkpoint: Path) -> np.ndarray:
    model_cfg = cfg["model"]
    model = PriceTransformer(**{
        key: model_cfg[key] for key in (
            "delta_vocab_size", "bucket_vocab_size", "delta_emb_dim", "bucket_emb_dim",
            "hidden_dim", "num_layers", "num_heads", "ffn_dim", "num_classes"
        )
    }, dropout=0.0, seq_len=ds.seq_len)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_positions in np.array_split(positions, max(1, (len(positions) + 63) // 64)):
            if not len(batch_positions):
                continue
            tokens = [ds[int(pos)][:3] for pos in batch_positions]
            streams = [torch.as_tensor(np.stack([row[i] for row in tokens]), dtype=torch.long) for i in range(3)]
            predictions.extend(model(*streams).argmax(dim=-1).numpy().tolist())
    return np.asarray(predictions, dtype=np.int8)


def summarize(predictions: np.ndarray, labels: np.ndarray, closes: np.ndarray, opens: np.ndarray,
              signals: np.ndarray, cfg: dict) -> dict:
    counts = np.bincount(predictions, minlength=3)
    scenarios = {}
    bt = cfg["backtest"]
    slip = bt.get("slippage", 0.0005)
    for name, commission, slippage in (
        ("commission_only", bt["commission"], 0.0),
        ("slippage_5bp", bt["commission"], slip),
        ("stress_8bp_10bp", 2 * bt["commission"], 2 * slip),
    ):
        result = BacktestEngine(
            commission=commission, stop_loss=bt["stop_loss"],
            take_profit=bt["take_profit"], max_hold=bt["max_hold"], slippage=slippage,
        ).run(closes, predictions, opens=opens, signal_indices=signals)
        scenarios[name] = {
            "commission_per_side": commission, "slippage_per_side": slippage,
            "total_pnl": result.total_pnl, "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown, "win_rate": result.win_rate,
            "trade_count": result.trade_count, "profit_factor": json_number(result.profit_factor),
            "avg_duration_candles": result.avg_duration,
        }
    return {
        "prediction_counts": dict(zip(LABELS, counts.tolist())),
        "accuracy": float(np.mean(predictions == labels)),
        "backtest": scenarios,
    }


def probability_scores(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Proper scores for soft labels; lower is better (uniform Brier = 2/3)."""
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.shape != (len(labels), len(LABELS)) or not len(labels)
            or not np.all(np.isfinite(values)) or np.any(values < 0)
            or np.any(values > 1)):
        raise ValueError("invalid teacher probability matrix")
    totals = values.sum(axis=1)
    if np.any(np.abs(totals - 1) > 0.02):
        raise ValueError("teacher probabilities do not sum to one")
    values = values / totals[:, None]
    truth = np.eye(len(LABELS))[labels]
    return {
        "multiclass_brier": float(np.mean(np.sum((values - truth) ** 2, axis=1))),
        "log_loss": float(np.mean(-np.log(np.maximum(values[np.arange(len(labels)), labels], 1e-12)))),
    }


def experiment_for_order(base: dict, order: tuple[str, ...]) -> dict:
    experiment = dict(base)
    if order != LABELS:
        # Keep the original order's experiment ID compatible with existing caches.
        experiment["criteria_order"] = list(order)
    return experiment


def ensemble_probabilities(caches: list[dict[int, dict]], signals: np.ndarray,
                           timestamps: np.ndarray, state_hashes: dict[int, str]) -> np.ndarray:
    """Align three complete caches by signal index and average class probabilities."""
    if len(caches) != len(CRITERIA_ORDERS):
        raise ValueError("ensemble requires exactly three order caches")
    combined = []
    for signal in signals:
        index = int(signal)
        distributions = []
        for cache in caches:
            row = cache.get(index)
            if (row is None or row.get("status") != "ok"
                    or row.get("signal_timestamp") != int(timestamps[index])
                    or row.get("state_sha256") != state_hashes[index]):
                raise ValueError(f"missing, failed, or misaligned cache row for signal {index}")
            choice = row.get("prediction")
            if type(choice) is not int or choice not in (0, 1, 2):
                raise ValueError(f"invalid prediction for signal {index}")
            try:
                _, probabilities = parse_choice({"answers": {"decision": {
                    "type": "choice", "choice": LABELS[choice],
                    "probabilities": row.get("probabilities"),
                }}})
            except ValueError as exc:
                raise ValueError(f"invalid probabilities for signal {index}") from exc
            distributions.append([probabilities[label] for label in LABELS])
        combined.append(np.mean(distributions, axis=0))
    return np.asarray(combined, dtype=np.float64)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            for row in rows:
                stream.write(canonical_bytes(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def evaluate_ensemble(args: argparse.Namespace, base_experiment: dict, total_hourly: int,
                      signals: np.ndarray, all_signals: np.ndarray, ds: KlinesDataset,
                      opens: np.ndarray, positions: np.ndarray, cfg: dict) -> None:
    source_ids = []
    caches = []
    source_summaries = []
    suffix = f"partial-{len(signals)}" if len(signals) < total_hourly else "full"
    for order, source_dir in zip(CRITERIA_ORDERS, args.ensemble_from):
        experiment = experiment_for_order(base_experiment, order)
        experiment_id = hashlib.sha256(canonical_bytes(experiment)).hexdigest()
        manifest = json.loads((source_dir / "manifest.json").read_text())
        expected_manifest = {"experiment_id": experiment_id, **experiment,
                             "total_hourly_signals": total_hourly}
        if manifest != expected_manifest:
            raise ValueError(f"ensemble source manifest does not match data, teacher or order: {source_dir}")
        cache_path = source_dir / "decisions.jsonl"
        if not cache_path.exists() or not cache_path.read_bytes().endswith(b"\n"):
            raise ValueError(f"incomplete cache: {cache_path}")
        cache = read_cache(cache_path, experiment_id, all_signals, ds.timestamps)
        source_summary = json.loads((source_dir / f"summary-{suffix}.json").read_text())
        if (source_summary.get("experiment_id") != experiment_id
                or source_summary.get("selected_signals") != len(signals)
                or source_summary.get("total_hourly_signals") != total_hourly
                or source_summary.get("teacher_status_counts") != {"ok": len(signals)}):
            raise ValueError(f"incomplete or misaligned source summary: {source_dir}")
        source_ids.append(experiment_id)
        caches.append(cache)
        source_summaries.append(source_summary)

    labels = ds.labels()[positions]
    label_counts = dict(zip(LABELS, np.bincount(labels, minlength=3).tolist()))
    baseline_names = ("flat", "long", "short", "momentum_60m", "contrarian_60m", "transformer")
    baseline_results = {name: source_summaries[0]["results"][name] for name in baseline_names}
    for source_summary in source_summaries:
        if (source_summary.get("label_counts") != label_counts
                or any(source_summary["results"][name] != baseline_results[name] for name in baseline_names)):
            raise ValueError("source summaries disagree on labels or baselines")

    state_hashes = {int(index): hashlib.sha256(make_state(ds, int(index), opens).encode()).hexdigest()
                    for index in signals}
    probabilities = ensemble_probabilities(caches, signals, ds.timestamps, state_hashes)
    teacher = probabilities.argmax(axis=1).astype(np.int8)
    ensemble_id = hashlib.sha256(canonical_bytes({"source_experiment_ids": source_ids})).hexdigest()
    run_dir = args.output_dir / f"{args.split}-ensemble-{ensemble_id[:16]}"
    ensemble_manifest = {
        "experiment_id": ensemble_id, "method": "mean_class_probabilities_argmax",
        "criteria_orders": [list(order) for order in CRITERIA_ORDERS],
        "source_experiment_ids": source_ids,
        "source_run_dirs": [str(path.resolve()) for path in args.ensemble_from],
        "shared_experiment": base_experiment,
        "total_hourly_signals": total_hourly,
    }
    teacher_result = summarize(teacher, labels, ds.closes, opens, signals, cfg)
    teacher_result["probability_scores"] = probability_scores(probabilities, labels)
    summary = {
        "experiment_id": ensemble_id, "source_experiment_ids": source_ids,
        "split": args.split, "partial": len(signals) < total_hourly,
        "selected_signals": len(signals), "total_hourly_signals": total_hourly,
        "teacher_status_counts": {"ok": len(signals)}, "label_counts": label_counts,
        "results": {"teacher": teacher_result,
                    **baseline_results},
    }
    decisions = [{"signal_index": int(index), "signal_timestamp": int(ds.timestamps[index]),
                  "state_sha256": state_hashes[int(index)],
                  "probabilities": dict(zip(LABELS, row.tolist())),
                  "prediction": int(prediction)}
                 for index, row, prediction in zip(signals, probabilities, teacher)]
    atomic_json(run_dir / "manifest.json", ensemble_manifest)
    write_jsonl_atomic(run_dir / "decisions.jsonl", decisions)
    output = run_dir / f"summary-{suffix}.json"
    atomic_json(output, summary)
    print(f"{len(signals)}/{total_hourly} hourly signals; three-order ensemble")
    print(f"Summary: {output}")
    print(f"Cache: {run_dir / 'decisions.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-id", required=True,
                        help="Operator-verified served checkpoint/revision for the cache key")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8008/v1/systemone")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/teacher_eval"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-signals", type=int, help="Only first N hourly signals; partial smoke run")
    parser.add_argument("--criteria-order", choices=tuple(",".join(order) for order in CRITERIA_ORDERS),
                        default=",".join(LABELS), metavar="ORDER",
                        help="DOWN,FLAT,UP (default), FLAT,UP,DOWN or UP,DOWN,FLAT")
    parser.add_argument("--all-orders", action="store_true",
                        help="Query all three orders per state, preserving each order's separate cache")
    parser.add_argument("--ensemble-from", type=Path, nargs=3, metavar=("NORMAL_DIR", "FLAT_FIRST_DIR", "UP_FIRST_DIR"),
                        help="Combine three completed order caches without teacher or Transformer inference")
    args = parser.parse_args()
    if args.max_signals is not None and args.max_signals < 1:
        parser.error("--max-signals must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (args.ensemble_from or args.all_orders) and args.criteria_order != ",".join(LABELS):
        parser.error("--criteria-order cannot be combined with --ensemble-from or --all-orders")
    if args.ensemble_from and args.all_orders:
        parser.error("--ensemble-from cannot be combined with --all-orders")
    order = tuple(args.criteria_order.split(","))

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["data"]["symbol"] != "BTCUSDT" or cfg["data"]["timeframe"] != "1m":
        raise ValueError("this experiment requires BTCUSDT 1m data")
    if cfg["sequence"]["length"] != 128 or cfg["sequence"]["target_horizon"] != 60 or cfg["sequence"]["target_threshold"] != 0.0015:
        raise ValueError("this experiment requires 128/60 windows and +/-0.15% labels")
    data_dir = args.data_dir or Path(cfg["data"]["data_dir"])
    files = make_split(data_dir, *cfg["data"][f"{args.split}_months"], symbol="BTCUSDT")
    months = expected_months(*cfg["data"][f"{args.split}_months"])
    if [path.stem for path in files] != months:
        raise ValueError(f"incomplete {args.split} parquet coverage: expected {months}, found {[path.stem for path in files]}")
    for path, month in zip(files, months):
        validate_split_coverage(pq.read_table(path, columns=["timestamp"])["timestamp"].to_numpy(), [month])
    checkpoint = args.checkpoint.resolve()
    token_paths = [checkpoint.parent / "volatility.npy", checkpoint.parent / "volume.npy"]
    hashes = {
        "data_files": {path.name: sha256_file(path) for path in files},
        "checkpoint": sha256_file(checkpoint),
        "tokenizers": {path.name: sha256_file(path) for path in token_paths},
    }
    tok = cfg["tokenizer"]
    ds = KlinesDataset(
        files, seq_len=128, target_horizon=60, target_threshold=0.0015,
        range_pct=tok["delta"]["range_pct"], step_pct=tok["delta"]["step_pct"],
        n_bins=tok["bucket"]["n_bins"],
        tokenizers=load_tokenizers(checkpoint.parent, tok["delta"]["range_pct"],
                                   tok["delta"]["step_pct"], tok["bucket"]["n_bins"]),
    )
    validate_split_coverage(ds.timestamps, months)
    all_positions = hourly_sample_positions(ds)
    total_hourly = len(all_positions)
    positions = all_positions
    if args.max_signals is not None:
        positions = positions[:args.max_signals]
    if not len(positions):
        raise ValueError("no complete hourly windows in split")
    signals = ds.sample_starts[positions] + ds.seq_len - 1
    all_signals = ds.sample_starts[all_positions] + ds.seq_len - 1
    bars = [pq.read_table(path, columns=["open"]) for path in files]
    opens = np.concatenate([bar["open"].to_numpy() for bar in bars])
    base_experiment = {
        "prompt_version": PROMPT_VERSION, "state_schema": STATE_SCHEMA,
        "model": MODEL_ID, "teacher_id": args.teacher_id,
        "question": QUESTION, "endpoint": args.endpoint,
        "config": cfg, "split": args.split, "months": months, "hashes": hashes,
    }
    if args.ensemble_from:
        evaluate_ensemble(args, base_experiment, total_hourly, signals, all_signals,
                          ds, opens, positions, cfg)
        return
    runs = []
    for selected_order in (CRITERIA_ORDERS if args.all_orders else (order,)):
        experiment = experiment_for_order(base_experiment, selected_order)
        experiment_id = hashlib.sha256(canonical_bytes(experiment)).hexdigest()
        run_dir = args.output_dir / f"{args.split}-{experiment_id[:16]}"
        atomic_json(run_dir / "manifest.json", {"experiment_id": experiment_id, **experiment,
                                                "total_hourly_signals": total_hourly})
        cache_path = run_dir / "decisions.jsonl"
        runs.append({"order": selected_order, "experiment_id": experiment_id,
                     "run_dir": run_dir, "cache_path": cache_path,
                     "cached": read_cache(cache_path, experiment_id, all_signals, ds.timestamps),
                     "consecutive_failures": 0})
    for index in signals:
        index = int(index)
        state = None
        state_hash = None
        for run in runs:
            cached = run["cached"]
            if index in cached and cached[index]["status"] == "ok":
                continue
            if state is None:
                state = make_state(ds, index, opens)
                state_hash = hashlib.sha256(state.encode()).hexdigest()
            prediction, probabilities, status = query_teacher(
                args.endpoint, state, args.timeout, run["order"])
            cached[index] = {
                "experiment_id": run["experiment_id"], "prompt_version": PROMPT_VERSION,
                "signal_index": index, "signal_timestamp": int(ds.timestamps[index]),
                "state_sha256": state_hash, "prediction": prediction,
                "probabilities": probabilities, "status": status,
            }
            append_jsonl(run["cache_path"], cached[index])
            if status != "ok":
                run["consecutive_failures"] += 1
                if run["consecutive_failures"] >= 3:
                    raise RuntimeError("three consecutive teacher failures; resumable cache retained")
            else:
                run["consecutive_failures"] = 0
            if len(cached) % 25 == 0:
                print(f"{len(cached)}/{total_hourly} decisions cached for {','.join(run['order'])}", flush=True)

    labels = ds.labels()[positions]
    momentum = np.where(ds.closes[signals] > ds.closes[signals - 60], 2,
                        np.where(ds.closes[signals] < ds.closes[signals - 60], 0, 1)).astype(np.int8)
    contrarian = np.where(momentum == 2, 0, np.where(momentum == 0, 2, 1)).astype(np.int8)
    baselines = {
        "flat": np.full(len(signals), 1, dtype=np.int8),
        "long": np.full(len(signals), 2, dtype=np.int8),
        "short": np.full(len(signals), 0, dtype=np.int8),
        "momentum_60m": momentum,
        "contrarian_60m": contrarian,
        "transformer": transformer_predictions(ds, positions, cfg, checkpoint),
    }
    baseline_results = {name: summarize(values, labels, ds.closes, opens, signals, cfg)
                        for name, values in baselines.items()}
    for run in runs:
        cached = run["cached"]
        teacher = np.asarray([cached[int(i)]["prediction"] for i in signals], dtype=np.int8)
        teacher_result = summarize(teacher, labels, ds.closes, opens, signals, cfg)
        if all(cached[int(i)]["status"] == "ok" for i in signals):
            probabilities = np.asarray([[cached[int(i)]["probabilities"][label] for label in LABELS]
                                        for i in signals], dtype=np.float64)
            teacher_result["probability_scores"] = probability_scores(probabilities, labels)
        summary = {
            "experiment_id": run["experiment_id"], "split": args.split,
            "partial": args.max_signals is not None and len(signals) < total_hourly,
            "selected_signals": len(signals), "total_hourly_signals": total_hourly,
            "teacher_status_counts": dict(Counter(cached[int(i)]["status"] for i in signals)),
            "label_counts": dict(zip(LABELS, np.bincount(labels, minlength=3).tolist())),
            "results": {"teacher": teacher_result, **baseline_results},
        }
        suffix = f"partial-{len(signals)}" if summary["partial"] else "full"
        output = run["run_dir"] / f"summary-{suffix}.json"
        atomic_json(output, summary)
        print(f"{len(signals)}/{total_hourly} hourly signals; order={','.join(run['order'])}; "
              f"status={summary['teacher_status_counts']}")
        print(f"Summary: {output}")
        print(f"Cache: {run['cache_path']}")


if __name__ == "__main__":
    main()
