import io
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_teacher import (
    CRITERIA_ORDERS, LABELS, MODEL_ID, QUESTION, append_jsonl, canonical_bytes,
    ensemble_probabilities, expected_months, experiment_for_order,
    hourly_sample_positions, make_state, parse_choice, probability_scores,
    query_teacher, read_cache, request_bytes, summarize, validate_split_coverage,
)


def test_hourly_grid_uses_completed_candle_and_valid_windows():
    ds = SimpleNamespace(
        seq_len=128,
        timestamps=np.arange(400, dtype=np.int64) * 60,
        # These are prevalidated, complete 128+60 windows from KlinesDataset.
        sample_starts=np.array([0, 52, 53, 112, 172], dtype=np.int64),
    )
    np.testing.assert_array_equal(hourly_sample_positions(ds), [1, 3, 4])


def test_state_excludes_future_candles():
    ds = SimpleNamespace(
        seq_len=128,
        timestamps=np.arange(200, dtype=np.int64) * 60,
        highs=np.arange(200, dtype=np.float32) + 2,
        lows=np.arange(200, dtype=np.float32),
        closes=np.arange(200, dtype=np.float32) + 1,
        volumes=np.ones(200, dtype=np.float32),
    )
    opens = np.arange(200, dtype=np.float32) + 0.5
    before = make_state(ds, 127, opens)
    ds.highs[128:] = ds.lows[128:] = ds.closes[128:] = ds.volumes[128:] = 99999
    opens[128:] = 99999
    assert make_state(ds, 127, opens) == before
    state = json.loads(before)
    assert set(state["close_return_pct"]) == {"1", "5", "15", "30", "60", "120"}
    assert "signal_close_utc" not in state
    assert "candles_1m" not in state
    assert len(before) < 600


def test_month_coverage_and_interrupted_cache_tail(tmp_path):
    assert expected_months("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    path = tmp_path / "decisions.jsonl"
    row = {"experiment_id": "experiment", "signal_index": 5, "signal_timestamp": 300,
           "prediction": 1, "probabilities": None, "status": "timeout"}
    append_jsonl(path, row)
    with path.open("ab") as stream:
        stream.write(b'{"interrupted":')
    assert read_cache(path, "experiment", np.array([5]), np.arange(10) * 60) == {5: row}
    assert path.read_bytes().endswith(b"\n")


def test_split_coverage_rejects_truncated_but_continuous_month():
    start = int(datetime(2025, 2, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp())
    timestamps = np.arange(start, end, 60, dtype=np.int64)
    validate_split_coverage(timestamps, ["2025-02"])
    for incomplete in (timestamps[1:], timestamps[:-1], np.delete(timestamps, 500)):
        with pytest.raises(ValueError, match="every UTC minute"):
            validate_split_coverage(incomplete, ["2025-02"])


def test_choice_schema_and_invalid_fallback_inputs():
    valid = {"answers": {"decision": {
        "type": "choice", "choice": "DOWN",
        "probabilities": {"DOWN": 0.6, "FLAT": 0.3, "UP": 0.1},
    }}}
    assert parse_choice(valid) == (0, {"DOWN": 0.6, "FLAT": 0.3, "UP": 0.1})
    for invalid in (
        {},
        {"answers": {"decision": {"type": "choice", "choice": "UP"}}},
        {"answers": {"decision": {"type": "choice", "choice": "UP",
                                   "probabilities": {"DOWN": 0.2, "FLAT": 0.3, "UP": float("nan")}}}},
    ):
        with pytest.raises(ValueError):
            parse_choice(invalid)


def test_teacher_abstains_on_bad_schema_and_timeout(monkeypatch):
    monkeypatch.setattr("scripts.evaluate_teacher.request.urlopen",
                        lambda _request, timeout: io.BytesIO(b'{"answers":{}}'))
    assert query_teacher("http://127.0.0.1:8008/v1/systemone", "state", 1) == (1, None, "invalid_response")

    def timed_out(_request, timeout):
        raise TimeoutError()

    monkeypatch.setattr("scripts.evaluate_teacher.request.urlopen", timed_out)
    assert query_teacher("http://127.0.0.1:8008/v1/systemone", "state", 1) == (1, None, "timeout")


@pytest.mark.parametrize("order", CRITERIA_ORDERS)
def test_http_request_preserves_criteria_order(monkeypatch, order):
    bodies = []

    def respond(http_request, timeout):
        bodies.append(http_request.data)
        return io.BytesIO(b'{"answers":{"decision":{"type":"choice","choice":"FLAT",'
                          b'"probabilities":{"DOWN":0.2,"FLAT":0.6,"UP":0.2}}}}')

    monkeypatch.setattr("scripts.evaluate_teacher.request.urlopen", respond)
    assert query_teacher("http://127.0.0.1:8008/v1/systemone", "state", 1, order)[2] == "ok"
    assert list(json.loads(bodies[0])["questions"]["decision"]["criteria"]) == list(order)
    assert experiment_for_order({"question": QUESTION}, LABELS) == {"question": QUESTION}
    if order != LABELS:
        assert canonical_bytes(experiment_for_order({"question": QUESTION}, order)) != canonical_bytes({"question": QUESTION})


def test_default_request_bytes_match_existing_cache_protocol():
    original = {"state": "state", "model": MODEL_ID, "questions": {"decision": QUESTION}}
    assert request_bytes("state", LABELS) == canonical_bytes(original)


def test_ensemble_aligns_by_signal_and_rejects_missing_or_error_rows():
    signals = np.array([2, 4])
    timestamps = np.arange(5) * 60
    hashes = {2: "state-2", 4: "state-4"}

    def row(index, down, flat, up):
        return {"signal_index": index, "signal_timestamp": int(timestamps[index]),
                "state_sha256": hashes[index], "prediction": 1, "status": "ok",
                "probabilities": {"DOWN": down, "FLAT": flat, "UP": up}}

    caches = [
        {4: row(4, 0.1, 0.2, 0.7), 2: row(2, 0.8, 0.1, 0.1)},
        {2: row(2, 0.1, 0.7, 0.2), 4: row(4, 0.1, 0.1, 0.8)},
        {4: row(4, 0.7, 0.2, 0.1), 2: row(2, 0.3, 0.2, 0.5)},
    ]
    means = ensemble_probabilities(caches, signals, timestamps, hashes)
    np.testing.assert_allclose(means, [[0.4, 1 / 3, 0.8 / 3], [0.3, 1 / 6, 1.6 / 3]])
    np.testing.assert_array_equal(means.argmax(axis=1), [0, 2])
    for bad in (None, {**caches[2][4], "status": "timeout"}):
        broken = [dict(cache) for cache in caches]
        if bad is None:
            del broken[2][4]
        else:
            broken[2][4] = bad
        with pytest.raises(ValueError, match="signal 4"):
            ensemble_probabilities(broken, signals, timestamps, hashes)


def test_summary_enters_on_next_open():
    config = {"backtest": {"commission": 0.0004, "stop_loss": -1,
                           "take_profit": 1, "max_hold": 1}}
    result = summarize(
        predictions=np.array([2]), labels=np.array([2]),
        closes=np.array([100., 101., 120., 120.]),
        opens=np.array([100., 101., 120., 120.]),
        signals=np.array([1]), cfg=config,
    )
    assert result["backtest"]["commission_only"]["trade_count"] == 1
    assert result["backtest"]["commission_only"]["total_pnl"] == pytest.approx((1 - 0.0004) ** 2 - 1)
    assert result["backtest"]["slippage_5bp"]["total_pnl"] < result["backtest"]["commission_only"]["total_pnl"]
    assert result["backtest"]["stress_8bp_10bp"]["commission_per_side"] == pytest.approx(0.0008)
    assert result["backtest"]["stress_8bp_10bp"]["slippage_per_side"] == pytest.approx(0.001)


def test_soft_label_scores_normalize_rounded_probabilities():
    labels = np.array([0, 2])
    perfect = probability_scores(np.array([[1., 0., 0.], [0., 0., 1.]]), labels)
    assert perfect["multiclass_brier"] == 0
    assert perfect["log_loss"] == 0
    uniform = probability_scores(np.full((2, 3), 0.3333), labels)
    assert uniform["multiclass_brier"] == pytest.approx(2 / 3)
    assert uniform["log_loss"] == pytest.approx(np.log(3))
    with pytest.raises(ValueError, match="sum to one"):
        probability_scores(np.full((2, 3), 0.2), labels)
