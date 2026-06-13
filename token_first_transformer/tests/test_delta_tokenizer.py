import numpy as np
import pytest
from tokenizer.delta_tokenizer import DeltaTokenizer


def test_vocab_size():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    assert tok.vocab_size == 122


def test_pad_and_cls_ids():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    assert tok.pad_id == 0
    assert tok.cls_id == 1


def test_zero_delta_maps_to_middle_bin():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    token_id = tok.encode_single(0.0)
    assert token_id == 1 + 60


def test_positive_delta_encodes():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    token_id = tok.encode_single(0.0005)
    assert token_id == 62


def test_negative_delta_encodes():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    token_id = tok.encode_single(-0.0005)
    assert token_id == 60


def test_out_of_range_clips():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    high = tok.encode_single(0.05)
    assert high == tok.vocab_size - 1
    low = tok.encode_single(-0.05)
    assert low == 2


def test_encode_batch():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    deltas = np.array([0.0, 0.0005, -0.0005, 0.05, -0.05], dtype=np.float32)
    ids = tok.encode_batch(deltas)
    assert ids.shape == (5,)
    assert ids[0] == 1 + 60
    assert ids[1] == 62
    assert ids[2] == 60
    assert ids[3] == tok.vocab_size - 1
    assert ids[4] == 2


def test_from_closes():
    tok = DeltaTokenizer(range_pct=3.0, step_pct=0.05)
    closes = np.array([100.0, 101.0, 99.0, 100.5, 100.5], dtype=np.float32)
    ids = tok.from_closes(closes)
    assert ids[0] == tok.pad_id
    assert len(ids) == 5
    assert ids[4] == 1 + 60
