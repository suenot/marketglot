import numpy as np
import pytest
from tokenizer.bucket_tokenizer import BucketTokenizer


def test_default_vocab_size():
    tok = BucketTokenizer(n_bins=8)
    assert tok.vocab_size == 10


def test_pad_id():
    tok = BucketTokenizer(n_bins=8)
    assert tok.pad_id == 0


def test_fit_creates_boundaries():
    tok = BucketTokenizer(n_bins=4)
    data = np.arange(100, dtype=np.float32)
    tok.fit(data)
    assert tok.boundaries is not None
    assert len(tok.boundaries) == 3


def test_encode_after_fit():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    tok.fit(data)
    ids = tok.encode_batch(data)
    assert ids.min() >= 2
    assert ids.max() <= 5
    assert ids[0] == 2
    assert ids[-1] == 5


def test_encode_single():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tok.fit(data)
    token = tok.encode_single(5.0)
    assert 2 <= token <= 5


def test_encode_below_min():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tok.fit(data)
    assert tok.encode_single(5.0) == 2


def test_encode_above_max():
    tok = BucketTokenizer(n_bins=4)
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tok.fit(data)
    assert tok.encode_single(50.0) == 5


def test_save_load(tmp_path):
    tok = BucketTokenizer(n_bins=4)
    data = np.arange(100, dtype=np.float32)
    tok.fit(data)
    path = tmp_path / "bounds.npy"
    tok.save(path)
    tok2 = BucketTokenizer(n_bins=4)
    tok2.load(path)
    np.testing.assert_array_equal(tok.boundaries, tok2.boundaries)
    sample = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    np.testing.assert_array_equal(tok.encode_batch(sample), tok2.encode_batch(sample))
