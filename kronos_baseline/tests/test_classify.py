"""Unit tests for the network-free parts of the Kronos baseline wrapper."""
import pytest

from kronos_baseline.kronos_signal import (
    DOWN, FLAT, UP, classify_return, _resolve_kronos_path,
)


def test_classify_return_thresholds():
    # threshold_pct=0.15  ->  deadband +-0.0015
    assert classify_return(0.0020, 0.15) == UP
    assert classify_return(-0.0020, 0.15) == DOWN
    assert classify_return(0.0005, 0.15) == FLAT
    assert classify_return(0.0, 0.15) == FLAT


def test_classify_return_boundary_is_exclusive():
    # exactly at the boundary stays FLAT (strict >, <)
    assert classify_return(0.0015, 0.15) == FLAT
    assert classify_return(-0.0015, 0.15) == FLAT
    assert classify_return(0.0015001, 0.15) == UP


def test_class_order_matches_repo_convention():
    assert (DOWN, FLAT, UP) == (0, 1, 2)


def test_resolve_kronos_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_kronos_path(str(tmp_path / "nope"))


def test_resolve_kronos_path_ok(tmp_path):
    (tmp_path / "model").mkdir()
    assert _resolve_kronos_path(str(tmp_path)) == tmp_path.resolve()
