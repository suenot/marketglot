import io
import json
import urllib.error
from pathlib import Path

import pytest

from warehouse import client as client_mod
from warehouse.client import WarehouseClient


class FakeResponse:
    """Minimal stand-in for the object returned by urlopen."""

    def __init__(self, body: bytes):
        self._buf = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read() if n is None or n < 0 else self._buf.read(n)

    def close(self) -> None:
        self._buf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def cli():
    return WarehouseClient(
        api_base="https://api.example.com/api/v1/",
        s3_base="https://s3.example.com/bucket/",
    )


def test_base_urls_stripped(cli):
    assert cli.api_base == "https://api.example.com/api/v1"
    assert cli.s3_base == "https://s3.example.com/bucket"


def test_file_url_construction(cli):
    url = cli.file_url("XRPUSDT", "bybit", "2026-06-09", "12_delta.parquet.zst")
    assert url == (
        "https://s3.example.com/bucket/XRPUSDT/orderbook/bybit/"
        "2026-06-09/12_delta.parquet.zst"
    )


def test_get_orderbook_info(monkeypatch, cli):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        body = json.dumps(
            {"exchanges": {"bybit": {"days": 9, "first": "2026-06-01", "last": "2026-06-09"}}}
        ).encode()
        return FakeResponse(body)

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake_urlopen)
    info = cli.get_orderbook_info("XRPUSDT")
    assert captured["url"] == "https://api.example.com/api/v1/data/XRPUSDT/orderbook"
    assert info["bybit"]["days"] == 9


def test_list_days(monkeypatch, cli):
    def fake_urlopen(req, timeout=None):
        return FakeResponse(json.dumps({"days": ["2026-06-02", "2026-06-01"]}).encode())

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake_urlopen)
    days = cli.list_days("XRPUSDT", "bybit")
    assert days == ["2026-06-01", "2026-06-02"]  # sorted


def test_list_files(monkeypatch, cli):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        files = [
            {"file": "12_delta.parquet.zst", "size_bytes": 100, "url": "u1"},
            {"file": "12_snapshot.parquet.zst", "size_bytes": 50, "url": "u2"},
        ]
        return FakeResponse(json.dumps(files).encode())

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake_urlopen)
    files = cli.list_files("XRPUSDT", "bybit", "2026-06-09")
    assert captured["url"] == (
        "https://api.example.com/api/v1/data/XRPUSDT/orderbook/bybit/2026-06-09"
    )
    assert len(files) == 2
    assert files[0]["file"] == "12_delta.parquet.zst"


def test_download_day_writes_files_and_paths(monkeypatch, tmp_path, cli):
    listing = [
        {"file": "00_snapshot.parquet.zst", "size_bytes": 4, "url": "https://x/snap"},
        {"file": "00_delta.parquet.zst", "size_bytes": 4, "url": "https://x/delta"},
    ]
    monkeypatch.setattr(cli, "list_files", lambda *a, **k: listing)

    bodies = {"https://x/snap": b"SNAP", "https://x/delta": b"DELT"}

    def fake_urlopen(req, timeout=None):
        return FakeResponse(bodies[req.full_url])

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake_urlopen)

    paths = cli.download_day("XRPUSDT", "bybit", "2026-06-09", tmp_path)
    day_dir = tmp_path / "XRPUSDT" / "bybit" / "2026-06-09"
    assert sorted(p.name for p in paths) == [
        "00_delta.parquet.zst",
        "00_snapshot.parquet.zst",
    ]
    assert (day_dir / "00_snapshot.parquet.zst").read_bytes() == b"SNAP"
    assert (day_dir / "00_delta.parquet.zst").read_bytes() == b"DELT"
    # No leftover .part temp files.
    assert not list(day_dir.glob("*.part"))


def test_download_day_skip_existing(monkeypatch, tmp_path, cli):
    listing = [{"file": "00_delta.parquet.zst", "size_bytes": 4, "url": "https://x/d"}]
    monkeypatch.setattr(cli, "list_files", lambda *a, **k: listing)

    day_dir = tmp_path / "XRPUSDT" / "bybit" / "2026-06-09"
    day_dir.mkdir(parents=True)
    existing = day_dir / "00_delta.parquet.zst"
    existing.write_bytes(b"OLD")

    def fail_urlopen(req, timeout=None):
        raise AssertionError("should not download when skip_existing keeps file")

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fail_urlopen)
    paths = cli.download_day("XRPUSDT", "bybit", "2026-06-09", tmp_path, skip_existing=True)
    assert paths == [existing]
    assert existing.read_bytes() == b"OLD"  # untouched


def test_download_day_no_skip_redownloads(monkeypatch, tmp_path, cli):
    listing = [{"file": "00_delta.parquet.zst", "size_bytes": 4, "url": "https://x/d"}]
    monkeypatch.setattr(cli, "list_files", lambda *a, **k: listing)

    day_dir = tmp_path / "XRPUSDT" / "bybit" / "2026-06-09"
    day_dir.mkdir(parents=True)
    existing = day_dir / "00_delta.parquet.zst"
    existing.write_bytes(b"OLD")

    monkeypatch.setattr(
        client_mod.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(b"NEW")
    )
    cli.download_day("XRPUSDT", "bybit", "2026-06-09", tmp_path, skip_existing=False)
    assert existing.read_bytes() == b"NEW"


def test_retry_on_5xx_then_success(monkeypatch, cli):
    # Avoid real sleeping between retries.
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return FakeResponse(json.dumps({"exchanges": {}}).encode())

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", flaky_urlopen)
    info = cli.get_orderbook_info("XRPUSDT")
    assert info == {}
    assert calls["n"] == 2


def test_no_retry_on_4xx(monkeypatch, cli):
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def urlopen_404(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", urlopen_404)
    with pytest.raises(urllib.error.HTTPError):
        cli.get_orderbook_info("XRPUSDT")
    assert calls["n"] == 1  # no retries on client error


def test_retry_exhausted_reraises(monkeypatch, cli):
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)

    def always_500(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", always_500)
    with pytest.raises(urllib.error.HTTPError):
        cli.get_orderbook_info("XRPUSDT")
