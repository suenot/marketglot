"""HTTP client for the warehouse market-data service.

Uses only the standard library (``urllib.request``); no third-party HTTP
dependency. Provides metadata lookups and an atomic, resumable day download.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Read body in 1 MiB chunks while streaming a file to disk.
_CHUNK_SIZE = 1024 * 1024
# Back-off pauses (seconds) before retry attempts 1, 2 and 3.
_RETRY_PAUSES = (10, 30, 60)
_TIMEOUT_SEC = 60


class WarehouseClient:
    """Client for the warehouse REST API and anonymous S3 file store."""

    def __init__(self, api_base: str, s3_base: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.s3_base = s3_base.rstrip("/")

    # -- metadata ---------------------------------------------------------

    def get_orderbook_info(self, symbol: str) -> dict:
        """Return ``{exchange: {days, first, last}}`` for ``symbol``."""
        url = f"{self.api_base}/data/{symbol}/orderbook"
        return self._get_json(url).get("exchanges", {})

    def list_days(self, symbol: str, exchange: str) -> list[str]:
        """Return the sorted list of available ``YYYY-MM-DD`` days."""
        url = f"{self.api_base}/data/{symbol}/orderbook/{exchange}"
        data = self._get_json(url)
        if isinstance(data, dict):
            days = data.get("days", [])
        else:
            days = data
        return sorted(str(d) for d in days)

    def list_files(self, symbol: str, exchange: str, date: str) -> list[dict]:
        """Return ``[{file, size_bytes, url}, ...]`` for one day."""
        url = f"{self.api_base}/data/{symbol}/orderbook/{exchange}/{date}"
        data = self._get_json(url)
        if isinstance(data, dict):
            return data.get("files", [])
        return data

    # -- download ---------------------------------------------------------

    def download_day(
        self,
        symbol: str,
        exchange: str,
        date: str,
        dest_root: Path,
        skip_existing: bool = True,
    ) -> list[Path]:
        """Download every file of one day into ``dest_root``.

        Files land at ``{dest_root}/{symbol}/{exchange}/{date}/{filename}``.
        Each file is streamed to a ``.part`` temp file and atomically moved
        with ``os.replace``. A non-empty existing file is skipped when
        ``skip_existing`` is true. Returns the list of local paths present
        after the call (downloaded or already there).
        """
        dest_root = Path(dest_root)
        day_dir = dest_root / symbol / exchange / date
        day_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for entry in self.list_files(symbol, exchange, date):
            filename = entry["file"]
            url = entry["url"]
            dest = day_dir / filename
            if skip_existing and dest.exists() and dest.stat().st_size > 0:
                paths.append(dest)
                continue
            self._download_file(url, dest)
            paths.append(dest)
        return paths

    def file_url(self, symbol: str, exchange: str, date: str, filename: str) -> str:
        """Build the canonical S3 URL for one file (used when no API list)."""
        return f"{self.s3_base}/{symbol}/orderbook/{exchange}/{date}/{filename}"

    # -- internals --------------------------------------------------------

    def _get_json(self, url: str):
        raw = self._request_with_retry(lambda: self._urlopen(url))
        return json.loads(raw.read().decode("utf-8"))

    def _download_file(self, url: str, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".part")

        def _stream() -> None:
            resp = self._urlopen(url)
            try:
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
            finally:
                resp.close()

        self._request_with_retry(_stream)
        os.replace(tmp, dest)

    def _urlopen(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "orderbook-encoder/0.1"})
        return urllib.request.urlopen(req, timeout=_TIMEOUT_SEC)

    @staticmethod
    def _request_with_retry(call):
        """Run ``call`` retrying on 5xx HTTP errors and timeouts.

        Up to three attempts with the pauses from ``_RETRY_PAUSES``. Returns
        whatever ``call`` returns; re-raises the last error on exhaustion.
        """
        last_err: Exception | None = None
        for attempt in range(len(_RETRY_PAUSES) + 1):
            try:
                return call()
            except urllib.error.HTTPError as err:
                last_err = err
                if err.code < 500:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_err = err
            if attempt < len(_RETRY_PAUSES):
                time.sleep(_RETRY_PAUSES[attempt])
        assert last_err is not None
        raise last_err
