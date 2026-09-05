"""Disk-backed search cache + curated demo results (stdlib only).

Two jobs:
1. Cache: identical queries reuse a stored result instead of re-hitting S2.
2. Demos: landing-page example queries always serve a curated snapshot, so
   they are instant, always impressive, and never spend a user's quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

DEMO_DIR = Path(__file__).resolve().parent / "data" / "demos"


def normalize_query(q: str) -> str:
    """Whitespace/case-insensitive key so trivial edits still hit cache."""
    return " ".join((q or "").split()).strip().lower()


def cache_key(query: str, **params: Any) -> str:
    payload = json.dumps(
        {"q": normalize_query(query), **{k: params[k] for k in sorted(params)}},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class SearchCache:
    """One JSON file per entry; TTL in seconds. Safe for the threaded server."""

    def __init__(self, path: Path, ttl_s: int = 7 * 86400, max_entries: int = 2000) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.ttl_s = int(ttl_s)
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()

    def _file(self, key: str) -> Path:
        return self.path / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        f = self._file(key)
        if not f.is_file():
            return None
        try:
            if self.ttl_s > 0 and time.time() - f.stat().st_mtime > self.ttl_s:
                f.unlink(missing_ok=True)
                return None
            with f.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        f = self._file(key)
        tmp = f.with_suffix(".tmp")
        try:
            with self._lock:
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(value, fh, ensure_ascii=False)
                tmp.replace(f)
                self._evict_if_needed()
        except OSError:
            pass

    def _evict_if_needed(self) -> None:
        """Drop oldest entries once the directory grows past max_entries."""
        try:
            files = sorted(
                self.path.glob("*.json"), key=lambda x: x.stat().st_mtime
            )
        except OSError:
            return
        excess = len(files) - self.max_entries
        for f in files[:excess] if excess > 0 else []:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

    def stats(self) -> dict[str, Any]:
        try:
            n = sum(1 for _ in self.path.glob("*.json"))
        except OSError:
            n = 0
        return {"entries": n, "ttl_s": self.ttl_s, "dir": str(self.path)}


class DemoStore:
    """Curated results for the landing-page example queries."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.dir = Path(directory) if directory else DEMO_DIR
        self._index: dict[str, Path] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        index: dict[str, Path] = {}
        if self.dir.is_dir():
            for f in sorted(self.dir.glob("*.json")):
                try:
                    with f.open(encoding="utf-8") as fh:
                        doc = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
                for q in doc.get("queries") or []:
                    index[normalize_query(str(q))] = f
        self._index = index
        self._cache = {}

    def match(self, query: str) -> Optional[str]:
        return normalize_query(query) if normalize_query(query) in self._index else None

    def get(self, query: str) -> Optional[dict[str, Any]]:
        key = normalize_query(query)
        f = self._index.get(key)
        if not f:
            return None
        if key in self._cache:
            return self._cache[key]
        try:
            with f.open(encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        payload = doc.get("payload") or {}
        self._cache[key] = payload
        return payload

    def queries(self) -> list[str]:
        return sorted(self._index.keys())
