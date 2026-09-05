"""Semantic Scholar API key pool + optional rotating HTTP proxy."""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse


def _split_keys(*blobs: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for blob in blobs:
        if not blob:
            continue
        for piece in str(blob).replace(";", ",").split(","):
            k = piece.strip().strip('"').strip("'")
            if not k or k in seen:
                continue
            # drop obvious non-keys
            if "KEY=" in k or k.endswith("=") or len(k) < 8:
                continue
            seen.add(k)
            out.append(k)
    return out


def load_s2_keys_from_env(
    extra: Optional[list[str]] = None,
) -> list[str]:
    keys = _split_keys(
        os.environ.get("SEMANTIC_SCHOLAR_API_KEYS", ""),
        os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        os.environ.get("S2_API_KEY", ""),
    )
    if extra:
        keys = _split_keys(",".join(keys), ",".join(extra))
    return keys


def key_fingerprint(key: str) -> str:
    k = key or ""
    if len(k) <= 10:
        return "***"
    return f"{k[:6]}…{k[-4:]}"


@dataclass
class KeyStat:
    key: str
    next_free: float = 0.0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0
    last_error: str = ""

    @property
    def fp(self) -> str:
        return key_fingerprint(self.key)


class S2KeyPool:
    """Per-key 1 rps pacing with least-wait selection + 429 cooldown."""

    def __init__(
        self,
        keys: list[str],
        *,
        min_interval: float = 1.05,
        rate_limit_cooldown: float = 8.0,
    ) -> None:
        if not keys:
            # anonymous single slot
            self._stats = [KeyStat(key="")]
        else:
            self._stats = [KeyStat(key=k) for k in keys]
        self.min_interval = float(min_interval)
        self.rate_limit_cooldown = float(rate_limit_cooldown)
        self._lock = threading.Lock()
        self._rr = 0

    @property
    def size(self) -> int:
        return len(self._stats)

    def snapshot(self) -> list[dict]:
        with self._lock:
            now = time.time()
            rows = []
            for s in self._stats:
                rows.append(
                    {
                        "fp": s.fp or "(anon)",
                        "successes": s.successes,
                        "failures": s.failures,
                        "rate_limits": s.rate_limits,
                        "wait_s": round(max(0.0, s.next_free - now), 3),
                        "last_error": s.last_error,
                    }
                )
            return rows

    def acquire(self) -> tuple[str, float]:
        """Pick a key and sleep until it is free. Returns (key, waited_s)."""
        with self._lock:
            now = time.time()
            # Prefer ready keys; else the soonest next_free.
            ready = [s for s in self._stats if s.next_free <= now]
            if ready:
                # round-robin among ready keys for even LB
                self._rr %= len(self._stats)
                # walk from rr to find a ready slot
                stat = None
                for i in range(len(self._stats)):
                    cand = self._stats[(self._rr + i) % len(self._stats)]
                    if cand.next_free <= now:
                        stat = cand
                        self._rr = (self._rr + i + 1) % len(self._stats)
                        break
                if stat is None:
                    stat = ready[0]
                wait = 0.0
            else:
                stat = min(self._stats, key=lambda s: s.next_free)
                wait = max(0.0, stat.next_free - now)
            # reserve slot immediately so concurrent callers don't collide
            start = max(now, stat.next_free)
            stat.next_free = start + self.min_interval
            key = stat.key
        if wait > 0:
            time.sleep(wait)
        return key, wait

    def mark_success(self, key: str) -> None:
        with self._lock:
            st = self._find(key)
            if st:
                st.successes += 1
                st.last_error = ""

    def mark_rate_limit(self, key: str, *, retry_after: Optional[float] = None) -> None:
        cool = retry_after if retry_after and retry_after > 0 else self.rate_limit_cooldown
        with self._lock:
            st = self._find(key)
            if st:
                st.rate_limits += 1
                st.failures += 1
                st.last_error = "429"
                st.next_free = max(st.next_free, time.time() + cool)

    def mark_failure(self, key: str, err: str) -> None:
        with self._lock:
            st = self._find(key)
            if st:
                st.failures += 1
                st.last_error = (err or "")[:120]
                # light penalty
                st.next_free = max(st.next_free, time.time() + 0.5)

    def _find(self, key: str) -> Optional[KeyStat]:
        for s in self._stats:
            if s.key == key:
                return s
        return self._stats[0] if self._stats else None


@dataclass
class ProxyPool:
    """Bright Data (or generic) proxy with per-request random session → random IP."""

    username: str
    password: str
    host: str = "brd.superproxy.io"
    port: int = 44445
    enabled: bool = True
    # When True, append -session-<rand> so DC zone picks a fresh peer IP.
    rotate_session: bool = True
    calls: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> Optional["ProxyPool"]:
        # Full URL wins: http://user:pass@host:port
        url = (os.environ.get("BRD_PROXY_URL") or os.environ.get("TOPPER_HTTP_PROXY") or "").strip()
        if url:
            p = urlparse(url)
            if p.hostname and p.username:
                return cls(
                    username=urllib_unquote(p.username),
                    password=urllib_unquote(p.password or ""),
                    host=p.hostname,
                    port=p.port or 44445,
                    enabled=os.environ.get("TOPPER_PROXY_ENABLED", "1") not in {"0", "false", "no"},
                    # Default OFF: still egress via BRD IP pool; session rotate
                    # only when you explicitly need a fresh peer every request.
                    rotate_session=os.environ.get("TOPPER_PROXY_ROTATE", "0")
                    in {"1", "true", "yes"},
                )
        user = (os.environ.get("BRD_PROXY_USER") or "").strip()
        passwd = (os.environ.get("BRD_PROXY_PASS") or "").strip()
        if not user or not passwd:
            return None
        host = (os.environ.get("BRD_PROXY_HOST") or "brd.superproxy.io").strip()
        port = int(os.environ.get("BRD_PROXY_PORT") or "44445")
        return cls(
            username=user,
            password=passwd,
            host=host,
            port=port,
            enabled=os.environ.get("TOPPER_PROXY_ENABLED", "1") not in {"0", "false", "no"},
            rotate_session=os.environ.get("TOPPER_PROXY_ROTATE", "0")
            in {"1", "true", "yes"},
        )

    def proxy_url(self) -> str:
        """Return a proxy URL; random session suffix when rotate_session."""
        user = self.username
        if self.rotate_session:
            # Avoid double-appending if caller already fixed a session.
            if "-session-" not in user:
                user = f"{user}-session-topper{random.randrange(1_000_000, 9_999_999)}"
        auth = f"{quote(user, safe='')}:{quote(self.password, safe='')}"
        return f"http://{auth}@{self.host}:{self.port}"

    def mark(self, ok: bool) -> None:
        with self._lock:
            self.calls += 1
            if not ok:
                self.failures += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "host": self.host,
                "port": self.port,
                "rotate_session": self.rotate_session,
                "user_prefix": self.username[:40],
                "calls": self.calls,
                "failures": self.failures,
            }


def urllib_unquote(s: str) -> str:
    from urllib.parse import unquote

    return unquote(s or "")


def build_opener(proxy_url: Optional[str]):
    """stdlib opener; proxy_url None → direct."""
    if not proxy_url:
        return urllib_request_module().build_opener()
    proxy = proxy_url
    handler = urllib_request_module().ProxyHandler({"http": proxy, "https": proxy})
    return urllib_request_module().build_opener(handler)


def urllib_request_module():
    import urllib.request as u

    return u
