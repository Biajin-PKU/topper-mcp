"""Semantic Scholar Graph API search (primary source) with key+proxy pools."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from topper.models import PaperCard
from topper.s2_pool import (
    ProxyPool,
    S2KeyPool,
    build_opener,
    key_fingerprint,
    load_s2_keys_from_env,
)

API = "https://api.semanticscholar.org/graph/v1/paper/search"
BULK_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"

# Fields shared by both endpoints. `authors.affiliations` is deliberately absent:
# it roughly doubles relevance latency and the bulk endpoint rejects it outright.
# Affiliations are backfilled for the final shortlist via the batch endpoint.
_BASE_FIELDS = [
    "paperId",
    "title",
    "year",
    "venue",
    "citationCount",
    "authors",
    "externalIds",
    "url",
    "openAccessPdf",
    "publicationVenue",
    "abstract",
    "publicationDate",
]
FIELDS = ",".join(_BASE_FIELDS)
BULK_FIELDS = ",".join(_BASE_FIELDS)
BATCH_FIELDS = "paperId,authors,authors.affiliations"


class SemanticScholarSource:
    """Relevance search with key-pool LB and optional rotating IP proxy."""

    name = "s2"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_keys: Optional[list[str]] = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        min_interval: float = 1.05,
        use_proxy: Optional[bool] = None,
        proxy_pool: Optional[ProxyPool] = None,
        key_pool: Optional[S2KeyPool] = None,
        prefer_bulk: Optional[bool] = None,
        bulk_sort: str = "citationCount:desc",
        backfill_affiliations: Optional[bool] = None,
        affiliation_batch_limit: int = 100,
    ) -> None:
        if key_pool is not None:
            self.key_pool = key_pool
        else:
            keys = list(api_keys or [])
            if api_key:
                keys = [api_key] + keys
            if not keys:
                keys = load_s2_keys_from_env()
            # de-dupe preserve order
            seen = set()
            uniq = []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    uniq.append(k)
            self.key_pool = S2KeyPool(uniq, min_interval=min_interval)

        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.prefer_bulk = (
            prefer_bulk
            if prefer_bulk is not None
            else (os.environ.get("TOPPER_S2_BULK") or "1").lower() not in {"0", "false", "no"}
        )
        self.bulk_sort = bulk_sort
        self.backfill_affiliations = (
            backfill_affiliations
            if backfill_affiliations is not None
            else (os.environ.get("TOPPER_S2_AFFILIATIONS") or "1").lower()
            not in {"0", "false", "no"}
        )
        self.affiliation_batch_limit = int(affiliation_batch_limit)

        env_proxy = ProxyPool.from_env()
        self.proxy_pool = proxy_pool if proxy_pool is not None else env_proxy
        if use_proxy is None:
            # default on when a proxy pool is configured
            self.use_proxy = bool(self.proxy_pool and self.proxy_pool.enabled)
        else:
            self.use_proxy = bool(use_proxy)

        # last call meta for diagnostics
        self.last_meta: dict[str, Any] = {}

    @property
    def api_key(self) -> Optional[str]:
        """Back-compat: first key fingerprint presence."""
        snap = self.key_pool.snapshot()
        if not snap:
            return None
        fp = snap[0]["fp"]
        return None if fp in {"(anon)", "***"} and self.key_pool.size == 1 else "pool"

    def pool_status(self) -> dict[str, Any]:
        return {
            "keys": self.key_pool.snapshot(),
            "key_count": self.key_pool.size,
            "proxy": None
            if not self.proxy_pool
            else {**self.proxy_pool.snapshot(), "use_proxy": self.use_proxy},
            "last": dict(self.last_meta),
        }

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 25,
        year: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        venue: Optional[str] = None,
        fields_of_study: Optional[str] = None,
    ) -> list[PaperCard]:
        if not query or not query.strip():
            return []

        want = max(1, min(int(limit), 100))
        common: dict[str, str] = {"query": query.strip()}
        if year:
            common["year"] = year
        if min_citation_count is not None and min_citation_count > 0:
            common["minCitationCount"] = str(int(min_citation_count))
        if venue:
            common["venue"] = venue
        if fields_of_study:
            common["fieldsOfStudy"] = fields_of_study

        rows: list[dict[str, Any]] = []
        if self.prefer_bulk:
            # Bulk is ~2x faster than relevance and returns up to 1000 rows per
            # call, so one request replaces many. Sorting by citations puts the
            # landmark papers first instead of relying on extra query rounds.
            bulk_params = dict(common)
            bulk_params["fields"] = BULK_FIELDS
            bulk_params["sort"] = self.bulk_sort
            try:
                data = self._get(bulk_params, base=BULK_API)
                rows = data.get("data") or []
            except Exception:  # noqa: BLE001 — bulk rejects some query syntax
                rows = []

        if not rows:
            rel_params = dict(common)
            rel_params["fields"] = FIELDS
            rel_params["limit"] = str(want)
            data = self._get(rel_params, base=API)
            rows = data.get("data") or []

        rows = rows[:want]
        cards = [self._to_card(r) for r in rows]
        if self.backfill_affiliations:
            self._attach_affiliations(cards)
        return cards

    def _attach_affiliations(self, cards: list[PaperCard]) -> None:
        """Backfill institutions for a shortlist via one batch call.

        The search endpoints omit `authors.affiliations` (bulk rejects it, and on
        relevance it roughly doubles latency), so institutions are fetched once
        for the cards we actually keep.
        """
        ids = [c.raw.get("paperId") for c in cards if c.raw.get("paperId")]
        ids = [i for i in ids if i][: self.affiliation_batch_limit]
        if not ids:
            return
        try:
            payload = json.dumps({"ids": ids}).encode("utf-8")
            key, _waited = self.key_pool.acquire()
            headers = {
                "User-Agent": "top-paper-retriever/0.0.1 (zhice-rh; s2-pool)",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if key:
                headers["x-api-key"] = key
            url = f"{BATCH_API}?{urllib.parse.urlencode({'fields': BATCH_FIELDS})}"
            proxy_url = None
            if self.use_proxy and self.proxy_pool and self.proxy_pool.enabled:
                proxy_url = self.proxy_pool.proxy_url()
            opener = build_opener(proxy_url)
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with opener.open(req, timeout=self.timeout) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            self.key_pool.mark_success(key)
        except Exception:  # noqa: BLE001 — institutions are cosmetic, never fail a search
            return

        by_id: dict[str, list[str]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            pid = row.get("paperId")
            if not pid:
                continue
            affs: list[str] = []
            for a in row.get("authors") or []:
                for aff in (a or {}).get("affiliations") or []:
                    if aff and aff not in affs:
                        affs.append(aff)
            if affs:
                by_id[pid] = affs
        for c in cards:
            pid = c.raw.get("paperId")
            if pid and pid in by_id and not c.institutions:
                c.institutions = by_id[pid][:6]

    def _get(self, params: dict[str, str], base: str = API) -> dict[str, Any]:
        url = f"{base}?{urllib.parse.urlencode(params)}"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            key, waited = self.key_pool.acquire()
            proxy_url = None
            if self.use_proxy and self.proxy_pool and self.proxy_pool.enabled:
                proxy_url = self.proxy_pool.proxy_url()  # random session → random IP

            headers = {
                "User-Agent": "top-paper-retriever/0.0.1 (zhice-rh; s2-pool)",
                "Accept": "application/json",
            }
            if key:
                headers["x-api-key"] = key

            t0 = time.time()
            try:
                opener = build_opener(proxy_url)
                req = urllib.request.Request(url, headers=headers)
                with opener.open(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    payload = json.loads(raw)
                dt = time.time() - t0
                self.key_pool.mark_success(key)
                if self.proxy_pool and proxy_url:
                    self.proxy_pool.mark(True)
                self.last_meta = {
                    "ok": True,
                    "key_fp": key_fingerprint(key) if key else "(anon)",
                    "waited_s": round(waited, 3),
                    "latency_s": round(dt, 3),
                    "proxy": bool(proxy_url),
                    "attempt": attempt + 1,
                    "bytes": len(raw),
                }
                return payload
            except urllib.error.HTTPError as e:
                last_err = e
                dt = time.time() - t0
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                if self.proxy_pool and proxy_url:
                    self.proxy_pool.mark(e.code not in {429})  # 429 is S2, not proxy
                if e.code == 429:
                    retry_after = None
                    ra = e.headers.get("Retry-After") if e.headers else None
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            retry_after = None
                    self.key_pool.mark_rate_limit(key, retry_after=retry_after)
                    self.last_meta = {
                        "ok": False,
                        "error": "429",
                        "key_fp": key_fingerprint(key) if key else "(anon)",
                        "waited_s": round(waited, 3),
                        "latency_s": round(dt, 3),
                        "proxy": bool(proxy_url),
                        "attempt": attempt + 1,
                    }
                    if attempt + 1 < self.max_retries:
                        time.sleep(0.3)
                        continue
                else:
                    self.key_pool.mark_failure(key, f"HTTP {e.code}")
                    if e.code in {500, 502, 503, 504} and attempt + 1 < self.max_retries:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise RuntimeError(
                        f"Semantic Scholar HTTP {e.code}: {e.reason} {body}"
                    ) from e
            except urllib.error.URLError as e:
                last_err = e
                self.key_pool.mark_failure(key, f"URLError {e.reason}")
                if self.proxy_pool and proxy_url:
                    self.proxy_pool.mark(False)
                self.last_meta = {
                    "ok": False,
                    "error": f"URLError:{e.reason}",
                    "key_fp": key_fingerprint(key) if key else "(anon)",
                    "proxy": bool(proxy_url),
                    "attempt": attempt + 1,
                }
                if attempt + 1 < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"Semantic Scholar network error: {e.reason}") from e

        raise RuntimeError(f"Semantic Scholar failed: {last_err}")

    def _to_card(self, row: dict[str, Any]) -> PaperCard:
        paper_id = str(row.get("paperId") or "")
        title = (row.get("title") or "").strip() or "(untitled)"
        year = row.get("year")
        try:
            year_i = int(year) if year is not None else None
        except (TypeError, ValueError):
            year_i = None

        venue = row.get("venue")
        pv = row.get("publicationVenue") or {}
        if isinstance(pv, dict):
            alts = pv.get("alternate_names") or []
            name = pv.get("name")
            if alts:
                short = sorted((str(a) for a in alts if a), key=len)[:1]
                venue = short[0] if short else (name or venue)
            elif name:
                venue = name

        authors = []
        institutions: list[str] = []
        for a in row.get("authors") or []:
            if not isinstance(a, dict):
                continue
            n = a.get("name")
            if n:
                authors.append(str(n))
            for aff in a.get("affiliations") or []:
                # API may return plain strings or objects
                if isinstance(aff, str):
                    name = aff.strip()
                elif isinstance(aff, dict):
                    name = str(aff.get("name") or aff.get("institution") or "").strip()
                else:
                    name = str(aff).strip()
                if name and name not in institutions:
                    institutions.append(name)

        cited = row.get("citationCount") or 0
        try:
            cited_i = int(cited)
        except (TypeError, ValueError):
            cited_i = 0

        doi = None
        arxiv = None
        ext = row.get("externalIds") or {}
        if isinstance(ext, dict):
            if ext.get("DOI"):
                doi = str(ext["DOI"])
            if ext.get("ArXiv"):
                arxiv = str(ext["ArXiv"])

        oa_url = None
        oa = row.get("openAccessPdf") or {}
        if isinstance(oa, dict) and oa.get("url"):
            oa_url = str(oa["url"])
        if not oa_url and arxiv:
            oa_url = f"https://arxiv.org/pdf/{arxiv}"

        landing = row.get("url")
        if not landing and paper_id:
            landing = f"https://www.semanticscholar.org/paper/{paper_id}"

        abs_text = row.get("abstract")
        if isinstance(abs_text, str):
            abs_text = abs_text.strip() or None
        else:
            abs_text = None

        return PaperCard(
            id=f"s2:{paper_id or title[:40]}",
            title=title,
            year=year_i,
            venue=str(venue) if venue else None,
            doi=doi,
            cited_by_count=cited_i,
            publication_date=row.get("publicationDate"),
            abstract=abs_text,
            authors=authors,
            institutions=institutions,
            landing_url=str(landing) if landing else None,
            oa_url=oa_url,
            source=self.name,
            raw=row,
        )
