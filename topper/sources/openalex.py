"""OpenAlex works search (stdlib urllib)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from topper.models import PaperCard

DEFAULT_MAILTO = "top-paper-retriever@example.com"
API = "https://api.openalex.org/works"


class OpenAlexSource:
    name = "openalex"

    def __init__(
        self,
        *,
        mailto: str = DEFAULT_MAILTO,
        timeout: float = 30.0,
        per_page_cap: int = 50,
    ) -> None:
        self.mailto = mailto
        self.timeout = timeout
        self.per_page_cap = per_page_cap

    def retrieve(self, query: str, *, limit: int = 25) -> list[PaperCard]:
        per_page = max(1, min(limit, self.per_page_cap))
        params = {
            "search": query,
            "per_page": str(per_page),
            "sort": "relevance_score:desc",
            "mailto": self.mailto,
        }
        url = f"{API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"top-paper-retriever/0.0.1 ({self.mailto})"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAlex HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAlex network error: {e.reason}") from e

        results = payload.get("results") or []
        cards = [self._to_card(r) for r in results]
        return cards[:limit]

    def _to_card(self, row: dict[str, Any]) -> PaperCard:
        openalex_id = str(row.get("id") or "")
        short_id = openalex_id.rsplit("/", 1)[-1] if openalex_id else ""
        title = (row.get("display_name") or row.get("title") or "").strip() or "(untitled)"
        year = row.get("publication_year")
        try:
            year_i: Optional[int] = int(year) if year is not None else None
        except (TypeError, ValueError):
            year_i = None

        venue = None
        primary = row.get("primary_location") or {}
        source = primary.get("source") or {}
        if isinstance(source, dict):
            venue = source.get("display_name") or source.get("name")
        if not venue:
            hl = row.get("host_venue") or {}
            if isinstance(hl, dict):
                venue = hl.get("display_name")

        authors: list[str] = []
        institutions: list[str] = []
        for auth in row.get("authorships") or []:
            a = auth.get("author") or {}
            name = a.get("display_name")
            if name:
                authors.append(str(name))
            for inst in auth.get("institutions") or []:
                iname = inst.get("display_name")
                if iname and iname not in institutions:
                    institutions.append(str(iname))

        oa = row.get("open_access") or {}
        oa_url = oa.get("oa_url")
        landing = None
        if isinstance(primary, dict):
            landing = primary.get("landing_page_url")

        cited = row.get("cited_by_count") or 0
        try:
            cited_i = int(cited)
        except (TypeError, ValueError):
            cited_i = 0

        doi = None
        ids = row.get("ids") or {}
        if isinstance(ids, dict) and ids.get("doi"):
            doi = str(ids["doi"]).replace("https://doi.org/", "")

        abstract = _rebuild_abstract(row.get("abstract_inverted_index"))

        return PaperCard(
            id=f"openalex:{short_id or title[:40]}",
            title=title,
            year=year_i,
            venue=str(venue) if venue else None,
            doi=doi,
            openalex_id=openalex_id or None,
            cited_by_count=cited_i,
            publication_date=row.get("publication_date"),
            abstract=abstract,
            authors=authors,
            institutions=institutions,
            landing_url=str(landing) if landing else None,
            oa_url=str(oa_url) if oa_url else None,
            source=self.name,
            raw=row,
        )


def _rebuild_abstract(inv: Any) -> Optional[str]:
    """OpenAlex serves abstracts as inverted index {token: [positions]}."""
    if not isinstance(inv, dict) or not inv:
        return None
    try:
        max_pos = max(p for positions in inv.values() for p in (positions or []))
    except ValueError:
        return None
    slots: list[str] = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for p in positions or []:
            if isinstance(p, int) and 0 <= p <= max_pos:
                slots[p] = str(token)
    text = " ".join(t for t in slots if t).strip()
    return text[:2000] if text else None
