"""Search pipeline: retrieve → tier → gate → score → sort."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional, Union

from topper.arxiv_bridge import is_arxiv_like
from topper.models import PaperCard, SearchPolicy
from topper.policy import (
    accepts,
    apply_field_prior,
    attach_tiers,
    infer_field_priors,
    score_card,
)
from topper.relevance import apply_relevance
from topper.scenarios import match_scenario, scenario_gold_substrings, title_hits_gold
from topper.sources.mock import MockSource
from topper.sources.openalex import OpenAlexSource
from topper.sources.s2 import SemanticScholarSource
from topper.tiers.registry import TierRegistry, get_default_registry


def get_source(name: str = "s2", **kwargs: Any):
    key = (name or "s2").lower()
    if key in {"s2", "semantic", "semanticscholar", "semantic-scholar"}:
        # map legacy single api_key into pool
        if "api_key" in kwargs and "api_keys" not in kwargs and kwargs.get("api_key"):
            kwargs = dict(kwargs)
            kwargs["api_keys"] = [kwargs.pop("api_key")]
        return SemanticScholarSource(**kwargs)
    if key in {"openalex", "oa"}:
        return OpenAlexSource(**kwargs)
    if key in {"mock", "demo", "offline"}:
        return MockSource()
    raise ValueError(f"unknown source: {name}")


def _fold_title(t: str) -> str:
    return " ".join((t or "").lower().split())


def _merge_cards(primary: list[PaperCard], extra: list[PaperCard]) -> list[PaperCard]:
    """De-dup by id / doi / folded title; keep primary order then extras."""
    seen_ids: set[str] = set()
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    out: list[PaperCard] = []

    def take(c: PaperCard) -> None:
        cid = (c.id or "").strip()
        doi = (c.doi or "").strip().lower()
        ft = _fold_title(c.title or "")
        if cid and cid in seen_ids:
            return
        if doi and doi in seen_doi:
            return
        if ft and ft in seen_title:
            return
        if cid:
            seen_ids.add(cid)
        if doi:
            seen_doi.add(doi)
        if ft:
            seen_title.add(ft)
        out.append(c)

    for c in primary:
        take(c)
    for c in extra:
        take(c)
    return out


def _s2_year_param(pol: SearchPolicy, now_year: int) -> Optional[str]:
    if pol.max_age_years is None:
        return None
    start = now_year - int(pol.max_age_years)
    return f"{start}-{now_year}"


def search(
    query: str,
    *,
    policy: Optional[SearchPolicy] = None,
    limit: int = 20,
    source: Union[str, object] = "s2",
    fetch_multiplier: int = 5,
    registry: Optional[TierRegistry] = None,
    now_year: Optional[int] = None,
    mailto: Optional[str] = None,
    s2_api_key: Optional[str] = None,
    s2_venue: Optional[str] = None,
    use_proxy: Optional[bool] = None,
    relevance_queries: Optional[list[str]] = None,
    primary_queries: Optional[list[str]] = None,
) -> list[PaperCard]:
    """Top-only search.

    Primary source is Semantic Scholar. Fetches more than `limit` when possible,
    attaches CCF/CAS tiers, drops non-top / off-topic rows, scores, returns top `limit`.

    Pipeline: retrieve → tier gate → prestige score → relevance selector → sort.
    """
    if not query or not str(query).strip():
        return []

    pol = policy or SearchPolicy()
    reg = registry or get_default_registry()
    year = now_year or datetime.utcnow().year

    src_name = source if isinstance(source, str) else getattr(source, "name", "")
    src_key = str(src_name).lower()

    if isinstance(source, str):
        kw: dict[str, Any] = {}
        if mailto and src_key in {"openalex", "oa"}:
            kw["mailto"] = mailto
        if src_key in {"s2", "semantic", "semanticscholar", "semantic-scholar", ""}:
            if s2_api_key:
                kw["api_key"] = s2_api_key
            if use_proxy is not None:
                kw["use_proxy"] = use_proxy
        src = get_source(source, **kw)
        src_key = source.lower()
    else:
        src = source

    fetch_n = max(limit * max(fetch_multiplier, 1), limit)

    retrieve_kwargs: dict[str, Any] = {"limit": fetch_n}
    if src_key in {"s2", "semantic", "semanticscholar", "semantic-scholar"}:
        y = _s2_year_param(pol, year)
        if y:
            retrieve_kwargs["year"] = y
        if pol.min_citations:
            retrieve_kwargs["min_citation_count"] = pol.min_citations
        if s2_venue:
            retrieve_kwargs["venue"] = s2_venue

    raw = src.retrieve(query, **retrieve_kwargs)

    # Optional dual-source: S2 primary + OpenAlex complement (same gate/rank after).
    # Skip for mock / explicit openalex-only / env off.
    dual = os.environ.get("TOPPER_DUAL_SOURCE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if (
        dual
        and src_key in {"s2", "semantic", "semanticscholar", "semantic-scholar", ""}
        and not isinstance(src, MockSource)
    ):
        try:
            oa_limit = max(8, min(limit, fetch_n // 2))
            oa_kw = {"mailto": mailto} if mailto else {}
            oa = OpenAlexSource(**oa_kw)
            extra = oa.retrieve(query, limit=oa_limit)
            raw = _merge_cards(list(raw), list(extra))
        except Exception:  # noqa: BLE001 — never fail S2 path on OA outage
            pass

    alt = list(relevance_queries or [])
    primary = list(primary_queries or [])

    # Scenario packs key off the user intent; multi_search passes that in alt.
    scenario_q = query
    if not match_scenario(scenario_q):
        for a in list(primary) + alt:
            if match_scenario(a):
                scenario_q = a
                break
    golds = scenario_gold_substrings(scenario_q) if match_scenario(scenario_q) else []
    prefer_majors, damp_majors, prefer_fms = infer_field_priors(
        [query, scenario_q, *primary, *alt[:6]]
    )

    kept: list[PaperCard] = []
    for card in raw:
        attach_tiers(card, reg)
        tier_ok = accepts(card, pol, now_year=year)
        arxiv_pending = False
        gold_hit = bool(golds and title_hits_gold(card.title or "", golds))
        if not tier_ok:
            # Flagship gold landmarks may be arXiv / missing tier labels — keep.
            if gold_hit:
                pass
            elif pol.admit_arxiv_candidates and is_arxiv_like(card):
                # Hold arXiv-like preprints for author-bridge admission later.
                arxiv_pending = True
            else:
                continue
        score_card(card, pol, now_year=year)
        card, rel = apply_relevance(
            card,
            query,
            alt_queries=alt,
            primary_queries=primary or None,
            min_score=pol.min_relevance,
            w_relevance=pol.w_relevance,
        )
        apply_field_prior(
            card,
            prefer_majors=prefer_majors,
            damp_majors=damp_majors,
            prefer_fms=prefer_fms,
        )
        # Hard drop known off-field prestige (e.g. 医学/生物学 on DiD/edu intents).
        # Soft score damp alone cannot keep mega-cite neighbors out of the shortlist
        # when a planner subquery accidentally matches them.
        major = (card.tiers.cas_major or "").strip()
        fms_disc = (card.tiers.fms_discipline or "").strip()
        if (
            prefer_majors
            and damp_majors
            and major in damp_majors
            and major not in prefer_majors
            and not (prefer_fms and fms_disc in prefer_fms)
            and not gold_hit
        ):
            continue
        # Topic gate always — including arXiv candidates (never off-topic bridge)
        if pol.require_relevance and not rel.passed and not gold_hit:
            continue
        if gold_hit:
            parts = dict(card.score_parts or {})
            parts["gold_landmark"] = 2.5
            card.score_parts = parts
            card.score = float(
                sum(v for v in parts.values() if isinstance(v, (int, float)))
            )
            ma = list(card.matched_authority or [])
            if "gold_landmark" not in ma:
                ma.append("gold_landmark")
            card.matched_authority = ma
        if arxiv_pending and not gold_hit:
            parts = dict(card.score_parts or {})
            parts["arxiv_pending"] = 1.0
            card.score_parts = parts
            # marker for multi_search; not yet in final list semantics
            ma = list(card.matched_authority or [])
            if "arxiv_pending" not in ma:
                ma.append("arxiv_pending")
            card.matched_authority = ma
        kept.append(card)

    kept.sort(key=lambda c: c.score, reverse=True)
    # allow extra headroom for pending arxiv before multi_search filters
    return kept[: max(limit, limit + 5)]
