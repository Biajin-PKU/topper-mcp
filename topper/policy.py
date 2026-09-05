"""Accept gate + scoring for top-only retrieval."""

from __future__ import annotations

import json
import math
from datetime import datetime
from functools import lru_cache
from typing import Optional, Sequence

from topper.authority import match_authors, match_orgs
from topper.models import PaperCard, SearchPolicy
from topper.normalize import fold
from topper.tiers.registry import TierRegistry, get_default_registry

# Intent → preferred CAS majors + FMS disciplines (soft rank; hard gate unchanged).
# Each rule: (keys, prefer_cas, damp_cas, prefer_fms).
_FIELD_RULES: tuple[
    tuple[tuple[str, ...], frozenset[str], frozenset[str], frozenset[str]], ...
] = (
    (
        (
            "education",
            "educational",
            "schooling",
            "pedagog",
            "classroom",
            "teacher",
            "student achievement",
            "高等教育",
            "基础教育",
            "教育干预",
            "教育经济",
            "教育学",
            "学业",
            "教师",
            "学生成绩",
        ),
        frozenset({"教育学", "经济学", "社会学", "心理学"}),
        frozenset({"医学", "生物学", "化学", "物理学", "材料科学", "地球科学"}),
        frozenset({"教育管理", "劳动与人口经济", "一般经济", "计量经济与统计"}),
    ),
    (
        (
            "urbanization",
            "urban economy",
            "city growth",
            "housing market",
            "land use",
            "real estate",
            "城镇化",
            "城市化",
            "城市经济",
            "土地财政",
            "房价",
        ),
        frozenset({"经济学", "地理学", "社会学", "环境科学与生态学", "管理学"}),
        frozenset({"医学", "生物学", "化学", "物理学"}),
        frozenset(
            {
                "区域研究与区域经济",
                "产业经济与发展经济",
                "资源环境管理",
                "一般经济",
            }
        ),
    ),
    (
        (
            "social capital",
            "social network",
            "trust",
            "civic",
            "社会资本",
            "社会网络",
            "信任",
        ),
        frozenset({"社会学", "经济学", "心理学", "管理学", "政治学"}),
        frozenset({"医学", "生物学", "化学", "物理学", "材料科学"}),
        frozenset({"一般管理", "组织管理", "公共政策与公共管理", "心理学"}),
    ),
    (
        (
            "difference-in-differences",
            "difference in differences",
            "did design",
            "twfe",
            "two-way fixed",
            "event study",
            "causal inference",
            "instrumental variable",
            "regression discontinuity",
            "双重差分",
            "因果推断",
            "工具变量",
            "断点回归",
        ),
        frozenset({"经济学", "社会学", "管理学", "统计学", "政治学", "教育学"}),
        frozenset(
            {
                "医学",
                "生物学",
                "化学",
                "物理学",
                "材料科学",
                "地球科学",
                "农林科学",
                "环境科学与生态学",
            }
        ),
        frozenset(
            {
                "计量经济与统计",
                "一般经济",
                "理论经济与实验经济",
                "劳动与人口经济",
                "产业经济与发展经济",
            }
        ),
    ),
)

# Mega-journals that span fields — never hard-drop, but soft-damp without design fit.
_GENERALIST_MAJORS = frozenset({"综合性期刊", "综合性"})


def infer_field_priors(
    texts: Sequence[str],
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return (prefer_cas, damp_cas, prefer_fms) from query / plan strings."""
    blob = " ".join(t for t in texts if t).lower()
    if not blob.strip():
        return frozenset(), frozenset(), frozenset()
    prefer: set[str] = set()
    damp: set[str] = set()
    prefer_fms: set[str] = set()
    for keys, pref, dmp, fms in _FIELD_RULES:
        if any(k.lower() in blob for k in keys):
            prefer |= set(pref)
            damp |= set(dmp)
            prefer_fms |= set(fms)
    # Never damp a major we also prefer (e.g. mixed intents).
    damp -= prefer
    return frozenset(prefer), frozenset(damp), frozenset(prefer_fms)


def apply_field_prior(
    card: PaperCard,
    *,
    prefer_majors: frozenset[str] = frozenset(),
    damp_majors: frozenset[str] = frozenset(),
    prefer_fms: frozenset[str] = frozenset(),
    w_field: float = 1.4,
) -> PaperCard:
    """Soft CAS-major / FMS-discipline prior on an already-scored card."""
    if not prefer_majors and not damp_majors and not prefer_fms:
        return card
    major = (card.tiers.cas_major or "").strip()
    fms_disc = (card.tiers.fms_discipline or "").strip()
    parts = dict(card.score_parts or {})
    delta = 0.0
    hit_prefer = False
    if major and major in prefer_majors:
        hit_prefer = True
        delta += w_field
        if card.tiers.cas_zone == 1:
            delta += 0.35
        elif card.tiers.cas_zone == 2:
            delta += 0.15
    if fms_disc and fms_disc in prefer_fms:
        hit_prefer = True
        delta += w_field * 0.85
        ft = (card.tiers.fms_tier or "").upper()
        if ft in {"A", "T1"}:
            delta += 0.3
        elif ft == "B":
            delta += 0.12
    if major and major in damp_majors and not hit_prefer:
        # Public-health / pure-medicine mega-cites on edu/econ intents.
        delta -= w_field * 1.15
        for k in ("citations", "tier"):
            if k in parts and isinstance(parts[k], (int, float)):
                parts[k] = float(parts[k]) * 0.55
    elif (
        prefer_majors
        and major in _GENERALIST_MAJORS
        and not hit_prefer
        and float(parts.get("relevance") or 0.0) < 2.4
    ):
        # Nature/Science without strong topical fit — shrink prestige.
        delta -= w_field * 0.7
        for k in ("citations", "tier"):
            if k in parts and isinstance(parts[k], (int, float)):
                parts[k] = float(parts[k]) * 0.5
    if abs(delta) < 1e-9:
        return card
    parts["field"] = float(parts.get("field") or 0.0) + delta
    card.score_parts = parts
    card.score = float(sum(v for v in parts.values() if isinstance(v, (int, float))))
    return card


def attach_tiers(card: PaperCard, registry: Optional[TierRegistry] = None) -> PaperCard:
    reg = registry or get_default_registry()
    venue = card.venue or ""
    card.tiers = reg.lookup(venue)
    card.venue_key = reg.resolve_key(venue) or card.venue_key
    return card


@lru_cache(maxsize=1)
def _flagship_keys() -> frozenset[str]:
    """Venues that count as top-tier regardless of which ranking tables are installed."""
    from topper import config

    path = config.data_dir() / "flagship_venues.json"
    try:
        with path.open(encoding="utf-8") as f:
            return frozenset(fold(v) for v in (json.load(f).get("venues") or []))
    except (OSError, json.JSONDecodeError):
        return frozenset()


def _whitelisted(card: PaperCard, policy: SearchPolicy) -> bool:
    keys = {fold(v) for v in policy.venue_whitelist} | _flagship_keys()
    if not keys:
        return False
    for cand in (card.venue_key, card.venue):
        if cand and fold(cand) in keys:
            return True
    return False


def accepts(card: PaperCard, policy: SearchPolicy, *, now_year: Optional[int] = None) -> bool:
    """Hard gate. Specialty min_citations / max_age also enforced here when set."""
    year_now = now_year or datetime.utcnow().year

    if policy.max_age_years is not None and card.year:
        if card.year < year_now - policy.max_age_years:
            return False
    if policy.min_citations and card.cited_by_count < policy.min_citations:
        return False

    if _whitelisted(card, policy):
        return True

    ccf_ok = (
        card.tiers.ccf is not None
        and card.tiers.ccf.upper() in {x.upper() for x in policy.ccf_levels}
    )
    cas_ok = (
        card.tiers.cas_zone is not None and card.tiers.cas_zone in policy.cas_zones
    )
    fms_ok = _fms_ok(card, policy)
    wos_ok = _wos_ok(card, policy)

    if policy.require_tier:
        return bool(ccf_ok or cas_ok or fms_ok or wos_ok)
    # Soft mode: always accept post citation/age filters
    return True


def _fms_ok(card: PaperCard, policy: SearchPolicy) -> bool:
    """True when FMS tier is in policy.fms_levels (empty levels → ignore)."""
    if not policy.fms_levels:
        return False
    ft = (card.tiers.fms_tier or "").upper()
    if not ft:
        return False
    return ft in {x.upper() for x in policy.fms_levels}


def _wos_ok(card: PaperCard, policy: SearchPolicy) -> bool:
    """True when the paper is in a requested Web-of-Science index (and quartile)."""
    if not policy.wos:
        return False
    t = card.tiers
    indexed = {"sci": t.sci, "ssci": t.ssci, "ahci": t.ahci}
    if not any(indexed.get(k) for k in policy.wos):
        return False
    if policy.jcr_quartiles and t.jcr_quartile not in policy.jcr_quartiles:
        return False
    return True


def score_card(
    card: PaperCard,
    policy: SearchPolicy,
    *,
    now_year: Optional[int] = None,
) -> PaperCard:
    year_now = now_year or datetime.utcnow().year
    parts: dict[str, float] = {}

    # citations: log1p, hard-capped so mega-cited neighbors cannot dominate topic
    cite_log = math.log1p(max(card.cited_by_count, 0))
    cap = float(getattr(policy, "citation_log_cap", 6.2) or 6.2)
    parts["citations"] = policy.w_citations * min(cite_log, cap)

    # recency: newer → higher; unknown year → neutral 0
    if card.year:
        age = max(year_now - card.year, 0)
        parts["recency"] = policy.w_recency * max(0.0, 1.0 - age / 10.0)
    else:
        parts["recency"] = 0.0

    # tier boost: CCF additive; CAS/JCR/FMS are journal-quality views → max;
    # WoS-index alone gives a small lift for journals not otherwise tiered
    tier = 0.0
    if card.tiers.ccf:
        tier += {"A": 1.0, "B": 0.6, "C": 0.25}.get(card.tiers.ccf.upper(), 0.0)
    cas_b = {1: 1.0, 2: 0.6, 3: 0.25, 4: 0.1}.get(card.tiers.cas_zone, 0.0) if card.tiers.cas_zone else 0.0
    jcr_b = (
        {"Q1": 0.8, "Q2": 0.5, "Q3": 0.25, "Q4": 0.1}.get(card.tiers.jcr_quartile, 0.0)
        if card.tiers.jcr_quartile
        else 0.0
    )
    fms_b = (
        {"A": 1.0, "T1": 1.0, "B": 0.6, "T2": 0.45, "C": 0.25, "D": 0.1}.get(
            (card.tiers.fms_tier or "").upper(), 0.0
        )
        if card.tiers.fms_tier
        else 0.0
    )
    tier += max(cas_b, jcr_b, fms_b)
    if (
        (card.tiers.sci or card.tiers.ssci or card.tiers.ahci)
        and not (card.tiers.ccf or card.tiers.cas_zone or card.tiers.fms_tier)
    ):
        tier += 0.25
    parts["tier"] = policy.w_tier * tier

    # authority
    auth_authors = match_authors(card.authors, policy.flagship_authors)
    auth_orgs = match_orgs(card.institutions, policy.flagship_orgs)
    matched = [f"author:{a}" for a in auth_authors] + [f"org:{o}" for o in auth_orgs]
    parts["authority"] = policy.w_authority * float(len(matched))
    card.matched_authority = matched

    card.score_parts = parts
    card.score = sum(parts.values())
    return card
