"""Shared card + policy types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TierLabels:
    """Normalized venue tier labels (None = unknown / not applicable)."""

    ccf: Optional[str] = None  # "A" | "B" | "C"
    cas_zone: Optional[int] = None  # 1 | 2 | 3 | 4 (中科院大类分区)
    cas_major: Optional[str] = None  # 中科院大类, e.g. "计算机科学" / "医学"
    cas_top: Optional[bool] = None  # 中科院 Top 期刊
    # FMS Journal Rating Guide (经管): A/B/C/D + T1/T2
    fms_tier: Optional[str] = None
    fms_discipline: Optional[str] = None
    sci: Optional[bool] = None  # SCIE (Science Citation Index Expanded)
    ssci: Optional[bool] = None  # SSCI (Social Sciences Citation Index)
    ahci: Optional[bool] = None  # A&HCI (Arts & Humanities Citation Index)
    jcr_quartile: Optional[str] = None  # best JCR quartile "Q1".."Q4"
    impact_factor: Optional[float] = None  # latest JCR impact factor
    source_as_of: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperCard:
    """Source-agnostic paper metadata. No full text."""

    id: str
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    venue_key: Optional[str] = None
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    cited_by_count: int = 0
    publication_date: Optional[str] = None
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    landing_url: Optional[str] = None
    oa_url: Optional[str] = None
    tiers: TierLabels = field(default_factory=TierLabels)
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)
    matched_authority: list[str] = field(default_factory=list)
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass(frozen=True)
class SearchPolicy:
    """Accept + rank policy for top-only retrieval."""

    ccf_levels: tuple[str, ...] = ("A", "B")
    cas_zones: tuple[int, ...] = (1, 2)
    # FMS tiers accepted as a hard-gate path (经管 journal guide). Empty = ignore FMS for gate.
    fms_levels: tuple[str, ...] = ("A", "B", "T1")
    # Web-of-Science indices accepted as a tier (subset of {"sci", "ssci", "ahci"}).
    # Empty = no WoS requirement. When non-empty, a journal must be in one of these
    # indices AND (if jcr_quartiles set) in an accepted JCR quartile.
    wos: tuple[str, ...] = ()
    jcr_quartiles: tuple[str, ...] = ()  # empty = no quartile restriction
    # If True, paper must match CCF or CAS or FMS or WoS (or venue_whitelist).
    require_tier: bool = True
    venue_whitelist: tuple[str, ...] = ()
    # Specialty
    min_citations: int = 0
    max_age_years: Optional[int] = 8
    # Authority boosts (soft): names matched case-insensitively as substrings
    flagship_authors: tuple[str, ...] = ()
    flagship_orgs: tuple[str, ...] = ()
    # Ranking weights — topic-first (citations must not drown topical fit).
    # Prestige is a tie-break inside the top-venue set, not the main sort key.
    w_citations: float = 0.25
    w_recency: float = 0.6
    w_tier: float = 1.2
    w_authority: float = 0.8
    # PaSa-style topical selector (lexical v0; LLM selector later)
    require_relevance: bool = True
    min_relevance: float = 0.28
    w_relevance: float = 6.0
    # Cap log-citation contribution so mega-cited off-neighbors cannot dominate.
    citation_log_cap: float = 6.2  # ~log1p(500)
    # arXiv bridge: allow preprint candidates through tier gate only as
    # pending; multi_search admits them iff author∩CCF-A (+ relevance).
    admit_arxiv_candidates: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
