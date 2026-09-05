"""Admit selected arXiv preprints when authors overlap CCF-A hits.

Rule (product):
  arXiv may enter the top list only if
    (1) topical relevance passes (same Selector gate as venue papers), AND
    (2) author-side bridge:
          - author name overlaps an already-accepted CCF-A paper in this run, OR
          - author/org matches the flagship authority lists.

Never import an author's off-topic arXiv just because of name overlap.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Set

from topper.authority import match_authors, match_orgs
from topper.models import PaperCard, SearchPolicy
from topper.normalize import fold
from topper.relevance import apply_relevance

_ARXIV_HINT = re.compile(
    r"\barxiv\b|corr/abs|preprint",
    re.I,
)


def is_arxiv_like(card: PaperCard) -> bool:
    blob = " ".join(
        [
            card.venue or "",
            card.venue_key or "",
            card.landing_url or "",
            card.oa_url or "",
            card.id or "",
        ]
    )
    if _ARXIV_HINT.search(blob):
        return True
    # DOI-less + open pdf on arxiv.org
    if card.oa_url and "arxiv.org" in card.oa_url.lower():
        return True
    if card.landing_url and "arxiv.org" in card.landing_url.lower():
        return True
    return False


def _norm_name(name: str) -> str:
    s = fold(name)
    # drop punctuation-ish
    s = re.sub(r"[^a-z0-9一-鿿\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def author_keyset(authors: Sequence[str]) -> Set[str]:
    keys: Set[str] = set()
    for a in authors or []:
        n = _norm_name(a)
        if not n:
            continue
        keys.add(n)
        parts = n.split()
        # last-token key for "J. Smith" ~ "smith" (weak); keep full + last
        if parts:
            keys.add(parts[-1])
            if len(parts) >= 2:
                keys.add(f"{parts[0][0]} {parts[-1]}")  # first initial + last
    return {k for k in keys if len(k) >= 2}


def authors_overlap(a: Sequence[str], b: Sequence[str]) -> list[str]:
    """Return matched identity keys (for provenance)."""
    ka, kb = author_keyset(a), author_keyset(b)
    # Prefer longer keys (full names) when reporting
    hits = sorted(ka & kb, key=len, reverse=True)
    # Filter pure last-name-only collisions if a longer key also exists? keep all.
    return hits


def ccf_a_author_pool(cards: Iterable[PaperCard]) -> Set[str]:
    pool: Set[str] = set()
    for c in cards:
        if (c.tiers.ccf or "").upper() == "A":
            pool |= author_keyset(c.authors)
    return pool


def bridge_reasons(
    card: PaperCard,
    *,
    ccf_a_authors: Set[str],
    policy: SearchPolicy,
) -> list[str]:
    """Why an arXiv card may be bridged (author/org only — not relevance)."""
    reasons: list[str] = []
    card_keys = author_keyset(card.authors)
    overlap = sorted(card_keys & ccf_a_authors, key=len, reverse=True)
    # Require at least one "strong" overlap: full name or initial+last (has space)
    strong = [k for k in overlap if " " in k or len(k) >= 6]
    if strong:
        reasons.append(f"ccf_a_author:{strong[0]}")
    elif overlap and len(overlap[0]) >= 5:
        # single long token last name — weaker; still allow with flag
        reasons.append(f"ccf_a_author_weak:{overlap[0]}")

    auth_a = match_authors(card.authors, policy.flagship_authors)
    auth_o = match_orgs(card.institutions, policy.flagship_orgs)
    for a in auth_a:
        reasons.append(f"flagship_author:{a}")
    for o in auth_o:
        reasons.append(f"flagship_org:{o}")
    return reasons


def try_admit_arxiv(
    card: PaperCard,
    *,
    query: str,
    alt_queries: Optional[Sequence[str]],
    policy: SearchPolicy,
    ccf_a_authors: Set[str],
    now_year: Optional[int] = None,
) -> tuple[bool, PaperCard, str]:
    """Return (admitted, card, reason).

    BOTH must hold:
      - bridge_reasons non-empty (CCF-A author overlap or flagship)
      - relevance.passed against query (+ expansions)
    """
    if not is_arxiv_like(card):
        return False, card, "not_arxiv"

    # age / cites still apply lightly via policy helpers
    from datetime import datetime

    year_now = now_year or datetime.utcnow().year
    if policy.max_age_years is not None and card.year:
        if card.year < year_now - policy.max_age_years:
            return False, card, "too_old"
    if policy.min_citations and card.cited_by_count < policy.min_citations:
        return False, card, "low_cites"

    reasons = bridge_reasons(card, ccf_a_authors=ccf_a_authors, policy=policy)
    if not reasons:
        return False, card, "no_author_bridge"

    # Topic gate — mandatory (never import author's other topics)
    weak_only = bool(reasons) and all(
        r.startswith("ccf_a_author_weak:") for r in reasons
    )
    min_rel = policy.min_relevance * (1.35 if weak_only else 1.0)
    card, rel = apply_relevance(
        card,
        query,
        alt_queries=list(alt_queries or []),
        min_score=min_rel,
        w_relevance=policy.w_relevance,
    )
    if policy.require_relevance and not rel.passed:
        return False, card, "off_topic"

    # Mark provenance for UI summary
    tag = f"arxiv_bridge:{reasons[0]}"
    matched = list(card.matched_authority or [])
    if tag not in matched:
        matched.append(tag)
    for r in reasons[1:3]:
        t = f"arxiv_bridge:{r}"
        if t not in matched:
            matched.append(t)
    card.matched_authority = matched
    # small boost so bridged arXiv can surface among peers
    parts = dict(card.score_parts or {})
    parts["arxiv_bridge"] = 0.8
    card.score_parts = parts
    card.score = float(sum(v for v in parts.values() if isinstance(v, (int, float))))
    return True, card, reasons[0]
