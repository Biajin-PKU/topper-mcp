"""Query–paper relevance: a deterministic topical gate.

Scores title+abstract against the query and drops papers that pass the venue
gate but do not fit the topic. Needs no LLM; an LLM judge can be swapped in
behind the same interface (see `selector.py`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from topper.models import PaperCard
from topper.scenarios import (
    match_scenario,
    scenario_gold_substrings,
    scenario_noise_substrings,
    title_hits_gold,
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-+]{1,}|[一-鿿]{1,}")

# Ultra-common tokens that should not drive relevance alone.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "as", "at", "is", "are", "be", "this", "that", "these", "those",
    "we", "our", "their", "its", "into", "via", "using", "based", "based",
    "paper", "study", "method", "methods", "model", "models", "approach",
    "towards", "toward", "using", "new", "novel",
    "的", "了", "与", "和", "及", "在", "对", "中", "为", "是", "一种", "基于",
}


def _stem(t: str) -> str:
    """Very light English stem so scientist≈scientific, agents≈agent."""
    if len(t) <= 4 or not t.isascii():
        return t
    for suf in ("ation", "ition", "iness", "ement", "ance", "ence", "ing", "ers", "ies", "ied", "ed", "ly", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for m in _TOKEN.finditer(text):
        t = m.group(0).lower()
        if t in _STOP:
            continue
        if t.isdigit():
            continue
        if len(t) == 1 and not ("一" <= t <= "鿿"):
            continue
        out.append(_stem(t))
    return out


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if n <= 1 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass(frozen=True)
class RelevanceResult:
    score: float  # 0..1-ish
    title_overlap: float
    abstract_overlap: float
    phrase_bonus: float
    matched_terms: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "title_overlap": round(self.title_overlap, 4),
            "abstract_overlap": round(self.abstract_overlap, 4),
            "phrase_bonus": round(self.phrase_bonus, 4),
            "matched_terms": list(self.matched_terms)[:12],
            "passed": self.passed,
        }


def _overlap(query_toks: set[str], doc_toks: set[str]) -> float:
    if not query_toks:
        return 0.0
    hit = query_toks & doc_toks
    # recall-oriented vs query: how much of the query is covered
    return len(hit) / max(len(query_toks), 1)


# Generic connectors that match too many CS titles alone.
_WEAK = {
    "end", "to", "end-to-end", "end2end", "e2e", "real-time", "real", "time",
    "multi", "modal", "multi-modal", "system", "systems", "task", "tasks",
    "data", "learning", "deep", "neural", "network", "networks", "transformer",
    "transformers", "improved", "efficient", "via", "toward", "towards",
    # Social-science generics: alone they admit biology prestige on DiD/edu intents.
    "treatment", "treatments", "effect", "effects", "bias", "heterogeneity",
    "heterogeneous", "control", "response", "outcome", "outcomes", "policy",
    "impact", "impacts", "intervention", "interventions", "estimate", "estimator",
    "estimators", "estimation",
}

# If the primary intent is a named identification design, the paper must mention
# at least one design anchor (blocks "treatment/bias/heterogeneity" bleed).
_DESIGN_INTENT = (
    "difference-in-differences",
    "difference in differences",
    "diff-in-diff",
    "two-way fixed",
    "twfe",
    "staggered adoption",
    "event study",
    "goodman-bacon",
    "callaway",
    "synthetic control",
    "regression discontinuity",
    "instrumental variable",
    "双重差分",
    "断点回归",
    "工具变量",
    "合成控制",
)
_DESIGN_DOC_ANCHORS = (
    "difference-in-differences",
    "difference in differences",
    "differences-in-differences",
    "differences-in-difference",
    "diff-in-diff",
    "diff in diff",
    "did design",
    "twfe",
    "two-way fixed",
    "two way fixed",
    "staggered adoption",
    "staggered treatment",
    "event study",
    "event-study",
    "goodman-bacon",
    "goodman bacon",
    "callaway",
    "sun and abraham",
    "synthetic control",
    "regression discontinuity",
    "instrumental variable",
    "local average treatment",
    "wald estimator",
    "双重差分",
    "断点回归",
    "工具变量",
    "合成控制",
    "多期did",
    "多时点did",
)


def _score_one_query(
    q: str,
    *,
    title_toks: set[str],
    abs_toks: set[str],
    doc_toks: set[str],
    title_bi: set[tuple[str, ...]],
    abs_bi: set[tuple[str, ...]],
    abstract: str,
    landmark_bonus: float,
    min_score: float,
) -> Optional[RelevanceResult]:
    q_toks_list = tokenize(q)
    q_toks = set(q_toks_list)
    if not q_toks:
        return None
    strong_q = {t for t in q_toks if t not in _WEAK and len(t) > 2}
    if not strong_q:
        strong_q = set(q_toks)

    title_ov = _overlap(strong_q, title_toks)
    abs_ov = _overlap(strong_q, abs_toks)
    cov = _overlap(strong_q, doc_toks)
    matched = tuple(sorted(strong_q & doc_toks))

    q_bi = _ngrams(q_toks_list, 2)
    q_bi_strong = {bg for bg in q_bi if any(t not in _WEAK for t in bg)}
    bi_hits = 0
    if q_bi_strong:
        bi_hits = len(q_bi_strong & title_bi) + len(q_bi_strong & abs_bi)
    phrase = min(0.4, 0.15 * bi_hits)

    raw = 0.55 * title_ov + 0.30 * abs_ov + 0.15 * cov + phrase + landmark_bonus
    if not abstract and title_ov < 0.15 and landmark_bonus <= 0:
        raw *= 0.5
    score = max(0.0, min(1.8, raw))

    n_strong = len(strong_q)
    n_matched = len(matched)
    if landmark_bonus >= 0.4:
        passed = True
    elif n_strong >= 3:
        passed = n_matched >= 2 and (title_ov >= 0.15 or abs_ov >= 0.2 or bi_hits >= 1)
    elif n_strong == 2:
        passed = n_matched >= 2 or (n_matched >= 1 and bi_hits >= 1)
    else:
        passed = n_matched >= 1 and (title_ov >= 0.34 or bi_hits >= 1)
    passed = passed and score >= min_score

    return RelevanceResult(
        score=score,
        title_overlap=title_ov,
        abstract_overlap=abs_ov,
        phrase_bonus=phrase + landmark_bonus,
        matched_terms=matched,
        passed=passed,
    )


def score_relevance(
    query: str,
    card: PaperCard,
    *,
    alt_queries: Optional[Sequence[str]] = None,
    primary_queries: Optional[Sequence[str]] = None,
    min_score: float = 0.28,
) -> RelevanceResult:
    """Score topical fit of card to user query (+ optional expansions).

    Pass gate is driven by *primary* queries (core / user intent). Alternate
    expansions may raise the score but cannot alone admit an off-topic paper
    (fixes prestige bleed from weak synonym hits).
    """
    alts = [q for q in (alt_queries or []) if q]
    prim_src = [q for q in (primary_queries or []) if q]
    if not prim_src:
        # Default: user query + short latin alts only (not every fan-out string).
        prim_src = [query] + [q for q in alts if len(tokenize(q)) <= 14][:4]
    prim_src = list(dict.fromkeys([query, *prim_src]))
    support = list(dict.fromkeys([*prim_src, *alts]))

    title = card.title or ""
    abstract = card.abstract or ""
    venue = card.venue or ""

    scenario_q = query
    if not match_scenario(scenario_q):
        for a in support:
            if match_scenario(a or ""):
                scenario_q = a
                break

    noise_keys = scenario_noise_substrings(scenario_q)
    title_l = title.lower()
    abs_l = abstract.lower()
    doc_l = f"{title_l} {abs_l}"
    if noise_keys and any(n.lower() in title_l for n in noise_keys):
        return RelevanceResult(0.0, 0.0, 0.0, 0.0, (), False)

    intent_blob = " ".join(prim_src).lower()
    needs_design = any(m in intent_blob for m in _DESIGN_INTENT)
    has_design = (not needs_design) or any(a in doc_l for a in _DESIGN_DOC_ANCHORS)

    golds = scenario_gold_substrings(scenario_q)
    if golds and title_hits_gold(title, golds):
        return RelevanceResult(
            score=1.2,
            title_overlap=1.0,
            abstract_overlap=0.5,
            phrase_bonus=0.55,
            matched_terms=("gold_landmark",),
            passed=True,
        )

    landmark_bonus = 0.0
    for aq in support:
        aq_s = (aq or "").strip()
        if len(aq_s) >= 18 and aq_s.lower()[:48] in title_l:
            landmark_bonus = max(landmark_bonus, 0.55)
            break
        if len(aq_s.split()) >= 3:
            toks = [t for t in tokenize(aq_s) if t not in _WEAK][:4]
            if len(toks) >= 3 and all(t in tokenize(title) for t in toks[:3]):
                landmark_bonus = max(landmark_bonus, 0.4)

    title_toks = set(tokenize(title))
    abs_toks = set(tokenize(abstract))
    doc_toks = title_toks | abs_toks | set(tokenize(venue))
    title_bi = _ngrams(tokenize(title), 2)
    abs_bi = _ngrams(tokenize(abstract), 2)

    kw = dict(
        title_toks=title_toks,
        abs_toks=abs_toks,
        doc_toks=doc_toks,
        title_bi=title_bi,
        abs_bi=abs_bi,
        abstract=abstract,
        landmark_bonus=landmark_bonus,
        min_score=min_score,
    )

    best_primary: Optional[RelevanceResult] = None
    for q in prim_src:
        cand = _score_one_query(q, **kw)
        if cand is None:
            continue
        if best_primary is None or cand.score > best_primary.score:
            best_primary = cand

    best_any: Optional[RelevanceResult] = best_primary
    for q in support:
        cand = _score_one_query(q, **kw)
        if cand is None:
            continue
        if best_any is None or cand.score > best_any.score:
            best_any = cand

    if best_any is None:
        return RelevanceResult(0.0, 0.0, 0.0, 0.0, (), False)

    # Gate on primary; score can use best expansion.
    primary_ok = bool(best_primary and best_primary.passed) or landmark_bonus >= 0.4
    if not primary_ok:
        # Soft primary: high support score still needs some primary signal.
        p_score = best_primary.score if best_primary else 0.0
        primary_ok = p_score >= max(0.12, min_score * 0.45) and best_any.score >= min_score
        if not primary_ok:
            return RelevanceResult(
                score=best_any.score,
                title_overlap=best_any.title_overlap,
                abstract_overlap=best_any.abstract_overlap,
                phrase_bonus=best_any.phrase_bonus,
                matched_terms=best_any.matched_terms,
                passed=False,
            )

    passed = best_any.score >= min_score or landmark_bonus >= 0.4
    if needs_design and not has_design and landmark_bonus < 0.4:
        passed = False

    return RelevanceResult(
        score=best_any.score,
        title_overlap=best_any.title_overlap,
        abstract_overlap=best_any.abstract_overlap,
        phrase_bonus=best_any.phrase_bonus,
        matched_terms=best_any.matched_terms,
        passed=passed,
    )


def apply_relevance(
    card: PaperCard,
    query: str,
    *,
    alt_queries: Optional[Sequence[str]] = None,
    primary_queries: Optional[Sequence[str]] = None,
    min_score: float = 0.28,
    w_relevance: float = 6.0,
) -> tuple[PaperCard, RelevanceResult]:
    """Attach relevance to score_parts and boost card.score. Does not drop."""
    rel = score_relevance(
        query,
        card,
        alt_queries=alt_queries,
        primary_queries=primary_queries,
        min_score=min_score,
    )
    parts = dict(card.score_parts or {})
    parts["relevance"] = w_relevance * rel.score
    # Topic-first: damp prestige when topical fit is weak (even if gate passed).
    damp = 0.35 + 0.65 * min(1.0, max(rel.score, 0.0))
    for k in ("citations", "recency", "tier", "authority"):
        if k in parts and isinstance(parts[k], (int, float)):
            parts[k] = float(parts[k]) * damp
    card.score_parts = parts
    card.score = float(sum(v for v in parts.values() if isinstance(v, (int, float))))
    return card, rel


def filter_by_relevance(
    cards: Iterable[PaperCard],
    query: str,
    *,
    alt_queries: Optional[Sequence[str]] = None,
    primary_queries: Optional[Sequence[str]] = None,
    min_score: float = 0.28,
    w_relevance: float = 6.0,
) -> list[PaperCard]:
    kept: list[PaperCard] = []
    for c in cards:
        c2, rel = apply_relevance(
            c,
            query,
            alt_queries=alt_queries,
            primary_queries=primary_queries,
            min_score=min_score,
            w_relevance=w_relevance,
        )
        if rel.passed:
            kept.append(c2)
    kept.sort(key=lambda x: x.score, reverse=True)
    return kept
