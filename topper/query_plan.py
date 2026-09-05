"""Query analysis → query_families → multi-round plan.

An LLM fills a Search Brief (`core` / `synonym_expansion` / optional families)
via ``search_brief``; this module turns it into an ordered round schedule.

Requires a configured LLM; raises if none is available.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from topper.text import has_cjk
from topper.search_brief import ROUND_SCHEDULE, llm_available, plan_search_brief
from topper.scenarios import match_scenario, scenario_seed_queries

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# CJK punctuation / fullwidth forms must never survive into a query string.
_CJK_PUNCT = re.compile(r"[\u3000-\u303f\uff00-\uffef]")
_WS = re.compile(r"\s+")
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-+/]{1,}")


@dataclass
class PlannedQuery:
    family: str
    query: str
    round: int
    reason: str = ""
    lang: str = "en"  # language of this query string

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    original: str
    language: str
    intent: str
    families: dict[str, list[str]]
    rounds: list[list[PlannedQuery]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    display: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "language": self.language,
            "intent": self.intent,
            "families": self.families,
            "rounds": [[q.to_dict() for q in rnd] for rnd in self.rounds],
            "notes": list(self.notes),
            "display": dict(self.display),
            "flat": [q.to_dict() for rnd in self.rounds for q in rnd],
        }

    def flat_queries(self) -> list[PlannedQuery]:
        return [q for rnd in self.rounds for q in rnd]


def _norm(s: str) -> str:
    s = _CJK_PUNCT.sub(" ", s or "")
    return _WS.sub(" ", s.strip())


# S2 is a keyword engine: a long translated sentence dilutes the match.
# Keep the content-bearing terms, drop interrogative scaffolding.
_QUERY_STOPWORDS = frozenset(
    """
    a an and are as at be been being by can could do does for from had has have how
    in into is it its of on or over recent recently since so some such than that the
    their there these they this those to under upon was were what when where which
    who why will with within would year years substantial progress advance advances
    made there's whats
    """.split()
)

_MAX_QUERY_CHARS = 120


# Family variants differentiate themselves with a trailing modifier
# ("... survey", "... benchmark"). Truncation must never drop it.
_TAIL_KEEP = 3


# Question scaffolding that carries no retrieval signal.
_TAIL_NOISE = re.compile(
    r"\b(in\s+)?recent\s+years?\b[?.!]*|\?+\s*$",
    re.IGNORECASE,
)


def compress_query(text: str, *, max_chars: int = _MAX_QUERY_CHARS) -> str:
    """Trim a natural-language question down to a keyword query for S2.

    Content words are kept in order, stopwords dropped. The last few tokens are
    preserved verbatim so sibling variants stay distinguishable after trimming.
    """
    s = _norm(_TAIL_NOISE.sub(" ", text or ""))
    if not s:
        return ""
    if len(s) <= max_chars:
        return s

    tokens = s.split()
    tail = tokens[-_TAIL_KEEP:] if len(tokens) > _TAIL_KEEP else []
    head = tokens[: -_TAIL_KEEP] if tail else tokens
    tail_len = sum(len(x) + 1 for x in tail)

    kept: list[str] = []
    total = 0
    budget = max(20, max_chars - tail_len)
    for token in head:
        bare = token.strip(".,;:?!()[]\"'").lower()
        if not bare or bare in _QUERY_STOPWORDS:
            continue
        if total + len(token) + 1 > budget:
            break
        kept.append(token)
        total += len(token) + 1

    out = " ".join(kept + tail).strip()
    return out or s[:max_chars]


# Two queries whose token sets overlap this much retrieve the same papers.
_SIMILARITY_CUTOFF = 0.82


def _too_similar(a: str, b: str) -> bool:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / max(len(ta), len(tb))
    return overlap >= _SIMILARITY_CUTOFF


_HEAD_TERMS = 7
# Chinese puts the topic last ("...的联邦学习"), so a translated sentence often
# carries the subject at the end. Take from both ends, not just the front.
_HEAD_FRONT = 4


def topic_head(text: str, *, max_terms: int = _HEAD_TERMS) -> str:
    """The topic itself: leading qualifiers plus the trailing subject."""
    s = _norm(_TAIL_NOISE.sub(" ", text or ""))
    words: list[str] = []
    for token in s.split():
        bare = token.strip(".,;:?!()[]\"'").lower()
        # drop translator artefacts like "vs." that carry no retrieval signal
        if not bare or bare in _QUERY_STOPWORDS or bare in {"vs", "v"}:
            continue
        words.append(token.strip(".,;:?!"))
    if len(words) <= max_terms:
        return " ".join(words)
    front = words[:_HEAD_FRONT]
    tail = words[-(max_terms - _HEAD_FRONT):]
    return " ".join(front + tail)


def _unique(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in seq:
        s = _norm(s)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    latin = sum(1 for c in text if ("a" <= c.lower() <= "z") or c.isdigit())
    return latin / max(len(text), 1)


def _query_lang(text: str) -> str:
    return "zh" if has_cjk(text) else "en"


def _families_from_search_brief(q: str) -> tuple[dict[str, list[str]], list[str], str, str, str]:
    """LLM Search Brief → query families."""
    if not llm_available():
        raise RuntimeError(
            "The planner needs an LLM. Set TOPPER_LLM_API_KEY and "
            "TOPPER_LLM_MODEL (see .env.example), then run `topper doctor`."
        )
    brief = plan_search_brief(q)
    families = dict(brief.get("query_families") or {})
    families["original"] = [q]
    # Flagship gold titles only when scenario pack matches (curated recall aids).
    sc = match_scenario(q)
    seeds = scenario_seed_queries(q)
    if sc and seeds:
        families.setdefault("landmark_seed", [])
        for s in seeds:
            if s not in families["landmark_seed"]:
                families["landmark_seed"].append(s)
    notes = list(brief.get("notes") or [])
    anchors = brief.get("topic_anchors") or []
    en_seed = " ".join(anchors[:3]) if anchors else (families.get("core") or [q])[0]
    intent = str(brief.get("intent") or "general")
    lang = str(brief.get("language") or ("zh" if has_cjk(q) else "en"))
    return families, notes, en_seed, intent, lang


def build_query_plan(
    original: str,
    *,
    max_per_round: int = 3,
    max_rounds: int = 3,
    translate: bool = True,  # noqa: ARG001 — kept for call-site compat; unused (LLM only)
    include_cjk_search: bool = False,
) -> QueryPlan:
    """LLM fills the query families; this schedules them into search rounds."""
    del translate  # legacy flag kept for call-site compatibility
    if match_scenario(original):
        max_per_round = max(max_per_round, 5)
        max_rounds = max(max_rounds, 3)
    q = _norm(original)
    if not q:
        raise ValueError("empty query")

    families, notes, en_seed, intent, lang = _families_from_search_brief(q)

    schedule = [list(slot) for slot in ROUND_SCHEDULE]
    if families.get("landmark_seed") and "landmark_seed" not in schedule[0]:
        schedule[0] = ["landmark_seed"] + schedule[0]

    rounds: list[list[PlannedQuery]] = []
    used: set[str] = set()

    def consider(fam: str, query: str, r_i: int, bucket: list[PlannedQuery]) -> bool:
        ql = _query_lang(query)
        if ql == "zh" and not include_cjk_search:
            return False
        if ql == "en":
            query = compress_query(query)
        if not query:
            return False
        key = query.lower()
        if key in used:
            return False
        for prior in used:
            if _too_similar(key, prior):
                return False
        used.add(key)
        bucket.append(
            PlannedQuery(
                family=fam,
                query=query,
                round=r_i,
                reason=f"family={fam}; intent={intent}; seed={str(en_seed)[:40]}",
                lang=ql,
            )
        )
        return True

    for r_i, fam_names in enumerate(schedule[:max_rounds], start=1):
        bucket: list[PlannedQuery] = []
        for fam in fam_names:
            for query in families.get(fam) or []:
                consider(fam, query, r_i, bucket)
                if len(bucket) >= max_per_round:
                    break
            if len(bucket) >= max_per_round:
                break
        if bucket:
            rounds.append(bucket)

    leftover_landmarks = [
        s for s in (families.get("landmark_seed") or []) if s.lower() not in used
    ]
    if leftover_landmarks and match_scenario(q):
        r_i = len(rounds) + 1
        chunk = max(max_per_round, 8)
        for i in range(0, len(leftover_landmarks), chunk):
            bucket = []
            for query in leftover_landmarks[i : i + chunk]:
                consider("landmark_seed", query, r_i, bucket)
            if bucket:
                rounds.append(bucket)
                r_i += 1

    if not rounds:
        # LLM brief must still yield at least core; if compress wiped everything, use seed.
        seed = _norm(str(en_seed) or "")
        if has_cjk(seed):
            seed = " ".join(_LATIN_TOKEN.findall(seed)) or "research"
        if not seed:
            raise RuntimeError("RH brief produced no schedulable English queries")
        rounds = [
            [
                PlannedQuery(
                    family="core",
                    query=seed,
                    round=1,
                    reason="brief_core_seed",
                    lang="en",
                )
            ]
        ]

    display = {
        "title_zh": "顶刊检索展开",
        "title_en": "Top-paper query expansion",
        "intent": intent,
        "original": q,
        "english_seed": en_seed,
        "planner": "rh",
    }

    return QueryPlan(
        original=q,
        language=lang,
        intent=intent,
        families=families,
        rounds=rounds,
        notes=notes,
        display=display,
    )
