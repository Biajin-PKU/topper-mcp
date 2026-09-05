"""PaSa-style topical selector (batch LLM judge on title+abstract).

Optional second gate after lexical relevance. Off by default for latency;
enable with TOPPER_SELECTOR_LLM=1.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from topper.models import PaperCard
from topper.search_brief import _chat_json, llm_available

_SYSTEM = """You are an academic paper selector (PaSa-style).
Given a research question and candidate papers (title + abstract),
return JSON: {"keep": ["id1", "id2", ...], "drop": ["id3", ...], "notes": "..."}.

Rules:
- keep ONLY papers clearly on-topic for the question.
- Prestige/venue/citations do NOT justify keeping off-topic work.
- When unsure, drop.
- keep/drop ids must come from the candidate list.
"""


def selector_enabled() -> bool:
    return os.environ.get("TOPPER_SELECTOR_LLM", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def select_cards(
    query: str,
    cards: list[PaperCard],
    *,
    limit: int = 40,
    timeout: float = 60.0,
) -> list[PaperCard]:
    """Return cards judged on-topic. On failure/disabled, return input unchanged."""
    if not cards or not selector_enabled() or not llm_available():
        return cards
    head = cards[: max(1, limit)]
    tail = cards[len(head) :]
    payload = [
        {
            "id": c.id,
            "title": (c.title or "")[:160],
            "abstract": (c.abstract or "")[:280],
            "year": c.year,
            "venue": (c.venue or "")[:60],
        }
        for c in head
    ]
    try:
        raw = _chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\nCandidates:\n"
                        + json.dumps(payload, ensure_ascii=False)
                        + "\n\nReturn keep/drop JSON."
                    ),
                },
            ],
            timeout=float(os.environ.get("TOPPER_SELECTOR_TIMEOUT", str(timeout)) or timeout),
        )
    except Exception:  # noqa: BLE001
        return cards

    keep_ids = {str(x) for x in (raw.get("keep") or []) if x}
    drop_ids = {str(x) for x in (raw.get("drop") or []) if x}
    if not keep_ids and not drop_ids:
        return cards

    # If model only listed drops, keep the rest of head.
    if keep_ids:
        kept_head = [c for c in head if c.id in keep_ids]
    else:
        kept_head = [c for c in head if c.id not in drop_ids]

    # Never return empty if we had inputs — fall back.
    if not kept_head and head:
        return cards

    for c in kept_head:
        sp = dict(c.score_parts or {})
        sp["selector"] = 1.0
        c.score_parts = sp
        c.score = float(sum(v for v in sp.values() if isinstance(v, (int, float))))

    # dropped head cards are discarded; tail (beyond selector window) kept as-is
    out = kept_head + tail
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def select_result_dicts(
    query: str,
    rows: list[dict[str, Any]],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Dict-level wrapper used at multi_search finish."""
    if not selector_enabled() or not rows:
        return rows
    cards = []
    for r in rows[:limit]:
        cards.append(
            PaperCard(
                id=str(r.get("id") or ""),
                title=str(r.get("title") or ""),
                abstract=str(r.get("abstract") or ""),
                year=r.get("year"),
                venue=r.get("venue"),
                cited_by_count=int(r.get("cited_by_count") or 0),
                score=float(r.get("score") or 0),
                score_parts=dict(r.get("score_parts") or {}),
            )
        )
    selected = select_cards(query, cards, limit=limit)
    if selected is cards or not selected:
        return rows
    id_order = [c.id for c in selected]
    by_id = {str(r.get("id") or ""): r for r in rows}
    out = [by_id[i] for i in id_order if i in by_id]
    # append any rows beyond limit unchanged
    seen = set(id_order)
    for r in rows:
        rid = str(r.get("id") or "")
        if rid not in seen:
            out.append(r)
    return out
