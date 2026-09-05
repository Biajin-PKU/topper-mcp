"""Multi-round top-only search with query-family fan-out."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from topper.arxiv_bridge import ccf_a_author_pool, try_admit_arxiv
from topper.models import PaperCard, SearchPolicy
from topper.pipeline import search
from topper.query_plan import QueryPlan, build_query_plan


ProgressCb = Callable[[dict[str, Any]], None]


@dataclass
class MultiSearchState:
    plan: QueryPlan
    seen_ids: set[str] = field(default_factory=set)
    results: list[PaperCard] = field(default_factory=list)
    pending_arxiv: list[PaperCard] = field(default_factory=list)
    rounds_done: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    arxiv_admitted: int = 0

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "rounds_done": self.rounds_done,
            "rounds_total": len(self.plan.rounds),
            "count": len(self.results),
            "arxiv_bridged": self.arxiv_admitted,
            "results": [r.to_dict() for r in self.results],
        }


def _card_id(c: PaperCard) -> str:
    return c.id or f"{c.title}|{c.year}|{c.venue}"


def _is_pending_arxiv(card: PaperCard) -> bool:
    ma = card.matched_authority or []
    if "arxiv_pending" in ma:
        return True
    sp = card.score_parts or {}
    return float(sp.get("arxiv_pending") or 0) > 0


def _flush_arxiv_bridge(
    state: MultiSearchState,
    *,
    pol: SearchPolicy,
    alt_for_rel: list[str],
) -> list[PaperCard]:
    """Admit pending arXiv only with author bridge AND relevance (already checked)."""
    if not state.pending_arxiv:
        return []
    pool = ccf_a_author_pool(state.results)
    still: list[PaperCard] = []
    newly: list[PaperCard] = []
    for card in state.pending_arxiv:
        cid = _card_id(card)
        if cid in state.seen_ids and any(_card_id(r) == cid for r in state.results):
            continue
        ok, card2, reason = try_admit_arxiv(
            card,
            query=state.plan.original,
            alt_queries=alt_for_rel,
            policy=pol,
            ccf_a_authors=pool,
        )
        if ok:
            # clear pending marker
            card2.matched_authority = [
                x for x in (card2.matched_authority or []) if x != "arxiv_pending"
            ]
            sp = dict(card2.score_parts or {})
            sp.pop("arxiv_pending", None)
            card2.score_parts = sp
            if cid not in state.seen_ids:
                state.seen_ids.add(cid)
            # avoid dup titles already in results
            if not any(_card_id(r) == cid for r in state.results):
                state.results.append(card2)
                newly.append(card2)
                state.arxiv_admitted += 1
        else:
            # keep waiting if only missing CCF-A author pool; drop off-topic permanently
            if reason in {"no_author_bridge"} and not pool:
                still.append(card)
            elif reason == "no_author_bridge":
                # pool exists but this author didn't match — drop
                pass
            # off_topic / too_old already rejected
    state.pending_arxiv = still
    state.results.sort(key=lambda c: c.score, reverse=True)
    return newly


def run_round(
    state: MultiSearchState,
    *,
    round_index: int,
    policy: Optional[SearchPolicy] = None,
    limit_per_query: int = 8,
    use_proxy: Optional[bool] = None,
    fetch_multiplier: int = 4,
    source: str = "s2",
) -> dict[str, Any]:
    """Execute one plan round; mutate state; return round event payload."""
    if round_index < 1 or round_index > len(state.plan.rounds):
        return {
            "type": "round_skip",
            "round": round_index,
            "reason": "out_of_range",
        }

    planned = state.plan.rounds[round_index - 1]
    pol = policy or SearchPolicy()
    new_cards: list[PaperCard] = []
    query_reports: list[dict[str, Any]] = []

    # All planned strings + original — Selector-style topical fit
    alt_for_rel = [state.plan.original] + [
        q.query for rnd in state.plan.rounds for q in rnd
    ]
    # Primary gate: user text + core family only (not every synonym/fan-out).
    primary_for_rel = [state.plan.original] + [
        q.query
        for rnd in state.plan.rounds
        for q in rnd
        if q.family in {"core", "landmark_seed"} and q.query
    ]
    # topic anchors from display seed if present
    en_seed = (state.plan.display or {}).get("english_seed")
    if en_seed:
        primary_for_rel.append(str(en_seed))

    for pq in planned:
        t0 = time.time()
        try:
            hits = search(
                pq.query,
                policy=pol,
                limit=limit_per_query,
                source=source,
                use_proxy=use_proxy,
                fetch_multiplier=fetch_multiplier,
                relevance_queries=alt_for_rel,
                primary_queries=primary_for_rel,
            )
            err = None
        except Exception as e:  # noqa: BLE001 — surface per-query soft fail
            hits = []
            err = str(e)
        elapsed = time.time() - t0
        added = 0
        for h in hits:
            cid = _card_id(h)
            if _is_pending_arxiv(h):
                # park — do not enter results until author bridge + topic OK
                if cid not in {_card_id(x) for x in state.pending_arxiv}:
                    state.pending_arxiv.append(h)
                continue
            if cid in state.seen_ids:
                continue
            state.seen_ids.add(cid)
            h.score_parts = dict(h.score_parts or {})
            h.score_parts["query_family"] = 0.0
            state.results.append(h)
            new_cards.append(h)
            added += 1
        state.results.sort(key=lambda c: c.score, reverse=True)
        query_reports.append(
            {
                "family": pq.family,
                "query": pq.query,
                "round": pq.round,
                "hits": len(hits),
                "added": added,
                "elapsed_s": round(elapsed, 3),
                "error": err,
            }
        )

    bridged = _flush_arxiv_bridge(state, pol=pol, alt_for_rel=alt_for_rel)
    new_cards.extend(bridged)

    state.rounds_done = max(state.rounds_done, round_index)
    event = {
        "type": "round_done",
        "round": round_index,
        "rounds_total": len(state.plan.rounds),
        "queries": query_reports,
        "arxiv_bridged_this_round": len(bridged),
        "arxiv_bridged_total": state.arxiv_admitted,
        "new_count": len(new_cards),
        "total_count": len(state.results),
        "new_results": [c.to_dict() for c in new_cards],
        "results": [c.to_dict() for c in state.results],
    }
    state.events.append(event)
    return event


def multi_search_stream(
    query: str,
    *,
    policy: Optional[SearchPolicy] = None,
    max_rounds: int = 3,
    max_per_round: int = 3,
    limit_per_query: int = 8,
    use_proxy: Optional[bool] = None,
    fetch_multiplier: int = 4,
    source: str = "s2",
    translate: bool = True,
    stop_when_total: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """Yield SSE-ready events: plan → round_start → round_done* → done."""
    from topper.scenarios import match_scenario

    # Flagship demo scenario: denser fan-out + more per-query headroom.
    # Run every planned round (incl. leftover landmark drain) — no early stop.
    scenario = match_scenario(query)
    if scenario:
        max_per_round = max(max_per_round, 5)
        max_rounds = max(max_rounds, 3)
        limit_per_query = max(limit_per_query, 10)
        if stop_when_total is not None:
            stop_when_total = max(stop_when_total, 36)
        stop_when_total_effective = None  # finish all landmark rounds
    else:
        stop_when_total_effective = stop_when_total

    t_plan0 = time.time()
    plan = build_query_plan(
        query,
        max_per_round=max_per_round,
        max_rounds=max_rounds,
        translate=translate,
    )
    plan_s = round(time.time() - t_plan0, 3)
    state = MultiSearchState(plan=plan)
    yield {
        "type": "plan",
        "plan": plan.to_dict(),
        "plan_s": plan_s,
        "message": f"已分析意图「{plan.intent}」，展开 {sum(len(r) for r in plan.rounds)} 路查询 · {len(plan.rounds)} 轮",
    }

    for r_i in range(1, len(plan.rounds) + 1):
        yield {
            "type": "round_start",
            "round": r_i,
            "rounds_total": len(plan.rounds),
            "queries": [q.to_dict() for q in plan.rounds[r_i - 1]],
            "message": f"第 {r_i}/{len(plan.rounds)} 轮检索…",
        }
        event = run_round(
            state,
            round_index=r_i,
            policy=policy,
            limit_per_query=limit_per_query,
            use_proxy=use_proxy,
            fetch_multiplier=fetch_multiplier,
            source=source,
        )
        yield event
        # final arxiv bridge pass after each round already ran; one more at stop
        pol = policy or SearchPolicy()
        alt = [plan.original] + [q.query for rnd in plan.rounds for q in rnd]
        _flush_arxiv_bridge(state, pol=pol, alt_for_rel=alt)

        # Adaptive stop: enough strong topical hits after ≥1 round (non-scenario).
        if (
            not scenario
            and r_i >= 1
            and _strong_topical_count(state.results, pol) >= _adaptive_strong_k()
            and (stop_when_total_effective is None or len(state.results) >= min(12, stop_when_total_effective or 12))
        ):
            yield _finish(
                query,
                state,
                plan,
                reason="adaptive_quality",
                cap=stop_when_total,
            )
            return

        if stop_when_total_effective and len(state.results) >= stop_when_total_effective:
            yield _finish(
                query,
                state,
                plan,
                reason="target_reached",
                cap=stop_when_total_effective,
            )
            return

    pol = policy or SearchPolicy()
    alt = [plan.original] + [q.query for rnd in plan.rounds for q in rnd]
    _flush_arxiv_bridge(state, pol=pol, alt_for_rel=alt)
    yield _finish(query, state, plan, reason="all_rounds", cap=stop_when_total)


def _adaptive_strong_k() -> int:
    try:
        return max(4, int(os.environ.get("TOPPER_ADAPTIVE_STRONG_K", "8") or 8))
    except ValueError:
        return 8


def _strong_topical_count(cards: list[PaperCard], pol: SearchPolicy) -> int:
    """Count cards whose relevance component is clearly on-topic."""
    thr = max(float(pol.min_relevance), 0.28) * float(pol.w_relevance) * 0.85
    n = 0
    for c in cards:
        sp = c.score_parts or {}
        rel = float(sp.get("relevance") or 0.0)
        if rel >= thr:
            n += 1
    return n


def _finish(
    query: str,
    state: MultiSearchState,
    plan: QueryPlan,
    *,
    reason: str,
    cap: Optional[int],
) -> dict[str, Any]:
    from topper.landscape import build_landscape
    from topper.selector import select_result_dicts

    results_out = state.results[:cap] if cap else state.results
    result_dicts = [c.to_dict() for c in results_out]
    result_dicts = select_result_dicts(query, result_dicts)
    landscape = build_landscape(query, result_dicts)
    payload: dict[str, Any] = {
        "type": "done",
        "reason": reason,
        "total_count": len(state.results),
        "arxiv_bridged": state.arxiv_admitted,
        "results": result_dicts,
        "plan": plan.to_dict(),
        "rounds_done": state.rounds_done,
    }
    if landscape:
        payload["landscape"] = landscape
    return payload


def multi_search(
    query: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Blocking helper: run full stream, return final snapshot."""
    final: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for ev in multi_search_stream(query, **kwargs):
        events.append(ev)
        if ev.get("type") == "done":
            final = ev
    final["events"] = events
    return final
