"""Flagship scenario packs for reliable internal-beta demos."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

def _scenario_dir() -> Path:
    """Curated packs live beside the tier tables, so `TOPPER_DATA_DIR` moves both."""
    from topper import config

    return config.data_dir() / "scenarios"


@lru_cache(maxsize=1)
def load_scenarios() -> list[dict[str, Any]]:
    directory = _scenario_dir()
    if not directory.is_dir():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


def clear_scenario_cache() -> None:
    load_scenarios.cache_clear()


def match_scenario(query: str) -> Optional[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return None
    # also fold simple separators
    q_norm = re.sub(r"\s+", " ", q)
    for sc in load_scenarios():
        for m in sc.get("match_any") or []:
            if str(m).lower() in q_norm:
                return sc
    return None


def scenario_seed_queries(query: str) -> list[str]:
    sc = match_scenario(query)
    if not sc:
        return []
    return [str(x) for x in (sc.get("seed_queries") or []) if x]


def scenario_gold_substrings(query: str) -> list[str]:
    sc = match_scenario(query)
    if not sc:
        return []
    return [str(x) for x in (sc.get("gold_title_substrings") or []) if x]


def scenario_noise_substrings(query: str) -> list[str]:
    sc = match_scenario(query)
    if not sc:
        return []
    return [str(x) for x in (sc.get("noise_title_substrings") or []) if x]


def title_hits_gold(title: str, golds: list[str]) -> bool:
    t = (title or "").lower()
    return any(g.lower() in t for g in golds)


def title_hits_noise(title: str, noises: list[str]) -> bool:
    t = (title or "").lower()
    return any(n.lower() in t for n in noises)


def evaluate_against_scenario(
    query: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    sc = match_scenario(query)
    if not sc:
        return {"matched": False}
    golds = scenario_gold_substrings(query)
    noises = scenario_noise_substrings(query)
    hit = []
    noise_hit = []
    for r in results:
        title = r.get("title") or ""
        if title_hits_gold(title, golds):
            # dedupe by gold key
            for g in golds:
                if g.lower() in title.lower() and g not in hit:
                    hit.append(g)
        if title_hits_noise(title, noises):
            noise_hit.append(title[:80])
    acc = sc.get("acceptance") or {}
    ccf_a = sum(1 for r in results if (r.get("tiers") or {}).get("ccf") == "A")
    report = {
        "matched": True,
        "scenario_id": sc.get("id"),
        "n_results": len(results),
        "gold_hits": hit,
        "gold_hit_count": len(hit),
        "gold_total": len(golds),
        "noise_hits": noise_hit,
        "noise_count": len(noise_hit),
        "ccf_a": ccf_a,
        "pass": (
            len(results) >= int(acc.get("min_results") or 0)
            and len(hit) >= int(acc.get("min_gold_hits") or 0)
            and len(noise_hit) <= int(acc.get("max_noise_hits") or 99)
            and ccf_a >= int(acc.get("min_ccf_a") or 0)
        ),
        "acceptance": acc,
    }
    return report
