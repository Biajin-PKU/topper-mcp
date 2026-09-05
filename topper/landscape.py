"""Research-direction landscape from retrieved top papers.

Groups the result set into schools of thought and active teams.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Optional

from topper.scenarios import match_scenario, title_hits_gold


def _title(r: dict[str, Any]) -> str:
    return str(r.get("title") or "").strip()


_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Well-known affiliation-name variants → one canonical display name.
# Keys are the folded form (see _fold_org).
_ORG_ALIASES: dict[str, str] = {
    "allen institute for ai": "Allen Institute for AI",
    "allen institute of ai": "Allen Institute for AI",
    "ai2": "Allen Institute for AI",
}

# Words that make a comma/slash piece look like a full organization name
# (used to decide whether "A, B" is two orgs vs one org with a campus qualifier).
_ORG_KEYWORDS = (
    "university",
    "institute",
    "institutes",
    "college",
    "academy",
    "laboratory",
    "laboratories",
    "lab",
    "labs",
    "school",
    "center",
    "centre",
    "hospital",
    "clinic",
    "research",
    "department",
    "ministry",
    "corporation",
    "company",
    "inc",
    "ltd",
    "gmbh",
    "openai",
    "deepmind",
    "anthropic",
)


def _fold_org(name: str) -> str:
    """Normalize an org name for grouping (lower, strip, abbrev, punctuation)."""
    s = str(name or "").lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("artificial intelligence", "ai")
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _looks_like_org(piece: str) -> bool:
    return any(re.search(rf"\b{re.escape(kw)}\b", piece) for kw in _ORG_KEYWORDS)


def _org_pieces(name: str) -> list[tuple[str, str]]:
    """Return (canonical_key, display) pairs for a raw institution string.

    Handles three cases:
    * name variants ("Allen Institute for/of AI" → one key),
    * combined strings ("University of Arizona, Allen Institute for AI" → two keys),
    * embedded known orgs inside a longer raw string.
    """
    raw = " ".join(str(name or "").split())
    folded = _fold_org(raw)
    if not folded or len(folded) < 3:
        return []

    # 1) whole string is a known variant
    if folded in _ORG_ALIASES:
        disp = _ORG_ALIASES[folded]
        return [(disp, disp)]

    # 2) combined "A, B / C" — split only when every piece is itself an org
    raw_parts = re.split(r"\s*[,;/|]\s*", raw)
    if len(raw_parts) > 1:
        folded_parts = [_fold_org(p) for p in raw_parts]
        if folded_parts and all(_looks_like_org(fp) for fp in folded_parts):
            out: list[tuple[str, str]] = []
            for rp, fp in zip(raw_parts, folded_parts):
                canon = _ORG_ALIASES.get(fp)
                out.append((canon, canon) if canon else (fp, rp.strip()))
            return out

    # 3) known org embedded in a longer raw string
    for ak, disp in _ORG_ALIASES.items():
        if f" {ak}" in f" {folded}":
            return [(disp, disp)]

    # 4) plain org
    return [(folded, raw)]


def _match_papers(results: list[dict[str, Any]], keys: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in results:
        t = _title(r)
        if not t or t in seen:
            continue
        if keys and title_hits_gold(t, keys):
            seen.add(t)
            out.append(
                {
                    "title": t,
                    "year": str(r.get("year") or ""),
                    "venue": str(r.get("venue") or ""),
                    "id": str(r.get("id") or ""),
                }
            )
    return out


def _bind_curated(pack: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    schools_out = []
    for s in pack.get("schools") or []:
        keys = [str(k) for k in (s.get("paper_keys") or []) if k]
        hits = _match_papers(results, keys)
        schools_out.append(
            {
                "id": s.get("id") or "",
                "name_zh": s.get("name_zh") or s.get("name") or "",
                "name_en": s.get("name_en") or "",
                "desc_zh": s.get("desc_zh") or "",
                "sota_hint_zh": s.get("sota_hint_zh") or "",
                "papers": hits[:6],
            }
        )

    teams_out = []
    for t in pack.get("teams") or []:
        keys = [str(k) for k in (t.get("paper_keys") or []) if k]
        hits = _match_papers(results, keys)
        teams_out.append(
            {
                "name": t.get("name") or "",
                "note_zh": t.get("note_zh") or "",
                "papers": hits[:4],
            }
        )

    benches_out = []
    for b in pack.get("benchmarks") or []:
        keys = [str(k) for k in (b.get("paper_keys") or [b.get("name")]) if k]
        hits = _match_papers(results, keys)
        benches_out.append(
            {
                "name": b.get("name") or "",
                "focus_zh": b.get("focus_zh") or "",
                "papers": hits[:3],
            }
        )

    return {
        "source": "curated",
        "scenario_id": pack.get("scenario_id") or "",
        "title_zh": pack.get("title_zh") or "研究方向梳理",
        "summary_zh": pack.get("summary_zh") or "",
        "schools": schools_out,
        "teams": teams_out,
        "benchmarks": benches_out,
    }


_BENCH_KW = (
    "benchmark",
    "bench",
    "evaluation",
    "evaluating",
    "dataset",
    "leaderboard",
    "评测",
    "基准",
)
_SURVEY_KW = ("survey", "review", "roadmap", "综述", "展望")
_AGENT_KW = ("agent", "scientist", "autonomous", "automated research", "智能体", "自动化")


def _heuristic(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Cheap direction sketch from the current top-paper list."""
    schools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in results:
        t = _title(r)
        if not t:
            continue
        tl = t.lower()
        item = {
            "title": t,
            "year": str(r.get("year") or ""),
            "venue": str(r.get("venue") or ""),
            "id": str(r.get("id") or ""),
        }
        if any(k in tl for k in _BENCH_KW):
            schools["benchmark"].append(item)
        elif any(k in tl for k in _SURVEY_KW):
            schools["survey"].append(item)
        elif any(k in tl for k in _AGENT_KW):
            schools["system"].append(item)
        else:
            schools["method"].append(item)

    label = {
        "system": ("系统 / Agent 路线", "端到端或代理式系统"),
        "method": ("方法与模型", "具体算法、架构或训练方法"),
        "benchmark": ("评测与基准", "benchmark / dataset / evaluation"),
        "survey": ("综述与路线图", "survey / roadmap"),
    }
    schools_out = []
    for key in ("system", "method", "benchmark", "survey"):
        papers = schools.get(key) or []
        if not papers:
            continue
        name, desc = label[key]
        # rank by citation if present
        ranked = sorted(
            papers,
            key=lambda p: next(
                (
                    int(r.get("cited_by_count") or 0)
                    for r in results
                    if _title(r) == p["title"]
                ),
                0,
            ),
            reverse=True,
        )
        schools_out.append(
            {
                "id": key,
                "name_zh": name,
                "name_en": key,
                "desc_zh": desc,
                "sota_hint_zh": f"本批高引代表：{ranked[0]['title'][:60]}" if ranked else "",
                "papers": ranked[:5],
            }
        )

    org_c: Counter[str] = Counter()
    org_display: dict[str, str] = {}
    org_papers: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in results:
        t = _title(r)
        item = {
            "title": t,
            "year": str(r.get("year") or ""),
            "venue": str(r.get("venue") or ""),
            "id": str(r.get("id") or ""),
        }
        for o in r.get("institutions") or []:
            for key, disp in _org_pieces(o):
                if not key:
                    continue
                org_c[key] += 1
                org_display.setdefault(key, disp)
                if len(org_papers[key]) < 3:
                    org_papers[key].append(item)

    teams_out = []
    for key, cnt in org_c.most_common(6):
        if cnt < 1:
            continue
        teams_out.append(
            {
                "name": org_display.get(key, key),
                "note_zh": f"本批结果中出现 {cnt} 篇",
                "papers": org_papers.get(key) or [],
            }
        )

    benches_out = []
    for s in schools_out:
        if s["id"] == "benchmark":
            for p in s["papers"][:6]:
                benches_out.append(
                    {
                        "name": p["title"].split(":")[0].strip()[:48],
                        "focus_zh": "开源/公开评测（从标题识别）",
                        "papers": [p],
                    }
                )

    return {
        "source": "heuristic",
        "scenario_id": "",
        "title_zh": "研究方向梳理（基于本批顶刊）",
        "summary_zh": (
            f"围绕「{query.strip()[:40]}」，从本批 {len(results)} 篇顶刊中"
            "归纳方法路线、持续产出机构与评测基准；非穷尽综述。"
        ),
        "schools": schools_out,
        "teams": teams_out,
        "benchmarks": benches_out,
    }


def build_landscape(
    query: str,
    results: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return a landscape dict, or None if there is nothing useful to show.

    Flagship scenarios keep curated packs. Everything else prefers a
    cite-grounded ReportAST (LLM when available) adapted to this UI shape.
    """
    if not results:
        return None
    sc = match_scenario(query)
    if sc and isinstance(sc.get("landscape"), dict):
        pack = dict(sc["landscape"])
        pack["scenario_id"] = sc.get("id") or ""
        return _bind_curated(pack, results)
    try:
        from topper.report import build_report, report_to_landscape

        rep = build_report(query, results)
        if rep:
            return report_to_landscape(rep)
    except Exception:  # noqa: BLE001 — never fail the search path on report
        pass
    return _heuristic(query, results)
