"""Cite-grounded research brief from the current top-paper set (PaperQA-style).

RH contract sketch: synthesize_literature_brief(cards) -> ReportAST.
Uses LLM when available; otherwise a deterministic skeleton still bound to paper ids.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from topper.search_brief import _chat_json, llm_available

_SYSTEM = """You write a short Chinese research briefing from a FIXED paper list only.

Output ONE JSON object (no markdown):
{
  "title_zh": "…",
  "summary_zh": "2–4 sentences; every factual claim must reference [P#] ids",
  "lines": [
    {
      "name_zh": "路线名",
      "desc_zh": "1–2 sentences with [P#] citations",
      "paper_ids": ["id1","id2"],
      "sota_hint_zh": "optional, with [P#] if any"
    }
  ],
  "gaps_zh": ["本批未覆盖或仍开放的问题…"],
  "reading_order": ["id…"]
}

Rules:
- ONLY use papers from the provided list. Never invent titles, venues, or years.
- Cite with [P#] matching the given indices. Prefer 3–6 lines.
- If evidence is thin, say so; put unknowns in gaps_zh.
- Chinese body text; keep paper titles in original language when quoting.
"""


def _card_view(r: dict[str, Any], idx: int) -> dict[str, Any]:
    tiers = r.get("tiers") or {}
    return {
        "P": idx,
        "id": str(r.get("id") or f"row-{idx}"),
        "title": (r.get("title") or "")[:180],
        "year": r.get("year"),
        "venue": (r.get("venue") or "")[:80],
        "cites": r.get("cited_by_count") or 0,
        "ccf": tiers.get("ccf"),
        "cas": tiers.get("cas_zone"),
        "abstract": ((r.get("abstract") or "")[:420]),
        "institutions": (r.get("institutions") or [])[:4],
    }


def _heuristic_report(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic fallback: cluster by simple title cues, always cite ids."""
    from collections import defaultdict

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels = {
        "survey": ("综述与路线图", "survey / review 线索"),
        "benchmark": ("评测与数据", "benchmark / evaluation / dataset"),
        "method": ("方法与系统", "具体方法、模型或系统工作"),
    }
    for i, r in enumerate(results, 1):
        t = (r.get("title") or "").lower()
        view = _card_view(r, i)
        if any(k in t for k in ("survey", "review", "roadmap", "综述")):
            buckets["survey"].append(view)
        elif any(k in t for k in ("benchmark", "evaluat", "dataset", "leaderboard")):
            buckets["benchmark"].append(view)
        else:
            buckets["method"].append(view)

    lines = []
    for key in ("method", "benchmark", "survey"):
        papers = buckets.get(key) or []
        if not papers:
            continue
        name, desc = labels[key]
        ids = [p["id"] for p in papers[:4]]
        cites = "、".join(f"[P{p['P']}]" for p in papers[:3])
        lines.append(
            {
                "name_zh": name,
                "desc_zh": f"{desc}；本批代表作包括 {cites}。",
                "paper_ids": ids,
                "sota_hint_zh": f"高引线索：[P{papers[0]['P']}] {(papers[0].get('title') or '')[:60]}",
                "papers": [
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "year": str(p.get("year") or ""),
                        "venue": p.get("venue") or "",
                    }
                    for p in papers[:5]
                ],
            }
        )

    org_c: dict[str, int] = {}
    org_papers: dict[str, list[dict[str, str]]] = {}
    for i, r in enumerate(results, 1):
        item = {
            "id": str(r.get("id") or ""),
            "title": r.get("title") or "",
            "year": str(r.get("year") or ""),
            "venue": r.get("venue") or "",
        }
        for o in r.get("institutions") or []:
            name = str(o).strip()
            if not name:
                continue
            org_c[name] = org_c.get(name, 0) + 1
            org_papers.setdefault(name, [])
            if len(org_papers[name]) < 3:
                org_papers[name].append(item)

    teams = [
        {
            "name": n,
            "note_zh": f"本批出现 {c} 篇",
            "papers": org_papers.get(n) or [],
        }
        for n, c in sorted(org_c.items(), key=lambda x: -x[1])[:6]
    ]

    id_by_p = {i: str(r.get("id") or f"row-{i}") for i, r in enumerate(results, 1)}
    reading = [id_by_p[i] for i in range(1, min(8, len(results) + 1))]
    summary = (
        f"围绕「{query.strip()[:40]}」，基于本批 {len(results)} 篇顶刊/顶会结果整理主线；"
        f"下列论断均绑定列表内论文编号，非穷尽综述。"
    )
    return {
        "source": "heuristic_report",
        "title_zh": "研究方向简报（基于本批顶刊）",
        "summary_zh": summary,
        "lines": lines,
        "gaps_zh": ["完整机制对比与跨库遗漏需结合全文精读，本批仅覆盖检索命中。"],
        "reading_order": reading,
        "teams": teams,
        "papers_index": [_card_view(r, i) for i, r in enumerate(results, 1)],
    }


def _bind_lines(
    lines_in: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(r.get("id") or ""): r for r in results if r.get("id")}
    # also map P indices
    by_p = {i: r for i, r in enumerate(results, 1)}
    out = []
    for ln in lines_in or []:
        if not isinstance(ln, dict):
            continue
        ids: list[str] = []
        for x in ln.get("paper_ids") or []:
            s = str(x)
            if s in by_id:
                ids.append(s)
            else:
                m = re.match(r"P?(\d+)$", s, re.I)
                if m:
                    r = by_p.get(int(m.group(1)))
                    if r and r.get("id"):
                        ids.append(str(r["id"]))
        # scrape [P#] from text
        blob = f"{ln.get('desc_zh') or ''} {ln.get('sota_hint_zh') or ''}"
        for m in re.finditer(r"\[P(\d+)\]", blob):
            r = by_p.get(int(m.group(1)))
            if r and r.get("id"):
                ids.append(str(r["id"]))
        ids = list(dict.fromkeys(ids))
        papers = []
        for pid in ids[:6]:
            r = by_id.get(pid)
            if not r:
                continue
            papers.append(
                {
                    "id": pid,
                    "title": r.get("title") or "",
                    "year": str(r.get("year") or ""),
                    "venue": r.get("venue") or "",
                }
            )
        out.append(
            {
                "name_zh": str(ln.get("name_zh") or "研究线索")[:40],
                "desc_zh": str(ln.get("desc_zh") or "")[:400],
                "paper_ids": ids[:6],
                "sota_hint_zh": str(ln.get("sota_hint_zh") or "")[:240],
                "papers": papers,
            }
        )
    return out


def _llm_report(query: str, results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not llm_available() or not results:
        return None
    if os.environ.get("TOPPER_REPORT_LLM", "1").strip() in {"0", "false", "no"}:
        return None
    views = [_card_view(r, i) for i, r in enumerate(results[:24], 1)]
    try:
        raw = _chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{query}\n\nPapers (cite by P only):\n"
                        + json.dumps(views, ensure_ascii=False)
                        + "\n\nReturn the JSON briefing now."
                    ),
                },
            ],
            timeout=float(os.environ.get("TOPPER_REPORT_TIMEOUT", "90") or 90),
        )
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    lines = _bind_lines(list(raw.get("lines") or []), results)
    if not lines:
        return None
    by_id = {str(r.get("id") or ""): r for r in results}
    reading = []
    for x in raw.get("reading_order") or []:
        s = str(x)
        if s in by_id:
            reading.append(s)
        else:
            m = re.match(r"P?(\d+)$", s, re.I)
            if m:
                r = results[int(m.group(1)) - 1] if 1 <= int(m.group(1)) <= len(results) else None
                if r and r.get("id"):
                    reading.append(str(r["id"]))
    reading = list(dict.fromkeys(reading))[:10]
    gaps = [str(g) for g in (raw.get("gaps_zh") or []) if g][:6]
    # teams from institutions in cited papers
    base = _heuristic_report(query, results)
    return {
        "source": "llm_report",
        "title_zh": str(raw.get("title_zh") or "研究方向简报")[:80],
        "summary_zh": str(raw.get("summary_zh") or "")[:800],
        "lines": lines,
        "gaps_zh": gaps or base.get("gaps_zh") or [],
        "reading_order": reading or base.get("reading_order") or [],
        "teams": base.get("teams") or [],
        "papers_index": views,
    }


def build_report(query: str, results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return ReportAST-like dict, or None if no results."""
    if not results:
        return None
    llm = _llm_report(query, results)
    if llm:
        return llm
    return _heuristic_report(query, results)


def report_to_landscape(report: dict[str, Any]) -> dict[str, Any]:
    """Adapt ReportAST into the existing landscape UI shape."""
    schools = []
    for i, ln in enumerate(report.get("lines") or []):
        schools.append(
            {
                "id": f"line-{i}",
                "name_zh": ln.get("name_zh") or "",
                "name_en": "",
                "desc_zh": ln.get("desc_zh") or "",
                "sota_hint_zh": ln.get("sota_hint_zh") or "",
                "papers": ln.get("papers") or [],
            }
        )
    benches = []
    for ln in report.get("lines") or []:
        name = (ln.get("name_zh") or "").lower()
        if "评测" in (ln.get("name_zh") or "") or "benchmark" in name:
            for p in (ln.get("papers") or [])[:4]:
                benches.append(
                    {
                        "name": (p.get("title") or "")[:48],
                        "focus_zh": ln.get("desc_zh") or "",
                        "papers": [p],
                    }
                )
    gaps = report.get("gaps_zh") or []
    summary = report.get("summary_zh") or ""
    if gaps:
        joiner = "" if not summary or summary.rstrip().endswith(("。", "；", "\n")) else "。"
        summary = summary + joiner + "开放问题：" + "；".join(gaps[:2])
    return {
        "source": report.get("source") or "report",
        "scenario_id": "",
        "title_zh": report.get("title_zh") or "研究方向简报",
        "summary_zh": summary,
        "schools": schools,
        "teams": report.get("teams") or [],
        "benchmarks": benches,
        "report": report,
    }


def report_to_markdown(
    report: dict[str, Any],
    *,
    query: str = "",
    results: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Export ReportAST (+ optional paper table) as plain Markdown."""
    lines: list[str] = []
    title = (report.get("title_zh") or "研究方向简报").strip()
    lines.append(f"# {title}")
    if query:
        lines.append("")
        lines.append(f"> 检索问题：{query.strip()}")
    summary = (report.get("summary_zh") or "").strip()
    if summary:
        lines.extend(["", "## 摘要", "", summary])

    body_lines = report.get("lines") or []
    if body_lines:
        lines.extend(["", "## 研究主线", ""])
        for i, ln in enumerate(body_lines, 1):
            name = (ln.get("name_zh") or f"线索 {i}").strip()
            lines.append(f"### {i}. {name}")
            desc = (ln.get("desc_zh") or "").strip()
            if desc:
                lines.append("")
                lines.append(desc)
            hint = (ln.get("sota_hint_zh") or "").strip()
            if hint:
                lines.append("")
                lines.append(f"- SOTA 线索：{hint}")
            papers = ln.get("papers") or []
            if papers:
                lines.append("")
                for p in papers:
                    bit = (p.get("title") or "").strip() or p.get("id") or ""
                    y = p.get("year") or ""
                    v = p.get("venue") or ""
                    meta = " · ".join(x for x in (str(y), str(v)) if x)
                    lines.append(f"- {bit}" + (f" ({meta})" if meta else ""))
            lines.append("")

    gaps = [str(g).strip() for g in (report.get("gaps_zh") or []) if str(g).strip()]
    if gaps:
        lines.extend(["## 开放问题 / 缺口", ""])
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    teams = report.get("teams") or []
    if teams:
        lines.extend(["## 团队 / 机构", ""])
        for t in teams[:8]:
            n = (t.get("name") or "").strip()
            if not n:
                continue
            note = (t.get("note_zh") or "").strip()
            lines.append(f"- **{n}**" + (f"：{note}" if note else ""))
        lines.append("")

    reading = report.get("reading_order") or []
    by_id = {str(r.get("id") or ""): r for r in (results or []) if r.get("id")}
    idx = {
        str(p.get("id") or ""): p
        for p in (report.get("papers_index") or [])
        if p.get("id")
    }
    if reading:
        lines.extend(["## 建议阅读顺序", ""])
        for i, pid in enumerate(reading, 1):
            r = by_id.get(str(pid)) or idx.get(str(pid)) or {}
            t = (r.get("title") or str(pid)).strip()
            y = r.get("year") or ""
            lines.append(f"{i}. {t}" + (f" ({y})" if y else ""))
        lines.append("")

    rows = results or []
    if rows:
        lines.extend(["## 本批论文", ""])
        for i, r in enumerate(rows, 1):
            t = (r.get("title") or "").strip() or "(untitled)"
            y = r.get("year") or ""
            v = r.get("venue") or ""
            cites = r.get("cited_by_count")
            tiers = r.get("tiers") or {}
            bits = [x for x in (str(y) if y else "", str(v) if v else "") if x]
            if tiers.get("ccf"):
                bits.append(f"CCF-{tiers['ccf']}")
            if tiers.get("cas_zone") is not None:
                bits.append(f"CAS-{tiers['cas_zone']}")
            if cites is not None and cites != "":
                bits.append(f"引用 {cites}")
            link = r.get("landing_url") or r.get("oa_url") or ""
            head = f"{i}. **{t}**"
            if bits:
                head += f" — {' · '.join(bits)}"
            lines.append(head)
            if link:
                lines.append(f"   - {link}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
