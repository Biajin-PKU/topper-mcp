"""MCP server: expose TOPPER's retrieval to any MCP client.

Run it:

    topper-mcp                      # stdio, the transport MCP clients expect

Claude Desktop / Claude Code config:

    {
      "mcpServers": {
        "topper": {
          "command": "topper-mcp",
          "env": {
            "TOPPER_LLM_API_KEY": "sk-...",
            "TOPPER_LLM_MODEL": "gpt-4o-mini",
            "SEMANTIC_SCHOLAR_API_KEY": "..."
          }
        }
      }
    }

The tool functions below are the single source of truth: the MCP schema is
derived from their signatures, and the CLI calls the same functions.
"""

from __future__ import annotations

from typing import Any, Optional

from topper import config
from topper.models import SearchPolicy


def _tier_label(t: dict[str, Any]) -> str:
    """The tier labels a paper carries, if any table covered its venue."""
    bits = []
    if t.get("ccf"):
        bits.append(f"CCF-{t['ccf']}")
    if t.get("cas_zone"):
        bits.append(f"CAS-{t['cas_zone']}" + ("(Top)" if t.get("cas_top") else ""))
    if t.get("jcr_quartile"):
        bits.append(f"JCR-{t['jcr_quartile']}")
    if t.get("fms_tier"):
        bits.append(f"FMS-{t['fms_tier']}")
    return " · ".join(bits)


def admission_label(paper: dict[str, Any]) -> str:
    """Why this paper cleared the gate.

    Flagship-list venues and author-bridged preprints qualify without a tier
    row, so they get a label of their own.
    """
    label = _tier_label(paper.get("tiers") or {})
    if label:
        return label
    from topper.normalize import fold
    from topper.policy import _flagship_keys

    for cand in (paper.get("venue_key"), paper.get("venue")):
        if cand and fold(cand) in _flagship_keys():
            return "flagship"
    if any(str(x).startswith("arxiv_bridge") for x in (paper.get("matched_authority") or [])):
        return "preprint by a CCF-A author"
    return ""


def _require_config() -> Optional[dict[str, Any]]:
    """Config problems become tool output, not a dead server."""
    config.load_dotenv()
    gaps = config.missing_required()
    hard = [g for g in gaps if g.startswith(("TOPPER_LLM_API_KEY", "TOPPER_LLM_MODEL"))]
    if hard:
        return {"error": "not configured", "fix": hard}
    return None


def search_top_papers(
    query: str,
    limit: int = 20,
    years: int = 8,
    ccf_levels: Optional[list[str]] = None,
    cas_zones: Optional[list[int]] = None,
    min_citations: int = 0,
) -> dict[str, Any]:
    """Search ONLY top-tier academic venues (CCF A/B, CAS zone 1/2, flagship journals).

    Use this instead of a web search when the user wants the literature a field
    actually treats as authoritative — a reading list, related work, the state
    of the art, or "what are the key papers on X". Returns metadata and links,
    never paywalled full text.

    Each paper comes back with the reason it qualified, so you can tell the user
    *why* it counts as top-tier. Also returns a landscape: schools of thought
    and active teams.

    Costs one LLM call to plan the search plus several source round-trips, so a
    query typically takes 30-120s. Ask one well-formed research question rather
    than many keyword probes.

    Args:
        query: One research question, Chinese or English. A full question beats
            keywords — the planner uses it to derive the field's own terminology.
        limit: Maximum papers to return (max 50).
        years: Only papers from the last N years.
        ccf_levels: CCF tiers to accept, e.g. ["A", "B"].
        cas_zones: CAS zones to accept, e.g. [1, 2].
        min_citations: Drop papers below this citation count.
    """
    from topper.multi_search import multi_search

    bad = _require_config()
    if bad:
        return bad
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(limit or 20), 50))

    policy = SearchPolicy(
        ccf_levels=tuple(str(c).upper() for c in (ccf_levels or ["A", "B"])),
        cas_zones=tuple(int(z) for z in (cas_zones or [1, 2])),
        max_age_years=int(years or 8),
        min_citations=int(min_citations or 0),
    )
    out = multi_search(
        query,
        policy=policy,
        max_rounds=3,
        max_per_round=3,
        limit_per_query=10,
        stop_when_total=limit,
    )
    papers = [
        {
            "title": r.get("title"),
            "year": r.get("year"),
            "venue": r.get("venue"),
            "tier": admission_label(r),
            "citations": r.get("cited_by_count"),
            "authors": (r.get("authors") or [])[:8],
            "url": r.get("landing_url") or r.get("oa_url"),
            "doi": r.get("doi"),
        }
        for r in (out.get("results") or [])[:limit]
    ]
    payload: dict[str, Any] = {"query": query, "count": len(papers), "papers": papers}
    land = out.get("landscape") or {}
    if land.get("summary_zh") or land.get("schools"):
        payload["landscape"] = {
            "summary": land.get("summary_zh"),
            "schools": [s.get("name_zh") for s in (land.get("schools") or [])],
            "teams": [t.get("name") for t in (land.get("teams") or [])],
        }
    return payload


def explain_venue(venue: str) -> dict[str, Any]:
    """Look up the tier labels (CCF / CAS zone / JCR / WoS index) for one venue.

    Use it to justify why a paper counts as top-tier, or to check coverage
    before trusting a filter — an "unknown" answer may mean the venue is not
    top-tier, or simply that the table covering it is not installed.

    Args:
        venue: Journal or conference name, e.g. "Journal of Econometrics".
    """
    from topper.tiers.registry import get_default_registry

    venue = (venue or "").strip()
    if not venue:
        raise ValueError("venue is required")
    tiers = get_default_registry().lookup(venue).to_dict()
    known = any(
        tiers.get(k) is not None
        for k in ("ccf", "cas_zone", "fms_tier", "jcr_quartile", "sci", "ssci", "ahci")
    )
    label = admission_label({"venue": venue, "tiers": tiers})
    out: dict[str, Any] = {
        "venue": venue,
        "known": known or bool(label),
        "tiers": tiers,
        "label": label,
    }
    if not out["known"]:
        out["note"] = (
            "No tier label found. Either the venue is genuinely not top-tier, or "
            "the table that covers it is not installed — see topper/data/README.md."
        )
    return out


def preview_search_plan(query: str) -> dict[str, Any]:
    """Show the query families and topic anchors the planner would search with.

    Cheap (one LLM call, no source round-trips). Use it when a search returned
    too little and you want to see whether the terminology was right — most
    recall problems are a planner that missed the field's own vocabulary.

    Args:
        query: The same research question you would pass to search_top_papers.
    """
    from topper.query_plan import build_query_plan

    bad = _require_config()
    if bad:
        return bad
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    return build_query_plan(query, max_per_round=3, max_rounds=3).to_dict()


TOOL_FUNCS = [search_top_papers, explain_venue, preview_search_plan]


def main() -> None:
    config.load_dotenv()
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:  # pragma: no cover - install hint is the whole point
        raise SystemExit(
            "The MCP server needs the MCP SDK:  pip install 'topper-mcp[mcp]'"
        )

    server = MCPServer(
        "topper",
        instructions=(
            "Top-only academic paper retrieval. The venue tier gate runs before "
            "ranking, so results are already filtered to CCF A/B, CAS zone 1/2 and "
            "flagship journals. Searches take 30-120s; prefer one well-formed "
            "research question over several keyword probes."
        ),
        version="0.1.0",
    )
    for fn in TOOL_FUNCS:
        server.add_tool(fn)
    server.run("stdio")


if __name__ == "__main__":
    main()
