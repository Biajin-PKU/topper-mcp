"""Command line entry: `topper doctor | tiers | plan | search`.

`doctor` checks configuration and installed tables without making an API call.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from topper import config


def cmd_doctor(_: argparse.Namespace) -> int:
    from topper.tiers.registry import get_default_registry

    config.load_dotenv()
    print("TOPPER doctor\n")

    print("LLM")
    print(f"  base_url : {config.llm_base_url()}")
    print(f"  model    : {config.llm_model() or '(unset)'}")
    fallbacks = config.llm_model_chain()[1:]
    print(f"  fallbacks: {', '.join(fallbacks) if fallbacks else '(none)'}")
    print(f"  api_key  : {'set' if config.llm_api_key() else 'MISSING'}")

    print("\nSources")
    keys = config.s2_api_keys()
    print(f"  semantic scholar keys: {len(keys)}" + ("" if keys else "  (throttled)"))
    print(f"  openalex mailto      : {config.openalex_mailto() or '(unset)'}")

    print("\nTier tables")
    stats = get_default_registry().stats()
    for k in ("ccf_venues", "cas_journals", "fms_journals", "jcr_quartiled_journals"):
        n = stats.get(k, 0)
        mark = "" if n else "   <- not installed, see topper/data/README.md"
        print(f"  {k:24} {n}{mark}")
    print(f"  data_dir                 {stats.get('data_dir')}")

    gaps = config.missing_required()
    if gaps:
        print("\nStill to configure:")
        for g in gaps:
            print(f"  - {g}")
        return 1
    print("\nReady.")
    return 0


def cmd_tiers(args: argparse.Namespace) -> int:
    from topper.tiers.registry import get_default_registry

    labels = get_default_registry().lookup(args.venue).to_dict()
    print(json.dumps({"venue": args.venue, "tiers": labels}, ensure_ascii=False, indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    config.load_dotenv()
    from topper.mcp_server import preview_search_plan

    print(json.dumps(preview_search_plan(args.query), ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from topper.mcp_server import search_top_papers

    config.load_dotenv()
    payload: dict[str, Any] = search_top_papers(
        query=args.query,
        limit=args.limit,
        years=args.years,
        ccf_levels=[c.strip().upper() for c in args.ccf.split(",") if c.strip()],
        cas_zones=[int(z) for z in args.cas.split(",") if z.strip()],
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"{payload['count']} papers for: {payload['query']}\n")
    for i, p in enumerate(payload["papers"], 1):
        tier = f"[{p['tier']}]" if p["tier"] else "[-]"
        print(f"{i:2}. {p['year']} {tier} {p['title']}")
        print(f"    {p['venue'] or '(venue not reported)'}  ·  cited {p['citations']}")
        if p.get("url"):
            print(f"    {p['url']}")
    land = payload.get("landscape") or {}
    if land.get("summary"):
        print(f"\n{land['summary']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="topper", description="Top-only paper retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check configuration and installed tier tables")

    p = sub.add_parser("tiers", help="look up one venue's tier labels")
    p.add_argument("venue")

    p = sub.add_parser("plan", help="show the query plan without searching")
    p.add_argument("query")

    p = sub.add_parser("search", help="run a top-only search")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--ccf", default="A,B")
    p.add_argument("--cas", default="1,2")
    p.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    return {
        "doctor": cmd_doctor,
        "tiers": cmd_tiers,
        "plan": cmd_plan,
        "search": cmd_search,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
