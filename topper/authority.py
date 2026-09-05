"""Flagship author / org matching (soft boosts)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from topper.normalize import fold

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load(name: str) -> dict[str, Any]:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_flagship_authors() -> list[dict[str, Any]]:
    return list((_load("flagship_authors.json").get("authors") or []))


@lru_cache(maxsize=1)
def load_flagship_orgs() -> list[dict[str, Any]]:
    return list((_load("flagship_orgs.json").get("orgs") or []))


def match_authors(
    paper_authors: list[str],
    extra_names: tuple[str, ...] = (),
) -> list[str]:
    """Return matched authority keys/names."""
    hay = [fold(a) for a in paper_authors if a]
    hits: list[str] = []
    for row in load_flagship_authors():
        names = [fold(n) for n in (row.get("names") or [])]
        if any(n and any(n in h or h in n for h in hay) for n in names):
            hits.append(str(row.get("key") or names[0]))
    for name in extra_names:
        n = fold(name)
        if n and any(n in h or h in n for h in hay):
            hits.append(name)
    # stable unique
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def match_orgs(
    institutions: list[str],
    extra_names: tuple[str, ...] = (),
) -> list[str]:
    hay = [fold(i) for i in institutions if i]
    hits: list[str] = []
    for row in load_flagship_orgs():
        names = [fold(n) for n in (row.get("names") or [])]
        if any(n and any(n in h or h in n for h in hay) for n in names):
            hits.append(str(row.get("key") or names[0]))
    for name in extra_names:
        n = fold(name)
        if n and any(n in h or h in n for h in hay):
            hits.append(name)
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out
