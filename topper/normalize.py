"""Venue string normalization for tier lookup."""

from __future__ import annotations

import re
import unicodedata


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_VOL_ISSUE = re.compile(r"\b(vol|volume|no|number|pp|pages?)\b\.?")

# Tokens that often appear in primary-location display names but hurt matching.
_STRIP_TOKENS = {
    "proceedings",
    "proc",
    "of",
    "the",
    "ieee",
    "acm",
    "international",
    "conference",
    "symposium",
    "workshop",
    "journal",
    "transactions",
    "on",
    "and",
    "annual",
    "meeting",
}


def fold(text: str) -> str:
    """Lowercase ASCII fold."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def normalize_venue(text: str) -> str:
    """Aggressive key for matching aliases."""
    s = fold(text)
    s = _YEAR.sub(" ", s)
    s = _VOL_ISSUE.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    parts = [p for p in s.split() if p and p not in _STRIP_TOKENS]
    return " ".join(parts)


def compact(text: str) -> str:
    return normalize_venue(text).replace(" ", "")
