"""Small text helpers shared across the engine."""

from __future__ import annotations

import re

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def has_cjk(text: str) -> bool:
    """True when the string contains CJK characters — used to decide whether a
    query still needs to be turned into the field's English terminology."""
    return bool(_CJK.search(text or ""))
