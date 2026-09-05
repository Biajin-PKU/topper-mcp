from __future__ import annotations

from typing import Protocol

from topper.models import PaperCard


class Source(Protocol):
    name: str

    def retrieve(self, query: str, *, limit: int = 25) -> list[PaperCard]:
        ...
