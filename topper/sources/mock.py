"""Deterministic offline source for tests and demos."""

from __future__ import annotations

from topper.models import PaperCard
from topper.normalize import fold


_CORPUS: list[PaperCard] = [
    PaperCard(
        id="mock:1",
        title="Attention Is All You Need",
        year=2017,
        venue="NeurIPS",
        cited_by_count=120000,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        institutions=["Google Brain"],
        source="mock",
    ),
    PaperCard(
        id="mock:2",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        year=2019,
        venue="NAACL",
        cited_by_count=80000,
        authors=["Jacob Devlin"],
        institutions=["Google"],
        source="mock",
    ),
    PaperCard(
        id="mock:3",
        title="Deep Residual Learning for Image Recognition",
        year=2016,
        venue="CVPR",
        cited_by_count=150000,
        authors=["Kaiming He"],
        institutions=["Microsoft Research"],
        source="mock",
    ),
    PaperCard(
        id="mock:4",
        title="A Random Mid-tier Workshop Paper",
        year=2024,
        venue="Unknown Local Workshop",
        cited_by_count=2,
        authors=["A. Author"],
        institutions=["Somewhere College"],
        source="mock",
    ),
    PaperCard(
        id="mock:5",
        title="Nature Machine Intelligence survey on foundation models",
        year=2023,
        venue="Nature Machine Intelligence",
        cited_by_count=900,
        authors=["Demo Author"],
        institutions=["Stanford University"],
        source="mock",
    ),
    PaperCard(
        id="mock:6",
        title="Scaling Laws for Neural Language Models",
        year=2020,
        venue="arXiv",
        cited_by_count=5000,
        authors=["Someone"],
        institutions=["OpenAI"],
        source="mock",
    ),
]


class MockSource:
    name = "mock"

    def retrieve(self, query: str, *, limit: int = 25) -> list[PaperCard]:
        q = fold(query)
        hits = []
        for c in _CORPUS:
            blob = fold(
                " ".join(
                    [
                        c.title,
                        c.venue or "",
                        " ".join(c.authors),
                        " ".join(c.institutions),
                    ]
                )
            )
            if not q or any(tok in blob for tok in q.split()):
                # copy so callers can mutate tiers/score safely
                hits.append(
                    PaperCard(
                        id=c.id,
                        title=c.title,
                        year=c.year,
                        venue=c.venue,
                        cited_by_count=c.cited_by_count,
                        authors=list(c.authors),
                        institutions=list(c.institutions),
                        source=self.name,
                    )
                )
        return hits[:limit]
