from topper.arxiv_bridge import (
    author_keyset,
    authors_overlap,
    is_arxiv_like,
    try_admit_arxiv,
)
from topper.models import PaperCard, SearchPolicy, TierLabels


def test_is_arxiv_like():
    assert is_arxiv_like(
        PaperCard(id="1", title="x", venue="arXiv.org", oa_url="https://arxiv.org/pdf/1")
    )
    assert not is_arxiv_like(PaperCard(id="2", title="x", venue="NeurIPS"))


def test_authors_overlap_full_name():
    hits = authors_overlap(
        ["Alice Smith", "Bob Jones"],
        ["A. Smith", "Carol Lee"],
    )
    assert hits  # initial+last or smith


def test_admit_requires_topic_and_bridge():
    pol = SearchPolicy(require_relevance=True, min_relevance=0.18)
    pool = author_keyset(["Alice Wonderland"])

    on_topic = PaperCard(
        id="arx1",
        title="The AI Scientist: Fully Automated Scientific Discovery",
        abstract="An autonomous research agent that runs experiments and writes papers.",
        year=2024,
        venue="arXiv",
        authors=["Alice Wonderland"],
        oa_url="https://arxiv.org/pdf/2408.00000",
        tiers=TierLabels(),
    )
    ok, card, reason = try_admit_arxiv(
        on_topic,
        query="end-to-end automated scientific research agents",
        alt_queries=["AI scientist autonomous research"],
        policy=pol,
        ccf_a_authors=pool,
    )
    assert ok, reason
    assert any("arxiv_bridge" in x for x in card.matched_authority)

    off_topic = PaperCard(
        id="arx2",
        title="A Study of Baking Sourdough at Home",
        abstract="Recipes and fermentation temperature charts for bread.",
        year=2024,
        venue="arXiv",
        authors=["Alice Wonderland"],
        oa_url="https://arxiv.org/pdf/2408.00001",
    )
    ok2, _, reason2 = try_admit_arxiv(
        off_topic,
        query="end-to-end automated scientific research agents",
        alt_queries=["AI scientist"],
        policy=pol,
        ccf_a_authors=pool,
    )
    assert not ok2
    assert reason2 == "off_topic"


def test_no_bridge_without_author_overlap():
    pol = SearchPolicy()
    card = PaperCard(
        id="arx3",
        title="Autonomous Scientific Discovery with LLM Agents",
        abstract="We present an agent for automated scientific research and experiments.",
        year=2025,
        venue="arXiv.org",
        authors=["Nobody Special"],
        oa_url="https://arxiv.org/pdf/2501.00001",
    )
    ok, _, reason = try_admit_arxiv(
        card,
        query="automated scientific research agent",
        alt_queries=[],
        policy=pol,
        ccf_a_authors=author_keyset(["Alice Wonderland"]),
    )
    assert not ok
    assert reason == "no_author_bridge"
