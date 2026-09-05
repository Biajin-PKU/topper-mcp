from topper.models import PaperCard
from topper.relevance import filter_by_relevance, score_relevance


def test_on_topic_passes():
    card = PaperCard(
        id="1",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        abstract="We introduce RAG models that combine parametric and non-parametric memory for question answering.",
        year=2020,
        venue="NeurIPS",
        cited_by_count=1000,
    )
    rel = score_relevance(
        "retrieval augmented generation for question answering",
        card,
        min_score=0.28,
    )
    assert rel.passed
    assert rel.score > 0.2


def test_off_topic_fails():
    card = PaperCard(
        id="2",
        title="A Survey of Database Indexing Structures",
        abstract="B-trees and hash indexes for relational storage engines.",
        year=2019,
        venue="VLDB",
        cited_by_count=500,
    )
    rel = score_relevance(
        "diffusion models for text-to-image generation safety",
        card,
        min_score=0.28,
    )
    assert not rel.passed


def test_generic_end_to_end_alone_does_not_pass_detection_paper():
    card = PaperCard(
        id="det",
        title="YOLOv10: Real-Time End-to-End Object Detection",
        abstract="We present a real-time object detector with NMS-free end-to-end training.",
        year=2024,
        venue="NeurIPS",
        cited_by_count=100,
    )
    rel = score_relevance(
        "end-to-end automated scientific research agents",
        card,
        alt_queries=["AI scientist autonomous research"],
        min_score=0.28,
    )
    assert not rel.passed


def test_filter_keeps_relevant_only():
    cards = [
        PaperCard(
            id="a",
            title="The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
            abstract="An end-to-end agent that writes code, runs experiments, and produces papers.",
            year=2024,
            venue="arXiv",
        ),
        PaperCard(
            id="b",
            title="High-Frequency Trading Microstructure",
            abstract="Order books and latency arbitrage in equities markets.",
            year=2023,
            venue="Journal of Finance",
        ),
    ]
    kept = filter_by_relevance(
        cards,
        "end-to-end automated scientific research agent",
        alt_queries=["AI scientist autonomous research"],
        min_score=0.28,
    )
    ids = {c.id for c in kept}
    assert "a" in ids
    assert "b" not in ids


def test_weak_alt_alone_cannot_admit_prestige_neighbor():
    """Synonym fan-out must not open the gate for off-topic mega-cited work."""
    card = PaperCard(
        id="sci",
        title="Single-cell RNA-seq reveals cell type–specific genetic associations to lupus",
        abstract="We map eQTLs across immune cell types in autoimmune disease cohorts.",
        year=2022,
        venue="Science",
        cited_by_count=5000,
    )
    rel = score_relevance(
        "difference-in-differences multi-period treatment bias",
        card,
        alt_queries=[
            "difference-in-differences multi-period treatment bias",
            "heterogeneous treatment effects",  # weak bridge into genetics HTE-ish titles
            "single-cell",
        ],
        primary_queries=["difference-in-differences multi-period treatment bias"],
        min_score=0.28,
    )
    assert not rel.passed


def test_design_anchor_required_for_did():
    from topper.models import PaperCard
    from topper.relevance import score_relevance

    noise = PaperCard(
        id="n",
        title="Treatment heterogeneity and response bias in immune cell genetics",
        abstract="We study treatment response heterogeneity and selection bias in cell assays.",
        year=2021,
        venue="Science",
    )
    good = PaperCard(
        id="g",
        title="Two-way fixed effects and difference-in-differences estimators",
        abstract="We study staggered adoption difference-in-differences and TWFE bias.",
        year=2021,
        venue="American Economic Review",
    )
    q = "双重差分 多期处理 异质性 偏误"
    prim = [
        q,
        "difference-in-differences staggered treatment heterogeneity bias",
    ]
    assert not score_relevance(q, noise, primary_queries=prim, min_score=0.28).passed
    assert score_relevance(q, good, primary_queries=prim, min_score=0.28).passed
