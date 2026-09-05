from topper.models import PaperCard, SearchPolicy
from topper.policy import accepts, attach_tiers, score_card
from topper.tiers.registry import TierRegistry


def test_gate_keeps_ccf_a_drops_workshop():
    reg = TierRegistry()
    pol = SearchPolicy(ccf_levels=("A", "B"), cas_zones=(1, 2), require_tier=True)

    good = PaperCard(id="1", title="x", year=2023, venue="CVPR", cited_by_count=10)
    bad = PaperCard(id="2", title="y", year=2023, venue="Local Workshop", cited_by_count=10)
    attach_tiers(good, reg)
    attach_tiers(bad, reg)
    assert accepts(good, pol, now_year=2026)
    assert not accepts(bad, pol, now_year=2026)


def test_cas_gate():
    reg = TierRegistry()
    pol = SearchPolicy(ccf_levels=("A",), cas_zones=(1,), require_tier=True)
    card = PaperCard(id="1", title="x", year=2022, venue="Nature", cited_by_count=100)
    attach_tiers(card, reg)
    assert card.tiers.cas_zone == 1
    assert accepts(card, pol, now_year=2026)


def test_min_citations_and_age():
    reg = TierRegistry()
    pol = SearchPolicy(min_citations=100, max_age_years=5, require_tier=True)
    card = PaperCard(id="1", title="x", year=2010, venue="NeurIPS", cited_by_count=5)
    attach_tiers(card, reg)
    assert not accepts(card, pol, now_year=2026)


def test_wos_gate_ssci():
    reg = TierRegistry()
    pol = SearchPolicy(wos=("ssci",), require_tier=True)
    card = PaperCard(
        id="1",
        title="x",
        year=2023,
        venue="American Economic Review",
        cited_by_count=10,
    )
    attach_tiers(card, reg)
    assert card.tiers.ssci is True
    assert accepts(card, pol, now_year=2026)


def test_wos_gate_quartile():
    reg = TierRegistry()
    pol = SearchPolicy(wos=("sci",), jcr_quartiles=("Q1",), require_tier=True)
    card = PaperCard(id="1", title="x", year=2023, venue="Nature", cited_by_count=10)
    attach_tiers(card, reg)
    assert accepts(card, pol, now_year=2026)


def test_authority_boost_in_score():
    reg = TierRegistry()
    pol = SearchPolicy(require_tier=True)
    card = PaperCard(
        id="1",
        title="x",
        year=2020,
        venue="NeurIPS",
        cited_by_count=10,
        authors=["Kaiming He"],
        institutions=["Microsoft Research"],
    )
    attach_tiers(card, reg)
    score_card(card, pol, now_year=2026)
    assert any(a.startswith("author:") for a in card.matched_authority)
    assert any(a.startswith("org:") for a in card.matched_authority)
    assert card.score_parts["authority"] > 0


def test_field_prior_prefers_education_over_medicine():
    from topper.models import TierLabels
    from topper.policy import apply_field_prior, infer_field_priors

    prefer, damp, prefer_fms = infer_field_priors(
        ["教育干预的因果效应与选择偏误", "educational intervention selection bias"]
    )
    assert "教育学" in prefer
    assert "医学" in damp
    assert "教育管理" in prefer_fms

    edu = PaperCard(
        id="e",
        title="Schooling returns",
        year=2020,
        venue="Economics of Education Review",
        cited_by_count=50,
        tiers=TierLabels(
            cas_zone=3,
            cas_major="教育学",
            fms_tier="B",
            fms_discipline="教育管理",
            ssci=True,
        ),
        score_parts={"citations": 1.0, "tier": 0.5},
        score=1.5,
    )
    med = PaperCard(
        id="m",
        title="School-based health",
        year=2020,
        venue="The Lancet",
        cited_by_count=5000,
        tiers=TierLabels(cas_zone=1, cas_major="医学", sci=True),
        score_parts={"citations": 4.0, "tier": 1.2},
        score=5.2,
    )
    apply_field_prior(
        edu, prefer_majors=prefer, damp_majors=damp, prefer_fms=prefer_fms
    )
    apply_field_prior(
        med, prefer_majors=prefer, damp_majors=damp, prefer_fms=prefer_fms
    )
    assert edu.score > med.score
    assert (edu.score_parts or {}).get("field", 0) > 0
    assert (med.score_parts or {}).get("field", 0) < 0


def test_field_prior_hard_drop_in_pipeline():
    """Damped CAS majors must not enter results when intent priors are active."""
    from topper.pipeline import search
    from topper.sources.mock import MockSource

    class _BioBleed(MockSource):
        def retrieve(self, query: str, **kwargs):  # noqa: ARG002
            return [
                PaperCard(
                    id="bio",
                    title="Single-cell eQTL mapping identifies cell type genetic control",
                    year=2022,
                    venue="Nature Genetics",
                    cited_by_count=800,
                    abstract="single-cell eQTL autoimmune genetics transcriptome",
                ),
                PaperCard(
                    id="did",
                    title="Two-way fixed effects and difference-in-differences estimators",
                    year=2021,
                    venue="American Economic Review",
                    cited_by_count=200,
                    abstract="difference-in-differences staggered treatment twfe bias",
                ),
            ]

    rows = search(
        "双重差分 多期处理 异质性 偏误",
        source=_BioBleed(),
        limit=10,
        primary_queries=["difference-in-differences staggered treatment"],
        relevance_queries=["twfe event study"],
    )
    ids = [r.id for r in rows]
    assert "did" in ids
    assert "bio" not in ids


def test_fms_gate_and_score():
    reg = TierRegistry()
    # FMS-only path: drop CCF/CAS/WoS so FMS must carry the gate.
    pol = SearchPolicy(
        ccf_levels=(),
        cas_zones=(),
        wos=(),
        fms_levels=("A", "B", "T1"),
        require_tier=True,
    )
    card = PaperCard(
        id="1",
        title="x",
        year=2020,
        venue="Economics of Education Review",
        cited_by_count=40,
    )
    attach_tiers(card, reg)
    assert card.tiers.fms_tier == "B"
    assert card.tiers.fms_discipline == "教育管理"
    assert accepts(card, pol, now_year=2026)
    score_card(card, pol, now_year=2026)
    assert card.score_parts["tier"] > 0

    pol_strict = SearchPolicy(
        ccf_levels=(),
        cas_zones=(),
        wos=(),
        fms_levels=("A", "T1"),
        require_tier=True,
    )
    assert not accepts(card, pol_strict, now_year=2026)


def test_citation_log_is_capped():
    reg = TierRegistry()
    pol = SearchPolicy(require_tier=True, w_citations=1.0, citation_log_cap=6.2)
    low = PaperCard(id="1", title="a", year=2023, venue="NeurIPS", cited_by_count=500)
    high = PaperCard(id="2", title="b", year=2023, venue="NeurIPS", cited_by_count=500_000)
    attach_tiers(low, reg)
    attach_tiers(high, reg)
    score_card(low, pol, now_year=2026)
    score_card(high, pol, now_year=2026)
    assert abs(low.score_parts["citations"] - high.score_parts["citations"]) < 1e-6
