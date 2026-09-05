from topper.models import SearchPolicy
from topper.pipeline import search


def test_mock_search_top_only():
    hits = search(
        "transformer residual",
        source="mock",
        policy=SearchPolicy(require_tier=True, max_age_years=None, min_citations=0),
        limit=10,
        now_year=2026,
    )
    assert hits
    venues = {h.venue for h in hits}
    assert "Unknown Local Workshop" not in venues
    # arXiv alone is not in CCF/CAS seed → should be dropped when require_tier
    assert "arXiv" not in venues


def test_mock_high_cite_filter():
    hits = search(
        "attention",
        source="mock",
        policy=SearchPolicy(require_tier=True, min_citations=10000, max_age_years=None),
        limit=10,
        now_year=2026,
    )
    assert hits
    assert all(h.cited_by_count >= 10000 for h in hits)
