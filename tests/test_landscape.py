from topper.landscape import build_landscape
from topper.scenarios import clear_scenario_cache


def test_curated_landscape_when_a_pack_matches():
    """A scenario pack takes over the landscape; no pack ships, so use a fixture."""
    clear_scenario_cache()
    results = [
        {"title": "Alpha Method For Fixture Topic", "year": 2024},
        {"title": "Beta Framework Revisited", "year": 2025, "venue": "NeurIPS"},
        {"title": "Gamma Approach To Everything", "year": 2025},
        {"title": "FixtureBench: Evaluating Fixtures", "year": 2023},
        {"title": "ProbeSuite For Fixture Topic", "year": 2024},
    ]
    ls = build_landscape("fixture curated topic", results)
    assert ls is not None
    assert ls["source"] == "curated"
    assert len(ls["schools"]) >= 3
    assert any(s["papers"] for s in ls["schools"])
    names = " ".join(b["name"] for b in ls["benchmarks"])
    assert "FixtureBench" in names or "ProbeSuite" in names


def test_heuristic_landscape_generic_query(monkeypatch):
    # Force non-LLM report path for unit stability.
    monkeypatch.setenv("TOPPER_REPORT_LLM", "0")
    results = [
        {
            "id": "p1",
            "title": "A Survey of Graph Neural Networks for Molecular Property Prediction",
            "year": 2024,
            "cited_by_count": 100,
            "institutions": ["MIT", "Stanford"],
        },
        {
            "id": "p2",
            "title": "Uncertainty-Aware GNN Benchmark for Drug Discovery",
            "year": 2025,
            "cited_by_count": 20,
            "institutions": ["MIT"],
        },
        {
            "id": "p3",
            "title": "Efficient Graph Transformer Architecture for Molecules",
            "year": 2023,
            "cited_by_count": 50,
            "institutions": ["Stanford"],
        },
    ]
    ls = build_landscape("分子图神经网络 不确定性", results)
    assert ls is not None
    assert ls["source"] in {"heuristic", "heuristic_report", "llm_report", "report"}
    assert ls["schools"]
    assert any(t["name"] in {"MIT", "Stanford"} for t in ls["teams"])
    # cite-grounded path keeps paper ids on lines when report AST is present
    if ls.get("report"):
        assert ls["report"]["lines"]
