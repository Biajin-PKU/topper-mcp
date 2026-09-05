from topper.report import (
    build_report,
    report_to_landscape,
    report_to_markdown,
)


def test_heuristic_report_binds_paper_ids(monkeypatch):
    monkeypatch.setenv("TOPPER_REPORT_LLM", "0")
    results = [
        {
            "id": "a",
            "title": "A Survey of Federated Learning Systems",
            "year": 2024,
            "venue": "IEEE TKDE",
            "cited_by_count": 100,
            "institutions": ["MIT"],
        },
        {
            "id": "b",
            "title": "FedAvg: Communication-Efficient Learning of Deep Networks",
            "year": 2017,
            "venue": "AISTATS",
            "cited_by_count": 20000,
            "institutions": ["Google"],
        },
    ]
    rep = build_report("federated learning", results)
    assert rep is not None
    assert rep["source"] == "heuristic_report"
    assert rep["lines"]
    assert all(ln.get("paper_ids") for ln in rep["lines"])
    ls = report_to_landscape(rep)
    assert ls["schools"]
    assert "report" in ls
    md = report_to_markdown(rep, query="federated learning", results=results)
    assert md.startswith("# ")
    assert "FedAvg" in md or "Federated" in md
    assert "研究主线" in md


def test_report_empty_results(monkeypatch):
    monkeypatch.setenv("TOPPER_REPORT_LLM", "0")
    assert build_report("x", []) is None
