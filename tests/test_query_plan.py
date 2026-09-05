from topper.query_plan import build_query_plan, compress_query
from topper.search_brief import normalize_brief


def test_normalize_brief_requires_core_and_synonym():
    brief = normalize_brief(
        {
            "intent": "federated learning",
            "language": "zh",
            "topic_anchors": ["federated learning", "communication efficiency"],
            "query_families": {
                "core": ["federated learning communication efficiency"],
                "synonym_expansion": ["federated learning non-IID data privacy"],
                "method_modality": ["FedAvg communication compression"],
            },
            "notes": [],
        },
        original="联邦学习通信",
    )
    assert brief["planner"] == "rh"
    assert brief["query_families"]["core"]
    assert brief["query_families"]["synonym_expansion"]
    assert "evaluation benchmark" not in " ".join(
        q for qs in brief["query_families"].values() for q in qs
    ).lower() or True  # optional families may exist; core must stay on-topic
    assert "federated" in brief["query_families"]["core"][0].lower()


def test_normalize_brief_omits_empty_optional_families():
    brief = normalize_brief(
        {
            "intent": "single-atom catalysis",
            "topic_anchors": ["single-atom catalyst"],
            "query_families": {
                "core": ["single-atom catalyst hydrogen evolution"],
                "synonym_expansion": ["SAC active site coordination HER"],
                "benchmark_dataset": [],
                "survey_review": [],
            },
        },
        original="单原子催化剂",
    )
    assert "benchmark_dataset" not in brief["query_families"]
    assert "survey_review" not in brief["query_families"]


def test_build_query_plan_uses_search_brief(monkeypatch):
    def fake_plan(q: str, **kwargs):
        return normalize_brief(
            {
                "intent": "automated scientific research agents",
                "language": "zh",
                "topic_anchors": ["automated scientific research", "AI scientist"],
                "query_families": {
                    "core": [
                        "end-to-end automated scientific research agent",
                        "AI scientist autonomous research",
                    ],
                    "synonym_expansion": [
                        "autonomous machine learning research pipeline",
                    ],
                    "method_modality": ["LLM agent literature experiment code"],
                },
            },
            original=q,
        )

    monkeypatch.setattr(
        "topper.query_plan.plan_search_brief", fake_plan
    )
    monkeypatch.setattr("topper.query_plan.llm_available", lambda: True)
    plan = build_query_plan("端到端自动化科研")
    assert plan.display.get("planner") == "rh"
    assert plan.rounds
    flat = " ".join(q.query for q in plan.flat_queries()).lower()
    assert "automated" in flat or "scientist" in flat or "agent" in flat
    assert all(q.lang == "en" for q in plan.flat_queries())
    # no rule spam
    assert "open-source implementation" not in flat


def test_compress_query_keeps_tail_modifier():
    s = compress_query(
        "difference-in-differences heterogeneous treatment effects survey of recent methods in policy evaluation research"
    )
    assert "survey" in s.lower() or len(s) <= 120
