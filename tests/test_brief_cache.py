import json
from pathlib import Path

from topper import search_brief


def test_brief_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPPER_BRIEF_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TOPPER_BRIEF_CACHE", "1")
    monkeypatch.delenv("TOPPER_BRIEF_CACHE_TTL_S", raising=False)

    calls = {"n": 0}

    def fake_chat(messages, timeout=45.0):  # noqa: ARG001
        calls["n"] += 1
        return {
            "intent": "federated learning",
            "language": "en",
            "topic_anchors": ["federated learning"],
            "query_families": {
                "core": ["federated learning non-iid"],
                "synonym_expansion": ["federated optimization heterogeneous"],
            },
        }

    monkeypatch.setattr(search_brief, "_chat_json", fake_chat)
    monkeypatch.setattr(search_brief, "_load_openai_env", lambda: ("http://x", "k", "gpt-test"))

    b1 = search_brief.plan_search_brief("Federated Learning")
    b2 = search_brief.plan_search_brief("  federated   learning ")
    assert calls["n"] == 1
    assert b1["query_families"]["core"]
    assert "cache:brief_hit" in (b2.get("notes") or [])
    files = list(Path(tmp_path).glob("*.json"))
    assert len(files) == 1
    raw = json.loads(files[0].read_text(encoding="utf-8"))
    assert raw["intent"]


def test_brief_cache_can_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPPER_BRIEF_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TOPPER_BRIEF_CACHE", "0")
    calls = {"n": 0}

    def fake_chat(messages, timeout=45.0):  # noqa: ARG001
        calls["n"] += 1
        return {
            "intent": "x",
            "topic_anchors": ["x"],
            "query_families": {"core": ["x y z"], "synonym_expansion": ["x y"]},
        }

    monkeypatch.setattr(search_brief, "_chat_json", fake_chat)
    monkeypatch.setattr(search_brief, "_load_openai_env", lambda: ("http://x", "k", "gpt-test"))
    search_brief.plan_search_brief("hello world topic")
    search_brief.plan_search_brief("hello world topic")
    assert calls["n"] == 2
