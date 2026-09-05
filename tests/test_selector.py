from topper.models import PaperCard
from topper.selector import select_cards, selector_enabled


def test_selector_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("TOPPER_SELECTOR_LLM", "0")
    assert not selector_enabled()
    cards = [
        PaperCard(id="1", title="Federated Learning", abstract="non-iid clients"),
        PaperCard(id="2", title="Unrelated Astronomy", abstract="galaxy survey"),
    ]
    out = select_cards("federated learning non-iid", cards)
    assert out is cards or [c.id for c in out] == ["1", "2"]


def test_selector_mock_keep(monkeypatch):
    monkeypatch.setenv("TOPPER_SELECTOR_LLM", "1")
    from topper import selector as sel

    monkeypatch.setattr(sel, "llm_available", lambda: True)

    def fake_chat(messages, timeout=60.0):  # noqa: ARG001
        return {"keep": ["1"], "drop": ["2"], "notes": "ok"}

    monkeypatch.setattr(sel, "_chat_json", fake_chat)
    cards = [
        PaperCard(id="1", title="Federated Learning", abstract="non-iid"),
        PaperCard(id="2", title="Galaxy Formation", abstract="dark matter"),
    ]
    out = select_cards("federated learning", cards)
    assert [c.id for c in out] == ["1"]
