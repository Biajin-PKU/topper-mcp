"""The parts that only exist in the open-source packaging."""

import json

import pytest

from topper import config
from topper.mcp_server import TOOL_FUNCS, _tier_label, admission_label, explain_venue
from topper.models import PaperCard, SearchPolicy
from topper.policy import accepts, attach_tiers


def _card(venue: str, year: int = 2024) -> PaperCard:
    c = PaperCard(id="x", title="t", year=year, venue=venue)
    attach_tiers(c)
    return c


POLICY = SearchPolicy(ccf_levels=("A", "B"), cas_zones=(1, 2), max_age_years=8)


@pytest.mark.parametrize(
    "venue,expected",
    [
        ("Nature", True),  # flagship list
        ("The Lancet", True),
        ("Journal of Econometrics", True),  # flagship list, no CAS row needed
        ("NeurIPS", True),  # CCF
        ("Journal of Obscure Studies", False),
    ],
)
def test_gate_works_without_licensed_tables(venue, expected):
    """A fresh clone ships no CAS/JCR catalogue; the gate must still be useful."""
    assert accepts(_card(venue), POLICY) is expected


def test_flagship_list_does_not_assert_tier_values():
    """The bundled flagship list is names only — no zone/quartile is claimed."""
    with (config.data_dir() / "flagship_venues.json").open(encoding="utf-8") as f:
        doc = json.load(f)
    assert all(isinstance(v, str) for v in doc["venues"])


def test_missing_required_names_the_variable(monkeypatch):
    for var in ("TOPPER_LLM_API_KEY", "OPENAI_API_KEY", "TOPPER_LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    gaps = " ".join(config.missing_required())
    assert "TOPPER_LLM_API_KEY" in gaps
    assert "TOPPER_LLM_MODEL" in gaps


def test_model_chain_puts_primary_first(monkeypatch):
    monkeypatch.setenv("TOPPER_LLM_MODEL", "primary")
    monkeypatch.setenv("TOPPER_LLM_FALLBACK_MODELS", "backup-a, backup-b ,primary")
    assert config.llm_model_chain() == ["primary", "backup-a", "backup-b"]


def test_tier_label_is_human_readable():
    assert _tier_label({"ccf": "A"}) == "CCF-A"
    assert "Top" in _tier_label({"cas_zone": 1, "cas_top": True})
    assert _tier_label({}) == ""


def test_admission_label_never_blank_for_an_admitted_paper():
    """Nature clears the gate via the flagship list and carries no tier row."""
    assert admission_label({"venue": "Nature", "tiers": {}}) == "flagship"
    assert admission_label({"venue": "X", "tiers": {"ccf": "A"}}) == "CCF-A"
    assert "CCF-A author" in admission_label(
        {"venue": "arXiv", "tiers": {}, "matched_authority": ["arxiv_bridge:ccf_a_author:x"]}
    )
    assert admission_label({"venue": "Journal of Obscure Studies", "tiers": {}}) == ""


def test_tool_functions_carry_the_schema():
    """MCP derives each tool schema from the signature and docstring, so both
    have to be there — a missing docstring ships a tool nobody can use."""
    import inspect

    for fn in TOOL_FUNCS:
        doc = inspect.getdoc(fn)
        assert doc and len(doc) > 80, f"{fn.__name__} needs a usable description"
        assert "Args:" in doc, f"{fn.__name__} must document its parameters"
        sig = inspect.signature(fn)
        for name, param in sig.parameters.items():
            assert param.annotation is not inspect.Parameter.empty, (
                f"{fn.__name__}.{name} needs a type annotation for the schema"
            )


def test_explain_venue_reports_unknown_without_crashing():
    out = explain_venue("Journal of Obscure Studies")
    assert out["known"] is False
    assert "note" in out


def test_explain_venue_rejects_empty():
    with pytest.raises(ValueError):
        explain_venue("")


# CCF-A levels for well-known venues in each category, pinned so a table
# regeneration cannot change them.
CCF_A_FLAGSHIPS = [
    "SIGCOMM", "MobiCom", "INFOCOM", "NSDI",              # 计算机网络
    "SIGGRAPH", "ACM MM", "IEEE VIS", "IEEE Virtual Reality",  # 图形学与多媒体
    "CAV", "LICS", "STOC", "FOCS", "SODA",                # 计算机科学理论
    "EuroSys", "USENIX ATC", "ISCA", "ASPLOS",            # 体系结构
    "NDSS", "CCS", "USENIX Security",                     # 安全
    "CSCW", "CHI", "UbiComp",                             # 人机交互
    "RTSS", "WWW",                                        # 交叉/综合
    "NeurIPS", "CVPR", "ACL", "ICML", "AAAI",             # 人工智能
    "SIGMOD", "VLDB", "KDD", "SIGIR", "ICDE",             # 数据库/挖掘/检索
    "ICSE", "FSE", "OSDI", "SOSP", "POPL", "PLDI",        # 软工/系统/PL
]


@pytest.mark.parametrize("venue", CCF_A_FLAGSHIPS)
def test_ccf_a_flagships_are_labelled_a(venue):
    from topper.tiers.registry import get_default_registry

    assert get_default_registry().lookup(venue).to_dict()["ccf"] == "A", (
        f"{venue} is CCF-A in the recommended catalog"
    )
