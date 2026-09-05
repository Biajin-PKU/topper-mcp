from topper.tiers.registry import TierRegistry


def test_ccf_neurips():
    reg = TierRegistry()
    t = reg.lookup("NeurIPS")
    assert t.ccf == "A"
    assert reg.resolve_key("Advances in Neural Information Processing Systems") == "neurips"


def test_ccf_emnlp_b():
    reg = TierRegistry()
    assert reg.lookup("EMNLP").ccf == "B"


def test_cas_nature_mi():
    reg = TierRegistry()
    t = reg.lookup("Nature Machine Intelligence")
    assert t.cas_zone == 1


def test_unknown_venue():
    reg = TierRegistry()
    t = reg.lookup("Totally Fake Workshop 2099")
    assert t.ccf is None
    assert t.cas_zone is None


def test_sci_nature():
    reg = TierRegistry()
    t = reg.lookup("Nature")
    assert t.sci is True
    assert t.ssci is False
    assert t.cas_zone == 1
    assert t.jcr_quartile == "Q1"
    assert t.impact_factor is not None


def test_ssci_journal():
    reg = TierRegistry()
    t = reg.lookup("American Economic Review")
    assert t.ssci is True
    assert t.sci is False
    assert t.cas_major == "经济学"
    assert t.fms_tier == "A"
    assert t.fms_discipline == "一般经济"


def test_fms_education_journal():
    reg = TierRegistry()
    t = reg.lookup("Economics of Education Review")
    assert t.fms_tier == "B"
    assert t.fms_discipline == "教育管理"
    assert reg.stats()["fms_journals"] >= 1000


def test_ahci_journal():
    reg = TierRegistry()
    t = reg.lookup("Journal of Philosophy")
    assert t.ahci is True
    assert t.ssci is False


def test_issn_lookup():
    reg = TierRegistry()
    # Nature's ISSN resolves to the same journal
    t = reg.lookup("0028-0836")
    assert t.cas_zone == 1
    assert t.sci is True
