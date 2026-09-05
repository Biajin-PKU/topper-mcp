from topper.sources.s2 import SemanticScholarSource


def test_s2_to_card_prefers_short_venue_alias():
    src = SemanticScholarSource()
    card = src._to_card(
        {
            "paperId": "abc",
            "title": "Hello",
            "year": 2020,
            "venue": "Neural Information Processing Systems",
            "publicationVenue": {
                "name": "Neural Information Processing Systems",
                "alternate_names": ["NeurIPS", "NIPS"],
            },
            "citationCount": 12,
            "authors": [
                {"name": "A. Author", "affiliations": ["Stanford University"]},
                {"name": "B. Author", "affiliations": ["Google DeepMind"]},
            ],
            "externalIds": {"DOI": "10.1/x", "ArXiv": "2001.1"},
            "url": "https://www.semanticscholar.org/paper/abc",
            "openAccessPdf": {"url": ""},
        }
    )
    assert card.id == "s2:abc"
    assert card.venue == "NIPS" or card.venue == "NeurIPS"
    assert card.cited_by_count == 12
    assert card.doi == "10.1/x"
    assert card.oa_url.endswith("2001.1")
    assert "Stanford University" in card.institutions
    assert "Google DeepMind" in card.institutions
