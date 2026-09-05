"""Cache key normalization, TTL/eviction, and demo lookup."""

import json
import time

import pytest

from topper.search_cache import (
    DemoStore,
    SearchCache,
    cache_key,
    normalize_query,
)


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  Hello   WORLD  ") == "hello world"
    assert normalize_query("") == ""


def test_cache_key_stable_across_trivial_edits():
    a = cache_key("Graph neural networks", limit=10, ccf="A,B")
    b = cache_key("  graph   neural networks ", ccf="A,B", limit=10)
    assert a == b
    assert cache_key("other query", limit=10, ccf="A,B") != a


def test_cache_roundtrip_and_miss(tmp_path):
    c = SearchCache(tmp_path / "c", ttl_s=60)
    k = cache_key("q", limit=1)
    assert c.get(k) is None
    c.put(k, {"results": [1, 2]})
    assert c.get(k) == {"results": [1, 2]}
    assert c.stats()["entries"] == 1


def test_cache_expires(tmp_path):
    c = SearchCache(tmp_path / "c", ttl_s=1)
    k = cache_key("q", limit=1)
    c.put(k, {"results": []})
    assert c.get(k) is not None
    time.sleep(1.1)
    assert c.get(k) is None


def test_cache_evicts_oldest(tmp_path):
    c = SearchCache(tmp_path / "c", ttl_s=0, max_entries=3)
    for i in range(5):
        c.put(cache_key(f"q{i}", limit=1), {"i": i})
        time.sleep(0.01)
    assert c.stats()["entries"] <= 3


def test_demo_store_matches_query(tmp_path):
    d = tmp_path / "demos"
    d.mkdir()
    (d / "sample.json").write_text(
        json.dumps(
            {
                "id": "sample",
                "queries": ["Test Demo Query"],
                "payload": {"results": [{"title": "x"}], "count": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = DemoStore(d)
    assert store.get("test demo query")["count"] == 1
    assert store.get("  Test  Demo  Query ")["count"] == 1
    assert store.get("unrelated") is None
    assert store.queries() == ["test demo query"]


def test_demo_store_empty_dir(tmp_path):
    store = DemoStore(tmp_path / "nope")
    assert store.queries() == []
    assert store.get("anything") is None
