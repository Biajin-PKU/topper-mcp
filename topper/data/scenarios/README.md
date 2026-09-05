# Scenario packs (optional)

A scenario pack is a curated answer key for one recurring topic: the landmark
papers you already know should surface, plus the schools and benchmarks you
want grouped. When a query matches a pack, those seeds are searched alongside
the LLM-planned queries and the landmark titles bypass the relevance gate.

Packs are field-specific. Drop your own `*.json` in this directory (or under
`$TOPPER_DATA_DIR/scenarios/`).

```json
{
  "id": "my_topic",
  "name_zh": "主题中文名",
  "name_en": "Topic name",
  "match_any": ["trigger phrase", "触发词"],
  "seed_queries": ["Exact Title Of A Landmark Paper"],
  "gold_substrings": ["distinctive title fragment"],
  "schools": [
    {"name_zh": "学派名", "name_en": "School name", "match": ["title fragment"]}
  ],
  "benchmarks": [{"name": "BenchmarkName", "match": ["benchmarkname"]}]
}
```

`match_any` is matched case-insensitively against the raw user query, so keep
the triggers distinctive — a pack that fires on "agent" will hijack unrelated
searches.
