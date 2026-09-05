<p align="center">
  <img src="assets/topper-lockup.png" alt="TOPPER — Top Paper Retriever" width="520">
</p>

<p align="center">
  <strong>Top-only academic paper retrieval</strong> — a Python library, a CLI, and an MCP server.<br>
  <sub>CCF A/B · CAS zone 1/2 · flagship journals</sub>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-server-8B0012">
</p>

<p align="center">
  <a href="README.md">简体中文</a> · <b>English</b>
</p>

---

Describe the research you are looking for in a sentence, and get back the
top-conference and top-journal papers in that direction.

---

## What it does

```
research question
  → LLM Search Brief        (query families + the field's own terminology)
  → multi-round retrieval   (Semantic Scholar + OpenAlex, deduped)
  → tier gate               (CCF / CAS zone / JCR / WoS / flagship list)
  → relevance gate          (topical fit, so prestige alone cannot get in)
  → score + landscape       (schools of thought, active teams)
```

## Install

```bash
pip install "topper-mcp[mcp]"      # with the MCP server
pip install topper-mcp             # library + CLI only (stdlib-only engine)

cp .env.example .env               # fill in two things, see below
topper doctor                      # tells you exactly what is still missing
```

## Configure

Two things are required. `topper doctor` checks both and names what is missing.

| Variable | Required | What it is |
|---|---|---|
| `TOPPER_LLM_API_KEY` | **yes** | Any OpenAI-compatible endpoint — OpenAI, a local vLLM/Ollama gateway, a relay. |
| `TOPPER_LLM_MODEL` | **yes** | e.g. `gpt-4o-mini`. No default; must be set. |
| `TOPPER_LLM_BASE_URL` | no | Defaults to `https://api.openai.com/v1`. |
| `TOPPER_LLM_FALLBACK_MODELS` | no | Comma-separated; tried in order when the primary model is unavailable. |
| `SEMANTIC_SCHOLAR_API_KEY` | no | Works without one, but throttled to roughly a request every few seconds. [Free key](https://www.semanticscholar.org/product/api). |
| `OPENALEX_MAILTO` | no | OpenAlex politeness policy; without it you share the slow pool. |
| `TOPPER_DATA_DIR` | no | Where tier tables live. See [`topper/data/README.md`](topper/data/README.md). |

### Use it as an MCP server

```json
{
  "mcpServers": {
    "topper": {
      "command": "topper-mcp",
      "env": {
        "TOPPER_LLM_API_KEY": "sk-...",
        "TOPPER_LLM_MODEL": "gpt-4o-mini",
        "SEMANTIC_SCHOLAR_API_KEY": "..."
      }
    }
  }
}
```

Three tools:

| Tool | Purpose |
|---|---|
| `search_top_papers` | Search |
| `explain_venue` | Tier labels for one journal or conference |
| `preview_search_plan` | Query plan only, no source calls |

### Use it from the shell

```bash
topper search "钙钛矿太阳能电池的稳定性衰减机理与效率提升的界面钝化策略"
topper plan   "difference-in-differences with staggered adoption"   # no source calls
topper tiers  "Journal of Econometrics"
```

### Use it as a library

```python
from topper import search, SearchPolicy

hits = search(
    "graph neural networks",
    policy=SearchPolicy(ccf_levels=("A", "B"), cas_zones=(1, 2), max_age_years=5),
    limit=20,
)
```

---

## Coverage

### Shipped

| Catalogue | Size | Levels | Default gate |
|---|---|---|---|
| CCF recommended catalog 2026 | 670 (383 conferences / 287 journals) | A 87 · B 234 · C 349 | A/B only |
| Flagship list (hand-curated) | 38 | — | all |

**CCF tier-A venues, by category**

| Category | Tier-A venues |
|---|---|
| Artificial intelligence | NeurIPS, ICML, CVPR, ICCV, ACL, AAAI, IJCAI, … |
| Databases / data mining / retrieval | SIGMOD, VLDB, ICDE, KDD, SIGIR |
| Software engineering / systems / PL | ICSE, FSE, OSDI, SOSP, POPL, PLDI, ASE, … |
| Architecture / parallel & distributed | ISCA, ASPLOS, MICRO, HPCA, SC, FAST, EuroSys, … |
| Networking | SIGCOMM, MobiCom, INFOCOM, NSDI |
| Security | CCS, S&P, USENIX Security, NDSS, CRYPTO, … |
| Theory | STOC, FOCS, SODA, CAV, LICS |
| Graphics & multimedia | SIGGRAPH, ACM MM, IEEE VIS, IEEE VR |
| HCI & ubiquitous computing | CHI, CSCW, UbiComp |
| Interdisciplinary / emerging | WWW, RTSS |

**The 38 flagship journals**

| Field | Journals |
|---|---|
| General | Nature, Science, Cell, Nature Communications, Science Advances, PNAS |
| Medicine | NEJM, The Lancet, JAMA, BMJ, Nature Medicine |
| Life sciences | Nature Methods / Biotechnology / Genetics / Neuroscience, Immunity, Cancer Cell, Molecular Cell, Neuron |
| Physics, chemistry, materials | PRL, JACS, Angewandte Chemie, Advanced Materials, Nature Materials / Physics / Chemistry, Chem, Joule |
| Economics & social science | AER, Econometrica, QJE, JPE, REStud, Journal of Econometrics, Psychological Bulletin / Review, ASR, APSR |

### Bring your own

The engine reads these, but each catalogue carries its own terms of use and is
not redistributed here. Drop them in `topper/data/` (or wherever
`TOPPER_DATA_DIR` points) and they take effect; the format is in
[`topper/data/README.md`](topper/data/README.md). Nothing breaks without them —
`topper doctor` reports which tables are missing.

| Catalogue | Categories | Levels |
|---|---|---|
| CAS zones (中科院分区) | 21 | 1–4 |
| JCR | 254 | Q1–Q4 |
| SCI / SSCI / A&HCI | — | indexed or not |
| FMS journal rating (经管) | 33 | A/B/C/D/T1/T2 |

### Two more things worth knowing

Metadata and links only; no paywalled full text.

A search costs one LLM call plus several source round-trips: typically 30–120
seconds. The brief is cached on disk, so repeating a question is much faster.

## Development

```bash
pip install -e ".[dev,mcp]"
pytest -q
```

## License

MIT for the code.

Two carve-outs:

- **Catalogues** keep their own terms. See [`topper/data/README.md`](topper/data/README.md).
- **Brand assets** (`assets/`, the TOPPER name and mark) are not covered by the
  MIT grant. Fork the code freely; please ship it under your own name and logo.
