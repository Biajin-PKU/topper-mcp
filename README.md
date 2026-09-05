<p align="center">
  <img src="assets/topper-lockup.png" alt="TOPPER — Top Paper Retriever" width="520">
</p>

<p align="center">
  <strong>只搜顶刊的学术论文检索</strong> —— Python 库、命令行、MCP 服务三种用法<br>
  <sub>CCF A/B · 中科院 1/2 区 · 精选旗舰刊</sub>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-server-8B0012">
</p>

<p align="center">
  <b>简体中文</b> · <a href="README.en.md">English</a>
</p>

---

用一段话描述你要找的研究，返回这个方向上的顶会顶刊论文。

---

## 它做了什么

```
一个研究问题
  → LLM 写检索简报      （query families + 领域自己的术语）
  → 多轮检索            （Semantic Scholar + OpenAlex，合并去重）
  → 档次门              （CCF / 中科院分区 / JCR / WoS / 旗舰刊名单）
  → 相关性门            （主题契合度，光有声望进不来）
  → 打分 + 全景          （学派、活跃团队）
```

## 安装

```bash
pip install "topper-mcp[mcp]"      # 含 MCP 服务
pip install topper-mcp             # 只要库和命令行（引擎零依赖，纯标准库）

cp .env.example .env               # 只有两项必填，见下
topper doctor                      # 它会逐条告诉你还缺什么
```

## 配置

必填两项，`topper doctor` 会检查并点名缺失的那一项。

| 变量 | 必填 | 说明 |
|---|---|---|
| `TOPPER_LLM_API_KEY` | **是** | 任何 OpenAI 兼容端点——OpenAI、本地 vLLM/Ollama 网关、中转都行 |
| `TOPPER_LLM_MODEL` | **是** | 例如 `gpt-4o-mini`。无默认值，需显式指定 |
| `TOPPER_LLM_BASE_URL` | 否 | 默认 `https://api.openai.com/v1` |
| `TOPPER_LLM_FALLBACK_MODELS` | 否 | 逗号分隔，主模型不可用时依次尝试 |
| `SEMANTIC_SCHOLAR_API_KEY` | 否 | 不填也能跑，但会被限流到几秒一次。[免费申请](https://www.semanticscholar.org/product/api) |
| `OPENALEX_MAILTO` | 否 | OpenAlex 的礼貌策略；不填就走慢速公共池 |
| `TOPPER_DATA_DIR` | 否 | 档次表的位置，见 [`topper/data/README.md`](topper/data/README.md) |

### 当 MCP 服务用

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

三个工具：

| 工具 | 作用 |
|---|---|
| `search_top_papers` | 检索 |
| `explain_venue` | 查一个期刊或会议的档次标签 |
| `preview_search_plan` | 只生成检索计划，不打数据源 |

### 当命令行用

```bash
topper search "钙钛矿太阳能电池的稳定性衰减机理与界面钝化策略"
topper plan   "双重差分 多期处理 异质性"    # 只出检索计划，不打数据源
topper tiers  "Journal of Econometrics"
```

### 当库用

```python
from topper import search, SearchPolicy

hits = search(
    "graph neural networks",
    policy=SearchPolicy(ccf_levels=("A", "B"), cas_zones=(1, 2), max_age_years=5),
    limit=20,
)
```

---

## 覆盖范围

### 开箱可用

| 名录 | 数量 | 等级分布 | 默认门 |
|---|---|---|---|
| CCF 推荐目录 2026 | 670（会议 383 / 期刊 287） | A 87 · B 234 · C 349 | 只放 A/B |
| 旗舰刊名单（人工整理） | 38 | — | 全放 |

**CCF 十大方向的 A 类代表**

| 方向 | A 类代表 |
|---|---|
| 人工智能 | NeurIPS、ICML、CVPR、ICCV、ACL、AAAI、IJCAI 等 |
| 数据库 / 数据挖掘 / 内容检索 | SIGMOD、VLDB、ICDE、KDD、SIGIR |
| 软件工程 / 系统软件 / 程序设计语言 | ICSE、FSE、OSDI、SOSP、POPL、PLDI、ASE 等 |
| 计算机体系结构 / 并行与分布计算 | ISCA、ASPLOS、MICRO、HPCA、SC、FAST、EuroSys 等 |
| 计算机网络 | SIGCOMM、MobiCom、INFOCOM、NSDI |
| 网络与信息安全 | CCS、S&P、USENIX Security、NDSS、CRYPTO 等 |
| 计算机科学理论 | STOC、FOCS、SODA、CAV、LICS |
| 计算机图形学与多媒体 | SIGGRAPH、ACM MM、IEEE VIS、IEEE VR |
| 人机交互与普适计算 | CHI、CSCW、UbiComp |
| 交叉 / 综合 / 新兴 | WWW、RTSS |

**旗舰刊 38 本**

| 学科 | 刊 |
|---|---|
| 综合 | Nature、Science、Cell、Nature Communications、Science Advances、PNAS |
| 医学 | NEJM、The Lancet、JAMA、BMJ、Nature Medicine |
| 生命科学 | Nature Methods / Biotechnology / Genetics / Neuroscience、Immunity、Cancer Cell、Molecular Cell、Neuron |
| 物理 化学 材料 | PRL、JACS、Angewandte Chemie、Advanced Materials、Nature Materials / Physics / Chemistry、Chem、Joule |
| 经济 社科 | AER、Econometrica、QJE、JPE、REStud、Journal of Econometrics、Psychological Bulletin / Review、ASR、APSR |

### 需要自己装

引擎能读，但这些名录各有使用条款，本仓不做再分发。放进 `topper/data/`（或
`TOPPER_DATA_DIR` 指向的目录）即可生效，格式见
[`topper/data/README.md`](topper/data/README.md)。缺表不影响运行，`topper doctor`
会报告哪几张没装。

| 名录 | 学科数 | 等级 |
|---|---|---|
| 中科院分区 | 21 | 1–4 区 |
| JCR | 254 | Q1–Q4 |
| SCI / SSCI / A&HCI | — | 是否收录 |
| FMS 经管期刊评级 | 33 | A/B/C/D/T1/T2 |

### 另外两件要知道的

只返回元数据和链接，不抓取付费全文。

一次检索是一次 LLM 调用加若干次数据源往返，通常 30–120 秒。检索简报会缓存到磁盘，
同一个问题再问一次快得多。

## 开发

```bash
pip install -e ".[dev,mcp]"
pytest -q
```

## 许可

代码是 MIT。

两处例外：

- **名录数据**保留各自的条款，见 [`topper/data/README.md`](topper/data/README.md)
- **品牌资产**（`assets/` 目录、TOPPER 名称与标识）不在 MIT 授权范围内。代码随意 fork，请用你自己的名字和标识发布
