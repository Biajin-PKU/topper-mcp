"""Every environment variable TOPPER reads, in one place.

Two things must be configured before the engine can run:

1. **A paper source.** Semantic Scholar works without a key but is throttled
   hard; a free key raises the limit a lot. OpenAlex needs no key, only a
   contact email (its politeness policy).
2. **An LLM.** The planner turns one research question into a Search Brief
   (query families + topic anchors). Any OpenAI-compatible `/v1/chat/completions`
   endpoint works — OpenAI, a local vLLM/Ollama gateway, or a relay.

Tier tables (CCF / CAS / JCR) are data, not config — see `topper/data/README.md`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DOTENV_LOADED = False


def load_dotenv(path: Optional[Path] = None) -> None:
    """Read `.env` from the current directory once. Real env always wins."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED and path is None:
        return
    env_path = Path(path) if path else Path.cwd() / ".env"
    if path is None:
        _DOTENV_LOADED = True
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return default


# --- LLM (planner / brief / selector) ---------------------------------------


def llm_base_url() -> str:
    return _env(
        "TOPPER_LLM_BASE_URL",
        "OPENAI_BASE_URL",
        default="https://api.openai.com/v1",
    ).rstrip("/")


def llm_api_key() -> str:
    return _env("TOPPER_LLM_API_KEY", "OPENAI_API_KEY")


def llm_model() -> str:
    """Primary planner model. No default; must be set explicitly."""
    return _env("TOPPER_LLM_MODEL", "OPENAI_MODEL")


def llm_model_chain() -> list[str]:
    """Primary first, then `TOPPER_LLM_FALLBACK_MODELS` (comma separated)."""
    chain = [llm_model()]
    for m in _env("TOPPER_LLM_FALLBACK_MODELS").split(","):
        m = m.strip()
        if m and m not in chain:
            chain.append(m)
    return [m for m in chain if m]


def llm_timeout_s() -> float:
    try:
        return float(_env("TOPPER_LLM_TIMEOUT_S", default="45"))
    except ValueError:
        return 45.0


# --- Paper sources -----------------------------------------------------------


def s2_api_keys() -> list[str]:
    """Semantic Scholar keys. Several are pooled round-robin when given."""
    raw = _env("SEMANTIC_SCHOLAR_API_KEYS", "SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY")
    return [k.strip() for k in raw.split(",") if k.strip()]


def openalex_mailto() -> str:
    """OpenAlex asks for a contact address; without it you share the slow pool."""
    return _env("OPENALEX_MAILTO", "UNPAYWALL_EMAIL")


# --- Tier tables -------------------------------------------------------------


def data_dir() -> Path:
    """Where CCF / CAS / JCR tables live. Defaults to the bundled directory."""
    override = _env("TOPPER_DATA_DIR")
    return Path(override) if override else Path(__file__).resolve().parent / "data"


# --- Caches ------------------------------------------------------------------


def cache_dir() -> Path:
    override = _env("TOPPER_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "topper"


def brief_cache_enabled() -> bool:
    return _env("TOPPER_BRIEF_CACHE", default="1").lower() not in {"0", "false", "no"}


# --- Optional egress proxy ---------------------------------------------------


def proxy_enabled() -> bool:
    return _env("TOPPER_PROXY_ENABLED", default="0").lower() not in {"0", "false", "no"}


def missing_required() -> list[str]:
    """Human-readable list of what still has to be set. Used by `topper doctor`."""
    gaps: list[str] = []
    if not llm_api_key():
        gaps.append("TOPPER_LLM_API_KEY — the planner cannot run without an LLM")
    if not llm_model():
        gaps.append("TOPPER_LLM_MODEL — e.g. gpt-4o-mini, or whatever your endpoint serves")
    if not s2_api_keys():
        gaps.append(
            "SEMANTIC_SCHOLAR_API_KEY — optional, but without it Semantic Scholar "
            "throttles to roughly one request per few seconds (free key: "
            "https://www.semanticscholar.org/product/api)"
        )
    return gaps
