"""Search Brief planner: an LLM turns one research question into query families.

The brief contract:
  core / synonym_expansion            (required)
  method_modality / survey_review /
  benchmark_dataset / venue_probe     (optional — omitted when unnatural)
  landmark_seed                       (only when a curated scenario matches)

Each family is written in the field's own English terminology, so the
downstream keyword search matches how the literature is actually titled.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from topper import config

# Query-family names the planner is allowed to emit.
QUERY_FAMILY_KEYS = (
    "core",
    "synonym_expansion",
    "method_modality",
    "survey_review",
    "benchmark_dataset",
    "venue_probe",
)

# Round schedule: high-precision first, optional families only if LLM filled them.
ROUND_SCHEDULE = (
    ("core",),
    ("synonym_expansion", "method_modality"),
    ("landmark_seed", "survey_review", "venue_probe", "benchmark_dataset"),
)

_SYSTEM = """You are a Search Brief planner for top-only academic retrieval (CCF A/B, CAS 1/2, flagship journals).

Given a user research question (Chinese or English), output ONE JSON object only (no markdown) with:

{
  "intent": "short English label of the research task",
  "language": "zh" | "en",
  "topic_anchors": ["canonical English topic phrases that MUST appear in core queries"],
  "query_families": {
    "core": ["..."],
    "synonym_expansion": ["..."],
    "method_modality": ["..."],
    "survey_review": ["..."],
    "benchmark_dataset": ["..."],
    "venue_probe": ["..."]
  },
  "notes": ["optional short notes"]
}

Rules for query_families (Semantic Scholar keyword search):
- Every query string MUST be scholarly English keywords, not a full interrogative sentence.
- Prefer 4–12 content words; keep distinctive domain terms (do not drop the topic).
- core: 2–4 high-precision queries. Each MUST include at least one topic_anchors phrase.
- synonym_expansion: 2–4 paraphrases / alternate scholarly wording. Still on-topic.
- method_modality: 0–3 method/mechanism angles ONLY if natural for the topic; else OMIT the key.
- survey_review: 0–2 ONLY if user wants survey/review OR a classic survey is central; else OMIT.
- benchmark_dataset: 0–2 ONLY if user asks for benchmarks/datasets/evaluation suites OR the field is benchmark-centric; else OMIT. Never add "benchmark" to unrelated chemistry/economics/materials queries.
- venue_probe: 0–2 optional "topic + venue-class keyword" probes; else OMIT.
- Do NOT invent off-topic CS/LLM filler (NumPy, generic "scientific research", "open-source implementation") unless the user asked for that.
- Do NOT translate word-by-word into broken English; use standard field terminology (e.g. 双重差分 → difference-in-differences, 联邦学习 → federated learning, 单原子催化剂 → single-atom catalyst).
- Omit optional family keys entirely when unused (do not output empty lists).
- topic_anchors: 1–4 canonical English phrases.
"""


def _load_openai_env() -> tuple[str, str, str]:
    """base_url, api_key, model — see `topper.config` for the full surface."""
    config.load_dotenv()
    return config.llm_base_url(), config.llm_api_key(), config.llm_model()


def _model_candidates(primary: str) -> list[str]:
    """Primary model first, then the configured fallbacks."""
    chain = config.llm_model_chain()
    out: list[str] = []
    for m in [primary, *chain]:
        if m and m not in out:
            out.append(m)
    return out


def llm_available() -> bool:
    _, key, _ = _load_openai_env()
    return bool(key)


def _chat_json(messages: list[dict[str, str]], *, timeout: float = 0.0) -> dict[str, Any]:
    base, key, primary = _load_openai_env()
    timeout = timeout or config.llm_timeout_s()
    if not key:
        raise RuntimeError(
            "No LLM configured. Set TOPPER_LLM_API_KEY and TOPPER_LLM_MODEL "
            "(see .env.example), or run `topper doctor`."
        )

    url = f"{base}/chat/completions"
    last_err: Exception | None = None
    payload: dict[str, Any] | None = None
    import time as _time

    # Prefer fast fail-over across models over long same-model retries.
    max_attempts = int(os.environ.get("TOPPER_PLANNER_RETRIES", "2") or 2)
    max_attempts = max(1, min(max_attempts, 4))

    for model in _model_candidates(primary):
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        for attempt in range(max_attempts):
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "User-Agent": "topper/0.1",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")[:400]
                last_err = RuntimeError(f"planner HTTP {e.code} model={model}: {err_body}")
                if e.code in {429, 502, 503}:
                    _time.sleep(1.5 * (attempt + 1))
                    continue
                # non-retryable for this model — try next model
                break
            except Exception as e:  # noqa: BLE001 — retry timeout/reset
                last_err = e
                _time.sleep(1.5 * (attempt + 1))
                continue
        if payload is not None:
            break
    else:
        raise last_err or RuntimeError("planner failed")

    if payload is None:
        raise last_err or RuntimeError("planner failed")

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"planner empty choices: {str(payload)[:200]}")
    content = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("planner empty content")
    return _parse_json_object(content)


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def _parse_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    m = _JSON_FENCE.search(s)
    if m:
        s = m.group(1).strip()
    # trim leading junk before first {
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        s = s[i : j + 1]
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise RuntimeError("planner JSON root must be object")
    return obj


def _clean_query_list(vals: Any, *, require_anchor: Optional[list[str]] = None) -> list[str]:
    if not vals:
        return []
    if not isinstance(vals, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    anchors = [a.lower() for a in (require_anchor or []) if a]
    for v in vals:
        if not isinstance(v, str):
            continue
        q = re.sub(r"\s+", " ", v).strip().strip('"').strip("'")
        if len(q) < 3 or len(q) > 160:
            continue
        # drop pure CJK lines — S2 is English-first
        if re.search(r"[一-鿿]", q) and not re.search(r"[A-Za-z]{3,}", q):
            continue
        key = q.lower()
        if key in seen:
            continue
        if anchors:
            # soft: prefer keeping only if some anchor token overlap; don't drop all if LLM failed
            pass
        seen.add(key)
        out.append(q)
    return out


def normalize_brief(raw: dict[str, Any], *, original: str) -> dict[str, Any]:
    """Coerce LLM output into the query-family shape the engine executes."""
    qf_in = raw.get("query_families") if isinstance(raw.get("query_families"), dict) else {}
    anchors = []
    for a in raw.get("topic_anchors") or []:
        if isinstance(a, str) and a.strip():
            anchors.append(re.sub(r"\s+", " ", a).strip())
    anchors = list(dict.fromkeys(anchors))[:6]

    families: dict[str, list[str]] = {}
    for key in QUERY_FAMILY_KEYS:
        cleaned = _clean_query_list(qf_in.get(key))
        if cleaned:
            families[key] = cleaned

    # Ensure core exists: fall back to anchors joined
    if not families.get("core"):
        if anchors:
            families["core"] = [" ".join(anchors[:3])]
        else:
            # last resort latin tokens from user text
            latin = " ".join(re.findall(r"[A-Za-z][A-Za-z0-9\-+]{2,}", original or ""))
            families["core"] = [latin or "research"]

    if not families.get("synonym_expansion"):
        # A single core query with no paraphrase gives the fan-out nothing to work with.
        c0 = families["core"][0]
        families["synonym_expansion"] = [c0 if len(families["core"]) == 1 else families["core"][-1]]

    # If core queries lost anchors but anchors exist, prepend anchor query
    if anchors:
        anchor_q = " ".join(anchors[:3])
        core = families["core"]
        blob = " ".join(core).lower()
        if not any(a.lower() in blob for a in anchors):
            families["core"] = [anchor_q] + core

    intent = raw.get("intent") if isinstance(raw.get("intent"), str) else "general"
    intent = (intent or "general").strip() or "general"
    lang = raw.get("language") if isinstance(raw.get("language"), str) else None
    notes = [str(n) for n in (raw.get("notes") or []) if n][:8]
    notes = ["planner:search_brief", f"model_intent:{intent}"] + notes

    return {
        "original": original,
        "intent": intent,
        "language": lang,
        "topic_anchors": anchors,
        "query_families": families,
        "notes": notes,
        "planner": "rh",
    }


def _brief_cache_dir() -> str:
    override = os.environ.get("TOPPER_BRIEF_CACHE_DIR") or ""
    if override.strip():
        return override.strip()
    return os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "top-paper-retriever",
        "briefs",
    )


def _brief_cache_key(q: str, model_hint: str) -> str:
    import hashlib

    payload = json.dumps(
        {"q": " ".join(q.split()).strip().lower(), "model": model_hint, "v": 2},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _brief_cache_get(key: str) -> Optional[dict[str, Any]]:
    if os.environ.get("TOPPER_BRIEF_CACHE", "1").strip() in {"0", "false", "no"}:
        return None
    path = os.path.join(_brief_cache_dir(), f"{key}.json")
    try:
        import time as _time

        ttl = float(os.environ.get("TOPPER_BRIEF_CACHE_TTL_S", str(7 * 86400)))
        if not os.path.isfile(path):
            return None
        if ttl > 0 and _time.time() - os.path.getmtime(path) > ttl:
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _brief_cache_put(key: str, value: dict[str, Any]) -> None:
    if os.environ.get("TOPPER_BRIEF_CACHE", "1").strip() in {"0", "false", "no"}:
        return
    try:
        d = _brief_cache_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{key}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def plan_search_brief(original: str, *, timeout: float = 0.0) -> dict[str, Any]:
    """Ask the planner model for a Search Brief (disk-cached)."""
    q = (original or "").strip()
    if not q:
        raise ValueError("empty query")
    _, _, primary = _load_openai_env()
    # Cache key ignores fallback model chain — same intent → same brief.
    ck = _brief_cache_key(q, primary)
    hit = _brief_cache_get(ck)
    if hit and isinstance(hit.get("query_families"), dict):
        out = dict(hit)
        out["original"] = q
        notes = list(out.get("notes") or [])
        if "cache:brief_hit" not in notes:
            notes = ["cache:brief_hit", *notes]
        out["notes"] = notes
        return out

    to = float(os.environ.get("TOPPER_PLANNER_TIMEOUT", str(timeout)) or timeout)
    raw = _chat_json(
        [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"User research question:\n{q}\n\nReturn the JSON Search Brief now.",
            },
        ],
        timeout=to,
    )
    brief = normalize_brief(raw, original=q)
    _brief_cache_put(ck, brief)
    return brief
