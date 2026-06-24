"""
engine/deepseek_client.py — Thin DeepSeek API client wrapper.

Added 2026-05-10. Used for:
  - Tool 2 judge agent (different LLM from Gemini critic → real multi-LLM debate)
  - Tool 4 outcome reasoner (V4-flash reasoning chain)
  - Cost-arbitrage path (output ~9x cheaper than Gemini 2.5 Flash)

Per project axis: LLM (DeepSeek included) NEVER enters alpha decision loop.
DeepSeek operates in operations / governance / research-co-pilot layer only.

Pricing reference (2026-05-10, may change):
  Input:  $0.14/M tokens (cache miss) / $0.014/M (cache hit)
  Output: $0.28/M tokens
  → ~9x cheaper than Gemini 2.5 Flash on output-heavy tasks

2026-05-10 Sprint 2C-1: cost recording migrated to engine.llm_cost_ledger
(unified JSONL ledger). Old `.streamlit/deepseek_llm_cost.json` was probe-only
and already cleaned; no historical data to preserve.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import socket
import time
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# Pricing per million tokens (USD)
PRICE_INPUT_USD_PER_M:  float = 0.14
PRICE_OUTPUT_USD_PER_M: float = 0.28


@dataclasses.dataclass(frozen=True)
class DeepSeekResponse:
    """Standard DeepSeek API response shape."""
    content:           str
    reasoning_content: Optional[str]   # V4-flash o1-style reasoning chain (may be None)
    model:             str
    cost_usd:          float
    latency_ms:        int
    prompt_tokens:     int
    completion_tokens: int
    reasoning_tokens:  int
    cache_hit_tokens:  int
    raw_response:      dict


def _load_credentials() -> tuple[str, str, str]:
    """Read API key + model + base_url from .streamlit/secrets.toml.

    Returns (api_key, model, base_url).
    """
    try:
        import streamlit as st
        secrets = st.secrets
        cfg = secrets["DEEPSEEK"]
        return cfg["API_KEY"], cfg["DEFAULT_MODEL"], cfg["BASE_URL"]
    except Exception:
        # Fallback: read TOML directly (for non-Streamlit contexts)
        try:
            import toml
            from pathlib import Path
            secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
            with open(secrets_path, encoding="utf-8") as f:
                data = toml.load(f)
            cfg = data["DEEPSEEK"]
            return cfg["API_KEY"], cfg["DEFAULT_MODEL"], cfg["BASE_URL"]
        except Exception as exc:
            raise RuntimeError(f"Could not load DeepSeek credentials: {exc!s}")


def call_deepseek(
    prompt:      str,
    *,
    model:       Optional[str] = None,
    max_tokens:  int = 2000,
    temperature: float = 0.1,
    timeout_s:   int = 60,
) -> DeepSeekResponse:
    """Single-shot DeepSeek chat completion.

    Args:
        prompt:      user message text
        model:       override model (default reads from secrets, typically "deepseek-v4-flash")
        max_tokens:  V4-flash needs ≥200 to leave room for reasoning chain
        temperature: 0.1 default for stability
        timeout_s:   network timeout

    Returns:
        DeepSeekResponse with content + reasoning + cost + latency.

    Raises:
        RuntimeError on API/network failure (caller decides fallback).
    """
    api_key, default_model, base_url = _load_credentials()
    model_to_use = model or default_model

    req_body = json.dumps({
        "model":       model_to_use,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=req_body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    socket.setdefaulttimeout(timeout_s)

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body[:300]}")
    except Exception as exc:
        raise RuntimeError(f"DeepSeek call failed: {exc!s}")
    latency_ms = int((time.time() - t0) * 1000)

    msg = raw.get("choices", [{}])[0].get("message", {}) or {}
    content = msg.get("content", "") or ""
    reasoning_content = msg.get("reasoning_content")

    usage = raw.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    reasoning_tokens = int(
        (usage.get("completion_tokens_details", {}) or {}).get("reasoning_tokens", 0) or 0
    )
    cache_hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(usage.get("prompt_cache_miss_tokens", prompt_tokens) or prompt_tokens)

    cost = (
        cache_miss * PRICE_INPUT_USD_PER_M / 1_000_000
        + cache_hit * (PRICE_INPUT_USD_PER_M / 10) / 1_000_000  # cache hit is 10x cheaper
        + completion_tokens * PRICE_OUTPUT_USD_PER_M / 1_000_000
    )

    response = DeepSeekResponse(
        content=content,
        reasoning_content=reasoning_content,
        model=raw.get("model", model_to_use),
        cost_usd=round(cost, 6),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_hit_tokens=cache_hit,
        raw_response=raw,
    )
    try:
        _record_cost(response)
    except Exception as exc:
        logger.warning("DeepSeek cost ledger write failed: %s", exc)
    return response


def _record_cost(response: DeepSeekResponse) -> None:
    """Append a call entry to the unified llm_cost_ledger.

    Migrated 2026-05-10 (Sprint 2C-1) from `.streamlit/deepseek_llm_cost.json`
    to `data/llm_cost_ledger.jsonl`. DeepSeek-specific extras (reasoning_tokens,
    cache_hit_tokens) ride along in entry.extra so they survive in the unified
    ledger.
    """
    from engine.llm_cost_ledger import record_call
    record_call(
        agent_id          = "deepseek",
        provider          = "deepseek",
        model             = response.model,
        prompt_tokens     = response.prompt_tokens,
        completion_tokens = response.completion_tokens,
        cost_usd          = response.cost_usd,
        latency_ms        = response.latency_ms,
        extra             = {
            "reasoning_tokens": response.reasoning_tokens,
            "cache_hit_tokens": response.cache_hit_tokens,
        },
    )


def get_deepseek_cumulative_cost() -> dict:
    """Aggregate cumulative cost from the unified ledger (DeepSeek-scoped).

    Backwards-compatible public API: returns dict with ytd_year /
    ytd_spend_usd / lifetime_spend_usd / total_calls / last_call_ts /
    recent (last 50 entries). engine.llm_budget.get_budget_status() reads
    this shape — preserve key set.
    """
    from engine.llm_cost_ledger import get_calls
    now = datetime.datetime.utcnow()
    cur_year = now.year
    year_start = datetime.date(cur_year, 1, 1)

    all_calls = get_calls(agent_id="deepseek")
    ytd_calls = [c for c in all_calls
                 if (_d := _entry_date_safe(c.ts)) is not None and _d >= year_start]

    ytd_spend      = round(sum(c.cost_usd for c in ytd_calls), 6)
    lifetime_spend = round(sum(c.cost_usd for c in all_calls), 6)
    total_calls    = len(all_calls)
    last_call_ts   = all_calls[-1].ts if all_calls else None

    # "recent" preserves DeepSeek-specific fields (reasoning_tokens / cache_hit)
    recent_entries = []
    for c in all_calls[-50:]:
        recent_entries.append({
            "at":                c.ts,
            "model":             c.model,
            "cost_usd":          c.cost_usd,
            "latency_ms":        c.latency_ms,
            "prompt_tokens":     c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "reasoning_tokens":  int(c.extra.get("reasoning_tokens", 0) or 0),
            "cache_hit_tokens":  int(c.extra.get("cache_hit_tokens", 0) or 0),
        })

    return {
        "ytd_year":           cur_year,
        "ytd_spend_usd":      ytd_spend,
        "lifetime_spend_usd": lifetime_spend,
        "total_calls":        total_calls,
        "last_call_ts":       last_call_ts,
        "recent":             recent_entries,
    }


def _entry_date_safe(ts: str):
    """Parse ts string to datetime.date; return None on failure."""
    try:
        clean = ts.rstrip("Z")
        return datetime.datetime.fromisoformat(clean).date()
    except (ValueError, AttributeError):
        return None


def is_available() -> bool:
    """Quick check whether DeepSeek is configured + reachable.

    Returns False on missing credentials OR network failure (no exception).
    """
    try:
        api_key, _, _ = _load_credentials()
        if not api_key or not api_key.startswith("sk-"):
            return False
        # Don't actually call API — just credential check
        return True
    except Exception:
        return False
