"""
engine/llm/providers/anthropic_provider.py — Anthropic Messages API adapter.

Wraps `anthropic.Anthropic().messages.create()` with our project's
conventions: secrets.toml key loading, system-prompt prompt caching
(5-min ephemeral) on by default, tool use support, normalized result
shape that's provider-agnostic.

Design choices (per claude-api skill guidance):
  - SDK only (no raw httpx) — official anthropic Python SDK
  - Top-level `cache_control={"type": "ephemeral"}` on system blocks
    to auto-cache the system prompt (90% input-cost reduction on
    repeated calls with the same persona)
  - For Sonnet 4.6 + Haiku 4.5: NO `temperature` / `top_p` / `top_k`
    (Anthropic 4.6+ removed/deprecated; would 400 on Opus 4.7)
  - Adaptive thinking OFF by default for narrator-style short outputs
    (cost + latency); caller can enable via thinking=True
  - Typed-exception handling (anthropic.RateLimitError, .APIStatusError)
    not string-matched error messages
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _RawCallResult:
    """Shared shape across providers — orchestrator wraps into LLMCallResult."""
    text:               str
    tool_calls:         list[dict]
    stop_reason:        str
    model:              str           # actual model returned by API
    input_tokens:       int           # uncached input
    output_tokens:      int
    cache_read_tokens:  int           # 0 if no cache hit
    cache_write_tokens: int
    latency_ms:         int
    raw_usage:          dict          # full usage object for forensics


def _get_api_key() -> str:
    """Resolve Anthropic API key from env, streamlit secrets, or a
    direct tomllib read of .streamlit/secrets.toml (in that order).

    The direct tomllib path is needed because st.secrets only resolves
    inside an actual Streamlit runtime — when this provider is invoked
    from FastAPI or a plain script (no streamlit ScriptRunContext), the
    streamlit path silently fails. Without the tomllib fallback every
    FastAPI-side caller would have to maintain its own key loader."""
    # 1. env var (works everywhere; the canonical CI / prod path)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # 2. streamlit context (legacy / streamlit-app callers)
    try:
        import streamlit as st
        v = st.secrets.get("ANTHROPIC_API_KEY", "")
        if v:
            return v
    except Exception:
        pass
    # 3. direct tomllib read of .streamlit/secrets.toml
    try:
        import pathlib as _pl
        secrets_path = _pl.Path(__file__).resolve().parents[3] / ".streamlit" / "secrets.toml"
        if secrets_path.is_file():
            try:
                import tomllib  # py>=3.11
            except ImportError:
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            v = data.get("ANTHROPIC_API_KEY") or ""
            if v:
                return v
    except Exception:
        pass
    return ""


def call_anthropic(
    *,
    model:        str,
    system:       str,
    user:         Optional[str] = None,
    messages:     Optional[list[dict]] = None,
    tools:        Optional[list[dict]] = None,
    max_tokens:   int = 1024,
    cache_system: bool = True,
    thinking:     bool = False,
    effort:       str = "low",       # low | medium | high (Sonnet 4.6 / Opus only)
) -> _RawCallResult:
    """Call Anthropic Messages API with our defaults.

    Args:
      model:        full model id (e.g. "claude-sonnet-4-6", "claude-haiku-4-5")
      system:       system prompt (cached if cache_system=True and tokens ≥ 1024)
      user:         user message
      tools:        list of tool definitions (Anthropic schema) — caller's
                    responsibility to format correctly
      max_tokens:   completion cap (Haiku 4.5 max 64K, Sonnet 4.6 max 64K,
                    streaming required > ~16K)
      cache_system: wrap system in ephemeral cache_control (default True;
                    only kicks in if system ≥ 1024 tokens for Haiku, ≥ 2048
                    for Sonnet 4.6)
      user:         single user message (single-turn convenience; ignored
                    if `messages` provided)
      messages:     full conversation history in Anthropic format
                    [{role, content}, ...] — use for multi-turn / tool-use
                    round-trips. Exactly one of `user` or `messages`
                    must be supplied.
      thinking:     enable adaptive thinking (default OFF — costs tokens
                    and latency; not needed for narrator-style short output)
      effort:       output_config effort level (low | medium | high) —
                    Sonnet 4.6 default is "high" but we lower to "low" for
                    narrator workloads to save tokens; tool-using agents
                    should pass "medium" or "high".
    """
    if messages is None and user is None:
        raise ValueError("must supply either `user` (single-turn) or `messages`")

    import anthropic

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Anthropic API key not found. Add ANTHROPIC_API_KEY to "
            ".streamlit/secrets.toml or set as env var."
        )

    # Resilience config (per claude-api skill guidance + deferred backlog
    # #1 hardening 2026-05-19):
    #   max_retries=4    — SDK auto-retries on 429 (rate limit) / 408 / 409
    #                      / 5xx with exponential backoff. Default is 2;
    #                      we raise to 4 to absorb provider-side transient
    #                      blips without bubbling a red error to the user
    #                      mid-conversation. 4 retries × exponential
    #                      backoff caps at ~30s tail, acceptable for
    #                      interactive chat.
    #   timeout=60.0     — per-request wall-clock cap. SDK default is
    #                      10min which is way too long for chat UX
    #                      (user sees indefinite spinner). 60s covers
    #                      worst-case Sonnet 4.6 tool-using turn.
    client = anthropic.Anthropic(
        api_key      = api_key,
        max_retries  = 4,
        # 2026-06-14: bumped 60->120 after synthesis with Phase B belief
        # context started consistently timing out at 60s. Synthesis is
        # a background cron call not interactive chat — 120s wall-clock
        # is fine. Tool-use turns with belief layer + anchor library +
        # doctrine snippets together can push Sonnet 4.6 into 60-90s
        # latency under provider load.
        timeout      = 120.0,
    )

    # System prompt: pass as list of blocks so we can attach cache_control.
    # Plain string also works but loses the cache attachment.
    if cache_system and system:
        system_param: Any = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_param = system

    # messages takes priority; user is single-turn convenience
    msgs = messages if messages is not None else [{"role": "user", "content": user}]

    kwargs: dict[str, Any] = {
        "model":      model,
        "max_tokens": max_tokens,
        "system":     system_param,
        "messages":   msgs,
    }
    if tools:
        kwargs["tools"] = tools
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    # Effort only applies to Sonnet 4.6 + Opus 4.5+. Skip for Haiku to avoid 400.
    if effort and not model.startswith("claude-haiku"):
        kwargs["output_config"] = {"effort": effort}

    start = time.time()
    try:
        response = client.messages.create(**kwargs)
    except anthropic.RateLimitError as exc:
        # SDK already retried max_retries times before raising. Make
        # the user-facing surface informative rather than the raw
        # SDK message.
        logger.exception(
            "Anthropic rate-limit exhausted after %d retries: %s",
            4, exc,
        )
        raise RuntimeError(
            "Anthropic API rate-limit exhausted after 4 retries — "
            "try again in 30-60s, or check your tier limits at "
            "console.anthropic.com/settings/limits."
        ) from exc
    except anthropic.APITimeoutError as exc:
        logger.exception("Anthropic API timeout (60s): %s", exc)
        raise RuntimeError(
            "Anthropic API timeout (60s wall-clock) — provider slow "
            "or transient network issue. Retry the question."
        ) from exc
    except anthropic.APIConnectionError as exc:
        logger.exception("Anthropic API connection error: %s", exc)
        raise RuntimeError(
            "Cannot reach Anthropic API (network / DNS issue). "
            "Check connectivity and retry."
        ) from exc
    except anthropic.APIStatusError as exc:
        # Catch-all for other 4xx/5xx after SDK retries
        logger.exception(
            "Anthropic API error (status=%s): %s",
            getattr(exc, "status_code", "?"), exc,
        )
        raise
    elapsed_ms = int((time.time() - start) * 1000)

    # Parse content blocks
    text_chunks: list[str] = []
    tool_calls: list[dict] = []
    for block in response.content:
        if block.type == "text":
            text_chunks.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id":    block.id,
                "name":  block.name,
                "input": block.input,
            })

    usage = response.usage
    return _RawCallResult(
        text               = "\n".join(text_chunks),
        tool_calls         = tool_calls,
        stop_reason        = response.stop_reason or "",
        model              = response.model,
        input_tokens       = int(usage.input_tokens or 0),
        output_tokens      = int(usage.output_tokens or 0),
        cache_read_tokens  = int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        latency_ms         = elapsed_ms,
        raw_usage          = {
            "input_tokens":                int(usage.input_tokens or 0),
            "output_tokens":               int(usage.output_tokens or 0),
            "cache_read_input_tokens":     int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        },
    )
