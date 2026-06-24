"""
engine/llm/providers/deepseek_provider.py — DeepSeek API adapter (OpenAI-compatible).

Implementation per 2026-05-19 verify-then-build: live ping confirmed
DeepSeek /v1/chat/completions is OpenAI-format. Adapter converts
between Anthropic-shape (our internal contract) and OpenAI-shape (wire):

  Tools:
    Anthropic   {name, description, input_schema}
    OpenAI      {type:"function", function:{name, description, parameters}}

  Assistant tool_use:
    Anthropic   content block {type:"tool_use", id, name, input:dict}
    OpenAI      message.tool_calls[].{id, function:{name, arguments:json_str}}

  User tool_result:
    Anthropic   content block {type:"tool_result", tool_use_id, content, is_error?}
    OpenAI      message {role:"tool", tool_call_id, content}
                (no is_error flag — encode in content text)

Key facts (verified 2026-05-19 live):
  - Endpoint: POST https://api.deepseek.com/v1/chat/completions
  - Auth: Bearer token from DEEPSEEK_API_KEY (flat secrets.toml key)
  - Models available: deepseek-v4-flash / deepseek-v4-pro
  - Prompt caching IS supported (prompt_cache_hit_tokens in usage)
  - No SDK; raw httpx with manual retry/timeout (Anthropic-style)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import time
from typing import Any, Optional

import httpx

from engine.llm.providers.anthropic_provider import _RawCallResult

logger = logging.getLogger(__name__)


_BASE_URL = "https://api.deepseek.com/v1"
_TIMEOUT_SECONDS = 60.0
_MAX_RETRIES = 4


def _get_api_key() -> str:
    """Read DEEPSEEK_API_KEY from streamlit secrets (flat) or env."""
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
        # Legacy: try [DEEPSEEK] section API_KEY field
        ds_section = st.secrets.get("DEEPSEEK", {})
        if hasattr(ds_section, "get"):
            key = ds_section.get("API_KEY", "") or ds_section.get("api_key", "")
            if key:
                return key
    except Exception:
        pass
    return os.environ.get("DEEPSEEK_API_KEY", "")


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic → OpenAI schema/message converters
# ──────────────────────────────────────────────────────────────────────────────
def _anthropic_tools_to_openai(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """Convert Anthropic tool schemas to OpenAI function-call format."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t["input_schema"],
            },
        }
        for t in tools
    ]


def _anthropic_msgs_to_openai(
    system_text: str,
    messages: list[dict],
) -> list[dict]:
    """Convert Anthropic-format messages (incl content blocks) to OpenAI format.

    Anthropic shape:
      [{"role":"user", "content":"text"} | {"role":"user", "content":[blocks]}]
    OpenAI shape:
      [{"role":"system", "content":"..."}, {"role":"user", "content":"..."}, ...]
      Assistant tool calls go on message.tool_calls (separate from content).
      Tool results become {"role":"tool", "tool_call_id":..., "content":...}.
    """
    out: list[dict] = []
    if system_text:
        out.append({"role": "system", "content": system_text})

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # User message may carry tool_result blocks (post-tool-call response)
                # OR plain text blocks. Split:
                tool_results: list[dict] = []
                text_parts:   list[str]  = []
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "tool_result":
                        # Each tool_result becomes its own OpenAI {"role":"tool"} msg
                        tr_content = blk.get("content", "")
                        if isinstance(tr_content, list):
                            # Anthropic allows tool_result.content to be list of blocks
                            tr_content = "\n".join(
                                b.get("text", "") for b in tr_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        # If is_error, prepend marker so DeepSeek sees it clearly
                        if blk.get("is_error"):
                            tr_content = f"[TOOL_ERROR] {tr_content}"
                        tool_results.append({
                            "role":         "tool",
                            "tool_call_id": blk.get("tool_use_id", ""),
                            "content":      tr_content,
                        })
                    elif blk.get("type") == "text":
                        text_parts.append(blk.get("text", ""))
                if text_parts:
                    out.append({"role": "user", "content": "\n".join(text_parts)})
                out.extend(tool_results)

        elif role == "assistant":
            # Assistant may have text + tool_use blocks. OpenAI splits these:
            #   content = concatenated text
            #   tool_calls = array of {id, type:"function", function:{name,arguments}}
            text_parts: list[str]  = []
            tool_calls: list[dict] = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "text":
                        text_parts.append(blk.get("text", ""))
                    elif blk.get("type") == "tool_use":
                        tool_calls.append({
                            "id":   blk.get("id", ""),
                            "type": "function",
                            "function": {
                                "name":      blk.get("name", ""),
                                "arguments": json.dumps(blk.get("input", {}),
                                                        ensure_ascii=False),
                            },
                        })
            asst_msg: dict = {"role": "assistant"}
            if text_parts:
                asst_msg["content"] = "\n".join(text_parts)
            else:
                asst_msg["content"] = None  # OpenAI accepts None when tool_calls present
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            out.append(asst_msg)
    return out


def _openai_response_to_raw(
    resp_json: dict,
    elapsed_ms: int,
) -> _RawCallResult:
    """Convert OpenAI-shape /v1/chat/completions response to _RawCallResult."""
    choice    = resp_json["choices"][0]
    msg       = choice["message"]
    usage     = resp_json.get("usage", {})
    text      = msg.get("content") or ""

    # OpenAI tool_calls → Anthropic-shape tool_calls (id, name, input dict)
    tool_calls: list[dict] = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args_str = fn.get("arguments", "{}")
        try:
            input_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            input_dict = {"_unparsed_args": args_str}
        tool_calls.append({
            "id":    tc.get("id", ""),
            "name":  fn.get("name", ""),
            "input": input_dict,
        })

    # OpenAI finish_reason → Anthropic-shape stop_reason
    finish = choice.get("finish_reason", "")
    stop_reason_map = {
        "stop":        "end_turn",
        "tool_calls":  "tool_use",
        "length":      "max_tokens",
        "content_filter": "refusal",
    }
    stop_reason = stop_reason_map.get(finish, finish)

    # DeepSeek usage shape — has prompt_cache_hit_tokens / prompt_cache_miss_tokens
    cache_hit  = int(usage.get("prompt_cache_hit_tokens",  0) or 0)
    cache_miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    # input_tokens = prompt_tokens minus the cached portion (charged separately)
    # DeepSeek charges cache_hit at reduced rate; treat cache_hit as cache_read
    # and the un-cached remainder as input_tokens. Note: prompt_tokens INCLUDES
    # both hit and miss, so we subtract.
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    input_tokens  = max(0, prompt_tokens - cache_hit)

    return _RawCallResult(
        text               = text,
        tool_calls         = tool_calls,
        stop_reason        = stop_reason,
        model              = resp_json.get("model", ""),
        input_tokens       = input_tokens,
        output_tokens      = int(usage.get("completion_tokens", 0) or 0),
        cache_read_tokens  = cache_hit,
        cache_write_tokens = 0,    # DeepSeek doesn't charge for cache writes separately
        latency_ms         = elapsed_ms,
        raw_usage          = {
            "prompt_tokens":          prompt_tokens,
            "completion_tokens":      int(usage.get("completion_tokens", 0) or 0),
            "prompt_cache_hit":       cache_hit,
            "prompt_cache_miss":      cache_miss,
            "reasoning_tokens":       int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                or 0
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Manual retry/backoff (no SDK, so we implement it explicitly)
# ──────────────────────────────────────────────────────────────────────────────
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _post_with_retry(
    url:      str,
    headers:  dict,
    body:     dict,
    timeout:  float,
    max_retries: int,
) -> httpx.Response:
    """POST with exponential backoff on retryable status codes + transport errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code < 400:
                return resp
            if resp.status_code not in _RETRYABLE_STATUS or attempt == max_retries:
                return resp   # caller inspects + raises
            # Retryable error — fall through to backoff
            logger.warning(
                "DeepSeek %d on attempt %d/%d: %s — retrying",
                resp.status_code, attempt + 1, max_retries + 1, resp.text[:160],
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            logger.warning(
                "DeepSeek transport error on attempt %d/%d: %s — retrying",
                attempt + 1, max_retries + 1, exc,
            )

        # Exponential backoff with jitter
        delay = min(2.0 ** attempt + random.uniform(0, 0.5), 30.0)
        time.sleep(delay)

    # Should be unreachable, but guard
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("DeepSeek _post_with_retry: unreachable")


# ──────────────────────────────────────────────────────────────────────────────
# Main entrypoint — matches anthropic_provider.call_anthropic signature
# ──────────────────────────────────────────────────────────────────────────────
def call_deepseek(
    *,
    model:        str,
    system:       str,
    user:         Optional[str] = None,
    messages:     Optional[list[dict]] = None,
    tools:        Optional[list[dict]] = None,
    max_tokens:   int = 1024,
    cache_system: bool = True,   # ignored — DeepSeek caches automatically based on prefix
    thinking:     bool = False,  # V4 has built-in reasoning_tokens; toggle not exposed
    effort:       str = "low",   # not directly supported; ignored
) -> _RawCallResult:
    """Call DeepSeek /v1/chat/completions (OpenAI-compatible).

    Anthropic-shape signature for drop-in routing alongside call_anthropic.
    Converts schemas/messages internally; returns the shared _RawCallResult.
    """
    if messages is None and user is None:
        raise ValueError("must supply either `user` (single-turn) or `messages`")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "DeepSeek API key not found. Add DEEPSEEK_API_KEY to "
            ".streamlit/secrets.toml or set as env var."
        )

    # messages takes priority; user is single-turn convenience
    anth_msgs = messages if messages is not None else [{"role": "user", "content": user}]
    openai_msgs = _anthropic_msgs_to_openai(system, anth_msgs)
    openai_tools = _anthropic_tools_to_openai(tools)

    body: dict[str, Any] = {
        "model":      model,
        "messages":   openai_msgs,
        "max_tokens": max_tokens,
    }
    if openai_tools:
        body["tools"] = openai_tools

    start = time.time()
    try:
        resp = _post_with_retry(
            url         = f"{_BASE_URL}/chat/completions",
            headers     = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            body        = body,
            timeout     = _TIMEOUT_SECONDS,
            max_retries = _MAX_RETRIES,
        )
    except httpx.TimeoutException as exc:
        logger.exception("DeepSeek timeout after %d retries: %s", _MAX_RETRIES, exc)
        raise RuntimeError(
            f"DeepSeek API timeout ({_TIMEOUT_SECONDS}s wall-clock per attempt, "
            f"{_MAX_RETRIES} retries exhausted) — provider slow or down."
        ) from exc
    except httpx.NetworkError as exc:
        logger.exception("DeepSeek network error after retries: %s", exc)
        raise RuntimeError(
            "Cannot reach DeepSeek API (network / DNS issue). "
            "Check connectivity and retry."
        ) from exc

    elapsed_ms = int((time.time() - start) * 1000)

    if resp.status_code >= 400:
        body_preview = resp.text[:300]
        logger.error(
            "DeepSeek API status %d after retries: %s",
            resp.status_code, body_preview,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                "DeepSeek 401 Unauthorized — check DEEPSEEK_API_KEY validity at "
                "platform.deepseek.com/api_keys."
            )
        if resp.status_code == 429:
            raise RuntimeError(
                "DeepSeek 429 rate-limit exhausted after retries — wait 30-60s, "
                "or check account balance at platform.deepseek.com/usage."
            )
        raise RuntimeError(
            f"DeepSeek API error (status={resp.status_code}): {body_preview}"
        )

    try:
        resp_json = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"DeepSeek returned non-JSON response: {resp.text[:200]}"
        ) from exc

    return _openai_response_to_raw(resp_json, elapsed_ms)
