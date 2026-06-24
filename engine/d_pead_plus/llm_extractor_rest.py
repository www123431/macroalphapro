"""
engine/d_pead_plus/llm_extractor_rest.py — Day-2 REST API rewrite.

Day-1 (2026-05-13) discovered: google-genai SDK in Vertex ADC mode stalls
indefinitely after N calls due to gRPC connection state degradation.
Python threading cannot truly cancel native gRPC I/O.

Day-2 (2026-05-14) FIX: direct httpx POST to Vertex AI REST endpoint,
bypassing google-genai SDK entirely. httpx timeout is network-level
(reliable, no daemon thread leaks).

Hash-locked LOCKS preserved from llm_extractor.py:
  - Model: gemini-2.5-flash
  - Temperature: 0.0, top_p: 1.0
  - Response schema: 5 features (tone/forward_confidence/macro_headwind/
    evasion/linguistic_complexity)
  - Prompt template hash: f01e18fbf998ec19 (unchanged — schema swap from
    SDK to REST does NOT change PROMPT_HASH because all hash inputs
    preserved: prompt + model + temperature + top_p + schema)

DOCTRINE: this is FEATURE EXTRACTION layer. Decision-layer modules
(feature_combiner, backtest, verdict) MUST NOT import this module.
Enforced by engine.d_pead_plus.doctrine.
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

# Import locked constants from original SDK-based module for consistency
from engine.d_pead_plus.llm_extractor import (
    LLM_MODEL_LOCKED,
    LLM_TEMPERATURE_LOCKED,
    LLM_TOP_P_LOCKED,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_PROVIDER_LOCKED,
    LLM_RESPONSE_SCHEMA,
    SYSTEM_PROMPT_LOCKED,
    USER_TEMPLATE_LOCKED,
    PROMPT_HASH_LOCKED,
    PROMPT_HASH_SHORT_LOCKED,
    LLMExtractionRecord,
    FEATURES_PARQUET,
    CACHE_DIR,
    save_extractions,
    load_existing_extractions,
    _compute_cost_usd,
    _log_to_cost_ledger,
)

logger = logging.getLogger(__name__)

# Vertex REST endpoint config
VERTEX_LOCATION_LOCKED: str = "us-central1"
VERTEX_API_URL_TEMPLATE: str = (
    "https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
    "/locations/{location}/publishers/google/models/{model}:generateContent"
)
ADC_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/cloud-platform",)

# httpx timeout (per-call, network-level — reliable)
HTTP_TIMEOUT_SECONDS: float = 120.0
HTTP_CONNECT_TIMEOUT_SECONDS: float = 10.0


@dataclass
class _AuthState:
    """Cached Vertex ADC token + project."""
    token:        str
    project_id:   str
    expires_at:   datetime.datetime
    credentials:  object


def _get_auth_state(force_refresh: bool = False, cached: list = []) -> _AuthState:
    """Get/refresh Vertex ADC token. Cached in module-level list to survive
    between calls. Refreshes when within 5 min of expiry."""
    now = datetime.datetime.utcnow()
    if cached and not force_refresh:
        state = cached[0]
        # Refresh if within 5 min of expiry (token TTL typically 60 min)
        if (state.expires_at - now).total_seconds() > 300:
            return state

    from google.auth import default as default_auth
    from google.auth.transport.requests import Request as AuthRequest

    credentials, project_id = default_auth(scopes=list(ADC_SCOPES))
    credentials.refresh(AuthRequest())

    expires_at = credentials.expiry or (now + datetime.timedelta(minutes=55))
    state = _AuthState(
        token=credentials.token,
        project_id=project_id,
        expires_at=expires_at,
        credentials=credentials,
    )
    cached.clear()
    cached.append(state)
    logger.info("Vertex ADC refreshed; project=%s expires=%s", project_id, expires_at)
    return state


def _build_request_body(company_name: str, call_date: datetime.date, transcript_text: str) -> dict:
    """Build the JSON body for Vertex AI generateContent REST call."""
    user_prompt = USER_TEMPLATE_LOCKED.format(
        company_name=company_name,
        call_date=call_date.isoformat(),
        transcript_text=transcript_text,
    )
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT_LOCKED}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature":       LLM_TEMPERATURE_LOCKED,
            "topP":              LLM_TOP_P_LOCKED,
            "maxOutputTokens":   LLM_MAX_OUTPUT_TOKENS,
            "responseMimeType":  "application/json",
            "responseSchema":    LLM_RESPONSE_SCHEMA,
            # Matches SDK llm_extractor.py:178 (thinking_budget=0). Gemini 2.5 Flash
            # 'thinking' otherwise consumes the entire 500-token budget before output.
            "thinkingConfig":    {"thinkingBudget": 0},
        },
    }


def _call_gemini_rest(
    company_name:    str,
    call_date:       datetime.date,
    transcript_text: str,
    client:          httpx.Client,
) -> dict:
    """One Gemini REST API call. Returns parsed JSON + token usage.

    Raises:
      httpx.TimeoutException  — wall-clock timeout (reliable, network-level)
      httpx.HTTPStatusError   — 4xx/5xx response
      ValueError              — response JSON missing required schema fields
    """
    auth = _get_auth_state()
    url = VERTEX_API_URL_TEMPLATE.format(
        location=VERTEX_LOCATION_LOCKED,
        project=auth.project_id,
        model=LLM_MODEL_LOCKED,
    )
    body = _build_request_body(company_name, call_date, transcript_text)
    headers = {
        "Authorization": f"Bearer {auth.token}",
        "Content-Type":  "application/json",
    }

    resp = client.post(url, json=body, headers=headers)

    # 401 → refresh token + retry once
    if resp.status_code == 401:
        logger.warning("Vertex 401; refreshing token + retrying")
        auth = _get_auth_state(force_refresh=True)
        headers["Authorization"] = f"Bearer {auth.token}"
        resp = client.post(url, json=body, headers=headers)

    resp.raise_for_status()
    payload = resp.json()

    # Extract response text from Vertex response format
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError(f"Vertex response has no candidates: {payload}")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise ValueError(f"Vertex response candidate has no parts: {candidates[0]}")
    text = parts[0].get("text", "")
    if not text:
        raise ValueError(f"Vertex response part has no text: {parts[0]}")

    parsed = json.loads(text)
    for field in ("tone_score", "forward_confidence", "macro_headwind_flag",
                   "evasion_score", "linguistic_complexity"):
        if field not in parsed:
            raise ValueError(f"LLM response missing required field {field}: {parsed}")

    usage = payload.get("usageMetadata", {})
    in_tok  = int(usage.get("promptTokenCount", 0) or 0)
    out_tok = int(usage.get("candidatesTokenCount", 0) or 0) + int(usage.get("thoughtsTokenCount", 0) or 0)

    return {
        "parsed":        parsed,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
    }


def run_extraction_rest(
    transcripts_index_df:  pd.DataFrame,
    transcripts_text_df:   pd.DataFrame,
    *,
    max_transcripts:        Optional[int] = None,
    skip_already_extracted: bool          = True,
    sleep_between_calls_s:  float         = 0.1,
    flush_every_n:          int           = 10,
    timeout_seconds:        float         = HTTP_TIMEOUT_SECONDS,
) -> list[LLMExtractionRecord]:
    """REST API version of run_extraction. Same interface as SDK-based.

    Reliability improvements over SDK version:
      - httpx network-level timeout (no daemon thread leaks)
      - No persistent gRPC connection state degradation
      - 401 token refresh + retry (transparent)
      - Idempotent (skips already-extracted via parquet cache)
      - Periodic flush every N records (kill-safe)
    """
    idx_df  = transcripts_index_df.copy()
    text_df = transcripts_text_df.copy()
    text_df["transcript_id"] = text_df["transcript_id"].astype(int)
    idx_df["transcript_id"]  = idx_df["transcript_id"].astype(int)

    merged = idx_df.merge(
        text_df[["transcript_id", "full_text", "total_chars"]],
        on="transcript_id", how="inner",
    )

    if skip_already_extracted:
        existing = load_existing_extractions()
        if not existing.empty:
            existing_ids = set(existing["transcript_id"].astype(int).tolist())
            n_before = len(merged)
            merged = merged[~merged["transcript_id"].isin(existing_ids)]
            logger.info("Idempotency: skipping %d already-extracted (remaining %d)",
                        n_before - len(merged), len(merged))

    if max_transcripts is not None:
        merged = merged.head(max_transcripts)

    n_total = len(merged)
    if n_total == 0:
        logger.info("run_extraction_rest: nothing to extract.")
        return []

    logger.info("Starting %d LLM extractions via REST (model=%s, prompt_hash=%s, "
                "flush_every=%d, timeout=%ds)",
                n_total, LLM_MODEL_LOCKED, PROMPT_HASH_SHORT_LOCKED, flush_every_n,
                int(timeout_seconds))

    records: list[LLMExtractionRecord] = []
    pending_unflushed: list[LLMExtractionRecord] = []

    # Single shared httpx client (HTTP keep-alive but per-call timeout reliable)
    timeout_config = httpx.Timeout(timeout_seconds, connect=HTTP_CONNECT_TIMEOUT_SECONDS)
    transport = httpx.HTTPTransport(retries=2)

    try:
        with httpx.Client(timeout=timeout_config, transport=transport) as client:
            for i, (_, row) in enumerate(merged.iterrows(), 1):
                transcript_id   = int(row["transcript_id"])
                permno          = int(row["permno"])
                rdq_d           = pd.to_datetime(row["rdq"]).date()
                company_name    = str(row["company_name"])
                call_date       = pd.to_datetime(row["call_date"]).date()
                transcript_text = str(row["full_text"])

                t0 = time.time()
                try:
                    result = _call_gemini_rest(company_name, call_date, transcript_text, client)
                except httpx.TimeoutException:
                    logger.warning("  %d/%d transcript_id=%d: TIMEOUT (skipped)",
                                    i, n_total, transcript_id)
                    continue
                except Exception as exc:
                    logger.warning("  %d/%d transcript_id=%d: %s (skipped)",
                                    i, n_total, transcript_id, exc)
                    continue
                latency_ms = int((time.time() - t0) * 1000)

                parsed  = result["parsed"]
                in_tok  = result["input_tokens"]
                out_tok = result["output_tokens"]
                cost    = _compute_cost_usd(in_tok, out_tok)
                _log_to_cost_ledger(in_tok, out_tok, cost, latency_ms)

                rec = LLMExtractionRecord(
                    transcript_id          = transcript_id,
                    permno                 = permno,
                    rdq                    = rdq_d,
                    company_name           = company_name,
                    call_date              = call_date,
                    tone_score             = float(parsed["tone_score"]),
                    forward_confidence     = float(parsed["forward_confidence"]),
                    macro_headwind_flag    = bool(parsed["macro_headwind_flag"]),
                    evasion_score          = float(parsed["evasion_score"]),
                    linguistic_complexity  = float(parsed["linguistic_complexity"]),
                    prompt_hash            = PROMPT_HASH_LOCKED,
                    model_version          = LLM_MODEL_LOCKED,
                    extract_ts_utc         = datetime.datetime.utcnow(),
                    input_tokens           = in_tok,
                    output_tokens          = out_tok,
                    cost_usd               = cost,
                )
                records.append(rec)
                pending_unflushed.append(rec)

                if i % flush_every_n == 0 or i == n_total:
                    logger.info("  Progress: %d/%d (last cost $%.4f, cum $%.4f, latency=%dms)",
                                i, n_total, cost, sum(r.cost_usd for r in records), latency_ms)
                    save_extractions(pending_unflushed)
                    pending_unflushed = []

                if sleep_between_calls_s > 0:
                    time.sleep(sleep_between_calls_s)
    finally:
        if pending_unflushed:
            logger.info("Final flush: %d unflushed records", len(pending_unflushed))
            try:
                save_extractions(pending_unflushed)
            except Exception:
                logger.exception("Final flush failed — investigate")

    logger.info("REST extraction complete: %d records, total cost $%.4f",
                len(records), sum(r.cost_usd for r in records))
    return records


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=5, help="Max transcripts (smoke test)")
    args = p.parse_args()

    from engine.d_pead_plus.transcripts_loader import load_cached_transcripts
    idx, txt = load_cached_transcripts()
    print(f"Cached: index={len(idx)}, text={len(txt)}")
    records = run_extraction_rest(idx, txt, max_transcripts=args.smoke)
    print(f"\nDone: {len(records)} records extracted; "
          f"total cost ${sum(r.cost_usd for r in records):.4f}")
