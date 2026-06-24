"""
engine/d_pead_plus/llm_extractor.py — Gemini 2.5 Flash earnings call feature extraction.

Spec id=74 §2.5 LOCK: Hash-locked prompt template + model + temperature.
                     5 structured features per transcript:
                       tone_score (float -1..+1)
                       forward_confidence (float 0..1)
                       macro_headwind_flag (bool)
                       evasion_score (float 0..1)
                       linguistic_complexity (float 0..1)

LLM ONLY extracts features. NEVER sees returns / portfolio / future data /
other firms' features. Decision layer (feature_combiner / backtest / verdict)
contains zero LLM calls.

DOCTRINE: this module IS the LLM call site. Decision layer modules must
NOT import this module (enforced by engine.d_pead_plus.doctrine).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Spec id=74 §2.5 LOCKED constants (changing requires NEW spec)
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL_LOCKED:       str   = "gemini-2.5-flash"
LLM_TEMPERATURE_LOCKED: float = 0.0
LLM_TOP_P_LOCKED:       float = 1.0
LLM_MAX_OUTPUT_TOKENS:  int   = 500
LLM_PROVIDER_LOCKED:    str   = "vertex"

# Output JSON schema (structured response_schema for Gemini)
LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tone_score":            {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "forward_confidence":    {"type": "number", "minimum": 0.0,  "maximum": 1.0},
        "macro_headwind_flag":   {"type": "boolean"},
        "evasion_score":         {"type": "number", "minimum": 0.0,  "maximum": 1.0},
        "linguistic_complexity": {"type": "number", "minimum": 0.0,  "maximum": 1.0},
    },
    "required": [
        "tone_score",
        "forward_confidence",
        "macro_headwind_flag",
        "evasion_score",
        "linguistic_complexity",
    ],
    "propertyOrdering": [
        "tone_score",
        "forward_confidence",
        "macro_headwind_flag",
        "evasion_score",
        "linguistic_complexity",
    ],
}

# Hash-locked prompt template (per spec §2.5)
SYSTEM_PROMPT_LOCKED: str = (
    "You are a financial analyst extracting structured features from an "
    "earnings call transcript. Output ONLY valid JSON matching the schema. "
    "Do not include any other text."
)

USER_TEMPLATE_LOCKED: str = """Below is the full transcript of {company_name}'s earnings call dated {call_date}.
Extract the following 5 features as a JSON object:

{{
  "tone_score": <float between -1 and +1, where -1 is very negative,
                 0 is neutral, +1 is very positive overall tone>,
  "forward_confidence": <float between 0 and 1, where 1 is highest confidence
                         in forward guidance statements>,
  "macro_headwind_flag": <true if macro headwinds explicitly cited as material
                          concern, else false>,
  "evasion_score": <float between 0 and 1, where 1 is highest detected evasion
                    in management answers (per Larcker-Zakolyukina 2012 deception
                    linguistic markers: hesitancy, deflection, hedging)>,
  "linguistic_complexity": <float between 0 and 1, where 1 is highest Gunning
                            Fog Index proxy (complex sentence structure, jargon
                            density, abstract terminology)>
}}

CRITICAL RULES:
- Use ONLY information present in the transcript itself.
- Do NOT reference any external context about this company.
- Do NOT use any knowledge about this company's stock performance.
- Do NOT consider the company's future or past returns in any way.
- Output JSON only, no explanation.

TRANSCRIPT:
{transcript_text}"""


def _compute_prompt_hash() -> str:
    """SHA256 of (system + template + model + temp + top_p + schema)."""
    canonical = (
        f"system={SYSTEM_PROMPT_LOCKED}\n"
        f"template={USER_TEMPLATE_LOCKED}\n"
        f"model={LLM_MODEL_LOCKED}\n"
        f"temperature={LLM_TEMPERATURE_LOCKED}\n"
        f"top_p={LLM_TOP_P_LOCKED}\n"
        f"max_output={LLM_MAX_OUTPUT_TOKENS}\n"
        f"schema={json.dumps(LLM_RESPONSE_SCHEMA, sort_keys=True)}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PROMPT_HASH_LOCKED:       str = _compute_prompt_hash()
PROMPT_HASH_SHORT_LOCKED: str = PROMPT_HASH_LOCKED[:16]

# Persistence
CACHE_DIR = Path("data/d_pead_plus")
FEATURES_PARQUET = CACHE_DIR / "_llm_extracted_features.parquet"


@dataclass(frozen=True)
class LLMExtractionRecord:
    """One LLM extraction record per transcript."""
    transcript_id:          int
    permno:                 int
    rdq:                    datetime.date
    company_name:           str
    call_date:              datetime.date
    tone_score:             float
    forward_confidence:     float
    macro_headwind_flag:    bool
    evasion_score:          float
    linguistic_complexity:  float
    prompt_hash:            str
    model_version:          str
    extract_ts_utc:         datetime.datetime
    input_tokens:           int
    output_tokens:          int
    cost_usd:               float


def _call_gemini_extract(
    company_name:     str,
    call_date:        datetime.date,
    transcript_text:  str,
    timeout_seconds:  int = 120,
) -> dict:
    """Single LLM call with hash-locked prompt; returns parsed JSON + meta.

    Uses engine.key_pool.get_model() per project pattern (Vertex ADC + retries).

    TIMEOUT: hard wall-clock cap via ThreadPoolExecutor.
    Gemini API occasionally stalls indefinitely; without timeout the whole
    extraction loop hangs. 120s is generous (typical call 3-6s); anything
    longer indicates API-side issue and we should skip that record.
    """
    import concurrent.futures

    user_prompt = USER_TEMPLATE_LOCKED.format(
        company_name=company_name,
        call_date=call_date.isoformat(),
        transcript_text=transcript_text,
    )
    full_prompt = f"{SYSTEM_PROMPT_LOCKED}\n\n{user_prompt}"

    from engine.key_pool import get_pool
    pool = get_pool()
    model = pool.get_model(
        model_name      = LLM_MODEL_LOCKED,
        response_schema = LLM_RESPONSE_SCHEMA,
        temperature     = LLM_TEMPERATURE_LOCKED,
        thinking_budget = 0,
    )

    # Wall-clock timeout via daemon thread (Windows-compatible).
    # ThreadPoolExecutor.shutdown() blocks for workers, which defeats timeout;
    # daemon thread is freed naturally on process exit and join(timeout=...)
    # returns control to main thread even if worker is still running.
    import threading
    result_holder: list = []
    exc_holder: list = []

    def _do_call():
        try:
            result_holder.append(model.generate_content(full_prompt))
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_do_call, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        # Thread still running — give up; daemon thread will be killed on
        # process exit (or eventually return; we don't wait).
        logger.warning("Gemini call exceeded %ds timeout for %s; skipping (worker thread daemon-leaked)",
                        timeout_seconds, company_name[:40])
        raise TimeoutError(f"Gemini call >{timeout_seconds}s")
    if exc_holder:
        raise exc_holder[0]
    if not result_holder:
        raise RuntimeError("Gemini worker thread finished but no result/exception")
    resp = result_holder[0]
    pool.report_success(has_content=True)

    text  = getattr(resp, "text", None) or str(resp)
    usage = getattr(resp, "usage_metadata", None)
    in_tok  = int(getattr(usage, "prompt_token_count",     0) or 0)
    out_tok = int((getattr(usage, "candidates_token_count", 0) or 0)
                  + (getattr(usage, "thoughts_token_count", 0) or 0))

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.error("llm_extractor JSON parse failed; first 200: %r", text[:200])
        raise

    # Validate schema
    for field in ("tone_score", "forward_confidence", "macro_headwind_flag",
                   "evasion_score", "linguistic_complexity"):
        if field not in parsed:
            raise ValueError(f"LLM response missing required field: {field}")

    return {
        "parsed":        parsed,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
    }


def _compute_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Gemini 2.5 Flash on VERTEX AI pricing 2026-05 (per 1M tokens):
       input: $0.30, output: $2.50

    NOTE: This is the VERTEX AI tier, NOT Google AI Studio direct
    ($0.075 / $0.30 — 4-8x cheaper). Production calls go through Vertex ADC
    endpoint, so Vertex pricing applies.

    Empirical verification 2026-05-13/14 Sprint I extraction:
      109M input + 688K output tokens
      → Vertex billed $34.46  (matches GCP console)
      → AI Studio would have billed $8.39  (4.1x cheaper, but not the API used)
    """
    return (input_tokens / 1_000_000.0) * 0.30 + (output_tokens / 1_000_000.0) * 2.50


def _log_to_cost_ledger(in_tok: int, out_tok: int, cost: float, latency_ms: int = 0) -> None:
    """Append cost entry to engine.llm_cost_ledger (correct signature)."""
    try:
        from engine.llm_cost_ledger import record_call
        record_call(
            agent_id          = "d_pead_plus_llm_extractor",
            provider          = "gemini",
            model             = LLM_MODEL_LOCKED,
            prompt_tokens     = in_tok,
            completion_tokens = out_tok,
            cost_usd          = cost,
            latency_ms        = latency_ms,
            scope             = f"prompt_hash={PROMPT_HASH_SHORT_LOCKED}",
        )
    except Exception:
        logger.exception("failed to log to cost ledger (continuing; not blocking extraction)")


def load_existing_extractions() -> pd.DataFrame:
    """Load cached extraction parquet (idempotency support)."""
    if not FEATURES_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(FEATURES_PARQUET)


def save_extractions(records: list[LLMExtractionRecord]) -> None:
    """Append + dedupe by transcript_id; preserve prior runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([{
        "transcript_id":          r.transcript_id,
        "permno":                 r.permno,
        "rdq":                    r.rdq,
        "company_name":           r.company_name,
        "call_date":              r.call_date,
        "tone_score":             r.tone_score,
        "forward_confidence":     r.forward_confidence,
        "macro_headwind_flag":    r.macro_headwind_flag,
        "evasion_score":          r.evasion_score,
        "linguistic_complexity":  r.linguistic_complexity,
        "prompt_hash":            r.prompt_hash,
        "model_version":          r.model_version,
        "extract_ts_utc":         r.extract_ts_utc,
        "input_tokens":           r.input_tokens,
        "output_tokens":          r.output_tokens,
        "cost_usd":               r.cost_usd,
    } for r in records])
    if FEATURES_PARQUET.exists():
        df_existing = pd.read_parquet(FEATURES_PARQUET)
        df = pd.concat([df_existing, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["transcript_id"], keep="last")
    else:
        df = df_new
    df.to_parquet(FEATURES_PARQUET)
    logger.info("Saved %d new records; total cached: %d", len(df_new), len(df))


def run_extraction(
    transcripts_index_df:  pd.DataFrame,
    transcripts_text_df:   pd.DataFrame,
    *,
    max_transcripts:       Optional[int] = None,
    skip_already_extracted: bool = True,
    sleep_between_calls_s: float = 0.1,   # Vertex per-call latency already ~3-6s; minimal sleep needed
    flush_every_n:         int   = 10,    # save partial progress frequently for kill-safety
) -> list[LLMExtractionRecord]:
    """Run LLM extraction for given transcripts.

    Idempotent: skips already-extracted transcript_ids if skip_already_extracted=True.
    Rate-limited by sleep_between_calls_s (defends against Vertex RPM quota).

    Args:
        transcripts_index_df: from transcripts_loader.fetch_transcript_index
        transcripts_text_df:   from transcripts_loader.fetch_transcript_text
        max_transcripts:       cap on N for dev / dry runs
        skip_already_extracted: skip transcript_ids in existing cache
        sleep_between_calls_s: pause between LLM calls

    Returns: list of new LLMExtractionRecord
    """
    idx_df  = transcripts_index_df.copy()
    text_df = transcripts_text_df.copy()
    text_df["transcript_id"] = text_df["transcript_id"].astype(int)
    idx_df["transcript_id"]  = idx_df["transcript_id"].astype(int)

    # Join index + text
    merged = idx_df.merge(text_df[["transcript_id", "full_text", "total_chars"]],
                          on="transcript_id", how="inner")

    # Idempotency filter
    if skip_already_extracted:
        existing = load_existing_extractions()
        if not existing.empty:
            existing_ids = set(existing["transcript_id"].astype(int).tolist())
            n_before = len(merged)
            merged = merged[~merged["transcript_id"].isin(existing_ids)]
            logger.info("Idempotency: skipping %d already-extracted transcripts (remaining %d)",
                        n_before - len(merged), len(merged))

    if max_transcripts is not None:
        merged = merged.head(max_transcripts)

    records: list[LLMExtractionRecord] = []
    n_total = len(merged)
    if n_total == 0:
        logger.info("run_extraction: nothing to extract.")
        return records

    logger.info("Starting %d LLM extractions (model=%s, prompt_hash=%s, flush_every=%d)",
                n_total, LLM_MODEL_LOCKED, PROMPT_HASH_SHORT_LOCKED, flush_every_n)

    pending_unflushed: list[LLMExtractionRecord] = []
    try:
        for i, (_, row) in enumerate(merged.iterrows(), 1):
            transcript_id = int(row["transcript_id"])
            permno        = int(row["permno"])
            rdq_d         = pd.to_datetime(row["rdq"]).date()
            company_name  = str(row["company_name"])
            call_date     = pd.to_datetime(row["call_date"]).date()
            transcript_text = str(row["full_text"])

            t0 = time.time()
            try:
                result = _call_gemini_extract(company_name, call_date, transcript_text)
            except Exception as exc:
                logger.exception("LLM extraction failed transcript_id=%d: %s", transcript_id, exc)
                continue
            latency_ms = int((time.time() - t0) * 1000)

            parsed = result["parsed"]
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
                # Periodic flush of accumulated unflushed records
                save_extractions(pending_unflushed)
                pending_unflushed = []

            if sleep_between_calls_s > 0:
                time.sleep(sleep_between_calls_s)
    finally:
        # Always flush remaining records on exit (kill-safety)
        if pending_unflushed:
            logger.info("Final flush: %d unflushed records", len(pending_unflushed))
            try:
                save_extractions(pending_unflushed)
            except Exception:
                logger.exception("Final flush failed (records lost — investigate)")

    logger.info("Extraction complete: %d records, total cost $%.4f",
                len(records), sum(r.cost_usd for r in records))
    return records


def get_locked_constants() -> dict:
    """Public access to locked constants (for spec audit + reproducibility check)."""
    return {
        "LLM_MODEL_LOCKED":       LLM_MODEL_LOCKED,
        "LLM_TEMPERATURE_LOCKED": LLM_TEMPERATURE_LOCKED,
        "LLM_TOP_P_LOCKED":       LLM_TOP_P_LOCKED,
        "LLM_MAX_OUTPUT_TOKENS":  LLM_MAX_OUTPUT_TOKENS,
        "LLM_PROVIDER_LOCKED":    LLM_PROVIDER_LOCKED,
        "PROMPT_HASH_LOCKED":     PROMPT_HASH_LOCKED,
        "PROMPT_HASH_SHORT":      PROMPT_HASH_SHORT_LOCKED,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== D-PEAD-Plus LLM Extractor — Locked Constants ===")
    for k, v in get_locked_constants().items():
        if k == "PROMPT_HASH_LOCKED":
            print(f"  {k:<25} {v}")
        else:
            print(f"  {k:<25} {v}")
    print()
    print("Module loads clean. Use run_extraction(idx_df, text_df) to invoke.")
