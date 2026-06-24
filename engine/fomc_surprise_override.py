"""
engine/fomc_surprise_override.py — FOMC-Day Surprise Override (Step 1 MVP).

Pre-registration: docs/spec_fomc_surprise_override.md
Spec id (engine.preregistration.SpecRegistry): 48
Spec hash (registered 2026-05-08): 036b2805f0d6 (12-char prefix; full SHA-1 in registry)

Purpose
-------
On FOMC press-statement release days (~8/yr, calendar-known), call Gemini 2.5
Flash exactly once to classify the statement's surprise level relative to
ex-ante market expectations. If LLM returns EXTREME_SURPRISE *and* the current
quant v3 multivariate regime is NOT risk-on, fire an emergency override that
multiplies REGIME_SCALE by 0.5 for 5 trading days.

Division of labor (escape wrapping)
-----------------------------------
LLM does what the quant statistical regime classifier *cannot* do — text-based
first-occurrence event classification. The quant statistical regime classifier
(v3 multivariate MSM) reasons over numerical features (yield_spread, VIX) on
historical sample distributions; for one-shot text events with shifting
language, statistical regime doesn't apply. LLM never enters cross-sectional
rank, signal generation, or position sizing (data plane). LLM only modifies
REGIME_SCALE meta-parameter (control plane), once per FOMC day, binary trigger.

Boundary invariant (project rule "0-LLM-in-evaluation")
-------------------------------------------------------
LLM is called only inside `_call_llm` for surprise classification. Verdict
computation paths (counterfactual P&L attribution, label-aggregation,
trigger-decision logic) are pure deterministic functions. See spec §3.7.

Forbidden modifications (HARKing R1-R4, per spec §六)
-----------------------------------------------------
- Multiplier bound [0.5, 1.0] (one-way defensive)
- Duration ≤ HARD_DURATION_CAP (10 trading days)
- Trigger AND-gate cannot be relaxed to OR (anti-LLM-单边)
- LLM cannot be invoked on non-FOMC days (calendar-locked dispatch)
- LLM cannot ingest post-FOMC market data (lookahead bias)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (spec §2.6, §2.7, §2.3, §4.4)
# ─────────────────────────────────────────────────────────────────────────────

HARD_OVERRIDE_MULTIPLIER: float = 0.5
"""Spec §2.6 — base_scale × this when override active. One-way defensive."""

HARD_DURATION_DAYS: int = 5
"""Spec §2.7 — trading days override remains active post-trigger."""

HARD_DURATION_CAP: int = 10
"""Spec §2.7 — absolute max even with future spec amendment (defense-in-depth)."""

HARD_MULTIPLIER_LOWER: float = 0.5
"""Spec §2.7 — clamp lower bound on effective scale relative to base."""

HARD_MULTIPLIER_UPPER: float = 1.0
"""Spec §2.7 — clamp upper bound; LLM cannot trigger aggressive scale-up."""

LLM_MODEL_VERSION: str = "gemini-2.5-flash"
"""Spec §2.3 — locked 90 days per anomaly_screener / W3 narrative tag pattern."""

LLM_TEMPERATURE: float = 0.0
LLM_THINKING_BUDGET: int = 1500
"""Classification task, not deep reasoning; cost control."""

MAX_RETRIES_ON_PARSE_FAILURE: int = 2
"""Spec §2.3 — fail-safe to NORMAL on 3rd failure."""

ANNUAL_BUDGET_USD: float = 1.0
"""Spec §2.3 + §4.4 — typical 8 calls × $0.05 = $0.40, $1 = 2.5× safety margin."""

PER_CALL_BUDGET_USD: float = 0.10
"""Spec §2.3 — per-call cap; 2× expected $0.05."""

# Cost pricing (Gemini 2.5 Flash, same as anomaly_llm_detector)
COST_PER_1M_INPUT_TOKENS: float = 0.30
COST_PER_1M_OUTPUT_TOKENS: float = 2.50

# Storage paths
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fomc_surprise_override"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = _DATA_DIR / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_COST_LEDGER_PATH = _DATA_DIR / "cost_ledger.json"
_DECISION_HISTORY_PATH = _DATA_DIR / "decision_history.parquet"
_COUNTERFACTUAL_PNL_PATH = _DATA_DIR / "counterfactual_pnl.parquet"
_OVERRIDE_STATE_PATH = _DATA_DIR / "override_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# FOMC calendar (reuse engine/decision_context.py existing list)
# ─────────────────────────────────────────────────────────────────────────────


def _get_fomc_dates() -> tuple[datetime.date, ...]:
    """
    Reuse the existing FOMC schedule from engine/decision_context.py.

    decision_context.py maintains _FOMC_DATES_2024_2026 as the canonical
    project-wide FOMC calendar (refresh annually). We import + use it
    directly rather than maintain a duplicate list (drift risk).

    Future extension to 2027+ via decision_context.py update + spec
    amendment (kind=clarification per spec §六).
    """
    from engine.decision_context import _FOMC_DATES_2024_2026
    return _FOMC_DATES_2024_2026


def is_fomc_day(d: datetime.date) -> bool:
    """Return True iff `d` is a scheduled FOMC press-statement release day."""
    if not isinstance(d, datetime.date):
        raise TypeError(f"is_fomc_day expected datetime.date, got {type(d)}")
    # Compare as date (not datetime); strip time component if needed
    if isinstance(d, datetime.datetime):
        d = d.date()
    return d in _get_fomc_dates()


def _get_prior_fomc_date(d: datetime.date) -> Optional[datetime.date]:
    """Return the most recent FOMC date strictly before `d`, or None if first."""
    fomc_dates = sorted(_get_fomc_dates())
    prior = [x for x in fomc_dates if x < d]
    return prior[-1] if prior else None


# ─────────────────────────────────────────────────────────────────────────────
# Trading-day utility (spec §2.6 §rule-9 N13)
# ─────────────────────────────────────────────────────────────────────────────


def _trading_days_elapsed(start: datetime.date, current: datetime.date) -> int:
    """
    Approximate trading days elapsed using pandas bdate_range.

    bdate_range counts business days (Mon-Fri) excluding weekends but NOT
    NYSE holidays. For HARD_DURATION_DAYS=5 (1 trading week), holiday-induced
    offset is at most 1 day per duration window (e.g., MLK day, July 4,
    Thanksgiving). Acceptable approximation for v1 short window.

    If HARD_DURATION_CAP extended (>10 days) in future amendments, switch
    to pandas_market_calendars XNYS calendar.
    """
    if start > current:
        return 0
    return max(0, len(pd.bdate_range(start, current)) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# JSON output schema (spec §2.4)
# ─────────────────────────────────────────────────────────────────────────────

LLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "surprise_label": {
            "type": "string",
            "enum": ["NORMAL", "MILD_SURPRISE", "EXTREME_SURPRISE"],
        },
        "direction": {
            "type": "string",
            "enum": ["dovish", "hawkish", "mixed", "neutral"],
        },
        "rationale":  {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["surprise_label", "direction", "rationale", "confidence"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction (spec §2.4 surprise label semantics)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTIONS = """You are a FOMC press-statement surprise classifier for a quantitative
trading system. Your job is to label this statement's surprise level relative
to ex-ante market expectations and prior FOMC stance, using ONLY the inputs
provided. You are a CLASSIFIER, NOT a forecaster.

LABELS (mutually exclusive):
- NORMAL          : statement consistent with prior narrative + current rate
                    band; no policy pivot; no emergency action.
- MILD_SURPRISE   : statement language tone shifts ≥1 std dev hawkish/dovish
                    vs prior FOMC narrative score; OR small (≤25bp) deviation
                    from current target band.
- EXTREME_SURPRISE: rate decision deviates ≥50bp from current target band; OR
                    emergency inter-meeting action; OR unprecedented language
                    pivot (e.g., balance-sheet expansion first announcement).

DIRECTION (interpretive only, does NOT enter trigger logic):
- dovish   : language / decision favors easing
- hawkish  : language / decision favors tightening
- mixed    : conflicting signals (some hawk + some dove)
- neutral  : no clear directional bias

RULES:
- ONLY use the inputs provided. Do not speculate beyond the text.
- Do not invent forward-rate paths or future policy actions.
- Anchor labels on the QUANTITATIVE THRESHOLDS above (≥50bp, ≥1σ tone, etc.).
- Rationale ≤300 chars, factual, references specific statement language.
- Confidence on 1-5 Likert:
    1 = weak / single ambiguous signal
    2 = single solid signal
    3 = two converging signals
    4 = three converging signals
    5 = four+ converging signals OR unambiguous emergency action

Output JSON only, no prose."""


def build_prompt(
    fomc_date:                datetime.date,
    statement_text:           str,
    narrative_score_current:  float,
    narrative_score_prior:    Optional[float],
    effective_fed_funds_rate: Optional[float],
    fed_funds_target_band:    tuple[Optional[float], Optional[float]],
    prior_30d_spy_return:     Optional[float],
    v3_regime_label:          str,
) -> str:
    """
    Compose the deterministic prompt sent to Gemini. Same inputs → same prompt.

    Spec §2.2 input lock; truncates statement_text to 8000 chars to keep
    input tokens reasonable (Gemini 2.5 Flash token cost dominated by output).
    """
    # Statement may be 5000-20000 chars; truncate while preserving start (Fed
    # opens with the rate decision in para 1, the most signal-rich content)
    statement_excerpt = (statement_text or "")[:8000]

    fed_band_str = "n/a"
    if fed_funds_target_band[0] is not None and fed_funds_target_band[1] is not None:
        fed_band_str = f"{fed_funds_target_band[0]:.2f}% – {fed_funds_target_band[1]:.2f}%"

    effr_str = f"{effective_fed_funds_rate:.2f}%" if effective_fed_funds_rate is not None else "n/a"

    prior_score_str = (
        f"{narrative_score_prior:+.4f}" if narrative_score_prior is not None else "n/a (first FOMC in series)"
    )

    spy_ret_str = f"{prior_30d_spy_return:+.2%}" if prior_30d_spy_return is not None else "n/a"

    delta_str = "n/a"
    if narrative_score_prior is not None:
        delta = narrative_score_current - narrative_score_prior
        delta_str = f"{delta:+.4f}"

    return f"""FOMC DATE: {fomc_date}

EX-ANTE CONTEXT (deterministic numerical inputs; not LLM-derived)
- narrative_score (current statement, z-norm): {narrative_score_current:+.4f}
- narrative_score (prior FOMC):                 {prior_score_str}
- narrative_score Δ (current − prior):          {delta_str}
- effective fed funds rate (FRED):              {effr_str}
- fed funds target band (FRED upper / lower):   {fed_band_str}
- prior 30d SPY return (ex-FOMC, returns only): {spy_ret_str}
- v3 multivariate quant regime label:           {v3_regime_label}

FOMC PRESS STATEMENT (just released; first 8000 chars)
{statement_excerpt}

INSTRUCTIONS
{_SYSTEM_INSTRUCTIONS}"""


# ─────────────────────────────────────────────────────────────────────────────
# Cache (SHA-256 prompt hash → response on disk; reproducibility per spec §2.8)
# ─────────────────────────────────────────────────────────────────────────────


def _hash_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _cache_path(fomc_date: datetime.date) -> Path:
    return _CACHE_DIR / f"fomc_{fomc_date.strftime('%Y%m%d')}.json"


def _cache_get(fomc_date: datetime.date, prompt_hash: str) -> dict | None:
    p = _cache_path(fomc_date)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        # Same FOMC date but different prompt (e.g., text revised) → cache miss
        if payload.get("prompt_hash") != prompt_hash:
            return None
        return payload
    except Exception:
        return None


def _cache_put(
    fomc_date:    datetime.date,
    prompt:       str,
    response:     str,
    *,
    input_tokens: int,
    output_tokens: int,
    cost_usd:     float,
) -> None:
    p = _cache_path(fomc_date)
    payload = {
        "fomc_date":     fomc_date.isoformat(),
        "prompt_hash":   _hash_bytes(prompt),
        "response_hash": _hash_bytes(response),
        "prompt":        prompt,
        "response":      response,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      cost_usd,
        "timestamp":     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_version": LLM_MODEL_VERSION,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Cost ledger (spec §4.4 standalone; not engine.llm_budget SystemConfig)
# ─────────────────────────────────────────────────────────────────────────────


class BudgetExceeded(RuntimeError):
    """Annual or per-call budget cap breached."""


def _load_cost_ledger() -> list[dict]:
    if not _COST_LEDGER_PATH.exists():
        return []
    try:
        return json.loads(_COST_LEDGER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_cost_ledger(ledger: list[dict]) -> None:
    _COST_LEDGER_PATH.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _trailing_365d_total(as_of: datetime.date, ledger: list[dict]) -> float:
    cutoff = as_of - datetime.timedelta(days=365)
    return sum(
        float(e.get("cost_usd", 0.0))
        for e in ledger
        if datetime.date.fromisoformat(e.get("date", "1970-01-01")) >= cutoff
    )


def _check_and_record_cost(call_cost_usd: float, *, as_of: datetime.date) -> None:
    """
    Append cost entry to ledger; raise BudgetExceeded if cap breached.
    Per-call cap and trailing 365d annual cap both enforced.
    """
    if call_cost_usd > PER_CALL_BUDGET_USD:
        raise BudgetExceeded(
            f"per_call cost ${call_cost_usd:.4f} > cap ${PER_CALL_BUDGET_USD:.2f}"
        )
    ledger = _load_cost_ledger()
    trailing = _trailing_365d_total(as_of, ledger)
    if trailing + call_cost_usd > ANNUAL_BUDGET_USD:
        raise BudgetExceeded(
            f"annual trailing-365d cost ${trailing + call_cost_usd:.4f} "
            f"> cap ${ANNUAL_BUDGET_USD:.2f}"
        )
    ledger.append({
        "date":     as_of.isoformat(),
        "cost_usd": round(call_cost_usd, 6),
    })
    _save_cost_ledger(ledger)


def get_cost_status(as_of: Optional[datetime.date] = None) -> dict:
    """Return current cost status for UI / audit."""
    if as_of is None:
        as_of = datetime.date.today()
    ledger = _load_cost_ledger()
    trailing = _trailing_365d_total(as_of, ledger)
    return {
        "trailing_365d_total_usd": round(trailing, 6),
        "annual_cap_usd":           ANNUAL_BUDGET_USD,
        "per_call_cap_usd":         PER_CALL_BUDGET_USD,
        "ledger_entries_count":     len(ledger),
        "fraction_of_annual_cap":   trailing / ANNUAL_BUDGET_USD if ANNUAL_BUDGET_USD > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compute cost from token usage
# ─────────────────────────────────────────────────────────────────────────────


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * COST_PER_1M_INPUT_TOKENS +
            output_tokens * COST_PER_1M_OUTPUT_TOKENS) / 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (with cache + cost + retry; spec §2.3)
# ─────────────────────────────────────────────────────────────────────────────


def _call_llm(
    prompt:    str,
    fomc_date: datetime.date,
    *,
    as_of:     Optional[datetime.date] = None,
) -> dict:
    """
    Call Gemini 2.5 Flash with locked config. Cache-first; on miss invoke API.

    Returns dict with parsed JSON + token usage + cost + cache_hit indicator.
    Raises BudgetExceeded on cost cap; RuntimeError on parse failure after
    MAX_RETRIES_ON_PARSE_FAILURE retries.

    LLM is invoked here ONLY (boundary invariant: 0-LLM-in-eval).
    """
    if as_of is None:
        as_of = fomc_date
    p_hash = _hash_bytes(prompt)
    cached = _cache_get(fomc_date, p_hash)
    if cached:
        try:
            parsed = json.loads(cached["response"])
        except Exception:
            parsed = None
        if parsed is not None:
            logger.info("fomc_override: cache hit fomc_date=%s prompt_hash=%s", fomc_date, p_hash[:12])
            return {
                "parsed":        parsed,
                "response_text": cached["response"],
                "prompt_hash":   p_hash,
                "response_hash": cached.get("response_hash", _hash_bytes(cached["response"])),
                "input_tokens":  cached.get("input_tokens", 0),
                "output_tokens": cached.get("output_tokens", 0),
                "cost_usd":      cached.get("cost_usd", 0.0),
                "cache_hit":     True,
            }

    from engine.key_pool import get_pool
    pool = get_pool()

    last_exc: Exception | None = None
    text = ""
    in_tok = 0
    out_tok = 0
    cost = 0.0
    for attempt in range(MAX_RETRIES_ON_PARSE_FAILURE + 1):
        try:
            model = pool.get_model(
                model_name=LLM_MODEL_VERSION,
                response_schema=LLM_OUTPUT_SCHEMA,
                temperature=LLM_TEMPERATURE,
                thinking_budget=LLM_THINKING_BUDGET,
            )
            resp = model.generate_content(prompt)
            pool.report_success(has_content=True)
            text = getattr(resp, "text", None) or str(resp)
            usage = getattr(resp, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = (getattr(usage, "candidates_token_count", 0) or 0) + \
                      (getattr(usage, "thoughts_token_count", 0) or 0)
            cost = _compute_cost(in_tok, out_tok)
            # Validate parse before recording cost (failed parse not chargeable
            # is a stretch — still chargeable since LLM produced output)
            parsed = json.loads(text)
            # Successful parse — record cost + cache
            _check_and_record_cost(cost, as_of=as_of)
            _cache_put(fomc_date, prompt, text,
                       input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
            return {
                "parsed":        parsed,
                "response_text": text,
                "prompt_hash":   p_hash,
                "response_hash": _hash_bytes(text),
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "cost_usd":      cost,
                "cache_hit":     False,
            }
        except json.JSONDecodeError as exc:
            last_exc = exc
            logger.warning(
                "fomc_override: parse failure attempt %d/%d on fomc_date=%s; first 200 chars: %s",
                attempt + 1, MAX_RETRIES_ON_PARSE_FAILURE + 1, fomc_date, text[:200],
            )
            continue
        except BudgetExceeded:
            raise  # propagate; caller handles
        except Exception as exc:
            last_exc = exc
            logger.error("fomc_override: API call failed attempt %d: %s", attempt + 1, exc)
            continue

    raise RuntimeError(
        f"fomc_override: LLM call failed after {MAX_RETRIES_ON_PARSE_FAILURE + 1} attempts "
        f"(last exc: {last_exc!r})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation + clamping (deterministic, no LLM in this path)
# ─────────────────────────────────────────────────────────────────────────────


_VALID_LABELS = {"NORMAL", "MILD_SURPRISE", "EXTREME_SURPRISE"}
_VALID_DIRECTIONS = {"dovish", "hawkish", "mixed", "neutral"}


def validate_and_classify(llm_output: dict) -> dict:
    """
    Schema-validate LLM output; clamp to neutral defaults on partial failure.

    Returns a normalized dict with all 4 required fields populated. On any
    schema violation, returns NORMAL/neutral/confidence=1 fallback (safe).
    """
    if not isinstance(llm_output, dict):
        return _neutral_fallback("output_not_dict")

    label = llm_output.get("surprise_label")
    direction = llm_output.get("direction")
    rationale = llm_output.get("rationale", "")
    confidence = llm_output.get("confidence")

    if label not in _VALID_LABELS:
        return _neutral_fallback(f"invalid_label_{label}")
    if direction not in _VALID_DIRECTIONS:
        return _neutral_fallback(f"invalid_direction_{direction}")
    if not isinstance(confidence, int) or not (1 <= confidence <= 5):
        return _neutral_fallback(f"invalid_confidence_{confidence}")

    return {
        "surprise_label": label,
        "direction":      direction,
        "rationale":      str(rationale)[:300],  # spec §2.4 char limit
        "confidence":     int(confidence),
        "fallback":       False,
    }


def _neutral_fallback(reason: str) -> dict:
    logger.warning("fomc_override: validation fallback to NORMAL (reason=%s)", reason)
    return {
        "surprise_label": "NORMAL",
        "direction":      "neutral",
        "rationale":      f"validation_fallback:{reason}",
        "confidence":     1,
        "fallback":       True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trigger logic + override application (spec §2.5, §2.6)
# ─────────────────────────────────────────────────────────────────────────────


def trigger_emergency_override(
    surprise_label:           str,
    current_v3_regime_label:  str,
) -> bool:
    """
    Spec §2.5 — emergency override AND-gate.

    Fires iff BOTH:
      - LLM label = EXTREME_SURPRISE
      - quant v3 regime is NOT 'risk-on'

    LLM cannot trigger directional aggressive override (one-way defensive).
    """
    return (
        surprise_label == "EXTREME_SURPRISE"
        and current_v3_regime_label in {"risk-off", "transition"}
    )


def apply_override_to_regime_scale(
    base_scale:    float,
    triggered_at:  datetime.date | None,
    as_of:         datetime.date,
) -> float:
    """
    Spec §2.6 — pure function; deterministic; 0 LLM.

    If override active (triggered_at within HARD_DURATION_DAYS trading days
    of as_of), return base_scale × HARD_OVERRIDE_MULTIPLIER (clamped). Else
    return base_scale unchanged.
    """
    if triggered_at is None:
        return base_scale
    days_since = _trading_days_elapsed(triggered_at, as_of)
    if days_since < 0 or days_since >= HARD_DURATION_DAYS:
        return base_scale
    new_scale = base_scale * HARD_OVERRIDE_MULTIPLIER
    # Hard clamp (defense-in-depth)
    return max(HARD_MULTIPLIER_LOWER * base_scale,
               min(HARD_MULTIPLIER_UPPER * base_scale, new_scale))


# ─────────────────────────────────────────────────────────────────────────────
# Persistent override state (read-side for portfolio.py hook)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OverrideState:
    triggered_at:        datetime.date | None
    surprise_label:      str | None
    direction:           str | None
    days_remaining:      int
    multiplier_applied:  float  # base_scale relative
    fomc_date:           datetime.date | None  # the FOMC date that caused trigger
    raw: dict = field(default_factory=dict)


def _load_override_state_disk() -> dict:
    if not _OVERRIDE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(_OVERRIDE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_override_state_disk(state: dict) -> None:
    _OVERRIDE_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def get_active_override_state(as_of: datetime.date) -> OverrideState:
    """
    Return current override state for portfolio.py hook (spec §4.2-§4.3).

    Reads disk-cached state; computes days_remaining from triggered_at.
    Returns OverrideState with triggered_at=None when no active override.
    """
    raw = _load_override_state_disk()
    triggered_at_str = raw.get("triggered_at")
    if not triggered_at_str:
        return OverrideState(
            triggered_at=None, surprise_label=None, direction=None,
            days_remaining=0, multiplier_applied=1.0, fomc_date=None, raw=raw,
        )
    try:
        triggered_at = datetime.date.fromisoformat(triggered_at_str)
    except Exception:
        return OverrideState(
            triggered_at=None, surprise_label=None, direction=None,
            days_remaining=0, multiplier_applied=1.0, fomc_date=None, raw=raw,
        )
    days_since = _trading_days_elapsed(triggered_at, as_of)
    days_remaining = max(0, HARD_DURATION_DAYS - days_since)
    if days_remaining == 0:
        # Override expired — return inactive state but preserve raw for audit
        return OverrideState(
            triggered_at=None, surprise_label=None, direction=None,
            days_remaining=0, multiplier_applied=1.0, fomc_date=None, raw=raw,
        )
    fomc_date = None
    if raw.get("fomc_date"):
        try:
            fomc_date = datetime.date.fromisoformat(raw["fomc_date"])
        except Exception:
            pass
    return OverrideState(
        triggered_at=triggered_at,
        surprise_label=raw.get("surprise_label"),
        direction=raw.get("direction"),
        days_remaining=days_remaining,
        multiplier_applied=HARD_OVERRIDE_MULTIPLIER,
        fomc_date=fomc_date,
        raw=raw,
    )


def _persist_override_trigger(
    fomc_date:       datetime.date,
    triggered_at:    datetime.date,
    surprise_label:  str,
    direction:       str,
    rationale:       str,
    confidence:      int,
) -> None:
    """Write triggered override state to disk for portfolio.py to read."""
    state = {
        "fomc_date":     fomc_date.isoformat(),
        "triggered_at":  triggered_at.isoformat(),
        "surprise_label": surprise_label,
        "direction":     direction,
        "rationale":     rationale,
        "confidence":    int(confidence),
        "duration_days": HARD_DURATION_DAYS,
        "multiplier":    HARD_OVERRIDE_MULTIPLIER,
        "persisted_at":  datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _save_override_state_disk(state)


# ─────────────────────────────────────────────────────────────────────────────
# Input gathering (deterministic; falls back to None on data fetch failure)
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_fred_csv(series_id: str, start: str, end: str):
    """Generic FRED CSV → pd.Series. Inlined from the deprecated engine.regime
    module (2026-05-29) — keep here so this file stays self-contained."""
    import pandas as pd, requests
    from io import StringIO
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        date_col, val_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df[(df.index >= start) & (df.index <= end)]
        return pd.to_numeric(df[val_col], errors="coerce").dropna()
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        import pandas as pd
        return pd.Series(dtype=float)


def _fetch_fed_funds_data(as_of: datetime.date) -> dict:
    """
    Fetch FRED FEDFUNDS / DFEDTARU / DFEDTARL.

    Returns dict with effective rate + target band; values are None on fetch
    failure (resilience — LLM still classifies based on what's available).
    """
    out = {"effective_fed_funds_rate": None, "target_upper": None, "target_lower": None}
    end_str = as_of.isoformat()
    start_str = (as_of - datetime.timedelta(days=90)).isoformat()
    for key, sid in (
        ("effective_fed_funds_rate", "FEDFUNDS"),
        ("target_upper",             "DFEDTARU"),
        ("target_lower",             "DFEDTARL"),
    ):
        try:
            s = _fetch_fred_csv(sid, start_str, end_str)
            if s is not None and not s.empty:
                out[key] = float(s.dropna().iloc[-1])
        except Exception as exc:
            logger.warning("fomc_override: %s fetch failed: %s", sid, exc)
    return out


def _fetch_spy_30d_return(as_of: datetime.date) -> Optional[float]:
    """
    Compute prior 30 calendar-day SPY return (close-to-close).

    Returns None on data fetch failure.
    """
    try:
        from engine.signal import _fetch_closes
        # Need prices through as_of - 1 day (no lookahead on FOMC day itself)
        end = as_of - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=45)  # buffer for non-trading days
        closes = _fetch_closes("SPY", start=start.isoformat(), end=end.isoformat())
        if closes is None or len(closes) < 25:
            return None
        # Use most recent vs ~30 cal-day prior close
        s = closes.dropna()
        if len(s) < 22:
            return None
        latest = float(s.iloc[-1])
        target_idx = max(0, len(s) - 22)  # ~22 trading days = ~30 calendar days
        prior = float(s.iloc[target_idx])
        if prior <= 0:
            return None
        return latest / prior - 1.0
    except Exception as exc:
        logger.warning("fomc_override: SPY 30d return fetch failed: %s", exc)
        return None


def _fetch_v3_regime_label(as_of: datetime.date) -> str:
    """
    Get current v3 multivariate MSM regime label at as_of - 1 day (ex-ante).

    Returns "risk-on" / "risk-off" / "transition", or "transition" as
    safe default on fetch failure (preserves AND-gate behavior).
    """
    try:
        from engine.regime import get_regime_on
        # Use as_of - 1 day to avoid lookahead on FOMC day
        result = get_regime_on(as_of - datetime.timedelta(days=1))
        if result is None or not getattr(result, "regime", None):
            logger.warning("fomc_override: v3 regime fetch returned None")
            return "transition"
        return str(result.regime)
    except Exception as exc:
        logger.warning("fomc_override: v3 regime fetch failed: %s", exc)
        return "transition"


def _fetch_narrative_score_for(meeting_date: datetime.date) -> Optional[float]:
    """
    Fetch FOMC statement for `meeting_date` and compute narrative_score.
    Returns None on fetch / scoring failure.
    """
    try:
        from engine.narrative_classifier import (
            fetch_fomc_press_statement,
            compute_narrative_score,
        )
        text = fetch_fomc_press_statement(meeting_date)
        if not text:
            return None
        return float(compute_narrative_score(text))
    except Exception as exc:
        logger.warning(
            "fomc_override: narrative_score fetch failed for %s: %s",
            meeting_date, exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DecisionLog write (project-standard ORM, 0 LLM in this path)
# ─────────────────────────────────────────────────────────────────────────────


def _write_decision_log(
    fomc_date:        datetime.date,
    classification:   dict,
    triggered:        bool,
    v3_regime_label:  str,
    inputs_summary:   dict,
) -> int | None:
    """
    Persist decision row to engine.memory.DecisionLog (project-standard ORM).

    Returns row id, or None on failure (logged, not raised).
    """
    try:
        from engine.memory import DecisionLog, SessionFactory
    except Exception as exc:
        logger.error("fomc_override: DecisionLog import failed: %s", exc)
        return None

    sess = SessionFactory()
    try:
        row = DecisionLog(
            tab_type="fomc_override",
            decision_date=fomc_date,
            ai_conclusion=classification.get("rationale", ""),
            direction=classification.get("direction"),
            confidence_score=int(classification.get("confidence", 1)) * 20,  # 1-5 → 20-100
            economic_logic="FOMC press-statement surprise classification (spec id=48)",
            macro_regime=v3_regime_label,
            macro_regime_view=v3_regime_label,
            news_categories_used=json.dumps({
                "surprise_label": classification.get("surprise_label"),
                "fallback":       classification.get("fallback", False),
                "triggered":      triggered,
                "spec_id":        48,
            }, ensure_ascii=False),
            quant_metrics=json.dumps(inputs_summary, ensure_ascii=False, default=str),
            is_backtest=False,
            key_thesis=(classification.get("rationale", "") or "")[:300],
        )
        sess.add(row)
        sess.commit()
        rid = row.id
        logger.info(
            "fomc_override: DecisionLog id=%s fomc_date=%s label=%s triggered=%s",
            rid, fomc_date, classification.get("surprise_label"), triggered,
        )
        return rid
    except Exception as exc:
        sess.rollback()
        logger.error("fomc_override: DecisionLog write failed: %s", exc)
        return None
    finally:
        sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — orchestrates one FOMC day
# ─────────────────────────────────────────────────────────────────────────────


def process_fomc_day(
    as_of:               datetime.date,
    *,
    skip_llm_call:       bool = False,
    inject_classification: dict | None = None,
) -> dict:
    """
    Orchestrate a single FOMC-day decision cycle.

    Spec §4.5 / §三 verdict framework.

    Steps:
      1. Validate as_of is a FOMC day (else noop).
      2. Gather deterministic inputs (statement text, narrative scores, FRED
         data, SPY 30d return, v3 regime label).
      3. Build prompt and call LLM (cached, retried, cost-capped).
      4. Validate / clamp LLM output (fallback NORMAL on schema failure).
      5. Decide trigger (AND-gate: EXTREME_SURPRISE + v3 ≠ risk-on).
      6. Persist override state to disk (if triggered).
      7. Write DecisionLog row (regardless of trigger ON/OFF).

    Args:
        as_of:                 the FOMC date being processed.
        skip_llm_call:         if True, skip LLM entirely (testing); requires
                               inject_classification.
        inject_classification: pre-computed classification for testing.

    Returns:
        dict with action, label, direction, triggered, decision_log_id, etc.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"process_fomc_day expected date, got {type(as_of)}")
    if isinstance(as_of, datetime.datetime):
        as_of = as_of.date()

    if not is_fomc_day(as_of):
        return {"action": "noop_not_fomc_day", "as_of": as_of.isoformat()}

    # Gather deterministic inputs
    statement_text = ""
    try:
        from engine.narrative_classifier import fetch_fomc_press_statement
        statement_text = fetch_fomc_press_statement(as_of) or ""
    except Exception as exc:
        logger.warning("fomc_override: statement fetch failed for %s: %s", as_of, exc)

    narrative_score_current = None
    if statement_text:
        try:
            from engine.narrative_classifier import compute_narrative_score
            narrative_score_current = float(compute_narrative_score(statement_text))
        except Exception as exc:
            logger.warning("fomc_override: current narrative_score failed: %s", exc)

    prior_fomc = _get_prior_fomc_date(as_of)
    narrative_score_prior = None
    if prior_fomc is not None:
        narrative_score_prior = _fetch_narrative_score_for(prior_fomc)

    fed_data = _fetch_fed_funds_data(as_of)
    spy_30d_ret = _fetch_spy_30d_return(as_of)
    v3_regime_label = _fetch_v3_regime_label(as_of)

    # If statement_text is empty (fetch failure), fall back to neutral
    # immediately — cannot classify without text input.
    if not statement_text or narrative_score_current is None:
        classification = _neutral_fallback("missing_statement_or_score")
        cost_usd = 0.0
        cache_hit = False
        prompt_hash = ""
    else:
        prompt = build_prompt(
            fomc_date=as_of,
            statement_text=statement_text,
            narrative_score_current=narrative_score_current,
            narrative_score_prior=narrative_score_prior,
            effective_fed_funds_rate=fed_data.get("effective_fed_funds_rate"),
            fed_funds_target_band=(
                fed_data.get("target_lower"),
                fed_data.get("target_upper"),
            ),
            prior_30d_spy_return=spy_30d_ret,
            v3_regime_label=v3_regime_label,
        )
        if skip_llm_call:
            if inject_classification is None:
                raise ValueError("skip_llm_call=True requires inject_classification")
            classification = validate_and_classify(inject_classification)
            cost_usd = 0.0
            cache_hit = False
            prompt_hash = _hash_bytes(prompt)
        else:
            try:
                llm_out = _call_llm(prompt, as_of, as_of=as_of)
                classification = validate_and_classify(llm_out["parsed"])
                cost_usd = llm_out["cost_usd"]
                cache_hit = llm_out["cache_hit"]
                prompt_hash = llm_out["prompt_hash"]
            except BudgetExceeded as exc:
                logger.error("fomc_override: budget exceeded — fallback NORMAL: %s", exc)
                classification = _neutral_fallback("budget_exceeded")
                cost_usd = 0.0
                cache_hit = False
                prompt_hash = _hash_bytes(prompt)
            except Exception as exc:
                logger.error("fomc_override: LLM call failed — fallback NORMAL: %s", exc)
                classification = _neutral_fallback("llm_call_failed")
                cost_usd = 0.0
                cache_hit = False
                prompt_hash = _hash_bytes(prompt)

    # Trigger decision (deterministic AND-gate)
    triggered = trigger_emergency_override(
        surprise_label=classification["surprise_label"],
        current_v3_regime_label=v3_regime_label,
    )

    # Persist override state (if triggered)
    if triggered:
        _persist_override_trigger(
            fomc_date=as_of,
            triggered_at=as_of,
            surprise_label=classification["surprise_label"],
            direction=classification["direction"],
            rationale=classification["rationale"],
            confidence=classification["confidence"],
        )

    # Write DecisionLog row (regardless of trigger ON/OFF)
    inputs_summary = {
        "narrative_score_current": narrative_score_current,
        "narrative_score_prior":   narrative_score_prior,
        "effective_fed_funds_rate": fed_data.get("effective_fed_funds_rate"),
        "target_upper":            fed_data.get("target_upper"),
        "target_lower":            fed_data.get("target_lower"),
        "prior_30d_spy_return":    spy_30d_ret,
        "v3_regime_label":         v3_regime_label,
        "prompt_hash":             prompt_hash[:16] if prompt_hash else None,
        "cost_usd":                round(cost_usd, 6),
        "cache_hit":               cache_hit,
        "fallback":                classification.get("fallback", False),
        "spec_hash_prefix":        "036b2805f0d6",  # FOMC override spec hash
    }
    decision_log_id = _write_decision_log(
        fomc_date=as_of,
        classification=classification,
        triggered=triggered,
        v3_regime_label=v3_regime_label,
        inputs_summary=inputs_summary,
    )

    return {
        "action":             "fomc_day_processed",
        "as_of":              as_of.isoformat(),
        "surprise_label":     classification["surprise_label"],
        "direction":          classification["direction"],
        "rationale":          classification["rationale"],
        "confidence":         classification["confidence"],
        "fallback":           classification.get("fallback", False),
        "v3_regime_label":    v3_regime_label,
        "triggered":          triggered,
        "decision_log_id":    decision_log_id,
        "cost_usd":           round(cost_usd, 6),
        "cache_hit":          cache_hit,
        "narrative_score_current": narrative_score_current,
        "narrative_score_prior":   narrative_score_prior,
    }
