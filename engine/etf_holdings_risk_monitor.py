"""
engine/etf_holdings_risk_monitor.py — ETF Holdings LLM Risk Monitor (Sprint Week 2).

Pre-registration: docs/spec_etf_holdings_llm_risk_monitor.md
Spec id (engine.preregistration.SpecRegistry): 49
Spec hash lineage:
  v1 (2026-05-08 initial):                 0c3696fc4145
  v2 (2026-05-09 max-of fallback +3 trials): 02a27ba0cc20
  v3 (2026-05-14 5 senior improvements + Sprint I lessons, post-Sprint-I-FAIL): 9cc868d2a8a6
Current hash: 9cc868d2a8a6 (v3, locked 2026-05-14)

Purpose
-------
Monthly LLM screening of unique top-10 holdings across 24 equity ETFs (~120-211
unique names per current universe) for fundamental risk signals. Per-ETF
aggregation produces aggregate risk score (1-5 weighted-avg by holding weight);
score ≥ 3.5 triggers MAX_WEIGHT cap (25% → 15%) for 5 trading days.

Division of labor (escape wrapping, spec §rule-9 N9)
----------------------------------------------------
LLM does: name-level fundamental NLP on SEC 8-K + news (量化做不了 — BAB only
reads β/vol/cross-sectional rank, no corporate fundamentals narrative).
Quant does: BAB cross-sectional rank, vol targeting, position sizing (LLM 不进
data plane).
Hook: portfolio.py Step 6 (position cap) — orthogonal to Step 5 regime overlay.
LLM modifies MAX_WEIGHT meta-parameter (control plane), not weights directly.

Boundary invariant (project rule "0-LLM-in-evaluation")
-------------------------------------------------------
LLM is called only inside `_call_llm_screen_name` for per-name classification.
Verdict computation paths (counterfactual P&L, aggregation, trigger logic) are
pure deterministic functions. See spec §3.7.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (spec §2.3, §2.6, §2.7)
# ─────────────────────────────────────────────────────────────────────────────

# Trigger logic (spec §2.7)
CAP_TRIGGER_THRESHOLD: float = 3.5
"""ETF aggregate risk score ≥ this → fire cap. Spec §2.7."""

HARD_CAP_MULTIPLIER: float = 0.6
"""MAX_WEIGHT × this when cap active (25% × 0.6 = 15%). Spec §2.7."""

HARD_CAP_DURATION_DAYS: int = 5
"""Trading days cap remains active. Spec §2.7."""

HARD_CAP_DURATION_CAP: int = 10
"""Absolute max even with future amend (defense-in-depth). Spec §2.7."""

HARD_CAP_FLOOR: float = 0.5
"""Multiplier ≥ this fraction of base (no overriding to >50% reduction). Spec §2.7."""

HARD_CAP_UPPER: float = 1.0
"""Multiplier ≤ this fraction of base (LLM cannot raise cap, one-way defensive). Spec §2.7."""

# v3 amendment 2026-05-14 — deployment mode (spec §2.10)
ETF_HOLDINGS_DEPLOYMENT_MODE: str = "paper_only"
"""Deployment safety gate per spec v3 §2.10. Values:
  'paper_only':           caps affect paper trade only; real DEFAULT_INITIAL_ALLOCATION never touched.
  'live_pending_approval': caps create PendingApproval; transition needs (24mo verdict ≥ DESCRIPTIVE_INFRASTRUCTURE_PASS)
                            + (≥10 cap activations accumulated) + (Tier 3 supervisor explicit sign-off)
                            + (§2.12 calibration set precision ≥ 0.75).

Default 'paper_only' until ALL 4 conditions met. Changing this constant requires supervisor PR review.
"""

# LLM config (spec §2.3)
LLM_MODEL_VERSION: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.0
LLM_THINKING_BUDGET: int = 1500
MAX_RETRIES_ON_PARSE_FAILURE: int = 2

# Budget constants (spec §2.3 v3 Vertex-corrected 2026-05-14)
# Soft annual budget = $540 (1800 calls × ~$0.30 typical with 100x safety margin).
# Hard halt = $720 (33% safety margin above soft budget).
# ANNUAL_BUDGET_USD is the auto-block threshold used by _check_and_record_cost — it
# MUST equal hard halt $720 to match Watchdog rule_etf_holdings_cost_budget at
# `auto_audit_rules.py:4017-4018` (hard_halt_threshold=720, spec_budget_annual=540).
# Was $120 from v1 AI Studio mispricing — corrected with v3 Vertex tier $0.30/M
# input + $2.50/M output. See docs/spec_etf_holdings_llm_risk_monitor.md §2.3.
ANNUAL_BUDGET_USD: float = 720.0       # HARD HALT (auto-block) — spec §2.3 v3
ANNUAL_SOFT_BUDGET_USD: float = 540.0   # Soft budget (for UI / Watchdog display)
PER_CALL_BUDGET_USD: float = 0.10

# Pricing (Gemini 2.5 Flash, same as anomaly_llm_detector + fomc_surprise_override)
COST_PER_1M_INPUT_TOKENS: float = 0.30
COST_PER_1M_OUTPUT_TOKENS: float = 2.50

# Severity-priority overrides (spec §2.10)
SEVERE_THRESHOLD: float = 4.5
"""Aggregate score ≥ this requires mandatory supervisor approval (even post-onboarding). Spec §2.10."""

SUPERVISOR_ONBOARDING_FIRST_N: int = 3
"""First N cap activations (any ETF) require mandatory supervisor approval. Spec §2.10."""

# v2 amendment 2026-05-09 (hypothesis_amend +3 trials): Max-of fallback trigger
# (per spec §2.7 amendment + §六 partial lift of "max-of forbidden")
# Rationale: weighted-avg dilutes single-name severe risk (e.g. 1 holding score=5
# at 5% weight + 9 holdings score=1 at 95% weight → aggregate 1.2, no fire).
# Max-of fallback fires if ANY holding score ≥ SEVERE_SINGLE_NAME_SCORE AND
# normalized weight ≥ SEVERE_SINGLE_NAME_WEIGHT_FLOOR.
SEVERE_SINGLE_NAME_SCORE: float = 4.5
"""Single-name LLM score ≥ this triggers cap if weight floor met (max-of fallback)."""

SEVERE_SINGLE_NAME_WEIGHT_FLOOR: float = 0.05
"""Single-name normalized weight ≥ 5% is required for max-of fallback (filter trivial weight)."""

# Storage paths
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "etf_holdings_risk_monitor"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = _DATA_DIR / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_COST_LEDGER_PATH = _DATA_DIR / "cost_ledger.json"
_CAP_STATE_PATH = _DATA_DIR / "cap_state.json"
_DECISION_HISTORY_PATH = _DATA_DIR / "decision_history.parquet"
_COUNTERFACTUAL_PNL_PATH = _DATA_DIR / "counterfactual_pnl.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# JSON output schema (spec §2.4)
# ─────────────────────────────────────────────────────────────────────────────

LLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name":        {"type": "string"},
        "risk_score":  {"type": "integer", "minimum": 1, "maximum": 5},
        "event_class": {
            "type": "string",
            "enum": [
                "earnings_warning",
                "sec_filing_material",
                "litigation",
                "accounting_irregularity",
                "regulatory_action",
                "management_turnover",
                "supply_chain_disruption",
                "other_fundamental",
                "no_signal",
            ],
        },
        "rationale":     {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "as_of_date":    {"type": "string"},
    },
    "required": ["name", "risk_score", "event_class", "rationale", "as_of_date"],
}

_VALID_RISK_SCORES = {1, 2, 3, 4, 5}
_VALID_EVENT_CLASSES = {
    "earnings_warning",
    "sec_filing_material",
    "litigation",
    "accounting_irregularity",
    "regulatory_action",
    "management_turnover",
    "supply_chain_disruption",
    "other_fundamental",
    "no_signal",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cost ledger (standalone per spec §2.3 + §4.4 — not engine.llm_budget)
# ─────────────────────────────────────────────────────────────────────────────


class BudgetExceeded(RuntimeError):
    """Annual or per-call cost cap breached."""


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


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * COST_PER_1M_INPUT_TOKENS +
            output_tokens * COST_PER_1M_OUTPUT_TOKENS) / 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-name LLM cache (SHA-256 prompt hash → response on disk)
# ─────────────────────────────────────────────────────────────────────────────


def _hash_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _cache_path_for_name(name: str, as_of: datetime.date) -> Path:
    return _CACHE_DIR / f"{name.upper()}_{as_of.strftime('%Y%m')}.json"


def _cache_get(name: str, as_of: datetime.date, prompt_hash: str) -> dict | None:
    p = _cache_path_for_name(name, as_of)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        # Same name + month but different prompt → cache miss
        if payload.get("prompt_hash") != prompt_hash:
            return None
        return payload
    except Exception:
        return None


def _cache_put(
    name:          str,
    as_of:         datetime.date,
    prompt:        str,
    response:      str,
    *,
    input_tokens:  int,
    output_tokens: int,
    cost_usd:      float,
) -> None:
    p = _cache_path_for_name(name, as_of)
    payload = {
        "name":          name.upper(),
        "as_of":         as_of.isoformat(),
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
# Prompt construction (spec §2.5)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTIONS = """You are a fundamental risk classifier for individual stock holdings of a
quantitative ETF portfolio. Your job is to label this stock's fundamental risk
level (1-5 Likert) based ONLY on the provided ex-ante context (recent SEC
filings, news, price action). You are a CLASSIFIER, not a forecaster.

═══ EVIDENCE PRIORITY (anti-reflexivity) ═══
This system explicitly prioritizes LEADING regulator-mandated disclosures
over LAGGING market-sentiment news, because the LLM's training data largely
reflects market consensus already. Emphasize:

  TIER A (LEADING, weight HEAVY):
    - SEC 8-K filings — regulator-mandated material event disclosure;
      typically leads market repricing by 1-2 weeks
    - SEC 10-Q/10-K filings — periodic financial disclosure with hard data
    - Direct regulatory action (SEC enforcement / FDA action / antitrust)

  TIER B (CONTEXTUAL, weight MEDIUM):
    - Pre-announcement guidance changes, executive departures
    - Litigation filings (court records, not media coverage)

  TIER C (LAGGING, weight LOW):
    - News sentiment articles
    - Analyst commentary / downgrades after-the-fact
    - Social media chatter
    - Price-action-driven narrative

Score primarily based on TIER A evidence. Tier B/C only confirm or contextualize.
If ONLY Tier C signals exist (no 8-K / no regulatory action / no hard event),
default toward score 1-2 (lagging consensus is not leading information).

RISK SCORE (1-5):
- 1: no signal (no Tier A/B material event, only routine Tier C news flow)
- 2: minor signal (single mild Tier B event, e.g. management commentary,
     sector rumor, minor analyst downgrade)
- 3: moderate signal (single material Tier A event with clear evidence:
     8-K earnings warning, executive 8-K departure, single-product issue)
- 4: significant signal (multiple converging Tier A events OR major
     regulatory/legal action, material impairment with 8-K)
- 5: severe signal (existential Tier A risk: bankruptcy filing, fraud
     accusation, SEC enforcement action, massive shareholder loss)

EVENT CLASSES (mutually exclusive, choose ONE):
- earnings_warning      : pre-announcement guidance reduction or beat-miss
- sec_filing_material   : 8-K filing on material event (acquisition, asset write-down)
- litigation            : new lawsuit / class action / antitrust / arbitration
- accounting_irregularity: restatement / auditor change / control weakness
- regulatory_action     : SEC investigation / FDA action / antitrust scrutiny
- management_turnover   : CEO/CFO/critical exec departure or replacement
- supply_chain_disruption: factory closure / supplier issue / shortage
- other_fundamental     : other material fundamental signal not above
- no_signal             : routine month, no material events

RULES:
- ONLY use the provided ex-ante context. Do not speculate beyond the data.
- Do not reference future earnings results or post-event market response.
- Anchor labels on TIER A evidence preferentially (filings > news).
- Rationale ≤200 chars, factual citation of specific evidence.
- evidence_refs: list of specific filing/news source identifiers cited;
  prefer 8-K item numbers / SEC filing accession numbers over news URLs.

Output JSON only, no prose."""


def build_prompt(
    name:                 str,
    as_of:                datetime.date,
    sector:               Optional[str],
    recent_8k_filings:    list[dict],
    recent_news:          list[dict],
    price_30d_return:     Optional[float],
    next_earnings_date:   Optional[datetime.date],
) -> str:
    """
    Compose the deterministic prompt for per-name LLM screening.
    Same inputs → same prompt (SHA-256 hash for reproducibility).
    """
    sector_str = sector if sector else "n/a"
    earnings_str = next_earnings_date.isoformat() if next_earnings_date else "n/a"
    price_str = f"{price_30d_return:+.2%}" if price_30d_return is not None else "n/a"

    filings_str = "(no recent 8-K filings in 30d window)"
    if recent_8k_filings:
        rows = []
        for f in recent_8k_filings[:5]:  # cap to top 5 to keep prompt size bounded
            date_str = f.get("date", "?")
            item_str = f.get("item", "?")
            summary = (f.get("summary") or "")[:250]
            rows.append(f"  - [{date_str}] item {item_str}: {summary}")
        filings_str = "\n".join(rows)

    news_str = "(no recent news in 30d window)"
    if recent_news:
        rows = []
        for i, n in enumerate(recent_news[:8], 1):  # cap to top 8
            date_str = n.get("publish_date", "?")
            source = n.get("source", "?")
            title = (n.get("title") or "")[:200]
            rows.append(f"  {i}. [{date_str}] ({source}) {title}")
        news_str = "\n".join(rows)

    return f"""TICKER: {name}
AS OF: {as_of.isoformat()}
SECTOR: {sector_str}
NEXT EARNINGS: {earnings_str}
PRIOR 30D RETURN: {price_str}

RECENT SEC 8-K FILINGS (last 30 days, top 5):
{filings_str}

RECENT NEWS (last 30 days, top 8):
{news_str}

INSTRUCTIONS
{_SYSTEM_INSTRUCTIONS}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (spec §2.3 — boundary: 0-LLM-in-eval, LLM only here)
# ─────────────────────────────────────────────────────────────────────────────


def _call_llm_screen_name(
    prompt:     str,
    name:       str,
    as_of:      datetime.date,
) -> dict:
    """
    Call Gemini 2.5 Flash with locked config. Cache-first; on miss invoke API.

    Returns dict with parsed JSON + token usage + cost + cache_hit indicator.
    Raises BudgetExceeded on cost cap; RuntimeError on parse failure after
    MAX_RETRIES_ON_PARSE_FAILURE retries.
    """
    p_hash = _hash_bytes(prompt)
    cached = _cache_get(name, as_of, p_hash)
    if cached:
        try:
            parsed = json.loads(cached["response"])
            logger.debug(
                "etf_holdings_monitor: cache hit name=%s month=%s",
                name, as_of.strftime("%Y%m"),
            )
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
        except Exception:
            pass  # fall through to re-fetch

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
            parsed = json.loads(text)
            _check_and_record_cost(cost, as_of=as_of)
            _cache_put(name, as_of, prompt, text,
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
                "etf_holdings_monitor: parse fail attempt %d/%d for %s: first 200 = %s",
                attempt + 1, MAX_RETRIES_ON_PARSE_FAILURE + 1, name, text[:200],
            )
            continue
        except BudgetExceeded:
            raise
        except Exception as exc:
            last_exc = exc
            logger.error("etf_holdings_monitor: API error attempt %d for %s: %s",
                         attempt + 1, name, exc)
            continue

    raise RuntimeError(
        f"etf_holdings_monitor: LLM call failed for {name} after "
        f"{MAX_RETRIES_ON_PARSE_FAILURE + 1} attempts (last: {last_exc!r})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation + neutral fallback (deterministic, 0 LLM)
# ─────────────────────────────────────────────────────────────────────────────


def validate_and_classify_screening(llm_output: dict) -> dict:
    """
    Schema-validate per-name LLM output; fallback to risk_score=1 (no signal)
    on partial failure.
    """
    if not isinstance(llm_output, dict):
        return _neutral_screening_fallback("output_not_dict")

    name = llm_output.get("name")
    risk_score = llm_output.get("risk_score")
    event_class = llm_output.get("event_class")
    rationale = llm_output.get("rationale", "")
    evidence_refs = llm_output.get("evidence_refs", [])
    as_of_date = llm_output.get("as_of_date", "")

    if not isinstance(risk_score, int) or risk_score not in _VALID_RISK_SCORES:
        return _neutral_screening_fallback(f"invalid_risk_score_{risk_score}")
    if event_class not in _VALID_EVENT_CLASSES:
        return _neutral_screening_fallback(f"invalid_event_class_{event_class}")

    return {
        "name":          str(name or "").upper().strip(),
        "risk_score":    int(risk_score),
        "event_class":   str(event_class),
        "rationale":     str(rationale)[:200],
        "evidence_refs": list(evidence_refs) if isinstance(evidence_refs, list) else [],
        "as_of_date":    str(as_of_date),
        "fallback":      False,
    }


def _neutral_screening_fallback(reason: str) -> dict:
    logger.warning("etf_holdings_monitor: validation fallback to risk_score=1 (reason=%s)", reason)
    return {
        "name":          "",
        "risk_score":    1,
        "event_class":   "no_signal",
        "rationale":     f"validation_fallback:{reason}",
        "evidence_refs": [],
        "as_of_date":    "",
        "fallback":      True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-ETF aggregation (deterministic, pure function — spec §2.6)
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_etf_risk(
    holdings:    list[dict],
    name_scores: dict[str, int],
) -> float:
    """
    Spec §2.6 — aggregate ETF risk score = weighted-avg holding risk × top-10
    normalized weight.

    Args:
        holdings:    [{name, weight, rank}, ...] from ingestion
        name_scores: {ticker: risk_score 1-5} from LLM screening

    Returns:
        float ∈ [1.0, 5.0]; 1.0 fallback if holdings empty / invalid.
    """
    if not holdings:
        return 1.0
    top10_weight_sum = sum(h.get("weight", 0.0) for h in holdings)
    if top10_weight_sum < 1e-6:
        return 1.0
    weighted = sum(
        (h.get("weight", 0.0) / top10_weight_sum) *
        name_scores.get(str(h.get("name", "")).upper(), 1)
        for h in holdings
    )
    return max(1.0, min(5.0, weighted))


# ─────────────────────────────────────────────────────────────────────────────
# Cap trigger + application (deterministic, pure)
# ─────────────────────────────────────────────────────────────────────────────


def trigger_etf_cap(
    etf_aggregate_score: float,
    holdings:    list[dict] | None = None,
    name_scores: dict[str, int] | None = None,
) -> bool:
    """
    Spec §2.7 (v2 amendment 2026-05-09, hypothesis_amend +3 trials) —
    deterministic trigger with PRIMARY + FALLBACK conditions.

    PRIMARY: weighted-avg aggregate score ≥ CAP_TRIGGER_THRESHOLD (3.5)
    FALLBACK (max-of): any single holding has
                       LLM score ≥ SEVERE_SINGLE_NAME_SCORE (4.5) AND
                       top-10-normalized weight ≥ SEVERE_SINGLE_NAME_WEIGHT_FLOOR (5%)

    Backward-compat: holdings/name_scores both None → primary check only
    (existing call sites pre-v2 amendment).

    Pure function; 0 LLM in this path.
    """
    if etf_aggregate_score >= CAP_TRIGGER_THRESHOLD:
        return True

    # Max-of fallback (requires both args)
    if holdings is None or name_scores is None:
        return False

    total_weight = sum(h.get("weight", 0.0) for h in holdings)
    if total_weight < 1e-6:
        return False

    for h in holdings:
        name = str(h.get("name", "")).upper()
        score = name_scores.get(name, 1)
        normalized_weight = h.get("weight", 0.0) / total_weight
        if (
            score >= SEVERE_SINGLE_NAME_SCORE
            and normalized_weight >= SEVERE_SINGLE_NAME_WEIGHT_FLOOR
        ):
            return True

    return False


def _trading_days_elapsed(start: datetime.date, current: datetime.date) -> int:
    """Approximate trading days using pandas bdate_range. Same as fomc_surprise_override pattern."""
    import pandas as pd
    if start > current:
        return 0
    return max(0, len(pd.bdate_range(start, current)) - 1)


def apply_cap_to_max_weight(
    base_max_weight: float,
    cap_active:      bool,
    days_since_trigger: int,
) -> float:
    """
    Spec §2.7 — pure function; deterministic; 0 LLM.

    If cap active within HARD_CAP_DURATION_DAYS, return base × HARD_CAP_MULTIPLIER
    (clamped to [HARD_CAP_FLOOR × base, HARD_CAP_UPPER × base]).
    """
    if not cap_active:
        return base_max_weight
    if days_since_trigger < 0 or days_since_trigger >= HARD_CAP_DURATION_DAYS:
        return base_max_weight
    new_max = base_max_weight * HARD_CAP_MULTIPLIER
    return max(HARD_CAP_FLOOR * base_max_weight,
               min(HARD_CAP_UPPER * base_max_weight, new_max))


def get_per_ticker_max_weight_dict(
    base_max_weight: float = 0.25,
    as_of:           Optional[datetime.date] = None,
    paper_trade_mode: bool = True,
) -> dict[str, float]:
    """Public API for engine.portfolio.construct_portfolio Step 6 hook.

    Returns {ticker: effective_max_weight} for ETFs currently under active cap.
    ETFs not in dict use the default base_max_weight (caller's clip default).

    Spec §2.10 v3 deployment mode safety:
      paper_trade_mode=True (DEFAULT): returns caps regardless of
                                       ETF_HOLDINGS_DEPLOYMENT_MODE
      paper_trade_mode=False:          returns caps ONLY if ETF_HOLDINGS_DEPLOYMENT_MODE
                                       == 'live_pending_approval'
                                       (defense-in-depth against accidental real-money cap)

    Args:
      base_max_weight:  caller's MAX_WEIGHT constant (engine.portfolio.MAX_WEIGHT, default 0.25)
      as_of:            evaluation date (default today UTC)
      paper_trade_mode: True if caller is paper trade orchestrator; False for real-money path

    Returns:
      dict {etf_ticker: effective_max_weight_after_cap}; empty if no active caps OR
      blocked by paper_trade_mode safety.

    Pure function; 0 LLM in this path.
    """
    # Defense-in-depth: real-money path with deployment mode 'paper_only' → block
    if (not paper_trade_mode) and ETF_HOLDINGS_DEPLOYMENT_MODE == "paper_only":
        logger.warning(
            "ETF Holdings caps blocked: paper_trade_mode=False but "
            "ETF_HOLDINGS_DEPLOYMENT_MODE='paper_only'. Returning empty dict."
        )
        return {}

    if as_of is None:
        as_of = datetime.datetime.utcnow().date()

    state = _load_cap_state()
    if not state:
        return {}

    result: dict[str, float] = {}
    for etf, entry in state.items():
        try:
            triggered_at = datetime.date.fromisoformat(entry["triggered_at"])
        except Exception:
            continue
        days_since = _trading_days_elapsed(triggered_at, as_of)
        effective = apply_cap_to_max_weight(
            base_max_weight  = base_max_weight,
            cap_active       = True,
            days_since_trigger = days_since,
        )
        if effective < base_max_weight:   # only emit if cap actually changes
            result[etf.upper()] = effective
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cap state management (severity-priority — spec §2.8)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CapEntry:
    triggered_at:     datetime.date
    aggregate_score:  float
    expires_at:       datetime.date
    rationale:        str = ""

    def to_dict(self) -> dict:
        return {
            "triggered_at":    self.triggered_at.isoformat(),
            "aggregate_score": round(self.aggregate_score, 4),
            "expires_at":      self.expires_at.isoformat(),
            "rationale":       self.rationale,
        }


def _load_cap_state() -> dict[str, dict]:
    if not _CAP_STATE_PATH.exists():
        return {}
    try:
        return json.loads(_CAP_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cap_state(state: dict[str, dict]) -> None:
    _CAP_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def cleanup_expired_cap_state(
    as_of:  Optional[datetime.date] = None,
    *,
    buffer_calendar_days: int = 3,
) -> int:
    """Phase A3 (2026-05-14) — purge expired entries from cap_state.json.

    Removes entries where `expires_at` < (as_of - buffer_calendar_days).
    Returns number of entries removed.

    Buffer: keeps recently-expired entries for `buffer_calendar_days` so
    a slightly-late paper trade run still sees the cap state used by
    yesterday's decisions (audit traceability).

    Called by:
      1. scripts/run_etf_holdings_monitor_monthly.py at start of monthly run
      2. engine.portfolio.paper_trade_combined Step 3b (defense-in-depth)
    """
    if as_of is None:
        as_of = datetime.date.today()
    cutoff = as_of - datetime.timedelta(days=buffer_calendar_days)

    state = _load_cap_state()
    if not state:
        return 0

    n_before = len(state)
    keep: dict[str, dict] = {}
    for etf, entry in state.items():
        try:
            expires_at = datetime.date.fromisoformat(entry.get("expires_at", "1970-01-01"))
        except Exception:
            keep[etf] = entry   # malformed → keep (safer than delete)
            continue
        if expires_at >= cutoff:
            keep[etf] = entry
        else:
            logger.info("cleanup_expired_cap_state: purged %s (expired %s, cutoff %s)",
                        etf, expires_at.isoformat(), cutoff.isoformat())

    n_removed = n_before - len(keep)
    if n_removed > 0:
        _save_cap_state(keep)
    return n_removed


def get_active_cap_state(as_of: datetime.date) -> dict[str, dict]:
    """
    Spec §2.8 — return per-ETF active cap state (only entries within duration window).

    Used by portfolio.py Step 6 hook to read effective MAX_WEIGHT overrides.
    """
    raw = _load_cap_state()
    active: dict[str, dict] = {}
    for etf, entry in raw.items():
        try:
            triggered_at = datetime.date.fromisoformat(entry["triggered_at"])
        except Exception:
            continue
        days_since = _trading_days_elapsed(triggered_at, as_of)
        if days_since >= 0 and days_since < HARD_CAP_DURATION_DAYS:
            active[etf] = entry
    return active


def _persist_cap_trigger(
    etf:              str,
    triggered_at:     datetime.date,
    aggregate_score:  float,
    rationale:        str,
) -> None:
    """
    Spec §2.8 severity-priority — replace state if new score higher; ignore if
    new ≤ current active.
    """
    state = _load_cap_state()
    existing = state.get(etf)
    expires_at = triggered_at + datetime.timedelta(days=HARD_CAP_DURATION_DAYS * 2)  # cal day buffer

    if existing:
        try:
            existing_at = datetime.date.fromisoformat(existing["triggered_at"])
            existing_score = float(existing["aggregate_score"])
            still_active = _trading_days_elapsed(existing_at, triggered_at) < HARD_CAP_DURATION_DAYS
            if still_active and aggregate_score <= existing_score:
                logger.info(
                    "etf_holdings_monitor: %s new score %.2f ≤ existing active %.2f, ignored (severity-priority)",
                    etf, aggregate_score, existing_score,
                )
                return
        except Exception:
            pass  # fall through to replace

    entry = CapEntry(
        triggered_at=triggered_at,
        aggregate_score=aggregate_score,
        expires_at=expires_at,
        rationale=rationale[:200],
    )
    state[etf] = entry.to_dict()
    _save_cap_state(state)
    logger.info(
        "etf_holdings_monitor: cap fired etf=%s score=%.2f triggered_at=%s",
        etf, aggregate_score, triggered_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision log write (project-standard ORM, 0 LLM in this path)
# ─────────────────────────────────────────────────────────────────────────────


def _write_decision_log_cap_activation(
    etf:              str,
    aggregate_score:  float,
    rationale:        str,
    n_holdings_above_3: int,
    triggered_at:     datetime.date,
) -> int | None:
    """Persist cap activation row to DecisionLog ORM. Returns id or None on failure."""
    try:
        from engine.memory import DecisionLog, SessionFactory
    except Exception as exc:
        logger.error("etf_holdings_monitor: DecisionLog import failed: %s", exc)
        return None

    sess = SessionFactory()
    try:
        row = DecisionLog(
            tab_type="etf_holdings_cap",
            decision_date=triggered_at,
            ticker=etf,
            ai_conclusion=rationale[:500],
            direction="低配",  # cap = reduce exposure
            confidence_score=int(aggregate_score * 20),  # 1-5 → 20-100
            economic_logic="ETF holdings fundamental NLP risk aggregation (spec id=49)",
            news_categories_used=json.dumps({
                "aggregate_score":    round(aggregate_score, 4),
                "n_holdings_flagged": n_holdings_above_3,
                "spec_id":            49,
                "spec_hash_prefix":   "9cc868d2a8a6",   # v3 locked 2026-05-14
            }, ensure_ascii=False),
            quant_metrics=json.dumps({
                "cap_multiplier":     HARD_CAP_MULTIPLIER,
                "duration_days":      HARD_CAP_DURATION_DAYS,
                "trigger_threshold":  CAP_TRIGGER_THRESHOLD,
            }, ensure_ascii=False),
            is_backtest=False,
            key_thesis=rationale[:300],
        )
        sess.add(row)
        sess.commit()
        return row.id
    except Exception as exc:
        sess.rollback()
        logger.error("etf_holdings_monitor: DecisionLog write failed for %s: %s", etf, exc)
        return None
    finally:
        sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — process monthly rebalance
# ─────────────────────────────────────────────────────────────────────────────


def screen_name(
    name:                 str,
    as_of:                datetime.date,
    *,
    sector:               Optional[str] = None,
    recent_8k_filings:    Optional[list[dict]] = None,
    recent_news:          Optional[list[dict]] = None,
    price_30d_return:     Optional[float] = None,
    next_earnings_date:   Optional[datetime.date] = None,
    skip_llm_call:        bool = False,
    inject_classification: dict | None = None,
) -> dict:
    """
    Per-name LLM screening (public API).

    On LLM/budget failure → fallback risk_score=1 (no_signal); never crashes.
    Cached on disk per (name, YYYYMM); same prompt 24h apart → same response.

    Args:
        skip_llm_call:         if True, skip LLM (testing); requires inject_classification
        inject_classification: pre-computed classification for testing
    """
    prompt = build_prompt(
        name=name,
        as_of=as_of,
        sector=sector,
        recent_8k_filings=recent_8k_filings or [],
        recent_news=recent_news or [],
        price_30d_return=price_30d_return,
        next_earnings_date=next_earnings_date,
    )

    if skip_llm_call:
        if inject_classification is None:
            raise ValueError("skip_llm_call=True requires inject_classification")
        return validate_and_classify_screening(inject_classification)

    try:
        llm_out = _call_llm_screen_name(prompt, name, as_of)
        return validate_and_classify_screening(llm_out["parsed"])
    except BudgetExceeded as exc:
        logger.error("etf_holdings_monitor: budget exceeded for %s — fallback no_signal: %s", name, exc)
        return _neutral_screening_fallback("budget_exceeded")
    except Exception as exc:
        logger.error("etf_holdings_monitor: LLM call failed for %s — fallback no_signal: %s", name, exc)
        return _neutral_screening_fallback("llm_call_failed")


def process_monthly_rebalance(as_of: datetime.date) -> dict:
    """
    Spec §三 main orchestration entry point. Called by monthly run script.

    Steps:
      1. Fetch top 10 holdings for 24 equity ETFs (engine.etf_holdings_ingestion)
      2. Deduplicate to unique names (~120-211 typical)
      3. LLM screen each unique name (cached + retried + cost-capped)
      4. Per-ETF aggregate via weighted-avg
      5. Trigger cap for ETFs ≥ CAP_TRIGGER_THRESHOLD
      6. Persist cap state (severity-priority)
      7. Write DecisionLog rows for cap activations

    Returns: summary dict with cap activations, scores, costs, etc.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")

    from engine.etf_holdings_ingestion import (
        fetch_all_equity_etf_holdings,
        deduplicate_holdings_to_unique_names,
    )

    holdings_by_etf = fetch_all_equity_etf_holdings(as_of)
    unique_names = deduplicate_holdings_to_unique_names(holdings_by_etf)

    # LLM screen each unique name (caller passes ex-ante context; for v1 main module,
    # we call with minimal context — daily_run script enriches with SEC + news + price)
    name_scores: dict[str, int] = {}
    name_screening_results: dict[str, dict] = {}
    n_llm_calls = 0
    total_cost = 0.0
    n_fallbacks = 0
    for name in sorted(unique_names):
        result = screen_name(name, as_of)  # uses default inputs; daily_run enriches
        name_scores[name] = result["risk_score"]
        name_screening_results[name] = result
        if result.get("fallback"):
            n_fallbacks += 1
        if not result.get("fallback") and not result.get("from_cache", False):
            n_llm_calls += 1

    # Per-ETF aggregation
    etf_aggregates: dict[str, float] = {}
    etf_n_high: dict[str, int] = {}  # holdings above 3 (informational)
    for etf, holdings in holdings_by_etf.items():
        score = aggregate_etf_risk(holdings, name_scores)
        etf_aggregates[etf] = score
        etf_n_high[etf] = sum(
            1 for h in holdings
            if name_scores.get(str(h.get("name", "")).upper(), 1) >= 3
        )

    # Cap trigger + persist (v2 amendment: pass holdings + name_scores for max-of fallback)
    cap_activations: list[dict] = []
    for etf, score in sorted(etf_aggregates.items(), key=lambda x: -x[1]):
        _holdings = holdings_by_etf.get(etf, [])
        if trigger_etf_cap(score, holdings=_holdings, name_scores=name_scores):
            top_contributors = sorted(
                holdings_by_etf.get(etf, []),
                key=lambda h: -name_scores.get(str(h.get("name", "")).upper(), 1) *
                              h.get("weight", 0.0),
            )[:3]
            top_names = [str(h.get("name", "")).upper() for h in top_contributors]
            rationale = (
                f"Aggregate risk {score:.2f} ≥ {CAP_TRIGGER_THRESHOLD}; "
                f"top contributors: {', '.join(top_names)}; "
                f"{etf_n_high[etf]} holdings ≥ score 3"
            )
            _persist_cap_trigger(etf, as_of, score, rationale)
            decision_id = _write_decision_log_cap_activation(
                etf=etf,
                aggregate_score=score,
                rationale=rationale,
                n_holdings_above_3=etf_n_high[etf],
                triggered_at=as_of,
            )
            cap_activations.append({
                "etf":              etf,
                "aggregate_score":  score,
                "rationale":        rationale,
                "n_holdings_high":  etf_n_high[etf],
                "decision_log_id":  decision_id,
            })

    return {
        "as_of":             as_of.isoformat(),
        "n_etfs_screened":   len(holdings_by_etf),
        "n_unique_names":    len(unique_names),
        "n_llm_calls":       n_llm_calls,
        "n_fallbacks":       n_fallbacks,
        "total_cost_usd":    round(total_cost, 6),
        "etf_aggregates":    {k: round(v, 4) for k, v in etf_aggregates.items()},
        "cap_activations":   cap_activations,
        "n_cap_activations": len(cap_activations),
    }
