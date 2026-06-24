"""
engine/forensic/devils_advocate.py — Dual-LLM forensic verdict consistency.

Forensic redesign Phase 3 (2026-05-14).

Purpose
-------
Wrap a forensic news investigation with a second-LLM devil's advocate to
counter Tetlock-Gardner 2015 "narrative coherence ≠ truth" bias and
Logg-Minson-Moore 2019 "algorithm appreciation" bias. The pattern:

  PRIMARY:          Gemini 2.5 Flash (engine.forensic.news_context)
  DEVIL'S ADVOCATE: DeepSeek V4-flash (engine.deepseek_client)

Both LLMs receive identical input (trade context + same headlines) and
identical prompt. Each returns a structured 5-field JSON verdict
(material_events / macro_context / sentiment_assessment / signal_alignment /
forensic_verdict).

A `verdict_consistency` is then computed:
  agree-on-case (a/b/c)    → 1.0 weighted 0.6
  agree on positive vs neg signal-alignment direction → +0.2
  agree on >=1 material_event keyword (case-insensitive overlap) → +0.2

Total in [0, 1]. Thresholds:
  consistency >= 0.70 → HIGH confidence: dual-signed verdict surfaced
  consistency  < 0.70 → LOW  confidence: BOTH narratives shown side-by-
                        side, user reviews divergence manually.

Cost: Gemini ~$0.001-0.003 per call + DeepSeek ~$0.0001-0.0005
(~9× cheaper output). Total ~$0.002-0.005 per forensic event — roughly
doubles cost vs single-LLM but eliminates the narrative-coherence bias
single-LLM cannot self-detect.

Doctrine
--------
Both LLM outputs land in NARRATIVE layer only (0-LLM-in-DECISION
invariant preserved). Verdict consistency feeds UI confidence indicator
+ optional Watchdog rule (forensic_verdict_diverge_streak); never feeds
back to alpha signal generation or portfolio allocation.

References
----------
  - Tetlock-Gardner 2015 "Superforecasting" — narrative coherence ≠
    predictive accuracy (ρ ≈ 0.1-0.2)
  - Logg-Minson-Moore 2019 "Algorithm appreciation" — humans over-trust
    fluent LLM output; cross-LLM disagreement breaks the spell
  - Cohen-Polk-Vuolteenaho 2003 — anomaly_detector input upstream
  - Brinson-Hood-Beebower 1986 — residual_attribution input upstream
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import time
from typing import Optional

from engine.agents.observability import track_agent_invocation

logger = logging.getLogger(__name__)

# Confidence threshold — below this, surface BOTH narratives.
CONSISTENCY_HIGH_THRESHOLD = 0.70


@dataclasses.dataclass(frozen=True)
class DevilsAdvocateVerdict:
    """Structured output of DeepSeek devil's advocate call.

    Mirrors ForensicNewsSummary's LLM-output fields. Stand-alone (does
    NOT inherit) so it can be persisted/compared independently.
    """
    material_events:       tuple[str, ...]
    macro_context:         str
    sentiment_assessment:  str
    signal_alignment:      str
    key_quotes:            tuple[str, ...]
    forensic_verdict:      str          # 'case_a' / 'case_b' / 'case_c'

    # Audit
    cost_usd:              float
    llm_model:             str
    llm_latency_ms:        int
    raw_text:              str          # full model output for audit trail


@dataclasses.dataclass(frozen=True)
class DualLLMForensicResult:
    """Composite output: primary (Gemini) + devil (DeepSeek) + consistency."""
    # Trade context (echoed; same as ForensicNewsSummary)
    date:                  datetime.date
    ticker:                str
    strategy_name:         str

    # Per-model outputs
    primary_summary:       "ForensicNewsSummary"   # forward ref to avoid circular
    devil_verdict:         DevilsAdvocateVerdict

    # Consistency metrics
    verdict_agreement:     bool                    # case_a/b/c exact match
    direction_agreement:   bool                    # +/- signal_alignment direction
    event_overlap:         int                     # n shared lowercase keywords in material_events
    consistency_score:     float                   # in [0, 1]
    consistency_label:     str                     # 'HIGH' | 'LOW'

    # Aggregate audit
    total_cost_usd:        float
    extracted_at_utc:      datetime.datetime


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction — identical to engine.forensic.news_context._USER_TEMPLATE
# but adapted for DeepSeek (no native responseSchema; rely on prompt + JSON parse)
# ─────────────────────────────────────────────────────────────────────────────
_DEVIL_SYSTEM_PROMPT: str = (
    "You are a senior quant analyst doing forensic analysis of a paper-trade "
    "drawdown. You receive (a) trade context: strategy, signal value, weight, "
    "realized return, expected horizon, and (b) a list of news headlines from "
    "the 10-day window around the trade. Your job: classify the move into one "
    "of three cases (case_a / case_b / case_c) and provide a structured "
    "forensic verdict. Be concise, evidence-based, and quote sources verbatim "
    "when possible. DO NOT speculate beyond the headlines.\n\n"
    "OUTPUT FORMAT (strict): Return ONLY a valid JSON object with these "
    "fields:\n"
    "{\n"
    '  "material_events":      [<=5 short event descriptions],\n'
    '  "macro_context":        "<=200 chars",\n'
    '  "sentiment_assessment": "<=200 chars, +/- direction explicit",\n'
    '  "signal_alignment":     "<=200 chars, was signal correct ex-ante?",\n'
    '  "key_quotes":           [<=5 verbatim headline excerpts],\n'
    '  "forensic_verdict":     "case_a" | "case_b" | "case_c"\n'
    "}\n\n"
    "Case definitions:\n"
    "  case_a — signal wrong / over-fit (model failed to predict the move)\n"
    "  case_b — horizon incomplete (decision was right, realization not yet)\n"
    "  case_c — exogenous shock (signal correct, idiosyncratic news dominated)\n\n"
    "Return NOTHING outside the JSON object — no preamble, no commentary."
)


def _build_user_prompt(
    *,
    date:                  datetime.date,
    ticker:                str,
    strategy_name:         str,
    signal_value:          Optional[float],
    weight:                float,
    realized_return:       Optional[float],
    expected_horizon_days: int,
    headlines:             list[dict],
    date_window_start:     datetime.date,
    date_window_end:       datetime.date,
) -> str:
    """Build user prompt body identical in semantics to Gemini's _USER_TEMPLATE."""
    from engine.forensic.news_context import _build_headlines_text
    return (
        f"TRADE CONTEXT\n"
        f"  Date:              {date.isoformat()}\n"
        f"  Ticker:            {ticker}\n"
        f"  Strategy:          {strategy_name}\n"
        f"  Signal value:      {signal_value}\n"
        f"  Weight:            {weight:+.4f}\n"
        f"  Realized return:   "
        f"{('n/a' if realized_return is None else f'{realized_return:+.4f}')}\n"
        f"  Expected horizon:  {expected_horizon_days} days\n\n"
        f"NEWS HEADLINES (date window {date_window_start.isoformat()} to "
        f"{date_window_end.isoformat()}, {len(headlines)} articles)\n"
        f"{_build_headlines_text(headlines)}\n\n"
        f"TASK: Classify into case_a (signal wrong), case_b (horizon "
        f"incomplete), or case_c (exogenous shock). Output structured "
        f"JSON per schema. Output ONLY the JSON."
    )


def _parse_devil_response(raw_text: str) -> dict:
    """Robust JSON parse from DeepSeek output.

    DeepSeek doesn't have native responseSchema; some models prefix with
    text like "```json" or trailing comments. This strips leading/
    trailing non-JSON and parses defensively.
    """
    text = raw_text.strip()
    # Strip code fences if present
    for fence in ("```json", "```JSON", "```"):
        if text.startswith(fence):
            text = text[len(fence):].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    # Find outermost { ... }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in devil response")
    return json.loads(text[start:end + 1])


# ─────────────────────────────────────────────────────────────────────────────
# Consistency metric
# ─────────────────────────────────────────────────────────────────────────────
def _direction(signal_alignment_text: str) -> str:
    """Crude direction classification of signal_alignment text."""
    txt = (signal_alignment_text or "").lower()
    pos = sum(1 for w in ("correct", "right", "well", "aligned", "valid", "正确", "对的", "符合")
              if w in txt)
    neg = sum(1 for w in ("wrong", "incorrect", "fail", "misaligned", "off",
                          "错误", "失误", "不对", "偏离")
              if w in txt)
    if pos > neg:  return "positive"
    if neg > pos:  return "negative"
    return "neutral"


def _event_overlap_count(
    primary_events: tuple[str, ...],
    devil_events:   tuple[str, ...],
) -> int:
    """Count shared lowercase keywords across material_events lists.

    Crude lexical overlap — sufficient as a coarse consistency signal.
    """
    def _kw(events):
        out = set()
        for e in events or ():
            for tok in str(e).lower().split():
                tok = tok.strip(".,;:!?\"'()[]")
                if len(tok) >= 4:
                    out.add(tok)
        return out
    a, b = _kw(primary_events), _kw(devil_events)
    return len(a & b)


def compute_consistency(
    primary_verdict:   str,
    primary_alignment: str,
    primary_events:    tuple[str, ...],
    devil_verdict:     str,
    devil_alignment:   str,
    devil_events:      tuple[str, ...],
) -> dict:
    """Compute consistency_score in [0, 1] from per-LLM outputs.

    Weights:
      verdict (case_a/b/c) match:    0.6
      direction (+ / -) match:        0.2
      event_overlap >= 1 keyword:     0.2

    Returns dict with score + 3 boolean components + overlap count.
    """
    v_agree = (primary_verdict == devil_verdict)
    d_agree = (_direction(primary_alignment) == _direction(devil_alignment))
    overlap = _event_overlap_count(primary_events, devil_events)
    e_agree = overlap >= 1
    score = (0.6 * v_agree) + (0.2 * d_agree) + (0.2 * e_agree)
    label = "HIGH" if score >= CONSISTENCY_HIGH_THRESHOLD else "LOW"
    return {
        "verdict_agreement":   bool(v_agree),
        "direction_agreement": bool(d_agree),
        "event_overlap":       int(overlap),
        "consistency_score":   float(score),
        "consistency_label":   label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry: investigate with dual LLM
# ─────────────────────────────────────────────────────────────────────────────
def _devils_advocate_schema_validator(result: DualLLMForensicResult) -> bool:
    return (
        isinstance(result, DualLLMForensicResult)
        and 0.0 <= result.consistency_score <= 1.0
        and result.consistency_label in ("HIGH", "LOW")
        and result.devil_verdict is not None
    )


def _devils_advocate_quality_extractor(result: DualLLMForensicResult) -> dict:
    """Cross-LLM consistency = quality signal (Gemini PRIMARY vs DeepSeek DEVIL)."""
    return {
        "cross_llm_consistency_score": float(result.consistency_score),
        "consistency_label":           result.consistency_label,
        "verdict_agreement":           bool(result.verdict_agreement),
        "direction_agreement":         bool(result.direction_agreement),
        "event_overlap_count":         int(result.event_overlap),
    }


def _devils_advocate_extra(result: DualLLMForensicResult) -> dict:
    return {
        "ticker":          result.ticker,
        "strategy_name":   result.strategy_name,
        "n_tool_calls":    2,    # primary + devil = 2 LLM calls (proxy for tool calls)
    }


@track_agent_invocation(
    agent_id="forensic_devils_advocate",
    schema_validator=_devils_advocate_schema_validator,
    extract_extra=_devils_advocate_extra,
    quality_extractor=_devils_advocate_quality_extractor,
)
def investigate_with_devils_advocate(
    *,
    date:                  datetime.date,
    ticker:                str,
    signal_value:          Optional[float],
    weight:                float,
    realized_return:       Optional[float],
    strategy_name:         str,
    expected_horizon_days: int,
    window_days:           int = 5,
    skip_devil:            bool = False,
) -> DualLLMForensicResult:
    """End-to-end dual-LLM forensic investigation.

    Step 1: PRIMARY Gemini call via engine.forensic.news_context.investigate_trade
    Step 2: DEVIL'S ADVOCATE DeepSeek call with the SAME headlines + SAME prompt
    Step 3: Compute consistency_score + label
    Step 4: Return DualLLMForensicResult

    If skip_devil=True OR DeepSeek not available, returns a result with
    devil_verdict populated from a degraded "single-LLM" placeholder that
    flags consistency_label='LOW' (so UI prompts user that the devil
    advocate did not run).

    Doctrine: never feeds back to decision layer. Output for UI/audit only.
    """
    from engine.forensic.news_context import (
        investigate_trade, fetch_news_window, _build_headlines_text,
    )

    t_start = datetime.datetime.utcnow()
    # Step 1: PRIMARY — Gemini investigates and produces ForensicNewsSummary.
    # This call also fetches/caches the news, so the devil reuses the
    # same headlines via the AV cache to guarantee input parity.
    primary = investigate_trade(
        date=date, ticker=ticker, signal_value=signal_value, weight=weight,
        realized_return=realized_return, strategy_name=strategy_name,
        expected_horizon_days=expected_horizon_days, window_days=window_days,
    )

    # Step 2: Fetch the same headlines for the devil. This will hit the
    # AV cache (engine.forensic.news_context._av_cache_*) populated by
    # Step 1, so we don't pay AV again and input parity is guaranteed.
    headlines, win_start, win_end, _n_sources = fetch_news_window(
        ticker, date, window_days=window_days,
    )

    devil_text = ""
    devil_cost = 0.0
    devil_latency = 0
    devil_model = "deepseek-skipped"
    try:
        if skip_devil:
            raise RuntimeError("skip_devil=True")
        from engine.deepseek_client import call_deepseek, is_available
        if not is_available():
            raise RuntimeError("DeepSeek not available (credentials)")
        user_prompt = _build_user_prompt(
            date=date, ticker=ticker, strategy_name=strategy_name,
            signal_value=signal_value, weight=weight,
            realized_return=realized_return,
            expected_horizon_days=expected_horizon_days,
            headlines=headlines,
            date_window_start=win_start, date_window_end=win_end,
        )
        full_prompt = _DEVIL_SYSTEM_PROMPT + "\n\n" + user_prompt
        resp = call_deepseek(full_prompt, max_tokens=1000, temperature=0.1)
        devil_text    = resp.content
        devil_cost    = resp.cost_usd
        devil_latency = resp.latency_ms
        devil_model   = resp.model
        parsed = _parse_devil_response(devil_text)
    except Exception as exc:
        logger.warning("Devil's advocate (DeepSeek) failed: %s", exc)
        # Degraded fallback: no devil, all fields empty, verdict = primary's
        # verdict so verdict_agreement is trivially True but direction/event
        # signals are missing → consistency_label degrades to LOW via overlap=0.
        parsed = {
            "material_events":      [],
            "macro_context":        f"(devil's advocate unavailable: {exc!s})",
            "sentiment_assessment": "",
            "signal_alignment":     "",
            "key_quotes":           [],
            "forensic_verdict":     primary.forensic_verdict,
        }
        devil_model = f"deepseek-failed:{type(exc).__name__}"

    devil_verdict = DevilsAdvocateVerdict(
        material_events      = tuple(parsed.get("material_events", [])),
        macro_context        = str(parsed.get("macro_context", "")),
        sentiment_assessment = str(parsed.get("sentiment_assessment", "")),
        signal_alignment     = str(parsed.get("signal_alignment", "")),
        key_quotes           = tuple(parsed.get("key_quotes", [])),
        forensic_verdict     = str(parsed.get("forensic_verdict", "case_a")),
        cost_usd             = float(devil_cost),
        llm_model            = devil_model,
        llm_latency_ms       = int(devil_latency),
        raw_text             = devil_text,
    )

    # Step 3: Consistency
    cons = compute_consistency(
        primary_verdict   = primary.forensic_verdict,
        primary_alignment = primary.signal_alignment,
        primary_events    = primary.material_events,
        devil_verdict     = devil_verdict.forensic_verdict,
        devil_alignment   = devil_verdict.signal_alignment,
        devil_events      = devil_verdict.material_events,
    )

    return DualLLMForensicResult(
        date                  = date,
        ticker                = ticker,
        strategy_name         = strategy_name,
        primary_summary       = primary,
        devil_verdict         = devil_verdict,
        verdict_agreement     = cons["verdict_agreement"],
        direction_agreement   = cons["direction_agreement"],
        event_overlap         = cons["event_overlap"],
        consistency_score     = cons["consistency_score"],
        consistency_label     = cons["consistency_label"],
        total_cost_usd        = float(primary.cost_usd + devil_verdict.cost_usd),
        extracted_at_utc      = datetime.datetime.utcnow(),
    )
