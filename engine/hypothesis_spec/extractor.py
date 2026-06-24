"""hypothesis_spec.extractor — LLM-driven claim → structured spec.

Takes a free-text hypothesis claim (potentially with mechanism_family +
mechanism_subtype hints) and produces a HypothesisSpec.

Single-shot Claude call with a tightly-bounded JSON output schema. We
use tool-call output (forced JSON) so the LLM cannot drift into prose.
The Anthropic SDK's structured-output / forced tool is used to enforce
this — if Claude tries to return text instead of the tool call, the call
is treated as a failure.

Why LLM and not regex/heuristics
--------------------------------
Real claims look like:
  "Cross-sectional momentum from 12 to 1 months on developed-market
   currencies, with a low-vol filter and monthly rebalance."
A regex parser would need a hand-curated 200-rule grammar. An LLM with
the enum vocabulary in its prompt handles the variety with one model
call (~$0.005, ~3s).

The risk is hallucinated values. Mitigations:
  1. Forced tool call with strict enum schema (claude rejects invalid)
  2. confidence field — the model self-reports certainty
  3. On extraction confidence < threshold OR validation fails, the spec
     is stored with extraction.confidence < 0.5 and the UI surfaces a
     warning + 'edit spec' button
  4. Re-extraction of the same claim with same extractor version is
     deterministic-ish (temperature=0) and the spec_hash will match

Usage
-----
spec = extract_spec(
    source_hypothesis_id = "h_abc...",
    claim_text           = "...",
    mechanism_family     = "CARRY",       # hint
    mechanism_subtype    = "fx_carry_g10", # hint
)
"""
from __future__ import annotations

import json
import logging
import datetime as _dt
from typing import Optional

from engine.hypothesis_spec.schema import (
    HypothesisSpec, SignalLeg, Universe, PortfolioConstruction,
    RiskManagement, PredictedOutcome, Extraction,
)
from engine.hypothesis_spec.enums import (
    ClaimType, FamilyV2, AssetClass, SignalType, Sign, UniverseSubset,
    Weighting, Rebalance, PredictedDirection,
)

logger = logging.getLogger(__name__)


# B.2-A2 (2026-06-05): bumped from v1 -> v2 with ClaimType pre-classification.
# F12  (2026-06-05): v3 = anti-UNKNOWN prompt + auto-downgrade post-processor.
EXTRACTOR_VERSION = "claude_sonnet_4_6_v3"

# F12: how many UNKNOWN core fields on a FACTOR_HYPOTHESIS forces auto-
# downgrade to METHODOLOGY. Cross-checked against the doctrine: a
# tradable strategy needs (signal, where, how_much, when). Allow at
# most 1 UNKNOWN before reclassifying.
_F12_MAX_UNKNOWN_BEFORE_DOWNGRADE = 2


def _system_prompt() -> str:
    return f"""You are a quantitative-research metadata extractor. Your task is to
read a hypothesis claim and emit a STRUCTURED specification.

# Hard rules
1. You MUST call the `emit_hypothesis_spec` tool with valid JSON matching
   its schema. Do NOT produce any text outside the tool call.
2. Every enum field MUST be one of the listed allowed values. If the
   claim is ambiguous, use UNKNOWN — never invent.
3. Be conservative: only assert a value when the claim explicitly or
   strongly implies it. Default to UNKNOWN for missing fields, NOT to
   a "reasonable guess".
4. self-report confidence honestly: 1.0 means "spec is unambiguous"
   and 0.3 means "I am guessing on >50% of fields".

# STEP 1 — classify claim_type FIRST (B.2-A2)
Before anything else, decide which KIND of claim this is. Only
FACTOR_HYPOTHESIS claims describe a tradable strategy and flow into
the Composer pipeline. Other types are still stored as research
evidence but must NOT be forced into a factor family.

claim_type options:
  FACTOR_HYPOTHESIS  — Tradable strategy claim: a signal that produces
                       a returns series. MUST have a signal mechanism
                       (carry / momentum / value / etc.), an asset class,
                       and an implied long/short rule.
                       Examples:
                         "FX carry on G10 monthly outperforms equal-weight"
                         "Cross-sectional 12-1 momentum earns 6% per year"
                         "Short-term reversal (1-month) on US large caps"

  METHODOLOGY        — Claim about research METHOD: multiple-testing
                       thresholds, sample-size theory, overfitting bias,
                       factor model explanatory power, replication rates.
                       NOT itself tradable.
                       Examples:
                         "BHY FDR adjustment finds more factors than Bonferroni"
                         "t-ratio 2.0 is insufficient given 316 published factors"
                         "Holdout method applied 20 times produces false positives"
                         "q-factor model explains 115 of 161 anomalies"

  MICROSTRUCTURE     — Market structure / implementation cost measurement:
                       market impact, bid-ask, slippage, price-impact decay.
                       Used to size capacity but NOT a factor itself.
                       Examples:
                         "Mean market impact for institutional trades = 12 bps"
                         "70 percent of price impact is permanent"
                         "Buy-to-cover costs 7.3 bps more than buy-long"

  CAPACITY           — AUM / fund-size ceiling, scalability, break-even.
                       Examples:
                         "Size and value survive trading costs at very high AUM"
                         "Value + momentum together have higher capacity due to netting"

  DECAY_STUDY        — Post-publication / out-of-sample decay of known
                       anomalies. About the AGGREGATE behavior of many
                       factors, not a single new tradable signal.
                       Examples:
                         "Post-publication decay across 82 anomalies = 35 percent"
                         "Accounting anomalies decline both before and after discovery"

  FACTOR_STRUCTURE   — Factor correlations, principal components, model
                       fit statistics. About the SHAPE of factor space.
                       Examples:
                         "Average cross-sectional factor correlation = 0.15-0.20"
                         "5-factor model insensitive to 2x3 vs 2x2x2x2 sorts"

  DOMAIN_FACT        — Real-economy observation, not a directly tradable
                       claim. Sometimes the basis for a downstream factor
                       but as stated it is not a strategy.
                       Examples:
                         "Customer-supplier firms have correlated sales"
                         "Liquidity risk premium = 24 bps per month in cross-section"

  OTHER              — None of the above (true catch-all). Use sparingly;
                       prefer one of the typed categories above.

If claim_type is NOT FACTOR_HYPOTHESIS, set family=OTHER and the
remaining factor fields (legs/universe/construction/risk/outcome)
to UNKNOWN defaults — they will be ignored downstream. Still emit
the claim so it persists as research evidence.

# STEP 2 — only if claim_type=FACTOR_HYPOTHESIS, classify family
family decision tree (apply in ORDER, stop at first match):
  - mentions "carry" / "roll yield" / "forward discount" / "term premium"     → CARRY
  - mentions "momentum" / "trend" / "TSMOM" / "12-1" / "12-month"             → MOMENTUM
  - mentions "reversal" / "1-month reversal" / "long-term reversal"           → REVERSAL
  - mentions "value" / "book-to-market" / "B/M" / "P/E" / "PPP"               → VALUE
  - mentions "quality" / "QMJ" / "profitability" (operating margin)           → QUALITY or PROFITABILITY
  - mentions "low vol" / "BAB" / "betting against beta" / "low beta"          → LOW_VOL
  - mentions "size" / "small cap" / "microcap" premium                        → SIZE
  - mentions "investment" / "asset growth" / "CMA"                            → INVESTMENT
  - mentions "VRP" / "variance swap" / "delta-hedged" / "implied vs realized" → VOL_RISK_PREMIUM
  - mentions "term structure of vol" / "volatility curve"                     → TERM_STRUCTURE
  - mentions "short interest" / "days to cover"                               → SHORT_INTEREST
  - mentions "attention" / "Google Trends" / "SVI" / "Wikipedia views"        → ATTENTION
  - mentions "PEAD" / "earnings drift" / "SUE" / "analyst revision"           → EARNINGS_DRIFT
  - mentions "sentiment" / "media tone" / "FinBERT" / "WSJ pessimism"         → SENTIMENT
  - mentions "supply chain" / "customer-supplier link"                        → SUPPLY_CHAIN
  - mentions "implied vol skew" / "put-call ratio" / "OPTIONS_*"              → OPTIONS_IMPLIED
  - mentions "13F" / "holdings breadth" / "institutional ownership"           → HOLDINGS_BASED
  - cross-asset trend / TSMOM across asset classes                            → CROSS_ASSET_MOMENTUM
  - none fit → OTHER (very rare for a FACTOR_HYPOTHESIS)

# Allowed values (USE EXACTLY)
claim_type:    {", ".join(t.value for t in ClaimType)}
family:        {", ".join(f.value for f in FamilyV2)}
asset_class:   {", ".join(a.value for a in AssetClass)}
signal_type:   {", ".join(s.value for s in SignalType)}
sign:          {", ".join(s.value for s in Sign)}
subset:        {", ".join(u.value for u in UniverseSubset)}
weighting:     {", ".join(w.value for w in Weighting)}
rebalance:     {", ".join(r.value for r in Rebalance)}
direction:     {", ".join(d.value for d in PredictedDirection)}

# Factor-spec examples (only when claim_type=FACTOR_HYPOTHESIS)
- "FX carry on G10 monthly, BAB-style weighting"
  → claim_type=FACTOR_HYPOTHESIS, family=CARRY, asset_class=FX,
    signal_type=CARRY_FORWARD_DISCOUNT, subset=G10,
    weighting=INV_VOL (BAB ≈ inv_vol), rebalance=MONTHLY

- "Time-series momentum 12-month on commodity futures"
  → claim_type=FACTOR_HYPOTHESIS, family=MOMENTUM, asset_class=COMMODITY,
    signal_type=MOMENTUM_TSMOM_12, sign=TIMESERIES, rebalance=MONTHLY

- "Carry × momentum filter on cross-asset basket"
  → claim_type=FACTOR_HYPOTHESIS, family=CARRY, asset_class=COMBINED,
    TWO legs:
    leg 1: signal_type=CARRY_ROLL_YIELD, role=primary
    leg 2: signal_type=MOMENTUM_12_1,     role=filter

# Non-factor examples (claim_type != FACTOR_HYPOTHESIS)
- "HLZ argue minimum t-ratio for new factors should be 3.0 not 2.0"
  → claim_type=METHODOLOGY, family=OTHER, legs/universe/etc all UNKNOWN
- "Mean market impact for institutional equity trades ≈ 12 bps"
  → claim_type=MICROSTRUCTURE, family=OTHER, legs/universe/etc all UNKNOWN
- "Post-publication Sharpe decay averages 35 percent across 82 anomalies"
  → claim_type=DECAY_STUDY, family=OTHER, legs/universe/etc all UNKNOWN
- "Customer-supplier linked firms have correlated operating income"
  → claim_type=DOMAIN_FACT, family=OTHER, legs/universe/etc all UNKNOWN

# F12 (2026-06-05) — UNKNOWN is NOT a safe default
#
# Pre-F12 the backfill of 209 hypotheses produced ~120 UNKNOWN-bucket
# gaps (58 WEIGHTING=UNKNOWN, 24 SIGNAL=UNKNOWN, 16 REBALANCE=UNKNOWN,
# 23 UNIVERSE=EQUITY__UNKNOWN). All 88 FACTOR_HYPOTHESIS specs were
# missing_components, blocking Composer.
#
# DOCTRINE: each UNKNOWN field on a FACTOR_HYPOTHESIS is EVIDENCE that
# the claim is NOT actually a tradable strategy specification — it's
# probably a METHODOLOGY claim (factor model spec / replication
# critique / multiple-testing argument), a DOMAIN_FACT (cross-sectional
# correlation observation), or a FACTOR_STRUCTURE statement (factor
# model fit). A TRADABLE strategy specifies WHAT to trade
# (signal_type), WHERE (universe + subset), HOW MUCH (weighting), and
# WHEN to rebalance.
#
# Rules:
#   1. If you cannot identify ANY of {{signal_type, asset_class,
#      weighting, rebalance}} from the claim text, the claim is NOT
#      FACTOR_HYPOTHESIS. Re-classify to METHODOLOGY / DOMAIN_FACT /
#      FACTOR_STRUCTURE / DECAY_STUDY based on what it IS about.
#   2. UNKNOWN signal_type on FACTOR_HYPOTHESIS = invalid output. If
#      the claim mentions buying/selling assets but you can't pick a
#      SignalType enum value, pick the CLOSEST one and put your
#      uncertainty in the leg.note field, but DO NOT pick UNKNOWN.
#   3. UNKNOWN weighting on FACTOR_HYPOTHESIS = invalid. If unstated,
#      default to EQUAL (the academic baseline for factor portfolios).
#      Mention in outcome.rationale that weighting was unstated.
#   4. UNKNOWN rebalance on FACTOR_HYPOTHESIS = invalid. If unstated,
#      default to MONTHLY (the default cadence for equity factors).
#      Same rationale note.
#   5. UNKNOWN subset on FACTOR_HYPOTHESIS = use ALL for the asset_class.
#   6. asset_class can legitimately be UNKNOWN only if the claim is
#      cross-asset → use COMBINED instead.
#
# In short: UNKNOWN is a signal you've mis-classified claim_type.
# Re-classify rather than dump UNKNOWN.

# Confidence calibration
1.0  every field unambiguous from the claim
0.8  claim_type clear + family + asset_class + signal_type clear
0.5  claim_type clear but factor fields required best-guess (or non-factor
     claim with all required fields trivially UNKNOWN)
0.3  claim is so vague that >50% of fields are UNKNOWN or guessed
0.0  refuse — claim is not a quant-research claim at all
"""


def _tool_schema() -> dict:
    """The forced-output schema. Anthropic's tool_use enforces this."""
    return {
        "name": "emit_hypothesis_spec",
        "description": "Emit the structured hypothesis spec",
        "input_schema": {
            "type": "object",
            "required": ["claim_type", "family", "legs", "universe", "construction",
                         "risk", "outcome", "confidence"],
            "properties": {
                "claim_type": {"type": "string", "enum": [t.value for t in ClaimType]},
                "family":     {"type": "string", "enum": [f.value for f in FamilyV2]},
                "legs": {
                    "type":  "array",
                    "items": {
                        "type": "object",
                        "required": ["signal_type", "sign", "lookback_periods", "role"],
                        "properties": {
                            "signal_type":      {"type": "string", "enum": [s.value for s in SignalType]},
                            "sign":             {"type": "string", "enum": [s.value for s in Sign]},
                            "lookback_periods": {"type": "array", "items": {"type": "integer"}},
                            "quantile":         {"type": "number"},
                            "role":             {"type": "string", "enum": ["primary", "filter", "overlay"]},
                            "note":             {"type": "string"},
                        },
                    },
                },
                "universe": {
                    "type": "object",
                    "required": ["asset_class", "subset"],
                    "properties": {
                        "asset_class":  {"type": "string", "enum": [a.value for a in AssetClass]},
                        "subset":       {"type": "string", "enum": [u.value for u in UniverseSubset]},
                        "custom_tickers": {"type": ["array", "null"],
                                          "items": {"type": "string"}},
                        "min_history_months": {"type": "integer"},
                    },
                },
                "construction": {
                    "type": "object",
                    "required": ["weighting", "rebalance"],
                    "properties": {
                        "weighting":      {"type": "string", "enum": [w.value for w in Weighting]},
                        "rebalance":      {"type": "string", "enum": [r.value for r in Rebalance]},
                        "skip_first_day": {"type": "boolean"},
                        "holding_period_n": {"type": "integer"},
                    },
                },
                "risk": {
                    "type": "object",
                    "properties": {
                        "vol_target_annual":   {"type": ["number", "null"]},
                        "max_leverage":        {"type": ["number", "null"]},
                        "turnover_cap_annual": {"type": ["number", "null"]},
                        "max_position":        {"type": ["number", "null"]},
                        "drawdown_stop":       {"type": ["number", "null"]},
                    },
                },
                "outcome": {
                    "type": "object",
                    "required": ["predicted_direction"],
                    "properties": {
                        "predicted_direction": {"type": "string", "enum": [d.value for d in PredictedDirection]},
                        "predicted_sharpe_lo": {"type": ["number", "null"]},
                        "predicted_sharpe_hi": {"type": ["number", "null"]},
                        "rationale":           {"type": "string"},
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    }


def _build_user_msg(*, claim_text: str, family_hint: Optional[str],
                   subtype_hint: Optional[str]) -> str:
    parts = []
    if family_hint:
        parts.append(f"family hint (from existing classification): {family_hint}")
    if subtype_hint:
        parts.append(f"mechanism_subtype hint: {subtype_hint}")
    parts.append(f"\nCLAIM:\n{claim_text}")
    return "\n".join(parts)


def extract_spec(
    *,
    source_hypothesis_id: str,
    claim_text:           str,
    mechanism_family:     Optional[str] = None,
    mechanism_subtype:    Optional[str] = None,
    git_sha:              str = "",
    workload_override:    Optional[str] = None,
) -> Optional[HypothesisSpec]:
    """Run the LLM extractor. Returns a HypothesisSpec or None on
    hard failure (no api key / api error / bad output).

    workload_override (R1 cost-route audit, 2026-06-05): if provided,
    routes via that workload key instead of 'spec_drafter'. Used by
    audit_cost_route_spec_drafter.py to A/B-test deepseek vs anthropic.
    Default None preserves current production routing."""
    if not claim_text or not claim_text.strip():
        logger.warning("extract_spec: empty claim_text")
        return None

    try:
        from engine.llm.call import call as llm_call
    except ImportError as exc:
        logger.exception("extract_spec: engine.llm.call unavailable: %s", exc)
        return None

    user_msg = _build_user_msg(
        claim_text   = claim_text,
        family_hint  = mechanism_family,
        subtype_hint = mechanism_subtype,
    )
    tool = _tool_schema()

    try:
        result = llm_call(
            workload   = workload_override or "spec_drafter",
            system     = _system_prompt(),
            user       = user_msg,
            agent_id   = "spec_drafter",
            tools      = [tool],
            max_tokens = 1200,
            scope      = "hypothesis_spec_extract",
        )
    except Exception as exc:
        logger.exception("extract_spec: llm_call failed: %s", exc)
        return None

    # Pull the tool call payload
    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_hypothesis_spec":
            payload = tc.input
            break
    if payload is None:
        logger.warning("extract_spec: model returned text instead of tool_call")
        return None

    try:
        return _payload_to_spec(
            payload,
            source_hypothesis_id = source_hypothesis_id,
            claim_text           = claim_text,
            git_sha              = git_sha,
        )
    except Exception as exc:
        logger.exception("extract_spec: payload → spec failed: %s", exc)
        return None


def _payload_to_spec(
    payload:              dict,
    *,
    source_hypothesis_id: str,
    claim_text:           str,
    git_sha:              str,
) -> HypothesisSpec:
    """Defensive conversion. Anything missing/invalid → UNKNOWN."""
    confidence = float(payload.get("confidence", 0.5))

    # ClaimType (B.2-A2): step-1 classification. Defaults to UNKNOWN
    # which will be caught by Composer / direction_proposer filter.
    try:
        claim_type = ClaimType(payload.get("claim_type") or "UNKNOWN")
    except ValueError:
        claim_type = ClaimType.UNKNOWN

    # Family
    try:
        family = FamilyV2(payload["family"])
    except (KeyError, ValueError):
        family = FamilyV2.OTHER

    # Cross-check: if claim_type is non-factor, family MUST be OTHER.
    # Defensively overwrite — the prompt asks the LLM to do this but
    # we don't trust the LLM to never drift.
    if claim_type not in (ClaimType.FACTOR_HYPOTHESIS, ClaimType.UNKNOWN):
        if family != FamilyV2.OTHER:
            logger.info(
                "extract_spec: claim_type=%s but family=%s; "
                "forcing family=OTHER for non-factor claim",
                claim_type.value, family.value,
            )
            family = FamilyV2.OTHER

    # Legs
    legs_raw = payload.get("legs") or []
    if not isinstance(legs_raw, list) or not legs_raw:
        legs_raw = [{
            "signal_type":      "UNKNOWN",
            "sign":             "UNKNOWN",
            "lookback_periods": [12],
            "role":             "primary",
        }]
    legs = []
    for L in legs_raw:
        try:
            st = SignalType(L.get("signal_type") or "UNKNOWN")
        except ValueError:
            st = SignalType.UNKNOWN
        try:
            sg = Sign(L.get("sign") or "UNKNOWN")
        except ValueError:
            sg = Sign.UNKNOWN
        lp = L.get("lookback_periods") or [12]
        if not isinstance(lp, list) or not all(isinstance(x, int) for x in lp):
            lp = [12]
        legs.append(SignalLeg(
            signal_type      = st,
            sign             = sg,
            lookback_periods = tuple(lp),
            quantile         = float(L.get("quantile", 0.30)),
            role             = str(L.get("role", "primary")),
            note             = str(L.get("note", "")),
        ))

    # Universe
    u_raw = payload.get("universe") or {}
    try:
        ac = AssetClass(u_raw.get("asset_class") or "UNKNOWN")
    except ValueError:
        ac = AssetClass.UNKNOWN
    try:
        sub = UniverseSubset(u_raw.get("subset") or "UNKNOWN")
    except ValueError:
        sub = UniverseSubset.UNKNOWN
    custom = u_raw.get("custom_tickers")
    universe = Universe(
        asset_class        = ac,
        subset             = sub,
        custom_tickers     = tuple(custom) if isinstance(custom, list) else None,
        # `or 36` covers BOTH missing-key and explicit-null-value cases:
        # Claude omits the field when it has no preference; Deepseek tends
        # to emit explicit `null` in the JSON tool call — both should
        # default to 36 rather than crash int(None). Provider-agnostic.
        min_history_months = int(u_raw.get("min_history_months") or 36),
    )

    # Construction
    c_raw = payload.get("construction") or {}
    try:
        w = Weighting(c_raw.get("weighting") or "UNKNOWN")
    except ValueError:
        w = Weighting.UNKNOWN
    try:
        rb = Rebalance(c_raw.get("rebalance") or "UNKNOWN")
    except ValueError:
        rb = Rebalance.UNKNOWN
    construction = PortfolioConstruction(
        weighting        = w,
        rebalance        = rb,
        skip_first_day   = bool(c_raw.get("skip_first_day", True)),
        holding_period_n = int(c_raw.get("holding_period_n", 1)),
    )

    # Risk
    r_raw = payload.get("risk") or {}
    risk = RiskManagement(
        vol_target_annual   = r_raw.get("vol_target_annual"),
        max_leverage        = r_raw.get("max_leverage"),
        turnover_cap_annual = r_raw.get("turnover_cap_annual"),
        max_position        = r_raw.get("max_position"),
        drawdown_stop       = r_raw.get("drawdown_stop"),
    )

    # Outcome
    o_raw = payload.get("outcome") or {}
    try:
        pd_ = PredictedDirection(o_raw.get("predicted_direction") or "UNKNOWN")
    except ValueError:
        pd_ = PredictedDirection.UNKNOWN
    outcome = PredictedOutcome(
        predicted_direction = pd_,
        predicted_sharpe_lo = o_raw.get("predicted_sharpe_lo"),
        predicted_sharpe_hi = o_raw.get("predicted_sharpe_hi"),
        rationale           = str(o_raw.get("rationale", "")),
    )

    # F12 (2026-06-05) auto-downgrade: if claim_type=FACTOR_HYPOTHESIS but
    # too many core fields are UNKNOWN, the LLM's classification is wrong.
    # Re-classify as METHODOLOGY (most common true target) + leave breadcrumb.
    if claim_type == ClaimType.FACTOR_HYPOTHESIS:
        unknown_signal_legs = sum(1 for L in legs
                                   if L.signal_type == SignalType.UNKNOWN)
        unknown_count = sum([
            unknown_signal_legs == len(legs),     # ALL legs UNKNOWN
            ac == AssetClass.UNKNOWN,
            sub == UniverseSubset.UNKNOWN,
            w == Weighting.UNKNOWN,
            rb == Rebalance.UNKNOWN,
        ])
        if unknown_count >= _F12_MAX_UNKNOWN_BEFORE_DOWNGRADE:
            logger.info(
                "F12 auto-downgrade: hyp=%s had %d UNKNOWN core fields under "
                "FACTOR_HYPOTHESIS — reclassified to METHODOLOGY",
                source_hypothesis_id, unknown_count,
            )
            claim_type = ClaimType.METHODOLOGY
            family     = FamilyV2.OTHER
            # Cut confidence — the original LLM verdict was wrong
            confidence = min(confidence, 0.45)

    extraction = Extraction(
        method       = EXTRACTOR_VERSION,
        confidence   = confidence,
        extracted_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        extractor_v  = "v3",
    )

    return HypothesisSpec.new(
        source_hypothesis_id = source_hypothesis_id,
        claim_type           = claim_type,
        family               = family,
        claim_text           = claim_text,
        legs                 = tuple(legs),
        universe             = universe,
        construction         = construction,
        risk                 = risk,
        outcome              = outcome,
        extraction           = extraction,
        git_sha              = git_sha,
    )
