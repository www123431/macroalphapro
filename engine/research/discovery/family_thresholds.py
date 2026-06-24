"""engine/research/discovery/family_thresholds.py — family-aware
confidence thresholds + LLM-family-recognition bonus.

Per [[project-e2e-smoke-v3-funnel-findings-2026-05-30]]:
The v3 smoke showed the Bitcoin ETF carry paper scored only 0.10 on
deterministic confidence because its abstract lacks Sharpe/t-stat/sample
markers — but the LLM correctly identified family="carry". This is a
real factor proposal that the calculator missed.

Senior fix (per "宁可漏报到 borderline,不要漏到 trash" doctrine):
  1. Two-tier queue: review (≥ family threshold) + borderline (≥ 0.30
     but below family threshold) + skip (< 0.30)
  2. LLM family bonus: when LLM recognized a known family but
     confidence < 0.30, bump to 0.30 floor so it lands in borderline
     for human spot-check
  3. Family-aware threshold: families with mechanism-narrative abstract
     conventions (carry / vol_carry / lead_lag) get looser threshold
     than families with numeric-marker conventions (value / momentum)

DESIGN PRINCIPLE: confidence-based filtering is correct; the fix is
RECOGNIZING the LLM's family identification as a structural signal
that supplements weak text markers, not replacing them.
"""
from __future__ import annotations

# Family → minimum confidence for review queue
FAMILY_THRESHOLDS: dict[str, float] = {
    # Strict-markers families (value/momentum literature typically
    # reports Sharpe + t-stat + sample window in abstract)
    "value":           0.50,
    "momentum":        0.50,
    "quality":         0.50,
    "low_vol":         0.50,
    "profitability":   0.50,
    "factor_model":    0.50,
    "investment":      0.50,
    "residual_momentum": 0.50,
    # Mechanism-narrative families (often describe mechanism, leave
    # numerics to body — looser threshold appropriate)
    "carry":           0.40,
    "vol_carry":       0.40,
    "tsmom":           0.45,
    "lead_lag":        0.40,
    "term_structure":  0.40,
    "cross_asset_carry": 0.40,
    "cross_asset_tsmom": 0.45,
    # Event-driven / behavioral (mixed; allow slight slack)
    "pead":            0.45,
    "post_earnings_drift": 0.45,
    "merger_arb":      0.45,
    "behavioral":      0.45,
    "event_drift":     0.45,
    "news_attention":  0.45,
    "earnings_quality": 0.45,
    # Replication / negative-evidence venues
    "replication":     0.40,
    # Unknown / fallback
    "unknown":         0.50,
}

DEFAULT_THRESHOLD = 0.50
BORDERLINE_THRESHOLD = 0.30
FAMILY_BONUS_FLOOR = 0.30


def threshold_for_family(family: str | None) -> float:
    """Return min confidence for review-queue routing based on family.

    Unknown or unspecified family → DEFAULT_THRESHOLD (strict).
    """
    if family in (None, "", "unknown", "Unknown"):
        return DEFAULT_THRESHOLD
    return FAMILY_THRESHOLDS.get(family, DEFAULT_THRESHOLD)


def adjust_confidence_for_llm_family(
    base_confidence: float, family: str | None,
) -> tuple[float, bool]:
    """If LLM mapped paper to a known family but confidence is below
    FAMILY_BONUS_FLOOR, bump to FAMILY_BONUS_FLOOR.

    Returns: (adjusted_confidence, bonus_was_applied)
    """
    if family in (None, "", "unknown", "Unknown"):
        return base_confidence, False
    if base_confidence < FAMILY_BONUS_FLOOR:
        return FAMILY_BONUS_FLOOR, True
    return base_confidence, False


def classify_confidence(
    confidence: float, family: str | None,
) -> str:
    """Route to one of: 'review' (≥ family threshold),
    'borderline' (≥ 0.30 but below threshold), 'skip' (< 0.30)."""
    threshold = threshold_for_family(family)
    if confidence >= threshold:
        return "review"
    if confidence >= BORDERLINE_THRESHOLD:
        return "borderline"
    return "skip"


def explain_routing(
    base_confidence: float, family: str | None,
) -> dict:
    """Full routing decision breakdown for audit trail."""
    adjusted, bonus = adjust_confidence_for_llm_family(
        base_confidence, family,
    )
    routing = classify_confidence(adjusted, family)
    threshold = threshold_for_family(family)
    return {
        "base_confidence":         round(base_confidence, 4),
        "family":                  family or "unknown",
        "family_threshold":        threshold,
        "family_bonus_applied":     bonus,
        "adjusted_confidence":     round(adjusted, 4),
        "borderline_floor":        BORDERLINE_THRESHOLD,
        "routing":                 routing,
    }
