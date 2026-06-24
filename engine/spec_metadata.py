"""
engine/spec_metadata.py — Per-spec metadata registry (Watchdog amendment 2).

Pre-registration: docs/spec_ops_watchdog_agent_v1.md (id=63) §2.1 amendment 2
(hash c0a5f989 — post-Path-E TC fix lesson, 2026-05-12).

Purpose
-------
The SpecRegistry table (engine.db_models.SpecRegistry) tracks WHEN each spec
was registered, its content hash, and amendment_log. It does NOT carry the
per-spec OPERATIONAL CONSTANTS that auditing rules need (e.g. each
strategy's locked tc_bps_per_event).

This module provides that registry as a hardcoded dict — populated manually
each time a strategy spec is registered. The dict is the single source of
truth for `engine.auto_audit_rules.rule_realized_tc_vs_spec_rate` (Watchdog
mode 9 per amendment 2).

INVARIANT (per feedback_etf_tc_tier_model.md standing rule):
  ETF specs MUST use tier-specific TC, not single-stock paradigm:
    Tier 1 ETF (SPY/QQQ/IWB/IWM/XLK):    0.5-1.5 bp roundtrip
    Tier 2 ETF (EWG/EWC/INDA regional):  1.5-3   bp roundtrip
    Tier 3 ETF (ICLN/SLV thematic):      2-5     bp roundtrip
    Single-stock universe:               10-30   bp roundtrip
  Future ETF spec registrations MUST audit TC by ticker tier; this registry
  must be updated alongside the spec_lock.

Adding a spec entry
-------------------
When registering a new strategy spec:
  1. Determine `tc_bps_per_event` via TC audit (see standing rule).
  2. Add the entry below, keyed by spec_id.
  3. If this spec is the new dominant strategy for its sleeve, set
     `is_primary=True` and toggle the OLD primary's `is_primary=False`.
  4. Update Watchdog mode 9 tests in tests/test_auto_audit_rules_watchdog_phase1.py
     so the per-spec branch coverage stays current.
"""
from __future__ import annotations

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Registry (amendment 2, 2026-05-12)
# ─────────────────────────────────────────────────────────────────────────────
#
# Keys = SpecRegistry.id (assigned at register_spec time).
# Values:
#   tc_bps_per_event:  expected per-event TC in basis points
#   sleeve_id:         which sleeve this strategy writes trades into
#   is_primary:        if True, this is the dominant strategy in its sleeve
#                       (used by rule_realized_tc_vs_spec_rate as the
#                        comparison anchor when multiple specs share a sleeve)
#   notes:             human-readable context
#
SPEC_TC_BPS_PER_EVENT: dict[int, dict] = {
    # ETF tier-1 sleeve (etf_l1)
    44: {
        "tc_bps_per_event": 8.0,
        "sleeve_id":        "etf_l1",
        "is_primary":       True,
        "notes":            "B++ Mass FDR 2026-05-04 (33 ETF Tier-1 universe; "
                            "historical TC convention pre-amendment 2 doctrine)",
    },
    61: {
        "tc_bps_per_event": 8.0,
        "sleeve_id":        "etf_l1",
        "is_primary":       True,
        "notes":            "Path K1 size-expanded 2026-05-12 (43 ETFs; "
                            "same 8bp convention as B++; production swap)",
    },
    64: {
        "tc_bps_per_event": 4.0,
        "sleeve_id":        "etf_l1",
        "is_primary":       False,
        "notes":            "Path E v1 + amendment 1 (FOMC event-driven; "
                            "4bp per-event after tier-specific correction "
                            "per feedback_etf_tc_tier_model.md)",
    },
    # Single-stock sleeve (ss_sp500)
    62: {
        "tc_bps_per_event": 30.0,
        "sleeve_id":        "ss_sp500",
        "is_primary":       True,
        "notes":            "Path D DHS Behavioral 2-factor 2026-05-12 "
                            "(single-stock roundtrip, 60d post-rdq events)",
    },
    # Crypto sleeve (crypto_btc_eth) — DEPRECATED 2026-05-13 evening.
    # Specs preserved in registry for falsification chain audit. Sleeve removed
    # from engine.portfolio_sleeves.ALLOWED_SLEEVES same day; replaced by
    # cta_defensive (Path O) which empirically delivers crisis alpha (PQTIX
    # 3/3 crisis-positive 2018/2020/2022) vs crypto SAA's ρ=0.32 marginal
    # diversification ("self-deceiving diverse" per user 2026-05-13).
    71: {
        "tc_bps_per_event": 25.0,
        "sleeve_id":        "crypto_btc_eth",
        "is_primary":       False,
        "notes":            "Path N Crypto TSMOM v1 alpha test 2026-05-13 FAIL "
                            "(Sharpe 0.26, 0/5 gates). DEPRECATED 2026-05-13 evening: "
                            "sleeve removed from ALLOWED_SLEEVES. Preserved for "
                            "falsification chain audit only.",
    },
    72: {
        "tc_bps_per_event": 25.0,
        "sleeve_id":        "crypto_btc_eth",
        "is_primary":       False,   # demoted on sleeve removal
        "notes":            "Path N-SAA Crypto Overlay v1 2026-05-13. "
                            "DEPRECATED 2026-05-13 evening: ρ=0.32 to equity = marginal "
                            "diversification; superseded by Path O CTA sleeve "
                            "(PQTIX/DBMF managed futures, empirically crisis-positive "
                            "in 2018/2020/2022). 0 production trades ever executed.",
    },
    # D-PEAD-Plus LLM sentiment supplement (Sprint I 2026-05-13 deep night,
    # spec id=74 hash d0532f8f). Tests LLM-as-INPUT-FEATURE (Pattern 1) for
    # D-PEAD baseline (id=62) augmentation. 0-LLM-in-DECISION doctrine amendment
    # applied: LLM extracts 5 features from earnings calls (Gemini 2.5 Flash temp=0,
    # hash-locked prompt), deterministic OLS combines with SUE, decision layer
    # pure Python. Pre-registration: 6 quarters 2024-Q2+ (post-Gemini-cutoff for
    # validity), dev/OOS time-split 3:6 quarters, single OOS run commitment,
    # 5 gates with PRIMARY=IC delta + bootstrap CI. Expected P(STRICT_PASS) 15-20%
    # per 12-lock design audit. NOT YET RUN — spec only registered; build pending.
    74: {
        "tc_bps_per_event": 30.0,        # SS-Tier-1 standing rule (single-stock S&P 500)
        "sleeve_id":        "ss_sp500",
        "is_primary":       False,       # D-PEAD baseline (id=62) still primary
        "notes":            "D-PEAD-Plus v1 LLM sentiment supplement 2026-05-13. "
                            "Pre-registered LOCKED. Hash 6d8e614e (post-amendment-1). "
                            "Amendment 1 (2026-05-13 same day, clarification +0 trials): "
                            "data source CRSP -> yfinance+Compustat due to NUS CRSP data lag "
                            "(msf max 2024-12-31 vs spec window 2026-Q2). All design locks preserved. "
                            "Tests LLM-as-feature Pattern 1 with 0-LLM-in-DECISION doctrine. "
                            "Loughran-McDonald 2011 + Frankel 2010 + Larcker 2012 + Hassan 2019 anchors.",
    },
    # CTA Defensive Overlay sleeve (cta_defensive) — Path O v1 (2026-05-13 evening,
    # spec id=73 hash 9630c2bb). SAA_DEPLOYABLE 5/5 gates: max DD 34%→30% (-3.6pp),
    # Sharpe-neutral (0.799→0.802), ρ=-0.19, all 3 crisis windows positive
    # (2018+6.34%, 2020+6.47%, 2022+11.61%).
    73: {
        "tc_bps_per_event": 25.0,
        "sleeve_id":        "cta_defensive",
        "is_primary":       True,
        "notes":            "Path O CTA Defensive Overlay v1 2026-05-13 SAA_DEPLOYABLE "
                            "(PQTIX 100% / 10% portfolio allocation; annual + 2% drift "
                            "rebalance; PIMCO TRENDS Managed Futures Strategy I-class "
                            "as crisis-positive manager outsourcing per Yale Endowment / "
                            "Bridgewater All Weather pattern; Moskowitz 2012 + "
                            "Hurst-Ooi-Pedersen 2017 + Faber 2007 anchors; SAA not alpha)",
    },
}

# Deviation threshold (amendment 2 §implementation contract): flag HIGH if
# |realized - locked| / locked > MODE_9_DEVIATION_THRESHOLD.
MODE_9_DEVIATION_THRESHOLD: float = 0.5

# Minimum trades-per-sleeve-window for deviation check to fire. Below this
# the median is too noisy.
MODE_9_MIN_TRADES_FOR_CHECK: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Public lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_spec_tc_metadata(spec_id: int) -> Optional[dict]:
    """Return the metadata dict for one spec_id, or None."""
    return SPEC_TC_BPS_PER_EVENT.get(int(spec_id))


def get_primary_tc_for_sleeve(sleeve_id: str) -> Optional[dict]:
    """
    Return the primary-strategy metadata for a sleeve (the entry with
    is_primary=True), or None if no spec is registered for this sleeve.

    Returns dict: {tc_bps_per_event, sleeve_id, is_primary, notes, spec_id}.
    """
    for sid, meta in SPEC_TC_BPS_PER_EVENT.items():
        if meta.get("sleeve_id") == sleeve_id and meta.get("is_primary"):
            return {**meta, "spec_id": sid}
    return None


def get_all_specs_for_sleeve(sleeve_id: str) -> list[dict]:
    """All registered specs that write trades into a given sleeve (sorted by
    spec_id). Each entry includes spec_id for the rule's audit trail."""
    return sorted(
        ({**meta, "spec_id": sid}
         for sid, meta in SPEC_TC_BPS_PER_EVENT.items()
         if meta.get("sleeve_id") == sleeve_id),
        key=lambda d: d["spec_id"],
    )


def get_known_sleeves() -> list[str]:
    """Distinct sleeve_id values referenced in the registry, sorted."""
    return sorted({meta["sleeve_id"] for meta in SPEC_TC_BPS_PER_EVENT.values()})
