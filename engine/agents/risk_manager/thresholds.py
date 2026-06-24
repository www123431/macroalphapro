"""
engine/agents/risk_manager/thresholds.py — Tier-3-locked risk thresholds.

Phase 3 of Risk Manager Agent v1.0 (spec id=69). 2026-05-19 §2.1a
amend split Q1 single-cap into Q1a (book absolute) + Q1b (intra
per-sleeve-class). Current hash via lookup_spec(69) / SpecRegistry.

All numeric thresholds for the 12 detectors live here as ONE frozen
dataclass. Mutating any field requires:
  1. Spec amendment row in spec_metadata.amendment_log
  2. New entry in THRESHOLDS_GOVERNANCE_LOG below
  3. Hash bump on the spec file (re-register_spec)

The lockdown test in tests/test_risk_manager_thresholds_locked.py
(Phase 9) compares the live RISK_THRESHOLDS values against a separate
hardcoded LOCKED_RISK_THRESHOLDS table, so any silent drift fails CI
before the change can be committed.

Anchor for each threshold: spec_risk_manager_agent_v1.md §2.1 + §2.1a
+ §3.1. Re-read spec section before changing.
"""
from __future__ import annotations

import dataclasses

from engine.strategies import SleeveClass


# ──────────────────────────────────────────────────────────────────────────────
# Two-tier single-ticker caps (spec §2.1a, Q1a + Q1b resolution).
#
# 2026-05-19 amendment — the original §2.1a (Q1) collapsed two distinct
# risk concepts into one cap, then patched cross-sleeve overlap with a
# Basel-III-style "conservative minimum" rule. GLD/TLT case made the
# composition flaw visible: a ticker held by both K1 BAB (equity factor,
# tight cap) and AC TLT/GLD (insurance, loose cap) hit the equity cap
# and HALTed, even though insurance's 5% allocation alone was within its
# own design budget.
#
# Institutional pattern (BlackRock Aladdin / AQR / Bridgewater PARC):
# decompose the single cap into two independent gates that defend
# different risks. No conservative-min, no exceptions, no sleeve-pair
# special cases — just two layers operating at different abstraction
# levels.
#
# ▸ Mode 1a (BOOK_SINGLE_TICKER_ABS_CAP) — operational/issuer risk.
#   Defends against a single ETF blowing up (delisting, tracking error,
#   counterparty failure). Applies UNIFORMLY across all sleeves.
#
# ▸ Mode 1b (SLEEVE_CLASS_INTRA_CAPS) — strategy concentration risk.
#   Defends against one strategy over-leaning on a single ticker.
#   Checks the strategy's INTRA-STRATEGY weight (signal.weights), not
#   the book-level aggregate. Per-sleeve-class because strategies have
#   different N (BAB ~45 ETFs / D-PEAD ~1500 stocks / AC 2 tickers).
#
# Combined book exposure = sum(sleeve_target × intra_weight) is then
# naturally bounded by these two caps without any ad-hoc aggregation
# rule. Cross-sleeve ticker overlap stops being a special case.
# ──────────────────────────────────────────────────────────────────────────────

# Mode 1a — book-level absolute cap. Applies to ALL tickers regardless
# of sleeve membership. Threshold = "any one ETF should not be > 25%
# of the book" (Aladdin single-name exposure limit standard).
BOOK_SINGLE_TICKER_ABS_CAP: float = 0.25


# Mode 1b — per-strategy intra-strategy ticker cap, by sleeve_class.
# These are caps on signal.weights values (a strategy's INTRA gross
# concentration), not on combined book weights. See gates.py
# gate_mode_1b_intra_sleeve_cap for the evaluation contract.
SLEEVE_CLASS_INTRA_CAPS: dict[SleeveClass, float] = {
    SleeveClass.ALPHA_EQUITY_LS:    0.15,   # BAB tertile: ~7-12% per ETF typical;
                                            #   15% covers β-neutralized edge cases
    SleeveClass.ALPHA_SINGLE_STOCK: 0.05,   # 1500-name universe → no single name > 5%
    SleeveClass.INSURANCE:          0.50,   # legitimately 50/50 (AC TLT/GLD design)
    SleeveClass.CTA_OVERLAY:        1.00,   # single-fund overlay (PQTIX = 100% intra)
}


@dataclasses.dataclass(frozen=True)
class RiskThresholds:
    """All numeric thresholds for the 12 detectors. Single source of truth.

    Field semantics map 1:1 to spec §2.1 modes 1-10 plus the 6b/7b model-
    integrity tiers added by Q4 resolution.
    """
    # Mode 2 — relative sleeve drift (Q5 resolution: relative not absolute)
    sleeve_drift_relative_max:    float = 0.10    # 10% relative to target weight

    # Mode 3 — gross leverage cap (Tier-3 1.5× nominal + 10pp band)
    gross_leverage_max:           float = 1.60

    # Mode 4 — net exposure bounds
    net_exposure_min:             float = -0.50
    net_exposure_max:             float =  1.50

    # Mode 5 — Herfindahl-Hirschman concentration cap (Tier-3)
    hhi_max:                      float = 0.25

    # Mode 6 / 6b — 1-day VaR-95 (Q4 two-tier)
    var_95_soft_warn:             float = -0.03   # SOFT WARN: < -3% NAV
    var_95_hard_halt:             float = -0.09   # HARD HALT: < -9% NAV (3× threshold)

    # Mode 7 / 7b — 1-day ES-95 (Q4 two-tier)
    es_95_soft_warn:              float = -0.05   # SOFT WARN: < -5% NAV
    es_95_hard_halt:              float = -0.15   # HARD HALT: < -15% NAV (3× threshold)

    # Mode 8 — short-side aggregate cap (fraction of gross)
    short_side_max_of_gross:      float = 0.50

    # Mode 9 — minimum number of OK strategies (out of registry total)
    min_ok_strategies:            int   = 3

    # Mode 10 — max ticker count appearing long+short across strategies
    cross_cancel_ticker_max:      int   = 5

    # G3 ops alert (Q3 resolution): warn-narrate when VaR three-method dispersion
    # exceeds 20% (deployment gate G3 still uses 30%).
    var_method_dispersion_warn:   float = 0.20    # 20% relative dispersion warn
    var_method_dispersion_deploy: float = 0.30    # 30% deployment gate threshold


# Singleton — import this name; do NOT construct a fresh instance at runtime.
# Tests verify the singleton == LOCKED_RISK_THRESHOLDS to detect drift.
RISK_THRESHOLDS: RiskThresholds = RiskThresholds()


THRESHOLDS_GOVERNANCE_LOG: list[str] = [
    "2026-05-18 initial lockdown — Phase 3 of Risk Manager spec id=69 (initial hash f763a71774734af1, recorded in SpecRegistry); "
    "thresholds aligned with spec §2.1 + §2.1a + §3.1 (Q1-Q5 senior-reviewed values)",
    "2026-05-19 spec §2.1a amend (Q1 → Q1a + Q1b) — single-ticker cap split "
    "into two-tier structure to resolve cross-sleeve ticker overlap "
    "structurally. SLEEVE_CLASS_CAPS retired; replaced by BOOK_SINGLE_TICKER_"
    "ABS_CAP=0.25 (Mode 1a, operational/issuer risk) + SLEEVE_CLASS_INTRA_CAPS "
    "(Mode 1b, per-strategy intra-strategy concentration). No more "
    "conservative-min special case; both gates run independently. Trigger: "
    "shadow-phase 2026-05-19 surfaced GLD/TLT false-positive HALT in K1 "
    "BAB × AC TLT/GLD overlap. Spec hash rehashed.",
]
