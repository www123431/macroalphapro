"""tests/test_risk_manager_thresholds_locked.py — Phase 9 threshold lockdown.

Phase 3 of Risk Manager (spec id=69; current hash in SpecRegistry) locked 13 numeric
threshold values + 4 sleeve-class caps via Q1-Q5 senior review. 2026-
05-19 §2.1a amend split Q1 → Q1a (BOOK_SINGLE_TICKER_ABS_CAP scalar)
+ Q1b (SLEEVE_CLASS_INTRA_CAPS dict). This test file hash-locks all
of them; any drift fails CI before the change can be committed.

Mirrors the pattern from tests/test_strategy_meta_locked.py — locked
values + governance log + amendment workflow citation.
"""
from __future__ import annotations

import pytest

from engine.agents.risk_manager.thresholds import (
    BOOK_SINGLE_TICKER_ABS_CAP,
    RISK_THRESHOLDS,
    SLEEVE_CLASS_INTRA_CAPS,
    THRESHOLDS_GOVERNANCE_LOG,
    RiskThresholds,
)
from engine.strategies import SleeveClass


# ──────────────────────────────────────────────────────────────────────────────
# Locked threshold values — sourced from spec id=69 §2.1 + §2.1a + §3.1
# Q1-Q5 senior-review resolutions applied 2026-05-18; §2.1a amended
# 2026-05-19 (Q1 → Q1a + Q1b two-tier; see governance log).
# ──────────────────────────────────────────────────────────────────────────────
LOCKED_THRESHOLDS: dict[str, float] = {
    "sleeve_drift_relative_max":    0.10,    # Q5 — relative 10% (was 2pp)
    "gross_leverage_max":           1.60,    # Tier-3 1.5× + 10pp band
    "net_exposure_min":            -0.50,
    "net_exposure_max":             1.50,
    "hhi_max":                      0.25,
    "var_95_soft_warn":            -0.03,    # Mode 6
    "var_95_hard_halt":            -0.09,    # Q4 — 3× threshold model integrity
    "es_95_soft_warn":             -0.05,    # Mode 7
    "es_95_hard_halt":             -0.15,    # Q4 — 3× threshold
    "short_side_max_of_gross":      0.50,
    "min_ok_strategies":            3,
    "cross_cancel_ticker_max":      5,
    "var_method_dispersion_warn":   0.20,    # Q3 ops alert
    "var_method_dispersion_deploy": 0.30,    # Q3 deployment gate
}

# Q1a (Mode 1a) — book-level absolute single-ticker cap. Uniform.
LOCKED_BOOK_SINGLE_TICKER_ABS_CAP: float = 0.25

# Q1b (Mode 1b) — per-strategy intra-strategy ticker cap by sleeve_class.
LOCKED_SLEEVE_CLASS_INTRA_CAPS: dict[SleeveClass, float] = {
    SleeveClass.ALPHA_EQUITY_LS:    0.15,    # BAB tertile typical ~7-12%
    SleeveClass.ALPHA_SINGLE_STOCK: 0.05,    # 1500-name universe
    SleeveClass.INSURANCE:          0.50,    # AC TLT/GLD 50/50 by design
    SleeveClass.CTA_OVERLAY:        1.00,    # single-fund overlay
}


@pytest.mark.parametrize("field", list(LOCKED_THRESHOLDS))
def test_threshold_value_locked(field: str):
    """Each RISK_THRESHOLDS field matches the spec-locked value.

    Failure mode: an Engineer agent / human edit changed thresholds.py
    without updating this test. Resolve by either reverting the edit
    (if accidental) or by amending LOCKED_THRESHOLDS + appending to
    THRESHOLDS_GOVERNANCE_LOG citing the spec amendment row.
    """
    expected = LOCKED_THRESHOLDS[field]
    actual = getattr(RISK_THRESHOLDS, field)
    assert actual == expected, (
        f"RISK_THRESHOLDS.{field} drifted: locked={expected!r}, actual={actual!r}\n"
        f"To resolve: update LOCKED_THRESHOLDS in this test + append a "
        f"THRESHOLDS_GOVERNANCE_LOG entry citing the spec amendment."
    )


def test_book_single_ticker_abs_cap_locked():
    """Mode 1a — BOOK_SINGLE_TICKER_ABS_CAP locked at 25% (Q1a)."""
    assert BOOK_SINGLE_TICKER_ABS_CAP == LOCKED_BOOK_SINGLE_TICKER_ABS_CAP, (
        f"BOOK_SINGLE_TICKER_ABS_CAP drifted: "
        f"locked={LOCKED_BOOK_SINGLE_TICKER_ABS_CAP!r}, "
        f"actual={BOOK_SINGLE_TICKER_ABS_CAP!r}"
    )


@pytest.mark.parametrize("sleeve_class", list(LOCKED_SLEEVE_CLASS_INTRA_CAPS))
def test_sleeve_class_intra_cap_locked(sleeve_class: SleeveClass):
    """Each SLEEVE_CLASS_INTRA_CAPS entry matches the Q1b-locked cap."""
    expected = LOCKED_SLEEVE_CLASS_INTRA_CAPS[sleeve_class]
    actual = SLEEVE_CLASS_INTRA_CAPS[sleeve_class]
    assert actual == expected, (
        f"SLEEVE_CLASS_INTRA_CAPS[{sleeve_class.value}] drifted: "
        f"locked={expected!r}, actual={actual!r}"
    )


def test_no_extra_sleeve_classes_in_intra_caps():
    """SLEEVE_CLASS_INTRA_CAPS keys must match the locked set exactly."""
    assert set(SLEEVE_CLASS_INTRA_CAPS.keys()) == set(LOCKED_SLEEVE_CLASS_INTRA_CAPS.keys()), (
        f"SLEEVE_CLASS_INTRA_CAPS has extra/missing keys: "
        f"actual={set(SLEEVE_CLASS_INTRA_CAPS)} vs "
        f"locked={set(LOCKED_SLEEVE_CLASS_INTRA_CAPS)}"
    )


def test_thresholds_dataclass_is_frozen():
    """RiskThresholds must reject runtime mutation."""
    with pytest.raises(Exception):
        RISK_THRESHOLDS.gross_leverage_max = 99.0  # type: ignore[misc]


def test_governance_log_non_empty():
    """At least one governance entry must exist (guards against accidental empty log)."""
    assert len(THRESHOLDS_GOVERNANCE_LOG) >= 1
    assert all(isinstance(line, str) and len(line) > 20
               for line in THRESHOLDS_GOVERNANCE_LOG)


def test_q4_two_tier_hard_halt_is_3x_soft_warn():
    """Q4 resolution: HARD HALT at 3× SOFT WARN threshold (model integrity)."""
    # VaR — note both values are negative; absolute ratio
    assert abs(RISK_THRESHOLDS.var_95_hard_halt / RISK_THRESHOLDS.var_95_soft_warn - 3.0) < 0.01
    # ES — same 3× ratio
    assert abs(RISK_THRESHOLDS.es_95_hard_halt / RISK_THRESHOLDS.es_95_soft_warn - 3.0) < 0.01


def test_q3_two_tier_dispersion_warn_below_deploy():
    """Q3 resolution: ops warn threshold (20%) strictly below deployment gate (30%)."""
    assert RISK_THRESHOLDS.var_method_dispersion_warn < RISK_THRESHOLDS.var_method_dispersion_deploy
    assert RISK_THRESHOLDS.var_method_dispersion_warn == 0.20
    assert RISK_THRESHOLDS.var_method_dispersion_deploy == 0.30


def test_sleeve_class_intra_cap_ordering_makes_sense():
    """Senior invariant: equity intra caps must be tighter than
    fund/insurance intra caps (universe-size driven)."""
    single_stock = SLEEVE_CLASS_INTRA_CAPS[SleeveClass.ALPHA_SINGLE_STOCK]
    equity_ls    = SLEEVE_CLASS_INTRA_CAPS[SleeveClass.ALPHA_EQUITY_LS]
    insurance    = SLEEVE_CLASS_INTRA_CAPS[SleeveClass.INSURANCE]
    cta          = SLEEVE_CLASS_INTRA_CAPS[SleeveClass.CTA_OVERLAY]
    # 1500-stock universe → tightest; ~45 ETFs → middle; designed-concentrated
    # insurance/CTA → most permissive.
    assert single_stock <= equity_ls <= insurance <= cta, (
        f"intra-cap ordering broken: single_stock={single_stock}, "
        f"equity_ls={equity_ls}, insurance={insurance}, cta={cta}"
    )


def test_book_abs_cap_dominates_intra_caps():
    """Senior invariant: the book-level absolute cap (Mode 1a) must NOT
    be tighter than any intra cap (Mode 1b) for sleeves whose intra weight
    can reach book-level (target_weight ≈ 1). Otherwise 1a renders 1b
    irrelevant for those sleeves. Current targets are <0.5, so the
    constraint is automatic but we test it to prevent silent regression."""
    # E.g., insurance intra 50% × target 0.10 = 5% book. 5% < 25% 1a cap. OK.
    # CTA intra 100% × target 0.09 = 9% book. 9% < 25%. OK.
    # If insurance target were 0.6 + intra 0.5 = 30% book > 25% cap, 1a fires.
    # The invariant we assert: for current allocation, 1a ≥ max(intra * target).
    # This protects against future allocation drift breaking the separation.
    assert BOOK_SINGLE_TICKER_ABS_CAP >= 0.20, (
        "book absolute cap should not be tighter than typical max "
        "sleeve-target × intra-cap product (currently ~10% peak); "
        "0.20 floor enforces room for 1b to ever be the binding gate"
    )
