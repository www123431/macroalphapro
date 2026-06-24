"""
tests/test_wave_fixes_2026_05_07.py — invariants for the Wave 1+2+5 fixes.

These guard against regressions on:
  · H1 — DSR Bailey-LdP 2014 eq.10 convention (kurt + 2)/4
  · H5 — _direction_from_pos helper covering signal_tsmom + target_weight fallback
  · Wave 5 — rule_capability_vs_data_congruence project-age skip + cadence ladder

Each test cites the source file:line of the production code under test so that
when the assertion fires in CI, debugging starts with the right symbol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# H1 — DSR kurt convention (engine/backtest.py:340-359)
# ─────────────────────────────────────────────────────────────────────────────

def _dsr_denom(kurt_excess: float, skew: float, sr: float, formula: str) -> float:
    """Return the denominator term used by the DSR formula.

    formula='blp' is the post-fix (kurt + 2)/4 convention.  We re-implement
    here so the test does not depend on the full backtest.compute_metrics
    surface (which pulls regime / yfinance / etc).
    """
    if formula == "blp":
        return 1 - skew * sr + (kurt_excess + 2) / 4.0 * sr ** 2
    if formula == "buggy":
        return 1 - skew * sr + (kurt_excess - 1) / 4.0 * sr ** 2
    raise ValueError(formula)


def test_dsr_blp_denom_normal_sr1():
    """Normal returns (skew=0, excess kurt=0), SR=1: denom should be 1.5."""
    denom = _dsr_denom(kurt_excess=0.0, skew=0.0, sr=1.0, formula="blp")
    assert abs(denom - 1.5) < 1e-9, f"BLP convention expects 1.5 for normal SR=1, got {denom}"


def test_dsr_old_buggy_denom_normal_sr1():
    """Documents the regression: old formula yielded 0.75 (variance underestimate)."""
    denom = _dsr_denom(kurt_excess=0.0, skew=0.0, sr=1.0, formula="buggy")
    assert abs(denom - 0.75) < 1e-9, f"Old buggy formula should yield 0.75, got {denom}"


def test_engine_backtest_uses_blp_formula():
    """Source-code invariant: engine/backtest.py contains the (kurt + 2)/4 form
    and does NOT contain the old (kurt - 1)/4 form."""
    import engine.backtest as bt
    src = open(bt.__file__, encoding="utf-8").read()
    assert "(kurt + 2) / 4" in src, "engine/backtest.py must contain BLP-correct (kurt + 2)/4"
    assert "(kurt - 1) / 4" not in src, "engine/backtest.py must not contain the old (kurt - 1)/4"


def test_dsr_blp_strictly_lower_dsr_in_visible_regime():
    """Round-trip through the full Φ((SR-SR*) * sqrt(T-1) / sqrt(denom))
    pipeline at SR=1, T=20, n_trials=2 (mid-regime where bug is visible)."""
    from scipy import stats
    sr = 1.0
    t = 20
    n_trials = 2
    skew = 0.0
    kurt = 0.0
    eg = 0.5772156649
    sr_star = (
        (1 - eg) * stats.norm.ppf(1 - 1 / max(n_trials, 2))
        + eg     * stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e))
    )
    denom_blp = _dsr_denom(kurt_excess=kurt, skew=skew, sr=sr, formula="blp")
    denom_buggy = _dsr_denom(kurt_excess=kurt, skew=skew, sr=sr, formula="buggy")
    dsr_blp   = float(stats.norm.cdf((sr - sr_star) * np.sqrt(t - 1) / np.sqrt(denom_blp)))
    dsr_buggy = float(stats.norm.cdf((sr - sr_star) * np.sqrt(t - 1) / np.sqrt(denom_buggy)))
    # The buggy formula has smaller denom → larger z-stat → higher DSR (overconfident)
    assert dsr_blp < dsr_buggy - 0.01, (
        f"BLP DSR ({dsr_blp:.4f}) must be strictly less than buggy DSR "
        f"({dsr_buggy:.4f}) in visible regime"
    )


# ─────────────────────────────────────────────────────────────────────────────
# H5 — _direction_from_pos tests REMOVED 2026-05-22 (test-debt cleanup).
# Evidence: engine.portfolio_tracker._direction_from_pos (formerly :599-622) was removed in
# a refactor; it is now defined NOWHERE and has NO live caller (the position->direction
# derivation was inlined into daily_batch). These 7 tests imported a non-existent symbol.
# The direction concept (超配/低配/标配) remains alive and is exercised via integration
# (daily_batch / db_models.direction columns / decision_context._direction_zh).
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Wave 5 — rule_capability_vs_data_congruence
# (engine/auto_audit_rules.py near end of file)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def project_age_seed():
    """Seed cycle_states so _project_age_days() > 0 in the isolated tempfile DB.

    The test conftest creates an empty DB; without at least one cycle_states
    row the age helper returns 0 and the capability rule short-circuits to
    None.  This fixture inserts a 7-day-old row and rolls back via the
    Wave-2 cleanup contract."""
    import datetime
    from engine.memory import init_db, SessionFactory
    from engine.db_models import CycleState
    init_db()
    seed_started = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    with SessionFactory() as s:
        # Idempotent — only seed if no cycle_states present
        if s.query(CycleState).count() == 0:
            s.add(CycleState(
                cycle_type="daily",
                as_of_date=seed_started.date(),
                status="completed",
                started_at=seed_started,
                finished_at=seed_started + datetime.timedelta(seconds=10),
                elapsed_s=10.0,
            ))
            s.commit()
    yield 7   # project age in days
    # No teardown — tempfile DB is removed when the pytest process exits.


def test_capability_rule_silent_when_within_cadence_margin(project_age_seed):
    """Rule must NOT fire when staleness < cadence even if skip_until=0.
    Guards against hyperactive false positives."""
    from engine.auto_audit_rules import (
        rule_capability_vs_data_congruence, CAPABILITY_REGISTRY,
    )
    # Mutate harking entry: skip threshold 0, cadence large → staleness < cadence
    idx = next(i for i, c in enumerate(CAPABILITY_REGISTRY)
               if c["capability"] == "harking_detection_active")
    orig_skip = CAPABILITY_REGISTRY[idx]["skip_until_age_d"]
    orig_cadence = CAPABILITY_REGISTRY[idx]["cadence_days"]
    CAPABILITY_REGISTRY[idx]["skip_until_age_d"] = 0
    CAPABILITY_REGISTRY[idx]["cadence_days"] = 9999  # impossibly large
    try:
        result = rule_capability_vs_data_congruence()
        if result is not None:
            harking_issue = next(
                (i for i in result["snapshot"]["issues"]
                 if i["capability"] == "harking_detection_active"),
                None,
            )
            assert harking_issue is None, "harking should be silent within cadence margin"
    finally:
        CAPABILITY_REGISTRY[idx]["skip_until_age_d"] = orig_skip
        CAPABILITY_REGISTRY[idx]["cadence_days"]    = orig_cadence


def test_capability_rule_high_when_2x_cadence(project_age_seed):
    """Adversarial: skip=0, cadence=1 day. On a project with age>=2 days
    rule fires HIGH for the empty harking_flags table."""
    from engine.auto_audit_rules import (
        rule_capability_vs_data_congruence, CAPABILITY_REGISTRY,
        _project_age_days,
    )
    assert _project_age_days() >= 2, (
        f"fixture should have seeded age >= 2d, got {_project_age_days()}"
    )
    idx = next(i for i, c in enumerate(CAPABILITY_REGISTRY)
               if c["capability"] == "harking_detection_active")
    orig_skip = CAPABILITY_REGISTRY[idx]["skip_until_age_d"]
    orig_cadence = CAPABILITY_REGISTRY[idx]["cadence_days"]
    CAPABILITY_REGISTRY[idx]["skip_until_age_d"] = 0
    CAPABILITY_REGISTRY[idx]["cadence_days"] = 1
    try:
        result = rule_capability_vs_data_congruence()
        assert result is not None
        target = next(
            i for i in result["snapshot"]["issues"]
            if i["capability"] == "harking_detection_active"
        )
        assert target["kind"] == "stale_or_empty"
        assert target["severity"] == "HIGH"
    finally:
        CAPABILITY_REGISTRY[idx]["skip_until_age_d"] = orig_skip
        CAPABILITY_REGISTRY[idx]["cadence_days"]    = orig_cadence


def test_capability_rule_schema_missing_branch(project_age_seed):
    """Defensive: schema mismatch surfaces as HIGH with kind=table_or_column_missing."""
    from engine.auto_audit_rules import (
        rule_capability_vs_data_congruence, CAPABILITY_REGISTRY,
    )
    CAPABILITY_REGISTRY.append({
        "capability":          "_test_phantom_for_pytest",
        "downstream_table":    "harking_flags",
        "downstream_filter":   "1=1",
        "downstream_time_col": "no_such_column_here",
        "cadence_days":        30,
        "skip_until_age_d":    0,
        "silenceable":         False,
    })
    try:
        result = rule_capability_vs_data_congruence()
        assert result is not None
        target = next(
            i for i in result["snapshot"]["issues"]
            if i["capability"] == "_test_phantom_for_pytest"
        )
        assert target["kind"] == "table_or_column_missing"
        assert target["severity"] == "HIGH"
    finally:
        CAPABILITY_REGISTRY.pop()


def test_capability_rule_registered_in_weekly():
    """Wave 5 wiring sanity — rule must be in WEEKLY_RULES for cron to fire it."""
    from engine.auto_audit_rules import WEEKLY_RULES, rule_capability_vs_data_congruence
    assert rule_capability_vs_data_congruence in WEEKLY_RULES
