"""tests/test_risk_manager_cb_absorption.py — Phase 9 G4 verdict gate.

Covers:
  - G4 parity: when no RM alerts exist, unified_circuit_state returns
    field-equal CircuitBreakerState to legacy evaluate()
  - RM SEVERE escalation: unified_circuit_state.level=='severe' when
    RiskManagerAlert table has SEVERE entry
  - Read/write separation: unified_circuit_state has NO side effect
  - persist_risk_manager_severe writes via legacy set_external_halt_flag
  - get_circuit_state_breakdown distinguishes legacy / RM / unified

Test isolation: uses date(2099, ...) sentinels + manual_reset cleanup.
"""
from __future__ import annotations

import dataclasses
import datetime

import pytest

from engine.agents.risk_manager.cb_absorption import (
    unified_circuit_state,
    persist_risk_manager_severe,
    get_circuit_state_breakdown,
    _query_risk_manager_worst_today,
)
from engine.agents.risk_manager.gates import Breach
from engine.agents.risk_manager.persist import persist_breaches_to_db
from engine.circuit_breaker import evaluate as legacy_eval, manual_reset
from engine.db_models import RiskManagerAlert
from engine.memory import init_db, SessionFactory


@pytest.fixture
def cleanup_test_state():
    """Per-test sentinel date + persistent CB + DB cleanup."""
    init_db()
    test_date = datetime.date(2099, 1, 1)

    # Pre-test cleanup
    try: manual_reset("test pre-cleanup")
    except Exception: pass
    s = SessionFactory()
    try:
        s.query(RiskManagerAlert).filter(RiskManagerAlert.date == test_date).delete()
        s.commit()
    finally:
        s.close()

    yield test_date

    # Post-test cleanup
    try: manual_reset("test post-cleanup")
    except Exception: pass
    s = SessionFactory()
    try:
        s.query(RiskManagerAlert).filter(RiskManagerAlert.date == test_date).delete()
        s.commit()
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# G4 parity — field equality when no RM alerts
# ──────────────────────────────────────────────────────────────────────────────
class TestG4Parity:
    def test_no_rm_alerts_field_equal_legacy(self, cleanup_test_state):
        # Use a date far in the past where no RM alerts could exist
        parity_date = datetime.date(2020, 1, 1)
        legacy = legacy_eval(parity_date)
        unified = unified_circuit_state(parity_date)
        assert dataclasses.asdict(legacy) == dataclasses.asdict(unified), (
            f"G4 parity broken: {dataclasses.asdict(legacy)} != {dataclasses.asdict(unified)}"
        )

    def test_g4_parity_object_identity_NOT_guaranteed(self, cleanup_test_state):
        """Docstring contract: identity NOT preserved across separate calls."""
        parity_date = datetime.date(2020, 1, 1)
        l1 = legacy_eval(parity_date)
        l2 = legacy_eval(parity_date)
        # Legacy itself returns different instances per call (verified upstream)
        # so unified can't preserve identity to either; only field equality
        assert l1 is not l2     # confirm legacy never gives identity


# ──────────────────────────────────────────────────────────────────────────────
# RM SEVERE escalation
# ──────────────────────────────────────────────────────────────────────────────
class TestEscalation:
    def test_rm_severe_dominates_legacy_none(self, cleanup_test_state):
        # Insert HARD_HALT RM alert
        b = Breach("1", "HARD_HALT", "test escalation", 0.10, 0.05,
                   ("TEST_X",), {}, "s")
        persist_breaches_to_db([b], cleanup_test_state,
                               phase="pre_trade", halt_decision=True)
        # Now unified should escalate
        unified = unified_circuit_state(cleanup_test_state)
        assert unified.level == "severe"
        # And legacy alone (no persistence side effect from unified) should still be none
        legacy = legacy_eval(cleanup_test_state)
        assert legacy.level == "none"

    def test_unified_is_read_only(self, cleanup_test_state):
        """Critical: unified_circuit_state must NOT mutate persistent CB state."""
        b = Breach("1", "HARD_HALT", "no-side-effect test", 0.10, 0.05,
                   ("X",), {}, "s")
        persist_breaches_to_db([b], cleanup_test_state,
                               phase="pre_trade", halt_decision=True)
        # Before reading unified — legacy is clean
        assert legacy_eval(cleanup_test_state).level == "none"
        # Read unified (RM dominates)
        unified = unified_circuit_state(cleanup_test_state)
        assert unified.level == "severe"
        # After reading — legacy STILL clean (no persistence side effect)
        assert legacy_eval(cleanup_test_state).level == "none"


# ──────────────────────────────────────────────────────────────────────────────
# Explicit persistence (write path)
# ──────────────────────────────────────────────────────────────────────────────
class TestExplicitPersistence:
    def test_persists_severe_when_rm_severe(self, cleanup_test_state):
        b = Breach("1", "HARD_HALT", "explicit persist test", 0.10, 0.05,
                   ("Y",), {}, "s")
        persist_breaches_to_db([b], cleanup_test_state,
                               phase="pre_trade", halt_decision=True)
        # Persist via the explicit function
        state = persist_risk_manager_severe(cleanup_test_state,
                                             source="test_explicit")
        assert state is not None
        assert state.level == "severe"
        # Legacy now sees SEVERE
        assert legacy_eval(cleanup_test_state).level == "severe"

    def test_returns_none_when_no_severe(self, cleanup_test_state):
        # No RM alerts in DB → no SEVERE to persist
        result = persist_risk_manager_severe(cleanup_test_state, source="test_noop")
        assert result is None

    def test_returns_none_when_only_soft_warn(self, cleanup_test_state):
        # Insert SOFT_WARN only → not SEVERE-eligible
        b = Breach("2", "SOFT_WARN", "soft only", 0.15, 0.10,
                   ("etf_l1",), {}, "s")
        persist_breaches_to_db([b], cleanup_test_state,
                               phase="pre_trade", halt_decision=False)
        result = persist_risk_manager_severe(cleanup_test_state, source="test_warn_only")
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic breakdown helper
# ──────────────────────────────────────────────────────────────────────────────
class TestBreakdownHelper:
    def test_breakdown_legacy_only(self, cleanup_test_state):
        # No RM alerts → breakdown shows legacy only
        brk = get_circuit_state_breakdown(cleanup_test_state)
        assert brk["risk_manager"] is None
        assert brk["unified"]["source"] == "legacy"

    def test_breakdown_rm_dominates_correctly(self, cleanup_test_state):
        b = Breach("1", "HARD_HALT", "breakdown test", 0.10, 0.05,
                   ("Z",), {}, "s")
        persist_breaches_to_db([b], cleanup_test_state,
                               phase="pre_trade", halt_decision=True)
        brk = get_circuit_state_breakdown(cleanup_test_state)
        # Risk manager side shows severe
        assert brk["risk_manager"] is not None
        assert brk["risk_manager"]["level"] == "severe"
        # Legacy still clean (read-only contract)
        assert brk["legacy"]["level"] == "none"
        # Unified attributes source correctly
        assert brk["unified"]["source"] == "risk_manager"


# ──────────────────────────────────────────────────────────────────────────────
# Worst-today query helper
# ──────────────────────────────────────────────────────────────────────────────
class TestWorstTodayQuery:
    def test_returns_none_on_empty(self, cleanup_test_state):
        # No alerts for date
        assert _query_risk_manager_worst_today(cleanup_test_state) is None

    def test_returns_worst_when_multiple(self, cleanup_test_state):
        # Insert SOFT_WARN + HARD_HALT
        b1 = Breach("2", "SOFT_WARN", "soft", 0.15, 0.10, ("etf_l1",), {}, "s")
        b2 = Breach("1", "HARD_HALT", "hard", 0.10, 0.05, ("X",), {}, "s")
        persist_breaches_to_db([b1, b2], cleanup_test_state,
                               phase="pre_trade", halt_decision=True)
        worst = _query_risk_manager_worst_today(cleanup_test_state)
        assert worst is not None
        # cb_severity will be SEVERE for both rows (computed once per call)
        # so the worst-rank picks one of them — level is "severe" either way
        assert worst["level"] == "severe"
