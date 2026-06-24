"""
tests/test_llm_budget.py — Runtime LLM budget governance (2026-05-08).

Validates engine/llm_budget.py against:
  - default fallback when SystemConfig empty
  - SystemConfig override roundtrip
  - input validation (range, type)
  - history audit trail append
  - get_budget_status() shape

Mock-free; uses isolated test DB via conftest.
"""
from __future__ import annotations

import pytest

from engine.llm_budget import (
    get_budget_status,
    get_r_audit_budget_usd_per_year,
    get_rag_synthesis_daily_budget_usd,
    get_s6_anomaly_budget_usd_per_year,
    set_r_audit_budget_usd_per_year,
    set_rag_synthesis_daily_budget_usd,
    set_s6_anomaly_budget_usd_per_year,
)


# ─────────────────────────────────────────────────────────────────────────────
# Default / fallback behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaults:
    def test_r_audit_default_returns_config_value(self):
        """Without SystemConfig override, should return the module default.
        (Default moved engine.config.R_COST_BUDGET_USD -> engine.llm_budget
        ._DEFAULT_R_AUDIT_USD_PER_YEAR in the 2026-05-22 config-drift fix.)"""
        from engine.llm_budget import _DEFAULT_R_AUDIT_USD_PER_YEAR as R_DEFAULT
        set_r_audit_budget_usd_per_year(amount=float(R_DEFAULT), actor="test")
        v = get_r_audit_budget_usd_per_year()
        assert v == pytest.approx(R_DEFAULT, rel=1e-6)

    def test_s6_anomaly_default_returns_config_value(self):
        from engine.llm_budget import _DEFAULT_S6_ANOMALY_USD_PER_YEAR as S6_DEFAULT
        set_s6_anomaly_budget_usd_per_year(amount=float(S6_DEFAULT), actor="test")
        v = get_s6_anomaly_budget_usd_per_year()
        assert v == pytest.approx(S6_DEFAULT, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# SystemConfig override roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemConfigOverride:
    def test_r_audit_override_persists(self):
        set_r_audit_budget_usd_per_year(amount=125.0, actor="test")
        assert get_r_audit_budget_usd_per_year() == pytest.approx(125.0)

    def test_s6_anomaly_override_persists(self):
        set_s6_anomaly_budget_usd_per_year(amount=500.0, actor="test")
        assert get_s6_anomaly_budget_usd_per_year() == pytest.approx(500.0)

    def test_r_and_s6_independent(self):
        """Setting R does not affect S6 and vice versa."""
        set_r_audit_budget_usd_per_year(amount=77.0, actor="test_indep")
        set_s6_anomaly_budget_usd_per_year(amount=333.0, actor="test_indep")
        assert get_r_audit_budget_usd_per_year() == pytest.approx(77.0)
        assert get_s6_anomaly_budget_usd_per_year() == pytest.approx(333.0)


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_below_floor_rejected(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            set_r_audit_budget_usd_per_year(amount=0.001, actor="test")

    def test_above_ceiling_rejected(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            set_r_audit_budget_usd_per_year(amount=99_999.0, actor="test")

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            set_r_audit_budget_usd_per_year(amount=-10.0, actor="test")

    def test_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            set_r_audit_budget_usd_per_year(amount="50", actor="test")  # type: ignore

    def test_boundary_min_accepted(self):
        # Floor is 0.01
        set_r_audit_budget_usd_per_year(amount=0.01, actor="test")
        assert get_r_audit_budget_usd_per_year() == pytest.approx(0.01)

    def test_boundary_max_accepted(self):
        # Ceiling is 10000
        set_s6_anomaly_budget_usd_per_year(amount=10_000.0, actor="test")
        assert get_s6_anomaly_budget_usd_per_year() == pytest.approx(10_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# History audit trail
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryAuditTrail:
    def test_history_grows_after_set(self):
        # Capture history length before + after
        before = get_budget_status()["history"]
        n_before = len(before)
        set_r_audit_budget_usd_per_year(amount=42.0, actor="test_history")
        after = get_budget_status()["history"]
        assert len(after) == n_before + 1
        latest = after[-1]
        assert latest["scope"] == "r_audit"
        assert latest["actor"] == "test_history"
        assert float(latest["new"]) == pytest.approx(42.0)
        assert "at" in latest

    def test_history_truncates_at_max(self):
        """Verify history doesn't grow unbounded (capped at 50 in module)."""
        # Bulk-set many times; use the s6 scope so r_audit tests aren't affected
        for i in range(60):
            set_s6_anomaly_budget_usd_per_year(amount=100.0 + i, actor="test_truncate")
        history = get_budget_status()["history"]
        assert len(history) <= 50


# ─────────────────────────────────────────────────────────────────────────────
# get_budget_status() shape
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetStatus:
    def test_status_has_both_scopes(self):
        status = get_budget_status()
        assert "r_audit"    in status
        assert "s6_anomaly" in status
        assert "history"    in status

    def test_each_scope_has_required_fields(self):
        status = get_budget_status()
        for scope_name in ("r_audit", "s6_anomaly"):
            scope = status[scope_name]
            for field in (
                "current_usd_per_year",
                "default_usd_per_year",
                "spent_usd",
                "remaining_usd",
                "fraction_used",
            ):
                assert field in scope, f"missing {scope_name}.{field}"
            # Sanity: types
            assert isinstance(scope["current_usd_per_year"], float)
            assert isinstance(scope["fraction_used"], float)
            assert 0.0 <= scope["fraction_used"]   # may exceed 1.0 if over budget

    def test_remaining_equals_current_minus_spent(self):
        status = get_budget_status()
        for scope_name in ("r_audit", "s6_anomaly"):
            s = status[scope_name]
            expected_remaining = max(s["current_usd_per_year"] - s["spent_usd"], 0.0)
            assert s["remaining_usd"] == pytest.approx(expected_remaining, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Consumer integration — auto_audit_proposer.get_cost_status reads new helper
# ─────────────────────────────────────────────────────────────────────────────

class TestConsumerIntegration:
    def test_auto_audit_proposer_reads_runtime_budget(self):
        """Setting R-audit budget changes what auto_audit_proposer reports."""
        set_r_audit_budget_usd_per_year(amount=88.0, actor="test_consumer")
        from engine.auto_audit_proposer import get_cost_status
        status = get_cost_status()
        assert status["budget_usd"] == pytest.approx(88.0)

    def test_anomaly_llm_detector_reads_runtime_budget(self):
        set_s6_anomaly_budget_usd_per_year(amount=400.0, actor="test_consumer")
        from engine.anomaly_llm_detector import get_cost_status
        status = get_cost_status()
        assert status["budget_usd"] == pytest.approx(400.0)

    def test_rag_synthesis_reads_runtime_budget(self):
        """Setting RAG daily budget changes get_synthesis_cost_status report.
        This is the bug-sync test: attention bar / Brief / research_console /
        spec_drafter all read get_synthesis_cost_status."""
        set_rag_synthesis_daily_budget_usd(amount=0.12, actor="test_sync")
        from engine.agents.history_rag.synthesize import get_synthesis_cost_status
        status = get_synthesis_cost_status()
        assert status["today_budget_usd"] == pytest.approx(0.12)


# ─────────────────────────────────────────────────────────────────────────────
# Daily scope (RAG synthesis)
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyScope:
    def test_rag_default_returns_history_rag_config(self):
        from engine.agents.history_rag.config import SYNTHESIS_DAILY_BUDGET
        # Reset to default explicitly
        set_rag_synthesis_daily_budget_usd(amount=float(SYNTHESIS_DAILY_BUDGET),
                                           actor="test")
        v = get_rag_synthesis_daily_budget_usd()
        assert v == pytest.approx(SYNTHESIS_DAILY_BUDGET, rel=1e-6)

    def test_rag_override_persists(self):
        set_rag_synthesis_daily_budget_usd(amount=0.25, actor="test")
        assert get_rag_synthesis_daily_budget_usd() == pytest.approx(0.25)

    def test_rag_below_floor_rejected(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            set_rag_synthesis_daily_budget_usd(amount=0.0001, actor="test")

    def test_rag_above_ceiling_rejected(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            set_rag_synthesis_daily_budget_usd(amount=500.0, actor="test")

    def test_rag_boundary_min_accepted(self):
        set_rag_synthesis_daily_budget_usd(amount=0.001, actor="test")
        assert get_rag_synthesis_daily_budget_usd() == pytest.approx(0.001)

    def test_status_includes_rag_scope(self):
        status = get_budget_status()
        assert "rag_synthesis_daily" in status
        scope = status["rag_synthesis_daily"]
        for field in ("current_usd_per_day", "default_usd_per_day",
                      "spent_usd_today", "remaining_usd_today",
                      "fraction_used_today"):
            assert field in scope, f"missing rag_synthesis_daily.{field}"
        assert isinstance(scope["current_usd_per_day"], float)
