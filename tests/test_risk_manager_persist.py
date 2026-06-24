"""tests/test_risk_manager_persist.py — Phase 9 persistence layer tests.

Covers:
  - Deterministic alert_id (re-run produces same UUID)
  - Idempotent UPSERT (re-persist replaces, doesn't duplicate)
  - NaN observed_value coerces to None for JSON-safe DB storage
  - update_narrative populates Phase 7 fields without overwriting other fields
  - query_recent_alerts filters by severity_min correctly
  - cb_severity computed once per call → all rows from one cycle share the same value

Test isolation: uses date(2099, ...) sentinels so cleanup is unambiguous.
"""
from __future__ import annotations

import datetime

import pytest

from engine.agents.risk_manager.gates import Breach
from engine.agents.risk_manager.persist import (
    breach_to_alert_row,
    make_alert_id,
    persist_breaches_to_db,
    query_recent_alerts,
    update_narrative,
)


@pytest.fixture
def cleanup_test_date():
    """Per-test sentinel date + cleanup."""
    test_date = datetime.date(2099, 1, 15)
    yield test_date
    from engine.memory import SessionFactory
    from engine.db_models import RiskManagerAlert
    s = SessionFactory()
    try:
        s.query(RiskManagerAlert).filter(RiskManagerAlert.date == test_date).delete()
        s.commit()
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic alert_id
# ──────────────────────────────────────────────────────────────────────────────
class TestMakeAlertId:
    def test_same_inputs_same_id(self):
        date = datetime.date(2099, 1, 1)
        a = make_alert_id(date, "1", ("AAPL",))
        b = make_alert_id(date, "1", ("AAPL",))
        assert a == b

    def test_affected_order_insensitive(self):
        date = datetime.date(2099, 1, 1)
        # Canonical sort means order doesn't matter
        a = make_alert_id(date, "1", ("AAPL", "MSFT"))
        b = make_alert_id(date, "1", ("MSFT", "AAPL"))
        assert a == b

    def test_different_mode_different_id(self):
        date = datetime.date(2099, 1, 1)
        a = make_alert_id(date, "1", ("AAPL",))
        b = make_alert_id(date, "5", ("AAPL",))
        assert a != b


# ──────────────────────────────────────────────────────────────────────────────
# breach_to_alert_row — pure dict transform
# ──────────────────────────────────────────────────────────────────────────────
class TestBreachToAlertRow:
    def test_basic_fields(self):
        b = Breach("1", "HARD_HALT", "test", 0.06, 0.05, ("AAPL",), {"k": "v"}, "spec1")
        row = breach_to_alert_row(b, datetime.date(2099, 1, 1), "pre_trade",
                                   cb_severity="SEVERE", halt_decision=True)
        assert row["mode_id"] == "1"
        assert row["severity"] == "HARD_HALT"
        assert row["cb_severity"] == "SEVERE"
        assert row["halt_decision"] is True
        assert row["phase"] == "pre_trade"
        assert row["spec_anchor"] == "spec1"
        assert row["narrative_text"] is None    # Phase 7 fills later

    def test_nan_observed_value_coerces_to_none(self):
        b = Breach("1", "HARD_HALT", "test", float("nan"), 0.05, (), {}, "s")
        row = breach_to_alert_row(b, datetime.date(2099, 1, 1), "pre_trade",
                                   cb_severity="SEVERE", halt_decision=True)
        assert row["observed_value"] is None

    def test_nan_threshold_coerces_to_none(self):
        b = Breach("1", "HARD_HALT", "test", 0.05, float("nan"), (), {}, "s")
        row = breach_to_alert_row(b, datetime.date(2099, 1, 1), "pre_trade",
                                   cb_severity="SEVERE", halt_decision=True)
        assert row["threshold"] is None

    def test_json_fields_serialize_correctly(self):
        b = Breach("1", "HARD_HALT", "test", 0.06, 0.05, ("AAPL", "MSFT"),
                   {"k": "v", "n": 3}, "s")
        row = breach_to_alert_row(b, datetime.date(2099, 1, 1), "pre_trade",
                                   cb_severity="SEVERE", halt_decision=True)
        # affected_json is a JSON list (not Python tuple repr)
        import json
        assert json.loads(row["affected_json"]) == ["AAPL", "MSFT"]
        assert json.loads(row["extra_json"]) == {"k": "v", "n": 3}


# ──────────────────────────────────────────────────────────────────────────────
# Idempotent UPSERT
# ──────────────────────────────────────────────────────────────────────────────
class TestPersistIdempotency:
    def test_persist_returns_alert_ids(self, cleanup_test_date):
        breaches = [
            Breach("1", "HARD_HALT", "t1", 0.06, 0.05, ("A",), {}, "s"),
            Breach("5", "HARD_HALT", "t5", 0.30, 0.25, (), {}, "s"),
        ]
        ids = persist_breaches_to_db(breaches, cleanup_test_date,
                                      phase="pre_trade", halt_decision=True)
        assert len(ids) == 2
        assert all(isinstance(i, str) and len(i) == 36 for i in ids)   # uuid format

    def test_rerun_same_inputs_same_ids(self, cleanup_test_date):
        breaches = [
            Breach("1", "HARD_HALT", "t", 0.06, 0.05, ("AAPL",), {}, "s"),
        ]
        ids1 = persist_breaches_to_db(breaches, cleanup_test_date,
                                       phase="pre_trade", halt_decision=True)
        ids2 = persist_breaches_to_db(breaches, cleanup_test_date,
                                       phase="pre_trade", halt_decision=True)
        assert ids1 == ids2

    def test_empty_breaches_returns_empty(self, cleanup_test_date):
        ids = persist_breaches_to_db([], cleanup_test_date,
                                      phase="pre_trade", halt_decision=False)
        assert ids == []

    def test_cb_severity_consistent_across_rows(self, cleanup_test_date):
        """One UPSERT call → all rows share the same cb_severity (audit invariant)."""
        breaches = [
            Breach("1", "HARD_HALT", "", 0, 0, ("X",), {}, "s"),
            Breach("2", "SOFT_WARN", "", 0, 0, ("Y",), {}, "s"),
            Breach("3", "HARD_HALT", "", 0, 0, ("Z",), {}, "s"),
        ]
        persist_breaches_to_db(breaches, cleanup_test_date,
                               phase="pre_trade", halt_decision=True)
        # Read back: all 3 rows have cb_severity == "SEVERE" (HARD_HALT dominates)
        rows = query_recent_alerts(days_back=1, severity_min="LIGHT")
        test_rows = [r for r in rows if r["date"] == cleanup_test_date]
        assert len(test_rows) == 3
        assert all(r["cb_severity"] == "SEVERE" for r in test_rows)


# ──────────────────────────────────────────────────────────────────────────────
# update_narrative
# ──────────────────────────────────────────────────────────────────────────────
class TestUpdateNarrative:
    def test_updates_existing_row(self, cleanup_test_date):
        b = Breach("1", "HARD_HALT", "", 0.06, 0.05, ("AAPL",), {}, "s")
        ids = persist_breaches_to_db([b], cleanup_test_date,
                                      phase="pre_trade", halt_decision=True)
        ok = update_narrative(cleanup_test_date, ids[0],
                              narrative_text="test prose", cost_usd=0.001)
        assert ok is True

        rows = query_recent_alerts(days_back=1, severity_min="LIGHT")
        target = next(r for r in rows
                      if r["date"] == cleanup_test_date and r["alert_id"] == ids[0])
        assert target["narrative_text"] == "test prose"

    def test_nonexistent_alert_returns_false(self, cleanup_test_date):
        ok = update_narrative(cleanup_test_date, "no-such-id",
                              narrative_text="x", cost_usd=0.0)
        assert ok is False


# ──────────────────────────────────────────────────────────────────────────────
# query_recent_alerts
# ──────────────────────────────────────────────────────────────────────────────
class TestQueryRecentAlerts:
    def test_severity_min_filter(self, cleanup_test_date):
        # Insert one SOFT_WARN only
        b = Breach("2", "SOFT_WARN", "", 0.15, 0.10, ("etf_l1",), {}, "s")
        persist_breaches_to_db([b], cleanup_test_date,
                               phase="pre_trade", halt_decision=False)
        # Looking up with severity_min=SEVERE should miss
        rows = query_recent_alerts(days_back=1, severity_min="SEVERE")
        ours = [r for r in rows if r["date"] == cleanup_test_date]
        assert len(ours) == 0
        # severity_min=LIGHT catches
        rows = query_recent_alerts(days_back=1, severity_min="LIGHT")
        ours = [r for r in rows if r["date"] == cleanup_test_date]
        assert len(ours) == 1
