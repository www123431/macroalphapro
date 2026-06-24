"""Tests for engine.research.liveness_heartbeat (P0 liveness layer).

Strategy: monkeypatch LIVENESS_LEDGER to a tmp path. Exercise the four
recording / reading / no-show / weekend code paths.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    from engine.research import liveness_heartbeat as L
    p = tmp_path / "liveness_heartbeat.jsonl"
    monkeypatch.setattr(L, "LIVENESS_LEDGER", p)
    yield p


def test_record_run_writes_row_with_expected_fields(tmp_ledger):
    from engine.research import liveness_heartbeat as L
    row = L.record_run(
        as_of=_dt.date(2026, 6, 2),
        exit_code=0,
        n_orders=114,
        n_fills=114,
        equity_before=100046.19,
        n_strategies=4,
        gross_weight=0.7916,
        log_file=Path("data/paper_trade/daily_run_2026-06-02.log"),
    )
    assert row["status"] == "success"
    assert row["as_of"] == "2026-06-02"
    assert row["n_orders"] == 114
    # Ledger now has one persisted line
    assert tmp_ledger.is_file()
    [persisted] = [json.loads(ln) for ln in tmp_ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert persisted["status"] == "success"


def test_status_from_exit_maps_known_codes():
    from engine.research import liveness_heartbeat as L
    assert L.status_from_exit(0) == "success"
    assert L.status_from_exit(4) == "halt_cb"
    assert L.status_from_exit(5) == "halt_risk"
    assert L.status_from_exit(6) == "halt_dq"
    # Unknown exit code yields informational string, not exception
    assert L.status_from_exit(99) == "unknown_exit_99"


def test_record_run_append_only(tmp_ledger):
    from engine.research import liveness_heartbeat as L
    L.record_run(as_of=_dt.date(2026, 5, 31), exit_code=0)
    L.record_run(as_of=_dt.date(2026, 6, 1), exit_code=4, halted_at_step="step_circuit_breaker_preflight")
    L.record_run(as_of=_dt.date(2026, 6, 2), exit_code=0)
    rows = L.read_recent(10)
    # newest first
    assert [r["as_of"] for r in rows] == ["2026-06-02", "2026-06-01", "2026-05-31"]
    assert rows[1]["status"] == "halt_cb"
    assert rows[1]["halted_at_step"] == "step_circuit_breaker_preflight"


def test_heartbeat_for_date_returns_latest_for_that_date(tmp_ledger):
    """A retry / re-run on the same as_of date should resolve to the
    most recent row, not the first."""
    from engine.research import liveness_heartbeat as L
    L.record_run(as_of=_dt.date(2026, 6, 2), exit_code=4)
    L.record_run(as_of=_dt.date(2026, 6, 2), exit_code=0)
    hb = L.heartbeat_for_date(_dt.date(2026, 6, 2))
    assert hb is not None
    assert hb["status"] == "success"   # the later run wins


def test_assess_liveness_ok_when_today_succeeded(tmp_ledger):
    from engine.research import liveness_heartbeat as L
    today = _dt.date(2026, 6, 2)   # Tuesday
    # Use a time *inside* the run window so a fresh row reads as today's run
    now = _dt.datetime(2026, 6, 2, 23, 30)   # 23:30 UTC = 07:30 SGT next day
    L.record_run(as_of=today, exit_code=0, n_orders=114, n_fills=114)
    verdict = L.assess_liveness(now_utc=now)
    assert verdict["verdict"] == "OK"


def test_assess_liveness_alert_no_show_past_deadline(tmp_ledger):
    """It's well past today's deadline + no heartbeat → ALERT_NO_SHOW."""
    from engine.research import liveness_heartbeat as L
    today = _dt.date(2026, 6, 2)   # Tuesday
    # 04:00 UTC the next day is well past deadline of 22:00+90min = 23:30
    now = _dt.datetime(2026, 6, 3, 4, 0)
    # Only have yesterday's row, none for today
    L.record_run(as_of=_dt.date(2026, 6, 1), exit_code=0)
    verdict = L.assess_liveness(now_utc=now)
    # 2026-06-03 is Wednesday — also a weekday — so this should ALERT
    assert verdict["verdict"] == "ALERT_NO_SHOW"


def test_assess_liveness_warn_when_today_halted(tmp_ledger):
    from engine.research import liveness_heartbeat as L
    today = _dt.date(2026, 6, 2)
    now = _dt.datetime(2026, 6, 2, 23, 30)
    L.record_run(as_of=today, exit_code=4,
                  halted_at_step="step_circuit_breaker_preflight")
    verdict = L.assess_liveness(now_utc=now)
    assert verdict["verdict"] == "WARN_STATUS"


def test_assess_liveness_saturday_with_friday_run_present_is_ok(tmp_ledger):
    """Saturday should NOT silently say 'weekend, don't care' — if
    Friday's run is missing we want to know on Saturday morning.
    Conversely, if Friday's run is present + healthy, OK on Saturday."""
    from engine.research import liveness_heartbeat as L
    L.record_run(as_of=_dt.date(2026, 6, 5), exit_code=0,
                  n_orders=10, n_fills=10)   # Friday
    sat = _dt.datetime(2026, 6, 6, 23, 30)
    verdict = L.assess_liveness(now_utc=sat)
    assert verdict["verdict"] == "OK"
    assert verdict["as_of"] == "2026-06-05"


def test_assess_liveness_saturday_no_friday_run_alerts(tmp_ledger):
    """Critical doctrine: weekends must NOT mask a missing Friday run."""
    from engine.research import liveness_heartbeat as L
    sat = _dt.datetime(2026, 6, 6, 23, 30)
    verdict = L.assess_liveness(now_utc=sat)
    assert verdict["verdict"] == "ALERT_NO_SHOW"
    assert verdict["as_of"] == "2026-06-05"   # Friday is the missing day


def test_assess_liveness_early_monday_morning_off_hours(tmp_ledger):
    """Monday before Friday's deadline has 'expired' from view: this
    can't happen with our weekday-deadline scan (Friday's deadline is
    always in the past by Monday). So actually this should ALERT for
    Friday missing. Test confirms."""
    from engine.research import liveness_heartbeat as L
    # Monday morning, no prior runs at all
    mon = _dt.datetime(2026, 6, 8, 10, 0)
    verdict = L.assess_liveness(now_utc=mon)
    # Friday June 5's deadline has long passed, no heartbeat for it
    assert verdict["verdict"] == "ALERT_NO_SHOW"
    assert verdict["as_of"] == "2026-06-05"


def test_assess_liveness_yesterday_ok_resolves_to_ok(tmp_ledger):
    """Early Tuesday morning, Monday's run is healthy, today's deadline
    has not yet passed → verdict resolves to Monday's heartbeat = OK."""
    from engine.research import liveness_heartbeat as L
    L.record_run(as_of=_dt.date(2026, 6, 1), exit_code=0,
                  n_orders=50, n_fills=50)
    early = _dt.datetime(2026, 6, 2, 10, 0)
    verdict = L.assess_liveness(now_utc=early)
    assert verdict["verdict"] == "OK"
    assert verdict["as_of"] == "2026-06-01"


def test_assess_liveness_empty_ledger_after_deadline_alerts(tmp_ledger):
    from engine.research import liveness_heartbeat as L
    late = _dt.datetime(2026, 6, 2, 23, 45)
    verdict = L.assess_liveness(now_utc=late)
    assert verdict["verdict"] == "ALERT_NO_SHOW"
