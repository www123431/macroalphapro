"""
tests/test_ops_watchdog_phase4_notifications.py — Phase 4 unit tests (25 cases).

Coverage:
  - emit_notification short paths (5)    : dry_run / severity_none / light /
                                            medium / severe channel matrix
  - dashboard widget (3)                  : write success, content shape, error path
  - Windows toast (4)                     : win10toast missing / present / error /
                                            duration map (medium=10s, severe=30s)
  - email (4)                             : no SMTP config / partial config /
                                            full config success (smtplib mocked) /
                                            smtplib raises -> fail soft
  - halt flag (4)                         : circuit_breaker reuse / fail-soft on
                                            import error / state schema / preserves
                                            existing manual_reset path
  - rule_watchdog_halt_flag_not_stuck (3) : clean / fresh-halt OK / stuck > 7d HIGH
  - orchestrator integration (2)          : dry_run skips notifications /
                                            severe → halt_flag attempted
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── Shared isolated_db fixture ──────────────────────────────────────────────
@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    from engine import memory as _memory                    # noqa: F401
    from engine import db_models as _db_models              # noqa: F401
    from engine import universe_manager as _um              # noqa: F401
    from engine import auto_audit as _auto_audit            # noqa: F401
    from engine.db_models import Base

    db_path = tmp_path / "test_watchdog_phase4.db"
    test_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(test_engine)
    _um._Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, expire_on_commit=False)

    monkeypatch.setattr(_memory,     "engine",        test_engine)
    monkeypatch.setattr(_memory,     "SessionFactory", TestSession)
    monkeypatch.setattr(_db_models,  "engine",        test_engine)
    monkeypatch.setattr(_db_models,  "SessionFactory", TestSession)
    monkeypatch.setattr(_um,         "SessionFactory", TestSession)
    monkeypatch.setattr(_auto_audit, "SessionFactory", TestSession)
    return TestSession


@pytest.fixture
def isolated_cb(monkeypatch, tmp_path):
    """Redirect circuit_breaker._STATE_FILE to tmp path so tests don't pollute
    production engine/state/circuit_breaker.json."""
    from engine import circuit_breaker as _cb
    test_state = tmp_path / "cb_test.json"
    monkeypatch.setattr(_cb, "_STATE_FILE", test_state)
    return test_state


@pytest.fixture
def isolated_widget(monkeypatch, tmp_path):
    """Redirect dashboard widget path so tests don't pollute data/."""
    from engine.agents.ops_watchdog import notifications as _n
    widget = tmp_path / "widget_state.json"
    monkeypatch.setattr(_n, "_widget_state_path", lambda: widget)
    return widget


# ═════════════════════════════════════════════════════════════════════════════
# emit_notification short paths (5)
# ═════════════════════════════════════════════════════════════════════════════
class TestEmitShortPaths:
    def test_dry_run_skips_all(self):
        from engine.agents.ops_watchdog.notifications import emit_notification
        r = emit_notification(severity="severe", summary="x", findings=[],
                              today_iso="2026-05-12", dry_run=True)
        assert r == {"dashboard": False, "toast": False, "email": False,
                     "halt_flag": False, "skipped_reason": "dry_run"}

    def test_severity_none_skips_all(self):
        from engine.agents.ops_watchdog.notifications import emit_notification
        r = emit_notification(severity="none", summary="x", findings=[],
                              today_iso="2026-05-12")
        assert r["skipped_reason"] == "severity_none_or_invalid"
        assert all(r[c] is False for c in ("dashboard", "toast", "email",
                                           "halt_flag"))

    def test_light_only_dashboard(self, isolated_widget):
        from engine.agents.ops_watchdog.notifications import emit_notification
        r = emit_notification(severity="light", summary="cosmetic",
                              findings=[], today_iso="2026-05-12")
        assert r["dashboard"] is True
        assert r["toast"] is False    # toast is medium+severe only
        assert r["email"] is False
        assert r["halt_flag"] is False

    def test_medium_dashboard_and_toast(self, isolated_widget, monkeypatch):
        """Medium should attempt toast + dashboard but NOT halt/email."""
        from engine.agents.ops_watchdog import notifications as _n
        captured = {"toast_called": False, "duration": None}

        def fake_toast(severity, summary, duration_seconds):
            captured["toast_called"] = True
            captured["duration"] = duration_seconds
            return True
        monkeypatch.setattr(_n, "_send_windows_toast", fake_toast)

        r = _n.emit_notification(severity="medium", summary="2 ops failures",
                                  findings=[], today_iso="2026-05-12")
        assert r["dashboard"] is True
        assert r["toast"] is True
        assert captured["duration"] == _n._TOAST_DURATION_MEDIUM_S
        assert r["email"] is False
        assert r["halt_flag"] is False

    def test_severe_all_channels_attempted(self, isolated_widget, monkeypatch):
        """Severe attempts all 4 channels; toast duration is the longer one."""
        from engine.agents.ops_watchdog import notifications as _n
        captured = {"duration": None}

        def fake_toast(severity, summary, duration_seconds):
            captured["duration"] = duration_seconds
            return True
        monkeypatch.setattr(_n, "_send_windows_toast", fake_toast)
        monkeypatch.setattr(_n, "_send_email_if_configured",
                            lambda **kw: True)
        monkeypatch.setattr(_n, "_set_halt_flag", lambda reason: True)

        r = _n.emit_notification(severity="severe", summary="critical halt",
                                  findings=[], today_iso="2026-05-12")
        assert r["dashboard"] is True
        assert r["toast"] is True
        assert captured["duration"] == _n._TOAST_DURATION_SEVERE_S
        assert r["email"] is True
        assert r["halt_flag"] is True


# ═════════════════════════════════════════════════════════════════════════════
# dashboard widget (3)
# ═════════════════════════════════════════════════════════════════════════════
class TestDashboardWidget:
    def test_writes_widget_state_json(self, isolated_widget):
        from engine.agents.ops_watchdog.notifications import (
            _write_dashboard_widget,
        )
        ok = _write_dashboard_widget(
            severity="medium",
            summary="2 ops failures detected",
            findings=[
                {"finding_id": 1, "rule_name": "rule_cycle_state_completion",
                 "severity": "MID"},
            ],
            today_iso="2026-05-12",
            repair_info={"n_attempted": 1, "n_succeeded": 1},
        )
        assert ok is True
        assert isolated_widget.exists()
        payload = json.loads(isolated_widget.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["spec_id"] == 63
        assert payload["severity"] == "medium"
        assert payload["n_findings"] == 1
        assert payload["findings_brief"][0]["rule_name"] == "rule_cycle_state_completion"
        assert payload["auto_repair"]["n_succeeded"] == 1

    def test_truncates_long_summary(self, isolated_widget):
        from engine.agents.ops_watchdog.notifications import (
            _write_dashboard_widget,
        )
        long_summary = "X" * 800
        ok = _write_dashboard_widget(
            severity="severe", summary=long_summary, findings=[],
            today_iso="2026-05-12",
        )
        assert ok is True
        payload = json.loads(isolated_widget.read_text(encoding="utf-8"))
        # Spec docstring says summary truncated to 500
        assert len(payload["summary"]) == 500

    def test_write_error_returns_false(self, monkeypatch, tmp_path):
        """If path write fails (e.g. unwritable dir), returns False not raise."""
        from engine.agents.ops_watchdog import notifications as _n
        # Point widget path to a non-creatable location (use a file path
        # whose parent is a regular file, making mkdir fail)
        bad_parent = tmp_path / "blocking_file"
        bad_parent.write_text("not a dir")
        monkeypatch.setattr(_n, "_widget_state_path",
                            lambda: bad_parent / "subdir" / "widget.json")
        ok = _n._write_dashboard_widget(
            severity="light", summary="x", findings=[],
            today_iso="2026-05-12",
        )
        assert ok is False


# ═════════════════════════════════════════════════════════════════════════════
# Windows toast (4)
# ═════════════════════════════════════════════════════════════════════════════
class TestWindowsToast:
    def test_win10toast_missing_returns_false(self, monkeypatch):
        """When win10toast not importable, channel fails soft."""
        from engine.agents.ops_watchdog import notifications as _n
        import sys
        # Hide win10toast even if installed
        monkeypatch.setitem(sys.modules, "win10toast", None)
        ok = _n._send_windows_toast(severity="medium",
                                     summary="x", duration_seconds=10)
        assert ok is False

    def test_win10toast_success_path(self, monkeypatch):
        """When win10toast is importable + show_toast succeeds → True."""
        from engine.agents.ops_watchdog import notifications as _n
        import sys
        captured = {"shown": False, "duration": None}

        class FakeToaster:
            def show_toast(self, title, msg, duration, threaded, icon_path):
                captured["shown"] = True
                captured["duration"] = duration

        fake_mod = type("FakeMod", (), {"ToastNotifier": FakeToaster})()
        monkeypatch.setitem(sys.modules, "win10toast", fake_mod)
        ok = _n._send_windows_toast(severity="medium",
                                     summary="x", duration_seconds=10)
        assert ok is True
        assert captured["shown"] is True
        assert captured["duration"] == 10

    def test_win10toast_exception_fail_soft(self, monkeypatch):
        """When show_toast raises, channel fails soft → False."""
        from engine.agents.ops_watchdog import notifications as _n
        import sys

        class BadToaster:
            def show_toast(self, *a, **kw):
                raise RuntimeError("synthetic_toast_failure")

        fake_mod = type("FakeMod", (), {"ToastNotifier": BadToaster})()
        monkeypatch.setitem(sys.modules, "win10toast", fake_mod)
        ok = _n._send_windows_toast(severity="severe",
                                     summary="x", duration_seconds=30)
        assert ok is False

    def test_duration_constants(self):
        from engine.agents.ops_watchdog.notifications import (
            _TOAST_DURATION_MEDIUM_S, _TOAST_DURATION_SEVERE_S,
        )
        # Spec §2.6: 10s medium / 30s severe persist
        assert _TOAST_DURATION_MEDIUM_S == 10
        assert _TOAST_DURATION_SEVERE_S == 30


# ═════════════════════════════════════════════════════════════════════════════
# email (4)
# ═════════════════════════════════════════════════════════════════════════════
class TestEmail:
    def test_no_smtp_config_returns_false(self, monkeypatch, tmp_path):
        """If .streamlit/secrets.toml has no [ops_watchdog.smtp], skip + False."""
        from engine.agents.ops_watchdog import notifications as _n
        monkeypatch.setattr(_n, "_read_smtp_config", lambda: None)
        ok = _n._send_email_if_configured(
            summary="x", findings=[], today_iso="2026-05-12",
        )
        assert ok is False

    def test_full_config_calls_smtplib(self, monkeypatch):
        from engine.agents.ops_watchdog import notifications as _n
        cfg = {
            "host": "smtp.test.local", "port": 587,
            "user": "u", "password": "p",
            "from_addr": "from@test", "to_addrs": ["to@test"],
            "use_tls": True,
        }
        monkeypatch.setattr(_n, "_read_smtp_config", lambda: cfg)
        captured = {"sent": False}

        class FakeSMTP:
            def __init__(self, host, port, timeout=None):
                self.host = host
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def ehlo(self): pass
            def starttls(self): pass
            def login(self, u, p): pass
            def sendmail(self, from_addr, to_addrs, msg):
                captured["sent"] = True

        monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
        ok = _n._send_email_if_configured(
            summary="critical halt", findings=[], today_iso="2026-05-12",
        )
        assert ok is True
        assert captured["sent"] is True

    def test_smtplib_raise_fails_soft(self, monkeypatch):
        from engine.agents.ops_watchdog import notifications as _n
        cfg = {"host": "x", "port": 587, "user": "u", "password": "p",
               "from_addr": "a", "to_addrs": ["b"], "use_tls": False}
        monkeypatch.setattr(_n, "_read_smtp_config", lambda: cfg)

        def boom(*a, **kw):
            raise ConnectionError("synthetic_smtp_down")
        monkeypatch.setattr("smtplib.SMTP", boom)
        ok = _n._send_email_if_configured(summary="x", findings=[],
                                          today_iso="2026-05-12")
        assert ok is False

    def test_partial_config_returns_false(self, monkeypatch, tmp_path):
        """Missing required fields in SMTP config -> _read_smtp_config rejects."""
        from engine.agents.ops_watchdog import notifications as _n
        # Force secrets.toml with partial config
        secrets = tmp_path / ".streamlit" / "secrets.toml"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(
            '[ops_watchdog.smtp]\nhost = "smtp.test"\nport = 587\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(_n, "_repo_root", lambda: tmp_path)
        ok = _n._send_email_if_configured(summary="x", findings=[],
                                          today_iso="2026-05-12")
        assert ok is False


# ═════════════════════════════════════════════════════════════════════════════
# halt flag (4)
# ═════════════════════════════════════════════════════════════════════════════
class TestHaltFlag:
    def test_writes_circuit_breaker_state(self, isolated_cb):
        from engine.agents.ops_watchdog.notifications import _set_halt_flag
        ok = _set_halt_flag(reason="mode_11 cadence drift detected")
        assert ok is True
        assert isolated_cb.exists()
        payload = json.loads(isolated_cb.read_text(encoding="utf-8"))
        assert payload["level"] == "severe"
        assert payload["auto_reset"] is False
        assert "ops_watchdog:" in payload["reason"]
        assert "mode_11" in payload["reason"]
        assert payload["triggered_at"] is not None

    def test_fail_soft_on_circuit_breaker_import_failure(self, monkeypatch):
        from engine.agents.ops_watchdog import notifications as _n
        import sys
        # Simulate engine.circuit_breaker missing
        saved = sys.modules.pop("engine.circuit_breaker", None)
        try:
            sys.modules["engine.circuit_breaker"] = None    # raises ImportError
            ok = _n._set_halt_flag(reason="x")
            assert ok is False
        finally:
            if saved is not None:
                sys.modules["engine.circuit_breaker"] = saved

    def test_reason_truncated_to_400_chars(self, isolated_cb):
        from engine.agents.ops_watchdog.notifications import _set_halt_flag
        long_reason = "Z" * 800
        ok = _set_halt_flag(reason=long_reason)
        assert ok is True
        payload = json.loads(isolated_cb.read_text(encoding="utf-8"))
        # "ops_watchdog: " prefix (14 chars) + 400 cap = total length 400
        # since slicing is [:400] on prefixed string
        assert len(payload["reason"]) == 400

    def test_set_halt_does_not_call_manual_reset(self, monkeypatch, isolated_cb):
        """INVARIANT: Watchdog SETs halt but NEVER clears. Verify the code
        path doesn't invoke manual_reset (only human dashboard can clear)."""
        from engine.agents.ops_watchdog import notifications as _n
        from engine import circuit_breaker as _cb
        called = {"manual_reset": False}

        def fake_reset(*a, **kw):
            called["manual_reset"] = True
        monkeypatch.setattr(_cb, "manual_reset", fake_reset)

        ok = _n._set_halt_flag(reason="severe finding")
        assert ok is True
        assert called["manual_reset"] is False


# ═════════════════════════════════════════════════════════════════════════════
# rule_watchdog_runs_daily (Phase 5 Tier R, spec §4.3) — 5 tests
# ═════════════════════════════════════════════════════════════════════════════
class TestWatchdogRunsDailyRule:
    def test_no_runs_ever_fires_high(self, isolated_db):
        """Empty AuditRun table → Watchdog has never fired → HIGH finding."""
        from engine.auto_audit_rules import rule_watchdog_runs_daily
        result = rule_watchdog_runs_daily()
        assert result is not None
        assert result["severity"] == "HIGH"
        assert result["snapshot"]["last_run_at"] is None

    def test_run_within_30h_clean(self, isolated_db):
        """AuditRun(scope='watchdog') 1h ago → within 30h window → None."""
        from engine.auto_audit_rules import rule_watchdog_runs_daily
        from engine.auto_audit_models import AuditRun
        fresh = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        with isolated_db() as s:
            s.add(AuditRun(scope="watchdog", n_rules_run=11, n_findings=0,
                            exit_status="ok", run_at=fresh))
            s.commit()
        assert rule_watchdog_runs_daily() is None

    def test_run_28h_ago_still_clean(self, isolated_db):
        """Cutoff is 30h, 28h ago is within tolerance → None."""
        from engine.auto_audit_rules import rule_watchdog_runs_daily
        from engine.auto_audit_models import AuditRun
        ts = datetime.datetime.utcnow() - datetime.timedelta(hours=28)
        with isolated_db() as s:
            s.add(AuditRun(scope="watchdog", n_rules_run=11, n_findings=2,
                            exit_status="ok", run_at=ts))
            s.commit()
        assert rule_watchdog_runs_daily() is None

    def test_run_36h_ago_fires_high(self, isolated_db):
        """36h ago > 30h cutoff → HIGH finding; snapshot reports last_age_hours."""
        from engine.auto_audit_rules import rule_watchdog_runs_daily
        from engine.auto_audit_models import AuditRun
        stale = datetime.datetime.utcnow() - datetime.timedelta(hours=36)
        with isolated_db() as s:
            s.add(AuditRun(scope="watchdog", n_rules_run=11, n_findings=3,
                            exit_status="ok", run_at=stale))
            s.commit()
        result = rule_watchdog_runs_daily()
        assert result is not None
        assert result["severity"] == "HIGH"
        assert result["snapshot"]["last_age_hours"] > 30.0
        assert result["snapshot"]["last_run_at"] is not None

    def test_other_scope_runs_dont_count(self, isolated_db):
        """An AuditRun(scope='critical') doesn't satisfy the watchdog-runs check."""
        from engine.auto_audit_rules import rule_watchdog_runs_daily
        from engine.auto_audit_models import AuditRun
        recent = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        with isolated_db() as s:
            s.add(AuditRun(scope="critical", n_rules_run=20, n_findings=0,
                            exit_status="ok", run_at=recent))
            s.commit()
        # 'critical' scope row exists but no 'watchdog' scope row → fire
        result = rule_watchdog_runs_daily()
        assert result is not None and result["severity"] == "HIGH"


# ═════════════════════════════════════════════════════════════════════════════
# rule_watchdog_halt_flag_not_stuck (3)
# ═════════════════════════════════════════════════════════════════════════════
class TestHaltStuckRule:
    def test_no_halt_returns_none(self, isolated_cb):
        from engine.auto_audit_rules import rule_watchdog_halt_flag_not_stuck
        # isolated_cb starts empty → no halt → None
        assert rule_watchdog_halt_flag_not_stuck() is None

    def test_fresh_halt_within_7d_no_finding(self, isolated_cb):
        """Halt set 1 day ago → not stuck → None."""
        from engine.auto_audit_rules import rule_watchdog_halt_flag_not_stuck
        from engine import circuit_breaker as _cb
        fresh = (datetime.datetime.utcnow() - datetime.timedelta(days=1))
        state = _cb.CircuitBreakerState(
            level=_cb.LEVEL_SEVERE,
            reason="ops_watchdog: mode_7_nav_anomaly",
            triggered_at=fresh.isoformat() + "Z",
            auto_reset=False,
        )
        _cb._save_persistent(state)
        assert rule_watchdog_halt_flag_not_stuck() is None

    def test_stuck_halt_over_7d_fires_high(self, isolated_cb):
        """Halt set 9 days ago and still SEVERE → HIGH finding."""
        from engine.auto_audit_rules import rule_watchdog_halt_flag_not_stuck
        from engine import circuit_breaker as _cb
        stale = (datetime.datetime.utcnow() - datetime.timedelta(days=9))
        state = _cb.CircuitBreakerState(
            level=_cb.LEVEL_SEVERE,
            reason="ops_watchdog: mode_5_weight_delta_unexplained",
            triggered_at=stale.isoformat() + "Z",
            auto_reset=False,
        )
        _cb._save_persistent(state)
        result = rule_watchdog_halt_flag_not_stuck()
        assert result is not None
        assert result["severity"] == "HIGH"
        assert result["snapshot"]["halt_age_days"] > 7.0
        assert "ops_watchdog:" in result["snapshot"]["halt_reason"]


# ═════════════════════════════════════════════════════════════════════════════
# orchestrator integration (2)
# ═════════════════════════════════════════════════════════════════════════════
class TestOrchestratorIntegration:
    def test_dry_run_skips_notifications(self, isolated_db, isolated_widget,
                                          isolated_cb, tmp_path, monkeypatch):
        from engine.agents.ops_watchdog import agent as _agent
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=True, save_trace=False)
        # Dry-run → all notification channels False; skipped_reason recorded
        assert result.notifications["skipped_reason"] == "dry_run"
        assert result.notifications["dashboard"] is False
        assert result.notifications["halt_flag"] is False

    def test_severe_triggers_halt_flag(self, isolated_db, isolated_widget,
                                        isolated_cb, tmp_path, monkeypatch):
        """End-to-end: seed mode-11 (severe) finding, no dry-run,
        verify halt_flag wrote to isolated_cb."""
        from engine.agents.ops_watchdog import agent as _agent
        from engine.auto_audit_models import AuditFinding, AuditRun
        from engine.quant_co_pilot.base import TraceResult

        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")

        # Stub run_daily_batch + LLM + Windows toast (avoid REAL user-visible
        # toast popping during pytest — test isolation hygiene). The toast
        # channel itself is unit-tested elsewhere via _send_windows_toast direct
        # tests; here we just need to verify the orchestrator INVOKES it.
        class FakeBatch:
            skipped = False
        monkeypatch.setattr("engine.daily_batch.run_daily_batch",
                            lambda **kw: FakeBatch())
        monkeypatch.setattr("engine.quant_co_pilot.base.run_react_agent",
                            lambda *a, **k: TraceResult(
                                query="", final_answer="severe halt rationale",
                                citations=[], annotated_answer="severe halt rationale",
                                steps=[], cost_usd=0.0, latency_ms=0,
                                abort_reason=None,
                                completed_at="2026-05-12T00:00:00Z",
                            ))
        monkeypatch.setattr(
            "engine.agents.ops_watchdog.notifications._send_windows_toast",
            lambda **kw: True,
        )

        def fake_run_audit(scope):
            from engine.memory import SessionFactory
            with SessionFactory() as s:
                run = AuditRun(scope=scope, n_rules_run=11, n_findings=1,
                               exit_status="ok")
                s.add(run); s.flush()
                s.add(AuditFinding(
                    run_id=run.id, rule_name="rule_rebalance_frequency_audit",
                    severity="HIGH",
                    snapshot_json='{"context":"mode_11 cadence"}', status="OPEN",
                    detected_at=datetime.datetime.utcnow(),
                ))
                s.commit()
                return {"run_id": run.id, "scope": scope,
                        "n_rules_run": 11, "n_findings": 1,
                        "exit_status": "ok"}
        monkeypatch.setattr("engine.auto_audit.run_audit", fake_run_audit)

        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        # Severe → halt flag should fire (other channels may fail-soft depending
        # on env; but halt_flag is core)
        assert result.triage["severity"] == "severe"
        assert result.notifications["halt_flag"] is True
        # Verify CB state file was written with ops_watchdog: prefix
        assert isolated_cb.exists()
        payload = json.loads(isolated_cb.read_text(encoding="utf-8"))
        assert payload["level"] == "severe"
        assert payload["reason"].startswith("ops_watchdog:")
