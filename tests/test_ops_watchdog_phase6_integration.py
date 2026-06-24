"""
tests/test_ops_watchdog_phase6_integration.py — spec §五 Gate 3 + Gate 4
end-to-end integration tests.

Phase 1-5 each have unit tests covering their module's slice. Phase 6 adds
two ONE-SHOT pytest scenarios that exercise the FULL Watchdog production
pipeline as a single chain — rules → triage → auto-repair → LLM ReAct →
notifications dispatch — with all external dependencies mocked but the
ORCHESTRATOR PATH itself unmocked.

These tests are the automated equivalent of the 2026-05-13 11:26 SGT manual
dogfood verification documented in `docs/capability_evidence/ops_watchdog_
v1_phase5_complete_2026-05-13.md`. They guard against future refactors
silently breaking the production chain.

Spec §五 Gate 3: cycle failure → 7 rules fire → ReAct → auto-repair → MEDIUM toast
Spec §五 Gate 4: weight cap violation → rule fires → auto-truncate → MEDIUM
                 (NOTE: post-amendment-3 mode 10 is deferred-stub; this test
                  verifies escalation behavior rather than auto-truncate)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── Shared isolated_db fixture (mirrors Phase 4 pattern) ────────────────────
@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    from engine import memory as _memory                    # noqa: F401
    from engine import db_models as _db_models              # noqa: F401
    from engine import universe_manager as _um              # noqa: F401
    from engine import auto_audit as _auto_audit            # noqa: F401
    from engine.db_models import Base

    db_path = tmp_path / "test_watchdog_phase6.db"
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
    """Redirect circuit_breaker._STATE_FILE to tmp path."""
    from engine import circuit_breaker as _cb
    test_state = tmp_path / "cb_e2e.json"
    monkeypatch.setattr(_cb, "_STATE_FILE", test_state)
    return test_state


@pytest.fixture
def isolated_widget(monkeypatch, tmp_path):
    from engine.agents.ops_watchdog import notifications as _n
    w = tmp_path / "widget_state_e2e.json"
    monkeypatch.setattr(_n, "_widget_state_path", lambda: w)
    return w


@pytest.fixture
def captured_side_effects(monkeypatch):
    """
    Capture orchestrator-emitted side effects across all 4 notification
    channels + auto-repair + LLM. Each tuple records (channel, payload).
    """
    captures: dict = {
        "llm_invocations": [],     # list of (agent_id, max_steps, cost_budget)
        "toast_calls":     [],     # list of (severity, summary, duration)
        "email_calls":     [],     # list of summary strings
        "run_daily_batch": [],     # list of as_of_date values
        "halt_set":        [],     # list of reasons
    }

    # LLM stub
    def fake_react(*args, **kwargs):
        captures["llm_invocations"].append({
            "agent_id":        kwargs.get("agent_id"),
            "max_steps":       kwargs.get("max_steps"),
            "cost_budget_usd": kwargs.get("cost_budget_usd"),
            "role_intro_len":  len(kwargs.get("role_intro", "") or ""),
        })
        from engine.quant_co_pilot.base import TraceResult
        return TraceResult(
            query=args[0] if args else "",
            final_answer="Phase 6 e2e: mode_1 cycle_failed detected; "
                          "auto-repair retry succeeded after run_daily_batch.",
            citations=[], annotated_answer="...annotated...",
            steps=[], cost_usd=0.012, latency_ms=850, abort_reason=None,
            completed_at="2026-05-13T12:00:00Z",
        )
    monkeypatch.setattr("engine.quant_co_pilot.base.run_react_agent", fake_react)

    # Toast stub
    def fake_toast(**kwargs):
        captures["toast_calls"].append(kwargs)
        return True
    monkeypatch.setattr(
        "engine.agents.ops_watchdog.notifications._send_windows_toast",
        fake_toast,
    )

    # Email stub
    def fake_email(**kwargs):
        captures["email_calls"].append(kwargs.get("summary", ""))
        return False  # simulate "no SMTP configured"
    monkeypatch.setattr(
        "engine.agents.ops_watchdog.notifications._send_email_if_configured",
        fake_email,
    )

    # run_daily_batch stub (auto-repair recipe target)
    class FakeBatch:
        skipped = False
    def fake_batch(*, as_of_date=None, force=False):
        captures["run_daily_batch"].append({"as_of_date": as_of_date,
                                             "force": force})
        return FakeBatch()
    monkeypatch.setattr("engine.daily_batch.run_daily_batch", fake_batch)

    return captures


# ═════════════════════════════════════════════════════════════════════════════
# Gate 3: cycle failure → MEDIUM end-to-end chain
# ═════════════════════════════════════════════════════════════════════════════
class TestE2EChainMediumCycleFailure:
    def test_full_chain_cycle_failed_to_medium_notifications(
        self, isolated_db, isolated_widget, isolated_cb, captured_side_effects,
        tmp_path, monkeypatch,
    ):
        """
        Spec §五 Gate 3 — end-to-end production chain:
          1. Pre-seed mode-1 (cycle silently failed) finding
          2. run_watchdog invoked (non-dry-run)
          3. triage → MEDIUM (mode_1 hardcoded medium severity)
          4. auto-repair → _repair_retry_idempotent_batch → run_daily_batch
             invoked → succeeded
          5. LLM ReAct invoked with agent_id='ops_watchdog' + Watchdog role_intro
          6. emit_notification dispatches:
             dashboard widget (medium → light/medium/severe all write)
             toast 10s (medium duration per spec §2.6)
             email skipped (medium does not fire email)
             halt_flag NOT set (only severe sets)
          7. Trace JSON v3 saved with all fields populated
        """
        from engine.agents.ops_watchdog import agent as _agent
        from engine.auto_audit_models import AuditFinding, AuditRun
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")

        # Pre-seed mode-1 finding via fake run_audit returning a known run_id
        def fake_run_audit(scope):
            from engine.memory import SessionFactory
            with SessionFactory() as s:
                run = AuditRun(scope=scope, n_rules_run=11, n_findings=1,
                               exit_status="ok")
                s.add(run); s.flush()
                s.add(AuditFinding(
                    run_id=run.id,
                    rule_name="rule_cycle_state_completion",
                    severity="MID",
                    snapshot_json=json.dumps({
                        "n_issues": 1,
                        "issues": [{"kind": "cycle_failed", "cycle_id": 99,
                                    "error_log": "synthetic Phase 6 e2e"}],
                    }),
                    status="OPEN",
                    detected_at=datetime.datetime.utcnow(),
                ))
                s.commit()
                return {"run_id": run.id, "scope": scope, "n_rules_run": 11,
                        "n_findings": 1, "exit_status": "ok"}
        monkeypatch.setattr("engine.auto_audit.run_audit", fake_run_audit)

        # ── Execute the full chain ────────────────────────────────────────
        result = _agent.run_watchdog(dry_run=False, save_trace=True)

        cap = captured_side_effects

        # ── 1. Findings collected ─────────────────────────────────────────
        assert result.n_findings == 1
        assert result.findings_summary[0]["rule_name"] == "rule_cycle_state_completion"

        # ── 2. Triage produced MEDIUM (mode_1 hardcoded) ─────────────────
        assert result.triage["severity"] == "medium"
        assert "mode_1_cycle_failed" in result.triage["modes_fired"]

        # ── 3. Auto-repair recipe fired and succeeded ─────────────────────
        assert result.auto_repair["n_attempted"] == 1
        assert result.auto_repair["n_succeeded"] == 1
        assert result.auto_repair["n_failed"] == 0
        # Confirm the recipe actually called run_daily_batch with force=True
        assert len(cap["run_daily_batch"]) == 1
        assert cap["run_daily_batch"][0]["force"] is True

        # ── 4. LLM ReAct invoked under ops_watchdog agent_id ──────────────
        assert result.llm_used is True
        assert len(cap["llm_invocations"]) == 1
        llm_inv = cap["llm_invocations"][0]
        assert llm_inv["agent_id"] == "ops_watchdog"
        assert llm_inv["max_steps"] == 8
        assert llm_inv["cost_budget_usd"] == 0.20
        assert llm_inv["role_intro_len"] > 100   # Watchdog role intro populated

        # ── 5. Notifications dispatched per medium severity ───────────────
        n = result.notifications
        assert n["dashboard"] is True          # any non-none → dashboard
        assert n["toast"] is True              # medium → toast
        assert n["email"] is False             # medium → email skipped
        assert n["halt_flag"] is False         # medium → halt NOT set
        # Toast was called with medium duration (10s)
        assert len(cap["toast_calls"]) == 1
        toast = cap["toast_calls"][0]
        assert toast["severity"] == "medium"
        assert toast["duration_seconds"] == 10

        # ── 6. Halt flag NOT written to circuit_breaker.json ─────────────
        assert not isolated_cb.exists()

        # ── 7. Trace JSON v3 saved with full payload ─────────────────────
        trace_path = Path(result.trace_json_path)
        assert trace_path.exists()
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["schema_version"] >= 3
        assert trace["spec_id"] == 63
        assert trace["triage"]["severity"] == "medium"
        assert trace["auto_repair"]["n_succeeded"] == 1
        assert trace["notifications"]["toast"] is True
        assert trace["notifications"]["halt_flag"] is False
        assert trace["llm_used"] is True


# ═════════════════════════════════════════════════════════════════════════════
# Gate 4: severe escalation chain (post-amendment-3 — mode 10 is deferred stub,
# so this gate now verifies ESCALATION behavior rather than auto-truncate)
# ═════════════════════════════════════════════════════════════════════════════
class TestE2EChainSevereDeferredEscalation:
    def test_severe_finding_triggers_halt_flag_via_public_cb_api(
        self, isolated_db, isolated_widget, isolated_cb, captured_side_effects,
        tmp_path, monkeypatch,
    ):
        """
        Spec §五 Gate 4 (post-amendment-3 reframe) — severe escalation chain:
          1. Pre-seed mode-11 (cadence drift, severe-hardcoded) finding
          2. run_watchdog non-dry-run
          3. triage → SEVERE
          4. auto-repair: mode_11 NOT in AUTO_REPAIR_RECIPES_LOCKED → 0 attempts
             (escalation, not auto-fix per Phase 3 Option A doctrine)
          5. Notifications: all 4 channels including halt_flag SET via the
             NEW public engine.circuit_breaker.set_external_halt_flag API
             (added Phase 6 cleanup for #4)
          6. Halt flag's reason has 'ops_watchdog:' prefix → Tier R
             rule_watchdog_halt_flag_not_stuck can distinguish source
        """
        from engine.agents.ops_watchdog import agent as _agent
        from engine.auto_audit_models import AuditFinding, AuditRun
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")

        def fake_run_audit(scope):
            from engine.memory import SessionFactory
            with SessionFactory() as s:
                run = AuditRun(scope=scope, n_rules_run=11, n_findings=1,
                               exit_status="ok")
                s.add(run); s.flush()
                s.add(AuditFinding(
                    run_id=run.id,
                    rule_name="rule_rebalance_frequency_audit",
                    severity="HIGH",
                    snapshot_json=json.dumps({
                        "n_violations": 1,
                        "violations": [{"month": "2025-12",
                                         "kind": "no_rebalance"}],
                    }),
                    status="OPEN",
                    detected_at=datetime.datetime.utcnow(),
                ))
                s.commit()
                return {"run_id": run.id, "scope": scope, "n_rules_run": 11,
                        "n_findings": 1, "exit_status": "ok"}
        monkeypatch.setattr("engine.auto_audit.run_audit", fake_run_audit)

        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        cap = captured_side_effects

        # Triage severe
        assert result.triage["severity"] == "severe"
        assert "mode_11_cadence_drift" in result.triage["modes_fired"]

        # Mode 11 NOT in AUTO_REPAIR_RECIPES → 0 attempts (escalation path)
        assert result.auto_repair["n_attempted"] == 0

        # All 4 channels fire for severe
        n = result.notifications
        assert n["dashboard"] is True
        assert n["toast"] is True
        # Email stub returns False (simulating no SMTP); halt SET via public API
        assert n["halt_flag"] is True

        # Toast duration = 30s for severe per spec §2.6
        assert len(cap["toast_calls"]) == 1
        assert cap["toast_calls"][0]["duration_seconds"] == 30

        # Halt flag file written via PUBLIC set_external_halt_flag API
        assert isolated_cb.exists()
        cb_payload = json.loads(isolated_cb.read_text(encoding="utf-8"))
        assert cb_payload["level"] == "severe"
        assert cb_payload["auto_reset"] is False
        assert cb_payload["reason"].startswith("ops_watchdog:")
        # Reason content is the LLM-narrated summary (mocked here as
        # `annotated_answer` text); just verify it's non-empty post-prefix.
        assert len(cb_payload["reason"]) > len("ops_watchdog: ")
