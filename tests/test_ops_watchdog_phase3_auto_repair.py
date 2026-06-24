"""
tests/test_ops_watchdog_phase3_auto_repair.py — Phase 3 unit tests (25 cases).

Coverage:
  - auto_repair.py recipes (10 tests): each recipe defensive, run_daily_batch
    mocked to avoid real production side effects; deferred stubs always fail;
    retry counter behavior; AuditProposal write path.
  - execute_repair_for_finding (5 tests): success on attempt 1 / retry then
    success / 3 retries exhausted / deferred returns immediately / unknown
    mode returns deferred.
  - Tier R guardrail (4 tests): clean file passes / poisoned file with raw
    SQL fires HIGH / pattern detection per line.
  - Orchestrator integration (6 tests): run_watchdog wires auto_repair only
    when not dry_run; auto_repair_summary fields populated; trace JSON v2
    schema includes auto_repair; integration with mocked run_daily_batch;
    deferred modes show up correctly.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── isolated_db fixture (mirrors Phase 2; patches engine.auto_audit too) ────
@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    from engine import memory as _memory                    # noqa: F401
    from engine import db_models as _db_models              # noqa: F401
    from engine import universe_manager as _um              # noqa: F401
    from engine import auto_audit as _auto_audit            # noqa: F401
    from engine.db_models import Base

    db_path = tmp_path / "test_watchdog_phase3.db"
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


# ═════════════════════════════════════════════════════════════════════════════
# auto_repair.py — recipe-level tests (10)
# ═════════════════════════════════════════════════════════════════════════════
class TestRecipes:
    def test_registry_has_6_modes(self):
        from engine.agents.ops_watchdog.auto_repair import AUTO_REPAIR_RECIPES_LOCKED
        assert len(AUTO_REPAIR_RECIPES_LOCKED) == 6
        for mode in ("mode_1_cycle_failed", "mode_2_yfinance_stale",
                     "mode_6_trade_execution_missing"):
            assert mode in AUTO_REPAIR_RECIPES_LOCKED

    def test_deferred_modes_set_has_3(self):
        from engine.agents.ops_watchdog.auto_repair import DEFERRED_MODES_LOCKED
        assert DEFERRED_MODES_LOCKED == frozenset({
            "mode_4_sleeve_drift",
            "mode_10_weight_cap_violation",
            "mode_12_regime_scale_misapplied",
        })

    def test_max_retry_attempts_is_3(self):
        from engine.agents.ops_watchdog.auto_repair import MAX_RETRY_ATTEMPTS
        assert MAX_RETRY_ATTEMPTS == 3

    def test_recipe_1_calls_run_daily_batch_force(self, monkeypatch):
        from engine.agents.ops_watchdog import auto_repair as _ar
        captured = {}

        class FakeResult:
            skipped = False

        def fake_run(*, as_of_date, force):
            captured["as_of_date"] = as_of_date
            captured["force"] = force
            return FakeResult()
        monkeypatch.setattr("engine.daily_batch.run_daily_batch", fake_run)

        ok, detail = _ar._repair_retry_idempotent_batch(
            finding={"snapshot": {"issues": [{"kind": "cycle_failed"}]}})
        assert ok is True
        assert captured["force"] is True
        assert detail["phase"] == "run_daily_batch_completed"

    def test_recipe_2_records_pre_state(self, monkeypatch):
        from engine.agents.ops_watchdog import auto_repair as _ar

        class FakeResult:
            skipped = False

        monkeypatch.setattr("engine.daily_batch.run_daily_batch",
                            lambda **kw: FakeResult())
        ok, detail = _ar._repair_force_fresh_fetch(
            finding={"snapshot": {"n_stale": 5, "n_missing": 2}})
        assert ok is True
        assert detail["n_stale_pre"] == 5
        assert detail["n_missing_pre"] == 2

    def test_recipe_6_records_n_orphans(self, monkeypatch):
        from engine.agents.ops_watchdog import auto_repair as _ar

        class FakeResult:
            skipped = False

        monkeypatch.setattr("engine.daily_batch.run_daily_batch",
                            lambda **kw: FakeResult())
        ok, detail = _ar._repair_retry_execution_if_signal_active(
            finding={"snapshot": {"n_orphans": 4}})
        assert ok is True
        assert detail["n_orphans_pre"] == 4

    def test_recipe_returns_false_on_run_daily_batch_exception(self, monkeypatch):
        from engine.agents.ops_watchdog import auto_repair as _ar

        def boom(**kw):
            raise RuntimeError("synthetic_batch_crash")
        monkeypatch.setattr("engine.daily_batch.run_daily_batch", boom)
        ok, detail = _ar._repair_retry_idempotent_batch(finding={})
        assert ok is False
        assert "synthetic_batch_crash" in detail["error"]

    def test_recipe_returns_false_on_import_failure(self, monkeypatch):
        """If engine.daily_batch import fails (extreme edge), recipe must
        not crash."""
        from engine.agents.ops_watchdog import auto_repair as _ar
        import sys
        saved = sys.modules.pop("engine.daily_batch", None)
        try:
            sys.modules["engine.daily_batch"] = type(
                "BadMod", (), {"run_daily_batch": None}
            )()
            # The recipe attempts a function call; will fail with TypeError.
            ok, detail = _ar._repair_retry_idempotent_batch(finding={})
            assert ok is False
        finally:
            if saved is not None:
                sys.modules["engine.daily_batch"] = saved

    def test_stub_deferred_always_fails(self):
        from engine.agents.ops_watchdog.auto_repair import _stub_deferred
        ok, detail = _stub_deferred({"rule_name": "rule_sleeve_id_integrity"})
        assert ok is False
        assert detail["phase"] == "deferred"
        assert "rule_sleeve_id_integrity" in detail["rule_name"]

    def test_deferred_modes_dispatched_to_stub(self):
        from engine.agents.ops_watchdog.auto_repair import (
            AUTO_REPAIR_RECIPES_LOCKED, _stub_deferred,
        )
        for mode in ("mode_4_sleeve_drift", "mode_10_weight_cap_violation",
                     "mode_12_regime_scale_misapplied"):
            assert AUTO_REPAIR_RECIPES_LOCKED[mode] is _stub_deferred


# ═════════════════════════════════════════════════════════════════════════════
# execute_repair_for_finding — retry / escalation (5)
# ═════════════════════════════════════════════════════════════════════════════
class TestExecuteRepair:
    def test_success_on_first_attempt(self, isolated_db, monkeypatch):
        from engine.agents.ops_watchdog.auto_repair import execute_repair_for_finding

        class FakeResult:
            skipped = False
        monkeypatch.setattr("engine.daily_batch.run_daily_batch",
                            lambda **kw: FakeResult())
        result = execute_repair_for_finding({
            "rule_name":  "rule_cycle_state_completion",   # mode_1
            "finding_id": None,
            "snapshot":   {"issues": []},
        })
        assert result.success is True
        assert result.deferred is False
        assert result.n_attempts == 1
        assert result.error is None
        assert len(result.attempts_log) == 1
        assert result.attempts_log[0]["status"] == "success"

    def test_retry_then_success(self, isolated_db, monkeypatch):
        from engine.agents.ops_watchdog.auto_repair import execute_repair_for_finding
        call_count = {"n": 0}

        class FakeResult:
            skipped = False

        def flaky(**kw):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("transient")
            return FakeResult()
        monkeypatch.setattr("engine.daily_batch.run_daily_batch", flaky)

        result = execute_repair_for_finding({
            "rule_name":  "rule_cycle_state_completion",
            "finding_id": None,
            "snapshot":   {},
        })
        assert result.success is True
        assert result.n_attempts == 2
        assert result.attempts_log[0]["status"] == "failed"
        assert result.attempts_log[1]["status"] == "success"

    def test_3_retries_exhausted(self, isolated_db, monkeypatch):
        from engine.agents.ops_watchdog.auto_repair import execute_repair_for_finding

        def always_fail(**kw):
            raise RuntimeError("permanent_failure")
        monkeypatch.setattr("engine.daily_batch.run_daily_batch", always_fail)

        result = execute_repair_for_finding({
            "rule_name":  "rule_cycle_state_completion",
            "finding_id": None,
            "snapshot":   {},
        })
        assert result.success is False
        assert result.deferred is False
        assert result.n_attempts == 3
        assert result.error is not None
        assert "permanent_failure" in result.error

    def test_deferred_mode_returns_immediately(self, isolated_db):
        from engine.agents.ops_watchdog.auto_repair import execute_repair_for_finding
        result = execute_repair_for_finding({
            "rule_name":  "rule_sleeve_id_integrity",       # mode_4 DEFERRED
            "finding_id": None,
            "snapshot":   {},
        })
        assert result.success is False
        assert result.deferred is True
        assert result.n_attempts == 1
        assert result.mode_key == "mode_4_sleeve_drift"
        assert result.error == "recipe_deferred"

    def test_unknown_mode_returns_deferred(self, isolated_db):
        from engine.agents.ops_watchdog.auto_repair import execute_repair_for_finding
        result = execute_repair_for_finding({
            "rule_name":  "not_a_watchdog_rule",
            "finding_id": None,
            "snapshot":   {},
        })
        assert result.deferred is True
        assert result.success is False
        assert result.error == "mode_not_in_recipe_table"


# ═════════════════════════════════════════════════════════════════════════════
# Tier R guardrail rule_watchdog_auto_repair_no_raw_sql (4)
# ═════════════════════════════════════════════════════════════════════════════
class TestNoRawSqlGuardrail:
    def test_clean_auto_repair_returns_none(self):
        from engine.auto_audit_rules import rule_watchdog_auto_repair_no_raw_sql
        # Real auto_repair.py should be clean
        result = rule_watchdog_auto_repair_no_raw_sql()
        assert result is None, f"Real auto_repair.py has raw SQL: {result}"

    def test_handles_missing_file(self, monkeypatch, tmp_path):
        """If auto_repair.py doesn't exist (e.g. pre-Phase-3 state), no finding."""
        from engine import auto_audit_rules as _aar
        import pathlib
        # Force the target path to a nonexistent location
        fake_dir = tmp_path / "engine" / "agents" / "ops_watchdog"
        fake_dir.mkdir(parents=True)
        # No auto_repair.py created
        original_file = pathlib.Path(_aar.__file__).resolve()
        # Use monkeypatch to redirect target detection
        # The rule uses Path(_aar.__file__).resolve().parent — patching __file__
        # is awkward; instead, just rename temporarily
        target = original_file.parent / "agents" / "ops_watchdog" / "auto_repair.py"
        backup = target.with_suffix(".py.bak")
        if target.exists():
            target.rename(backup)
        try:
            result = _aar.rule_watchdog_auto_repair_no_raw_sql()
            assert result is None
        finally:
            if backup.exists():
                backup.rename(target)

    def test_detects_text_call_pattern(self, monkeypatch, tmp_path):
        """Inject `text(...)` pattern → rule fires HIGH severity."""
        from engine import auto_audit_rules as _aar
        from pathlib import Path
        target = (Path(_aar.__file__).resolve().parent
                  / "agents" / "ops_watchdog" / "auto_repair.py")
        original = target.read_text(encoding="utf-8")
        poisoned = original + "\n\ndef _bad():\n    text('UPDATE x SET y=1')\n"
        try:
            target.write_text(poisoned, encoding="utf-8")
            result = _aar.rule_watchdog_auto_repair_no_raw_sql()
            assert result is not None
            assert result["severity"] == "HIGH"
            patterns = [v["pattern"] for v in result["snapshot"]["violations"]]
            assert "text(" in patterns
        finally:
            target.write_text(original, encoding="utf-8")

    def test_detects_update_simulated_pattern(self, monkeypatch, tmp_path):
        """Inject `UPDATE simulated_positions SET ...` → fires HIGH."""
        from engine import auto_audit_rules as _aar
        from pathlib import Path
        target = (Path(_aar.__file__).resolve().parent
                  / "agents" / "ops_watchdog" / "auto_repair.py")
        original = target.read_text(encoding="utf-8")
        poisoned = original + '\n\ndef _bad():\n    sql = "UPDATE simulated_trades SET cost_bps=0"\n'
        try:
            target.write_text(poisoned, encoding="utf-8")
            result = _aar.rule_watchdog_auto_repair_no_raw_sql()
            assert result is not None
            assert result["severity"] == "HIGH"
            patterns = [v["pattern"] for v in result["snapshot"]["violations"]]
            assert "UPDATE simulated_" in patterns
        finally:
            target.write_text(original, encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator integration (6)
# ═════════════════════════════════════════════════════════════════════════════
class TestOrchestratorIntegration:
    def test_dry_run_skips_auto_repair(self, isolated_db, tmp_path,
                                        monkeypatch):
        from engine.agents.ops_watchdog import agent as _agent
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=True, save_trace=False)
        # dry_run skips auto-repair entirely
        assert result.auto_repair["n_attempted"] == 0
        assert result.auto_repair["results"] == []

    def test_auto_repair_summary_fields_present(self, isolated_db, tmp_path,
                                                 monkeypatch):
        """WatchdogRunResult.auto_repair has expected schema even when empty."""
        from engine.agents.ops_watchdog import agent as _agent
        from engine import auto_audit_rules as _aar
        from engine import auto_audit as _aa
        monkeypatch.setattr(_aar, "WATCHDOG_RULES", [])
        monkeypatch.setattr(_aa,  "WATCHDOG_RULES", [])
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        assert set(result.auto_repair.keys()) >= {
            "n_attempted", "n_succeeded", "n_failed", "n_deferred", "results",
        }

    def test_trace_json_v2_includes_auto_repair(self, isolated_db, tmp_path,
                                                 monkeypatch):
        from engine.agents.ops_watchdog import agent as _agent
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=True, save_trace=True)
        payload = json.loads(Path(result.trace_json_path).read_text(
            encoding="utf-8"))
        # Phase 3 bumped schema to v2; Phase 4 bumped to v3 (added notifications).
        # Forward-compat: just assert >= 2 so test survives future schema additions.
        assert payload["schema_version"] >= 2
        assert "auto_repair" in payload
        assert payload["auto_repair"]["n_attempted"] == 0  # dry_run

    def test_active_recipe_fires_when_finding_exists(self, isolated_db, tmp_path,
                                                      monkeypatch):
        """Seed a mode-1 finding; non-dry-run; mocked run_daily_batch should be called."""
        from engine.agents.ops_watchdog import agent as _agent
        from engine.auto_audit_models import AuditFinding, AuditRun

        # Mock run_daily_batch so we don't actually trigger production
        captured = {"called": False}

        class FakeResult:
            skipped = False

        def fake_run(*, as_of_date=None, force=False):
            captured["called"] = True
            captured["force"] = force
            return FakeResult()
        monkeypatch.setattr("engine.daily_batch.run_daily_batch", fake_run)
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        # Mock Windows toast — orchestrator path goes through emit_notification
        # with mode_1 = medium → real toast would pop during pytest run. Hygiene.
        monkeypatch.setattr(
            "engine.agents.ops_watchdog.notifications._send_windows_toast",
            lambda **kw: True,
        )

        # Force run_audit to seed 1 mode-1 finding
        def fake_run_audit(scope):
            from engine.memory import SessionFactory
            with SessionFactory() as s:
                run = AuditRun(scope=scope, n_rules_run=11, n_findings=1,
                               exit_status="ok")
                s.add(run); s.flush()
                s.add(AuditFinding(
                    run_id=run.id, rule_name="rule_cycle_state_completion",
                    severity="MID",
                    snapshot_json='{"issues":[{"kind":"cycle_failed"}]}',
                    status="OPEN",
                    detected_at=datetime.datetime.utcnow(),
                ))
                s.commit()
                return {"run_id": run.id, "scope": scope,
                        "n_rules_run": 11, "n_findings": 1,
                        "exit_status": "ok"}
        monkeypatch.setattr("engine.auto_audit.run_audit", fake_run_audit)

        # Stub LLM ReAct so we don't actually call Gemini
        from engine.quant_co_pilot.base import TraceResult
        def fake_react(*args, **kwargs):
            return TraceResult(query="", final_answer="ok", citations=[],
                               annotated_answer="ok", steps=[], cost_usd=0.0,
                               latency_ms=0, abort_reason=None,
                               completed_at="2026-05-12T00:00:00Z")
        monkeypatch.setattr("engine.quant_co_pilot.base.run_react_agent",
                            fake_react)

        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        assert captured["called"] is True
        assert captured["force"] is True
        assert result.auto_repair["n_attempted"] == 1
        assert result.auto_repair["n_succeeded"] == 1
        assert result.auto_repair["n_failed"] == 0

    def test_deferred_mode_in_summary(self, isolated_db, tmp_path,
                                       monkeypatch):
        """Seed a mode-4 (deferred) finding; result should show n_deferred=1."""
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
                    run_id=run.id, rule_name="rule_sleeve_id_integrity",
                    severity="HIGH",
                    snapshot_json='{"n_issues":1}', status="OPEN",
                    detected_at=datetime.datetime.utcnow(),
                ))
                s.commit()
                return {"run_id": run.id, "scope": scope,
                        "n_rules_run": 11, "n_findings": 1,
                        "exit_status": "ok"}
        monkeypatch.setattr("engine.auto_audit.run_audit", fake_run_audit)

        # Stub LLM
        from engine.quant_co_pilot.base import TraceResult
        monkeypatch.setattr("engine.quant_co_pilot.base.run_react_agent",
                            lambda *a, **k: TraceResult(
                                query="", final_answer="ok", citations=[],
                                annotated_answer="ok", steps=[], cost_usd=0.0,
                                latency_ms=0, abort_reason=None,
                                completed_at="2026-05-12T00:00:00Z",
                            ))
        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        assert result.auto_repair["n_attempted"] == 1
        assert result.auto_repair["n_deferred"] == 1
        assert result.auto_repair["n_succeeded"] == 0

    def test_audit_proposal_persisted_on_success(self, isolated_db, tmp_path,
                                                  monkeypatch):
        """When recipe succeeds, AuditProposal row is written."""
        from engine.agents.ops_watchdog import agent as _agent
        from engine.auto_audit_models import AuditFinding, AuditProposal, AuditRun
        from engine.memory import SessionFactory

        class FakeResult:
            skipped = False
        monkeypatch.setattr("engine.daily_batch.run_daily_batch",
                            lambda **kw: FakeResult())
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")

        # Seed a finding via direct DB write (need its id for AuditProposal FK)
        with isolated_db() as s:
            run = AuditRun(scope="watchdog", n_rules_run=11, n_findings=1,
                           exit_status="ok")
            s.add(run); s.flush()
            f = AuditFinding(
                run_id=run.id, rule_name="rule_cycle_state_completion",
                severity="MID",
                snapshot_json='{"issues":[{"kind":"cycle_failed"}]}',
                status="OPEN",
                detected_at=datetime.datetime.utcnow(),
            )
            s.add(f); s.commit()
            finding_id = f.id
            run_id = run.id

        # Directly invoke execute_repair_for_finding (not full orchestrator)
        from engine.agents.ops_watchdog.auto_repair import (
            execute_repair_for_finding,
        )
        repair_result = execute_repair_for_finding({
            "rule_name":  "rule_cycle_state_completion",
            "finding_id": finding_id,
            "snapshot":   {"issues": [{"kind": "cycle_failed"}]},
        })
        assert repair_result.success is True
        assert repair_result.audit_proposal_id is not None

        # Verify AuditProposal row + AuditFinding.status=RESOLVED
        with isolated_db() as s:
            prop = s.query(AuditProposal).filter_by(
                finding_id=finding_id).first()
            assert prop is not None
            assert prop.generation_status == "success"
            assert prop.gate_status == "pass"
            assert prop.model_version == "auto_repair_v1"
            assert prop.cost_usd == 0.0
            payload = json.loads(prop.parsed_payload_json)
            assert payload["success"] is True
            assert payload["n_attempts"] == 1
            # AuditFinding.status updated
            finding_row = s.query(AuditFinding).filter_by(id=finding_id).first()
            assert finding_row.status == "RESOLVED"
