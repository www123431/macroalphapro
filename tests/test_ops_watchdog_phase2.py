"""
tests/test_ops_watchdog_phase2.py — Phase 2 unit tests (25 cases).

Coverage:
  - triage.py:  8 tests (constants integrity + aggregate_severity edge cases)
  - tools.py:   7 tests (dispatcher allow-list + per-tool defensive behavior)
  - prompt.py:  3 tests (role intro markers + query builder shape)
  - agent.py:   4 tests (dry-run skips LLM / triage flows through / trace JSON)
  - CLI:        3 tests (__main__ argument parsing + exit codes)
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ─────────────────────────────────────────────────────────────────────────────
# Shared isolated_db fixture (reuses pattern from Phase 1 tests)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    # Force-import every module that re-exports SessionFactory at top-level
    # BEFORE applying monkeypatch. Otherwise a fresh `from engine.memory
    # import SessionFactory` inside one of these modules (triggered for the
    # first time during the test) will capture the monkeypatched TestSession
    # and that binding never reverts when monkeypatch undoes.
    from engine import memory as _memory                    # noqa: F401
    from engine import db_models as _db_models              # noqa: F401
    from engine import universe_manager as _um              # noqa: F401
    from engine import auto_audit as _auto_audit            # noqa: F401
    from engine.db_models import Base

    db_path = tmp_path / "test_watchdog_phase2.db"
    test_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(test_engine)
    _um._Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, expire_on_commit=False)

    # Patch every module that holds a SessionFactory binding. monkeypatch
    # auto-reverts on test teardown so real DB references are restored.
    monkeypatch.setattr(_memory,     "engine",        test_engine)
    monkeypatch.setattr(_memory,     "SessionFactory", TestSession)
    monkeypatch.setattr(_db_models,  "engine",        test_engine)
    monkeypatch.setattr(_db_models,  "SessionFactory", TestSession)
    monkeypatch.setattr(_um,         "SessionFactory", TestSession)
    monkeypatch.setattr(_auto_audit, "SessionFactory", TestSession)
    return TestSession


# ═════════════════════════════════════════════════════════════════════════════
# triage.py — 8 tests
# ═════════════════════════════════════════════════════════════════════════════
class TestTriage:
    def test_severity_map_locked_has_13_modes(self):
        from engine.agents.ops_watchdog.triage import MODE_SEVERITY_MAP_LOCKED
        assert len(MODE_SEVERITY_MAP_LOCKED) == 13
        # All values are valid severity tokens
        valid = {"none", "light", "medium", "severe"}
        assert all(v in valid for v in MODE_SEVERITY_MAP_LOCKED.values())

    def test_rule_to_mode_locked_covers_all_phase1_rules(self):
        from engine.agents.ops_watchdog.triage import (
            RULE_TO_MODE_LOCKED, MODE_SEVERITY_MAP_LOCKED,
        )
        from engine.auto_audit_rules import WATCHDOG_RULES
        watchdog_rule_names = {r.__name__ for r in WATCHDOG_RULES}
        # Every Phase 1 rule must appear in RULE_TO_MODE
        missing = watchdog_rule_names - set(RULE_TO_MODE_LOCKED.keys())
        assert not missing, f"Phase 1 rules missing from RULE_TO_MODE: {missing}"
        # Every mode_key in RULE_TO_MODE must have a severity
        for rn, mk in RULE_TO_MODE_LOCKED.items():
            assert mk in MODE_SEVERITY_MAP_LOCKED, \
                f"mode {mk} (from rule {rn}) has no severity"

    def test_aggregate_empty_findings_returns_none_severity(self):
        from engine.agents.ops_watchdog.triage import aggregate_severity, SEVERITY_NONE
        result = aggregate_severity([])
        assert result["severity"] == SEVERITY_NONE
        assert result["n_findings"] == 0
        assert result["modes_fired"] == []

    def test_aggregate_single_medium_finding(self):
        from engine.agents.ops_watchdog.triage import aggregate_severity, SEVERITY_MEDIUM
        result = aggregate_severity(["rule_cycle_state_completion"])
        assert result["severity"] == SEVERITY_MEDIUM
        assert result["n_watchdog_findings"] == 1
        assert "mode_1_cycle_failed" in result["modes_fired"]

    def test_aggregate_severe_overrides_medium(self):
        from engine.agents.ops_watchdog.triage import aggregate_severity, SEVERITY_SEVERE
        result = aggregate_severity([
            "rule_cycle_state_completion",          # mode_1 medium
            "rule_rebalance_frequency_audit",       # mode_11 severe
            "rule_max_position_weight_vs_cap",      # mode_10 medium
        ])
        assert result["severity"] == SEVERITY_SEVERE

    def test_aggregate_escalates_light_to_medium_with_multiple(self):
        from engine.agents.ops_watchdog.triage import aggregate_severity, SEVERITY_MEDIUM
        # mode_4 sleeve_drift is LIGHT severity; 2+ findings → MEDIUM (§九)
        result = aggregate_severity([
            "rule_sleeve_id_integrity",   # mode_4 light (REUSED)
            "rule_sleeve_id_integrity",   # same rule fires twice → still 1 mode but 2 findings
        ])
        # n_mode_keys >= 2 means escalation; same rule twice → 2 mode_keys items
        assert result["severity"] == SEVERITY_MEDIUM
        assert result["escalation_applied"] is True

    def test_aggregate_unknown_rule_ignored(self):
        from engine.agents.ops_watchdog.triage import aggregate_severity, SEVERITY_NONE
        result = aggregate_severity([
            "rule_not_a_watchdog_rule",       # not in RULE_TO_MODE
            "another_unknown_rule",
        ])
        assert result["severity"] == SEVERITY_NONE
        assert result["n_findings"] == 2
        assert result["n_watchdog_findings"] == 0

    def test_auto_repairable_modes_subset_of_severity_map(self):
        from engine.agents.ops_watchdog.triage import (
            AUTO_REPAIRABLE_MODES_LOCKED, MODE_SEVERITY_MAP_LOCKED,
        )
        # Per spec §2.5: modes 1/2/4/6/10/12 are auto-repairable (6 total)
        assert len(AUTO_REPAIRABLE_MODES_LOCKED) == 6
        for m in AUTO_REPAIRABLE_MODES_LOCKED:
            assert m in MODE_SEVERITY_MAP_LOCKED


# ═════════════════════════════════════════════════════════════════════════════
# tools.py — 7 tests
# ═════════════════════════════════════════════════════════════════════════════
class TestTools:
    def test_registry_has_10_tools(self):
        from engine.agents.ops_watchdog.tools import (
            WATCHDOG_TOOL_REGISTRY, WATCHDOG_TOOL_NAMES,
        )
        assert len(WATCHDOG_TOOL_REGISTRY) == 10
        assert len(WATCHDOG_TOOL_NAMES) == 10
        # 5 NEW are present
        for name in ("read_audit_findings", "read_cycle_state",
                     "read_trade_log", "read_nav_change",
                     "read_historical_baseline"):
            assert name in WATCHDOG_TOOL_REGISTRY

    def test_dispatch_unknown_tool_returns_error(self):
        from engine.agents.ops_watchdog.tools import dispatch_watchdog_tool
        obs = dispatch_watchdog_tool("not_a_real_tool", {})
        assert "error" in obs and "unknown tool" in obs["error"]

    def test_dispatch_tool_arg_mismatch_returns_error(self):
        from engine.agents.ops_watchdog.tools import dispatch_watchdog_tool
        # read_historical_baseline requires `metric`
        obs = dispatch_watchdog_tool("read_historical_baseline", {"bogus_arg": 1})
        assert "error" in obs and "arg mismatch" in obs["error"]

    def test_read_audit_findings_empty_returns_empty_list(self, isolated_db):
        from engine.agents.ops_watchdog.tools import read_audit_findings
        result = read_audit_findings()
        assert result.success
        assert result.data["n_rows"] == 0
        assert result.data["rows"] == []

    def test_read_audit_findings_finds_seeded_row(self, isolated_db):
        from engine.agents.ops_watchdog.tools import read_audit_findings
        from engine.auto_audit_models import AuditFinding, AuditRun
        with isolated_db() as s:
            run = AuditRun(scope="watchdog", n_rules_run=1, n_findings=1)
            s.add(run)
            s.flush()
            s.add(AuditFinding(run_id=run.id, rule_name="rule_test",
                               severity="HIGH",
                               snapshot_json='{"x": 1}', status="OPEN"))
            s.commit()
        result = read_audit_findings()
        assert result.success and result.data["n_rows"] == 1
        assert result.data["rows"][0]["rule_name"] == "rule_test"
        assert result.data["rows"][0]["snapshot"] == {"x": 1}

    def test_read_historical_baseline_unknown_metric_errors(self, isolated_db):
        from engine.agents.ops_watchdog.tools import read_historical_baseline
        result = read_historical_baseline("not_a_metric")
        assert not result.success
        assert "unknown metric" in result.error_msg

    def test_read_historical_baseline_zero_observations(self, isolated_db):
        from engine.agents.ops_watchdog.tools import read_historical_baseline
        result = read_historical_baseline("nav_return", lookback_days=30)
        assert result.success
        assert result.data["n_obs"] == 0
        assert result.data["mean"] is None


# ═════════════════════════════════════════════════════════════════════════════
# prompt.py — 3 tests
# ═════════════════════════════════════════════════════════════════════════════
class TestPrompt:
    def test_role_intro_states_operations_layer_only(self):
        from engine.agents.ops_watchdog.prompt import WATCHDOG_ROLE_INTRO
        text = WATCHDOG_ROLE_INTRO.lower()
        # Must declare role and key invariants per §6
        assert "watchdog" in text
        assert "operations layer" in text
        assert "never" in text and "alpha decision" in text
        # Must say LLM does not decide severity / repair
        assert "do not decide severity" in text
        assert "do not decide whether to auto-repair" in text

    def test_role_intro_says_tool_required_before_final_answer(self):
        from engine.agents.ops_watchdog.prompt import WATCHDOG_ROLE_INTRO
        assert "MUST call at least one tool" in WATCHDOG_ROLE_INTRO

    def test_build_watchdog_query_includes_triage_and_findings(self):
        from engine.agents.ops_watchdog.prompt import build_watchdog_query
        triage = {
            "severity": "medium", "n_findings": 1,
            "modes_fired": ["mode_1_cycle_failed"],
        }
        q = build_watchdog_query(
            today_iso="2026-05-12",
            findings_preview=[{"rule_name": "rule_cycle_state_completion",
                                "severity": "MID", "snapshot": {"x": 1}}],
            triage_pre_summary=triage,
        )
        assert "2026-05-12" in q
        assert "rule_cycle_state_completion" in q
        assert "mode_1_cycle_failed" in q
        # Must instruct LLM to NOT propose actions
        assert "Do NOT propose actions" in q


# ═════════════════════════════════════════════════════════════════════════════
# agent.py — 4 tests
# ═════════════════════════════════════════════════════════════════════════════
class TestAgent:
    def test_dry_run_skips_llm_and_returns_triage(self, isolated_db, tmp_path,
                                                   monkeypatch):
        # Redirect trace dir to tmp
        from engine.agents.ops_watchdog import agent as _agent
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=True, save_trace=True)
        assert result.dry_run is True
        assert result.llm_used is False
        assert result.llm_cost_usd == 0.0
        assert result.audit_run_id is not None
        assert result.trace_json_path is not None
        assert Path(result.trace_json_path).exists()

    def test_no_findings_skips_llm_even_when_not_dry_run(self, isolated_db,
                                                         tmp_path, monkeypatch):
        from engine.agents.ops_watchdog import agent as _agent
        # Force an empty WATCHDOG_RULES list so run_audit produces 0 findings
        # regardless of DB state. (Some rules — e.g. rebalance_frequency_audit
        # — legitimately fire on empty DB because "no trades = no rebalances".)
        # Patch BOTH the source module and engine.auto_audit's bound name.
        from engine import auto_audit_rules as _aar
        from engine import auto_audit as _aa
        monkeypatch.setattr(_aar, "WATCHDOG_RULES", [])
        monkeypatch.setattr(_aa,  "WATCHDOG_RULES", [])
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        assert result.n_findings == 0
        assert result.triage["severity"] == "none"
        assert result.llm_used is False

    def test_trace_json_schema_version_and_spec_hash(self, isolated_db,
                                                     tmp_path, monkeypatch):
        from engine.agents.ops_watchdog import agent as _agent
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        result = _agent.run_watchdog(dry_run=True, save_trace=True)
        path = Path(result.trace_json_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        # schema_version 2 since Phase 3 added auto_repair field
        assert payload["schema_version"] >= 1
        assert payload["spec_id"] == 63
        # Post-amendment-1 hash; future amendments may bump this
        assert payload["spec_hash"] == "9d050804"

    def test_llm_invocation_uses_ops_watchdog_agent_id(self, isolated_db,
                                                       tmp_path, monkeypatch):
        """When LLM IS used, run_react_agent must receive agent_id='ops_watchdog'."""
        from engine.agents.ops_watchdog import agent as _agent
        from engine.agents.ops_watchdog.triage import SEVERITY_MEDIUM
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")

        # Seed one finding to force triage != none
        from engine.auto_audit_models import AuditFinding, AuditRun

        def fake_run_audit(scope):
            # Bypass real rules — return synthetic findings via direct AuditRun
            from engine.memory import SessionFactory
            with SessionFactory() as s:
                run = AuditRun(scope=scope, n_rules_run=11, n_findings=1,
                               exit_status="ok")
                s.add(run)
                s.flush()
                s.add(AuditFinding(run_id=run.id,
                                   rule_name="rule_cycle_state_completion",
                                   severity="MID",
                                   snapshot_json='{"kind":"cycle_failed"}',
                                   status="OPEN"))
                s.commit()
                return {"run_id": run.id, "scope": scope,
                        "n_rules_run": 11, "n_findings": 1,
                        "exit_status": "ok"}

        monkeypatch.setattr("engine.auto_audit.run_audit", fake_run_audit)

        call_capture = {}
        def fake_run_react(*args, **kwargs):
            call_capture.update(kwargs)
            from engine.quant_co_pilot.base import TraceResult
            return TraceResult(
                query="", final_answer="ok", citations=[],
                annotated_answer="ok", steps=[], cost_usd=0.0,
                latency_ms=0, abort_reason=None, completed_at="2026-05-12T00:00:00Z",
            )
        monkeypatch.setattr("engine.quant_co_pilot.base.run_react_agent",
                            fake_run_react)
        # Mock toast channel — mode_1 = medium → would pop real Windows toast
        # during pytest (test isolation hygiene per Phase 6 catch).
        monkeypatch.setattr(
            "engine.agents.ops_watchdog.notifications._send_windows_toast",
            lambda **kw: True,
        )

        result = _agent.run_watchdog(dry_run=False, save_trace=False)
        assert result.llm_used is True
        assert call_capture["agent_id"] == "ops_watchdog"
        # Spec §2.3 cost cap $0.20 (NOT Tool 1's $0.05)
        assert call_capture["cost_budget_usd"] == 0.20
        # Spec §2.3 max 8 steps
        assert call_capture["max_steps"] == 8


# ═════════════════════════════════════════════════════════════════════════════
# CLI — 3 tests
# ═════════════════════════════════════════════════════════════════════════════
class TestCli:
    def test_cli_help_does_not_crash(self, capsys):
        from engine.agents.ops_watchdog.__main__ import _parse_args
        with pytest.raises(SystemExit):
            _parse_args(["--help"])
        out = capsys.readouterr().out
        assert "spec id=63" in out

    def test_cli_dry_run_flag_parsed(self):
        from engine.agents.ops_watchdog.__main__ import _parse_args
        ns = _parse_args(["--dry-run"])
        assert ns.dry_run is True
        assert ns.verbose is False

    def test_cli_returns_zero_on_success(self, isolated_db, tmp_path,
                                          monkeypatch, capsys):
        from engine.agents.ops_watchdog import agent as _agent
        from engine.agents.ops_watchdog.__main__ import main
        monkeypatch.setattr(_agent, "_trace_dir",
                            lambda: tmp_path / "ops_watchdog")
        rc = main(["--dry-run", "--no-save-trace"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Watchdog" in out
