"""
tests/test_auto_audit_orchestrator.py — R-1.A run_audit invariants.

Critical: run_audit's error containment (one bad rule doesn't kill the
sweep) is core to the cron's value. Tests use fake rules registered
into the live registries — clean up after.
"""
import pytest


@pytest.fixture
def temp_critical_rules():
    """Snapshot/restore CRITICAL_RULES so each test gets a clean slate."""
    from engine import auto_audit_rules
    saved = list(auto_audit_rules.CRITICAL_RULES)
    auto_audit_rules.CRITICAL_RULES.clear()
    yield auto_audit_rules.CRITICAL_RULES
    auto_audit_rules.CRITICAL_RULES.clear()
    auto_audit_rules.CRITICAL_RULES.extend(saved)


@pytest.fixture
def temp_weekly_rules():
    from engine import auto_audit_rules
    saved = list(auto_audit_rules.WEEKLY_RULES)
    auto_audit_rules.WEEKLY_RULES.clear()
    yield auto_audit_rules.WEEKLY_RULES
    auto_audit_rules.WEEKLY_RULES.clear()
    auto_audit_rules.WEEKLY_RULES.extend(saved)


def test_run_audit_no_rules_status(temp_weekly_rules):
    """Empty registry returns exit_status='no_rules' (not 'ok')."""
    from engine.auto_audit import run_audit
    summary = run_audit("weekly")
    assert summary["n_rules_run"] == 0
    assert summary["n_findings"] == 0
    assert summary["exit_status"] == "no_rules"


def test_run_audit_clean_returns_ok(temp_critical_rules):
    """All rules return None → exit_status='ok', no findings."""
    from engine.auto_audit import run_audit
    temp_critical_rules.append(lambda: None)
    temp_critical_rules.append(lambda: None)
    summary = run_audit("critical")
    assert summary["n_rules_run"] == 2
    assert summary["n_findings"] == 0
    assert summary["n_errors"] == 0
    assert summary["exit_status"] == "ok"


def test_run_audit_finding_persisted(temp_critical_rules):
    """Rule returning a finding dict persists an AuditFinding row."""
    from engine.auto_audit import run_audit
    from engine.memory import SessionFactory
    from engine.auto_audit_models import AuditFinding

    def _rule_fires():
        return {"severity": "MID", "snapshot": {"detail": "test"}}
    temp_critical_rules.append(_rule_fires)

    summary = run_audit("critical")
    assert summary["n_findings"] == 1

    with SessionFactory() as s:
        f = s.query(AuditFinding).filter_by(run_id=summary["run_id"]).first()
        assert f is not None
        assert f.severity == "MID"
        assert f.rule_name == "_rule_fires"


def test_run_audit_one_bad_rule_does_not_abort(temp_critical_rules):
    """If 1 rule raises, others still execute; status='partial'."""
    from engine.auto_audit import run_audit

    def _rule_explode():
        raise RuntimeError("synthetic boom")

    def _rule_clean():
        return None

    def _rule_fires():
        return {"severity": "LOW", "snapshot": {}}

    temp_critical_rules.extend([_rule_explode, _rule_clean, _rule_fires])
    summary = run_audit("critical")
    assert summary["n_rules_run"] == 3
    assert summary["n_errors"] == 1
    assert summary["n_findings"] == 1
    assert summary["exit_status"] == "partial"


def test_run_audit_silenceable_dedup(temp_critical_rules):
    """If a finding for a rule was IGNORED in last 30d, new findings are suppressed."""
    import datetime
    import json
    from engine.auto_audit import run_audit
    from engine.memory import SessionFactory
    from engine.auto_audit_models import AuditFinding, AuditRun

    # Pre-seed an IGNORED finding for the test rule
    def _test_rule_for_silence():
        return {"severity": "LOW", "snapshot": {}}
    temp_critical_rules.append(_test_rule_for_silence)
    rule_name = "_test_rule_for_silence"

    with SessionFactory() as s:
        run = AuditRun(scope="critical", n_rules_run=1, n_findings=1,
                       exit_status="ok")
        s.add(run); s.flush()
        s.add(AuditFinding(
            run_id=run.id, rule_name=rule_name, severity="LOW",
            detected_at=datetime.datetime.utcnow(),
            snapshot_json=json.dumps({}), status="IGNORED",
        ))
        s.commit()

    summary = run_audit("critical")
    assert summary["n_suppressed"] == 1
    assert summary["n_findings"] == 0


def test_run_audit_unknown_scope_raises():
    from engine.auto_audit import run_audit
    with pytest.raises(ValueError, match="Unknown audit scope"):
        run_audit("never_a_scope")  # type: ignore
