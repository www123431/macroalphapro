"""
tests/test_auto_audit_promoter_executor.py — R-1.E promoter + executor.
"""
import json

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Promoter
# ─────────────────────────────────────────────────────────────────────────

def test_promote_writes_pending_approval(make_proposal):
    """gate-pass proposal → PendingApproval row + back-link on AuditProposal."""
    from engine.memory import SessionFactory, PendingApproval
    from engine.auto_audit_promoter import promote_to_pending_approval
    from engine.auto_audit_models import AuditProposal
    pid, _ = make_proposal()
    # Simulate gate having passed
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        p.gate_status = "pass"
        s.commit()
    res = promote_to_pending_approval(pid)
    assert res["ok"]
    assert res["pending_approval_id"] is not None
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        pa = s.get(PendingApproval, res["pending_approval_id"])
        assert pa.approval_type == "auto_audit_proposal"
        assert pa.status == "pending"
        assert pa.sector == "_audit_"  # sentinel for NOT NULL
        assert pa.ticker == "_AUDIT"
        assert p.pending_approval_id == pa.id


def test_promote_idempotent(make_proposal):
    """Second call returns existing pending_approval_id, doesn't create dupe."""
    from engine.memory import SessionFactory
    from engine.auto_audit_promoter import promote_to_pending_approval
    from engine.auto_audit_models import AuditProposal
    pid, _ = make_proposal()
    with SessionFactory() as s:
        s.get(AuditProposal, pid).gate_status = "pass"
        s.commit()
    r1 = promote_to_pending_approval(pid)
    r2 = promote_to_pending_approval(pid)
    assert r2["pending_approval_id"] == r1["pending_approval_id"]
    assert r2.get("skipped") == "already_promoted"


def test_promote_rejects_non_pass(make_proposal):
    """Without gate_status='pass', promotion is refused."""
    from engine.auto_audit_promoter import promote_to_pending_approval
    pid, _ = make_proposal()  # gate_status = None initially
    res = promote_to_pending_approval(pid)
    assert not res["ok"]


def test_promote_governance_required_priority_high(make_proposal,
                                                    make_proposal_payload):
    """governance_required=True → PendingApproval.priority = 'high'."""
    from engine.memory import SessionFactory, PendingApproval
    from engine.auto_audit_promoter import promote_to_pending_approval
    from engine.auto_audit_models import AuditProposal
    pid, _ = make_proposal()
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        p.gate_status = "pass"
        p.governance_required = True
        s.commit()
    res = promote_to_pending_approval(pid)
    with SessionFactory() as s:
        pa = s.get(PendingApproval, res["pending_approval_id"])
        assert pa.priority == "high"


# ─────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────

def test_executor_no_action_marks_finding_ignored(make_proposal,
                                                   make_proposal_payload):
    from engine.memory import SessionFactory, PendingApproval
    from engine.auto_audit_promoter import promote_to_pending_approval
    from engine.auto_audit_executor import execute_approved_proposal
    from engine.auto_audit_models import AuditProposal, AuditFinding
    payload = make_proposal_payload()
    payload["amendment_kind"] = "no_action"
    payload["options"][0]["risk_level"] = "LOW"
    pid, fid = make_proposal(payload=payload)
    with SessionFactory() as s:
        s.get(AuditProposal, pid).gate_status = "pass"
        s.commit()
    promote_to_pending_approval(pid)
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        pa_id = p.pending_approval_id
    res = execute_approved_proposal(pa_id, supervisor_id="test_user")
    assert res["ok"]
    assert res["action_taken"] == "marked_ignored"
    with SessionFactory() as s:
        f = s.get(AuditFinding, fid)
        assert f.status == "IGNORED"
        assert "no_action" in (f.notes or "")


def test_executor_amend_path_skips_unregistered_files(make_proposal,
                                                      make_proposal_payload):
    """If proposal cites a path NOT in spec_registry, executor reports skip
    not failure (supervisor must register first)."""
    from engine.memory import SessionFactory
    from engine.auto_audit_promoter import promote_to_pending_approval
    from engine.auto_audit_executor import execute_approved_proposal
    from engine.auto_audit_models import AuditProposal
    payload = make_proposal_payload()
    payload["options"][0]["files_to_touch"] = ["docs/spec_does_not_exist.md"]
    pid, _ = make_proposal(payload=payload)
    with SessionFactory() as s:
        s.get(AuditProposal, pid).gate_status = "pass"
        s.commit()
    promote_to_pending_approval(pid)
    with SessionFactory() as s:
        pa_id = s.get(AuditProposal, pid).pending_approval_id
    res = execute_approved_proposal(pa_id, supervisor_id="test_user")
    # Not "ok" since 0 amended; but no exception
    results = res["exec_detail"]["results"]
    assert any(r["status"] == "skipped_unregistered" for r in results)


def test_executor_rejects_short_rationale(make_proposal, make_proposal_payload):
    from engine.memory import SessionFactory
    from engine.auto_audit_promoter import promote_to_pending_approval
    from engine.auto_audit_executor import execute_approved_proposal
    from engine.auto_audit_models import AuditProposal
    payload = make_proposal_payload()
    payload["rationale_short"] = "too short"   # < 20 chars
    pid, _ = make_proposal(payload=payload)
    with SessionFactory() as s:
        # Force gate_status = 'pass' to bypass V4 (we want to test executor
        # defensive layer specifically)
        s.get(AuditProposal, pid).gate_status = "pass"
        s.commit()
    promote_to_pending_approval(pid)
    with SessionFactory() as s:
        pa_id = s.get(AuditProposal, pid).pending_approval_id
    res = execute_approved_proposal(pa_id, supervisor_id="test_user")
    assert not res["ok"]
    assert "rationale" in res["exec_detail"]["error"].lower()


def test_executor_rejects_non_pass_gate(make_proposal):
    from engine.memory import SessionFactory, PendingApproval
    from engine.auto_audit_executor import execute_approved_proposal
    from engine.auto_audit_models import AuditProposal
    pid, _ = make_proposal()
    # Manually create a stub PendingApproval row (gate_status not set)
    with SessionFactory() as s:
        import datetime
        pa = PendingApproval(
            approval_type="auto_audit_proposal",
            approval_class="llm_output",
            priority="normal",
            sector="_audit_",
            ticker="_AUDIT",
            triggered_condition="test",
            triggered_date=datetime.date.today(),
            status="pending",
        )
        s.add(pa); s.flush()
        pa_id = pa.id
        s.get(AuditProposal, pid).pending_approval_id = pa_id
        s.commit()
    res = execute_approved_proposal(pa_id, supervisor_id="test_user")
    assert not res["ok"]
    assert "gate_status" in res["exec_detail"]["error"]
