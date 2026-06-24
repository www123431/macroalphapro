"""tests/test_l2_propose_loop.py — L2 propose -> approve -> (record-only) loop.

Pins the L2-3 action seam end-to-end:
  - propose_action files ONE advisory PendingApproval (proposal-only, status=pending).
  - resolve_pending_approval accepts + PERSISTS the P-AUDIT v1 audit fields
    (review_rationale / review_category). This is a REGRESSION GUARD: those params were
    dropped in a refactor, which silently broke bulk_resolve -> /api/approvals/resolve ->
    the UI Approve/Reject path. Restored 2026-05-23.
  - An 'advisory' approval is RECORD-ONLY on approve: it must NOT auto-execute a trade
    (auto-execution is reserved for entry/risk_control/rebalance types). This is the
    0-LLM-in-DECISION guarantee: the LLM proposes, the human approves, and an advisory
    proposal changes nothing on the book.

Each test cleans up its own row so the real Approvals inbox is never polluted.
"""
from __future__ import annotations

import inspect
import json

from engine.memory import PendingApproval, SessionFactory, resolve_pending_approval


def _delete(approval_id: int) -> None:
    with SessionFactory() as s:
        row = s.get(PendingApproval, approval_id)
        if row is not None:
            s.delete(row)
            s.commit()


def test_resolver_accepts_audit_fields():
    """Regression guard: the dropped P-AUDIT v1 params are back in the signature."""
    params = inspect.signature(resolve_pending_approval).parameters
    assert "review_rationale" in params, "resolve_pending_approval lost review_rationale"
    assert "review_category" in params, "resolve_pending_approval lost review_category"


def test_propose_then_approve_is_record_only():
    # A NON-position proposal stays advisory + record-only. (A position directive with
    # ticker+weight now routes to an EXECUTABLE 'overlay' proposal — covered in
    # tests/test_overlay_executor.py, not here.)
    from engine.agents.persona.tools import execute_tool

    out, is_err = execute_tool("propose_action", {
        "kind": "risk_review",
        "detail": "TEST(pytest): flag the carry sleeve for a decay review",
        "rationale": "L2-3 regression test — safe to delete.",
    })
    assert not is_err, out
    res = json.loads(out)
    assert res.get("ok") is True
    pid = res["approval_id"]

    try:
        # filed as a pending advisory proposal
        with SessionFactory() as s:
            row = s.get(PendingApproval, pid)
            assert row is not None
            assert row.status == "pending"
            assert row.approval_type == "advisory"
            assert row.approval_class == "agent_proposal"

        # approve through the executor with the audit fields
        result = resolve_pending_approval(
            approval_id=pid, approved=True, resolved_by="pytest",
            review_rationale="L2-3 test: approving advisory proposal.",
            review_category="other",
        )
        assert result["ok"] is True
        # RECORD-ONLY: advisory matches no executable branch -> no trade side-effect
        assert result["exec_detail"] == {}, f"advisory must not auto-execute: {result}"

        # audit fields persisted on the row
        with SessionFactory() as s:
            row = s.get(PendingApproval, pid)
            assert row.status == "approved"
            assert row.resolved_by == "pytest"
            assert (row.review_rationale or "").startswith("L2-3 test")
            assert row.review_category == "other"
    finally:
        _delete(pid)


def test_propose_then_reject_via_bulk_resolver():
    """The UI path: bulk_resolve_pending_approvals -> resolve_pending_approval."""
    from engine.agents.persona.tools import execute_tool
    from engine.approval_workflow import bulk_resolve_pending_approvals

    out, _ = execute_tool("propose_action", {
        "kind": "risk_review", "detail": "TEST(pytest): flag K1 BAB for review",
        "rationale": "L2-3 reject-path regression test — safe to delete.",
    })
    pid = json.loads(out)["approval_id"]
    try:
        rr = bulk_resolve_pending_approvals(
            approval_ids=[pid], approved=False, resolved_by="pytest",
            review_rationale="L2-3 test: rejecting the proposal.",
            review_category="other",
        )
        assert rr["submitted"] == 1
        assert rr["skipped"] == [], rr
        assert rr["resolved"] and rr["resolved"][0]["ok"] is True, rr
        with SessionFactory() as s:
            row = s.get(PendingApproval, pid)
            assert row.status == "rejected"
            assert row.review_category == "other"
    finally:
        _delete(pid)
