"""tests/test_approval_charter.py — the HITL governance-inbox charter (2026-05-24).

Pins the routing predicate + the per-type "what Approve does" effect lines so the
inbox can never silently overstate what approving an item does, and so the retired
sector-overlay `risk_control` path stays record-only (never gated to the inbox).

Doctrine under test: human-ON-the-loop. Only entry/risk_control/rebalance auto-execute
in resolve_pending_approval(); everything else is record-only governance.
"""
from __future__ import annotations

from engine import approval_charter as ch


def test_executing_types_match_resolver():
    # The resolver (engine/memory.py resolve_pending_approval) only auto-executes
    # these types. The charter MUST agree — anything else claiming executes=True
    # would be a lie the UI repeats. `overlay` added 2026-05-24 (L2 executor).
    assert ch.EXECUTING_TYPES == frozenset({"entry", "risk_control", "rebalance", "overlay"})


def test_effect_executes_flag_is_faithful():
    for t in ("entry", "risk_control", "rebalance", "overlay"):
        assert ch.approval_effect(t)["executes"] is True, t
    for t in ("advisory", "anomaly_screener", "auto_audit_proposal",
              "factor_candidate", "universe_change", "cash_flow", "track_b"):
        assert ch.approval_effect(t)["executes"] is False, t


def test_effect_has_both_languages():
    for t in list(ch.EXECUTING_TYPES) + list(ch.GOVERNANCE_INBOX_TYPES) + ["totally_unknown"]:
        eff = ch.approval_effect(t)
        assert eff["en"] and eff["zh"], t
        assert isinstance(eff["executes"], bool)


def test_unknown_type_defaults_record_only():
    eff = ch.approval_effect("something_new")
    assert eff["executes"] is False
    assert "Record-only" in eff["en"]


def test_advisory_is_record_only():
    # The L2 CoS propose_action seam writes advisory/agent_proposal. Approving it
    # must never move the book (0-LLM-in-DECISION).
    assert ch.approval_effect("advisory")["executes"] is False
    assert ch.is_governance_inbox_item("advisory")


def test_risk_control_is_retired_discretionary():
    # Sector overlay stop signals: legacy rows still carry an executing effect,
    # but the type is retired from creating new pending inbox items.
    assert ch.is_retired_discretionary("risk_control")
    assert "risk_control" not in ch.GOVERNANCE_INBOX_TYPES


def test_governance_types_are_not_retired():
    for t in ch.GOVERNANCE_INBOX_TYPES:
        assert not ch.is_retired_discretionary(t), t


def test_retired_trace_fields_make_record_only_row():
    f = ch.retired_trace_fields()
    assert f["status"] == "approved"            # pre-resolved → never shows as pending
    assert f["approval_class"] == "routine_review"
    assert f["resolved_by"] == ch.SECTOR_RETIRED_RESOLVED_BY
    assert f["resolved_at"] is not None
    assert "retired" in f["review_rationale"].lower()


def test_charter_prose_present_both_langs():
    assert ch.CHARTER["en"] and ch.CHARTER["zh"]
    # the load-bearing claim: the live book does not route here
    assert "HARD-HALT" in ch.CHARTER["en"]
    assert "HARD-HALT" in ch.CHARTER["zh"]


def test_none_safe():
    assert ch.is_governance_inbox_item(None) is False
    assert ch.is_retired_discretionary(None) is False
    assert ch.approval_effect(None)["executes"] is False
