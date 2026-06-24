"""
tests/test_auto_audit_gate.py — R-1.D Layer 2 safety gate, all 10 V-rules.

Each V-rule gets at least one happy + one adversarial test case.
gate_proposal() integration test exercises persistence + governance flag.
"""
import json
import pytest


def test_baseline_payload_passes(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    ok, reasons, gov = validate_payload(make_proposal_payload())
    assert ok and not gov, f"baseline should pass: {reasons}"


def test_v1_missing_required_field(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    del p["amendment_kind"]
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V1" for r in reasons)


def test_v1_options_count_invalid(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["options"] = []
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V1" for r in reasons)


def test_v1_recommendation_index_out_of_range(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["recommendation_index"] = 99
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V1" for r in reasons)


def test_v2_forbidden_path_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["options"][0]["files_to_touch"] = ["engine/auto_audit.py"]
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V2" and r["severity"] == "fail" for r in reasons)


def test_v2_flagged_path_triggers_governance(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["options"][0]["files_to_touch"] = ["engine/portfolio.py"]
    ok, reasons, gov = validate_payload(p)
    assert ok and gov  # FLAGGED is not fail; only governance flag


def test_v3_unknown_amendment_kind_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["amendment_kind"] = "free_lunch"
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V3" for r in reasons)


def test_v4_short_rationale_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["rationale_short"] = "too short"
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V4" for r in reasons)


def test_v4_too_long_rationale_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["rationale_short"] = "X" * 600
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V4" for r in reasons)


def test_v5_diff_size_over_cap_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["options"][0]["diff_size_estimate"] = 75
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V5" and r["severity"] == "fail" for r in reasons)


def test_v6_empty_evidence_refs_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["evidence_refs"] = []
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V6" for r in reasons)


def test_v7_excess_effort_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["options"][0]["estimated_effort_min"] = 600
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V7" for r in reasons)


@pytest.mark.parametrize(
    "kind,risk,should_pass",
    [
        ("clarification",    "LOW",  True),
        ("clarification",    "MID",  True),
        ("clarification",    "HIGH", False),   # too high for clarification
        ("hypothesis_amend", "LOW",  False),   # too low for hypothesis_amend
        ("hypothesis_amend", "MID",  True),
        ("hypothesis_amend", "HIGH", True),
        ("threshold_tweak",  "HIGH", True),    # threshold_tweak is flexible
        ("no_action",        "LOW",  True),
        ("no_action",        "MID",  False),   # no_action requires LOW only
    ],
)
def test_v8_risk_kind_consistency_matrix(make_proposal_payload, kind, risk, should_pass):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["amendment_kind"] = kind
    p["options"][0]["risk_level"] = risk
    ok, reasons, _ = validate_payload(p)
    if should_pass:
        # might still fail on V5/V7/etc. — assert V8 specifically didn't fire
        v8_failures = [r for r in reasons if r["rule"] == "V8"]
        assert not v8_failures, f"V8 should not fire on {kind}+{risk}: {v8_failures}"
    else:
        assert not ok
        assert any(r["rule"] == "V8" for r in reasons)


def test_v9_production_signal_trigger_governance(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    # Use a kind that allows MID risk so V8 doesn't bite first
    p["amendment_kind"] = "threshold_tweak"
    p["options"][0]["risk_level"] = "MID"
    p["options"][0]["action"] = "Modify PRODUCTION_SIGNAL to switch to BAB"
    ok, reasons, gov = validate_payload(p)
    assert ok and gov
    assert any(r["rule"] == "V9" for r in reasons)


def test_v9_lowercase_bab_trigger_governance(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["amendment_kind"] = "threshold_tweak"
    p["options"][0]["risk_level"] = "MID"
    p["summary"] = "Something about betting against beta"
    ok, reasons, gov = validate_payload(p)
    assert ok and gov


def test_v10_self_reference_fails(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["evidence_refs"] = ["per Proposal #5 we should..."]
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V10" for r in reasons)


def test_v10_case_insensitive(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    p["evidence_refs"] = ["see PROPOSAL #12"]
    ok, reasons, _ = validate_payload(p)
    assert not ok
    assert any(r["rule"] == "V10" for r in reasons)


def test_v10_does_not_match_prose(make_proposal_payload):
    from engine.auto_audit_gate import validate_payload
    p = make_proposal_payload()
    # Mention "proposal" without # — should not trigger V10
    p["evidence_refs"] = ["Snapshot proposal context: verify hash"]
    ok, reasons, _ = validate_payload(p)
    assert ok or not any(r["rule"] == "V10" for r in reasons), \
        "V10 should require 'Proposal #N' pattern, not just word 'proposal'"


def test_payload_must_be_dict():
    from engine.auto_audit_gate import validate_payload
    ok, reasons, gov = validate_payload("not a dict")  # type: ignore
    assert not ok
    assert reasons[0]["rule"] == "V0"


# ─────────────────────────────────────────────────────────────────────────
# gate_proposal() integration: persistence + back-link + promoter trigger
# ─────────────────────────────────────────────────────────────────────────

def test_gate_proposal_persists_pass(make_proposal):
    """Pass proposal flips gate_status='pass' + writes PendingApproval row."""
    from engine.memory import PendingApproval, SessionFactory
    from engine.auto_audit_gate import gate_proposal
    from engine.auto_audit_models import AuditProposal
    pid, fid = make_proposal()
    res = gate_proposal(pid)
    assert res["gate_status"] == "pass"
    assert res["promotion"]["ok"]
    pa_id = res["promotion"]["pending_approval_id"]
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        assert p.gate_status == "pass"
        assert p.pending_approval_id == pa_id
        pa = s.get(PendingApproval, pa_id)
        assert pa.approval_type == "auto_audit_proposal"
        assert pa.status == "pending"


def test_gate_proposal_persists_fail(make_proposal, make_proposal_payload):
    """Fail proposal does NOT promote; gate_failure_reasons persisted."""
    from engine.memory import SessionFactory
    from engine.auto_audit_gate import gate_proposal
    from engine.auto_audit_models import AuditProposal
    bad = make_proposal_payload()
    bad["options"][0]["diff_size_estimate"] = 200  # V5 fail
    pid, _ = make_proposal(payload=bad)
    res = gate_proposal(pid)
    assert res["gate_status"] == "fail"
    assert res.get("promotion") in ({}, None)
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        assert p.gate_status == "fail"
        assert p.pending_approval_id is None
        reasons = json.loads(p.gate_failure_reasons_json)
        assert any(r["rule"] == "V5" for r in reasons)


def test_gate_proposal_governance_flag_persisted(make_proposal, make_proposal_payload):
    from engine.memory import PendingApproval, SessionFactory
    from engine.auto_audit_gate import gate_proposal
    from engine.auto_audit_models import AuditProposal
    flagged = make_proposal_payload()
    flagged["options"][0]["files_to_touch"] = ["engine/portfolio.py"]
    pid, _ = make_proposal(payload=flagged)
    res = gate_proposal(pid)
    assert res["gate_status"] == "pass"
    assert res["governance_required"] is True
    with SessionFactory() as s:
        p = s.get(AuditProposal, pid)
        assert p.governance_required is True
        pa = s.get(PendingApproval, p.pending_approval_id)
        assert pa.priority == "high"
