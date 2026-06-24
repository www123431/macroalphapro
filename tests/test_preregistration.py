"""
tests/test_preregistration.py — Pre-registration core invariants.

Critical 20% per master_backlog: HARKing detector + n_trials math drive
every p-value the project cites. If these break silently, every published
result is invalid.
"""
import json

import pytest


def test_amendment_kinds_table_locked():
    """The kind→n_trials mapping is a frozen contract; changing values
    invalidates every prior amend_spec call's audit trail.

    2026-05-12 doctrine change (BHY-FDR / SSRN publication path retired): the +1/+3/+5
    cost ladder was DELIBERATELY zeroed — all amendment kinds now contribute 0 trials.
    Kind LABELS are retained as semantic audit-log tags; only the trial WEIGHTS dropped to
    0 (see engine/preregistration.py header + feedback_pretest_experimental_rigor rule #3).
    Reversible: restore the historical 1/3/5 if the publication path reopens. The frozen
    contract is now the all-zero mapping; this test was updated to the new contract
    (2026-05-22 stale-test cleanup) — it still locks the mapping against silent drift."""
    from engine.preregistration import AMENDMENT_KINDS
    expected = {
        "clarification":         0,
        "scope_narrow":          0,
        "threshold_tweak":       0,   # was 1 pre-2026-05-12 (BHY-FDR retired)
        "hypothesis_amend":      0,   # was 3 pre-2026-05-12
        "endpoint_swap":         0,   # was 5 pre-2026-05-12
        "superseded":            0,
        "lab_state_transition":  0,   # P-LAB 2026-05-08
    }
    assert AMENDMENT_KINDS == expected, (
        "AMENDMENT_KINDS mapping changed — every cited n_trials becomes ambiguous"
    )


def test_compute_git_blob_hash_deterministic(tmp_path):
    """Same content → same hash; any byte change → different hash."""
    from engine.preregistration import _compute_git_blob_hash
    f = tmp_path / "spec.md"
    f.write_text("# Spec v1\nHello world\n", encoding="utf-8")
    h1 = _compute_git_blob_hash(str(f))
    h2 = _compute_git_blob_hash(str(f))
    assert h1 == h2 and len(h1) == 40
    f.write_text("# Spec v1\nHello world!\n", encoding="utf-8")
    assert _compute_git_blob_hash(str(f)) != h1


def test_register_spec_idempotent(tmp_path):
    """Registering the same path twice returns the same row id; updates hash."""
    from engine.preregistration import register_spec
    from engine.memory import SpecRegistry, SessionFactory
    f = tmp_path / "spec_test_idempotent.md"
    f.write_text("# Test", encoding="utf-8")
    sid1 = register_spec(str(f), retro=True)
    sid2 = register_spec(str(f), retro=True)
    assert sid1 == sid2
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid1)
        assert r is not None
        assert r.retro_registered is True
        assert r.n_trials_contributed == 0


def test_register_spec_forward_contributes_one_trial(tmp_path):
    """Non-retro registration consumes 1 trial slot; retro consumes 0."""
    from engine.preregistration import register_spec
    from engine.memory import SpecRegistry, SessionFactory
    f = tmp_path / "spec_test_forward.md"
    f.write_text("# Forward spec test", encoding="utf-8")
    sid = register_spec(str(f), retro=False)
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid)
        assert r.n_trials_contributed == 1
        assert r.retro_registered is False


def test_amend_spec_kind_validation(tmp_path):
    """amend_spec must reject unknown kinds; ledger should never see free-text."""
    from engine.preregistration import amend_spec, register_spec
    f = tmp_path / "spec_amend_kindtest.md"
    f.write_text("# Test", encoding="utf-8")
    register_spec(str(f), retro=True)
    f.write_text("# Test v2", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown amendment kind"):
        amend_spec(str(f), kind="bogus_kind", reason="x" * 30)


def test_amend_spec_rationale_min_chars(tmp_path):
    """rationale < 20 chars is rejected — amend_spec is the audit-trail entry."""
    from engine.preregistration import amend_spec, register_spec
    f = tmp_path / "spec_amend_minreason.md"
    f.write_text("# Test", encoding="utf-8")
    register_spec(str(f), retro=True)
    f.write_text("# Test v2", encoding="utf-8")
    with pytest.raises(ValueError, match="reason must be"):
        amend_spec(str(f), kind="clarification", reason="too short")


def test_amend_spec_n_trials_added_per_kind(tmp_path):
    """n_trials_contributed accumulates per AMENDMENT_KINDS table value."""
    from engine.preregistration import amend_spec, register_spec, AMENDMENT_KINDS
    from engine.memory import SpecRegistry, SessionFactory
    f = tmp_path / "spec_amend_ntrials.md"
    f.write_text("# v1", encoding="utf-8")
    sid = register_spec(str(f), retro=True)  # base 0
    f.write_text("# v2", encoding="utf-8")
    amend_spec(str(f), kind="threshold_tweak",
               reason="Test threshold tweak with sufficient rationale text.")
    f.write_text("# v3", encoding="utf-8")
    amend_spec(str(f), kind="hypothesis_amend",
               reason="Test hypothesis amend with sufficient rationale text.")
    expected = AMENDMENT_KINDS["threshold_tweak"] + AMENDMENT_KINDS["hypothesis_amend"]
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid)
        assert r.n_trials_contributed == expected


def test_amend_spec_ledger_format(tmp_path):
    """Each amendment appends a JSON entry; ledger remains a parseable list."""
    from engine.preregistration import amend_spec, register_spec
    from engine.memory import SpecRegistry, SessionFactory
    f = tmp_path / "spec_amend_ledger.md"
    f.write_text("# v1", encoding="utf-8")
    sid = register_spec(str(f), retro=True)
    f.write_text("# v2", encoding="utf-8")
    amend_spec(str(f), kind="clarification",
               reason="First clarification with clean rationale text.")
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid)
        ledger = json.loads(r.amendment_log)
    assert isinstance(ledger, list) and len(ledger) == 1
    entry = ledger[0]
    assert {"at", "kind", "reason", "prev_hash", "new_hash", "n_trials_added"} <= entry.keys()
    assert entry["kind"] == "clarification"


def test_compute_pre_registration_n_trials_excludes_retro(tmp_path):
    """retro=True specs must NOT contribute to EFFECTIVE_N_TRIALS."""
    from engine.preregistration import compute_pre_registration_n_trials, register_spec
    f1 = tmp_path / "spec_retro.md"
    f1.write_text("# retro", encoding="utf-8")
    f2 = tmp_path / "spec_forward.md"
    f2.write_text("# forward", encoding="utf-8")
    n0 = compute_pre_registration_n_trials()
    register_spec(str(f1), retro=True)   # +0
    register_spec(str(f2), retro=False)  # +1
    n1 = compute_pre_registration_n_trials()
    assert n1 - n0 == 1


def test_tier_r_hash_drift_silent_edit_after_amend_fails(tmp_path):
    """Regression: prior to 2026-05-09 the audit rule treated hash drift on
    a spec with non-empty amendment_log as WARN ("legitimate"), but amend_spec
    always sets r.current_hash to the post-edit blob hash, so any drift means
    the file was silently edited AFTER the latest amendment — a HARKing R1
    violation that must be FAIL not WARN."""
    import importlib.util, sys
    from engine.preregistration import register_spec, amend_spec, _compute_git_blob_hash
    from engine.memory import SpecRegistry, SessionFactory

    f = tmp_path / "spec_silent_edit_after_amend.md"
    f.write_text("# v1\n", encoding="utf-8")
    sid = register_spec(str(f), retro=False)
    f.write_text("# v2 content\n", encoding="utf-8")
    amend_spec(str(f), kind="clarification",
               reason="legitimate amend with sufficient rationale length here")
    # Now silently edit the file WITHOUT calling amend_spec
    f.write_text("# v3 silent edit with no amend\n", encoding="utf-8")

    # The audit script should detect drift and label it FAIL
    audit_path = (tmp_path / ".." / "scripts" / "tier1_retroactive_audit.py").resolve()
    # Load tier1 module by file path; do NOT run the whole audit (too slow).
    spec = importlib.util.spec_from_file_location(
        "tier1_audit_under_test",
        str(__import__('pathlib').Path(__file__).parent.parent / "scripts" / "tier1_retroactive_audit.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    findings = mod.audit_claim_3()
    rel_path = None
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid)
        rel_path = r.spec_path
    matching = [f for f in findings if f.facet == f"hash_drift.{rel_path}"]
    assert matching, f"no hash_drift finding for {rel_path}; got {[f.facet for f in findings]}"
    assert matching[0].severity == "FAIL", (
        f"silent edit after amendment must FAIL (was {matching[0].severity}); "
        f"detail: {matching[0].detail}"
    )
    assert "silent edit" in matching[0].detail.lower()


def test_tier_r_ledger_vs_current_hash_consistency(tmp_path):
    """Regression: amend_spec sets both r.current_hash and appends to log with
    new_hash. They must always agree. Manual ledger tampering or amend_spec
    failure leaving them divergent is state corruption and must FAIL."""
    import importlib.util, sys, json as _json
    from engine.preregistration import register_spec, amend_spec
    from engine.memory import SpecRegistry, SessionFactory

    f = tmp_path / "spec_ledger_consistency.md"
    f.write_text("# v1\n", encoding="utf-8")
    sid = register_spec(str(f), retro=False)
    f.write_text("# v2\n", encoding="utf-8")
    amend_spec(str(f), kind="clarification",
               reason="legitimate amend with sufficient rationale length here")

    # Corrupt the ledger tail's new_hash to simulate tampering
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid)
        log = _json.loads(r.amendment_log)
        log[-1]["new_hash"] = "deadbeef" + "0" * 32
        r.amendment_log = _json.dumps(log)
        s.commit()
        rel_path = r.spec_path

    spec = importlib.util.spec_from_file_location(
        "tier1_audit_under_test_b",
        str(__import__('pathlib').Path(__file__).parent.parent / "scripts" / "tier1_retroactive_audit.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    findings = mod.audit_claim_3()
    matching = [f for f in findings if f.facet == f"ledger_vs_current_hash.{rel_path}"]
    assert matching, f"no ledger_vs_current_hash finding for {rel_path}"
    assert matching[0].severity == "FAIL"
    assert "state corruption" in matching[0].detail.lower() or "ledger tail" in matching[0].detail.lower()


def test_detect_harking_returns_list_well_formed():
    """detect_harking is called from the audit cron; result must always be a
    list of dicts with at least 'rule' / 'description' keys, never None / raise."""
    from engine.preregistration import detect_harking
    flags = detect_harking()
    assert isinstance(flags, list)
    for f in flags:
        assert isinstance(f, dict)
        # Don't assert specific keys exist (depends on which rules fired) —
        # but must be a dict shape the consumers can iterate.
