"""tests/test_emit_candidate_skipped_pre_compute.py — Phase 2.0 step 8.

Direct tests for emit.candidate_skipped_pre_compute. Verifies:
  - Event type + subject + verdict mapping
  - metrics carry spec_hash + attack_vector + reasoning + confidence
  - tags include the canonical 'pre_compute_da' + 'skipped' labels
  - summary surfaces family + attack_vector
  - parent_event_ids propagate (lineage to candidate_pipeline_started
    on re-runs is not yet built, but the schema supports it)
  - reasoning truncated at 600 chars (audit doesn't need essays)
"""
from __future__ import annotations

from engine.research_store import emit, registry
from engine.research_store.schema import EventType, SubjectType, Verdict


def _patch_append(monkeypatch):
    captured: list = []
    from engine.research_store import store
    monkeypatch.setattr(store, "append", lambda ev: captured.append(ev))
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "_publish_to_bus", lambda ev: None)
    return captured


def _ensure_subject(subject_id: str):
    try:
        registry.require(subject_id)
    except Exception:
        registry.register_subject(
            subject_id=subject_id, subject_type=SubjectType.factor,
            description="(test guard for pre_compute skip)",
            created_by="test_emit_candidate_skipped_pre_compute",
        )


# ─────────────────────────────────────────────────────────────────────
# Shape
# ─────────────────────────────────────────────────────────────────────
def test_emit_writes_correct_event_type_and_subject(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    eid = emit.candidate_skipped_pre_compute(
        subject_id           = "auto_test_subject",
        spec_hash            = "abc123def456",
        source_hypothesis_id = "hyp-uuid-test-xyz",
        attack_vector        = "EARNINGS_DRIFT has 7 recent REDs in same cell",
        reasoning            = "Family already over-mined; same weighting tweak as prior REDs.",
        confidence           = 0.85,
        family               = "EARNINGS_DRIFT",
    )
    assert eid
    assert len(captured) == 1
    ev = captured[0]
    assert ev.event_type == EventType.candidate_skipped_pre_compute
    assert ev.subject_id == "auto_test_subject"
    assert ev.verdict == Verdict.NEUTRAL
    assert ev.family == "EARNINGS_DRIFT"


def test_metrics_carry_da_payload(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.candidate_skipped_pre_compute(
        subject_id           = "auto_test_subject",
        spec_hash            = "abc123",
        source_hypothesis_id = "hyp-xyz",
        attack_vector        = "test attack",
        reasoning            = "test reasoning",
        confidence           = 0.72,
    )
    m = captured[0].metrics
    assert m["spec_hash"] == "abc123"
    assert m["source_hypothesis_id"] == "hyp-xyz"
    assert m["attack_vector"] == "test attack"
    assert m["confidence"] == 0.72
    assert m["reasoning"] == "test reasoning"


def test_tags_include_canonical_labels(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.candidate_skipped_pre_compute(
        subject_id           = "auto_test_subject",
        spec_hash            = "x",
        source_hypothesis_id = "y",
        attack_vector        = "z",
        reasoning            = "w",
        confidence           = 0.5,
    )
    assert "pre_compute_da" in captured[0].tags
    assert "skipped" in captured[0].tags


def test_summary_includes_family_and_attack(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.candidate_skipped_pre_compute(
        subject_id           = "auto_test_subject",
        spec_hash            = "x",
        source_hypothesis_id = "11111111aaaa",
        attack_vector        = "CARRY graveyard hit on G10 cell",
        reasoning            = "z",
        confidence           = 0.7,
        family               = "CARRY",
    )
    s = captured[0].summary
    assert "CARRY" in s
    assert "graveyard hit" in s
    assert "11111111" in s   # subject_id slice


def test_long_reasoning_truncated(monkeypatch):
    """Audit payloads should NOT carry essays — caller's reasoning
    truncated to 600 chars at emit time."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.candidate_skipped_pre_compute(
        subject_id           = "auto_test_subject",
        spec_hash            = "x",
        source_hypothesis_id = "y",
        attack_vector        = "z",
        reasoning            = "Q" * 2000,
        confidence           = 0.5,
    )
    assert len(captured[0].metrics["reasoning"]) == 600


def test_parent_event_ids_propagate(monkeypatch):
    """Lineage to a hypothetical candidate_pipeline_started event
    should pass through (current autopilot_live doesn't emit start
    before the skip, but the field supports future wiring)."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.candidate_skipped_pre_compute(
        subject_id           = "auto_test_subject",
        spec_hash            = "x",
        source_hypothesis_id = "y",
        attack_vector        = "z",
        reasoning            = "w",
        confidence           = 0.5,
        parent_event_ids     = ("ev_upstream_1", "ev_upstream_2"),
    )
    assert captured[0].parent_event_ids == ("ev_upstream_1", "ev_upstream_2")
