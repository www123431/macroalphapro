"""tests/test_emit_memory_amendment_proposed.py — Phase 2.0 step 13.

Direct tests for emit.memory_amendment_proposed:
  - event type + subject (the doctrine memory file slug) + verdict
  - metrics carry full B payload (proposed_amendment + reasoning + confidence)
  - artifacts include the draft path
  - tags include the canonical labels
  - summary surfaces blocking_doctrine_id + first slice of amendment
"""
from __future__ import annotations

from engine.research_store import emit
from engine.research_store.schema import EventType, SubjectType, Verdict


def _patch_append(monkeypatch):
    captured: list = []
    from engine.research_store import store
    monkeypatch.setattr(store, "append", lambda ev: captured.append(ev))
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "_publish_to_bus", lambda ev: None)
    # Patch artifact existence check — these tests use sentinel paths
    # (the file write is a separate concern verified in
    # test_strengthener_approvals.py::test_write_amendment_draft_*).
    monkeypatch.setattr(emit_mod, "_validate_artifacts", lambda artifacts: None)
    return captured


def test_writes_event_type_and_subject(monkeypatch):
    captured = _patch_append(monkeypatch)
    eid = emit.memory_amendment_proposed(
        hypothesis_id              = "hyp-xyz",
        blocking_doctrine_id       = "project-test-doctrine-2026",
        proposed_amendment_summary = "Carve out X from the ban",
        b_reasoning                = "evidence A B C",
        draft_doc_path             = "data/strengthener/amendment_drafts/amendment_hyp-xyz.md",
        b_confidence               = 0.78,
    )
    assert eid
    ev = captured[0]
    assert ev.event_type == EventType.memory_amendment_proposed
    assert ev.subject_id == "project-test-doctrine-2026"
    assert ev.subject_type == SubjectType.memory_doctrine
    assert ev.verdict == Verdict.NEUTRAL


def test_metrics_carry_b_payload(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.memory_amendment_proposed(
        hypothesis_id              = "hyp1",
        blocking_doctrine_id       = "doc1",
        proposed_amendment_summary = "Carve out X",
        b_reasoning                = "reasoning text",
        draft_doc_path             = "drafts/x.md",
        b_confidence               = 0.71,
    )
    m = captured[0].metrics
    assert m["hypothesis_id"] == "hyp1"
    assert "Carve out X" in m["proposed_amendment_summary"]
    assert m["b_reasoning"] == "reasoning text"
    assert m["b_confidence"] == 0.71


def test_artifacts_include_draft_path(monkeypatch):
    """Audit consumers walk artifacts to find the actual proposed
    amendment markdown — must be present."""
    captured = _patch_append(monkeypatch)
    emit.memory_amendment_proposed(
        hypothesis_id              = "hyp1",
        blocking_doctrine_id       = "doc1",
        proposed_amendment_summary = "x",
        b_reasoning                = "y",
        draft_doc_path             = "data/strengthener/amendment_drafts/test.md",
    )
    assert captured[0].artifacts == {
        "amendment_draft": "data/strengthener/amendment_drafts/test.md",
    }


def test_tags_canonical_labels(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.memory_amendment_proposed(
        hypothesis_id              = "hyp1",
        blocking_doctrine_id       = "doc1",
        proposed_amendment_summary = "x",
        b_reasoning                = "y",
        draft_doc_path             = "x.md",
    )
    assert "memory_amendment" in captured[0].tags
    assert "proposed" in captured[0].tags


def test_summary_surfaces_doctrine_and_amendment(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.memory_amendment_proposed(
        hypothesis_id              = "h",
        blocking_doctrine_id       = "project-cross-asset-breadth-2026-05-28",
        proposed_amendment_summary = "Carve out EM sov QMJ from the ban",
        b_reasoning                = "x",
        draft_doc_path             = "x.md",
    )
    s = captured[0].summary
    assert "project-cross-asset-breadth-2026-05-28" in s
    assert "EM sov QMJ" in s


def test_summary_truncates_long_amendment(monkeypatch):
    """Amendments > 200 chars truncated in summary (full lives in
    metrics['proposed_amendment_summary'] up to 400)."""
    captured = _patch_append(monkeypatch)
    long_amend = "Q" * 600
    emit.memory_amendment_proposed(
        hypothesis_id              = "h",
        blocking_doctrine_id       = "doc",
        proposed_amendment_summary = long_amend,
        b_reasoning                = "x",
        draft_doc_path             = "x.md",
    )
    # Summary trimmed
    assert len(captured[0].summary) < 400
    # Metrics also capped at 400
    assert len(captured[0].metrics["proposed_amendment_summary"]) == 400


def test_auto_registers_doctrine_subject_idempotent(monkeypatch):
    """Subject auto-registration is idempotent — calling emit twice
    on the same blocking_doctrine_id should not raise."""
    _patch_append(monkeypatch)
    emit.memory_amendment_proposed(
        hypothesis_id="h1", blocking_doctrine_id="doc-twice-test",
        proposed_amendment_summary="x", b_reasoning="y", draft_doc_path="x.md",
    )
    # Second emit must not raise on registry conflict
    emit.memory_amendment_proposed(
        hypothesis_id="h2", blocking_doctrine_id="doc-twice-test",
        proposed_amendment_summary="x", b_reasoning="y", draft_doc_path="x.md",
    )
