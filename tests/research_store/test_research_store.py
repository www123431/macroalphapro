"""Tests for engine.research_store. Run as:

    pytest tests/research_store/ -v

Tests mock out the data dir to a tmp location so they don't pollute the
real store. Schema / registry / store / emit are exercised end-to-end with
a synthetic artifact file.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# Redirect store paths to a tmp dir for the whole test module
@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Patch the three module-level paths to point at tmp_path."""
    store_dir = tmp_path / "research_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    # Reload-safe: patch module attributes after they've been imported.
    from engine.research_store import registry, store
    monkeypatch.setattr(registry, "_STORE_DIR",     store_dir)
    monkeypatch.setattr(registry, "_SUBJECTS_PATH", store_dir / "subjects.yaml")
    monkeypatch.setattr(registry, "_ALIASES_PATH",  store_dir / "aliases.yaml")
    monkeypatch.setattr(store,    "_EVENTS_PATH",   store_dir / "events.jsonl")

    # Stub artifact-existence: emit's _validate_artifacts uses Path,
    # we'll create real files in test for that.
    yield {"store_dir": store_dir}


def test_register_subject_and_resolve(isolated_store):
    from engine.research_store import registry
    from engine.research_store.schema import SubjectType

    s = registry.register_subject(
        "test_subject_a",
        subject_type=SubjectType.factor,
        family="test_family",
        description="A test subject",
    )
    assert s.subject_id == "test_subject_a"
    assert s.subject_type == "factor"

    # Resolve direct
    assert registry.resolve("test_subject_a").subject_id == "test_subject_a"
    # Resolve unknown
    assert registry.resolve("nonexistent") is None


def test_subject_not_registered_error_carries_suggestions(isolated_store):
    from engine.research_store import registry
    from engine.research_store.exceptions import SubjectNotRegisteredError
    from engine.research_store.schema import SubjectType

    registry.register_subject("phase_a_v3", subject_type=SubjectType.factor)
    registry.register_subject("phase_b_v1", subject_type=SubjectType.factor)

    with pytest.raises(SubjectNotRegisteredError) as exc:
        registry.require("phase-a-v3")  # typo: dash vs underscore
    msg = str(exc.value)
    assert "phase_a_v3" in msg
    assert "did you mean" in msg.lower()


def test_register_alias_and_resolve(isolated_store):
    from engine.research_store import registry
    from engine.research_store.schema import SubjectType

    registry.register_subject("canonical_id", subject_type=SubjectType.factor)
    registry.register_alias("canonical_id", "old_name_v0")

    # Alias resolves to canonical
    resolved = registry.resolve("old_name_v0")
    assert resolved.subject_id == "canonical_id"


def test_register_alias_rejects_when_alias_is_registered_subject(isolated_store):
    from engine.research_store import registry
    from engine.research_store.schema import SubjectType

    registry.register_subject("a", subject_type=SubjectType.factor)
    registry.register_subject("b", subject_type=SubjectType.factor)
    with pytest.raises(ValueError) as exc:
        registry.register_alias(canonical="a", alias="b")
    assert "registered subject" in str(exc.value)


def test_emit_factor_verdict_end_to_end(isolated_store, tmp_path):
    from engine.research_store import emit, registry, store
    from engine.research_store.schema import SubjectType, EventType, Verdict

    # Set up artifact files (emit requires they exist)
    evidence_path = tmp_path / "evidence.md"
    evidence_path.write_text("# Evidence", encoding="utf-8")
    data_dir = tmp_path / "rundir"
    data_dir.mkdir()

    # Register subject
    registry.register_subject(
        "phase_a_v3",
        subject_type=SubjectType.factor,
        family="position_weighting",
    )

    event_id = emit.factor_verdict(
        subject_id="phase_a_v3",
        verdict="RED",
        metrics={"best_deflsr": 0.155, "n_cells": 16},
        artifacts={
            "evidence_doc": str(evidence_path),
            "data_dir":     str(data_dir),
        },
        summary="16/16 cells fail; 1/N retained.",
    )

    assert event_id and len(event_id) > 10

    # Round-trip via store
    events = store.all_events()
    assert len(events) == 1
    e = events[0]
    assert e.event_id == event_id
    assert e.event_type == EventType.factor_verdict_filed
    assert e.verdict == Verdict.RED
    assert e.subject_id == "phase_a_v3"
    assert e.family == "position_weighting"
    assert "evidence_doc" in e.artifacts


def test_emit_rejects_missing_artifact(isolated_store, tmp_path):
    from engine.research_store import emit, registry
    from engine.research_store.exceptions import ArtifactMissingError
    from engine.research_store.schema import SubjectType

    registry.register_subject("subj", subject_type=SubjectType.factor)
    with pytest.raises(ArtifactMissingError):
        emit.factor_verdict(
            subject_id="subj",
            verdict="RED",
            metrics={},
            artifacts={"evidence_doc": str(tmp_path / "does_not_exist.md")},
            summary="x",
        )


def test_emit_rejects_unregistered_subject(isolated_store, tmp_path):
    from engine.research_store import emit
    from engine.research_store.exceptions import SubjectNotRegisteredError

    evidence_path = tmp_path / "ev.md"
    evidence_path.write_text("x", encoding="utf-8")

    with pytest.raises(SubjectNotRegisteredError):
        emit.factor_verdict(
            subject_id="never_registered",
            verdict="RED", metrics={},
            artifacts={"evidence_doc": str(evidence_path)},
            summary="x",
        )


def test_emit_resolves_alias_to_canonical(isolated_store, tmp_path):
    from engine.research_store import emit, registry, store
    from engine.research_store.schema import SubjectType

    registry.register_subject("canonical_x", subject_type=SubjectType.factor)
    registry.register_alias("canonical_x", "alias_y")

    evidence_path = tmp_path / "ev.md"; evidence_path.write_text("x", encoding="utf-8")
    emit.factor_verdict(
        subject_id="alias_y",      # caller uses alias
        verdict="GREEN", metrics={},
        artifacts={"evidence_doc": str(evidence_path)},
        summary="ok",
    )
    e = store.all_events()[0]
    assert e.subject_id == "canonical_x"   # stored under canonical, not alias


def test_emit_summary_validation(isolated_store, tmp_path):
    from engine.research_store import emit, registry
    from engine.research_store.exceptions import InvalidEventError
    from engine.research_store.schema import SubjectType

    registry.register_subject("s", subject_type=SubjectType.factor)
    ev = tmp_path / "x.md"; ev.write_text("x", encoding="utf-8")

    # empty summary
    with pytest.raises(InvalidEventError):
        emit.factor_verdict(
            "s", "RED", {}, {"evidence_doc": str(ev)}, summary="",
        )

    # too-long summary
    with pytest.raises(InvalidEventError):
        emit.factor_verdict(
            "s", "RED", {}, {"evidence_doc": str(ev)}, summary="x" * 500,
        )


def test_duplicate_event_id_rejected(isolated_store, tmp_path):
    """Direct store.append() with a duplicate event_id should raise."""
    from engine.research_store import store
    from engine.research_store.exceptions import DuplicateEventError
    from engine.research_store.schema import EventType, ResearchEvent, SubjectType, Verdict

    ev = ResearchEvent(
        event_id="fixed-id-for-test",
        event_type=EventType.factor_verdict_filed,
        ts="2026-06-02T00:00:00Z", session_id="s", actor="test",
        subject_type=SubjectType.factor, subject_id="x",
        verdict=Verdict.GREEN, metrics={}, artifacts={},
        parent_event_ids=(), family=None, tags=(),
        summary="x", git_sha="abc",
    )
    store.append(ev)
    with pytest.raises(DuplicateEventError):
        store.append(ev)


def test_filter_events(isolated_store, tmp_path):
    from engine.research_store import emit, registry, store
    from engine.research_store.schema import SubjectType

    registry.register_subject("alpha", subject_type=SubjectType.factor, family="fam1")
    registry.register_subject("beta",  subject_type=SubjectType.factor, family="fam2")
    ev = tmp_path / "x.md"; ev.write_text("x", encoding="utf-8")

    emit.factor_verdict("alpha", "GREEN", {}, {"evidence_doc": str(ev)}, "ok")
    emit.factor_verdict("alpha", "RED",   {}, {"evidence_doc": str(ev)}, "later")
    emit.factor_verdict("beta",  "GREEN", {}, {"evidence_doc": str(ev)}, "ok")

    assert len(store.filter_events()) == 3
    assert len(store.filter_events(verdict="RED")) == 1
    assert len(store.filter_events(family="fam1")) == 2
    assert len(store.filter_events(subject_id="beta")) == 1


def test_round_trip_serialization(isolated_store):
    """Schema to_dict / from_dict round-trip preserves all fields."""
    from engine.research_store.schema import EventType, ResearchEvent, SubjectType, Verdict

    ev = ResearchEvent(
        event_id="abc", event_type=EventType.factor_verdict_filed,
        ts="2026-06-02T00:00:00Z", session_id="s1", actor="claude",
        subject_type=SubjectType.factor, subject_id="x",
        verdict=Verdict.MARGINAL,
        metrics={"sharpe": 0.83, "t": 4.36},
        artifacts={"evidence_doc": "docs/foo.md"},
        parent_event_ids=("parent1", "parent2"),
        family="fam_x", tags=("tag1", "tag2"),
        summary="round trip", git_sha="deadbeef",
    )
    d = ev.to_dict()
    ev2 = ResearchEvent.from_dict(d)
    assert ev2 == ev
