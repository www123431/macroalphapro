"""Tests for engine.sessions — full state machine + protocol checkers.

Run as: pytest tests/sessions/ -v
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    """Redirect store paths to tmp."""
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    from engine.sessions import store
    monkeypatch.setattr(store, "_SESSIONS_PATH", sdir / "sessions.jsonl")
    monkeypatch.setattr(store, "_ACTIVE_PATH",   sdir / "_active.json")
    yield {"sdir": sdir}


def test_open_creates_pending_preflight(isolated_sessions):
    from engine.sessions import lifecycle, store
    from engine.sessions.schema import SessionType, SessionState

    s = lifecycle.open_session(SessionType.research_new, title="Test factor X")
    assert s.state == SessionState.pending_preflight
    assert s.session_id
    assert s.preflight_digest is None

    # Active pointer set
    active = store.get_active()
    assert active is not None
    assert active["session_id"] == s.session_id
    assert active["session_type"] == "research_new"


def test_record_preflight_missing_fields(isolated_sessions):
    from engine.sessions import lifecycle
    from engine.sessions.exceptions import PreflightIncompleteError
    from engine.sessions.schema import PreflightDigest, SessionType

    s = lifecycle.open_session(SessionType.research_new, title="x")
    bad = PreflightDigest(goal="too short")

    with pytest.raises(PreflightIncompleteError) as exc:
        lifecycle.record_preflight(s.session_id, bad)
    msg = str(exc.value)
    assert "cockpit_reviewed" in msg
    assert "graveyard_search_query" in msg


def test_record_preflight_transitions_to_in_flight(isolated_sessions):
    from engine.sessions import lifecycle
    from engine.sessions.schema import PreflightDigest, SessionType, SessionState

    s = lifecycle.open_session(SessionType.research_new, title="x")
    good = PreflightDigest(
        cockpit_reviewed=True,
        decay_alerts_count=0, dq_breaches_count=0,
        graveyard_search_query="EM FX momentum",
        graveyard_hits_count=2,
        library_overlap_checked=True,
        goal="Test 12m momentum in EM FX. Need orthogonality vs deployed carry leg.",
    )
    updated = lifecycle.record_preflight(s.session_id, good)
    assert updated.state == SessionState.in_flight
    assert updated.preflight_digest is not None
    assert updated.preflight_ts


def test_exploration_minimal_preflight(isolated_sessions):
    """Exploration only requires goal (≥10 chars). Test escape hatch."""
    from engine.sessions import lifecycle
    from engine.sessions.schema import PreflightDigest, SessionType, SessionState

    s = lifecycle.open_session(SessionType.exploration, title="brainstorm")
    digest = PreflightDigest(goal="thinking about something")
    updated = lifecycle.record_preflight(s.session_id, digest)
    assert updated.state == SessionState.in_flight


def test_research_new_close_fails_without_verdict(isolated_sessions):
    """Close research_new without emitting any verdict → ExitConditionsUnmetError."""
    from engine.sessions import lifecycle
    from engine.sessions.exceptions import ExitConditionsUnmetError
    from engine.sessions.schema import PreflightDigest, SessionType

    s = lifecycle.open_session(SessionType.research_new, title="x")
    lifecycle.record_preflight(s.session_id, PreflightDigest(
        cockpit_reviewed=True,
        graveyard_search_query="anything",
        library_overlap_checked=True,
        goal="A long enough goal to pass the 30-char check easily.",
    ))
    with pytest.raises(ExitConditionsUnmetError) as exc:
        lifecycle.close_session(s.session_id)
    assert "factor_verdict_filed" in str(exc.value)


def test_ops_close_succeeds_without_artifacts(isolated_sessions):
    """Ops session always closes cleanly (no required artifacts)."""
    from engine.sessions import lifecycle
    from engine.sessions.schema import PreflightDigest, SessionType, SessionState

    s = lifecycle.open_session(SessionType.ops, title="check monitoring")
    lifecycle.record_preflight(s.session_id, PreflightDigest(
        goal="Check daily monitoring",
    ))
    closed = lifecycle.close_session(s.session_id)
    assert closed.state == SessionState.closed
    assert closed.exit_report and closed.exit_report.exit_satisfied


def test_abandon_succeeds_anytime(isolated_sessions):
    """Abandon bypasses exit_check — for legitimately no-artifact sessions."""
    from engine.sessions import lifecycle
    from engine.sessions.schema import PreflightDigest, SessionType, SessionState

    s = lifecycle.open_session(SessionType.research_new, title="x")
    lifecycle.record_preflight(s.session_id, PreflightDigest(
        cockpit_reviewed=True, graveyard_search_query="anything",
        library_overlap_checked=True,
        goal="A long enough goal for preflight validation here.",
    ))
    abandoned = lifecycle.abandon_session(s.session_id, reason="false-alarm idea")
    assert abandoned.state == SessionState.abandoned
    assert abandoned.exit_report
    assert any("false-alarm" in m for m in abandoned.exit_report.missing_requirements)


def test_close_clears_active_pointer(isolated_sessions):
    from engine.sessions import lifecycle, store
    from engine.sessions.schema import PreflightDigest, SessionType

    s = lifecycle.open_session(SessionType.ops, title="x")
    assert store.get_active() is not None
    lifecycle.record_preflight(s.session_id, PreflightDigest(goal="ops monitoring check"))
    lifecycle.close_session(s.session_id)
    assert store.get_active() is None


def test_session_roundtrip_serialization(isolated_sessions):
    from engine.sessions.schema import (
        UserSession, SessionType, SessionState,
        PreflightDigest, SessionExitReport,
    )
    s = UserSession(
        session_id="abc-123",
        session_type=SessionType.research_new,
        state=SessionState.closed,
        opened_ts="2026-06-02T10:00:00Z",
        preflight_ts="2026-06-02T10:05:00Z",
        closed_ts="2026-06-02T14:30:00Z",
        preflight_digest=PreflightDigest(
            cockpit_reviewed=True, graveyard_search_query="x",
            library_overlap_checked=True, goal="test goal at least 30 chars yes",
        ),
        exit_report=SessionExitReport(
            exit_satisfied=True,
            missing_requirements=(),
            emitted_event_ids=("ev1", "ev2"),
            git_commits=("abc",),
            closed_ts="2026-06-02T14:30:00Z",
        ),
        title="x",
    )
    d = s.to_dict()
    s2 = UserSession.from_dict(d)
    assert s2 == s


def test_emit_auto_tags_with_active_session(isolated_sessions, tmp_path, monkeypatch):
    """When emit is called while a session is active, auto-tag the event."""
    # Also isolate the research_store
    rstore_dir = tmp_path / "research_store"
    rstore_dir.mkdir(parents=True, exist_ok=True)
    from engine.research_store import registry as rregistry, store as rstore
    monkeypatch.setattr(rregistry, "_STORE_DIR",     rstore_dir)
    monkeypatch.setattr(rregistry, "_SUBJECTS_PATH", rstore_dir / "subjects.yaml")
    monkeypatch.setattr(rregistry, "_ALIASES_PATH",  rstore_dir / "aliases.yaml")
    monkeypatch.setattr(rstore,    "_EVENTS_PATH",   rstore_dir / "events.jsonl")

    from engine.sessions import lifecycle
    from engine.sessions.schema import PreflightDigest, SessionType
    from engine.research_store import emit, registry as rreg
    from engine.research_store.schema import SubjectType as RSubjectType

    # Open + advance a session
    s = lifecycle.open_session(SessionType.research_new, title="x")
    lifecycle.record_preflight(s.session_id, PreflightDigest(
        cockpit_reviewed=True, graveyard_search_query="x",
        library_overlap_checked=True,
        goal="A goal sufficiently long to pass preflight checks ok.",
    ))

    # Register subject + emit
    rreg.register_subject("test_subj", subject_type=RSubjectType.factor)
    evidence = tmp_path / "ev.md"; evidence.write_text("x", encoding="utf-8")
    emit.factor_verdict(
        subject_id="test_subj", verdict="RED",
        metrics={}, artifacts={"evidence_doc": str(evidence)},
        summary="test verdict for session auto-tag test",
    )

    # Verify the emitted event has session tags
    events = rstore.all_events()
    assert len(events) == 1
    ev = events[0]
    assert f"session:{s.session_id}" in ev.tags
    assert "session_type:research_new" in ev.tags
    assert ev.session_id == s.session_id
