"""engine.sessions.lifecycle — open / record_preflight / close / abandon API.

The public surface for orchestrating session state transitions. Enforces:
  - state machine validity (no skipping pending_preflight → closed)
  - preflight checker before in_flight
  - exit checker before closed
"""
from __future__ import annotations

import datetime as _dt
import uuid
from typing import Optional

from engine.sessions import store, protocols
from engine.sessions.exceptions import (
    SessionNotFoundError,
    PreflightIncompleteError,
    ExitConditionsUnmetError,
    InvalidStateTransitionError,
)
from engine.sessions.schema import (
    PreflightDigest, SessionExitReport, SessionState,
    SessionType, UserSession,
)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def open_session(session_type: SessionType, title: str) -> UserSession:
    """Open a new session in pending_preflight state. Sets active pointer."""
    session = UserSession(
        session_id=str(uuid.uuid4()),
        session_type=session_type,
        state=SessionState.pending_preflight,
        opened_ts=_utc_iso(),
        preflight_ts=None,
        closed_ts=None,
        preflight_digest=None,
        exit_report=None,
        title=title,
    )
    store.append_session_row(session)
    store.set_active(session.session_id, session_type.value)
    return session


def record_preflight(session_id: str, digest: PreflightDigest) -> UserSession:
    """Record preflight digest and transition pending_preflight → in_flight.
    Raises PreflightIncompleteError if required fields are not filled."""
    current = store.get_session(session_id)
    if current is None:
        raise SessionNotFoundError(session_id)
    if current.state != SessionState.pending_preflight:
        raise InvalidStateTransitionError(
            f"Session {session_id} in state {current.state.value}; can only record "
            f"preflight from pending_preflight."
        )

    proto = protocols.for_type(current.session_type)
    missing = proto.preflight_required_fields(digest)
    if missing:
        raise PreflightIncompleteError(session_id, missing)

    updated = UserSession(
        session_id=current.session_id,
        session_type=current.session_type,
        state=SessionState.in_flight,
        opened_ts=current.opened_ts,
        preflight_ts=_utc_iso(),
        closed_ts=None,
        preflight_digest=digest,
        exit_report=None,
        title=current.title,
        actor=current.actor,
    )
    store.append_session_row(updated)
    return updated


def _gather_session_events(session_id: str) -> list:
    """Pull all research_store events tagged with this session_id."""
    try:
        from engine.research_store import store as event_store
        all_events = event_store.all_events()
        return [
            e for e in all_events
            if any(t == f"session:{session_id}" for t in e.tags)
        ]
    except Exception:
        return []


def _gather_session_commits(session: UserSession) -> list[str]:
    """Pull git SHAs created during this session window (opened_ts → now)."""
    try:
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        # git log since opened_ts (ISO compatible with --since)
        out = subprocess.check_output(
            ["git", "log", "--since", session.opened_ts, "--pretty=%h"],
            cwd=repo_root, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        if not out:
            return []
        return out.splitlines()
    except Exception:
        return []


def close_session(session_id: str) -> UserSession:
    """Close a session — runs protocol exit_check. Raises ExitConditionsUnmetError
    if requirements not met (user must emit needed artifacts or abandon)."""
    current = store.get_session(session_id)
    if current is None:
        raise SessionNotFoundError(session_id)
    if current.state != SessionState.in_flight:
        raise InvalidStateTransitionError(
            f"Session {session_id} in state {current.state.value}; can only close "
            f"from in_flight."
        )

    events = _gather_session_events(session_id)
    commits = _gather_session_commits(current)

    proto = protocols.for_type(current.session_type)
    satisfied, missing = proto.exit_check(events, commits)
    if not satisfied:
        raise ExitConditionsUnmetError(
            session_id, current.session_type.value, missing,
        )

    report = SessionExitReport(
        exit_satisfied=True,
        missing_requirements=(),
        emitted_event_ids=tuple(e.event_id for e in events),
        git_commits=tuple(commits),
        closed_ts=_utc_iso(),
    )
    updated = UserSession(
        session_id=current.session_id,
        session_type=current.session_type,
        state=SessionState.closed,
        opened_ts=current.opened_ts,
        preflight_ts=current.preflight_ts,
        closed_ts=report.closed_ts,
        preflight_digest=current.preflight_digest,
        exit_report=report,
        title=current.title,
        actor=current.actor,
    )
    store.append_session_row(updated)
    # Clear active pointer only if this is the active one
    active = store.get_active()
    if active and active.get("session_id") == session_id:
        store.clear_active()
    return updated


def abandon_session(session_id: str, reason: str = "") -> UserSession:
    """Mark a session abandoned. Bypasses exit_check — for sessions that
    legitimately produce no artifacts (cancelled exploration, false-alarm audit).
    Records reason in exit_report.missing_requirements for audit."""
    current = store.get_session(session_id)
    if current is None:
        raise SessionNotFoundError(session_id)
    if current.state == SessionState.closed:
        raise InvalidStateTransitionError(
            f"Session {session_id} already closed; cannot abandon."
        )

    events = _gather_session_events(session_id)
    commits = _gather_session_commits(current)

    report = SessionExitReport(
        exit_satisfied=False,
        missing_requirements=(f"abandoned: {reason}",) if reason else ("abandoned",),
        emitted_event_ids=tuple(e.event_id for e in events),
        git_commits=tuple(commits),
        closed_ts=_utc_iso(),
    )
    updated = UserSession(
        session_id=current.session_id,
        session_type=current.session_type,
        state=SessionState.abandoned,
        opened_ts=current.opened_ts,
        preflight_ts=current.preflight_ts,
        closed_ts=report.closed_ts,
        preflight_digest=current.preflight_digest,
        exit_report=report,
        title=current.title,
        actor=current.actor,
    )
    store.append_session_row(updated)
    active = store.get_active()
    if active and active.get("session_id") == session_id:
        store.clear_active()
    return updated
