"""engine.sessions.store — append-only session log + active-singleton pointer.

Two files on disk:
  data/sessions/sessions.jsonl   — append-only history (one row per state transition)
  data/sessions/_active.json     — pointer to the active session_id (single user)

The pointer file is read by engine.research_store.emit to auto-tag events
with the active session_id, without Claude having to know it.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from engine.sessions.schema import UserSession

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SESSIONS_PATH = _REPO_ROOT / "data" / "sessions" / "sessions.jsonl"
_ACTIVE_PATH   = _REPO_ROOT / "data" / "sessions" / "_active.json"

_LOCK = threading.Lock()


def _ensure_dir() -> None:
    _SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_session_row(session: UserSession) -> None:
    """Append a session state to the log. Each transition writes a fresh row;
    the latest row per session_id reflects current state."""
    _ensure_dir()
    with _LOCK:
        with _SESSIONS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(session.to_dict(), ensure_ascii=False) + "\n")


def _read_all_rows() -> list[dict]:
    if not _SESSIONS_PATH.is_file():
        return []
    out = []
    with _SESSIONS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("sessions.jsonl line malformed; skipping")
    return out


def get_session(session_id: str) -> Optional[UserSession]:
    """Return the latest state for a session_id, or None if not found."""
    rows = [r for r in _read_all_rows() if r.get("session_id") == session_id]
    if not rows:
        return None
    return UserSession.from_dict(rows[-1])


def list_sessions(limit: int = 100, state: Optional[str] = None,
                  session_type: Optional[str] = None) -> list[UserSession]:
    """List unique sessions (latest state per session_id) newest first."""
    rows = _read_all_rows()
    # Reduce to latest per session_id
    latest: dict[str, dict] = {}
    for r in rows:
        sid = r.get("session_id")
        if sid:
            latest[sid] = r
    sessions = [UserSession.from_dict(r) for r in latest.values()]
    if state:
        sessions = [s for s in sessions if s.state.value == state]
    if session_type:
        sessions = [s for s in sessions if s.session_type.value == session_type]
    sessions.sort(key=lambda s: s.opened_ts, reverse=True)
    return sessions[:limit]


# ── Active session pointer (singleton) ──────────────────────────


def set_active(session_id: str, session_type: str) -> None:
    """Write the active session pointer file. Idempotent — overwrite OK."""
    _ensure_dir()
    with _LOCK:
        with _ACTIVE_PATH.open("w", encoding="utf-8") as fh:
            json.dump({"session_id": session_id, "session_type": session_type}, fh)


def clear_active() -> None:
    """Remove the active session pointer."""
    with _LOCK:
        if _ACTIVE_PATH.is_file():
            _ACTIVE_PATH.unlink()


def get_active() -> Optional[dict]:
    """Return {session_id, session_type} of the active session, or None."""
    if not _ACTIVE_PATH.is_file():
        return None
    try:
        with _ACTIVE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.exception("failed reading _active.json")
        return None
