"""engine.sessions — user-initiated typed session protocol.

Public API:
    lifecycle.open_session(session_type, title)              → UserSession
    lifecycle.record_preflight(session_id, digest)           → UserSession
    lifecycle.close_session(session_id)                      → UserSession
    lifecycle.abandon_session(session_id, reason)            → UserSession
    store.get_session(session_id)                            → UserSession | None
    store.list_sessions(limit, state, session_type)          → list[UserSession]
    store.get_active()                                       → dict | None
    protocols.for_type(session_type)                         → protocol module
    schema.SessionType / SessionState / PreflightDigest / UserSession

See CLAUDE.md "Session Protocol Doctrine" for when each session_type applies.
"""
from engine.sessions import lifecycle, protocols, schema, store
from engine.sessions.exceptions import (
    SessionError,
    SessionNotFoundError,
    PreflightIncompleteError,
    ExitConditionsUnmetError,
    InvalidStateTransitionError,
)
from engine.sessions.schema import (
    SessionType, SessionState, PreflightDigest, UserSession, SessionExitReport,
)

__all__ = [
    "lifecycle", "protocols", "schema", "store",
    "SessionType", "SessionState", "PreflightDigest", "UserSession",
    "SessionExitReport",
    "SessionError", "SessionNotFoundError", "PreflightIncompleteError",
    "ExitConditionsUnmetError", "InvalidStateTransitionError",
]
