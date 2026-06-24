"""engine.sessions.exceptions — typed errors for session lifecycle."""
from __future__ import annotations


class SessionError(Exception):
    """Base for all session errors."""


class SessionNotFoundError(SessionError):
    """Lookup by session_id returned nothing."""


class PreflightIncompleteError(SessionError):
    """Tried to transition to in_flight without all required preflight fields."""
    def __init__(self, session_id: str, missing: list[str]):
        self.session_id = session_id
        self.missing = missing
        super().__init__(
            f"Session {session_id} preflight incomplete. Missing:\n  - "
            + "\n  - ".join(missing)
        )


class ExitConditionsUnmetError(SessionError):
    """Tried to close a session but exit_check failed."""
    def __init__(self, session_id: str, session_type: str, missing: list[str]):
        self.session_id = session_id
        self.session_type = session_type
        self.missing = missing
        super().__init__(
            f"Session {session_id} ({session_type}) cannot close — exit conditions unmet.\n  - "
            + "\n  - ".join(missing)
            + "\nOptions: emit the required artifacts, OR abandon the session."
        )


class InvalidStateTransitionError(SessionError):
    """Attempted a lifecycle transition incompatible with current state."""
