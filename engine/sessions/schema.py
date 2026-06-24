"""engine.sessions.schema — typed contract for user-initiated research sessions.

A session is a first-class workflow object — same tier as a git commit or a
Linear ticket. It has a state machine (pending_preflight → in_flight → closed),
typed pre-flight requirements, and exit-condition checkers that gate close().

5 session types (post-2026-06-02 audit, 5th type added to prevent
over-bureaucratization per [[feedback-no-fear-of-rework-only-unusable]]):

  research_new   — test a new factor with strict gate
  audit          — investigate bug / suspicious number
  ops            — monitoring / alert response
  doctrine       — lock a lesson / amend memory
  exploration    — open-ended thinking; NO exit enforcement (escape hatch)

User-initiated. Cron-initiated runs use engine.cron_runs (separate schema)
and emit into the shared event store with kind:cron_run tag.
"""
from __future__ import annotations

import dataclasses as _dc
from enum import Enum
from typing import Optional


SCHEMA_VERSION = 1


class SessionType(str, Enum):
    research_new = "research_new"
    audit        = "audit"
    ops          = "ops"
    doctrine     = "doctrine"
    exploration  = "exploration"   # escape hatch — no exit emit required


class SessionState(str, Enum):
    pending_preflight = "pending_preflight"   # UI must fill preflight before transition
    in_flight         = "in_flight"           # Claude is working, events accumulating
    closed            = "closed"              # exit verified or explicitly bypassed
    abandoned         = "abandoned"           # user gave up; no exit verification


@_dc.dataclass(frozen=True)
class PreflightDigest:
    """The pre-flight checklist UI fills in before transitioning a session
    from pending_preflight → in_flight.

    NOT all fields apply to every session type. Per-type protocol checker
    enforces which are required (see engine.sessions.protocols).

    For exploration sessions, only `goal` is required (rest may be empty).
    """
    # State snapshot at session start (auto-pulled by UI)
    cockpit_reviewed:        bool = False
    decay_alerts_count:      int = 0
    dq_breaches_count:       int = 0

    # User input (required for most types)
    graveyard_search_query:  str = ""
    graveyard_hits_count:    int = 0
    library_overlap_checked: bool = False

    # Always required (sufficient even for exploration)
    goal:                    str = ""

    # Free-form notes the user wants to capture at session start
    notes:                   str = ""

    def to_dict(self) -> dict:
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreflightDigest":
        fields = {f.name for f in _dc.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@_dc.dataclass(frozen=True)
class SessionExitReport:
    """Output of close() — records what was verified vs missing."""
    exit_satisfied:        bool
    missing_requirements:  tuple[str, ...]
    emitted_event_ids:     tuple[str, ...]
    git_commits:           tuple[str, ...]
    closed_ts:             str

    def to_dict(self) -> dict:
        d = _dc.asdict(self)
        d["missing_requirements"] = list(self.missing_requirements)
        d["emitted_event_ids"]    = list(self.emitted_event_ids)
        d["git_commits"]          = list(self.git_commits)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionExitReport":
        return cls(
            exit_satisfied        = bool(d.get("exit_satisfied", False)),
            missing_requirements  = tuple(d.get("missing_requirements") or ()),
            emitted_event_ids     = tuple(d.get("emitted_event_ids") or ()),
            git_commits           = tuple(d.get("git_commits") or ()),
            closed_ts             = d.get("closed_ts", ""),
        )


@_dc.dataclass(frozen=True)
class UserSession:
    """The canonical session record.

    Identity:
        session_id     — UUID
        session_type   — one of 5 types
        state          — current state in lifecycle

    Lifecycle timestamps:
        opened_ts      — when session was opened (state created)
        preflight_ts   — when preflight_digest was recorded (state → in_flight)
        closed_ts      — when session was closed

    Pre-flight (filled by UI before in_flight):
        preflight_digest — typed checklist; per-type checker validates

    Exit (filled by close() and protocol checker):
        exit_report    — was exit verified, what was missing, linked artifacts
    """
    session_id:       str
    session_type:     SessionType
    state:            SessionState
    opened_ts:        str
    preflight_ts:     Optional[str]
    closed_ts:        Optional[str]

    preflight_digest: Optional[PreflightDigest]
    exit_report:      Optional[SessionExitReport]

    # User-facing labels
    title:            str            # short label (UI shown)
    actor:            str = "user"   # always "user" for UserSession; cron has separate schema

    schema_version:   int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "session_type":     self.session_type.value,
            "state":            self.state.value,
            "opened_ts":        self.opened_ts,
            "preflight_ts":     self.preflight_ts,
            "closed_ts":        self.closed_ts,
            "preflight_digest": self.preflight_digest.to_dict() if self.preflight_digest else None,
            "exit_report":      self.exit_report.to_dict() if self.exit_report else None,
            "title":            self.title,
            "actor":            self.actor,
            "schema_version":   self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserSession":
        pf = d.get("preflight_digest")
        er = d.get("exit_report")
        return cls(
            session_id       = d["session_id"],
            session_type     = SessionType(d["session_type"]),
            state            = SessionState(d["state"]),
            opened_ts        = d["opened_ts"],
            preflight_ts     = d.get("preflight_ts"),
            closed_ts        = d.get("closed_ts"),
            preflight_digest = PreflightDigest(**pf) if pf else None,
            exit_report      = SessionExitReport.from_dict(er) if er else None,
            title            = d.get("title", ""),
            actor            = d.get("actor", "user"),
            schema_version   = int(d.get("schema_version", 1)),
        )
