"""Operator Console — typed schema.

Implements D5 (multi-user-ready schema with actor_id) and D3 (data
tier tagging) from docs/architecture/operator_console.md.

These dataclasses are the contract between backend stations, the
job store, the API, and the frontend. Changes here ripple through
all layers — bump schema_version when modifying.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar


SCHEMA_VERSION = "1.0.0"


# ── Enums ────────────────────────────────────────────────────────


class DataTier(str, Enum):
    """D3 — every station declares its data dependency tier so the UI
    can warn users about what they can / can't run without their own
    WRDS subscription."""
    USER_DATA      = "user_data"       # works on any install (user-supplied input)
    DEMO_FIXTURE   = "demo_fixture"    # works on any install (bundled samples)
    SNAPSHOT_DATA  = "snapshot_data"   # works on any install (read-only project data)
    WRDS_REQUIRED  = "wrds_required"   # needs paid WRDS subscription + IP allowlist


class JobState(str, Enum):
    """Lifecycle state of a single station-execution job."""
    QUEUED              = "queued"
    RUNNING             = "running"
    COMPLETED           = "completed"
    FAILED              = "failed"
    CANCELLED           = "cancelled"
    HALTED_COST_CAP     = "halted_cost_cap"          # see R6 in design doc
    RECOVERED_UNKNOWN   = "recovered_unknown"        # see "server restart mid-job" risk


class SessionType(str, Enum):
    """Per CLAUDE.md Session Protocol Doctrine — 5 typed sessions.
    Each station declares which types it can run within."""
    RESEARCH_NEW = "research_new"
    AUDIT        = "audit"
    OPS          = "ops"
    DOCTRINE     = "doctrine"
    EXPLORATION  = "exploration"


class PreflightStatus(str, Enum):
    """Red/yellow/green for individual preflight checks."""
    GREEN  = "green"   # passes
    YELLOW = "yellow"  # warning (e.g. RED-lesson family warning per IR5) but not blocking
    RED    = "red"     # blocking; trigger rejected


# ── Core dataclasses ─────────────────────────────────────────────


@dataclass(frozen=True)
class StationSpec:
    """Static declaration of a station's metadata. Backend stations
    expose this as a ClassVar; the API surfaces it via /stations so
    the UI can render a launchpad without per-station code."""

    station_id:               str
    title:                    str
    description:              str
    data_tier:                DataTier
    requires_session_types:   set[SessionType]
    estimated_minutes:        int
    estimated_cost_usd:       float
    icon:                     str = "Layers"   # lucide-react icon name

    # i18n keys per IR1 (cross-cutting integration requirement).
    # If None, frontend falls back to title / description in English.
    title_key:                str | None = None
    description_key:          str | None = None

    # Capital-decision doctrine enforcement (per CLAUDE.md). A station
    # that "mutates capital" is one whose successful execution changes
    # which sleeves are deployed or how much they're funded. By doctrine,
    # such mutations MUST route through /approvals (human-in-the-loop) —
    # never write the deployed config YAML directly.
    #
    # Stations declare intent via this flag; registry.register() lints
    # the source file at import time:
    #   mutates_capital=False → source MUST NOT write deployed YAML
    #   mutates_capital=True  → source MUST reference _proposals.jsonl
    #                           (positive proof it routes to /approvals)
    # Bypass attempt raises CapitalDoctrineViolation at register time.
    mutates_capital:          bool = False


@dataclass(frozen=True)
class PreflightCheck:
    """Result of a single named preflight gate."""
    name:    str
    status:  PreflightStatus
    detail:  str = ""


@dataclass(frozen=True)
class PreflightResult:
    """Aggregate preflight: union of all checks plus overall blocker
    status. Trigger is rejected if any RED check exists."""

    checks:    list[PreflightCheck]
    can_trigger: bool

    @classmethod
    def from_checks(cls, checks: list[PreflightCheck]) -> "PreflightResult":
        has_red = any(c.status == PreflightStatus.RED for c in checks)
        return cls(checks=checks, can_trigger=not has_red)


@dataclass(frozen=True)
class CostEstimate:
    """D4 — pre-trigger cost preview shown to the user."""

    llm_cost_usd_est:  float
    compute_cost_usd_est: float = 0.0   # rare; placeholder for future
    confidence:        str = "approximate"  # approximate / exact

    @property
    def total_usd(self) -> float:
        return self.llm_cost_usd_est + self.compute_cost_usd_est


@dataclass(frozen=True)
class StationResult:
    """Standardized return shape from any station execute(). The
    `next_stations` field drives the lineage UI — what the user
    can do next given this result."""

    job_id:           str
    station_id:       str
    session_id:       str
    actor_id:         str                    # D5
    started_ts:       str                    # ISO-8601 UTC
    completed_ts:     str
    success:          bool
    artifacts:        dict[str, str] = field(default_factory=dict)  # name → path
    events_emitted:   list[str]     = field(default_factory=list)  # event_ids
    next_stations:    list["NextStationHint"] = field(default_factory=list)
    cost_actual_usd:  float = 0.0
    error_message:    str = ""               # populated when success=False


@dataclass(frozen=True)
class NextStationHint:
    """One option the user can take after this station completes.
    Includes pre-filled config to skip retyping."""
    station_id:        str
    label:             str           # "Synthesize hypothesis from this paper"
    suggested_config:  dict[str, Any] = field(default_factory=dict)


@dataclass
class CancellationToken:
    """Mutable; checked at stage boundaries per R3 (cannot cancel
    mid-stage). Stations must poll .cancelled at each stage start."""

    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ── Event store schema (mirrors research_store pattern) ──────────


class OperatorEventType(str, Enum):
    """The typed events emitted to data/operator_console/events.jsonl.
    Mirrors the research_store doctrine: one type per meaningful
    state change; immutable; queryable."""

    SESSION_STARTED                 = "session_started"
    SESSION_CLOSED                  = "session_closed"
    SESSION_ABANDONED               = "session_abandoned"
    STATION_TRIGGERED               = "station_triggered"
    STATION_COMPLETED               = "station_completed"
    STATION_FAILED                  = "station_failed"
    STATION_CANCELLED               = "station_cancelled"
    STATION_HALTED_COST_CAP         = "station_halted_cost_cap"
    OPERATOR_PREDICTION_FILED       = "operator_prediction_filed"
    DOCTRINE_PROPOSED               = "doctrine_proposed"
    HALT_FORENSIC_FILED             = "halt_forensic_filed"
