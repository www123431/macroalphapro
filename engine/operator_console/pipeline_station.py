"""PipelineStation — abstract base class for all 9 operator-console
stations (S1..S9) per docs/architecture/operator_console.md Section 4.

A station implements 5 elements:
    preflight()       — pre-conditions (D3 data tier check, D4 cost
                        cap check, IR5 RED-lesson warning, etc.)
    estimate_cost()   — cost preview shown before user clicks trigger
    render_config_form() — JSON Schema; UI renders generic form
    execute()         — the actual work; streams progress via emitter
    result_lineage()  — next-station hints after success

Subclasses must:
    - Declare STATION_SPEC (a static StationSpec instance)
    - Implement the 4 abstract methods
    - Honor CancellationToken at stage boundaries (R3 honesty)
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

from engine.operator_console.schema import (
    CancellationToken,
    CostEstimate,
    PreflightResult,
    StationResult,
    StationSpec,
)


class SSEEmitter(Protocol):
    """Protocol for progress streaming. Foundation will supply a
    concrete impl; stations only need this protocol surface.

    Each method maps to a named SSE event consumed by the frontend
    StationProgressStream component (see design doc Section D2)."""

    def stage_started(self, stage: str, expected_seconds: int = 0) -> None: ...
    def stage_progress(self, stage: str, pct: int, current: str = "") -> None: ...
    def stage_completed(self, stage: str, result: dict[str, Any]) -> None: ...
    def stage_failed(self, stage: str, error: str) -> None: ...
    def log_line(self, line: str) -> None: ...


class Session(Protocol):
    """Minimal session view a station needs. Foundation supplies the
    concrete session object via the trigger API."""

    session_id:  str
    session_type: str
    actor_id:    str


class PipelineStation(ABC):
    """Universal contract every station implements. The Foundation +
    API layer treats all stations through this interface; per-station
    code is purely the work."""

    # Subclasses MUST override this with a static StationSpec instance.
    # Enforced at class-creation time by __init_subclass__ — forgetting
    # raises TypeError on import, not on first .STATION_SPEC access.
    STATION_SPEC: ClassVar[StationSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract intermediate bases are allowed to omit STATION_SPEC;
        # only concrete (instantiable) leaves must declare one.
        if inspect.isabstract(cls):
            return
        if "STATION_SPEC" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare a class-level STATION_SPEC: "
                f"ClassVar[StationSpec] = StationSpec(...). See existing "
                f"stations in engine/operator_console/stations/ for examples.")
        if not isinstance(cls.__dict__["STATION_SPEC"], StationSpec):
            raise TypeError(
                f"{cls.__name__}.STATION_SPEC must be a StationSpec instance, "
                f"got {type(cls.__dict__['STATION_SPEC']).__name__}.")

    # ── Required abstract methods ────────────────────────────────

    @abstractmethod
    def preflight(self, session: Session, config: dict) -> PreflightResult:
        """Validate that this station can run right now.

        Standard checks every station should include:
          - Session is active and session.session_type is in
            STATION_SPEC.requires_session_types
          - Session cost ledger has room for estimate_cost(config)
          - D3 data tier dependencies satisfied
          - IR5 RED-lesson family warning (yellow status; not blocking)

        Return PreflightResult.from_checks([...]); helper combines
        the list of checks into the can_trigger boolean."""

    @abstractmethod
    def estimate_cost(self, config: dict) -> CostEstimate:
        """Predict LLM + compute cost for the trigger. Used by the
        UI to render a 'This action will cost ~$X.XX' preview before
        the user clicks trigger.

        Be honest: use worst-case (upper-bound) estimate. Underestimating
        triggers the cost-cap halt mid-execution per R6 in design doc."""

    @abstractmethod
    def render_config_form(self) -> dict:
        """Return a JSON Schema describing the configuration fields
        for this station. The frontend renders a generic form from
        this spec — no per-station React component required.

        Standard JSON Schema; use 'title', 'description', 'enum',
        'minimum', 'maximum', etc. UI-hint extensions namespaced
        under 'x-ui-' (e.g. 'x-ui-widget': 'text-area' / 'slider')."""

    @abstractmethod
    async def execute(
        self,
        session: Session,
        config: dict,
        emitter: SSEEmitter,
        cancellation: CancellationToken,
    ) -> StationResult:
        """The actual work. Run the pipeline, stream progress via
        emitter, persist artifacts, emit typed events, and return
        a StationResult.

        Cancellation contract per R3:
          - Check `cancellation.cancelled` at the START of each stage
          - If cancelled, emit stage_failed (or log_line) and return
            an unsuccessful StationResult with success=False
          - DO NOT attempt to abort an in-flight stage (e.g. don't
            kill a Bootstrap CI mid-resample); finish the current
            stage then check.

        Cost-cap contract per R6:
          - At each stage boundary, query the cost ledger
          - If accumulated cost > session_cap * 1.2, halt with
            success=False and JobState.HALTED_COST_CAP
        """

    @abstractmethod
    def result_lineage(self, result: StationResult) -> list[Any]:
        """Given a successful result, return the list of NextStationHint
        suggestions the UI should surface. Empty list = terminal node
        in the workflow.

        Example: S1 PaperIngest result.lineage = [
            NextStationHint(station_id='S2_hypothesis_synthesize',
                            label='Synthesize hypothesis from this paper',
                            suggested_config={'paper_ids': [paper_id]})
        ]"""

    # ── Convenience helpers (concrete; available to subclasses) ──

    @classmethod
    def spec(cls) -> StationSpec:
        """Public accessor for the static spec."""
        return cls.STATION_SPEC

    @classmethod
    def station_id(cls) -> str:
        return cls.STATION_SPEC.station_id
