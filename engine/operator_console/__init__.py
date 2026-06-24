"""Operator Console — UI-triggered pipeline orchestration.

Foundation for the 9-station Pipeline Station pattern documented in
docs/architecture/operator_console.md. See that doc for the full
design rationale; this package is the implementation.

Public surface:
    from engine.operator_console import (
        PipelineStation, StationSpec, StationResult,
        DataTier, JobState, SessionType,
        emit, store, cost_ledger,
    )
"""
from engine.operator_console.schema import (
    DataTier,
    JobState,
    SessionType,
    StationSpec,
    StationResult,
    PreflightCheck,
    PreflightResult,
    CostEstimate,
    CancellationToken,
)
from engine.operator_console.pipeline_station import PipelineStation
from engine.operator_console import emit, store, cost_ledger

__all__ = [
    "DataTier",
    "JobState",
    "SessionType",
    "StationSpec",
    "StationResult",
    "PreflightCheck",
    "PreflightResult",
    "CostEstimate",
    "CancellationToken",
    "PipelineStation",
    "emit",
    "store",
    "cost_ledger",
]
