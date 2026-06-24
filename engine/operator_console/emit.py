"""Operator Console — typed emit helpers.

Mirrors engine.research_store.emit pattern: each helper validates its
pre-conditions, then writes via engine.operator_console.store.emit_event.

Callers should never call store.emit_event() directly — go through these
helpers so pre-conditions are uniformly enforced.
"""
from __future__ import annotations

from engine.operator_console.schema import OperatorEventType
from engine.operator_console.store import emit_event


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"[operator_console.emit] precondition failed: {msg}")


def session_started(*, session_id: str, session_type: str, actor_id: str,
                    cost_cap_usd: float, exit_conditions: list[str]) -> str:
    _require(bool(session_id), "session_id required")
    _require(bool(session_type), "session_type required")
    _require(cost_cap_usd > 0, "cost_cap_usd must be positive")
    return emit_event(
        event_type=OperatorEventType.SESSION_STARTED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "session_type":     session_type,
            "cost_cap_usd":     cost_cap_usd,
            "exit_conditions":  exit_conditions,
        },
    )


def session_closed(*, session_id: str, actor_id: str,
                   exit_satisfied: bool, cost_actual_usd: float,
                   summary: str = "") -> str:
    _require(bool(session_id), "session_id required")
    return emit_event(
        event_type=OperatorEventType.SESSION_CLOSED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "exit_satisfied":  exit_satisfied,
            "cost_actual_usd": cost_actual_usd,
            "summary":         summary[:400],
        },
    )


def session_abandoned(*, session_id: str, actor_id: str, reason: str) -> str:
    _require(bool(reason), "reason required for abandon (no shame, but be honest)")
    return emit_event(
        event_type=OperatorEventType.SESSION_ABANDONED,
        session_id=session_id,
        actor_id=actor_id,
        payload={"reason": reason[:400]},
    )


def station_triggered(*, session_id: str, actor_id: str,
                      job_id: str, station_id: str,
                      config: dict, estimated_cost_usd: float) -> str:
    return emit_event(
        event_type=OperatorEventType.STATION_TRIGGERED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "job_id":             job_id,
            "station_id":         station_id,
            "config":             config,
            "estimated_cost_usd": estimated_cost_usd,
        },
    )


def station_completed(*, session_id: str, actor_id: str,
                      job_id: str, station_id: str,
                      cost_actual_usd: float,
                      artifacts: dict[str, str]) -> str:
    return emit_event(
        event_type=OperatorEventType.STATION_COMPLETED,
        session_id=session_id,
        actor_id=actor_id,
        parent_event_ids=[],   # caller can pass triggered_event_id if tracking
        payload={
            "job_id":          job_id,
            "station_id":      station_id,
            "cost_actual_usd": cost_actual_usd,
            "artifacts":       artifacts,
        },
    )


def station_failed(*, session_id: str, actor_id: str,
                   job_id: str, station_id: str,
                   stage_failed: str, error: str) -> str:
    _require(bool(stage_failed), "stage_failed required for forensic")
    return emit_event(
        event_type=OperatorEventType.STATION_FAILED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "job_id":       job_id,
            "station_id":   station_id,
            "stage_failed": stage_failed,
            "error":        error[:1000],
        },
    )


def station_cancelled(*, session_id: str, actor_id: str,
                      job_id: str, station_id: str,
                      stage_at_cancel: str) -> str:
    return emit_event(
        event_type=OperatorEventType.STATION_CANCELLED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "job_id":          job_id,
            "station_id":      station_id,
            "stage_at_cancel": stage_at_cancel,
        },
    )


def station_halted_cost_cap(*, session_id: str, actor_id: str,
                            job_id: str, station_id: str,
                            estimated_usd: float, actual_usd: float,
                            cap_usd: float) -> str:
    return emit_event(
        event_type=OperatorEventType.STATION_HALTED_COST_CAP,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "job_id":        job_id,
            "station_id":    station_id,
            "estimated_usd": estimated_usd,
            "actual_usd":    actual_usd,
            "cap_usd":       cap_usd,
        },
    )


def operator_prediction_filed(*, session_id: str, actor_id: str,
                              spec_id: str,
                              p_green: float, p_marginal: float, p_red: float,
                              rationale: str = "",
                              risk_seen: str = "",
                              confidence_basis: str = "",
                              would_change_mind: str = "") -> str:
    """File a structured operator prediction BEFORE seeing LLM
    prediction (D1 anchoring discipline from Tetlock 2017).

    The 4 narrative fields capture reasoning that pure probability
    triples lose — Sherman Kent CIA tradecraft."""
    s = p_green + p_marginal + p_red
    _require(abs(s - 1.0) < 0.01, f"probabilities must sum to ~1.0, got {s}")
    _require(bool(spec_id), "spec_id required (which dispatch is this prediction about?)")
    return emit_event(
        event_type=OperatorEventType.OPERATOR_PREDICTION_FILED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "spec_id":            spec_id,
            "p_green":            p_green,
            "p_marginal":         p_marginal,
            "p_red":              p_red,
            "rationale":          rationale[:600],
            "risk_seen":          risk_seen[:600],
            "confidence_basis":   confidence_basis[:600],
            "would_change_mind":  would_change_mind[:600],
        },
    )


def doctrine_proposed(*, session_id: str, actor_id: str,
                      title: str, body: str,
                      doctrine_type: str,
                      why: str, how_to_apply: str,
                      related_memories: list[str] | None = None) -> str:
    _require(doctrine_type in {"feedback", "project", "user", "reference"},
             f"doctrine_type must be one of feedback/project/user/reference, got {doctrine_type}")
    _require(bool(why), "why field is mandatory")
    _require(bool(how_to_apply), "how_to_apply field is mandatory")
    return emit_event(
        event_type=OperatorEventType.DOCTRINE_PROPOSED,
        session_id=session_id,
        actor_id=actor_id,
        payload={
            "title":            title,
            "body":             body,
            "doctrine_type":    doctrine_type,
            "why":              why,
            "how_to_apply":     how_to_apply,
            "related_memories": related_memories or [],
        },
    )


def halt_forensic_filed(*, session_id: str, actor_id: str,
                        halt_event_id: str,
                        root_cause: str,
                        severity: str,
                        next_action: str,
                        notes: str = "") -> str:
    _require(severity in {"low", "medium", "high", "critical"},
             f"severity must be low/medium/high/critical, got {severity}")
    _require(bool(root_cause), "root_cause required")
    _require(bool(next_action), "next_action required")
    return emit_event(
        event_type=OperatorEventType.HALT_FORENSIC_FILED,
        session_id=session_id,
        actor_id=actor_id,
        parent_event_ids=[halt_event_id],
        payload={
            "halt_event_id": halt_event_id,
            "root_cause":    root_cause[:800],
            "severity":      severity,
            "next_action":   next_action[:400],
            "notes":         notes[:1000],
        },
    )
