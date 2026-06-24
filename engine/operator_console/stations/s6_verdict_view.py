"""S6 — Verdict View (read-only drill of verdict + autopsy + belief delta).

Bundles a verdict event with its prediction-side counterpart (the
air-gapped prediction filed BEFORE dispatch) and the autopsy that
joined them. Lets the UI render the full Belief Layer lineage for a
single verdict in one click.

Read-only; no LLM; no compute; $0; ~1 minute. Per design doc this is
the terminal station of an E2E demo loop: S1 paper → S3 spec → S4
verdict → S6 view.

Design reference: docs/architecture/operator_console.md §5 (S6 spec).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

from engine.operator_console.pipeline_station import (
    PipelineStation,
    SSEEmitter,
    Session,
)
from engine.operator_console.schema import (
    CancellationToken,
    CostEstimate,
    DataTier,
    NextStationHint,
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
    SessionType,
    StationResult,
    StationSpec,
)
from engine.operator_console import emit as opcon_emit
from engine.operator_console import registry


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVENTS_PATH      = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
_AUTOPSIES_PATH   = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"
_PREDICTIONS_PATH = _REPO_ROOT / "data" / "research" / "predictions.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _find_verdict_event(verdict_event_id: str) -> dict | None:
    """Return the typed event row for this verdict_event_id, or None."""
    if not _EVENTS_PATH.is_file():
        return None
    for line in _EVENTS_PATH.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        if row.get("event_id") == verdict_event_id:
            return row
    return None


def _find_autopsy_by_verdict(verdict_event_id: str) -> dict | None:
    """Return the autopsy row that joined this verdict, or None."""
    if not _AUTOPSIES_PATH.is_file():
        return None
    for line in _AUTOPSIES_PATH.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        if row.get("verdict_event_id") == verdict_event_id:
            return row
    return None


def _find_prediction(prediction_id: str) -> dict | None:
    if not prediction_id or not _PREDICTIONS_PATH.is_file():
        return None
    for line in _PREDICTIONS_PATH.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        if row.get("prediction_id") == prediction_id:
            return row
    return None


class VerdictView(PipelineStation):
    """S6 — Read-only drill into a verdict's full Belief Layer lineage."""

    STATION_SPEC = StationSpec(
        station_id              = "S6_verdict_view",
        title                   = "Verdict View",
        description             = (
            "Drill into a verdict's full Belief Layer lineage: the "
            "air-gapped prediction filed before dispatch, the verdict "
            "itself, the autopsy that joined them, and the Brier "
            "component contributed to the family track record."
        ),
        data_tier               = DataTier.SNAPSHOT_DATA,
        # Available from any session type — it's a pure read-only drill.
        requires_session_types  = {
            SessionType.RESEARCH_NEW,
            SessionType.AUDIT,
            SessionType.OPS,
            SessionType.EXPLORATION,
            SessionType.DOCTRINE,
        },
        estimated_minutes       = 1,
        estimated_cost_usd      = 0.0,
        icon                    = "Eye",
        title_key               = "console.station.s6.title",
        description_key         = "console.station.s6.description",
    )

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck("session_active", PreflightStatus.RED,
                                         "No active session."))
        else:
            checks.append(PreflightCheck("session_active", PreflightStatus.GREEN,
                                         f"Session {session.session_id} ready."))

        verdict_event_id = str((config or {}).get("verdict_event_id", "")).strip()
        if not verdict_event_id:
            checks.append(PreflightCheck("verdict_event_id_provided", PreflightStatus.RED,
                                         "Provide verdict_event_id."))
        else:
            ev = _find_verdict_event(verdict_event_id)
            if ev is None:
                checks.append(PreflightCheck(
                    "verdict_event_resolvable", PreflightStatus.RED,
                    f"verdict_event_id '{verdict_event_id}' not in events.jsonl.",
                ))
            else:
                # Yellow if not a factor_verdict event (still readable but unusual)
                if ev.get("event_type") != "factor_verdict_filed":
                    checks.append(PreflightCheck(
                        "verdict_event_resolvable", PreflightStatus.YELLOW,
                        f"Found event but type is '{ev.get('event_type')}', not factor_verdict_filed.",
                    ))
                else:
                    checks.append(PreflightCheck(
                        "verdict_event_resolvable", PreflightStatus.GREEN,
                        f"Found factor_verdict_filed event for subject_id='{ev.get('subject_id', '?')}'.",
                    ))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict) -> CostEstimate:
        return CostEstimate(llm_cost_usd_est=0.0, confidence="exact")

    def render_config_form(self) -> dict:
        return {
            "type": "object",
            "title": "Verdict View input",
            "description": (
                "Drill into a single verdict's full Belief Layer lineage. "
                "Verdict event IDs come from S4 FORWARD Dispatch results "
                "(in artifacts.dispatch_event_id) or from "
                "/research/lessons listings."
            ),
            "properties": {
                "verdict_event_id": {
                    "type": "string",
                    "title": "verdict_event_id (event_id from events.jsonl)",
                    "description": "Usually a UUID. S4 dispatch results carry this as artifacts.dispatch_event_id.",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "e.g. 86b4ebac-ef9d-4c86-9b42-19d3d64d0c64",
                },
            },
            "required": ["verdict_event_id"],
        }

    async def execute(
        self,
        session: Session,
        config: dict,
        emitter: SSEEmitter,
        cancellation: CancellationToken,
    ) -> StationResult:
        started_ts = _utc_iso()
        actor_id = getattr(session, "actor_id", "principal")
        session_id = getattr(session, "session_id", "")
        verdict_event_id = str((config or {}).get("verdict_event_id", "")).strip()

        # ── Stage 1: load verdict event ──────────────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "load_verdict")
        emitter.stage_started("load_verdict", expected_seconds=1)
        ev = _find_verdict_event(verdict_event_id)
        if ev is None:
            emitter.stage_failed("load_verdict", "not found in events.jsonl")
            return self._failed(session, started_ts, "load_verdict",
                                f"verdict_event_id '{verdict_event_id}' not in events.jsonl")
        subject_id = ev.get("subject_id", "")
        payload = ev.get("payload", {}) or {}
        verdict = payload.get("verdict") or "(none in payload)"
        emitter.stage_completed("load_verdict", {
            "subject_id":  subject_id,
            "event_type":  ev.get("event_type", ""),
            "verdict":     verdict,
            "ts":          ev.get("ts", ""),
        })

        # ── Stage 2: join autopsy ────────────────────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "join_autopsy")
        emitter.stage_started("join_autopsy", expected_seconds=1)
        autopsy = _find_autopsy_by_verdict(verdict_event_id)
        if autopsy is None:
            emitter.stage_completed("join_autopsy", {
                "joined": False,
                "note": "No autopsy yet — autopsies are joined daily by belief refresh cron; if this verdict is fresh, autopsy may appear within 24h.",
            })
            prediction_id = ""
        else:
            prediction_id = autopsy.get("prediction_id", "")
            emitter.stage_completed("join_autopsy", {
                "joined":          True,
                "autopsy_id":      autopsy.get("autopsy_id", ""),
                "prediction_id":   prediction_id,
                "actual_verdict":  autopsy.get("actual_verdict", ""),
                "brier_component": autopsy.get("brier_component", 0),
                "surprise_direction": autopsy.get("surprise_direction", ""),
            })

        # ── Stage 3: load prediction (air-gapped) ────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "load_prediction")
        emitter.stage_started("load_prediction", expected_seconds=1)
        prediction = _find_prediction(prediction_id) if prediction_id else None
        if prediction is None:
            emitter.stage_completed("load_prediction", {
                "found": False,
                "note": "No prediction row matched; either prediction_id missing from autopsy or prediction not yet committed.",
            })
        else:
            emitter.stage_completed("load_prediction", {
                "found":            True,
                "prediction_id":    prediction.get("prediction_id", ""),
                "predicted_dist":   prediction.get("predicted_verdict_dist", {}),
                "prediction_basis": str(prediction.get("prediction_basis", ""))[:200],
                "session_id_at_predict": prediction.get("session_id", ""),
            })

        # ── Persist station_completed ─────────────────────────────
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = 0.0,
                artifacts       = {
                    "verdict_event_id": verdict_event_id,
                    "subject_id":       subject_id,
                    "verdict":          str(verdict),
                    "autopsy_joined":   "yes" if autopsy else "no",
                    "prediction_found": "yes" if prediction else "no",
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")

        # ── Result with full lineage bundle ───────────────────────
        completed_ts = _utc_iso()
        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            started_ts      = started_ts,
            completed_ts    = completed_ts,
            success         = True,
            artifacts       = {
                "verdict_event_id":     verdict_event_id,
                "subject_id":           subject_id,
                "verdict":              str(verdict),
                "event_ts":             ev.get("ts", ""),
                "autopsy_id":           (autopsy or {}).get("autopsy_id", ""),
                "brier_component":      str((autopsy or {}).get("brier_component", "")),
                "actual_verdict":       (autopsy or {}).get("actual_verdict", ""),
                "surprise_direction":   (autopsy or {}).get("surprise_direction", ""),
                "prediction_id":        prediction_id,
                "predicted_dist":       json.dumps(
                    (prediction or {}).get("predicted_verdict_dist", {})
                )[:300],
                "lineage_complete":     "yes" if (autopsy and prediction) else "partial",
            },
            events_emitted  = [],
            next_stations   = self._lineage_hints(ev, autopsy),
            cost_actual_usd = 0.0,
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        # Without re-loading, suggest the family page link
        return []

    @staticmethod
    def _lineage_hints(verdict_event: dict, autopsy: dict | None) -> list[NextStationHint]:
        hints: list[NextStationHint] = []
        payload = verdict_event.get("payload", {}) or {}
        verdict = payload.get("verdict", "")
        if verdict == "GREEN":
            hints.append(NextStationHint(
                station_id        = "S7_promote",
                label             = "PROMOTE this GREEN verdict to deployment (9-gate)",
                suggested_config  = {"verdict_event_id": verdict_event.get("event_id", "")},
            ))
        return hints

    def _cancelled(self, session: Session, started_ts: str, stage: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"Cancelled at stage '{stage}'.",
        )

    def _failed(self, session: Session, started_ts: str, stage: str, err: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"Stage '{stage}' failed: {err}",
        )


registry.register(VerdictView)
