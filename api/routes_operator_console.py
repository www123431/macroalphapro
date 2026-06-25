"""Operator Console — API routes.

Endpoints (per docs/architecture/operator_console.md §3):
    GET    /api/console/stations                 station registry (launchpad)
    GET    /api/console/stations/{station_id}    single spec + config schema
    POST   /api/console/preflight                run preflight without triggering
    POST   /api/console/trigger                  enqueue + execute
    GET    /api/console/status/{job_id}          current job state
    GET    /api/console/stream/{job_id}          SSE progress stream
    POST   /api/console/cancel/{job_id}          request cancellation
    GET    /api/console/cost_status              session cost ledger summary
    GET    /api/console/jobs                     recent jobs list

Phase 0a ships with empty station registry — endpoints work but
return 404 on /stations/{station_id} until S1 lands. This is by
design: foundation + registry first, then stations attach.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from engine.operator_console import store, cost_ledger, emit, registry
from engine.operator_console.schema import (
    JobState,
    OperatorEventType,
    SessionType,
)
from engine.operator_console.worker import (
    run_job, get_or_create_queue, request_cancellation,
)
# Importing the stations package auto-registers all attached stations.
# Phase 1.1 (2026-06-23): S1 Paper Ingest registers here.
from engine.operator_console import stations as _opcon_stations  # noqa: F401


router = APIRouter(prefix="/api/console", tags=["operator_console"])
logger = logging.getLogger(__name__)


# ── Request / response models ────────────────────────────────────


class PreflightRequest(BaseModel):
    station_id: str
    session_id: str
    config:     dict = {}


class TriggerRequest(BaseModel):
    station_id:      str
    session_id:      str
    config:          dict = {}
    idempotency_key: str | None = None
    """Optional client-supplied dedup key. If the same key is re-sent
    within store.IDEMPOTENCY_WINDOW_SECONDS (default 60s) for the same
    station+session+actor, the existing job_id is returned and no new
    job is created. Frontend should generate a UUID when the trigger
    form opens and re-send the same value on retry/double-click."""


class CostStatusResponse(BaseModel):
    session_id:        str
    cap_usd:           float
    spent_usd:         float
    remaining_usd:     float
    over_tolerance:    bool


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/stations")
def list_stations() -> dict:
    """All registered stations — drives the launchpad UI."""
    return {
        "stations":      registry.all_specs(),
        "n_registered":  len(registry.all_specs()),
    }


@router.get("/stations/{station_id}")
def get_station(station_id: str) -> dict:
    """Single station — spec + configuration form JSON Schema."""
    cls = registry.get(station_id)
    if cls is None:
        raise HTTPException(404, f"station '{station_id}' not registered")
    from dataclasses import asdict
    spec_dict = asdict(cls.STATION_SPEC)
    spec_dict["data_tier"] = cls.STATION_SPEC.data_tier.value
    spec_dict["requires_session_types"] = sorted(
        t.value for t in cls.STATION_SPEC.requires_session_types
    )
    return {
        "spec":        spec_dict,
        "config_form": cls().render_config_form(),
    }


@router.post("/preflight")
def run_preflight(
    req: PreflightRequest,
    x_actor_id: str = Header(default="principal"),
) -> dict:
    """Run preflight without triggering. Used by UI before showing
    the trigger button — surfaces yellow warnings, hides red blockers."""
    cls = registry.get(req.station_id)
    if cls is None:
        raise HTTPException(404, f"station '{req.station_id}' not registered")
    station = cls()

    # Stub session view; real implementation wires through sessions API
    class _SessionView:
        session_id   = req.session_id
        session_type = ""   # caller provides via separate session lookup
        actor_id     = x_actor_id

    result = station.preflight(_SessionView(), req.config)
    from dataclasses import asdict
    return {
        "can_trigger": result.can_trigger,
        "checks":      [asdict(c) for c in result.checks],
        "estimate":    {
            "total_usd": station.estimate_cost(req.config).total_usd,
        },
    }


@router.post("/trigger")
async def trigger_station(
    req: TriggerRequest,
    background: BackgroundTasks,
    x_actor_id: str = Header(default="principal"),
) -> dict:
    """Enqueue a station execution. Returns immediately with job_id;
    UI polls /status or subscribes to /stream.

    Phase 1.1: actually runs the station async via FastAPI
    BackgroundTasks. Worker pushes SSE events to a per-job queue."""
    cls = registry.get(req.station_id)
    if cls is None:
        raise HTTPException(404, f"station '{req.station_id}' not registered")
    station = cls()
    est = station.estimate_cost(req.config)

    # Cost cap gate (D4)
    allowed, reason = cost_ledger.can_trigger(
        session_id=req.session_id,
        session_cap_usd=cost_ledger.DEFAULT_CAP_USD,   # TODO wire session cap
        estimated_charge_usd=est.total_usd,
    )
    if not allowed:
        raise HTTPException(402, f"cost cap would be exceeded: {reason}")

    # Idempotency check BEFORE creating a new job. If the client passed
    # an idempotency_key and a recent matching job exists (≤ 60s old),
    # short-circuit: return the existing job_id, do not re-emit, do not
    # re-schedule the worker. Protects against double-clicks burning
    # cost twice.
    existing_job_id = (
        store._find_recent_job_by_idempotency_key(
            station_id      = req.station_id,
            session_id      = req.session_id,
            actor_id        = x_actor_id,
            idempotency_key = req.idempotency_key,
        )
        if req.idempotency_key else None
    )
    if existing_job_id is not None:
        existing = store.get_job(existing_job_id)
        return {
            "job_id":          existing_job_id,
            "state":           (existing or {}).get("state", JobState.RUNNING.value),
            "idempotent_hit":  True,
        }

    job_id = store.create_job(
        station_id      = req.station_id,
        session_id      = req.session_id,
        actor_id        = x_actor_id,
        config          = req.config,
        estimated_cost_usd = est.total_usd,
        idempotency_key = req.idempotency_key,
    )
    emit.station_triggered(
        session_id=req.session_id,
        actor_id=x_actor_id,
        job_id=job_id,
        station_id=req.station_id,
        config=req.config,
        estimated_cost_usd=est.total_usd,
    )
    store.update_job_state(job_id, state=JobState.RUNNING)

    # Schedule the worker to run after the response is sent. Worker
    # pushes SSE events to JOB_QUEUES[job_id]; subscribers connect via
    # /stream/{job_id}.
    background.add_task(run_job, job_id)

    return {"job_id": job_id, "state": JobState.RUNNING.value, "idempotent_hit": False}


@router.get("/status/{job_id}")
def job_status(job_id: str) -> dict:
    """Current state of a job. UI polls when not using SSE."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")
    return job


@router.post("/cancel/{job_id}")
def cancel_job(job_id: str, x_actor_id: str = Header(default="principal")) -> dict:
    """Request cancellation. Honored at next stage boundary (R3:
    cannot interrupt mid-stage)."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")
    if job["state"] not in {JobState.QUEUED.value, JobState.RUNNING.value}:
        raise HTTPException(
            409,
            f"cannot cancel job in state '{job['state']}'"
        )
    # Phase 1.1: signal the worker's CancellationToken; worker checks
    # at next stage boundary and exits with success=False per R3.
    signalled = request_cancellation(job_id)
    emit.station_cancelled(
        session_id=job["session_id"],
        actor_id=x_actor_id,
        job_id=job_id,
        station_id=job["station_id"],
        stage_at_cancel=(
            "signal sent; honored at next stage boundary"
            if signalled else
            "(no in-flight token; job may have already completed)"
        ),
    )
    # Do NOT update_job_state to CANCELLED here — that's the worker's
    # job once it actually exits at the stage boundary. Otherwise the
    # status race-condition would mark the job cancelled while it's
    # still running for ~30s.
    return {"job_id": job_id, "state": job["state"], "cancellation_requested": signalled}


@router.get("/cost_status")
def cost_status(session_id: str,
                cap_usd: float = cost_ledger.DEFAULT_CAP_USD) -> CostStatusResponse:
    """Current cost ledger for a session — drives CostCapBanner UI."""
    spent = cost_ledger.compute_session_spend(session_id)
    cap = min(cap_usd, cost_ledger.HARD_CEILING_USD)
    over, _ = cost_ledger.must_halt_mid_execution(
        session_id=session_id, session_cap_usd=cap,
    )
    return CostStatusResponse(
        session_id=session_id,
        cap_usd=cap,
        spent_usd=spent,
        remaining_usd=max(0.0, cap - spent),
        over_tolerance=over,
    )


@router.get("/jobs")
def list_jobs(
    session_id: str | None = None,
    state:      str | None = None,
    limit:      int = 50,
) -> dict:
    """Recent jobs. Drives a future /ops/console page."""
    state_enum = None
    if state:
        try:
            state_enum = JobState(state)
        except ValueError:
            raise HTTPException(400, f"unknown state '{state}'")

    rows = list(store.iter_jobs(
        state=state_enum,
        session_id=session_id,
        limit=limit,
    ))
    return {"jobs": rows, "n": len(rows)}


# ── SSE progress stream ──────────────────────────────────────────


@router.get("/stream/{job_id}")
async def stream_job(job_id: str):
    """Server-Sent Events stream for live progress (D2).

    Phase 1.1: real implementation. Connects to the per-job queue
    populated by the worker; yields stage_started / stage_progress /
    stage_completed / stage_failed / log / job_terminal events as
    they arrive. Closes the stream on `job_terminal` event."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")

    queue = get_or_create_queue(job_id)

    async def event_generator():
        # Send a snapshot first so the client immediately knows the
        # current state (e.g. if connecting late, after the worker
        # already finished and the queue cleanup window passed)
        yield {
            "event": "snapshot",
            "data": json.dumps({"job_id": job_id, "state": job["state"]}),
        }
        # Drain the queue. Each item is {"event": str, "data": json}
        # produced by worker.QueueSSEEmitter. Stop on job_terminal.
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    # Keep-alive ping so proxies don't close idle SSE
                    yield {"event": "ping", "data": "{}"}
                    # Re-check terminal: if job already done elsewhere
                    fresh = store.get_job(job_id)
                    if fresh and fresh["state"] not in {"queued", "running"}:
                        yield {"event": "job_terminal",
                               "data": json.dumps({"job_id": job_id, "state": fresh["state"]})}
                        return
                    continue
                yield item
                if item["event"] == "job_terminal":
                    return
        except asyncio.CancelledError:
            return

    return EventSourceResponse(event_generator())


# ── Restart recovery ─────────────────────────────────────────────


@router.post("/_recover_orphans")
def recover_orphans() -> dict:
    """On server restart, mark abandoned `running` jobs as
    `recovered_unknown` (R6). Idempotent; safe to call on each
    startup."""
    orphans = store.scan_orphaned_running_jobs()
    for jid in orphans:
        store.update_job_state(jid, state=JobState.RECOVERED_UNKNOWN)
        job = store.get_job(jid)
        if job:
            emit.station_failed(
                session_id=job.get("session_id", ""),
                actor_id=job.get("actor_id", "principal"),
                job_id=jid,
                station_id=job.get("station_id", "unknown"),
                stage_failed="server_restart_orphan",
                error="Job was running when server restarted; state unknown.",
            )
    return {"recovered": orphans, "n": len(orphans)}
