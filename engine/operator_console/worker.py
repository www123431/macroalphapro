"""Async station execution worker + in-process SSE event bus.

Per design doc D2: SSE for live progress. Per D1: async execution
via existing workflow_executor pattern; this module is the worker
that pulls a triggered job, instantiates the station, runs execute()
with an SSEEmitter that pushes events to a per-job queue, then marks
the job terminal.

In-process queue model:
    JOB_QUEUES: dict[job_id, asyncio.Queue]
    Each SSE endpoint subscriber dequeues events for its job_id.
    Worker pushes; subscriber pulls; queue is created lazily.

Server-restart caveat (R6): jobs running when uvicorn restarts are
orphaned — queue lost, worker died. routes_operator_console restart
scan marks them RECOVERED_UNKNOWN. Out-of-scope to persist queue
state in MVP; see docs/architecture/operator_console.md Risk #5.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from engine.operator_console import emit as opcon_emit
from engine.operator_console import registry, store
from engine.operator_console.pipeline_station import SSEEmitter
from engine.operator_console.schema import (
    CancellationToken,
    JobState,
    StationResult,
)


logger = logging.getLogger(__name__)


# ── Per-job queue + cancellation registry ────────────────────────


# job_id → asyncio.Queue of SSE event dicts
JOB_QUEUES: dict[str, asyncio.Queue] = {}

# job_id → CancellationToken (so /cancel API can flip the flag)
JOB_CANCELLATIONS: dict[str, CancellationToken] = {}


def get_or_create_queue(job_id: str) -> asyncio.Queue:
    """Lazy queue creation. Returned to both worker (push) and SSE
    endpoint (pull)."""
    q = JOB_QUEUES.get(job_id)
    if q is None:
        q = asyncio.Queue(maxsize=200)
        JOB_QUEUES[job_id] = q
    return q


def get_or_create_cancellation(job_id: str) -> CancellationToken:
    tok = JOB_CANCELLATIONS.get(job_id)
    if tok is None:
        tok = CancellationToken()
        JOB_CANCELLATIONS[job_id] = tok
    return tok


def request_cancellation(job_id: str) -> bool:
    """Flip the cancellation flag for an in-flight job. Returns True
    if a token existed (job was active), False if not (job already
    terminal or never started). Honored at next stage boundary per R3."""
    tok = JOB_CANCELLATIONS.get(job_id)
    if tok is None:
        return False
    tok.cancel()
    return True


def cleanup_job(job_id: str) -> None:
    """Drop the queue + cancellation token after the job reaches
    terminal state. Called by worker after marking job done."""
    JOB_QUEUES.pop(job_id, None)
    JOB_CANCELLATIONS.pop(job_id, None)


# ── SSE emitter (worker-side) ────────────────────────────────────


@dataclass
class QueueSSEEmitter:
    """Pushes events into the per-job queue. Implements the SSEEmitter
    Protocol used by station.execute().

    Each named method maps to an SSE event type that the frontend
    StationProgressStream consumer understands."""

    job_id: str
    queue:  asyncio.Queue

    def _put(self, event: str, payload: dict) -> None:
        try:
            self.queue.put_nowait({"event": event, "data": json.dumps(payload, ensure_ascii=False)})
        except asyncio.QueueFull:
            logger.warning("worker: SSE queue full for job_id=%s; dropping event=%s",
                           self.job_id, event)

    def stage_started(self, stage: str, expected_seconds: int = 0) -> None:
        self._put("stage_started", {
            "stage":            stage,
            "expected_seconds": expected_seconds,
        })

    def stage_progress(self, stage: str, pct: int, current: str = "") -> None:
        self._put("stage_progress", {
            "stage":   stage,
            "pct":     pct,
            "current": current,
        })

    def stage_completed(self, stage: str, result: dict[str, Any]) -> None:
        self._put("stage_completed", {
            "stage":  stage,
            "result": result,
        })

    def stage_failed(self, stage: str, error: str) -> None:
        self._put("stage_failed", {
            "stage": stage,
            "error": error,
        })

    def log_line(self, line: str) -> None:
        self._put("log", {"line": line})

    def terminal(self, state: str) -> None:
        """Emit the job_terminal SSE event signalling the stream is
        done. Public hook so callers (run_job) don't reach into _put."""
        self._put("job_terminal", {"job_id": self.job_id, "state": state})


# ── Worker ───────────────────────────────────────────────────────


@dataclass
class _SessionView:
    """Minimal Session struct passed to station.execute(). Real
    sessions API returns rich SessionRow; for now we pass the
    session_id + actor_id + a (best-effort) type."""
    session_id:   str
    session_type: str = ""
    actor_id:     str = "principal"


async def run_job(job_id: str) -> None:
    """Worker entry point. Looks up job, instantiates station, runs
    execute() with queued SSE emitter, marks terminal state.

    Called from the trigger endpoint via FastAPI BackgroundTasks
    (or asyncio.create_task). Returns when the job reaches a
    terminal state — no exceptions propagate (all caught + recorded
    as job_failed)."""
    job = store.get_job(job_id)
    if job is None:
        logger.error("worker: job_id=%s not found", job_id)
        return

    station_cls = registry.get(job["station_id"])
    if station_cls is None:
        store.update_job_state(job_id, state=JobState.FAILED,
                               error=f"station '{job['station_id']}' not registered at execute time")
        cleanup_job(job_id)
        return

    queue = get_or_create_queue(job_id)
    cancellation = get_or_create_cancellation(job_id)
    emitter = QueueSSEEmitter(job_id=job_id, queue=queue)
    session_view = _SessionView(
        session_id   = job.get("session_id", ""),
        actor_id     = job.get("actor_id", "principal"),
    )

    station = station_cls()
    try:
        result: StationResult = await station.execute(
            session      = session_view,
            config       = job.get("config", {}),
            emitter      = emitter,
            cancellation = cancellation,
        )
    except Exception as e:
        logger.exception("worker: job_id=%s station.execute() raised", job_id)
        emitter.stage_failed("execute", str(e)[:300])
        store.update_job_state(job_id, state=JobState.FAILED, error=str(e)[:1000])
        try:
            opcon_emit.station_failed(
                session_id   = session_view.session_id,
                actor_id     = session_view.actor_id,
                job_id       = job_id,
                station_id   = job["station_id"],
                stage_failed = "execute",
                error        = str(e),
            )
        except Exception:
            logger.exception("worker: failed to emit station_failed event")
        # Final terminal SSE event so subscribers close cleanly
        emitter.terminal(JobState.FAILED.value)
        cleanup_job(job_id)
        return

    # Determine terminal state from result
    if cancellation.cancelled and not result.success:
        terminal = JobState.CANCELLED
    elif result.success:
        terminal = JobState.COMPLETED
    else:
        terminal = JobState.FAILED

    store.update_job_state(job_id, state=terminal, result=result,
                           error=result.error_message or None)

    emitter.terminal(terminal.value)

    # Defer cleanup briefly so any in-flight SSE subscriber drains
    await asyncio.sleep(0.1)
    cleanup_job(job_id)
