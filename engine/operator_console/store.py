"""Operator Console — typed event + job store.

Two append-only jsonl files:

    data/operator_console/jobs.jsonl    one row per station-execution
                                        attempt; updated on state change

    data/operator_console/events.jsonl  one row per OperatorEventType
                                        emit; never mutated (mirrors
                                        research_store doctrine)

Reading is via filter_events() / get_job(); never scrape the jsonl
files directly from outside this module (typed query contract).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from engine.operator_console.schema import (
    JobState,
    OperatorEventType,
    StationResult,
)


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_JOBS_PATH   = _REPO_ROOT / "data" / "operator_console" / "jobs.jsonl"
_EVENTS_PATH = _REPO_ROOT / "data" / "operator_console" / "events.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ── Jobs ─────────────────────────────────────────────────────────


IDEMPOTENCY_WINDOW_SECONDS = 60
"""Idempotency lookback window. If create_job is called with the same
idempotency_key by the same actor + session + station within this many
seconds, the existing job_id is returned instead of creating a duplicate.
60s is the typical 'user double-clicks the trigger button' window;
anything later is genuinely a re-attempt the user intends."""


def _find_recent_job_by_idempotency_key(
    *,
    station_id: str,
    session_id: str,
    actor_id: str,
    idempotency_key: str,
    window_seconds: int = IDEMPOTENCY_WINDOW_SECONDS,
) -> str | None:
    """Scan jobs.jsonl for a recent QUEUED/RUNNING/COMPLETED job matching
    the idempotency key. Returns existing job_id on hit, else None.

    Reading the full jsonl on every create_job is O(N) but the window is
    short and N stays bounded in practice; if jobs.jsonl grows huge a
    secondary index would be the right escape hatch."""
    if not _JOBS_PATH.is_file():
        return None
    cutoff_ts = (_dt.datetime.now(_dt.timezone.utc)
                 - _dt.timedelta(seconds=window_seconds)).strftime(
                     "%Y-%m-%dT%H:%M:%S.%fZ")
    # Latest match wins (in case of multiple — shouldn't happen, but be safe)
    best: str | None = None
    corrupt_lines = 0
    with _JOBS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                corrupt_lines += 1
                continue
            if row.get("station_id") != station_id:
                continue
            if row.get("session_id") != session_id:
                continue
            if row.get("actor_id") != actor_id:
                continue
            if row.get("idempotency_key") != idempotency_key:
                continue
            if (row.get("created_ts") or "") < cutoff_ts:
                continue
            best = row.get("job_id") or best
    if corrupt_lines:
        logger.warning(
            "_find_recent_job_by_idempotency_key: skipped %d corrupt line(s) in %s",
            corrupt_lines, _JOBS_PATH)
    return best


def create_job(
    *,
    station_id: str,
    session_id: str,
    actor_id: str,
    config: dict,
    estimated_cost_usd: float,
    idempotency_key: str | None = None,
) -> str:
    """Allocate a job_id, persist the queued state, return job_id.

    Persisting BEFORE execution starts is the basis of R6 server-
    restart recovery: scanning for `status=running` rows on startup
    can mark orphans as `recovered_unknown`.

    Idempotency: if idempotency_key is supplied AND a job with the same
    (station_id, session_id, actor_id, idempotency_key) was created in
    the last IDEMPOTENCY_WINDOW_SECONDS, the existing job_id is returned
    — no new row is written. Protects against double-click duplicates +
    spurious retries.
    """
    if idempotency_key:
        existing = _find_recent_job_by_idempotency_key(
            station_id      = station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            idempotency_key = idempotency_key,
        )
        if existing is not None:
            logger.info(
                "create_job: idempotency-key hit (%s); returning existing job_id=%s",
                idempotency_key, existing)
            return existing

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    record = {
        "schema_version":     "1.0.0",
        "job_id":             job_id,
        "station_id":         station_id,
        "session_id":         session_id,
        "actor_id":           actor_id,
        "state":              JobState.QUEUED.value,
        "config":             config,
        "estimated_cost_usd": estimated_cost_usd,
        "idempotency_key":    idempotency_key,
        "created_ts":         _utc_iso(),
        "updated_ts":         _utc_iso(),
        "started_ts":         None,
        "completed_ts":       None,
        "result":             None,
        "error":              None,
    }
    _ensure_path(_JOBS_PATH)
    with _JOBS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return job_id


def update_job_state(job_id: str, *, state: JobState,
                     result: StationResult | None = None,
                     error: str | None = None) -> None:
    """Append a state-update row. The store is append-only; latest
    row for a given job_id wins on read. Mirrors the research_store
    'events are immutable, supersedence via parent_event_ids'
    pattern, simplified."""
    update = {
        "schema_version": "1.0.0",
        "job_id":         job_id,
        "state":          state.value,
        "updated_ts":     _utc_iso(),
    }
    if result is not None:
        update["result"]       = asdict(result)
        update["completed_ts"] = result.completed_ts
    if error is not None:
        update["error"] = error[:1000]   # cap

    _ensure_path(_JOBS_PATH)
    with _JOBS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(update, ensure_ascii=False) + "\n")


def get_job(job_id: str) -> dict | None:
    """Return the latest-merged view of a job (create row + all
    subsequent updates merged together).

    On large stores, consider migrating to an SQLite read-index
    (mirrors `data/research_store/_index/` pattern). For now,
    in-memory scan is fine — single-user, <1000 jobs/day max."""
    if not _JOBS_PATH.is_file():
        return None
    merged: dict | None = None
    corrupt_lines = 0
    with _JOBS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                corrupt_lines += 1
                continue
            if row.get("job_id") != job_id:
                continue
            if merged is None:
                merged = dict(row)
            else:
                merged.update(row)
    if corrupt_lines:
        logger.warning("get_job(%s): skipped %d corrupt line(s) in %s",
                       job_id, corrupt_lines, _JOBS_PATH)
    return merged


def iter_jobs(*, state: JobState | None = None,
              session_id: str | None = None,
              actor_id: str | None = None,
              limit: int | None = None) -> Iterator[dict]:
    """Iterate jobs matching filters, newest-first. Used by ops
    dashboard + restart recovery scan."""
    if not _JOBS_PATH.is_file():
        return
    merged: dict[str, dict] = {}
    corrupt_lines = 0
    with _JOBS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                corrupt_lines += 1
                continue
            jid = row.get("job_id")
            if not jid:
                continue
            if jid in merged:
                merged[jid].update(row)
            else:
                merged[jid] = dict(row)
    if corrupt_lines:
        logger.warning("iter_jobs: skipped %d corrupt line(s) in %s",
                       corrupt_lines, _JOBS_PATH)

    rows = list(merged.values())
    rows.sort(key=lambda r: r.get("created_ts", ""), reverse=True)

    n = 0
    for r in rows:
        if state is not None and r.get("state") != state.value:
            continue
        if session_id is not None and r.get("session_id") != session_id:
            continue
        if actor_id is not None and r.get("actor_id") != actor_id:
            continue
        yield r
        n += 1
        if limit is not None and n >= limit:
            return


def scan_orphaned_running_jobs() -> list[str]:
    """R6 — on FastAPI restart, jobs that were `running` are
    abandoned (the worker died). Return their job_ids so the caller
    can mark them RECOVERED_UNKNOWN.

    Idempotent: re-running after the marks are applied returns
    empty list."""
    orphans = []
    for row in iter_jobs(state=JobState.RUNNING):
        orphans.append(row["job_id"])
    return orphans


# ── Events ───────────────────────────────────────────────────────


def emit_event(
    *,
    event_type: OperatorEventType,
    session_id: str | None,
    actor_id: str,
    payload: dict,
    parent_event_ids: list[str] | None = None,
) -> str:
    """Append a typed event. Returns the new event_id.

    Pre-conditions are minimal at this layer (callers know their
    domain). Don't write to events.jsonl directly from outside this
    module — same doctrine as engine.research_store.emit."""
    event_id = f"opce_{uuid.uuid4().hex[:12]}"
    record = {
        "schema_version":   "1.0.0",
        "event_id":         event_id,
        "event_type":       event_type.value,
        "ts":               _utc_iso(),
        "session_id":       session_id,
        "actor_id":         actor_id,
        "parent_event_ids": parent_event_ids or [],
        "payload":          payload,
    }
    _ensure_path(_EVENTS_PATH)
    with _EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return event_id


def filter_events(
    *,
    event_type: OperatorEventType | None = None,
    session_id: str | None = None,
    actor_id: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read events with filters. Newest-first."""
    if not _EVENTS_PATH.is_file():
        return []
    out = []
    corrupt_lines = 0
    for line in _EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            corrupt_lines += 1
            continue
        if event_type is not None and row.get("event_type") != event_type.value:
            continue
        if session_id is not None and row.get("session_id") != session_id:
            continue
        if actor_id is not None and row.get("actor_id") != actor_id:
            continue
        out.append(row)
    if corrupt_lines:
        logger.warning("filter_events: skipped %d corrupt line(s) in %s",
                       corrupt_lines, _EVENTS_PATH)
    out.sort(key=lambda r: r.get("ts", ""), reverse=True)
    if limit is not None:
        out = out[:limit]
    return out
