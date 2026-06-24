"""engine.research_store.store — append-only event log.

JSONL on disk at data/research_store/events.jsonl. Append-only by contract;
never mutate prior rows. Idempotent on event_id (duplicate emit = silent
no-op so cron retries don't double-write).

Reading is currently linear scan (load all, filter). Fine up to ~50k events
(this project is unlikely to exceed 10k). When it does, swap for DuckDB
view layer reading the same jsonl.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable, Optional

from engine.research_store.exceptions import DuplicateEventError
from engine.research_store.schema import ResearchEvent

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EVENTS_PATH = _REPO_ROOT / "data" / "research_store" / "events.jsonl"

_LOCK = threading.Lock()


def _ensure_dir() -> None:
    _EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _read_all() -> list[dict]:
    """Read raw event dicts. Skips malformed lines with a warning."""
    if not _EVENTS_PATH.is_file():
        return []
    out: list[dict] = []
    with _EVENTS_PATH.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("events.jsonl line %d malformed; skipping", ln_no)
    return out


def append(event: ResearchEvent) -> None:
    """Append an event to the store. Raises DuplicateEventError if event_id
    already exists. Thread-safe via lock; multi-process not supported (this
    is single-user)."""
    _ensure_dir()
    with _LOCK:
        # Idempotency check — linear scan acceptable at current scale.
        # If we ever care: cache event_ids in memory and invalidate on
        # external mtime change.
        for row in _read_all():
            if row.get("event_id") == event.event_id:
                raise DuplicateEventError(event.event_id)
        with _EVENTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def all_events() -> list[ResearchEvent]:
    """Return all events in store order (insertion order = chronological).
    Lossy ordering across sub-second emits — use ts field for fine sort."""
    return [ResearchEvent.from_dict(d) for d in _read_all()]


def by_event_id(event_id: str) -> Optional[ResearchEvent]:
    for d in _read_all():
        if d.get("event_id") == event_id:
            return ResearchEvent.from_dict(d)
    return None


def filter_events(
    event_type: Optional[str] = None,
    subject_type: Optional[str] = None,
    subject_id: Optional[str] = None,
    verdict: Optional[str] = None,
    family: Optional[str] = None,
    since: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[ResearchEvent]:
    """Filter events. All criteria conjunctive (AND). `since` compares ts
    string lexically — safe because ts is ISO-8601 UTC."""
    out: list[ResearchEvent] = []
    for d in _read_all():
        if event_type   is not None and d.get("event_type")   != event_type:   continue
        if subject_type is not None and d.get("subject_type") != subject_type: continue
        if subject_id   is not None and d.get("subject_id")   != subject_id:   continue
        if verdict      is not None and d.get("verdict")      != verdict:      continue
        if family       is not None and d.get("family")       != family:       continue
        if since        is not None and (d.get("ts") or "")   <  since:        continue
        out.append(ResearchEvent.from_dict(d))
    # Newest first
    out.sort(key=lambda e: e.ts, reverse=True)
    if limit is not None:
        out = out[:limit]
    return out


def events_path() -> Path:
    """Expose the jsonl path for tools that need to inspect / tail it."""
    return _EVENTS_PATH
