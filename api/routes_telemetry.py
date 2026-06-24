"""api/routes_telemetry.py — local-only usage telemetry.

R4.2 — closes the "we don't know which surfaces are actually used"
gap surfaced in the R2 architect audit. Future trim / consolidation
decisions need data; right now they're vibe-driven.

Privacy posture: SINGLE USER, LOCAL. No PII, no remote send, no
sampling. The user IS the analytics consumer.

Storage: append-only jsonl under data/telemetry/events.jsonl.
Aggregation: on-demand via list_summary (no background indexer
needed at this scale; the file caps naturally at ~1000 events/day).

Endpoints
  POST /api/telemetry/event     fire-and-forget event record
  GET  /api/telemetry/summary   top pages + top events, last N days
                                (defaults to 7)
"""
from __future__ import annotations

import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVENTS    = _REPO_ROOT / "data" / "telemetry" / "events.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class TelemetryEventRequest(BaseModel):
    """Single telemetry record.

    `event` is a stable identifier (page_view, cta_click, search,
    etc.). `path` is the URL pathname (no query for privacy on
    multi-user setups; this is single-user so we include it).
    `payload` is free-form: action subject, target route, etc."""
    event:    str
    path:     str = ""
    payload:  dict = {}


@router.post("/event")
def file_event(req: TelemetryEventRequest):
    """Append a telemetry event. Fire-and-forget; returns 200 even
    if disk write fails (we don't want telemetry breaking UX)."""
    record = {
        "ts":      _utc_iso(),
        "event":   req.event,
        "path":    req.path or "",
        "payload": dict(req.payload or {}),
    }
    try:
        _EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with _EVENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Don't surface — telemetry should never break navigation.
        pass
    return {"ok": True}


@router.get("/summary")
def list_summary(days: int = 7):
    """Aggregate events over the last N days.

    Returns top pages by view count, top events by frequency, and
    a sparse hour histogram so the UI can render a simple usage
    pattern (heaviest hour of the day, etc.)."""
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=max(1, min(days, 90)))
    if not _EVENTS.is_file():
        return {
            "n_events":   0,
            "days":       days,
            "top_pages":  [],
            "top_events": [],
            "hour_histogram": {},
        }
    by_page:  Counter = Counter()
    by_event: Counter = Counter()
    by_hour:  Counter = Counter()
    n_total = 0
    n_in_window = 0
    for line in _EVENTS.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        n_total += 1
        ts_str = r.get("ts", "")
        try:
            ts = _dt.datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff: continue
        n_in_window += 1
        page  = r.get("path") or "(none)"
        event = r.get("event") or "(unknown)"
        by_page[page] += 1
        by_event[event] += 1
        by_hour[ts.hour] += 1
    return {
        "n_events":   n_in_window,
        "n_total":    n_total,
        "days":       days,
        "top_pages":  [{"path": p, "n": n} for p, n in by_page.most_common(15)],
        "top_events": [{"event": e, "n": n} for e, n in by_event.most_common(15)],
        "hour_histogram": dict(by_hour),
    }
