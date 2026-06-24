"""api/routes_sessions_stream.py — SSE event tail for the active
research session, plus a pending-approval poll endpoint Claude can
hit on a hook to discover new "test this next" intent.

Closes Collab-P1 (R2.5 audit): until today, when a PM approved a
forward vector in /research/forward the approval lived in
data/research_store/forward_vector_reviews.jsonl but nothing told
Claude. Likewise, when Claude emit-ed an event inside an active
session, the user had to refresh / scroll a different page to see
it. Both directions of the collaboration loop now have a real-time
channel.

Endpoints:

  GET /api/sessions/active/events/stream
      SSE that tails data/research_store/events.jsonl filtered by
      `tags=session:<active_session_id>`. Frontend opens this in
      /lab/today's SessionZone to render Claude's emits live.

  GET /api/sessions/forward-approvals/pending
      Returns the list of approved hypothesis_ids whose approval
      timestamp is newer than `since` (defaults to 10 min ago) AND
      that have NOT yet been touched by an active or recent
      research_new session. This is the "Claude, here's what the
      user approved that you haven't picked up" feed. Pollable from
      a cron or Claude Code hook on a 30-60s cadence.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_EVENTS     = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
_REVIEWS    = _REPO_ROOT / "data" / "research_store" / "forward_vector_reviews.jsonl"
_ACTIVE_SES = _REPO_ROOT / "data" / "sessions" / "_active.json"
_SESSIONS   = _REPO_ROOT / "data" / "sessions" / "sessions.jsonl"


def _read_active_session_id() -> Optional[str]:
    if not _ACTIVE_SES.is_file():
        return None
    try:
        return (json.loads(_ACTIVE_SES.read_text(encoding="utf-8")) or {}).get("session_id")
    except (json.JSONDecodeError, OSError):
        return None


def _events_matching_session(rows: list[dict], session_id: str) -> list[dict]:
    """Filter event rows for a session_id tag (or explicit session_id field)."""
    tag = f"session:{session_id}"
    matches = []
    for r in rows:
        if r.get("session_id") == session_id:
            matches.append(r); continue
        tags = r.get("tags") or []
        if isinstance(tags, list) and tag in tags:
            matches.append(r)
    return matches


async def _session_event_tail(session_id: str) -> AsyncGenerator[str, None]:
    """Tail events.jsonl, yielding each new event tagged for this session.

    Two-phase generator:
      1. Backfill: yield existing rows for this session (newest first,
         capped at 30) so the UI has a context window on connect.
      2. Live: poll the file's size+mtime; whenever it grows, yield
         the new lines that match the session.
    """
    # Phase 1 — backfill
    if _EVENTS.is_file():
        try:
            text = _EVENTS.read_text(encoding="utf-8")
            rows = []
            for line in text.splitlines()[-2000:]:
                s = line.strip()
                if not s: continue
                try: rows.append(json.loads(s))
                except json.JSONDecodeError: pass
            matches = _events_matching_session(rows, session_id)
            for r in matches[-30:]:
                yield f"event: backfill\ndata: {json.dumps(r)}\n\n"
        except OSError:
            pass

    # Phase 2 — live tail
    last_size = _EVENTS.stat().st_size if _EVENTS.is_file() else 0
    keepalive_ts = time.time()
    poll_sec = 2.0
    while True:
        await asyncio.sleep(poll_sec)
        try:
            size_now = _EVENTS.stat().st_size if _EVENTS.is_file() else 0
        except OSError:
            size_now = last_size
        if size_now > last_size:
            try:
                with _EVENTS.open("r", encoding="utf-8") as f:
                    f.seek(last_size)
                    new_text = f.read()
                    for line in new_text.splitlines():
                        s = line.strip()
                        if not s: continue
                        try: r = json.loads(s)
                        except json.JSONDecodeError: continue
                        if not _events_matching_session([r], session_id):
                            continue
                        yield f"event: new\ndata: {json.dumps(r)}\n\n"
                last_size = size_now
                keepalive_ts = time.time()
            except OSError:
                continue
        # Heartbeat every 25 seconds so proxies don't close idle connections
        if time.time() - keepalive_ts > 25:
            yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time()})}\n\n"
            keepalive_ts = time.time()


@router.get("/active/events/stream")
async def active_session_event_stream(
    session_id: Optional[str] = Query(
        None, description="explicit session_id; default = current active"),
):
    """SSE stream of research_store events tagged with the active
    session_id. Emits `backfill` events for the last 30 historical
    matches on connect, then `new` events live, plus periodic
    `heartbeat` events to keep proxies open.
    """
    sid = session_id or _read_active_session_id()
    if not sid:
        raise HTTPException(status_code=404,
            detail="no active session — open one from /lab/today")
    return StreamingResponse(
        _session_event_tail(sid),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────── Forward-approval poll (Claude side) ──────


def _load_review_history() -> list[dict]:
    """All review events in chronological order."""
    if not _REVIEWS.is_file():
        return []
    rows: list[dict] = []
    for line in _REVIEWS.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        try: rows.append(json.loads(s))
        except json.JSONDecodeError: pass
    return rows


def _sessions_touching_hypothesis(hypothesis_id: str) -> set[str]:
    """Find research_new sessions referencing this hypothesis_id (via
    pre-flight digest, exit report, or linked events). Used to know if
    the user/Claude has already picked up the approval."""
    if not _SESSIONS.is_file():
        return set()
    touched: set[str] = set()
    needle_a = f'"{hypothesis_id}"'
    needle_b = hypothesis_id[:12]   # fallback short match
    for line in _SESSIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("session_type") != "research_new": continue
        text = json.dumps(r, ensure_ascii=False)
        if needle_a in text or needle_b in text:
            touched.add(r.get("session_id", ""))
    return touched


@router.get("/forward-approvals/pending")
def pending_forward_approvals(
    since_minutes: int = Query(60, ge=1, le=1440,
        description="approvals newer than this many minutes ago"),
):
    """For Claude / cron hooks: list approved hypotheses whose approval
    is recent AND no research_new session has picked them up yet.

    Response: {n_pending, approvals: [{hypothesis_id, reviewed_ts,
    reviewed_by, note}]}. Empty list means "no work waiting".
    """
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(minutes=since_minutes)
    rows = _load_review_history()
    # Latest review per hypothesis_id (later rows win)
    latest: dict[str, dict] = {}
    for r in rows:
        hid = r.get("source_hypothesis_id")
        if not hid: continue
        latest[hid] = r

    out: list[dict] = []
    for hid, r in latest.items():
        if r.get("status") != "approved":
            continue
        ts_str = r.get("reviewed_ts", "")
        try:
            ts = _dt.datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        if _sessions_touching_hypothesis(hid):
            continue   # someone already opened a session for this
        out.append({
            "hypothesis_id":  hid,
            "reviewed_ts":    ts_str,
            "reviewed_by":    r.get("reviewed_by", "user"),
            "note":           r.get("note", ""),
        })

    # Newest pending first — Claude picks the freshest intent
    out.sort(key=lambda x: x.get("reviewed_ts", ""), reverse=True)

    return {
        "n_pending":     len(out),
        "since_minutes": since_minutes,
        "approvals":     out,
    }
