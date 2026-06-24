"""api/routes_agent_activity.py — recent agent emission feed.

GET /api/agents/activity?limit=20

Returns a unified feed of recent agent-emitted rows, newest first:
  - factor_verdict_filed events
  - audit_verifier lineage WARN/FAIL
  - graveyard_collision RISK/WARN
  - workflow_executor traces
  - decay_alert events
  - inbox rows tagged with agent_id

Feeds the Activity sidebar on Lab pages (U4) — so the user sees what
the agents have been doing without having to navigate to /agents.
"""
from __future__ import annotations

import json
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])

_REPO_ROOT = Path(__file__).resolve().parent.parent


class ActivityItem(BaseModel):
    ts:           str
    source:       str       # which agent / which file
    kind:         str       # event_type / verdict / status
    title:        str       # short headline
    detail:       Optional[str] = None
    href:         Optional[str] = None
    severity:     str       # "ok" | "warn" | "danger" | "info" | "muted"


class ActivityResponse(BaseModel):
    generated_ts: str
    items:        list[ActivityItem]


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(ts.rstrip("Z")[:19])
    except Exception:
        return None


def _read_jsonl(path: Path, mapper) -> list[ActivityItem]:
    if not path.is_file():
        return []
    out: list[ActivityItem] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                item = mapper(row)
                if item is not None:
                    out.append(item)
    except Exception:
        logger.warning("agent_activity: failed to scan %s", path, exc_info=True)
    return out


def _map_research_event(row: dict) -> Optional[ActivityItem]:
    et = row.get("event_type", "")
    verdict = row.get("verdict", "")
    sev = ("ok" if verdict == "GREEN" else
           "danger" if verdict == "RED" else
           "warn" if verdict == "MARGINAL" else "muted")
    return ActivityItem(
        ts        = row.get("ts", ""),
        source    = f"research_store::{et}",
        kind      = et,
        title     = f"[{verdict or 'NEUTRAL'}] {row.get('subject_id', '?')}",
        detail    = (row.get("summary") or "")[:180],
        href      = f"/research/lessons" if et == "factor_verdict_filed" else None,
        severity  = sev,
    )


def _map_audit_verifier(row: dict) -> Optional[ActivityItem]:
    v = row.get("verdict", "")
    if v in ("CLEAN", "SKIP"):
        return None    # boring — only surface WARN / FAIL
    sev = "danger" if v == "FAIL" else "warn"
    return ActivityItem(
        ts        = row.get("verified_ts", ""),
        source    = "audit_verifier",
        kind      = "lineage_audit",
        title     = f"[{v}] {row.get('subject_id', '?')} ({row.get('family', '?')})",
        detail    = f"checks failed: " + ", ".join(
            c.get("check", "?") for c in (row.get("checks") or [])
            if c.get("status") not in ("PASS",)
        ),
        href      = "/agents",
        severity  = sev,
    )


def _map_graveyard(row: dict) -> Optional[ActivityItem]:
    v = row.get("verdict", "")
    if v in ("CLEAN", "SKIP"):
        return None
    sev = "danger" if v == "RISK" else "warn"
    return ActivityItem(
        ts        = row.get("checked_ts", ""),
        source    = "graveyard_collision",
        kind      = "graveyard_warning",
        title     = f"[{v}] {row.get('candidate_name', '?')} ({row.get('family', '?')})",
        detail    = (row.get("reason") or "")[:180],
        href      = "/research/lessons?verdict=red",
        severity  = sev,
    )


def _map_workflow_trace(row: dict) -> Optional[ActivityItem]:
    status = row.get("status", "")
    sev = ("ok" if status == "ok" else
           "warn" if status == "skipped" else
           "danger" if status in ("error", "postcondition_fail") else "muted")
    wid = row.get("workflow_id", "?")
    dry = " (dry)" if row.get("dry_run") else ""
    return ActivityItem(
        ts        = row.get("ended_ts", ""),
        source    = f"workflow_executor::{wid}",
        kind      = "workflow_run",
        title     = f"[{status}] {wid}{dry}",
        detail    = (row.get("reason") or "")[:180],
        href      = "/agents",
        severity  = sev,
    )


def _map_inbox(row: dict) -> Optional[ActivityItem]:
    agent_id = row.get("agent_id")
    if not agent_id:
        return None
    priority = row.get("priority", "low")
    sev = ("danger" if priority == "high" else
           "warn" if priority == "medium" else "info")
    return ActivityItem(
        ts        = row.get("ts", ""),
        source    = f"inbox::{agent_id}",
        kind      = row.get("kind", "inbox"),
        title     = (row.get("title") or "")[:140],
        detail    = (row.get("body") or "")[:180],
        href      = row.get("where") or "/inbox",
        severity  = sev,
    )


@router.get("/activity", response_model=ActivityResponse)
def get_activity(limit: int = Query(20, ge=1, le=100)) -> ActivityResponse:
    items: list[ActivityItem] = []
    items.extend(_read_jsonl(
        _REPO_ROOT / "data" / "research_store" / "events.jsonl",
        _map_research_event,
    ))
    items.extend(_read_jsonl(
        _REPO_ROOT / "data" / "audit_verifier" / "lineage_results.jsonl",
        _map_audit_verifier,
    ))
    items.extend(_read_jsonl(
        _REPO_ROOT / "data" / "graveyard_collision" / "warnings.jsonl",
        _map_graveyard,
    ))
    items.extend(_read_jsonl(
        _REPO_ROOT / "data" / "agents" / "workflow_executor" / "traces.jsonl",
        _map_workflow_trace,
    ))
    items.extend(_read_jsonl(
        _REPO_ROOT / "data" / "research" / "research_ops_inbox.jsonl",
        _map_inbox,
    ))
    # Sort newest-first
    items.sort(key=lambda i: i.ts or "", reverse=True)
    return ActivityResponse(
        generated_ts = _utc_iso(),
        items        = items[:limit],
    )
