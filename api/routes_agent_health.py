"""api/routes_agent_health.py — agent ops health endpoint.

GET /api/agents/health → array of {agent_id, last_ts, last_status,
n_runs_7d, error_rate_7d, last_summary, source_path}

Surfaces every agent on the ops dashboard so the user can SEE that
autonomous agents are actually working. Without this, the user has no
visual signal whether daily_memo actually ran at 06:30 — only the
markdown file's existence, which they have to navigate to discover.

Sources scanned (all jsonl):
  daily_memo            data/agents/_health/daily_memo.jsonl
  direction_proposer    data/agents/_health/direction_proposer.jsonl
  audit_verifier        data/audit_verifier/lineage_results.jsonl
  graveyard_collision   data/graveyard_collision/warnings.jsonl
  chat_ask              data/research/chat_audit.jsonl  (filter agent=chat_ask)
  workflow_executor     data/agents/_health/workflow_executor.jsonl (Phase 2)

Each entry summarizes per-agent activity for the last 7 days.
"""
from __future__ import annotations

import json
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents/health", tags=["agents"])

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOOKBACK_DAYS = 7


# ── Source descriptors ─────────────────────────────────────


class _SourceDesc:
    def __init__(self, agent_id: str, path: str, ts_field: str,
                 summary_fields: list[str],
                 status_field: Optional[str] = None,
                 ok_status_values: Optional[set] = None,
                 filter_field: Optional[str] = None,
                 filter_value: Optional[str] = None):
        self.agent_id = agent_id
        self.path = path
        self.ts_field = ts_field
        self.summary_fields = summary_fields
        self.status_field = status_field
        self.ok_status_values = ok_status_values or {"ok", "PASS", "CLEAN"}
        self.filter_field = filter_field
        self.filter_value = filter_value


_SOURCES: list[_SourceDesc] = [
    _SourceDesc("daily_memo",
                "data/agents/_health/daily_memo.jsonl",
                "ts",
                ["date_key", "n_citations", "markdown_chars", "elapsed_s"],
                status_field="status",
                ok_status_values={"ok"}),
    _SourceDesc("direction_proposer",
                "data/agents/_health/direction_proposer.jsonl",
                "ts",
                ["diff", "new_count", "today_sig", "elapsed_s"],
                status_field="status",
                ok_status_values={"ok"}),
    _SourceDesc("audit_verifier",
                "data/audit_verifier/lineage_results.jsonl",
                "verified_ts",
                ["research_event_id", "subject_id", "family", "verdict"],
                status_field="verdict",
                ok_status_values={"CLEAN"}),
    _SourceDesc("graveyard_collision",
                "data/graveyard_collision/warnings.jsonl",
                "checked_ts",
                ["candidate_name", "family", "verdict", "n_scanned"],
                status_field="verdict",
                ok_status_values={"CLEAN"}),
    _SourceDesc("workflow_executor",
                "data/agents/_health/workflow_executor.jsonl",
                "ts",
                ["workflow_id", "trigger", "elapsed_s"],
                status_field="status",
                ok_status_values={"ok"}),
]


class AgentHealthRow(BaseModel):
    agent_id:        str
    last_ts:         Optional[str] = None
    last_status:     Optional[str] = None
    last_summary:    Optional[dict] = None
    n_runs_7d:       int = 0
    n_ok_7d:         int = 0
    n_error_7d:      int = 0
    error_rate_7d:   float = 0.0
    source_path:     str
    file_exists:     bool


class AgentHealthResponse(BaseModel):
    generated_ts:    str
    lookback_days:   int
    agents:          list[AgentHealthRow]


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(ts.rstrip("Z")[:19])
    except Exception:
        return None


def _read_health_for(src: _SourceDesc) -> AgentHealthRow:
    p = _REPO_ROOT / src.path
    row = AgentHealthRow(
        agent_id=src.agent_id,
        source_path=src.path,
        file_exists=p.is_file(),
    )
    if not p.is_file():
        return row
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=_LOOKBACK_DAYS)
    last_obj: Optional[dict] = None
    last_t: Optional[_dt.datetime] = None
    n_runs = n_ok = n_err = 0
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if src.filter_field and obj.get(src.filter_field) != src.filter_value:
                    continue
                ts_val = obj.get(src.ts_field, "")
                t = _parse_ts(str(ts_val))
                # Track absolute most recent regardless of cutoff
                if t and (last_t is None or t > last_t):
                    last_t = t
                    last_obj = obj
                if not t or t < cutoff:
                    continue
                n_runs += 1
                if src.status_field:
                    val = obj.get(src.status_field, "")
                    if val in src.ok_status_values:
                        n_ok += 1
                    else:
                        n_err += 1
    except Exception:
        logger.warning("agent_health: failed scanning %s", src.path,
                       exc_info=True)
        return row

    row.n_runs_7d = n_runs
    row.n_ok_7d = n_ok
    row.n_error_7d = n_err
    row.error_rate_7d = round(n_err / n_runs, 3) if n_runs > 0 else 0.0
    if last_obj is not None:
        row.last_ts = last_obj.get(src.ts_field)
        if src.status_field:
            row.last_status = str(last_obj.get(src.status_field) or "")
        row.last_summary = {k: last_obj.get(k) for k in src.summary_fields
                            if k in last_obj}
    return row


@router.get("", response_model=AgentHealthResponse)
def get_agent_health() -> AgentHealthResponse:
    rows = [_read_health_for(s) for s in _SOURCES]
    # Most recently active first; agents with no file go to the bottom
    rows.sort(key=lambda r: (r.last_ts or ""), reverse=True)
    return AgentHealthResponse(
        generated_ts=_utc_iso(),
        lookback_days=_LOOKBACK_DAYS,
        agents=rows,
    )
