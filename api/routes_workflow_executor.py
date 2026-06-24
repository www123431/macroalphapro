"""api/routes_workflow_executor.py — control plane for the autonomous
workflow executor.

Endpoints:
  GET  /api/agents/workflow_executor/status
       -> {paused, paused_ts?, paused_reason?, n_workflows, autorun_count,
           recent_runs: [...last 20...], failure_streak}

  POST /api/agents/workflow_executor/pause   {reason}
       -> {paused: True}                  Rule 9 — kill switch

  POST /api/agents/workflow_executor/resume
       -> {paused: False}

  GET  /api/agents/workflow_executor/workflows
       -> [{workflow_id, description, reversibility, blast_radius_max,
            autorun_allowed, last_run_ts, last_status}]

  POST /api/agents/workflow_executor/run/{workflow_id}
       Body: {dry_run?: bool, inputs?: dict}
       -> WorkflowResult              Manual trigger for testing
"""
from __future__ import annotations

import json
import logging
import datetime as _dt
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents/workflow_executor", tags=["agents"])

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_KILL_FLAG  = _REPO_ROOT / "data" / "agents" / "_kill_switches" / "workflow_executor.flag"
_TRACE_PATH = _REPO_ROOT / "data" / "agents" / "workflow_executor" / "traces.jsonl"


# ── Models ─────────────────────────────────────────────────


class WorkflowInfo(BaseModel):
    workflow_id:       str
    description:       str
    reversibility:     str
    blast_radius_max:  dict
    autorun_allowed:   bool
    last_run_ts:       Optional[str] = None
    last_status:       Optional[str] = None


class RecentRun(BaseModel):
    workflow_id:    str
    status:         str
    reason:         str
    trigger:        str
    ended_ts:       str
    elapsed_s:      float
    dry_run:        bool
    reversibility:  str
    error:          Optional[str] = None


class ExecutorStatus(BaseModel):
    paused:          bool
    paused_ts:       Optional[str] = None
    paused_reason:   Optional[str] = None
    n_workflows:     int
    autorun_count:   int
    failure_streak:  int
    recent_runs:     list[RecentRun]


class PauseRequest(BaseModel):
    reason: str = "manual_pause"


class RunRequest(BaseModel):
    dry_run: Optional[bool] = None
    inputs:  Optional[dict] = None


# ── Helpers ────────────────────────────────────────────────


def _load_kill_switch() -> Optional[dict]:
    if not _KILL_FLAG.is_file():
        return None
    try:
        return json.loads(_KILL_FLAG.read_text(encoding="utf-8"))
    except Exception:
        return {"paused_ts": "unknown", "reason": "(unparseable flag file)"}


def _recent_trace_rows(n: int = 20) -> list[dict]:
    if not _TRACE_PATH.is_file():
        return []
    try:
        rows = []
        with _TRACE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-n:][::-1]   # newest first
    except Exception:
        return []


# ── Endpoints ──────────────────────────────────────────────


@router.get("/status", response_model=ExecutorStatus)
def status() -> ExecutorStatus:
    """Snapshot of the executor: pause state, autorun whitelist,
    and last 20 trace rows."""
    try:
        from engine.agents.workflow_executor import (
            list_workflows, is_autorun_allowed,
        )
        from engine.agents.workflow_executor.runner import _recent_failure_streak
        wfs = list_workflows()
        autorun = sum(1 for c in wfs if is_autorun_allowed(c.workflow_id))
        streak = _recent_failure_streak()
    except Exception as exc:
        logger.exception("workflow_executor status import failed")
        wfs, autorun, streak = [], 0, 0
    ks = _load_kill_switch()
    recent = _recent_trace_rows(20)
    return ExecutorStatus(
        paused          = ks is not None,
        paused_ts       = (ks or {}).get("paused_ts"),
        paused_reason   = (ks or {}).get("reason"),
        n_workflows     = len(wfs),
        autorun_count   = autorun,
        failure_streak  = streak,
        recent_runs     = [
            RecentRun(
                workflow_id   = r.get("workflow_id", "?"),
                status        = r.get("status", "?"),
                reason        = r.get("reason", ""),
                trigger       = r.get("trigger", ""),
                ended_ts      = r.get("ended_ts", ""),
                elapsed_s     = float(r.get("elapsed_s") or 0),
                dry_run       = bool(r.get("dry_run", True)),
                reversibility = r.get("reversibility", "LEVEL_0"),
                error         = r.get("error"),
            ) for r in recent
        ],
    )


@router.post("/pause")
def pause(req: PauseRequest) -> dict:
    """Stop all autonomous execution. Rule 9. Idempotent."""
    try:
        from engine.agents.workflow_executor import set_paused
        set_paused(True, reason=req.reason[:200])
        return {"paused": True, "reason": req.reason[:200]}
    except Exception as exc:
        logger.exception("pause failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@router.post("/resume")
def resume() -> dict:
    """Resume autonomous execution. Idempotent."""
    try:
        from engine.agents.workflow_executor import set_paused
        set_paused(False)
        return {"paused": False}
    except Exception as exc:
        logger.exception("resume failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@router.get("/workflows", response_model=list[WorkflowInfo])
def workflows() -> list[WorkflowInfo]:
    """Every registered workflow + its autorun status + last run."""
    try:
        from engine.agents.workflow_executor import (
            list_workflows, is_autorun_allowed,
        )
        from engine.agents.workflow_executor.runner import _last_run_ts
    except Exception:
        return []
    out: list[WorkflowInfo] = []
    for cls in list_workflows():
        last = _last_run_ts(cls.workflow_id)
        last_status = None
        if last:
            # Cheap: pull the last row for this workflow_id
            for r in reversed(_recent_trace_rows(100)):
                if r.get("workflow_id") == cls.workflow_id:
                    last_status = r.get("status")
                    break
        out.append(WorkflowInfo(
            workflow_id      = cls.workflow_id,
            description      = cls.description,
            reversibility    = (cls.reversibility.value
                                if hasattr(cls.reversibility, "value")
                                else str(cls.reversibility)),
            blast_radius_max = dict(cls.blast_radius_max or {}),
            autorun_allowed  = is_autorun_allowed(cls.workflow_id),
            last_run_ts      = last,
            last_status      = last_status,
        ))
    return out


@router.post("/run/{workflow_id}")
def run_workflow(workflow_id: str, req: RunRequest) -> dict:
    """Manual trigger. Subject to all the same 10 rules — pause / kill
    switch still apply."""
    try:
        from engine.agents.workflow_executor import run_one
        r = run_one(
            workflow_id,
            trigger       = "manual_api",
            force_dry_run = req.dry_run,
            inputs        = req.inputs or {},
        )
        # WorkflowResult is a dataclass -> dict
        import dataclasses as _dc
        return _dc.asdict(r)
    except Exception as exc:
        logger.exception("run failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])
