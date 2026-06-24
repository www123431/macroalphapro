"""api/routes_chief_of_staff.py — Phase 2.0 step 15 backend.

POST /api/chief_of_staff/run
  {dry_run, session_id?}  →  SessionResult-shaped dict

Wraps `engine.agents.chief_of_staff.run_weekly_session` so the
/lab/today UI button can fire one session without the principal
opening a terminal.

Same fail-safe contract as the underlying runner: never raises into
the route (returns 200 with errors[] populated instead). 500 reserved
for unrecoverable import errors.

Cost: ≤ $0.10 (A's synthesis) + ≤ $0.50 (B's reviews capped at 10) +
$0 (D rules) + ≤ $0.05 (memo) per call. Realistic ~$0.10-0.20.

Cron sibling: scripts/run_weekly_session.py is the headless entry —
use that for the unattended Monday 03:00 UTC schedule. This endpoint
exists for on-demand "run it now" from the UI.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chief_of_staff", tags=["chief_of_staff"])


class WeeklySessionRequest(BaseModel):
    dry_run:    bool = True       # SAFE DEFAULT — explicit opt-in to persist
    session_id: Optional[str] = None


class WeeklySessionResponse(BaseModel):
    session_id:           str
    run_ts:               str
    dry_run:              bool
    d_result:             dict
    a_result:             dict
    b_result:             dict
    session_event_id:     Optional[str] = None
    errors:               list[str] = []
    d_emitted:            int = 0
    a_n_candidates:       int = 0
    a_n_written:          int = 0
    b_n_reviewed:         int = 0
    b_n_pending_approval: int = 0
    memo:                 Optional[dict] = None


@router.post("/run", response_model=WeeklySessionResponse)
def trigger_weekly_session(req: WeeklySessionRequest):
    """Fire ONE chief_of_staff weekly session (D → A → B → memo →
    session event). User-initiated only — never auto-fired by polling.

    dry_run=True (default) propagates to every substep: D skips emit,
    A skips persist, B skips persist, memo skips persist, session
    event skips emit. Use for preview without committing to events.jsonl /
    hypotheses.jsonl / verdicts.jsonl pollution.

    dry_run=False writes everything (production weekly run).
    """
    try:
        from engine.agents.chief_of_staff.runner import run_weekly_session
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"chief_of_staff import failed: {exc}")

    result = run_weekly_session(
        dry_run    = req.dry_run,
        session_id = req.session_id,
    )
    # SessionResult is a dataclass — convert to dict for FastAPI
    return WeeklySessionResponse(**_dc.asdict(result))
