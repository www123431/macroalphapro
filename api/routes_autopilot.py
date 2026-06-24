"""api/routes_autopilot.py — read-only F14a dry-run plan + F14b verdict surfaces.

GET /api/autopilot/dry-run/latest         → latest plan (deterministic recompute)
GET /api/autopilot/dry-run/latest.md      → raw markdown (for human-readable view)
GET /api/autopilot/verdicts/recent        → last N days of F14b live verdicts

Pre-F14b (live cron) — dry-run endpoints are pure metadata recompute over
the catalog and redundancy reports. /verdicts/recent reads the F14b live
run logs (data/autopilot/_live/<date>.json) and renders them as a typed
list for the daily-directive UI.

Per A+B substrate-first roadmap: F15 weaves /api/autopilot/dry-run/latest
into /lab/today; the verdict-history panel closes the loop by surfacing
what F14b actually produced over the past week.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/autopilot", tags=["autopilot"])

_AUTOPILOT_DIR = Path(__file__).resolve().parent.parent / "data" / "autopilot"
_LIVE_DIR      = _AUTOPILOT_DIR / "_live"


class CandidateDecisionOut(BaseModel):
    rank:                int
    source_hypothesis_id: str
    spec_hash:           str
    family:              str
    signal_type:         str
    universe_subset:     str
    weighting:           str
    rebalance:           str
    claim_preview:       str
    action:              str
    reason:              str
    redundancy_advice:   Optional[str] = None
    redundancy_n_red:    int = 0
    cell_n_papers:       int = 0
    cell_in_convergence: bool = False


class DryRunPlanOut(BaseModel):
    plan_ts:            str
    n_ready_specs:      int
    n_would_test:       int
    n_would_skip:       int
    estimated_cost_usd: float
    estimated_wall_s:   int
    decisions:          list[CandidateDecisionOut]


@router.get("/dry-run/latest", response_model=DryRunPlanOut)
def latest_dry_run(top: int = Query(5, ge=1, le=20)):
    """Recompute the dry-run plan now and return it. Recomputation is
    cheap (~100ms; catalog + redundancy joins are in-memory) and gives
    fresh-state semantics — important because catalog state changes
    as new papers / specs / verdicts land throughout the day.

    For the markdown view from the last cron run, use /dry-run/latest.md.
    """
    try:
        from engine.agents.autopilot import compute_dry_run_plan
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"import_failed:{exc}")
    plan = compute_dry_run_plan(top_n=top)
    return DryRunPlanOut(
        plan_ts            = plan.plan_ts,
        n_ready_specs      = plan.n_ready_specs,
        n_would_test       = plan.n_would_test,
        n_would_skip       = plan.n_would_skip,
        estimated_cost_usd = plan.estimated_cost_usd,
        estimated_wall_s   = plan.estimated_wall_s,
        decisions          = [CandidateDecisionOut(**_dc.asdict(d))
                              for d in plan.decisions],
    )


@router.get("/dry-run/latest.md")
def latest_dry_run_markdown():
    """Last-cron-rendered markdown. May be staler than /dry-run/latest
    if cron hasn't run today yet. Returns 404 if no cron output exists."""
    path = _AUTOPILOT_DIR / "latest.md"
    if not path.is_file():
        raise HTTPException(status_code=404,
                             detail="no autopilot/latest.md (cron has not run)")
    return {"path": str(path.relative_to(_AUTOPILOT_DIR.parent.parent)),
            "markdown": path.read_text(encoding="utf-8")}


# ────────────────────────────────────────────────────────────────────
# F14b live verdict history
# ────────────────────────────────────────────────────────────────────


class VerdictRow(BaseModel):
    """One day's F14b run. Mirrors LiveRunResult shape with the fields
    a UI surface actually needs."""
    date:                 str       # YYYY-MM-DD
    verdict:              str       # GREEN | MARGINAL | RED (post-DA)
    score:                int       # 0-4 (post-DA)
    family:               str
    signal_type:          str
    subject_id:           str
    source_hypothesis_id: str
    is_sharpe:            float
    oos_sharpe:           float
    t_stat:               float
    deflated_sr:          float
    max_dd:               float
    n_obs:                int
    capability_evidence_path: str
    ts:                   str
    # Devil's Advocate (Phase 4, 2026-06-05)
    raw_verdict:          str = ""        # pre-DA verdict if DA changed it
    raw_score:            int = 0
    da_fired:             bool = False
    da_tag:               str = "da_skipped"
    da_severity:          str = ""
    da_attack_vector:     str = ""
    da_confidence:        float = 0.0


class VerdictHistoryOut(BaseModel):
    days_requested: int
    n_runs:         int
    counts:         dict[str, int]   # {GREEN, MARGINAL, RED} cumulative
    rows:           list[VerdictRow]
    missing_dates:  list[str]        # dates in window with NO run (cron skipped or future)


@router.get("/verdicts/recent", response_model=VerdictHistoryOut)
def recent_verdicts(days: int = Query(7, ge=1, le=60)):
    """Read F14b verdict logs from data/autopilot/_live/<date>.json for
    the last `days` days. Missing dates are flagged separately so the
    UI can distinguish 'cron didn't fire' from 'cron fired RED'.

    Read-only: never triggers a run, only reads what's on disk. Use
    POST scripts/autopilot_live_run.py (or restart start.bat) to
    actually run today's verdict.
    """
    import json as _json
    import datetime as _dt

    today = _dt.datetime.utcnow().date()
    rows: list[VerdictRow] = []
    missing: list[str] = []
    counts = {"GREEN": 0, "MARGINAL": 0, "RED": 0}

    for k in range(days):
        d = today - _dt.timedelta(days=k)
        ds = d.isoformat()
        path = _LIVE_DIR / f"{ds}.json"
        if not path.is_file():
            missing.append(ds)
            continue
        try:
            raw = _json.loads(path.read_text(encoding="utf-8"))
            row = VerdictRow(
                date                 = ds,
                verdict              = str(raw.get("verdict", "RED")),
                score                = int(raw.get("score", 0)),
                family               = str(raw.get("family", "")),
                signal_type          = str(raw.get("signal_type", "")),
                subject_id           = str(raw.get("subject_id", "")),
                source_hypothesis_id = str(raw.get("source_hypothesis_id", "")),
                is_sharpe            = float(raw.get("is_sharpe") or 0.0),
                oos_sharpe           = float(raw.get("oos_sharpe") or 0.0),
                t_stat               = float(raw.get("t_stat") or 0.0),
                deflated_sr          = float(raw.get("deflated_sr") or 0.0),
                max_dd               = float(raw.get("max_dd") or 0.0),
                n_obs                = int(raw.get("n_obs") or 0),
                capability_evidence_path = str(raw.get("capability_evidence_path", "")),
                ts                   = str(raw.get("ts", "")),
                raw_verdict          = str(raw.get("raw_verdict", "")),
                raw_score            = int(raw.get("raw_score") or 0),
                da_fired             = bool(raw.get("da_fired", False)),
                da_tag               = str(raw.get("da_tag", "da_skipped")),
                da_severity          = str(raw.get("da_severity", "")),
                da_attack_vector     = str(raw.get("da_attack_vector", "")),
                da_confidence        = float(raw.get("da_confidence") or 0.0),
            )
            rows.append(row)
            if row.verdict in counts:
                counts[row.verdict] += 1
        except Exception as exc:
            logger.warning("recent_verdicts: parse failed for %s: %s", path, exc)
            missing.append(ds)

    rows.sort(key=lambda r: r.date, reverse=True)   # newest first
    return VerdictHistoryOut(
        days_requested = days,
        n_runs         = len(rows),
        counts         = counts,
        rows           = rows,
        missing_dates  = sorted(missing, reverse=True),
    )
