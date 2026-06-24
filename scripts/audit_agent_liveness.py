"""
Agent runtime liveness audit (2026-05-04).

Reason for existence: The 2026-05-04 global audit (14 probes) checked
schema / hash / convention but DID NOT check whether each registered agent
was actually running and writing data. As a result `macro_research` had been
silently dead (5 total runs, 3 test-data AlphaMemory rows, 0 reflections)
while the audit reported "0 critical bugs". This script closes that gap.

What it does for each known agent_id:
  - count of AgentRun rows in last 30 days
  - last_run timestamp (any status) + last_succeeded_run timestamp
  - success rate of last 30-day runs
  - count of "downstream" data rows produced in last 30 days (table varies
    by agent — heuristic mapping below)
  - flags `STALE` if last successful run > 14 days ago
  - flags `NEVER_RUN` if zero AgentRun rows total
  - flags `LOW_OUTPUT` if 30-day data writes < 1 (tunable)

Run as part of regular audit cadence, before any "no bugs found" claim.
"""
from __future__ import annotations

import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func


# ─────────────────────────────────────────────────────────────────────────────
# Known agents and their downstream data tables
# Keep this list in sync with engine/agents/* registrations.
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_AGENTS: list[dict] = [
    # macro_research moved to ARCHIVED_AGENTS in P5 cull (2026-05-07).
    # Implementation directory engine/agents/macro_research/ deleted;
    # weekly trigger removed from orchestrator on 2026-05-05.
    {
        "agent_id":            "sector_pipeline",
        "downstream_table":    "decision_logs",
        "downstream_filter":   "tab_type = 'sector'",
        "expected_cadence_days": 30,
    },
    {
        "agent_id":            "memory_curator",
        "downstream_table":    "memory_curator_reports",
        "downstream_filter":   "1=1",            # any row counts
        "downstream_time_col": "generated_at",   # not the default `created_at`
        "expected_cadence_days": 30,
        "no_agent_class":      True,             # function-based, no Agent class
        "callsite":            "engine.memory_curator.run_memory_curator",
    },
    {
        "agent_id":            "universe_review",
        "downstream_table":    "pending_approvals",
        "downstream_filter":   "approval_type = 'universe_change'",
        "expected_cadence_days": 90,
        "no_agent_class":      True,
        "callsite":            "engine.universe_review.run_universe_review",
    },
]


# Agents that historically appeared but are now archived/deleted.
# Listed here so future audits don't flag their stale agent_runs as bugs.
ARCHIVED_AGENTS: tuple[str, ...] = (
    "factor_mad",       # cleanup 2026-05-03 (project_cleanup_2026-05-03.md)
    "dummy_agent",      # 2026-05-02 test residue
    "narrative_overlay",  # rejected Phase 0 (2026-05-02 / 03)
    "risk_narrative_agent",  # cleanup 2026-05-03
    "macro_research",   # P5 cull 2026-05-07 (meta-audit kill 2026-05-05;
                        # dir engine/agents/macro_research/ deleted)
)


def _stale_threshold_days(expected_cadence_days: int) -> int:
    """Treat as stale if last_run > 2 × expected cadence."""
    return max(14, expected_cadence_days * 2)


def audit_one(agent_def: dict, today: datetime.date) -> dict:
    from engine.memory import AgentRun, SessionFactory

    aid = agent_def["agent_id"]
    is_function_agent = bool(agent_def.get("no_agent_class"))
    cutoff_30d = today - datetime.timedelta(days=30)
    cutoff_stale = today - datetime.timedelta(days=_stale_threshold_days(agent_def["expected_cadence_days"]))

    with SessionFactory() as s:
        n_total = (
            s.query(func.count(AgentRun.id))
             .filter(AgentRun.agent_id == aid).scalar() or 0
        )
        n_30d = (
            s.query(func.count(AgentRun.id))
             .filter(AgentRun.agent_id == aid)
             .filter(AgentRun.started_at >= cutoff_30d).scalar() or 0
        )
        n_30d_succeeded = (
            s.query(func.count(AgentRun.id))
             .filter(AgentRun.agent_id == aid)
             .filter(AgentRun.started_at >= cutoff_30d)
             .filter(AgentRun.status == "succeeded").scalar() or 0
        )

        last_run = (
            s.query(AgentRun)
             .filter(AgentRun.agent_id == aid)
             .order_by(AgentRun.started_at.desc()).first()
        )
        last_ok = (
            s.query(AgentRun)
             .filter(AgentRun.agent_id == aid, AgentRun.status == "succeeded")
             .order_by(AgentRun.started_at.desc()).first()
        )

        # Downstream data write count (last 30d) — raw SQL for table flexibility
        from sqlalchemy import text as _sql_text
        ts_col = agent_def.get("downstream_time_col", "created_at")
        try:
            ds_q = _sql_text(
                f"SELECT COUNT(*) FROM {agent_def['downstream_table']} "
                f"WHERE {agent_def['downstream_filter']} "
                f"AND {ts_col} >= :cutoff"
            )
            n_downstream_30d = s.execute(
                ds_q, {"cutoff": datetime.datetime.combine(cutoff_30d, datetime.time())}
            ).scalar() or 0
        except Exception:
            n_downstream_30d = -1  # signal: query failed (column / table may not exist)

    # Flags
    flags: list[str] = []
    if is_function_agent:
        # Function-based agents don't write AgentRun rows — judge solely on
        # downstream data freshness (with cadence-aware staleness).
        if n_downstream_30d == 0:
            flags.append("NO_DOWNSTREAM_DATA_30D")
    else:
        if n_total == 0:
            flags.append("NEVER_RUN")
        elif last_ok is None:
            flags.append("ALL_FAILED")
        elif last_ok.started_at and last_ok.started_at.date() < cutoff_stale:
            flags.append(f"STALE_{(today - last_ok.started_at.date()).days}D")
        if n_30d_succeeded > 0 and n_downstream_30d == 0:
            flags.append("AGENT_RUNS_BUT_NO_DATA_WRITES")
        if n_30d > 0 and n_30d_succeeded == 0:
            flags.append("ALL_30D_RUNS_FAILED")
        if n_total > 0 and n_total < agent_def["expected_cadence_days"] / 30:
            flags.append("LOW_VOLUME_HISTORICAL")

    return {
        "agent_id":             aid,
        "n_total_runs":         n_total,
        "n_30d_runs":           n_30d,
        "n_30d_succeeded":      n_30d_succeeded,
        "success_rate_30d":     (
            round(n_30d_succeeded / n_30d, 2) if n_30d else None
        ),
        "last_run_at":          str(last_run.started_at) if last_run else None,
        "last_succeeded_at":    str(last_ok.started_at) if last_ok else None,
        "n_downstream_30d":     n_downstream_30d,
        "expected_cadence_days": agent_def["expected_cadence_days"],
        "flags":                flags,
        "verdict":              "OK" if not flags else "ISSUES",
    }


def run_audit() -> list[dict]:
    today = datetime.date.today()
    return [audit_one(a, today) for a in KNOWN_AGENTS]


def detect_orphan_agent_runs() -> list[dict]:
    """Find agent_runs rows whose agent_id is NOT in KNOWN_AGENTS or
    ARCHIVED_AGENTS — i.e., a new agent class registered runs but isn't
    declared. Always a maintenance smell."""
    from engine.memory import AgentRun, SessionFactory
    from sqlalchemy import func, distinct

    declared = {a["agent_id"] for a in KNOWN_AGENTS} | set(ARCHIVED_AGENTS)
    with SessionFactory() as s:
        rows = s.query(
            AgentRun.agent_id,
            func.count(AgentRun.id),
        ).group_by(AgentRun.agent_id).all()

    orphans = [
        {"agent_id": aid, "runs": n}
        for aid, n in rows if aid not in declared
    ]
    return orphans


def detect_archived_with_recent_activity(today: datetime.date) -> list[dict]:
    """Agents listed in ARCHIVED_AGENTS that still have runs in last 30d —
    means cleanup is incomplete."""
    from engine.memory import AgentRun, SessionFactory
    cutoff = today - datetime.timedelta(days=30)
    findings: list[dict] = []
    with SessionFactory() as s:
        for aid in ARCHIVED_AGENTS:
            n_recent = (
                s.query(AgentRun)
                 .filter(AgentRun.agent_id == aid)
                 .filter(AgentRun.started_at >= cutoff)
                 .count()
            )
            if n_recent:
                findings.append({"agent_id": aid, "recent_runs_30d": n_recent})
    return findings


def _print_report(results: list[dict]) -> bool:
    """Returns True if all OK, False if any issues found."""
    print("=" * 78)
    print(f"Agent Liveness Audit  ·  {datetime.date.today()}")
    print("=" * 78)
    all_ok = True
    for r in results:
        verdict = r["verdict"]
        marker = "OK " if verdict == "OK" else "!! "
        if verdict != "OK":
            all_ok = False
        print(f"\n[{marker}] {r['agent_id']}")
        print(f"        n_total_runs       = {r['n_total_runs']}")
        print(f"        n_30d_runs         = {r['n_30d_runs']}  succeeded={r['n_30d_succeeded']}")
        print(f"        success_rate_30d   = {r['success_rate_30d']}")
        print(f"        last_run_at        = {r['last_run_at']}")
        print(f"        last_succeeded_at  = {r['last_succeeded_at']}")
        print(f"        n_downstream_30d   = {r['n_downstream_30d']}")
        print(f"        expected_cadence   = every {r['expected_cadence_days']} days")
        if r["flags"]:
            print(f"        flags              = {r['flags']}")
    # Orphan agent_id detection
    orphans = detect_orphan_agent_runs()
    if orphans:
        all_ok = False
        print("\n[!! ] ORPHAN agent_ids in agent_runs (declare or archive):")
        for o in orphans:
            print(f"        {o['agent_id']:30s}  runs={o['runs']}")

    # Archived agents with recent activity (cleanup smell)
    archived_active = detect_archived_with_recent_activity(datetime.date.today())
    if archived_active:
        all_ok = False
        print("\n[!! ] ARCHIVED agents with recent runs (cleanup incomplete):")
        for a in archived_active:
            print(f"        {a['agent_id']:30s}  recent_runs_30d={a['recent_runs_30d']}")

    print("\n" + "=" * 78)
    print("VERDICT: " + ("ALL OK" if all_ok else "ISSUES PRESENT"))
    print("=" * 78)
    return all_ok


if __name__ == "__main__":
    results = run_audit()
    ok = _print_report(results)
    sys.exit(0 if ok else 1)
