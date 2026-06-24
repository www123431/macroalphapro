"""engine/research/l4_cron.py — Frontier 2 (2026-06-01):
continuous-background L4 discovery loop.

Pattern:
  Temporal Schedule (cron spec, e.g. "0 9 * * *") fires daily →
  L4CronWorkflow picks the top seed from suggestion_engine that hasn't
  been run in the last N days → starts L4DiscoveryWorkflow as a CHILD
  with enable_reflection=True → returns child workflow_id.

Why a parent cron workflow (not just `client.create_schedule(L4Discovery)`):
  the seed picked must respect the cooling-period filter (don't re-run
  the same idea every day), which requires reading l4_cron_runs.jsonl.
  Temporal Schedules pass a FIXED argument list to the target workflow;
  to compute the arg at fire-time we need a parent workflow that calls
  an activity to pick the seed.

Honesty note: this generates COUNCIL VERDICTS (cheap), not empirical
pipeline runs. candidate_returns_path is None per fired iteration —
the inner pipeline only runs when a human/UI pairs a council outcome
with a real parquet. Keeps cost predictable (~$0.05-0.20/day in LLM
tokens vs. $1+ if pipeline ran).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
L4_CRON_RUNS_LEDGER = REPO_ROOT / "data" / "research" / "l4_cron_runs.jsonl"

# Cron-scoped task queue — separate from the council task queue so the
# cron worker can be scaled / paused independently of ad-hoc trigger
# traffic. Worker registers BOTH this TQ and TASK_QUEUE_L4.
TASK_QUEUE_L4_CRON = "l4-cron"

# Schedule identifier — fixed so enable/disable is idempotent.
L4_CRON_SCHEDULE_ID = "l4-daily-discovery"

# How recently a seed counts as "already run" — within this window we
# skip it and pick the next one. 7d means each seed gets ~weekly
# refresh; enough time for the council to forget priors but not so
# long that we churn through the suggestion pool.
SEED_COOLING_DAYS = 7


# ── Seed selection ────────────────────────────────────────────────────


def _read_recent_seeds(within_days: int = SEED_COOLING_DAYS) -> set[str]:
    """Read recently-run seed strings from the cron ledger.

    Returns a set of seed strings so the picker can O(1) skip them.
    Best-effort: ledger missing / unparseable → empty set."""
    if not L4_CRON_RUNS_LEDGER.is_file():
        return set()
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=within_days)
    seen: set[str] = set()
    with L4_CRON_RUNS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ts_str = row.get("ts", "").rstrip("Z")
                ts = _dt.datetime.fromisoformat(ts_str) if ts_str else None
                if ts and ts >= cutoff:
                    seed = row.get("seed") or ""
                    if seed:
                        seen.add(seed.strip())
            except Exception:
                continue
    return seen


def pick_seed_for_today(*, limit: int = 50) -> Optional[dict]:
    """Pick the highest-ranked suggestion that hasn't been run lately.

    Returns the suggestion dict (with seed, title, family, proposed_role)
    or None if every top-`limit` candidate was already run in the last
    SEED_COOLING_DAYS days — that's a real signal ("pool exhausted")
    and should be logged, not silently retried.
    """
    from engine.research.suggestion_engine import get_candidate_suggestions
    out = get_candidate_suggestions(limit=limit)
    recent = _read_recent_seeds()
    for sug in out.get("suggestions", []):
        seed = (sug.get("seed") or "").strip()
        if not seed or seed in recent:
            continue
        return sug
    return None


# ── Ledger ────────────────────────────────────────────────────────────


def _append_cron_run_log(entry: dict) -> str:
    """Append one row to l4_cron_runs.jsonl. Returns the row's id."""
    import uuid as _uuid
    row_id = _uuid.uuid4().hex[:12]
    try:
        L4_CRON_RUNS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "id":  row_id,
            "ts":  _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **entry,
        }
        with L4_CRON_RUNS_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.exception("l4 cron log append failed (non-fatal)")
    return row_id


def read_recent_cron_runs(limit: int = 50) -> list[dict]:
    """Read recent cron fires, newest first."""
    if not L4_CRON_RUNS_LEDGER.is_file():
        return []
    out: list[dict] = []
    with L4_CRON_RUNS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    out.reverse()
    return out[: max(1, int(limit))]


# ── Temporal activity: pick seed ──────────────────────────────────────


@dataclass
class PickSeedInput:
    limit: int = 50


@dataclass
class PickSeedOutput:
    found: bool
    seed: str = ""
    title: str = ""
    family: str = ""
    proposed_role: str = ""
    rationale: str = ""
    source: str = ""
    score: float = 0.0


@activity.defn(name="l4_cron_pick_seed_activity")
async def pick_seed_activity(inp: PickSeedInput) -> PickSeedOutput:
    """Wraps pick_seed_for_today as a Temporal activity so the cron
    workflow can read l4_cron_runs.jsonl at fire-time."""
    sug = await asyncio.to_thread(pick_seed_for_today, limit=inp.limit)
    if not sug:
        return PickSeedOutput(found=False)
    return PickSeedOutput(
        found=True,
        seed=str(sug.get("seed") or ""),
        title=str(sug.get("title") or ""),
        family=str(sug.get("family") or ""),
        proposed_role=str(sug.get("proposed_role") or ""),
        rationale=str(sug.get("rationale") or ""),
        source=str(sug.get("source") or ""),
        score=float(sug.get("score") or 0.0),
    )


@dataclass
class LogCronRunInput:
    seed: str
    title: str
    family: str
    source: str
    child_workflow_id: Optional[str] = None
    skipped_reason: Optional[str] = None


@activity.defn(name="l4_cron_log_activity")
async def log_cron_run_activity(inp: LogCronRunInput) -> str:
    """Persist a cron fire to the ledger from inside the workflow.

    Returns the row id so the workflow can correlate it with the child
    workflow handle if needed."""
    return await asyncio.to_thread(_append_cron_run_log, {
        "seed":              inp.seed,
        "title":             inp.title,
        "family":            inp.family,
        "source":            inp.source,
        "child_workflow_id": inp.child_workflow_id,
        "skipped_reason":    inp.skipped_reason,
    })


# ── Temporal cron workflow ────────────────────────────────────────────


@workflow.defn(name="L4CronWorkflow", sandboxed=False)
class L4CronWorkflow:
    """Fires once per Schedule tick. Picks a seed, kicks off a child
    L4DiscoveryWorkflow (reflection ON), logs the fire.

    Idempotent on the SEED level (cooling filter prevents same-seed
    duplicate fires within SEED_COOLING_DAYS). NOT idempotent on the
    WORKFLOW level — manual triggers will still run fresh."""

    @workflow.run
    async def run(self) -> dict:
        retry = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_attempts=3,
            backoff_coefficient=2.0,
        )

        seed_out: PickSeedOutput = await workflow.execute_activity(
            pick_seed_activity,
            PickSeedInput(limit=50),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry,
        )

        if not seed_out.found:
            # Pool exhausted — honest log + return
            await workflow.execute_activity(
                log_cron_run_activity,
                LogCronRunInput(
                    seed="", title="", family="", source="",
                    skipped_reason="suggestion_pool_exhausted_within_cooling_window",
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry,
            )
            return {
                "fired":          True,
                "child_started":  False,
                "skipped_reason": "suggestion_pool_exhausted_within_cooling_window",
            }

        # Start the discovery workflow as a CHILD so its lifecycle is
        # tied to this cron fire's parent for Temporal-side observability,
        # but we don't await its completion (fire-and-forget within cron
        # tick — discovery takes 30-60s and we don't want the cron tick
        # to block).
        from engine.research.l4_workflow import (
            L4DiscoveryWorkflow, TASK_QUEUE_L4,
        )
        child_handle = await workflow.start_child_workflow(
            L4DiscoveryWorkflow.run,
            args=[seed_out.seed, None, True],  # candidate_returns=None, reflection=True
            id=f"l4-cron-{workflow.info().workflow_id}-{seed_out.family[:20]}",
            task_queue=TASK_QUEUE_L4,
            parent_close_policy=workflow.ParentClosePolicy.ABANDON,
        )

        child_id = child_handle.id

        await workflow.execute_activity(
            log_cron_run_activity,
            LogCronRunInput(
                seed=seed_out.seed,
                title=seed_out.title,
                family=seed_out.family,
                source=seed_out.source,
                child_workflow_id=child_id,
            ),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=retry,
        )

        return {
            "fired":              True,
            "child_started":      True,
            "child_workflow_id":  child_id,
            "seed_title":         seed_out.title,
            "seed_family":        seed_out.family,
            "seed_source":        seed_out.source,
        }


# ── Client helpers: enable / disable / status ─────────────────────────


async def enable_l4_cron(
    cron_spec: str = "0 9 * * *",
    *,
    address: str = "localhost:7233",
    paused: bool = False,
) -> dict:
    """Create or update the L4 daily-discovery Schedule.

    cron_spec defaults to 09:00 daily server-local. Schedule body is
    L4CronWorkflow with no args (workflow itself fetches the seed at
    run-time, so the schedule arg list stays empty + stable).
    """
    from temporalio.client import (
        Client, Schedule, ScheduleActionStartWorkflow,
        ScheduleSpec, ScheduleState,
    )
    client = await Client.connect(address)
    handle = client.get_schedule_handle(L4_CRON_SCHEDULE_ID)

    # Temporal generates fired workflow ids as f"{id}-<timestamp>".
    # Setting id=schedule_id keeps fires grouped under the schedule
    # namespace for observability.
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            L4CronWorkflow.run,
            id=L4_CRON_SCHEDULE_ID,
            task_queue=TASK_QUEUE_L4_CRON,
        ),
        spec=ScheduleSpec(cron_expressions=[cron_spec]),
        state=ScheduleState(paused=paused),
    )

    # Idempotent: if it exists, update; else create.
    try:
        await handle.describe()
        await handle.update(lambda _input: schedule)
        action = "updated"
    except Exception:
        await client.create_schedule(L4_CRON_SCHEDULE_ID, schedule)
        action = "created"

    return {
        "ok":         True,
        "action":     action,
        "schedule_id": L4_CRON_SCHEDULE_ID,
        "cron":       cron_spec,
        "paused":     paused,
    }


async def disable_l4_cron(
    *,
    address: str = "localhost:7233",
    delete: bool = False,
) -> dict:
    """Pause (default) or delete the schedule.

    Pause is reversible — schedule resumes by calling enable with
    paused=False. Delete drops it entirely; next enable creates fresh.
    """
    from temporalio.client import Client
    client = await Client.connect(address)
    handle = client.get_schedule_handle(L4_CRON_SCHEDULE_ID)
    try:
        if delete:
            await handle.delete()
            return {"ok": True, "action": "deleted",
                    "schedule_id": L4_CRON_SCHEDULE_ID}
        await handle.pause(note="disabled via l4_cron.disable_l4_cron")
        return {"ok": True, "action": "paused",
                "schedule_id": L4_CRON_SCHEDULE_ID}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200],
                "schedule_id": L4_CRON_SCHEDULE_ID}


async def cron_status(
    *,
    address: str = "localhost:7233",
) -> dict:
    """Snapshot of the schedule + recent cron fires from the ledger."""
    from temporalio.client import Client
    schedule_info: dict = {"exists": False}
    try:
        client = await Client.connect(address)
        handle = client.get_schedule_handle(L4_CRON_SCHEDULE_ID)
        desc = await handle.describe()
        schedule_info = {
            "exists":   True,
            "paused":   bool(desc.schedule.state.paused),
            "cron":     list(desc.schedule.spec.cron_expressions) or [],
            "next_run": (
                desc.info.next_action_times[0].isoformat()
                if desc.info.next_action_times else None
            ),
            "running":  desc.info.running_workflows is not None and
                         len(desc.info.running_workflows) > 0,
            "n_recent_actions": desc.info.num_actions,
        }
    except Exception as exc:
        schedule_info = {"exists": False, "error": str(exc)[:200]}

    recent = await asyncio.to_thread(read_recent_cron_runs, 20)
    return {
        "schedule":    schedule_info,
        "recent_runs": recent,
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _cli() -> None:
    """python -m engine.research.l4_cron <enable|disable|status|pick>

    enable [CRON_SPEC] — create/update schedule (default "0 9 * * *")
    disable [--delete] — pause (or delete) schedule
    status             — show schedule + recent fires
    pick               — preview the seed today's cron would pick
                          (useful for dry-run before enabling)
    """
    import sys
    args = sys.argv[1:]
    cmd = args[0] if args else "status"

    if cmd == "pick":
        # No Temporal needed — pure data scan + ledger read
        sug = pick_seed_for_today(limit=50)
        print(json.dumps(sug or {"picked": None,
                                  "reason": "pool exhausted"},
                          indent=2, default=str))
        return

    if cmd == "status":
        out = asyncio.run(cron_status())
        print(json.dumps(out, indent=2, default=str))
        return

    if cmd == "enable":
        cron_spec = args[1] if len(args) > 1 else "0 9 * * *"
        out = asyncio.run(enable_l4_cron(cron_spec=cron_spec))
        print(json.dumps(out, indent=2, default=str))
        return

    if cmd == "disable":
        delete = "--delete" in args
        out = asyncio.run(disable_l4_cron(delete=delete))
        print(json.dumps(out, indent=2, default=str))
        return

    print(f"unknown command: {cmd!r}; "
          "use enable / disable / status / pick", file=__import__("sys").stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    _cli()
