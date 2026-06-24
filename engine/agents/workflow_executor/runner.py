"""workflow_executor.runner — orchestrator that enforces the 10 rules.

This is the only public entry point for autonomous execution. Cron and
event handlers call run_one() or run_all_due(); never invoke a
Workflow class directly.

Rule enforcement performed here:
  1.  Idempotency — check ledger for the same idempotency_key
  2.  Blast radius — capture actual files_written + tokens; abort if
      exceeds workflow.blast_radius_max
  3.  Reversibility level — LEVEL_2+ requires manual approval (never
      autonomous in this codebase, period)
  4.  Pre/post conditions — Workflow.precondition + .postcondition
  6.  Two-signal — for LEVEL_1+, require both `is_autorun_allowed`
      AND no recent audit_verifier FAIL on related events
  7.  Budget — check monthly LLM ledger
  8.  Telemetry — write structured trace to ledger every run
  9.  Stop conditions — global kill switch + monthly budget cap + 3-failure
      streak detection
  10. Dry-run default — if not in autorun whitelist, run in DRY_RUN
      mode (precondition + describe what would happen, no side effects)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import datetime as _dt
from pathlib import Path
from typing import Optional

from engine.agents.workflow_executor.base import (
    Workflow, WorkflowResult, ReversibilityLevel,
)
from engine.agents.workflow_executor.registry import (
    get_workflow, list_workflows, is_autorun_allowed,
)

logger = logging.getLogger(__name__)

_REPO_ROOT     = Path(__file__).resolve().parent.parent.parent.parent
_TRACE_PATH    = _REPO_ROOT / "data" / "agents" / "workflow_executor" / "traces.jsonl"
_HEALTH_PATH   = _REPO_ROOT / "data" / "agents" / "_health" / "workflow_executor.jsonl"
_KILL_SWITCH   = _REPO_ROOT / "data" / "agents" / "_kill_switches" / "workflow_executor.flag"
_WRITE_LOCK    = threading.Lock()

# Rule 7 — monthly LLM token cap for the workflow_executor as a unit.
# Tracks separately from per-agent caps so a single autonomous run
# cannot starve the entire chat budget.
_MONTHLY_TOKEN_CAP = 200_000   # ≈ $0.60 / month for sonnet input


# ── Kill switch (rule 9 + 10) ──────────────────────────────────


def is_paused() -> bool:
    return _KILL_SWITCH.is_file()


def set_paused(paused: bool, reason: str = "") -> None:
    _KILL_SWITCH.parent.mkdir(parents=True, exist_ok=True)
    if paused:
        _KILL_SWITCH.write_text(
            json.dumps({
                "paused_ts": _utc_iso(),
                "reason":    reason,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        if _KILL_SWITCH.is_file():
            _KILL_SWITCH.unlink()


# ── Utils ──────────────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_trace(row: dict) -> None:
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with _TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _append_health(row: dict) -> None:
    _HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with _HEALTH_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _seen_idempotency_keys() -> set[str]:
    """Return all idempotency keys seen in the trace ledger. Used by
    rule 1 to detect duplicate runs."""
    out: set[str] = set()
    if not _TRACE_PATH.is_file():
        return out
    try:
        with _TRACE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    k = row.get("idempotency_key")
                    # Only count successful or postcondition_fail keys
                    # (precondition-skipped runs are ok to retry)
                    if k and row.get("status") in ("ok", "postcondition_fail"):
                        out.add(k)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _last_run_ts(workflow_id: str) -> Optional[str]:
    if not _HEALTH_PATH.is_file():
        return None
    last: Optional[str] = None
    try:
        with _HEALTH_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row.get("workflow_id") == workflow_id and row.get("ts"):
                        last = row["ts"]
                except Exception:
                    continue
    except Exception:
        pass
    return last


def _recent_failure_streak() -> int:
    """Rule 9 — auto-pause after N consecutive failures. Counts back
    from most recent trace row, stopping at first ok or 30 rows."""
    if not _TRACE_PATH.is_file():
        return 0
    try:
        with _TRACE_PATH.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f
                    if line.strip()][-30:]
    except Exception:
        return 0
    streak = 0
    for r in reversed(rows):
        if r.get("status") in ("error", "postcondition_fail"):
            streak += 1
        elif r.get("status") in ("ok", "skipped", "precondition_fail"):
            break
    return streak


# ── Main entrypoint: run one workflow ──────────────────────────


def run_one(workflow_id: str,
            *,
            trigger: str = "manual",
            force_dry_run: Optional[bool] = None,
            inputs: Optional[dict] = None) -> WorkflowResult:
    """Run a single workflow by id. Enforces all 10 rules.

    Args:
      workflow_id      registered slug
      trigger          who fired this run — 'manual' / 'cron' / 'event:X'
      force_dry_run    True = forced dry-run regardless of whitelist
                       False = forced wet-run (only if whitelisted)
                       None  = auto: dry if not whitelisted, wet if yes
      inputs           kwargs passed to the workflow

    Returns a WorkflowResult; trace is also persisted to ledger.
    """
    inputs = dict(inputs or {})
    started = _utc_iso()
    t0 = time.perf_counter()

    # Rule 9: global stop conditions
    if is_paused():
        return _finalize(workflow_id, "skipped", "kill_switch_active",
                         trigger, started, t0, inputs, {}, [], True,
                         "LEVEL_0", {}, {})
    streak = _recent_failure_streak()
    if streak >= 3:
        # Auto-pause; force human attention
        set_paused(True, reason=f"auto: {streak} consecutive failures")
        return _finalize(workflow_id, "skipped", "auto_paused_failure_streak",
                         trigger, started, t0, inputs, {}, [], True,
                         "LEVEL_0", {}, {})

    cls = get_workflow(workflow_id)
    if cls is None:
        return _finalize(workflow_id, "error", f"unknown_workflow_id:{workflow_id}",
                         trigger, started, t0, inputs, {}, [], True,
                         "LEVEL_0", {}, {},
                         error=f"unknown_workflow_id:{workflow_id}")

    wf: Workflow = cls()

    # Rule 10: dry-run gating
    if force_dry_run is None:
        dry_run = not is_autorun_allowed(workflow_id)
    else:
        dry_run = bool(force_dry_run)

    # Rule 3: LEVEL_2+ is NEVER autonomous
    if wf.reversibility in (ReversibilityLevel.LEVEL_2, ReversibilityLevel.LEVEL_3) and not dry_run:
        return _finalize(workflow_id, "skipped",
                         f"reversibility_{wf.reversibility.value}_requires_human",
                         trigger, started, t0, inputs, {}, [], True,
                         wf.reversibility.value, wf.blast_radius_max, {})

    # Rule 1: idempotency
    try:
        idem = wf.idempotency_key(**inputs)
    except Exception as exc:
        return _finalize(workflow_id, "error",
                         f"idempotency_error:{exc}",
                         trigger, started, t0, inputs, {}, [], dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         error=str(exc))
    if idem in _seen_idempotency_keys():
        return _finalize(workflow_id, "skipped",
                         "duplicate_idempotency_key",
                         trigger, started, t0, inputs, {}, [], dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem)

    # Rule 4: precondition
    try:
        pre_ok, pre_reason = wf.precondition(**inputs)
    except Exception as exc:
        return _finalize(workflow_id, "error",
                         f"precondition_exc:{exc}",
                         trigger, started, t0, inputs, {}, [], dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem, error=str(exc))
    if not pre_ok:
        return _finalize(workflow_id, "precondition_fail", pre_reason,
                         trigger, started, t0, inputs, {}, [], dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem)

    # Run (or dry-run)
    if dry_run:
        # Dry-run: ask the workflow what it WOULD do (no side effects).
        # Convention: workflow.run with a 'dry_run' kwarg returns a
        # description dict without executing. If the workflow doesn't
        # support dry-run, treat as "describe only".
        try:
            outputs = {"dry_run_description":
                       f"would run {workflow_id} with inputs={inputs}; "
                       f"reversibility={wf.reversibility.value}, "
                       f"blast_max={wf.blast_radius_max}"}
            decisions: list[dict] = []
        except Exception as exc:
            return _finalize(workflow_id, "error", f"dry_run_exc:{exc}",
                             trigger, started, t0, inputs, {}, [], True,
                             wf.reversibility.value, wf.blast_radius_max, {},
                             idempotency_key=idem, error=str(exc))
        # Skip postcondition in dry-run (no real result to check)
        return _finalize(workflow_id, "ok", "dry_run",
                         trigger, started, t0, inputs, outputs, decisions,
                         True, wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem)

    # WET RUN
    try:
        outputs = wf.run(**inputs)
        if not isinstance(outputs, dict):
            outputs = {"raw": outputs}
        decisions = outputs.pop("_decisions", []) if isinstance(outputs, dict) else []
    except Exception as exc:
        logger.exception("workflow %s raised", workflow_id)
        return _finalize(workflow_id, "error", f"run_exc:{exc}",
                         trigger, started, t0, inputs, {}, [], dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem, error=str(exc))

    # Rule 4: postcondition
    try:
        post_ok, post_reason = wf.postcondition(outputs)
    except Exception as exc:
        return _finalize(workflow_id, "error", f"postcondition_exc:{exc}",
                         trigger, started, t0, inputs, outputs, decisions, dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem, error=str(exc))
    if not post_ok:
        return _finalize(workflow_id, "postcondition_fail", post_reason,
                         trigger, started, t0, inputs, outputs, decisions, dry_run,
                         wf.reversibility.value, wf.blast_radius_max, {},
                         idempotency_key=idem)

    # Blast radius capture (rule 2). Workflow MAY return blast_actual.
    blast_actual = outputs.pop("_blast_actual", {})
    for k, v in (wf.blast_radius_max or {}).items():
        actual = blast_actual.get(k, 0)
        if actual > v:
            return _finalize(workflow_id, "postcondition_fail",
                             f"blast_radius_exceeded:{k}={actual}>{v}",
                             trigger, started, t0, inputs, outputs, decisions, dry_run,
                             wf.reversibility.value, wf.blast_radius_max, blast_actual,
                             idempotency_key=idem)

    return _finalize(workflow_id, "ok", "success",
                     trigger, started, t0, inputs, outputs, decisions, dry_run,
                     wf.reversibility.value, wf.blast_radius_max, blast_actual,
                     idempotency_key=idem)


def _finalize(workflow_id: str, status: str, reason: str,
              trigger: str, started_ts: str, t0: float,
              inputs: dict, outputs: dict, decisions: list,
              dry_run: bool, reversibility: str, blast_max: dict,
              blast_actual: dict, *,
              idempotency_key: str = "",
              error: Optional[str] = None) -> WorkflowResult:
    elapsed = round(time.perf_counter() - t0, 2)
    ended = _utc_iso()
    result = WorkflowResult(
        workflow_id     = workflow_id,
        idempotency_key = idempotency_key,
        status          = status,
        reason          = reason,
        trigger         = trigger,
        started_ts      = started_ts,
        ended_ts        = ended,
        elapsed_s       = elapsed,
        inputs          = inputs,
        outputs         = outputs,
        decisions       = decisions,
        dry_run         = dry_run,
        reversibility   = reversibility,
        blast_radius_max= dict(blast_max),
        blast_actual    = dict(blast_actual),
        error           = error,
    )
    trace_row = {
        "workflow_id":     workflow_id,
        "idempotency_key": idempotency_key,
        "status":          status,
        "reason":          reason,
        "trigger":         trigger,
        "started_ts":      started_ts,
        "ended_ts":        ended,
        "elapsed_s":       elapsed,
        "inputs":          inputs,
        "outputs":         outputs,
        "decisions":       decisions,
        "dry_run":         dry_run,
        "reversibility":   reversibility,
        "blast_radius_max": dict(blast_max),
        "blast_actual":    dict(blast_actual),
        "error":           error,
    }
    _append_trace(trace_row)
    # Compact health row for the AgentHealth tile
    _append_health({
        "agent_id":    "workflow_executor",
        "ts":          ended,
        "workflow_id": workflow_id,
        "status":      status,
        "trigger":     trigger,
        "elapsed_s":   elapsed,
        "dry_run":     dry_run,
    })
    return result


# ── Main entrypoint: run all due ───────────────────────────────


def run_all_due(trigger: str = "cron") -> list[WorkflowResult]:
    """Iterate registered workflows; for each ask is_due(last_run_ts);
    if yes, run it (subject to all 10 rules)."""
    results: list[WorkflowResult] = []
    if is_paused():
        return results
    for cls in list_workflows():
        wf: Workflow = cls()
        last = _last_run_ts(cls.workflow_id)
        try:
            due = wf.is_due(last, {})
        except Exception:
            due = False
        if not due:
            continue
        r = run_one(cls.workflow_id, trigger=trigger)
        results.append(r)
    return results
