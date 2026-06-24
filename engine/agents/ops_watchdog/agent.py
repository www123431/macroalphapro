"""
engine/agents/ops_watchdog/agent.py — Watchdog orchestrator (Phase 2).

Entry point: `run_watchdog(...)`. Wires together:
  1. Auto-Audit run (scope="watchdog") — fires WATCHDOG_RULES, writes
     AuditFinding rows via the existing engine.auto_audit.run_audit path.
  2. Hardcoded triage — engine.agents.ops_watchdog.triage.aggregate_severity.
  3. LLM ReAct context pass — engine.quant_co_pilot.base.run_react_agent with
     `agent_id="ops_watchdog"` + Watchdog's role_intro + 10-tool dispatcher.
     Cost-capped at $0.20/run (spec §2.3); skipped when severity is "none"
     (no findings → no LLM spend) or when `dry_run=True`.
  4. Trace JSON write to data/ops_watchdog/{YYYY-MM-DD}_run.json (audit trail).

NOT in Phase 2:
  - Auto-repair recipes (Phase 3, engine/agents/ops_watchdog/auto_repair.py)
  - Notification dispatch (Phase 4, .../notifications.py)
  - Dashboard widget hooks (Phase 4)

Invariants enforced here:
  - Read-only on production tables (writes only to AuditRun / AuditFinding
    via run_audit; writes trace JSON; writes ledger via _call_llm).
  - LLM cost capped — Tool 1's run_react_agent enforces, plus this caller
    passes cost_budget_usd=$0.20.
  - Triage is hardcoded, not LLM-decided.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

from engine.agents.observability import track_agent_invocation

logger = logging.getLogger(__name__)

# Watchdog spec §2.3 — per-run LLM budget cap. Hardcoded here, NOT pulled from
# Tool 1 (Tool 1's locked cap is $0.05, different operating point).
WATCHDOG_COST_BUDGET_USD: float = 0.20
WATCHDOG_MAX_STEPS:       int   = 8
WATCHDOG_LATENCY_BUDGET_MS: int = 60000   # 60s wall clock cap for ReAct loop


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class WatchdogRunResult:
    """One Watchdog run summary."""
    started_at_iso:    str
    finished_at_iso:   str
    today_iso:         str
    dry_run:           bool
    audit_run_id:      Optional[int]
    audit_exit_status: Optional[str]
    n_rules_run:       int
    n_findings:        int
    findings_summary:  list[dict]    # subset for trace + dashboard
    triage:            dict          # aggregate_severity output
    auto_repair:       dict          # Phase 3: {n_attempted, n_succeeded, n_failed,
                                     # n_deferred, results: [RepairResult-as-dict]}
    notifications:     dict          # Phase 4: {dashboard, toast, email, halt_flag}
                                     # bool per channel (True = successfully fired)
    llm_used:          bool
    llm_cost_usd:      float
    llm_n_steps:       int
    llm_final_answer:  str
    llm_abort_reason:  Optional[str]
    trace_json_path:   Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _trace_dir() -> Path:
    return _repo_root() / "data" / "ops_watchdog"


def _collect_findings_for_run(audit_run_id: int, limit: int = 50) -> list[dict]:
    """
    Read AuditFinding rows for a given run_id. Returns ordered by severity
    desc (HIGH > MID > LOW) for prompt-prefix consumption.
    """
    try:
        from engine.auto_audit_models import AuditFinding
        from engine.memory import SessionFactory
    except Exception as exc:
        logger.warning("watchdog: failed to import audit models: %s", exc)
        return []

    sev_rank = {"HIGH": 3, "MID": 2, "LOW": 1}
    try:
        with SessionFactory() as s:
            rows = (s.query(AuditFinding)
                     .filter(AuditFinding.run_id == audit_run_id)
                     .limit(limit)
                     .all())
    except Exception as exc:
        logger.warning("watchdog: failed to read findings for run_id=%s: %s",
                       audit_run_id, exc)
        return []

    out = []
    for r in rows:
        try:
            snap = json.loads(r.snapshot_json) if r.snapshot_json else {}
        except Exception:
            snap = {"_parse_error": True}
        out.append({
            "finding_id":     r.id,
            "rule_name":      r.rule_name,
            "severity":       r.severity,
            "snapshot":       _truncate_snapshot(snap),
            "detected_at":    r.detected_at.isoformat() if r.detected_at else None,
        })
    out.sort(key=lambda f: -sev_rank.get(f["severity"], 0))
    return out


def _truncate_snapshot(snap: dict, max_chars: int = 500) -> dict:
    """Keep snapshot small for LLM prompt — drop large list contents."""
    if not isinstance(snap, dict):
        return {"_invalid_shape": True}
    out: dict[str, Any] = {}
    for k, v in snap.items():
        if isinstance(v, list) and len(v) > 3:
            out[k] = {"_list_truncated_to_3": v[:3], "_n_total": len(v)}
        elif isinstance(v, str) and len(v) > max_chars:
            out[k] = v[:max_chars] + "...[truncated]"
        else:
            out[k] = v
    return out


def _save_trace_json(result: WatchdogRunResult,
                     react_trace: Optional[dict]) -> Path:
    """Persist run trace under data/ops_watchdog/{YYYY-MM-DD}_run.json."""
    trace_dir = _trace_dir()
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{result.today_iso}_run.json"
    payload = {
        "schema_version":   3,             # Phase 4 added notifications field
        "spec_id":          63,
        "spec_hash":        "9d050804",   # post-amendment 1 (mode 13)
        "started_at_iso":   result.started_at_iso,
        "finished_at_iso":  result.finished_at_iso,
        "today_iso":        result.today_iso,
        "dry_run":          result.dry_run,
        "audit_run_id":     result.audit_run_id,
        "audit_exit_status": result.audit_exit_status,
        "n_rules_run":      result.n_rules_run,
        "n_findings":       result.n_findings,
        "findings_summary": result.findings_summary,
        "triage":           result.triage,
        "auto_repair":      result.auto_repair,
        "notifications":    result.notifications,
        "llm_used":         result.llm_used,
        "llm_cost_usd":     result.llm_cost_usd,
        "llm_n_steps":      result.llm_n_steps,
        "llm_final_answer": result.llm_final_answer,
        "llm_abort_reason": result.llm_abort_reason,
        "llm_react_trace":  react_trace,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def _react_trace_to_dict(trace: Any) -> dict:
    """Serialize TraceResult dataclass to JSON-friendly dict."""
    try:
        return {
            "query":            getattr(trace, "query", None),
            "final_answer":     getattr(trace, "final_answer", None),
            "n_citations":      len(getattr(trace, "citations", []) or []),
            "annotated_answer": getattr(trace, "annotated_answer", None),
            "n_steps":          len(getattr(trace, "steps", []) or []),
            "cost_usd":         getattr(trace, "cost_usd", 0.0),
            "latency_ms":       getattr(trace, "latency_ms", 0),
            "abort_reason":     getattr(trace, "abort_reason", None),
            "completed_at":     getattr(trace, "completed_at", None),
            "steps":            [dataclasses.asdict(s)
                                 for s in (getattr(trace, "steps", []) or [])],
        }
    except Exception as exc:
        return {"_serialize_failed": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def _watchdog_schema_validator(result: WatchdogRunResult) -> bool:
    """Schema check: result has required fields + sane structure."""
    valid_statuses = {None, "ok", "fail", "partial", "OK", "FAIL", "PARTIAL"}
    return (
        isinstance(result, WatchdogRunResult)
        and result.audit_exit_status in valid_statuses
        and isinstance(result.triage, dict)
        and "severity" in result.triage
        and isinstance(result.n_findings, int)
        and result.n_findings >= 0
    )


def _watchdog_quality_extractor(result: WatchdogRunResult) -> dict:
    """Extract quality signals: tool execution success rate + auto-repair stats."""
    auto_repair = result.auto_repair or {}
    n_attempted = int(auto_repair.get("n_attempted", 0))
    n_succeeded = int(auto_repair.get("n_succeeded", 0))
    tool_success_rate = (n_succeeded / n_attempted) if n_attempted > 0 else None

    return {
        "tool_success_rate":     tool_success_rate,
        "n_findings":            result.n_findings,
        "severity":              result.triage.get("severity"),
        "llm_used":              result.llm_used,
        "n_repairs_attempted":   n_attempted,
        "n_repairs_succeeded":   n_succeeded,
        "n_repairs_deferred":    int(auto_repair.get("n_deferred", 0)),
        "llm_n_steps":           result.llm_n_steps,
    }


def _watchdog_extra(result: WatchdogRunResult) -> dict:
    """Per-invocation context for the metrics record."""
    return {
        "today_iso":         result.today_iso,
        "audit_run_id":      result.audit_run_id,
        "n_tool_calls":      result.llm_n_steps,    # Watchdog ReAct steps as tool calls proxy
        "trace_json_path":   result.trace_json_path,
    }


@track_agent_invocation(
    agent_id="ops_watchdog",
    schema_validator=_watchdog_schema_validator,
    extract_extra=_watchdog_extra,
    quality_extractor=_watchdog_quality_extractor,
)
def run_watchdog(
    *,
    dry_run:    bool = False,
    today:      Optional[datetime.date] = None,
    save_trace: bool = True,
    verbose:    bool = False,
) -> WatchdogRunResult:
    """
    Execute one Watchdog cycle.

    Args:
        dry_run:    if True, skip LLM ReAct (rules + triage only). Useful for
                    CI smoke and Task Scheduler dev runs without API spend.
        today:      override "today" (for tests / replay); default = UTC date.
        save_trace: write trace JSON to data/ops_watchdog/. Default True.
        verbose:    log INFO-level progress messages.

    Returns:
        WatchdogRunResult dataclass.
    """
    from engine.auto_audit import run_audit
    from engine.agents.ops_watchdog.triage import (
        aggregate_severity, SEVERITY_NONE,
    )

    if today is None:
        today = datetime.date.today()
    started = datetime.datetime.utcnow()

    if verbose:
        logger.setLevel(logging.INFO)
        logger.info("Watchdog start: today=%s dry_run=%s", today, dry_run)

    # ── Phase 0: run WATCHDOG_RULES, persist AuditFinding rows ────────────
    audit_summary = run_audit("watchdog")
    audit_run_id  = audit_summary.get("run_id")
    findings      = _collect_findings_for_run(audit_run_id) if audit_run_id else []
    fired_rules   = [f["rule_name"] for f in findings]

    if verbose:
        logger.info("Watchdog audit run_id=%s n_findings=%d",
                    audit_run_id, len(findings))

    # ── Phase 1: hardcoded triage ────────────────────────────────────────
    triage_decision = aggregate_severity(fired_rules)
    if verbose:
        logger.info("Watchdog triage severity=%s modes_fired=%s",
                    triage_decision["severity"], triage_decision["modes_fired"])

    # ── Phase 1.5: auto-repair (Phase 3 — modes 1/2/6 active; 4/10/12 deferred) ─
    auto_repair_summary: dict = {
        "n_attempted":  0,
        "n_succeeded":  0,
        "n_failed":     0,
        "n_deferred":   0,
        "results":      [],
    }
    if not dry_run:
        try:
            from engine.agents.ops_watchdog.auto_repair import (
                execute_repairs_for_findings,
            )
            repair_results = execute_repairs_for_findings(findings)
            for rr in repair_results:
                rr_dict = dataclasses.asdict(rr)
                auto_repair_summary["results"].append(rr_dict)
                auto_repair_summary["n_attempted"] += 1
                if rr.deferred:
                    auto_repair_summary["n_deferred"] += 1
                elif rr.success:
                    auto_repair_summary["n_succeeded"] += 1
                else:
                    auto_repair_summary["n_failed"] += 1
            if verbose:
                logger.info(
                    "Watchdog auto_repair: attempted=%d succeeded=%d failed=%d deferred=%d",
                    auto_repair_summary["n_attempted"],
                    auto_repair_summary["n_succeeded"],
                    auto_repair_summary["n_failed"],
                    auto_repair_summary["n_deferred"],
                )
        except Exception as exc:
            logger.exception("Watchdog auto_repair dispatch failed: %s", exc)
            auto_repair_summary["dispatch_error"] = str(exc)

    # ── Phase 2: LLM ReAct context pass (conditional) ────────────────────
    llm_used        = False
    llm_cost_usd    = 0.0
    llm_n_steps     = 0
    llm_final       = ""
    llm_abort       = None
    react_trace_dict: Optional[dict] = None

    should_skip_llm = dry_run or triage_decision["severity"] == SEVERITY_NONE
    if not should_skip_llm:
        try:
            from engine.agents.ops_watchdog.prompt import (
                WATCHDOG_ROLE_INTRO, build_watchdog_query,
            )
            from engine.agents.ops_watchdog.tools import (
                WATCHDOG_TOOL_DESCRIPTIONS, WATCHDOG_TOOL_NAMES,
                dispatch_watchdog_tool,
            )
            from engine.quant_co_pilot.base import run_react_agent

            query = build_watchdog_query(
                today_iso          = today.isoformat(),
                findings_preview   = findings,
                triage_pre_summary = triage_decision,
            )

            trace = run_react_agent(
                query             = query,
                tool_dispatcher   = dispatch_watchdog_tool,
                tool_descriptions = WATCHDOG_TOOL_DESCRIPTIONS,
                max_steps         = WATCHDOG_MAX_STEPS,
                cost_budget_usd   = WATCHDOG_COST_BUDGET_USD,
                latency_budget_ms = WATCHDOG_LATENCY_BUDGET_MS,
                valid_tool_names  = set(WATCHDOG_TOOL_NAMES),
                agent_id          = "ops_watchdog",
                role_intro        = WATCHDOG_ROLE_INTRO,
            )
            llm_used        = True
            llm_cost_usd    = float(getattr(trace, "cost_usd", 0.0))
            llm_n_steps     = len(getattr(trace, "steps", []) or [])
            llm_final       = getattr(trace, "annotated_answer", "") or ""
            llm_abort       = getattr(trace, "abort_reason", None)
            react_trace_dict = _react_trace_to_dict(trace)
        except Exception as exc:
            logger.exception("Watchdog LLM ReAct failed: %s", exc)
            llm_abort = f"react_invocation_failed: {exc!s}"

    # ── Phase 4: dispatch notifications (skipped in dry_run by emit_notification) ─
    notification_result: dict = {
        "dashboard": False, "toast": False, "email": False, "halt_flag": False,
    }
    try:
        from engine.agents.ops_watchdog.notifications import emit_notification
        # Summary prefers LLM narrative when available; falls back to triage
        # decision string. Severe halt-flag reason needs to be human-readable.
        summary_text = (
            llm_final.strip()
            if llm_used and llm_final.strip()
            else (
                f"{triage_decision['severity'].upper()} — modes: "
                + ", ".join(triage_decision["modes_fired"])
                + f" ({len(findings)} findings)"
            )
        )
        notification_result = emit_notification(
            severity    = triage_decision["severity"],
            summary     = summary_text,
            findings    = findings,
            today_iso   = today.isoformat(),
            repair_info = auto_repair_summary,
            dry_run     = dry_run,
        )
        if verbose:
            logger.info("Watchdog notifications: %s", notification_result)
    except Exception as exc:
        logger.exception("Watchdog notification dispatch failed: %s", exc)
        notification_result["dispatch_error"] = str(exc)

    finished = datetime.datetime.utcnow()
    result = WatchdogRunResult(
        started_at_iso    = started.isoformat() + "Z",
        finished_at_iso   = finished.isoformat() + "Z",
        today_iso         = today.isoformat(),
        dry_run           = dry_run,
        audit_run_id      = audit_run_id,
        audit_exit_status = audit_summary.get("exit_status"),
        n_rules_run       = int(audit_summary.get("n_rules_run", 0) or 0),
        n_findings        = len(findings),
        findings_summary  = findings,
        triage            = triage_decision,
        auto_repair       = auto_repair_summary,
        notifications     = notification_result,
        llm_used          = llm_used,
        llm_cost_usd      = llm_cost_usd,
        llm_n_steps       = llm_n_steps,
        llm_final_answer  = llm_final,
        llm_abort_reason  = llm_abort,
        trace_json_path   = None,
    )

    # ── Phase 3: trace JSON persistence ──────────────────────────────────
    if save_trace:
        try:
            path = _save_trace_json(result, react_trace_dict)
            # Replace trace_json_path via dataclasses.replace (frozen)
            result = dataclasses.replace(result, trace_json_path=str(path))
            if verbose:
                logger.info("Watchdog trace saved: %s", path)
        except Exception as exc:
            logger.warning("Watchdog trace save failed: %s", exc)

    return result
