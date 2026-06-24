"""
engine/auto_audit.py — Orchestrator for the Auto-Audit Loop (R-1.A 2026-05-06)

Tier R rationale (docs/sprint_2026_05_thesis_grade.md):
The TSMOM → QL01-BAB migration was reactive to supervisor challenge, not
proactive system detection. This loop closes the gap: deterministic rules
detect contradictions, an LLM (R-1.C) drafts proposals, a deterministic
safety gate (R-1.D) validates them, and approved proposals flow into the
existing PendingApproval queue (R-1.E).

R-1.A scope (this file): cron-callable orchestrator + persistence.
LLM proposing and safety gating arrive in R-1.C / R-1.D.
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from typing import Any, Dict, List, Literal

from engine.auto_audit_models import AuditFinding, AuditRun
from engine.auto_audit_rules import CRITICAL_RULES, WATCHDOG_RULES, WEEKLY_RULES
from engine.memory import SessionFactory, init_db

logger = logging.getLogger(__name__)

# 2026-05-12 (Watchdog Phase 2): "watchdog" scope added — runs WATCHDOG_RULES
# at 06:10 SGT daily, 10min after MacroAlphaPro_DailyBatch's "critical" run.
Scope = Literal["critical", "weekly", "watchdog"]


def _select_rules(scope: Scope) -> List:
    if scope == "critical":
        return list(CRITICAL_RULES)
    if scope == "weekly":
        return list(WEEKLY_RULES)
    if scope == "watchdog":
        return list(WATCHDOG_RULES)
    raise ValueError(f"Unknown audit scope: {scope!r}")


def _classify_exit(n_rules: int, n_errors: int) -> str:
    if n_rules == 0:
        # No rules registered for this scope (only WEEKLY before R-1.B.3 lands
        # the remaining 5 weekly rules). Distinguishable from a real "ok" run
        # where rules ran cleanly.
        return "no_rules"
    if n_errors == 0:
        return "ok"
    if n_errors < n_rules:
        return "partial"
    return "error"


def run_audit(scope: Scope) -> Dict[str, Any]:
    """
    Execute every rule registered for `scope`, persist findings, return summary.

    Returns dict with:
      run_id, scope, n_rules_run, n_findings, n_errors, duration_sec, exit_status

    Errors in individual rules are caught + logged + counted; one bad rule
    does not abort the run. This matters for cron — a transient failure in
    one check should not silence every other check.
    """
    init_db()
    rules = _select_rules(scope)
    t0 = time.time()

    findings: List[Dict[str, Any]] = []
    n_errors = 0
    for rule in rules:
        rule_name = getattr(rule, "__name__", repr(rule))
        try:
            result = rule()
        except Exception:
            n_errors += 1
            logger.exception("auto_audit rule '%s' raised", rule_name)
            continue
        if result is None:
            continue
        # Defensive: ensure shape is what we expect
        if not isinstance(result, dict) or "severity" not in result:
            n_errors += 1
            logger.error("auto_audit rule '%s' returned malformed result: %r", rule_name, result)
            continue
        result.setdefault("rule_name", rule_name)
        findings.append(result)

    duration = time.time() - t0
    exit_status = _classify_exit(len(rules), n_errors)

    # Silenceable mechanism (R-1.B.3, 2026-05-06):
    # If a finding's rule_name has an IGNORED predecessor in the last 30 days,
    # suppress it. Supervisor's "ignore" act is a 30-day shut-up token per rule.
    # Per-rule granularity is the simple version; per-finding-signature can come
    # later if a rule produces heterogeneous findings worth distinguishing.
    SILENCE_LOOKBACK_DAYS = 30

    with SessionFactory() as session:
        ignored_cutoff = datetime.datetime.utcnow() - datetime.timedelta(
            days=SILENCE_LOOKBACK_DAYS,
        )
        ignored_rules = {
            row[0] for row in session.query(AuditFinding.rule_name)
            .filter(AuditFinding.status == "IGNORED")
            .filter(AuditFinding.detected_at >= ignored_cutoff)
            .distinct()
            .all()
        }

        kept_findings: List[Dict[str, Any]] = []
        n_suppressed = 0
        for f in findings:
            if f["rule_name"] in ignored_rules:
                n_suppressed += 1
                continue
            kept_findings.append(f)

        run = AuditRun(
            run_at=datetime.datetime.utcnow(),
            scope=scope,
            n_rules_run=len(rules),
            n_findings=len(kept_findings),
            n_errors=n_errors,
            n_suppressed=n_suppressed,
            duration_sec=duration,
            exit_status=exit_status,
        )
        session.add(run)
        session.flush()

        for f in kept_findings:
            session.add(AuditFinding(
                run_id        = run.id,
                rule_name     = f["rule_name"],
                severity      = f["severity"],
                snapshot_json = json.dumps(f.get("snapshot", {}), default=str),
                status        = "OPEN",
            ))
        session.commit()

        return {
            "run_id":       run.id,
            "scope":        scope,
            "n_rules_run":  len(rules),
            "n_findings":   len(kept_findings),
            "n_errors":     n_errors,
            "n_suppressed": n_suppressed,
            "duration_sec": round(duration, 3),
            "exit_status":  exit_status,
        }
