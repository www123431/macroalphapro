"""
engine/agents/ops_watchdog/auto_repair.py — Hardcoded auto-repair recipes.

Per spec §2.5 (LOCKED): mapping from mode_key → recipe function. LLM detection
of "this looks like mode X" triggers recipe X ONLY if X is in the table. LLM
never bypasses the mapping. Adding/removing modes requires spec amendment.

PHASE 3 SCOPE DEVIATION (2026-05-12, defer pending spec amendment 2):
  Spec §2.5 lists 6 auto-repair modes (1/2/4/6/10/12). Phase 3 actively
  implements 3 (modes 1/2/6 — transient/idempotent failures that resolve
  via run_daily_batch retry). Modes 4/10/12 are DEFERRED as `_stub_deferred`
  because the corresponding production-write paths (sleeve_id backfill,
  weight cap re-clamp, regime scale re-apply) lack surgical entry points
  in production modules AND auto-fixing those would mask the root-cause bug
  in the write path that produced the violation in the first place.
  Spec amendment 2 (planned Phase 6) will formally reduce hypothesis 2's
  "5 deterministic failure types" → "3 deterministic failure types" and
  reframe modes 4/10/12 as DETECT-only (escalate to PendingApproval, no
  auto-repair). See feedback_amendment_trial_cost_retired — kind=clarification
  +0 trials.

CRITICAL INVARIANTS (enforced by Tier R rule_watchdog_auto_repair_no_raw_sql):
  - Recipes call ONLY existing production functions (e.g.,
    engine.daily_batch.run_daily_batch). Recipes NEVER contain raw SQL
    targeting production tables (simulated_positions / simulated_trades /
    portfolio_nav_snapshots / universe_etfs).
  - Production writes happen INSIDE the production functions, which have
    their own invariant checks. Watchdog's responsibility is audit trail,
    not data integrity.
  - Each recipe attempt records to AuditProposal table (reusing existing
    R-1.C schema; LLM-specific fields zero-filled since this is
    recipe-driven, not LLM-generated).
  - Max 3 retries per recipe. After 3 fails → escalate to SEVERE (caller
    sets halt_next_batch flag + email).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Per spec §2.5 — max 3 attempts before escalation
MAX_RETRY_ATTEMPTS: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class RepairResult:
    """One auto-repair execution result for one AuditFinding."""
    mode_key:          str
    recipe_name:       str
    success:           bool                  # True iff at least one attempt succeeded
    deferred:          bool                  # True iff recipe is a deferred stub
    n_attempts:        int                   # how many tries made (≤ MAX_RETRY_ATTEMPTS)
    duration_sec:      float
    attempts_log:      list[dict]            # per-attempt {attempt, status, error, started_at, elapsed_s}
    error:             Optional[str]         # final error if !success
    audit_proposal_id: Optional[int]         # AuditProposal row id


# ─────────────────────────────────────────────────────────────────────────────
# Active recipes (Phase 3: modes 1 / 2 / 6 only)
#
# Each recipe takes the finding dict and returns (success: bool, detail: dict).
# Side effects (e.g. run_daily_batch invocation) go through existing production
# functions — recipes own NONE of the production-table SQL.
# ─────────────────────────────────────────────────────────────────────────────

def _repair_retry_idempotent_batch(finding: dict) -> tuple[bool, dict]:
    """
    Recipe for mode 1 (cycle silently failed).

    Calls existing production fn: engine.daily_batch.run_daily_batch(force=True).
    daily_batch handles its own idempotency + write-path correctness. Watchdog
    just triggers the retry; daily_batch decides what gets re-fetched/re-written.
    """
    try:
        from engine.daily_batch import run_daily_batch
    except Exception as exc:
        return False, {"phase": "import", "error": str(exc)}

    snapshot = finding.get("snapshot", {}) or {}
    issues   = snapshot.get("issues", []) or []
    # Re-run for today; downstream handles its own retry semantics.
    today    = datetime.date.today()
    started  = time.time()
    try:
        result = run_daily_batch(as_of_date=today, force=True)
    except Exception as exc:
        return False, {"phase": "run_daily_batch_exception",
                       "error": str(exc), "elapsed_s": time.time() - started}
    elapsed = time.time() - started

    skipped = getattr(result, "skipped", False)
    return True, {
        "phase":     "run_daily_batch_completed",
        "as_of":     today.isoformat(),
        "skipped":   bool(skipped),
        "elapsed_s": round(elapsed, 3),
        "trigger_issues": [i.get("kind") for i in issues][:5],
    }


def _repair_force_fresh_fetch(finding: dict) -> tuple[bool, dict]:
    """
    Recipe for mode 2 (yfinance stale data).

    Calls existing production fn: engine.daily_batch.run_daily_batch(force=True).
    daily_batch's data fetch path re-pulls yfinance for any stale tickers as
    a side effect of running step 1 (data_quality). No targeted refresh fn
    exists in production today; coarse re-run is the available lever.
    """
    try:
        from engine.daily_batch import run_daily_batch
    except Exception as exc:
        return False, {"phase": "import", "error": str(exc)}

    snapshot       = finding.get("snapshot", {}) or {}
    n_stale        = int(snapshot.get("n_stale", 0) or 0)
    n_missing      = int(snapshot.get("n_missing", 0) or 0)
    today          = datetime.date.today()
    started        = time.time()
    try:
        result = run_daily_batch(as_of_date=today, force=True)
    except Exception as exc:
        return False, {"phase": "run_daily_batch_exception",
                       "error": str(exc), "elapsed_s": time.time() - started}
    elapsed = time.time() - started

    return True, {
        "phase":        "run_daily_batch_completed",
        "as_of":        today.isoformat(),
        "skipped":      bool(getattr(result, "skipped", False)),
        "elapsed_s":    round(elapsed, 3),
        "n_stale_pre":  n_stale,
        "n_missing_pre": n_missing,
    }


def _repair_retry_execution_if_signal_active(finding: dict) -> tuple[bool, dict]:
    """
    Recipe for mode 6 (trade execution missing for active signal).

    Calls existing production fn: engine.daily_batch.run_daily_batch(force=True).
    Re-running idempotently regenerates trades from existing SignalRecord rows.
    """
    try:
        from engine.daily_batch import run_daily_batch
    except Exception as exc:
        return False, {"phase": "import", "error": str(exc)}

    snapshot   = finding.get("snapshot", {}) or {}
    n_orphans  = int(snapshot.get("n_orphans", 0) or 0)
    today      = datetime.date.today()
    started    = time.time()
    try:
        result = run_daily_batch(as_of_date=today, force=True)
    except Exception as exc:
        return False, {"phase": "run_daily_batch_exception",
                       "error": str(exc), "elapsed_s": time.time() - started}
    elapsed = time.time() - started

    return True, {
        "phase":          "run_daily_batch_completed",
        "as_of":          today.isoformat(),
        "skipped":        bool(getattr(result, "skipped", False)),
        "elapsed_s":      round(elapsed, 3),
        "n_orphans_pre":  n_orphans,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deferred stubs (Phase 3 Option A: modes 4 / 10 / 12)
#
# Reason: these modes signal ROOT-CAUSE BUGS in production write paths, not
# transient failures. Auto-fixing the symptom (sleeve_id, weight, scale) would
# mask the bug. Phase 6 spec amendment will formally remove them from the
# auto-repair table and reclassify as DETECT-only (escalate to PendingApproval).
# ─────────────────────────────────────────────────────────────────────────────

def _stub_deferred(finding: dict) -> tuple[bool, dict]:
    """No-op stub. Always returns False so caller escalates to SEVERE."""
    rule_name = finding.get("rule_name", "?")
    return False, {
        "phase":   "deferred",
        "reason":  (
            "Phase 3 Option A deferral: production module lacks surgical "
            "entry point for this mode AND auto-fixing the symptom would "
            "mask a root-cause write-path bug. Escalate to human review."
        ),
        "rule_name": rule_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOCKED mode → recipe mapping (spec §2.5)
# ─────────────────────────────────────────────────────────────────────────────

AUTO_REPAIR_RECIPES_LOCKED: dict[str, Callable[[dict], tuple[bool, dict]]] = {
    # Active (Phase 3)
    "mode_1_cycle_failed":             _repair_retry_idempotent_batch,
    "mode_2_yfinance_stale":           _repair_force_fresh_fetch,
    "mode_6_trade_execution_missing":  _repair_retry_execution_if_signal_active,
    # Deferred (Phase 3 Option A → Phase 6 spec amendment 2)
    "mode_4_sleeve_drift":             _stub_deferred,
    "mode_10_weight_cap_violation":    _stub_deferred,
    "mode_12_regime_scale_misapplied": _stub_deferred,
}

DEFERRED_MODES_LOCKED: frozenset[str] = frozenset({
    "mode_4_sleeve_drift",
    "mode_10_weight_cap_violation",
    "mode_12_regime_scale_misapplied",
})


# ─────────────────────────────────────────────────────────────────────────────
# Execution engine: retry + persistence + escalation signal
# ─────────────────────────────────────────────────────────────────────────────

def _write_audit_proposal(
    finding_id:  Optional[int],
    recipe_name: str,
    success:     bool,
    attempts:    list[dict],
) -> Optional[int]:
    """
    Persist auto-repair outcome to AuditProposal (R-1.C existing infrastructure).
    LLM-specific fields are zero-filled since this is recipe-driven, not LLM.
    Fail-soft: ledger write hiccup must not crash the recipe.
    """
    if finding_id is None:
        return None
    try:
        from engine.auto_audit_models import AuditProposal
        from engine.memory import SessionFactory
    except Exception as exc:
        logger.warning("auto_repair: AuditProposal import failed: %s", exc)
        return None

    payload = {
        "recipe_name": recipe_name,
        "n_attempts":  len(attempts),
        "success":     success,
        "attempts":    attempts,
    }
    try:
        with SessionFactory() as s:
            # finding_id has unique constraint in schema → check before insert
            existing = (s.query(AuditProposal)
                         .filter(AuditProposal.finding_id == finding_id)
                         .first())
            if existing is not None:
                # Update existing row (retry on a re-discovered finding)
                existing.generated_at        = datetime.datetime.utcnow()
                existing.generation_status   = "success" if success else "generation_failed"
                existing.failure_reason      = None if success else \
                    "auto_repair_retries_exhausted"
                existing.parsed_payload_json = json.dumps(payload, default=str)
                existing.gate_status         = "pass" if success else "fail"
                s.commit()
                return existing.id

            row = AuditProposal(
                finding_id          = finding_id,
                generated_at        = datetime.datetime.utcnow(),
                model_version       = "auto_repair_v1",
                prompt_hash         = recipe_name,
                input_tokens        = 0,
                output_tokens       = 0,
                cost_usd            = 0.0,
                raw_response_text   = json.dumps({"recipe": recipe_name}),
                parsed_payload_json = json.dumps(payload, default=str),
                generation_status   = "success" if success else "generation_failed",
                failure_reason      = None if success else "auto_repair_retries_exhausted",
                gate_status         = "pass" if success else "fail",
                governance_required = False,
            )
            s.add(row)
            s.commit()
            return row.id
    except Exception as exc:
        logger.warning("auto_repair: AuditProposal write failed for finding %s: %s",
                       finding_id, exc)
        return None


def _mark_finding_resolved(finding_id: Optional[int]) -> None:
    """Update AuditFinding.status='RESOLVED' on successful repair."""
    if finding_id is None:
        return
    try:
        from engine.auto_audit_models import AuditFinding
        from engine.memory import SessionFactory
        with SessionFactory() as s:
            row = s.query(AuditFinding).filter(AuditFinding.id == finding_id).first()
            if row is not None and row.status == "OPEN":
                row.status = "RESOLVED"
                s.commit()
    except Exception as exc:
        logger.warning("auto_repair: AuditFinding.status update failed for id=%s: %s",
                       finding_id, exc)


def execute_repair_for_finding(finding: dict) -> RepairResult:
    """
    Execute the auto-repair recipe for a single AuditFinding.

    Algorithm:
      1. Resolve mode_key from rule_name via triage.rule_name_to_mode.
      2. Look up recipe in AUTO_REPAIR_RECIPES_LOCKED (hardcoded, NOT LLM-decided).
      3. If absent / deferred → return RepairResult(deferred=True) — caller
         escalates as if 3 retries had failed.
      4. Else: run recipe up to MAX_RETRY_ATTEMPTS times. Stop on first success.
      5. Persist AuditProposal row (audit trail).
      6. On success: AuditFinding.status='RESOLVED'.

    Args:
        finding: dict from agent._collect_findings_for_run() — must contain
                 rule_name, finding_id (=AuditFinding.id), snapshot.

    Returns:
        RepairResult with full attempts_log.
    """
    from engine.agents.ops_watchdog.triage import rule_name_to_mode

    rule_name  = finding.get("rule_name", "?")
    finding_id = finding.get("finding_id")
    mode_key   = rule_name_to_mode(rule_name) or "unknown_mode"
    recipe     = AUTO_REPAIR_RECIPES_LOCKED.get(mode_key)

    started_at = time.time()
    attempts_log: list[dict] = []

    if recipe is None:
        return RepairResult(
            mode_key=mode_key, recipe_name="none",
            success=False, deferred=True, n_attempts=0,
            duration_sec=0.0, attempts_log=[],
            error="mode_not_in_recipe_table",
            audit_proposal_id=None,
        )

    recipe_name = getattr(recipe, "__name__", "anonymous")
    is_stub = (recipe is _stub_deferred) or (mode_key in DEFERRED_MODES_LOCKED)

    if is_stub:
        ok, detail = recipe(finding)
        attempts_log.append({
            "attempt":    1,
            "status":     "deferred",
            "detail":     detail,
            "elapsed_s":  0.0,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        })
        prop_id = _write_audit_proposal(finding_id, recipe_name,
                                        success=False, attempts=attempts_log)
        return RepairResult(
            mode_key=mode_key, recipe_name=recipe_name,
            success=False, deferred=True, n_attempts=1,
            duration_sec=time.time() - started_at,
            attempts_log=attempts_log,
            error="recipe_deferred",
            audit_proposal_id=prop_id,
        )

    last_error: Optional[str] = None
    success = False
    for attempt_idx in range(1, MAX_RETRY_ATTEMPTS + 1):
        attempt_started = time.time()
        try:
            ok, detail = recipe(finding)
        except Exception as exc:
            ok, detail = False, {"phase": "recipe_exception",
                                  "error": str(exc)}
        elapsed_s = time.time() - attempt_started
        attempts_log.append({
            "attempt":    attempt_idx,
            "status":     "success" if ok else "failed",
            "detail":     detail,
            "elapsed_s":  round(elapsed_s, 3),
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        })
        if ok:
            success = True
            break
        last_error = detail.get("error") if isinstance(detail, dict) else None

    prop_id = _write_audit_proposal(finding_id, recipe_name,
                                    success=success, attempts=attempts_log)

    if success:
        _mark_finding_resolved(finding_id)

    return RepairResult(
        mode_key=mode_key, recipe_name=recipe_name,
        success=success, deferred=False, n_attempts=len(attempts_log),
        duration_sec=round(time.time() - started_at, 3),
        attempts_log=attempts_log,
        error=last_error if not success else None,
        audit_proposal_id=prop_id,
    )


def execute_repairs_for_findings(findings: list[dict]) -> list[RepairResult]:
    """
    Convenience: iterate `findings` and dispatch each whose mode is in the
    recipe table. Findings whose mode is unknown (not a Watchdog mode) are
    skipped silently — they're not in scope for auto-repair.
    """
    from engine.agents.ops_watchdog.triage import rule_name_to_mode

    out: list[RepairResult] = []
    for f in findings:
        mode_key = rule_name_to_mode(f.get("rule_name", "")) or ""
        if mode_key not in AUTO_REPAIR_RECIPES_LOCKED:
            continue
        out.append(execute_repair_for_finding(f))
    return out
