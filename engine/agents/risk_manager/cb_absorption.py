"""
engine/agents/risk_manager/cb_absorption.py — Phase 5 circuit breaker absorption.

Senior design pattern: NON-DISRUPTIVE shadow/overlay.

Background
----------
The existing engine.circuit_breaker module has 13 importing call sites
(pages/risk_console.py, app.py, engine.daily_batch, ops_watchdog, etc.).
Replacing it wholesale would require updating all 13 simultaneously, with
significant blast radius.

Instead this module ADDS a `unified_circuit_state(as_of)` function that:
  1. Calls the legacy `engine.circuit_breaker.evaluate(as_of)` unchanged
     (preserves VIX-spike + quota-pressure + persistent-SEVERE semantics)
  2. ALSO queries the RiskManagerAlert table (Phase 4) for today's worst
     `cb_severity` value
  3. Returns the MORE SEVERE of the two as a CircuitBreakerState

Pattern reference: this is the "Strangler Fig" + "Adapter" pattern from
Martin Fowler — the new system wraps and shadows the legacy without
modifying its internals. Real institutions use this to migrate
production trading systems over weeks/months without a "big bang" cut.

G4 parity invariant (verified by Phase 9 test)
----------------------------------------------
When no RiskManagerAlert rows exist for the date (e.g. day before the
Risk Manager Agent went live), `unified_circuit_state(d)` returns a
CircuitBreakerState whose FIELDS (level / reason / triggered_at /
auto_reset / vix_today / vix_prev / quota_frac) are equal to those of
legacy `evaluate(d)`. Object identity is NOT preserved (legacy
evaluate() constructs a new instance per call) — field equality is
the contract.

This guarantees Risk Console + Watchdog + daily_batch behavior is
byte-identical on dates pre-dating Risk Manager deployment.

Migration sequence (NOT this commit)
------------------------------------
After Phase 6 orchestrator hook lands and Phase 9 G4 test passes for
≥90 historical days, individual callers can OPTIONALLY migrate from
`engine.circuit_breaker.evaluate` → `unified_circuit_state` one at a
time. Risk Console (most important consumer) is the natural first to
migrate; daily_batch and ops_watchdog can stay on legacy.

The legacy `evaluate` itself is NEVER deleted — it remains the source
of truth for VIX/quota signals, which the unified function consumes.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Mapping from RiskManagerAlert.cb_severity → engine.circuit_breaker LEVEL_*.
# Identity mapping but explicit so a future scheme change in either side
# can adapt without renaming everywhere.
_RM_SEV_TO_CB_LEVEL = {
    "NONE":   "none",     # engine.circuit_breaker.LEVEL_NONE
    "LIGHT":  "light",    # LEVEL_LIGHT
    "MEDIUM": "medium",   # LEVEL_MEDIUM
    "SEVERE": "severe",   # LEVEL_SEVERE
}

# Rank order — must match engine.circuit_breaker._LEVEL_RANK.
_LEVEL_RANK = {"none": 0, "light": 1, "medium": 2, "severe": 3}


def _query_risk_manager_worst_today(as_of: datetime.date):
    """Read tool: return today's worst RiskManagerAlert as a CircuitBreakerState-shape dict.

    Returns dict with same field names as CircuitBreakerState so the caller
    can pick max severity by simple dict comparison. None if no alerts.
    """
    from engine.db_models import RiskManagerAlert
    from engine.memory import SessionFactory

    sess = SessionFactory()
    try:
        rows = (
            sess.query(RiskManagerAlert)
            .filter(RiskManagerAlert.date == as_of)
            .order_by(RiskManagerAlert.generated_at_utc.desc())
            .all()
        )
        if not rows:
            return None
        # Find worst severity
        worst_rank = -1
        worst_row = None
        for r in rows:
            rank = _LEVEL_RANK.get(_RM_SEV_TO_CB_LEVEL.get(r.cb_severity, "none"), 0)
            if rank > worst_rank:
                worst_rank = rank
                worst_row = r
        if worst_row is None:
            return None
        return {
            "level":         _RM_SEV_TO_CB_LEVEL[worst_row.cb_severity],
            "reason":        f"risk_manager mode {worst_row.mode_id}: "
                             f"{worst_row.rule_description[:280]}",
            "triggered_at":  (worst_row.generated_at_utc.isoformat() + "Z"
                              if worst_row.generated_at_utc else None),
            "auto_reset":    _RM_SEV_TO_CB_LEVEL[worst_row.cb_severity] != "severe",
            "source_table":  "RiskManagerAlert",
            "worst_alert_id": worst_row.alert_id,
        }
    finally:
        sess.close()


def unified_circuit_state(as_of: Optional[datetime.date] = None):
    """Return the more severe of (legacy circuit_breaker.evaluate, Risk Manager today).

    G4 parity: when no RiskManagerAlert exists for the date, the return value
    is the legacy CircuitBreakerState verbatim (not a copy — the same object).
    """
    from engine.circuit_breaker import (
        CircuitBreakerState, evaluate as _legacy_evaluate,
        LEVEL_SEVERE,
    )

    if as_of is None:
        as_of = datetime.date.today()

    legacy_state = _legacy_evaluate(as_of)
    rm_summary = _query_risk_manager_worst_today(as_of)

    if rm_summary is None:
        # G4 parity case — return legacy state object reference (field-equal
        # to a fresh `evaluate(as_of)` call; identity preserved within this
        # function but NOT across separate unified_circuit_state invocations).
        return legacy_state

    legacy_rank = _LEVEL_RANK.get(legacy_state.level, 0)
    rm_rank     = _LEVEL_RANK.get(rm_summary["level"],     0)

    if legacy_rank >= rm_rank:
        # Legacy dominates (or ties — tie goes to legacy for stability)
        return legacy_state

    # Risk Manager dominates → wrap as CircuitBreakerState.
    # NOTE: this function is READ-ONLY. It does NOT persist SEVERE to the
    # legacy CB state file. Callers (orchestrator Phase 6) that want
    # persistence MUST explicitly call
    # engine.circuit_breaker.set_external_halt_flag(reason, source='risk_manager').
    # Separation of concerns: read functions are pure, write functions
    # are explicit.
    return CircuitBreakerState(
        level        = rm_summary["level"],
        reason       = rm_summary["reason"],
        triggered_at = rm_summary["triggered_at"],
        auto_reset   = rm_summary["auto_reset"],
    )


def persist_risk_manager_severe(
    as_of:    Optional[datetime.date] = None,
    source:   str = "risk_manager",
) -> Optional[object]:
    """Explicit write: if Risk Manager has SEVERE alerts today, persist via
    the legacy circuit_breaker.set_external_halt_flag mechanism.

    Returns the CircuitBreakerState that was persisted, or None if no
    SEVERE-level RM alert exists today.

    Idempotency: set_external_halt_flag is itself idempotent — calling it
    multiple times with the same source/reason replaces the row in place
    (atomic file write under _lock in engine.circuit_breaker).

    Phase 6 orchestrator calls this AFTER the pre-trade gate decides halt:
        gate_breaches = evaluate_all_modes(...)
        halt = any_hard_halt(gate_breaches)
        if halt:
            persist_risk_manager_severe(today, source='risk_manager_pre_trade')
    """
    if as_of is None:
        as_of = datetime.date.today()
    rm_summary = _query_risk_manager_worst_today(as_of)
    if rm_summary is None or rm_summary["level"] != "severe":
        return None
    from engine.circuit_breaker import set_external_halt_flag
    return set_external_halt_flag(
        reason = rm_summary["reason"],
        source = source,
    )


def get_circuit_state_breakdown(as_of: Optional[datetime.date] = None) -> dict:
    """Diagnostic / dashboard helper — return both legacy and RM state separately
    so the Risk Console can show 'CB state = SEVERE (source: Risk Manager Mode 5
    HHI breach)' without losing the legacy VIX/quota state.
    """
    from engine.circuit_breaker import evaluate as _legacy_evaluate

    if as_of is None:
        as_of = datetime.date.today()

    legacy_state = _legacy_evaluate(as_of)
    rm_summary = _query_risk_manager_worst_today(as_of)
    unified = unified_circuit_state(as_of)

    return {
        "legacy": {
            "level":        legacy_state.level,
            "reason":       legacy_state.reason,
            "vix_today":    legacy_state.vix_today,
            "vix_prev":     legacy_state.vix_prev,
            "quota_frac":   legacy_state.quota_frac,
        },
        "risk_manager": rm_summary,
        "unified": {
            "level":  unified.level,
            "reason": unified.reason,
            "source": "legacy" if rm_summary is None
                      or _LEVEL_RANK[legacy_state.level] >= _LEVEL_RANK[rm_summary["level"]]
                      else "risk_manager",
        },
    }
