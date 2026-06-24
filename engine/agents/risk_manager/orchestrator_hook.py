"""
engine/agents/risk_manager/orchestrator_hook.py — Phase 6 daily-cycle integration.

Wires Risk Manager into the daily orchestrator at two points per spec §2.3:

  06:02  PRE-TRADE GATE — runs after run_paper_trade_day produces the
                          combined book but BEFORE persist_run_to_db.
                          HARD HALT modes block persistence.
  06:04  POST-TRADE GATE — runs AFTER persist_run_to_db succeeds.
                          Soft-warn only; never halts (book already saved).

Both hooks are pure functions: input = PaperTradeRunResult, output =
RiskManagerRunResult. Persistence side-effects (RiskManagerAlert writes,
legacy circuit-breaker set_external_halt_flag) are gated by the
`dry_run` parameter so the same code path can be used in unit tests.

Senior design choice (mid-build self-audit per [[feedback-iterative-
self-correction]] proactive rule):
  Pre-trade vs post-trade differ in halt semantics, NOT in the gate set.
  Both call evaluate_all_modes(...) — same 12 detectors. Pre-trade's
  halt_decision = any_hard_halt(breaches). Post-trade's halt_decision =
  False ALWAYS (book is already persisted; halting here would be too
  late). This keeps the gate library uniform and the phase-specific
  behavior in the hook layer where it belongs.

VaR/ES computation is OPTIONAL and degraded-gracefully: if the
synthesize_portfolio_returns call fails (insufficient history, yfinance
network failure, malformed positions), modes 6/6b/7/7b are no-ops and
the other 8 modes still run. This matches spec §2.1 "VaR/ES alerts are
SOFT because the metric is counterfactual" — pre-trade can still HARD
HALT on modes 1/3/4/5/9 which are cheap deterministic checks on
in-memory state.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd
    from engine.portfolio.paper_trade_combined import PaperTradeRunResult

from engine.agents.risk_manager.agent import RiskManagerRunResult
from engine.agents.risk_manager.gates import (
    evaluate_all_modes,
    any_hard_halt,
    classify_severity,
)
from engine.agents.risk_manager.persist import persist_breaches_to_db
from engine.agents.risk_manager.cb_absorption import persist_risk_manager_severe

logger = logging.getLogger(__name__)


HALT_FLAG_DIR = Path("data/risk_manager/halts")


# ──────────────────────────────────────────────────────────────────────────────
# VaR / ES computation helper (optional, degraded-gracefully)
# ──────────────────────────────────────────────────────────────────────────────
def _compute_var_es_optional(
    combined_book: "pd.Series",
) -> tuple[Optional[float], Optional[float]]:
    """Run synthesize_portfolio_returns + compute_var_block; return
    (var_95_historical, es_95_historical). On any failure return (None, None);
    callers must handle None as "VaR modes skipped, not error".
    """
    try:
        import pandas as pd
        from engine.risk_metrics import (
            synthesize_portfolio_returns, compute_var_block,
        )
        positions_df = pd.DataFrame({
            "ticker":        list(combined_book.index),
            "actual_weight": list(combined_book.values),
        })
        port_ret, _rets, meta = synthesize_portfolio_returns(positions_df)
        if meta.get("insufficient", True) or len(port_ret) < 30:
            return (None, None)
        vb = compute_var_block(port_ret, alpha=0.05)
        return (
            None if vb.historical != vb.historical else float(vb.historical),     # NaN guard
            None if vb.es_historical != vb.es_historical else float(vb.es_historical),
        )
    except Exception as exc:
        logger.warning(
            "risk_manager.orchestrator_hook: VaR/ES skipped (modes 6/6b/7/7b no-op): %s",
            exc,
        )
        return (None, None)


# ──────────────────────────────────────────────────────────────────────────────
# Helper — resolve current Risk Manager spec hash from SpecRegistry.
# Reads the live DB row at write time so the marker tracks amendments
# automatically; no literal hash strings pinned in source (those would
# create a fixed-point bug — see thresholds.py governance_log header).
# ──────────────────────────────────────────────────────────────────────────────
def _current_spec_hash_short() -> str:
    try:
        from engine.preregistration import list_specs
        for row in list_specs():
            if int(row.get("id", -1)) == 69:
                h = row.get("current_hash") or ""
                return h[:8] if h else "unknown"
    except Exception as exc:
        logger.warning("orchestrator_hook: could not resolve spec 69 hash: %s", exc)
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Helper — write _HALT.json marker file for downstream visibility
# ──────────────────────────────────────────────────────────────────────────────
def _write_halt_marker(result: "PaperTradeRunResult", breaches: list, severity: str) -> Path:
    """Atomic-ish write of a _HALT.json marker so downstream consumers
    (Streamlit dashboards, Watchdog, ops monitoring) can detect that
    today's run was rejected at the Risk Manager pre-trade gate.
    """
    HALT_FLAG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of":            result.as_of.isoformat(),
        "phase":            "pre_trade",
        "halt_decision":    True,
        "severity":         severity,
        "n_breaches":       len(breaches),
        "hard_halt_modes":  sorted({b.mode_id for b in breaches if b.severity == "HARD_HALT"}),
        "first_breach":     {
            "mode_id":          breaches[0].mode_id,
            "rule_description": breaches[0].rule_description,
            "affected":         list(breaches[0].affected),
        } if breaches else None,
        "spec_id":          69,
        "spec_hash_short":  _current_spec_hash_short(),
        "written_at_utc":   datetime.datetime.utcnow().isoformat() + "Z",
    }
    path = HALT_FLAG_DIR / f"{result.as_of.isoformat()}_HALT.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# PRE-TRADE GATE
# ──────────────────────────────────────────────────────────────────────────────
def pre_trade_gate(
    result:       "PaperTradeRunResult",
    *,
    compute_var:  bool = True,
    dry_run:      bool = False,
) -> RiskManagerRunResult:
    """Pre-trade gate. HARD HALT modes can block subsequent persistence.

    Args:
      result:       output of run_paper_trade_day (combined book + signals)
      compute_var:  if False, skip VaR/ES computation (modes 6/6b/7/7b no-op).
                    Useful for unit tests that don't have yfinance access.
      dry_run:      if True, do NOT persist alerts and do NOT call
                    persist_risk_manager_severe. Pure compute path.

    Returns RiskManagerRunResult with halt=True iff any HARD HALT breach.
    """
    from engine.strategies import get_registry

    started = datetime.datetime.utcnow()
    registry = get_registry()
    sleeve_target = registry.sleeve_allocation_dict()

    var_95, es_95 = (None, None)
    if compute_var:
        var_95, es_95 = _compute_var_es_optional(result.combined_portfolio)

    breaches = evaluate_all_modes(
        combined           = result.combined_portfolio,
        signals            = result.signals,
        sleeve_attribution = result.sleeve_attribution,
        sleeve_target      = sleeve_target,
        registry           = registry,
        var_95_historical  = var_95,
        es_95_historical   = es_95,
    )
    halt   = any_hard_halt(breaches)
    cb_sev = classify_severity(breaches)

    audit_ids: tuple[int, ...] = ()
    if not dry_run and breaches:
        alert_ids = persist_breaches_to_db(
            breaches, result.as_of, phase="pre_trade", halt_decision=halt,
        )
        audit_ids = tuple(alert_ids)
        if halt:
            persist_risk_manager_severe(
                result.as_of, source="risk_manager_pre_trade",
            )
            _write_halt_marker(result, breaches, cb_sev)
            logger.error(
                "RISK MANAGER PRE-TRADE HALT — %d breaches (%s); HARD_HALT modes: %s",
                len(breaches), cb_sev,
                sorted({b.mode_id for b in breaches if b.severity == "HARD_HALT"}),
            )

    finished = datetime.datetime.utcnow()
    return RiskManagerRunResult(
        started_at_iso    = started.isoformat(),
        finished_at_iso   = finished.isoformat(),
        today_iso         = result.as_of.isoformat(),
        phase             = "pre_trade",
        dry_run           = dry_run,
        n_modes_evaluated = 12,
        breaches          = tuple(breaches),
        halt              = halt,
        severity          = cb_sev,
        narratives        = (),       # populated by Phase 7 narrator if invoked
        llm_cost_usd      = 0.0,
        audit_alert_ids   = audit_ids,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST-TRADE GATE
# ──────────────────────────────────────────────────────────────────────────────
def post_trade_gate(
    result:       "PaperTradeRunResult",
    *,
    compute_var:  bool = True,
    dry_run:      bool = False,
) -> RiskManagerRunResult:
    """Post-trade gate. Never halts (book already persisted).

    Same 12 detectors as pre-trade run AGAIN against the now-persisted
    state. Any HARD HALT breach is logged loudly because it means the
    book made it through pre-trade but became dangerous after the
    ETF-Holdings cap overlay (paper_trade_combined step 3b) or some
    other post-aggregate transform.

    Returns RiskManagerRunResult with halt=False ALWAYS.
    """
    from engine.strategies import get_registry

    started = datetime.datetime.utcnow()
    registry = get_registry()
    sleeve_target = registry.sleeve_allocation_dict()

    var_95, es_95 = (None, None)
    if compute_var:
        var_95, es_95 = _compute_var_es_optional(result.combined_portfolio)

    breaches = evaluate_all_modes(
        combined           = result.combined_portfolio,
        signals            = result.signals,
        sleeve_attribution = result.sleeve_attribution,
        sleeve_target      = sleeve_target,
        registry           = registry,
        var_95_historical  = var_95,
        es_95_historical   = es_95,
    )
    cb_sev = classify_severity(breaches)

    # Post-trade NEVER halts — book is already persisted.
    halt = False

    # But log any HARD HALT loudly — it indicates the ETF Holdings cap overlay
    # or another post-aggregate transform pushed the book into HARD HALT
    # territory AFTER pre-trade passed.
    hard_halt_breaches = [b for b in breaches if b.severity == "HARD_HALT"]
    if hard_halt_breaches:
        logger.warning(
            "RISK MANAGER POST-TRADE — %d HARD_HALT breaches detected on persisted state; "
            "book is already saved. Investigate cap-overlay step or transient data. Modes: %s",
            len(hard_halt_breaches),
            sorted({b.mode_id for b in hard_halt_breaches}),
        )

    audit_ids: tuple[int, ...] = ()
    if not dry_run and breaches:
        alert_ids = persist_breaches_to_db(
            breaches, result.as_of, phase="post_trade", halt_decision=False,
        )
        audit_ids = tuple(alert_ids)

    finished = datetime.datetime.utcnow()
    return RiskManagerRunResult(
        started_at_iso    = started.isoformat(),
        finished_at_iso   = finished.isoformat(),
        today_iso         = result.as_of.isoformat(),
        phase             = "post_trade",
        dry_run           = dry_run,
        n_modes_evaluated = 12,
        breaches          = tuple(breaches),
        halt              = halt,
        severity          = cb_sev,
        narratives        = (),
        llm_cost_usd      = 0.0,
        audit_alert_ids   = audit_ids,
    )
