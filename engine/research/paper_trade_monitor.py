"""engine/research/paper_trade_monitor.py — SLM Phase 2 integration:
monthly tick that evaluates a PAPER_TRADE sleeve against its sequential
boundary and writes the role_specific_evidence_passed evidence into
the state store so PAPER_TRADE → SHADOW transitions can be authorized.

Workflow (called by a cron at month-end OR ad-hoc by deploy script):

  1. Read all sleeves in PAPER_TRADE state from the state store
  2. For each:
     a. Compute months_in_paper_trade = (today - paper_trade_started)
     b. Load trailing-window returns from the sleeve (via registry)
     c. Load auxiliary returns if role needs them (book/risk_source/baseline)
     d. evaluate_role_specific_metric(role, sleeve_returns, ...)
     e. boundary.decide(observed_t, m=months_in_paper_trade)
     f. Record the BoundaryResult + RoleMetricResult in a sleeve-
        specific paper_trade_log entry (data/research/paper_trade_log.jsonl)
     g. If decision==ACCEPT: this sleeve is eligible for transition —
        emit a hint that the human reviewer can run scripts/promote_to_shadow.py
     h. If decision==REJECT: auto-transition PAPER_TRADE → REJECTED
        (early-stop loss path is automatic; early-stop ACCEPT requires
        human confirmation per the 3-gate doctrine)

Doctrine:
  - The monitor does NOT auto-promote to SHADOW. ACCEPT is a SIGNAL
    for human review, not an authorization. ALLOCATE gate requires
    explicit human action per Promote/Wire/Allocate separation.
  - REJECT is automatic because early-stop loss saves money + is the
    safe side of the asymmetric cost.
  - All decisions logged to JSONL for audit + post-mortem (which
    sleeves were ACCEPTed early vs which dragged to terminal CONTINUE).

Pre-Phase-3 limitation: the monitor only knows about PAPER_TRADE state.
Phase 3 (capital ramp) extends to SHADOW state monitoring (different
boundary, different metric thresholds).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.research.role_specific_metric_eval import (
    RoleMetricResult, evaluate_role_specific_metric,
)
from engine.research.sequential_testing import (
    BoundaryResult, SequentialDecision,
    default_obf_boundary_paper_trade,
)
from engine.research.sleeve_registry import get_sleeve
from engine.research.strategy_lifecycle import (
    GateNotMetError, SleeveRole, StrategyState,
)
from engine.research.strategy_state_store import (
    DEFAULT_DB_PATH, get_strategy, list_strategies, transition,
)
from engine.research.three_layer_validator import (
    ThreeLayerDecision, ThreeLayerResult, evaluate_three_layer,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_TRADE_LOG = REPO_ROOT / "data" / "research" / "paper_trade_log.jsonl"


@dataclass
class MonitorTickResult:
    """Output of one tick() call for one sleeve.

    Post-Phase-2.5: includes three_layer_result for Sharpe-based roles
    (alpha_seeker / risk_premium_harvester). For non-Sharpe roles
    (insurance / diversifier / regime_overlay) the role metric eval
    remains primary and three_layer_result is None.
    """

    strategy_id: str
    role: Optional[SleeveRole]
    months_observed: int
    metric_result: Optional[RoleMetricResult]
    boundary_result: Optional[BoundaryResult]
    three_layer_result: Optional[ThreeLayerResult] = None
    action_taken: str = ""
    error: Optional[str] = None


def _append_log(entry: dict) -> None:
    PAPER_TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_TRADE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _months_between(start: _dt.datetime, end: _dt.datetime) -> int:
    """Approximate calendar-month delta between two timestamps.
    Used to determine which interim look number we are on."""
    delta_days = (end - start).total_seconds() / (24 * 3600)
    return max(1, int(delta_days // 30))


def tick_single_sleeve(
    *,
    strategy_id: str,
    role: SleeveRole,
    today: Optional[_dt.datetime] = None,
    book_returns: Optional[pd.Series] = None,
    risk_source_returns: Optional[pd.Series] = None,
    static_baseline_returns: Optional[pd.Series] = None,
    prior_mean_sharpe: Optional[float] = None,
    family: Optional[str] = None,
    n_trials_across_research: Optional[int] = None,
    auto_reject_on_early_stop_loss: bool = True,
    actor: str = "paper_trade_monitor",
    db_path: Path = DEFAULT_DB_PATH,
) -> MonitorTickResult:
    """Evaluate one PAPER_TRADE sleeve against its sequential boundary.

    Parameters:
      role: explicit role passed because the state store does not store
            role (role lives in factor_exposure.proposed_role of the
            library YAML — caller resolves via load_audit_blocks_from_yaml
            and passes here)
      today: defaults to UTC now
      book/risk_source/static_baseline returns: optional auxiliary
        series needed by non-Sharpe roles
      auto_reject_on_early_stop_loss: if True, REJECT decisions trigger
        an automatic state transition PAPER_TRADE → REJECTED (recommended)
    """
    today = today or _dt.datetime.now(_dt.timezone.utc)
    record = get_strategy(strategy_id, db_path=db_path)
    if record.current_state != StrategyState.PAPER_TRADE:
        return MonitorTickResult(
            strategy_id=strategy_id, role=role,
            months_observed=0, metric_result=None, boundary_result=None,
            error=f"sleeve not in PAPER_TRADE (is {record.current_state.value})",
        )
    if record.paper_trade_started is None:
        return MonitorTickResult(
            strategy_id=strategy_id, role=role,
            months_observed=0, metric_result=None, boundary_result=None,
            error="paper_trade_started timestamp missing in state row",
        )

    m = _months_between(record.paper_trade_started, today)

    # Load sleeve returns + trim to paper-trade window
    try:
        sleeve = get_sleeve(strategy_id)
    except KeyError as exc:
        return MonitorTickResult(
            strategy_id=strategy_id, role=role,
            months_observed=m, metric_result=None, boundary_result=None,
            error=f"sleeve not registered: {exc}",
        )
    full_returns = sleeve.returns()
    # Convert paper_trade_started to tz-naive for index alignment
    pt_start = record.paper_trade_started
    if pt_start.tzinfo is not None:
        pt_start = pt_start.replace(tzinfo=None)
    trailing = full_returns[full_returns.index >= pt_start]
    if trailing.empty:
        # No observed returns yet (paper trade started today, returns
        # series not extended) — treat as INSUFFICIENT
        trailing = full_returns.tail(min(m, len(full_returns)))

    # Compute role-specific metric (always — primary for non-Sharpe roles)
    metric = evaluate_role_specific_metric(
        role=role,
        sleeve_returns=trailing,
        book_returns=book_returns,
        risk_source_returns=risk_source_returns,
        static_baseline_returns=static_baseline_returns,
    )

    # Apply OBF boundary (kept for backward compat + non-Sharpe roles)
    boundary = default_obf_boundary_paper_trade()
    b_result: Optional[BoundaryResult]
    try:
        b_result = boundary.decide(observed_t=metric.t_stat, m=m)
    except ValueError as exc:
        # Trial exceeded total_months — REJECT (drift past planned window)
        return MonitorTickResult(
            strategy_id=strategy_id, role=role, months_observed=m,
            metric_result=metric, boundary_result=None,
            error=f"paper trade exceeded planned window: {exc}",
        )

    # Phase 2.5: for Sharpe-based roles, ALSO run the 3-layer validator
    # (Bayesian + DeflSR + OBF). The composite decision OVERRIDES the
    # OBF-only b_result for these roles. Non-Sharpe roles continue using
    # the role-specific metric evaluator alone (3-layer is Sharpe-centric).
    three_layer_result: Optional[ThreeLayerResult] = None
    composite_decision = b_result.decision  # default to OBF-only
    if role in (SleeveRole.ALPHA_SEEKER, SleeveRole.RISK_PREMIUM_HARVESTER):
        prior_for_layer1 = prior_mean_sharpe if prior_mean_sharpe is not None else 1.0
        three_layer_result = evaluate_three_layer(
            sleeve_returns=trailing,
            prior_mean_sharpe=prior_for_layer1,
            family=family,
            n_trials_across_research=n_trials_across_research,
            obf_boundary=boundary,
            obf_month=m,
        )
        # Map composite ThreeLayerDecision back to SequentialDecision
        # for the existing log + transition logic
        composite_map = {
            ThreeLayerDecision.ACCEPT:       SequentialDecision.ACCEPT,
            ThreeLayerDecision.REJECT:       SequentialDecision.REJECT,
            ThreeLayerDecision.CONTINUE:     SequentialDecision.CONTINUE,
            ThreeLayerDecision.INSUFFICIENT: SequentialDecision.INSUFFICIENT,
        }
        composite_decision = composite_map[three_layer_result.final_decision]
        # Replace b_result with synthesized composite for downstream logic
        b_result = BoundaryResult(
            month=m,
            observed_t=metric.t_stat,
            upper_critical_t=b_result.upper_critical_t,
            lower_critical_t=b_result.lower_critical_t,
            decision=composite_decision,
            rationale=three_layer_result.rationale,
        )

    # Persist log entry
    _append_log({
        "ts": today.isoformat(),
        "strategy_id": strategy_id,
        "role": role.value,
        "months_observed": m,
        "metric_name": metric.metric_name,
        "metric_value": metric.metric_value,
        "t_stat": metric.t_stat,
        "n_observations": metric.n_observations,
        "decision": b_result.decision.value,
        "upper_critical_t": b_result.upper_critical_t,
        "lower_critical_t": b_result.lower_critical_t,
        "rationale": b_result.rationale,
        "evidence_passed_minimum": metric.evidence_passed,
    })

    # Take action based on decision
    action = ""
    if b_result.decision == SequentialDecision.REJECT and auto_reject_on_early_stop_loss:
        try:
            transition(
                strategy_id=strategy_id,
                to_state=StrategyState.REJECTED,
                actor=actor,
                reason=(f"Auto-REJECT via sequential testing: {b_result.rationale}"),
                extra_evidence={"sequential_decision": "REJECT",
                                "month": m, "t_stat": metric.t_stat},
                db_path=db_path,
            )
            action = f"AUTO-TRANSITIONED to REJECTED (early-stop loss)"
        except (GateNotMetError, Exception) as exc:
            action = f"REJECT signaled but transition failed: {exc}"
    elif b_result.decision == SequentialDecision.ACCEPT:
        action = ("ACCEPT signal — eligible for PAPER_TRADE → SHADOW; "
                  "human review required to authorize (Promote ≠ Allocate)")
    elif b_result.decision == SequentialDecision.CONTINUE:
        action = f"CONTINUE — observe through month {m + 1}"
    else:
        action = f"INSUFFICIENT data — minimum {boundary.min_months_before_first_look}mo required"

    return MonitorTickResult(
        strategy_id=strategy_id, role=role,
        months_observed=m, metric_result=metric, boundary_result=b_result,
        three_layer_result=three_layer_result,
        action_taken=action,
    )


def tick_all_paper_trade_sleeves(
    *,
    today: Optional[_dt.datetime] = None,
    role_overrides: Optional[dict[str, SleeveRole]] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[MonitorTickResult]:
    """Bulk tick: iterate all sleeves in PAPER_TRADE state.

    role_overrides: optional dict mapping strategy_id → SleeveRole.
                    Caller resolves roles ahead of time (Phase 3 will
                    auto-resolve from library YAML).
    """
    records = list_strategies(state=StrategyState.PAPER_TRADE, db_path=db_path)
    results: list[MonitorTickResult] = []
    for rec in records:
        role = (role_overrides or {}).get(rec.strategy_id)
        if role is None:
            results.append(MonitorTickResult(
                strategy_id=rec.strategy_id, role=None,
                months_observed=0, metric_result=None, boundary_result=None,
                error="role not provided + auto-resolve not yet implemented",
            ))
            continue
        try:
            r = tick_single_sleeve(
                strategy_id=rec.strategy_id, role=role,
                today=today, db_path=db_path,
            )
            results.append(r)
        except Exception as exc:
            results.append(MonitorTickResult(
                strategy_id=rec.strategy_id, role=role,
                months_observed=0, metric_result=None, boundary_result=None,
                error=f"tick failed: {exc}",
            ))
    return results
