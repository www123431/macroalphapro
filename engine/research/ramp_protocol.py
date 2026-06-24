"""engine/research/ramp_protocol.py — SLM Phase 3: tier-gated capital
allocation for SHADOW → LIVE journey.

Doctrine: full target allocation is reached gradually through tiered
ramps, each tier gated by trailing-window performance vs honest deploy
target. Per Citadel incubator pattern + WorldQuant Alpha Pool ramp
protocol.

Ramp tier semantics:

  Tier 1 (entry):   allocation 1%  AUM   — initial real-capital toe-dip
                                            entered automatically when
                                            PAPER_TRADE → SHADOW gate clears
  Tier 2 (3mo):     allocation 5%        — first scale-up; requires
                                            trailing Sharpe >= 0.70 × honest
  Tier 3 (6mo):     allocation 15%       — institutional-bench size
  Tier 4 (9-12mo):  allocation = target  — full deploy

Each tier has:
  - min_months_in_state: how long the sleeve must have been in SHADOW
  - min_trailing_metric: role-specific bar to advance
  - target_allocation_pct: the allocation at this tier

If trailing metric falls below threshold:
  - Stay at current tier (no upgrade)
  - If trailing metric falls BELOW DOWNGRADE_THRESHOLD, drop a tier
    (e.g. Tier 3 → Tier 2)
  - Persistent downgrades through Tier 1 → escalate to DECAY_WATCH

Role-specific schedules:
  alpha_seeker            — standard (above)
  risk_premium_harvester  — 2x slower (premium can vanish)
  insurance               — gated on hedge_effectiveness not Sharpe
  diversifier             — gated on sustained cosine
  regime_overlay          — gated on accumulated switching attribution

Modules touch points:
  - This module defines RampTier + RampSchedule data
  - Reads strategy state (shadow_started timestamp) + current_allocation
    from strategy_state_store
  - Uses role_specific_metric_eval to compute the gate metric
  - Updates current_allocation via update_allocation()
  - Triggers SHADOW → LIVE transition when Tier 4 reached
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.research.role_specific_metric_eval import (
    RoleMetricResult, evaluate_role_specific_metric,
)
from engine.research.strategy_lifecycle import SleeveRole, StrategyState
from engine.research.strategy_state_store import (
    DEFAULT_DB_PATH, get_strategy, transition, update_allocation,
)


class RampAction(str, Enum):
    """Outcome of a ramp tick."""

    HOLD = "HOLD"                # at current tier; no action
    UPGRADE = "UPGRADE"          # advance to next tier; allocation increased
    DOWNGRADE = "DOWNGRADE"      # fell back a tier; allocation reduced
    PROMOTE_TO_LIVE = "PROMOTE_TO_LIVE"  # Tier 4 reached → SHADOW → LIVE
    INSUFFICIENT = "INSUFFICIENT"  # not enough months for any tier yet
    ESCALATE_TO_DECAY_WATCH = "ESCALATE_TO_DECAY_WATCH"  # persistent failure
    ERROR = "ERROR"


@dataclass(frozen=True)
class RampTier:
    """One tier in a ramp schedule."""

    step: int                          # 1, 2, 3, 4
    min_months_in_state: int
    target_allocation_pct: float       # absolute (0.01 = 1% AUM)
    min_trailing_metric: float         # role-specific threshold
    downgrade_metric_threshold: float  # below this → step back


@dataclass(frozen=True)
class RampSchedule:
    """A complete ramp schedule (4 tiers) for a SleeveRole.

    Lookup tier-by-step via `tier_for_step()`; max step is len(tiers).
    """

    role: SleeveRole
    tiers: tuple[RampTier, ...]

    def tier_for_step(self, step: int) -> Optional[RampTier]:
        for t in self.tiers:
            if t.step == step:
                return t
        return None

    def next_eligible_tier(
        self, months_in_state: int, current_step: int,
    ) -> Optional[RampTier]:
        """Highest tier whose months requirement is met AND step > current."""
        eligible = [t for t in self.tiers
                    if t.min_months_in_state <= months_in_state
                    and t.step > current_step]
        if not eligible:
            return None
        return max(eligible, key=lambda t: t.step)


# ── Default schedules per role ─────────────────────────────────────────


_DEFAULT_ALPHA_SEEKER_SCHEDULE = RampSchedule(
    role=SleeveRole.ALPHA_SEEKER,
    tiers=(
        RampTier(1, 0,  0.01, 0.00, -10.0),   # entry: no metric check, just present
        RampTier(2, 3,  0.05, 0.70, 0.30),    # 3mo: trailing ≥ 0.70 × target
        RampTier(3, 6,  0.15, 0.75, 0.40),
        RampTier(4, 9,  0.50, 0.80, 0.50),    # 9mo: full target band
    ),
)

# risk_premium harvester ramps 2x slower — months doubled
_DEFAULT_RP_HARVESTER_SCHEDULE = RampSchedule(
    role=SleeveRole.RISK_PREMIUM_HARVESTER,
    tiers=(
        RampTier(1, 0,  0.01, 0.00, -10.0),
        RampTier(2, 6,  0.05, 0.60, 0.20),
        RampTier(3, 12, 0.15, 0.65, 0.30),
        RampTier(4, 18, 0.40, 0.70, 0.40),
    ),
)

# Insurance: ramp gated on hedge_beta (more negative is better).
# Thresholds expressed as the t_stat magnitude evaluator returns
# (signed-positive when β is sufficiently negative).
_DEFAULT_INSURANCE_SCHEDULE = RampSchedule(
    role=SleeveRole.INSURANCE,
    tiers=(
        RampTier(1, 0,  0.005, 0.00, -100.0),   # initial 0.5% insurance budget
        RampTier(2, 3,  0.02,  1.5,  0.0),      # require hedge_t ≥ 1.5
        RampTier(3, 6,  0.05,  2.0,  0.5),      # ≥ 2.0 sustained
        RampTier(4, 9,  0.10,  2.5,  1.0),      # full insurance budget
    ),
)

_DEFAULT_DIVERSIFIER_SCHEDULE = RampSchedule(
    role=SleeveRole.DIVERSIFIER,
    tiers=(
        RampTier(1, 0,  0.01, 0.0,  -10.0),
        RampTier(2, 3,  0.03, 1.0,  0.0),      # cosine_t ≥ 1.0 (negative cosine)
        RampTier(3, 6,  0.07, 1.5,  0.5),
        RampTier(4, 9,  0.15, 2.0,  1.0),
    ),
)

_DEFAULT_REGIME_OVERLAY_SCHEDULE = RampSchedule(
    role=SleeveRole.REGIME_OVERLAY,
    tiers=(
        RampTier(1, 0,  0.0,  0.0,  -10.0),    # overlay has no direct allocation
        RampTier(2, 3,  0.0,  1.0,  0.0),
        RampTier(3, 6,  0.0,  1.5,  0.5),
        RampTier(4, 9,  0.0,  2.0,  1.0),      # graduates to overlay-as-LIVE
    ),
)


_SCHEDULES: dict[SleeveRole, RampSchedule] = {
    SleeveRole.ALPHA_SEEKER:            _DEFAULT_ALPHA_SEEKER_SCHEDULE,
    SleeveRole.RISK_PREMIUM_HARVESTER:  _DEFAULT_RP_HARVESTER_SCHEDULE,
    SleeveRole.INSURANCE:               _DEFAULT_INSURANCE_SCHEDULE,
    SleeveRole.DIVERSIFIER:             _DEFAULT_DIVERSIFIER_SCHEDULE,
    SleeveRole.REGIME_OVERLAY:          _DEFAULT_REGIME_OVERLAY_SCHEDULE,
}


def get_schedule_for_role(role: SleeveRole) -> RampSchedule:
    return _SCHEDULES[role]


# ── Tick result ─────────────────────────────────────────────────────────


@dataclass
class RampTickResult:
    """Output of one ramp tick."""

    strategy_id: str
    role: SleeveRole
    months_in_shadow: int
    current_step: int
    new_step: int
    current_allocation: float
    new_allocation: float
    action: RampAction
    metric: Optional[RoleMetricResult] = None
    rationale: str = ""
    error: Optional[str] = None


# ── Step inference ─────────────────────────────────────────────────────


def _infer_current_step_from_allocation(
    allocation: float, schedule: RampSchedule,
) -> int:
    """Find which tier the current allocation corresponds to (closest)."""
    if allocation <= 0:
        return 0
    # Pick the highest tier whose target_allocation <= current allocation
    eligible = [t for t in schedule.tiers if t.target_allocation_pct <= allocation + 1e-9]
    if not eligible:
        return 1
    return max(eligible, key=lambda t: t.step).step


# ── Main tick ──────────────────────────────────────────────────────────


def tick_ramp(
    *,
    strategy_id: str,
    role: SleeveRole,
    sleeve_returns: pd.Series,
    today: Optional[_dt.datetime] = None,
    book_returns: Optional[pd.Series] = None,
    risk_source_returns: Optional[pd.Series] = None,
    static_baseline_returns: Optional[pd.Series] = None,
    actor: str = "ramp_protocol",
    db_path: Path = DEFAULT_DB_PATH,
    target_total_allocation: float = 0.50,
) -> RampTickResult:
    """One ramp tick for one SHADOW-state sleeve.

    Evaluates the role-specific metric on `sleeve_returns` (already
    windowed to the SHADOW observation period by caller) and decides
    whether to UPGRADE / HOLD / DOWNGRADE / PROMOTE_TO_LIVE.

    target_total_allocation: the AUM weight when the sleeve reaches
    Tier 4 (= LIVE-ready). Caller derives from sleeve's target_band.
    """
    today = today or _dt.datetime.now(_dt.timezone.utc)
    record = get_strategy(strategy_id, db_path=db_path)
    if record.current_state != StrategyState.SHADOW:
        return RampTickResult(
            strategy_id=strategy_id, role=role,
            months_in_shadow=0, current_step=0, new_step=0,
            current_allocation=record.current_allocation_pct,
            new_allocation=record.current_allocation_pct,
            action=RampAction.ERROR,
            error=f"sleeve not in SHADOW (is {record.current_state.value})",
        )
    if record.shadow_started is None:
        return RampTickResult(
            strategy_id=strategy_id, role=role,
            months_in_shadow=0, current_step=0, new_step=0,
            current_allocation=record.current_allocation_pct,
            new_allocation=record.current_allocation_pct,
            action=RampAction.ERROR,
            error="shadow_started timestamp missing in state row",
        )

    months_in_shadow = max(1, int(
        ((today - record.shadow_started).total_seconds() / (24 * 3600)) // 30
    ))

    schedule = get_schedule_for_role(role)
    current_step = _infer_current_step_from_allocation(
        record.current_allocation_pct, schedule,
    )
    current_tier = schedule.tier_for_step(current_step) if current_step > 0 else None

    # Compute role-specific metric on the shadow window
    metric = evaluate_role_specific_metric(
        role=role, sleeve_returns=sleeve_returns,
        book_returns=book_returns,
        risk_source_returns=risk_source_returns,
        static_baseline_returns=static_baseline_returns,
    )

    # Decide: try to find a higher tier we're eligible for
    next_tier = schedule.next_eligible_tier(
        months_in_state=months_in_shadow, current_step=current_step,
    )

    # Eligibility metric: t_stat for tiers with role-evaluator
    # (insurance / diversifier / regime_overlay) uses signed t-stat
    # already produced; alpha_seeker / risk_premium_harvester compare
    # metric_value (Sharpe ratio) against the target threshold.
    if role in (SleeveRole.ALPHA_SEEKER, SleeveRole.RISK_PREMIUM_HARVESTER):
        eligibility_metric = metric.metric_value
    else:
        eligibility_metric = metric.t_stat

    # Try upgrade first
    if next_tier is not None and \
            eligibility_metric >= (next_tier.min_trailing_metric * target_total_allocation
                                    if role in (SleeveRole.ALPHA_SEEKER,
                                                SleeveRole.RISK_PREMIUM_HARVESTER)
                                    else next_tier.min_trailing_metric):
        # UPGRADE
        new_alloc = next_tier.target_allocation_pct
        if role in (SleeveRole.ALPHA_SEEKER, SleeveRole.RISK_PREMIUM_HARVESTER):
            new_alloc = min(new_alloc, target_total_allocation)
        update_allocation(
            strategy_id=strategy_id,
            current_allocation_pct=new_alloc,
            actor=actor, db_path=db_path,
        )
        # If we reached Tier 4 → SHADOW → LIVE
        if next_tier.step == max(t.step for t in schedule.tiers):
            try:
                transition(
                    strategy_id=strategy_id,
                    to_state=StrategyState.LIVE,
                    actor=actor,
                    reason=f"Ramp reached Tier 4: alloc={new_alloc:.4f} via {role.value}",
                    ramp_protocol_step=next_tier.step,
                    extra_evidence={
                        "role": role.value,
                        "ramp_metric": metric.metric_value,
                        "ramp_t_stat": metric.t_stat,
                    },
                    db_path=db_path,
                )
                action = RampAction.PROMOTE_TO_LIVE
                rationale = (f"Tier 4 reached + SHADOW → LIVE; alloc {new_alloc:.4f}; "
                             f"metric={metric.metric_value:+.3f}")
            except Exception as exc:
                action = RampAction.UPGRADE
                rationale = (f"Tier 4 alloc set but SHADOW → LIVE transition failed: "
                             f"{exc}; remains in SHADOW at {new_alloc:.4f}")
        else:
            action = RampAction.UPGRADE
            rationale = (f"Upgraded to Tier {next_tier.step}; alloc {new_alloc:.4f}; "
                         f"metric={metric.metric_value:+.3f}")
        return RampTickResult(
            strategy_id=strategy_id, role=role,
            months_in_shadow=months_in_shadow,
            current_step=current_step, new_step=next_tier.step,
            current_allocation=record.current_allocation_pct,
            new_allocation=new_alloc,
            action=action, metric=metric, rationale=rationale,
        )

    # Downgrade check (if current tier's downgrade threshold breached)
    if current_tier is not None and eligibility_metric < current_tier.downgrade_metric_threshold:
        if current_step <= 1:
            # Already at Tier 1 and still failing → escalate to DECAY_WATCH
            return RampTickResult(
                strategy_id=strategy_id, role=role,
                months_in_shadow=months_in_shadow,
                current_step=current_step, new_step=current_step,
                current_allocation=record.current_allocation_pct,
                new_allocation=record.current_allocation_pct,
                action=RampAction.ESCALATE_TO_DECAY_WATCH,
                metric=metric,
                rationale=(f"At Tier 1 still failing; eligibility_metric="
                           f"{eligibility_metric:+.3f} < downgrade threshold "
                           f"{current_tier.downgrade_metric_threshold:.3f}; "
                           f"caller should transition to DECAY_WATCH"),
            )
        prev_tier = schedule.tier_for_step(current_step - 1)
        if prev_tier is None:
            return RampTickResult(
                strategy_id=strategy_id, role=role,
                months_in_shadow=months_in_shadow,
                current_step=current_step, new_step=current_step,
                current_allocation=record.current_allocation_pct,
                new_allocation=record.current_allocation_pct,
                action=RampAction.HOLD,
                metric=metric,
                rationale="downgrade requested but no prior tier defined",
            )
        new_alloc = prev_tier.target_allocation_pct
        update_allocation(
            strategy_id=strategy_id,
            current_allocation_pct=new_alloc,
            actor=actor, db_path=db_path,
        )
        return RampTickResult(
            strategy_id=strategy_id, role=role,
            months_in_shadow=months_in_shadow,
            current_step=current_step, new_step=prev_tier.step,
            current_allocation=record.current_allocation_pct,
            new_allocation=new_alloc,
            action=RampAction.DOWNGRADE,
            metric=metric,
            rationale=(f"Downgraded to Tier {prev_tier.step}; alloc {new_alloc:.4f}; "
                       f"metric breach"),
        )

    # No upgrade + no downgrade → HOLD
    return RampTickResult(
        strategy_id=strategy_id, role=role,
        months_in_shadow=months_in_shadow,
        current_step=current_step, new_step=current_step,
        current_allocation=record.current_allocation_pct,
        new_allocation=record.current_allocation_pct,
        action=RampAction.HOLD,
        metric=metric,
        rationale=(f"At Tier {current_step}, alloc {record.current_allocation_pct:.4f}; "
                   f"eligibility_metric={eligibility_metric:+.3f}; no tier change"),
    )
