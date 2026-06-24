"""
engine/p_fund_sleeves.py — MS-3 per-sleeve performance attribution layer.

Status: NEW 2026-05-10 (MS-3). Multi-sleeve commit per
`project_final_vision_hybrid_2026-05-10.md`.

Purpose
-------
Computes per-sleeve TWR / MWR / HPR + cross-sleeve aggregate. Production
`engine/performance_metrics.py::compute_period_summary` is the canonical
portfolio-wide GIPS-compliant TWR/MWR/HPR engine (unchanged); this module
is the sleeve-aware wrapper.

GIPS compliance scope (honest disclose)
---------------------------------------
The project currently runs ONE paper-trading account at the portfolio
level. Per-sleeve NAV is therefore SYNTHETIC — derived from SimulatedMonthlyReturn
contribution rows tagged with sleeve_id (post-MS-1 schema). This is:

  ✅ Mathematically correct given current operating mode (single account,
     capital allocation is conceptual / Tier 3-governed)
  ⚠️ NOT GIPS-strict (GIPS requires per-composite independent NAV book +
     cash flow tracking; this module derives sleeve attribution from
     position-level returns)

When a real second-account / second-sleeve NAV is funded (post-Wave B
activation + capital reallocation), this module's `compute_per_sleeve_summary`
should be replaced with actual per-sleeve NAV reads (Approach A — schema
add sleeve_id to PortfolioNavSnapshot). For now (Approach B — synthetic
derivation from monthly returns), tests verify the etf_l1 100%-alloc case
returns identical TWR to the canonical portfolio-wide compute_period_summary.

Architecture invariants
-----------------------
- 0 LLM imports
- `compute_period_summary` (portfolio-wide) unchanged
- Multi-sleeve helpers ADDITIVE; safe to ignore from non-multi-sleeve callers
- For etf_l1 with 100% capital allocation, output IS current portfolio summary
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Result dataclass ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class SleevePeriodSummary:
    """Per-sleeve period attribution (mirrors compute_period_summary fields)."""
    sleeve_id:           str
    twr_period:          float    # period-rate TWR (not annualized) — geometric link of monthly contribs
    twr_annualized:      float    # annualized TWR
    cumulative_return:   float    # alias of twr_period for narrative clarity
    mean_monthly_return: float    # arithmetic mean of monthly contribution_pct
    n_months:            int      # number of contributing monthly observations
    contributing_period: tuple[Optional[datetime.date], Optional[datetime.date]]  # (first_month, last_month)
    note:                str = ""    # honest scope disclosure (synthetic vs NAV-tracked)


# ── Per-sleeve attribution from SimulatedMonthlyReturn ──────────────────────
def compute_per_sleeve_summary(
    sleeve_id:  str,
    start:      datetime.date,
    end:        datetime.date,
    *,
    session:    Any | None = None,
) -> SleevePeriodSummary:
    """Compute synthetic per-sleeve TWR/HPR over [start, end] from
    SimulatedMonthlyReturn.contribution rows tagged with sleeve_id.

    Mechanism:
      1. Sum contribution_per_month within sleeve_id over [start, end]
         (each contribution = weight_held × sector_return for that month;
         per-month total = sum of all sector contributions in the sleeve)
      2. Geometric link monthly returns → period TWR
      3. Annualize via (1 + twr_period) ** (12 / n_months) - 1

    Returns SleevePeriodSummary; empty (zero / NaN) if no data.

    Honest scope: per-sleeve attribution derived from monthly contributions,
    NOT independent per-sleeve NAV book. See module docstring §GIPS scope.
    """
    from engine.memory import SimulatedMonthlyReturn, SessionFactory
    from engine.portfolio_sleeves import ALLOWED_SLEEVES

    if sleeve_id not in ALLOWED_SLEEVES:
        raise ValueError(
            f"sleeve_id {sleeve_id!r} not in ALLOWED_SLEEVES {sorted(ALLOWED_SLEEVES)}"
        )

    own_session = session is None
    sess = session if session is not None else SessionFactory()
    try:
        rows = (
            sess.query(SimulatedMonthlyReturn)
                .filter(SimulatedMonthlyReturn.return_month >= start)
                .filter(SimulatedMonthlyReturn.return_month <= end)
                .filter(SimulatedMonthlyReturn.sleeve_id == sleeve_id)
                .order_by(SimulatedMonthlyReturn.return_month.asc())
                .all()
        )
    finally:
        if own_session:
            sess.close()

    if not rows:
        return SleevePeriodSummary(
            sleeve_id           = sleeve_id,
            twr_period          = 0.0,
            twr_annualized      = 0.0,
            cumulative_return   = 0.0,
            mean_monthly_return = 0.0,
            n_months            = 0,
            contributing_period = (None, None),
            note                = (
                f"no SimulatedMonthlyReturn rows for sleeve_id={sleeve_id} "
                f"in [{start}, {end}]"
            ),
        )

    # Aggregate per-month: sum contribution across sectors within each (month, sleeve)
    by_month: dict[datetime.date, float] = {}
    for r in rows:
        m = r.return_month
        c = float(r.contribution or 0.0)
        by_month[m] = by_month.get(m, 0.0) + c

    months_sorted = sorted(by_month.keys())
    monthly_rets = [by_month[m] for m in months_sorted]

    # Period TWR via geometric link
    cum = 1.0
    for r in monthly_rets:
        cum *= (1.0 + r)
    twr_period = cum - 1.0

    # Annualize
    n = len(monthly_rets)
    if n > 0:
        try:
            twr_ann = (1.0 + twr_period) ** (12.0 / n) - 1.0
        except (OverflowError, ValueError):
            twr_ann = float("nan")
    else:
        twr_ann = 0.0

    mean_monthly = sum(monthly_rets) / n if n > 0 else 0.0

    return SleevePeriodSummary(
        sleeve_id           = sleeve_id,
        twr_period          = float(twr_period),
        twr_annualized      = float(twr_ann),
        cumulative_return   = float(twr_period),
        mean_monthly_return = float(mean_monthly),
        n_months            = n,
        contributing_period = (months_sorted[0], months_sorted[-1]),
        note                = (
            "synthetic per-sleeve attribution from SimulatedMonthlyReturn; "
            "not GIPS-strict (no independent per-sleeve NAV book)"
        ),
    )


# ── Cross-sleeve summary (all active sleeves + portfolio aggregate) ─────────
def compute_cross_sleeve_summary(
    start:    datetime.date,
    end:      datetime.date,
    *,
    session:  Any | None = None,
) -> dict[str, Any]:
    """Compute per-sleeve summaries for ALL active sleeves + portfolio aggregate.

    Returns:
        {
            'per_sleeve':   {sleeve_id: SleevePeriodSummary, ...},
            'portfolio':    compute_period_summary(start, end) dict
                            (canonical portfolio-wide TWR/MWR/HPR; unchanged),
            'window':       {'start': ..., 'end': ...},
            'caveat':       honest scope note for downstream UI / report,
        }

    The 'portfolio' field uses the canonical engine.performance_metrics
    compute_period_summary so cross-sleeve total stays GIPS-anchored.
    Per-sleeve breakdown is the synthetic attribution layer.
    """
    from engine.memory import SessionFactory
    from engine.performance_metrics import compute_period_summary
    from engine.portfolio_sleeves import ALLOWED_SLEEVES

    own_session = session is None
    sess = session if session is not None else SessionFactory()
    try:
        per_sleeve: dict[str, SleevePeriodSummary] = {}
        for sleeve_id in sorted(ALLOWED_SLEEVES):
            per_sleeve[sleeve_id] = compute_per_sleeve_summary(
                sleeve_id=sleeve_id,
                start=start, end=end,
                session=sess,
            )
        # Canonical portfolio-wide summary (GIPS-anchored, unchanged)
        portfolio = compute_period_summary(start=start, end=end, session=sess)
    finally:
        if own_session:
            sess.close()

    return {
        "per_sleeve": per_sleeve,
        "portfolio":  portfolio,
        "window":     {"start": start.isoformat(), "end": end.isoformat()},
        "caveat":     (
            "Per-sleeve TWR is SYNTHETIC (derived from SimulatedMonthlyReturn). "
            "Portfolio-wide TWR/MWR/HPR is GIPS-anchored from PortfolioNavSnapshot. "
            "Multi-sleeve independent NAV book activates with Wave B + capital "
            "reallocation per Tier 3 governance."
        ),
    }
