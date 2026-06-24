"""engine/validation/quarter_distribution.py — Phase 5.4: per-quarter
return distribution + concentration-risk verdict.

Paper borrow ([[project-paper-borrow-ml-btc-costs-2026-06-01]] item 5.4):
the BTC paper showed median quarter ARC was 7.98% but mean was 331%
with std 866% — performance came from 2-3 lucky quarters. A strategy
whose annualized number depends on 5% of the time-axis has structural
fragility no Sharpe / DSR / PBB catches.

Senior design choices:
  1. Auto-N for drop-top-N stress (not paper's hardcoded N=3): scale
     with quarter count, drop the lower of (3, ceil(0.05 * N_quarters))
  2. Bootstrap CI on the MEDIAN (via 5.1 PBB infra) so concentration
     verdict has statistical backing, not just point estimates
  3. Returns a discrete VERDICT (LOW / MED / HIGH concentration) not
     just stats — operationally useful, not just descriptive

API:
  compute_quarter_distribution(daily_or_monthly_returns) -> QuarterDist
  classify_concentration(QuarterDist) -> {"verdict", "reasons"}

Used by:
  - candidate_pipeline_v2 (new node P-D9 concentration_risk)
  - Cockpit iteration drill-down (renders distribution + verdict)
  - Pre-promotion gate (HARD_WARN if drop-top-N ARC < 0)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuarterDist:
    """Per-quarter return distribution stats."""
    n_quarters:        int
    median_arc:        float    # quarter-level annualized return, median
    mean_arc:          float
    std_arc:           float
    min_arc:           float
    max_arc:           float
    n_profitable:      int
    pct_profitable:    float    # n_profitable / n_quarters
    drop_top_n:        int      # how many top quarters were dropped
    drop_top_arc:      float    # ARC after dropping top-N quarters
    median_ci_lo:      float    # PBB CI on median (lower 2.5%)
    median_ci_hi:      float    # PBB CI on median (upper 97.5%)
    quarter_returns:   list[float] = field(default_factory=list)


def _to_quarter_returns(returns: pd.Series) -> pd.Series:
    """Compound a return series into quarterly returns.

    Accepts daily or monthly input. Returns indexed by quarter end."""
    r = returns.copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    # Clip extreme single-period returns to avoid one-day blow-ups
    # dominating the compound (cosmetic — keeps stats meaningful)
    r = r.clip(-0.5, 0.5)
    # Quarter-end resample compounded returns
    quarterly = (1.0 + r).resample("QE").prod() - 1.0
    return quarterly.dropna()


def _arc_from_quarter_returns(q_returns: Sequence[float]) -> float:
    """Annualize a sequence of quarter returns by compounding then
    raising to (4 / n) power. NaN-safe."""
    q = np.asarray(q_returns, dtype=float)
    q = q[~np.isnan(q)]
    if len(q) == 0:
        return float("nan")
    total = float(np.prod(1.0 + q))
    if total <= 0:
        # Geometric mean of negatives — use signed annualized
        n = len(q)
        return float(np.mean(q) * 4)
    return total ** (4.0 / len(q)) - 1.0


def compute_quarter_distribution(
    returns: pd.Series,
    *,
    drop_top_n: Optional[int] = None,
    bootstrap_n_iter: int = 2000,
    rng_seed: Optional[int] = None,
) -> QuarterDist:
    """Compute the quarter-level distribution + drop-top-N stress.

    drop_top_n defaults to min(3, ceil(0.05 * n_quarters)) — paper used
    fixed 3, we scale so a 12-yr 48-quarter series drops ~3 while a
    40-yr 160-quarter series drops ~8. Capped at 25% of n_quarters
    (beyond that the metric becomes meaningless)."""
    q = _to_quarter_returns(returns)
    n_q = len(q)
    if n_q < 4:
        raise ValueError(f"need >= 4 quarters, got {n_q}")

    if drop_top_n is None:
        drop_top_n = max(1, min(int(math.ceil(0.05 * n_q)), 3))
    drop_top_n = max(1, min(drop_top_n, max(1, n_q // 4)))

    q_sorted = q.sort_values()
    q_keep = q_sorted.iloc[:-drop_top_n]
    arc_full = _arc_from_quarter_returns(q.values)
    arc_drop = _arc_from_quarter_returns(q_keep.values)

    n_profitable = int((q > 0).sum())

    # Bootstrap CI on the median quarter return (PBB stationary)
    from engine.validation.block_bootstrap import pbb_statistic
    median_ci_lo, median_ci_hi = float("nan"), float("nan")
    try:
        pbb = pbb_statistic(
            q.values, lambda x: float(np.median(x)),
            n_iter=bootstrap_n_iter, rng_seed=rng_seed,
        )
        median_ci_lo, median_ci_hi = pbb.ci_lo, pbb.ci_hi
    except Exception:
        logger.exception("median CI bootstrap failed (non-fatal)")

    return QuarterDist(
        n_quarters=n_q,
        median_arc=float(_arc_from_quarter_returns([float(q.median())] * 4)),
        mean_arc=arc_full,
        std_arc=float(q.std(ddof=1) * 2.0),  # crude annualize of quarter std
        min_arc=float(q.min()),
        max_arc=float(q.max()),
        n_profitable=n_profitable,
        pct_profitable=round(n_profitable / n_q, 3),
        drop_top_n=drop_top_n,
        drop_top_arc=arc_drop,
        median_ci_lo=float(median_ci_lo),
        median_ci_hi=float(median_ci_hi),
        quarter_returns=[float(x) for x in q.values],
    )


def classify_concentration(qd: QuarterDist) -> dict:
    """Discrete concentration-risk verdict + reason list.

    HIGH: any of
      - drop-top-N ARC < 0  (paper's smoking gun)
      - pct_profitable < 50%
      - mean / median ratio > 3.0 (heavy tail)
    MED: any of
      - pct_profitable < 65%
      - mean / median ratio > 1.5
      - drop_top_arc < mean_arc * 0.3
    LOW: otherwise
    """
    reasons = []
    high = False
    med = False

    if qd.drop_top_arc < 0:
        reasons.append(
            f"drop-top-{qd.drop_top_n} ARC = {qd.drop_top_arc:.1%} "
            "(strategy loses money without top quarters — paper's red flag)"
        )
        high = True
    if qd.pct_profitable < 0.50:
        reasons.append(
            f"only {qd.pct_profitable:.0%} of quarters profitable "
            "(<50% = structural)"
        )
        high = True
    elif qd.pct_profitable < 0.65:
        reasons.append(
            f"{qd.pct_profitable:.0%} of quarters profitable "
            "(below 65% threshold)"
        )
        med = True

    # Mean / median divergence
    if qd.median_arc != 0 and not math.isnan(qd.median_arc):
        ratio = abs(qd.mean_arc / qd.median_arc) if qd.median_arc != 0 else float("inf")
        if ratio > 3.0:
            reasons.append(
                f"mean/median ratio {ratio:.1f} >> 3 (heavy positive tail "
                "— performance concentrated)"
            )
            high = True
        elif ratio > 1.5:
            reasons.append(
                f"mean/median ratio {ratio:.1f} > 1.5 (moderate tail "
                "concentration)"
            )
            med = True

    # Drop-top stress as fraction of full ARC
    if qd.mean_arc > 0 and not math.isnan(qd.mean_arc):
        if qd.drop_top_arc < qd.mean_arc * 0.3:
            reasons.append(
                f"drop-top stress eats {(1 - qd.drop_top_arc / qd.mean_arc) * 100:.0f}% of ARC"
            )
            med = True

    verdict = "HIGH" if high else "MED" if med else "LOW"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "drop_top_n":   qd.drop_top_n,
        "drop_top_arc": qd.drop_top_arc,
        "pct_profitable": qd.pct_profitable,
        "mean_median_ratio": (
            abs(qd.mean_arc / qd.median_arc)
            if qd.median_arc != 0 and not math.isnan(qd.median_arc)
            else None
        ),
    }
