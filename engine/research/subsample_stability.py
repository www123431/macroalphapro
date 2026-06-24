"""engine.research.subsample_stability — Tier C L2-5 Commit 1.

Subsample stability decomposition for Tier C factor PnL series.
Pure function — IO-free, no DB / parquet reads. Wiring lives in
later commit.

Mathematical contract:
  Split factor PnL into N equal-length sub-periods. For each:
    Sharpe_i = ann_return_i / ann_vol_i
    NW-t_i   = Sharpe_i / Newey-West HAC SE_i
  Report stability metrics:
    worst/best Sharpe ratio  (institutional bar: > 0.4)
    monotone decay flag      (consecutive splits all decreasing)
    decay slope test         (OLS of monthly excess return vs time)

Why this matters:
  McLean-Pontiff 2016: published factors lose 32-58% of in-sample
  Sharpe POST-publication. A factor with 33-year total NW-t = 3.57
  can have its t entirely driven by the first 8-year window.
  Subsample stability surfaces this. Headline-only verdicts miss it.

  Also: factors with INVERSE-MONOTONE Sharpe pattern (each split
  better than the last) are suspicious — could be a non-causal
  trend (e.g., a sector boom over the sample period) rather than
  a stable alpha mechanism.

Pattern matches L2-2 replication + L2-3 cost stress + L2-4 anchor
regression — each is a separable rigor layer with its own pure
function + later wiring.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Below this many months total, n_splits >= 2 produces sub-windows
# too short for reliable NW-t. AQR / 2σ standard is min 24 mo per
# sub-window; with 4 splits → 96 mo total floor.
MIN_TOTAL_MONTHS_FOR_4_SPLIT = 96
MIN_MONTHS_PER_SUB           = 24

# Institutional bar — worst-window Sharpe must be >= this fraction
# of best-window Sharpe for a factor to be considered "stable".
# Below this it's regime-dependent, not a stable alpha.
INSTITUTIONAL_STABILITY_BAR = 0.4


# A.1 cleanup (audit-found 2026-06-09): local copy of NW lag formula
# removed; routes through shared engine.research.lens_helpers.
from engine.research.lens_helpers import (
    nw_lag_rule_of_thumb as _nw_lag_rule_of_thumb,
)


def _ann_sharpe_and_nw_t(returns: pd.Series) -> tuple[float, float]:
    """Return (annualized Sharpe, NW HAC t-stat). NaN if degenerate."""
    n = len(returns)
    if n < 12:
        return float("nan"), float("nan")
    mu  = float(returns.mean())
    sig = float(returns.std(ddof=1))
    if sig <= 0 or not math.isfinite(sig):
        return float("nan"), float("nan")
    sharpe_monthly = mu / sig
    sharpe_ann     = sharpe_monthly * math.sqrt(12.0)
    # NW HAC SE on the mean → t = mean / SE
    lag = _nw_lag_rule_of_thumb(n)
    x = returns.values - mu
    s0 = float(np.dot(x, x) / n)
    s  = s0
    for L in range(1, lag + 1):
        w     = 1.0 - L / (lag + 1.0)
        gamma = float(np.dot(x[:-L], x[L:]) / n)
        s    += 2 * w * gamma
    if s <= 0:
        return sharpe_ann, float("nan")
    se_mean = math.sqrt(s / n)
    t_stat  = mu / se_mean
    return sharpe_ann, t_stat


def compute_subsample_stability(
    pnl_net:    pd.Series,
    *,
    n_splits:   int = 4,
    min_per_sub: int = MIN_MONTHS_PER_SUB,
) -> Optional[dict]:
    """Split factor PnL into n_splits equal-length sub-periods, report
    per-window stats + aggregate stability metrics.

    Args:
      pnl_net: monthly factor PnL (decimal), month-end DatetimeIndex
      n_splits: number of equal-length sub-windows (default 4 →
                ~8-year windows on a 33-year sample)
      min_per_sub: refuse if any sub-window has fewer months than this

    Returns:
      None when:
        - pnl_net empty / not DatetimeIndex
        - any sub-window < min_per_sub
        - all sub-windows degenerate
      Else:
        {
          n_splits:        int
          n_total_months:  int
          windows: [
            {
              start: "YYYY-MM", end: "YYYY-MM",
              n_months: int,
              sharpe_ann: float | None,
              nw_t_stat: float | None,
              ann_return: float | None,
              ann_vol: float | None,
            },
            ...
          ],
          # Aggregates
          worst_best_sharpe_ratio: float | None  # min(Sharpe) / max(Sharpe)
                                                  # if max > 0; else None
          institutional_stable: bool             # ratio >= 0.4 AND
                                                  # min Sharpe > 0
          monotone_decay:       bool             # each split's Sharpe
                                                  # strictly < prior
          monotone_growth:      bool             # each split's Sharpe
                                                  # strictly > prior
                                                  # (suspicious — possible
                                                  # non-stationary trend)
          decay_slope_per_year: float | None     # OLS slope of monthly
                                                  # excess return on
                                                  # time index (decimal/yr)
          decay_slope_t:        float | None     # NW-t on the slope
                                                  # estimate
        }
    """
    if pnl_net is None or len(pnl_net) == 0:
        return None
    if not isinstance(pnl_net.index, pd.DatetimeIndex):
        logger.warning("subsample: index must be DatetimeIndex; got %s",
                          type(pnl_net.index).__name__)
        return None
    series = pnl_net.dropna().sort_index()
    n_total = len(series)
    if n_total < n_splits * min_per_sub:
        logger.info(
            "subsample: n_total=%d < n_splits=%d × min_per_sub=%d, refusing",
            n_total, n_splits, min_per_sub,
        )
        return None

    # ── 1. Build sub-window boundaries (equal-length integer split) ──
    boundaries = np.linspace(0, n_total, n_splits + 1).astype(int)
    windows = []
    for i in range(n_splits):
        lo, hi = boundaries[i], boundaries[i + 1]
        sub = series.iloc[lo:hi]
        if len(sub) < min_per_sub:
            logger.info("subsample: sub-window %d has %d < %d months",
                          i, len(sub), min_per_sub)
            return None
        sharpe, t_stat = _ann_sharpe_and_nw_t(sub)
        ann_ret = float(sub.mean()) * 12.0
        ann_vol = float(sub.std(ddof=1)) * math.sqrt(12.0) \
            if sub.std(ddof=1) > 0 else float("nan")
        windows.append({
            "start":      sub.index.min().strftime("%Y-%m"),
            "end":        sub.index.max().strftime("%Y-%m"),
            "n_months":   len(sub),
            "sharpe_ann": float(sharpe) if math.isfinite(sharpe) else None,
            "nw_t_stat":  float(t_stat) if math.isfinite(t_stat) else None,
            "ann_return": ann_ret if math.isfinite(ann_ret) else None,
            "ann_vol":    ann_vol if math.isfinite(ann_vol) else None,
        })

    # ── 2. Aggregates ──
    sharpes = [w["sharpe_ann"] for w in windows
                 if w["sharpe_ann"] is not None]
    worst_best_ratio: Optional[float] = None
    inst_stable = False
    if sharpes and max(sharpes) > 0:
        # Ratio is meaningful when the BEST window is positive. If
        # best Sharpe is negative the factor doesn't work in ANY
        # window — stability irrelevant.
        worst_best_ratio = float(min(sharpes) / max(sharpes))
        inst_stable = (worst_best_ratio >= INSTITUTIONAL_STABILITY_BAR
                         and min(sharpes) > 0)

    # Monotone flags (strict). Length-1 case → both False (trivial).
    monotone_decay  = (len(sharpes) >= 2
                         and all(sharpes[i + 1] < sharpes[i]
                                  for i in range(len(sharpes) - 1)))
    monotone_growth = (len(sharpes) >= 2
                         and all(sharpes[i + 1] > sharpes[i]
                                  for i in range(len(sharpes) - 1)))

    # ── 3. Decay slope test (OLS of monthly ret on time index) ──
    decay_slope_per_year: Optional[float] = None
    decay_slope_t: Optional[float] = None
    try:
        # Months since start as float
        t_months = np.arange(len(series), dtype=float)
        # Centered to reduce intercept-slope correlation
        t_centered = t_months - t_months.mean()
        y = series.values
        ss_t = float(np.dot(t_centered, t_centered))
        if ss_t > 0:
            slope_monthly = float(np.dot(t_centered, y) / ss_t)
            intercept     = float(y.mean()) - slope_monthly * t_centered.mean()
            # Residuals → NW HAC SE on the slope
            resid = y - (intercept + slope_monthly * t_months)
            lag   = _nw_lag_rule_of_thumb(len(series))
            # SE(slope) using NW HAC on the score function
            x_dem = t_centered
            n     = len(series)
            s0    = float(np.dot((x_dem * resid)**2, np.ones(n)) / n)
            s     = s0
            for L in range(1, lag + 1):
                w     = 1.0 - L / (lag + 1.0)
                lhs   = (x_dem * resid)[:-L]
                rhs   = (x_dem * resid)[L:]
                gamma = float(np.dot(lhs, rhs) / n)
                s    += 2 * w * gamma
            if s > 0:
                se_slope = math.sqrt(s * n / (ss_t ** 2))
                if se_slope > 0:
                    decay_slope_t = float(slope_monthly / se_slope)
            # Per-YEAR slope for human-readable reporting
            decay_slope_per_year = slope_monthly * 12.0
    except Exception as exc:
        logger.warning("subsample: decay slope test failed: %s", exc)

    return {
        "n_splits":                n_splits,
        "n_total_months":          n_total,
        "windows":                 windows,
        "worst_best_sharpe_ratio": worst_best_ratio,
        "institutional_stable":    inst_stable,
        "monotone_decay":          monotone_decay,
        "monotone_growth":         monotone_growth,
        "decay_slope_per_year":    decay_slope_per_year,
        "decay_slope_t":           decay_slope_t,
    }


def compute_for_tier_c_pnl_series(
    pnl_series_df: pd.DataFrame,
    *,
    n_splits:      int = 4,
    artifacts:     Optional[dict] = None,
) -> Optional[dict]:
    """Tier C wiring helper. Computes subsample stability on the
    template's declared default-cost net PnL column.

    B.2 (2026-06-09): column choice flows through the explicit
    artifacts contract (see engine.research.lens_helpers). Templates
    declare `pnl_default_col` in their artifacts; this helper reads
    that declaration with a legacy fallback for un-migrated templates.

    Args:
      pnl_series_df: the pnl_series_df DataFrame (templates' main
                     PnL artifact). Kept as a positional arg for
                     backwards compatibility with callers that pass
                     it directly without the artifacts dict.
      n_splits: equal-length sub-window count (default 4)
      artifacts: full template_result.artifacts dict. When provided,
                 `pnl_default_col` declaration is honored. When None,
                 falls back to legacy heuristic on pnl_series_df.

    Returns:
      JSON-safe dict from compute_subsample_stability, or None.
    """
    from engine.research.lens_helpers import (
        resolve_default_net_col, _legacy_pick_net_col,
    )
    if pnl_series_df is None or pnl_series_df.empty:
        return None
    if artifacts is not None:
        col = resolve_default_net_col(artifacts)
    else:
        col = _legacy_pick_net_col(pnl_series_df)
    if col is None or col not in pnl_series_df.columns:
        return None
    return compute_subsample_stability(
        pnl_series_df[col].dropna(),
        n_splits=n_splits,
    )


# ────────────────────────────────────────────────────────────────────
# Lens registry declaration (Phase 1 Commit 2, 2026-06-09)
# ────────────────────────────────────────────────────────────────────
def _runner_subsample(spec, template_result, prior_outputs):
    """B.2 (2026-06-09): pass the full artifacts dict so the
    pnl_default_col contract is honored. Templates that declare
    `pnl_default_col` get their explicit choice; un-migrated ones
    fall through to the legacy heuristic."""
    artifacts = template_result.artifacts or {}
    pnl_df = artifacts.get("pnl_series_df")
    if pnl_df is None or len(pnl_df) == 0:
        return None
    return compute_for_tier_c_pnl_series(
        pnl_df, n_splits=4, artifacts=artifacts,
    )


def _build_lens_declaration():
    from engine.research.lens_registry import LensDeclaration
    return LensDeclaration(
        name             = "subsample_stability",
        version          = "v1_2026-06-08",
        applicable_to    = {
            # Per spec §15.A3: insurance/diversifier routed to
            # Tier D entirely; NO Tier C lenses run on them.
            # subsample applies to alpha + overlay sleeves only.
            "investment_role": ("alpha", "overlay"),
            # All asset classes
        },
        input_protocols  = ("PnlSeriesDataFrameContract",),
        output_protocol  = "SubsampleStabilityOutput",
        conditional_on   = None,
        fallback_chain   = (),
        output_schema    = {
            "primary":   "windows",
            "secondary": ("worst_best_sharpe_ratio",
                          "institutional_stable",
                          "monotone_decay", "monotone_growth",
                          "decay_slope_per_year", "decay_slope_t"),
        },
        consumed_by      = (),    # leaf — no downstream consumers
        runner           = _runner_subsample,
    )


LENS_DECLARATION = _build_lens_declaration()
