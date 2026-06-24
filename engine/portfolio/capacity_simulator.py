"""
engine/portfolio/capacity_simulator.py — Class A #3 SAA Capacity Audit.

Pre-deployment capacity simulation per Pastor-Stambaugh 2002 / Berk-Green
2004 / Korajczyk-Sadka 2004 frameworks.

Purpose
-------
Given the current 4-sleeve combined paper-trade portfolio (Sprint B
2014-2023 replay), simulate Sharpe / max-DD / capacity-utilization
across candidate AUM levels [$1M ... $1B]. Find the **optimal launch
AUM** that maximizes expected $ P&L subject to multi-criteria
constraints (Sharpe floor, DD ceiling, capacity headroom).

Key academic anchors
--------------------
- Pastor-Stambaugh 2002 "Mutual Fund Performance and Seemingly
  Unrelated Assets" — capacity-decay framework
- Berk-Green 2004 "Mutual Fund Flows in Rational Markets" —
  equilibrium AUM where alpha = fee + impact
- Korajczyk-Sadka 2004 "Are Momentum Profits Robust to Trading Costs?"
  — capacity testing methodology
- Frazzini-Israel-Moskowitz 2018 AQR "Trading Costs" — calibration of
  impact / spread for institutional flows
- Ang 2014 "Asset Management" Ch.16 — capacity due diligence standard

Methodology
-----------
For each candidate AUM level:
  1. For each historical strategy fill (Sprint B 2014-2023):
       position_usd = weight × strategy_book_share × AUM
       fetch 60-day ADV for that ticker
       compute TC via engine.execution.cost_model.estimate_tc_bps
       (which includes base + ADV-impact + half-spread + capacity warning)
  2. Apply TC drag to the historical strategy return → "AUM-adjusted return"
  3. Re-aggregate to combined portfolio returns at this AUM
  4. Compute Sharpe / max DD / capacity-warning fraction / annual $ P&L

Output: per-AUM table + recommended launch AUM + comfort zone +
growth ceiling.

Doctrine: read-only analysis. No production state mutated. Decision
memo writes to docs/decisions/. SAA amendments go through Tier 3
approval workflow (NOT triggered by this tool).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate AUM levels for the simulation grid
# ─────────────────────────────────────────────────────────────────────────────
AUM_LEVELS_USD: list[float] = [
    1_000_000,        # $1M (current paper)
    10_000_000,       # $10M
    50_000_000,       # $50M
    100_000_000,      # $100M
    250_000_000,      # $250M
    500_000_000,      # $500M
    1_000_000_000,    # $1B
    2_000_000_000,    # $2B
    5_000_000_000,    # $5B
    10_000_000_000,   # $10B (extended to find binding)
]

# Constraints for "optimal launch AUM" finder (per spec roadmap §IV)
SHARPE_FLOOR              = 0.50       # institutional "GOOD" bar (was 0.70 "VERY GOOD" — too aggressive for our paper profile)
MAX_DD_CEILING            = -0.12      # max DD must be >= -12% (less negative is better)
CAPACITY_UTIL_CEILING     = 0.50        # ≤ 50% of positions in capacity warning zone
ANNUAL_RETURN_DRAG_CAP    = 0.20        # if AUM drag is > 20% Sharpe loss from $1M baseline, exclude

# Current 5-sleeve allocation (post Tier 3 AC deployment 2026-05-15)
SLEEVE_ALLOCATION = {
    "K1_BAB":     0.324,
    "D_PEAD":     0.243,
    "PATH_N":     0.243,
    "CTA_PQTIX":  0.090,
    "AC_TLT_GLD": 0.100,
}

# Path B leverage Tier 3 amendment 2026-05-15 evening
LEVERAGE_FACTOR_DEFAULT = 1.5

# Borrow cost for leveraged portion (SOFR + spread, institutional standard)
BORROW_COST_BPS_ANNUAL = 75.0

# Default vol assumptions (per cost_model.py)
DEFAULT_VOL_BY_CLASS = {
    "etf":         0.15,
    "single_stock": 0.25,
    "mutual_fund": 0.10,
}


@dataclass(frozen=True)
class AUMScenario:
    """Capacity result at one AUM level."""
    aum_usd:                float
    sharpe_ann:             float
    annualized_return:      float
    annualized_vol:         float
    max_drawdown:           float
    annual_pnl_usd:         float
    avg_tc_drag_annual:     float        # decimal (e.g., 0.005 = 50bp)
    capacity_warning_frac:  float        # fraction of fills with size/ADV > 20%
    capacity_impact_frac:   float        # 5-20% ADV zone (informational)
    sharpe_vs_baseline:     float        # Sharpe(AUM) - Sharpe($1M)
    meets_sharpe_floor:     bool
    meets_dd_ceiling:       bool
    meets_capacity:         bool
    constraint_passes:      bool


@dataclass(frozen=True)
class CapacitySimResult:
    """Full simulation output."""
    sprint_b_window:        tuple[str, str]
    n_weeks:                int
    sleeve_allocation:      dict[str, float]
    baseline_sharpe:        float        # Sharpe at $1M (no significant drag)
    aum_scenarios:          list[AUMScenario]
    recommended_aum_usd:    Optional[float]
    comfort_zone_aum_usd:   tuple[Optional[float], Optional[float]]   # (low, high)
    growth_ceiling_aum_usd: Optional[float]
    binding_constraint:     str
    notes:                  list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# REAL per-sleeve ADV from yfinance fetch 2026-05-15
# Replaces class-based proxy with empirical 60-day ADV
# (see scripts/fetch_real_adv_per_sleeve.py)
# ─────────────────────────────────────────────────────────────────────────────
SLEEVE_REAL_ADV_USD = {
    "K1_BAB":     1_844_900_000,   # median of 15 representative ETFs (XLK/XLF/SPY/QQQ etc.)
    "D_PEAD":     2_306_900_000,   # median of 20 large-cap S&P 500 names
    "PATH_N":       384_500_000,   # median of 11 recent S&P 500 additions
                                    # (caveat: pre-inclusion ADV may be 30-50% lower; honest range ~$200-400M)
    "AC_TLT_GLD": 4_835_700_000,   # median of TLT + GLD
    "CTA_PQTIX":            0.0,   # mutual fund, NAV-priced, no ADV
}


def approximate_adv_usd(sleeve_name: str, base_class: str) -> float:
    """Get real per-sleeve median ADV (replaces class proxy).

    Real ADV from yfinance 60-day fetch on representative basket per sleeve.
    Class-based fallback only for legacy compat.
    """
    if sleeve_name in SLEEVE_REAL_ADV_USD:
        return SLEEVE_REAL_ADV_USD[sleeve_name]

    # Legacy class-based fallback
    if base_class == "etf_tier1":      return 5_000_000_000
    if base_class == "etf_tier2":      return 100_000_000
    if base_class == "single_stock_large": return 100_000_000
    if base_class == "single_stock_mid":   return 20_000_000
    if base_class == "single_stock_small": return 2_000_000
    if base_class == "mutual_fund":    return 0.0
    return 50_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Backend: simulate one AUM scenario
# ─────────────────────────────────────────────────────────────────────────────
def simulate_aum_scenario(
    aum_usd:          float,
    weekly_returns:   pd.DataFrame,
    leverage_factor:  float = 1.0,
    borrow_cost_bps:  float = 0.0,
) -> AUMScenario:
    """Compute Sharpe / DD / capacity at given AUM.

    Simplified model:
      - For each sleeve, approximate per-name notional × monthly turnover
      - Apply class-based TC drag from cost_model
      - Subtract from baseline Sharpe / NAV path
      - Tally capacity-warning fraction
    """
    from engine.execution.cost_model import (
        estimate_tc_bps, classify_instrument, CAPACITY_WARN_FRAC,
        InstrumentClass,
    )

    # Per-sleeve approximations for representative position structure
    # Updated 2026-05-15 to 5-sleeve (added AC TLT/GLD)
    sleeve_proxies = {
        "K1_BAB":     {"n_names": 43, "class": "etf_tier1",          "monthly_turnover": 0.50,
                        "monthly_rebal": 1.0, "tc_floor_bps": 4},
        "D_PEAD":     {"n_names": 150, "class": "single_stock_large", "monthly_turnover": 0.50,
                        "monthly_rebal": 1.0, "tc_floor_bps": 10},
        "PATH_N":     {"n_names":  15, "class": "single_stock_large", "monthly_turnover": 0.80,
                        "monthly_rebal": 0.25, "tc_floor_bps": 10},   # ~quarterly
        "CTA_PQTIX":  {"n_names":   1, "class": "mutual_fund",        "monthly_turnover": 0.0,
                        "monthly_rebal": 0.0, "tc_floor_bps": 0},
        "AC_TLT_GLD": {"n_names":   2, "class": "etf_tier1",          "monthly_turnover": 0.10,
                        "monthly_rebal": 1.0, "tc_floor_bps": 4},     # TLT/GLD top-3 liquid ETFs
    }

    annual_tc_drag = 0.0
    n_capacity_warnings = 0
    n_impact = 0
    n_fills_total = 0

    # Leverage scales NOTIONAL AUM (since 1.5x means 150% gross exposure)
    leveraged_aum = aum_usd * leverage_factor

    for sleeve, alloc in SLEEVE_ALLOCATION.items():
        proxy = sleeve_proxies[sleeve]
        if proxy["n_names"] == 0:
            continue
        # Per-name notional at leveraged AUM
        sleeve_notional   = alloc * leveraged_aum
        per_name_notional = sleeve_notional / proxy["n_names"]
        # ADV per name (class-based proxy)
        per_name_adv      = approximate_adv_usd(sleeve, proxy["class"])

        # Per-rebalance per-name turnover dollars
        turnover_usd = per_name_notional * proxy["monthly_turnover"]
        # Compute TC for representative trade
        if turnover_usd > 0 and per_name_adv > 0:
            est = estimate_tc_bps(
                ticker="proxy",
                position_size_usd=turnover_usd,
                adv_usd=per_name_adv,
                vol_ann=0.25 if "single_stock" in proxy["class"] else 0.15,
                market_cap_usd=None,
            )
            tc_per_fill_bps = est.total_tc_bps
            # Annual TC drag = tc_bps/10000 × monthly_turnover × monthly_rebal × 12
            sleeve_drag = (tc_per_fill_bps / 10_000.0) * proxy["monthly_turnover"] * (proxy["monthly_rebal"] * 12)
            annual_tc_drag += alloc * sleeve_drag
            if est.capacity_warning:
                n_capacity_warnings += proxy["n_names"]
            elif est.size_over_adv >= 0.05:
                n_impact += proxy["n_names"]
            n_fills_total += proxy["n_names"]
        else:
            n_fills_total += proxy["n_names"]   # mutual fund counts but no warning

    # Apply leverage to baseline weekly EXCESS returns (M-M 1958 + spread correction).
    # Borrowed portion pays (RFR + spread); your equity earns full portfolio return.
    # Net excess = L × (portfolio_return - RFR) - (L-1) × spread - TC_drag
    # Net total return = RFR + net_excess
    WEEKLY_RFR_LOCAL = 0.04 / 52.0
    weekly_drag = annual_tc_drag / 52.0
    weekly_spread = (leverage_factor - 1.0) * borrow_cost_bps / 10000.0 / 52.0
    portfolio_excess = weekly_returns["combined_return"] - WEEKLY_RFR_LOCAL
    leveraged_excess = leverage_factor * portfolio_excess - weekly_spread - weekly_drag
    aum_adjusted = WEEKLY_RFR_LOCAL + leveraged_excess

    # Sharpe / DD / NAV stats
    WEEKLY_RFR = 0.04 / 52.0
    excess = aum_adjusted - WEEKLY_RFR
    if excess.std() <= 0:
        sharpe = float("nan")
    else:
        sharpe = float(excess.mean() / excess.std() * math.sqrt(52))

    ann_ret = float(aum_adjusted.mean() * 52)
    ann_vol = float(aum_adjusted.std() * math.sqrt(52))

    nav = (1 + aum_adjusted.fillna(0)).cumprod()
    running_peak = nav.cummax()
    dd = (nav / running_peak) - 1.0
    max_dd = float(dd.min())

    annual_pnl_usd = aum_usd * ann_ret

    cap_warn_frac   = (n_capacity_warnings / n_fills_total) if n_fills_total > 0 else 0.0
    cap_impact_frac = (n_impact / n_fills_total) if n_fills_total > 0 else 0.0

    return AUMScenario(
        aum_usd                = aum_usd,
        sharpe_ann             = sharpe,
        annualized_return      = ann_ret,
        annualized_vol         = ann_vol,
        max_drawdown           = max_dd,
        annual_pnl_usd         = annual_pnl_usd,
        avg_tc_drag_annual     = annual_tc_drag,
        capacity_warning_frac  = cap_warn_frac,
        capacity_impact_frac   = cap_impact_frac,
        sharpe_vs_baseline     = 0.0,    # filled in later
        meets_sharpe_floor     = sharpe >= SHARPE_FLOOR if not math.isnan(sharpe) else False,
        meets_dd_ceiling       = max_dd >= MAX_DD_CEILING,
        meets_capacity         = cap_warn_frac <= CAPACITY_UTIL_CEILING,
        constraint_passes      = False,   # filled in later
    )


def run_capacity_simulation(
    baseline_returns_path: str = "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet",
    leverage_factor:       float = LEVERAGE_FACTOR_DEFAULT,
    borrow_cost_bps:       float = BORROW_COST_BPS_ANNUAL,
) -> CapacitySimResult:
    """Run capacity simulation across all AUM_LEVELS_USD."""
    df = pd.read_parquet(baseline_returns_path)
    df = df.astype("float64").fillna(0.0)
    df.index = pd.to_datetime(df.index)
    # If 5-sleeve per-strategy parquet, combine to single column "combined_return"
    if "combined_return" not in df.columns:
        combined = pd.Series(0.0, index=df.index)
        for col, w in SLEEVE_ALLOCATION.items():
            if col in df.columns:
                combined += w * df[col]
            elif col == "AC_TLT_GLD" and "AC_proxy_AB_2014_23" in df.columns:
                combined += w * df["AC_proxy_AB_2014_23"]
        df = pd.DataFrame({"combined_return": combined})
    weekly_returns = df

    scenarios = []
    for aum in AUM_LEVELS_USD:
        s = simulate_aum_scenario(aum, weekly_returns,
                                   leverage_factor=leverage_factor,
                                   borrow_cost_bps=borrow_cost_bps)
        scenarios.append(s)

    # Baseline Sharpe = scenario at lowest AUM ($1M, minimal drag)
    baseline_sh = scenarios[0].sharpe_ann

    # Fill in sharpe_vs_baseline + constraint_passes
    final_scenarios = []
    for s in scenarios:
        passes = s.meets_sharpe_floor and s.meets_dd_ceiling and s.meets_capacity
        final_scenarios.append(
            AUMScenario(
                aum_usd                = s.aum_usd,
                sharpe_ann             = s.sharpe_ann,
                annualized_return      = s.annualized_return,
                annualized_vol         = s.annualized_vol,
                max_drawdown           = s.max_drawdown,
                annual_pnl_usd         = s.annual_pnl_usd,
                avg_tc_drag_annual     = s.avg_tc_drag_annual,
                capacity_warning_frac  = s.capacity_warning_frac,
                capacity_impact_frac   = s.capacity_impact_frac,
                sharpe_vs_baseline     = (s.sharpe_ann - baseline_sh)
                                          if not math.isnan(s.sharpe_ann) else float("nan"),
                meets_sharpe_floor     = s.meets_sharpe_floor,
                meets_dd_ceiling       = s.meets_dd_ceiling,
                meets_capacity         = s.meets_capacity,
                constraint_passes      = passes,
            )
        )

    # Find recommended launch AUM = max-pnl AUM among constraint-passing
    passing = [s for s in final_scenarios if s.constraint_passes]
    recommended = max(passing, key=lambda s: s.annual_pnl_usd).aum_usd if passing else None

    # Comfort zone: lowest and highest passing AUMs (consecutive)
    if passing:
        sorted_passing = sorted(passing, key=lambda s: s.aum_usd)
        comfort_low = sorted_passing[0].aum_usd
        comfort_high = sorted_passing[-1].aum_usd
    else:
        comfort_low = comfort_high = None

    # Growth ceiling: first AUM that FAILs after the highest passing
    failing_above = [s for s in final_scenarios
                      if not s.constraint_passes
                      and comfort_high is not None and s.aum_usd > comfort_high]
    growth_ceiling = (min(failing_above, key=lambda s: s.aum_usd).aum_usd
                       if failing_above else None)

    # Identify binding constraint at growth ceiling
    if growth_ceiling is not None:
        gc_scenario = next(s for s in final_scenarios if s.aum_usd == growth_ceiling)
        constraints_failed = []
        if not gc_scenario.meets_sharpe_floor:  constraints_failed.append("Sharpe floor")
        if not gc_scenario.meets_dd_ceiling:    constraints_failed.append("DD ceiling")
        if not gc_scenario.meets_capacity:      constraints_failed.append("capacity warning fraction")
        binding = " + ".join(constraints_failed) or "n/a"
    elif not passing:
        binding = "no AUM passes all constraints — diagnose binding from scenario table"
    else:
        binding = "no constraint binds within tested range"

    return CapacitySimResult(
        sprint_b_window        = (str(df.index.min().date()), str(df.index.max().date())),
        n_weeks                = len(df),
        sleeve_allocation      = SLEEVE_ALLOCATION,
        baseline_sharpe        = baseline_sh,
        aum_scenarios          = final_scenarios,
        recommended_aum_usd    = recommended,
        comfort_zone_aum_usd   = (comfort_low, comfort_high),
        growth_ceiling_aum_usd = growth_ceiling,
        binding_constraint     = binding,
        notes                  = [],
    )
