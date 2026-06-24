"""
engine/portfolio/cross_sectional_momentum.py — Path V signal module.

Spec: docs/spec_path_v_cross_sectional_momentum_v1.md
Spec id: 69 · current hash: cd9fecf5 (post Amendment 1 scope_narrow)

Implements Jegadeesh-Titman 1993 cross-sectional momentum on S&P 500
historical-constituent universe (per Amendment 1 v1.1).

Algorithm (locked, no parameter tuning post-impl):
  1. Universe: 298 tickers from CRSP DSF panel (survivorship-free verified)
  2. Resample daily closes to weekly (W-FRI last)
  3. Monthly rebalance: first Friday of each month
  4. Per stock i at rebalance week t:
       mom_raw_i(t) = price_i(t-4w) / price_i(t-52w) - 1
                     (12-month cumulative return, skip last 4 weeks for
                     short-term reversal per J-T 1993 convention)
  5. Cross-sectional percentile rank at t (within currently-valid tickers)
  6. Top decile (rank >= 0.90) → equal weight, 3.5% per-name cap
  7. Position held until next monthly rebalance
  8. TC: 10bp per side applied to |Δ weight| × NAV at each rebalance bar
  9. Return weekly time series: w_t · r_t+1 (next-week return earned by
     positions decided at end of week t)

Doctrine guardrails:
  - 0-LLM-in-DECISION (this module has zero LLM)
  - HARKing red line: no spec parameter changes post-implementation
  - Survivorship: NaN at any week means stock not in universe that week
  - Look-ahead: rebalance decision uses data only up to and including
    rebalance week close
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (per spec — DO NOT mutate without amendment_log entry)
# ─────────────────────────────────────────────────────────────────────────────
CRSP_PANEL_PATH:    str   = "data/factor_ensemble_singlename/_crsp_dsf_panel.parquet"
WINDOW_START:       str   = "2014-09-12"
WINDOW_END:         str   = "2023-12-29"

# Signal parameters
LOOKBACK_WEEKS:     int   = 52   # full 12-month lookback
SKIP_WEEKS:         int   = 4    # skip last 4 weeks (~1 month for short-term reversal)

# Position parameters
TOP_DECILE_PCTILE:  float = 0.90
PER_NAME_CAP:       float = 0.035   # 3.5% per spec Amendment 1

# Execution
TC_BPS_PER_SIDE:    float = 10.0    # US single-stock T1 baseline (cost_model.py)
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0

# Universe filter — exclude index/ETF proxies that happen to be in panel
EXCLUDE_TICKERS:    frozenset[str] = frozenset({"SPY", "BRK"})  # SPY (ETF), BRK has share-class issues

# Rebalance: monthly, first trading day of each month
# Use pandas BMS (business month start) — closest to "first trading day of month"
REBAL_FREQ:         str   = "BMS"


# ─────────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CSMBacktestResult:
    """Output of run_csm_backtest()."""
    weekly_returns_gross:  pd.Series          # before TC
    weekly_returns_net:    pd.Series          # after TC drag at rebalance weeks
    weekly_tc_drag:        pd.Series          # per-week TC subtraction
    positions_history:     pd.DataFrame       # rebalance_date × ticker → weight
    rebalance_dates:       list[pd.Timestamp]
    n_weeks:               int
    n_rebalances:          int
    avg_n_names_per_rebal: float
    avg_turnover:          float
    notes:                 list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load panel + clean
# ─────────────────────────────────────────────────────────────────────────────
def load_crsp_panel(
    panel_path:   str = CRSP_PANEL_PATH,
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
) -> pd.DataFrame:
    """Load CRSP DSF panel, coerce dtypes, filter window, exclude proxies.

    Returns wide DataFrame: date index × ticker columns, float64 values
    with NaN for periods where ticker not in universe (pre-listing or
    post-delisting).
    """
    df = pd.read_parquet(panel_path)
    # Pandas nullable Float64 → plain float64 (some columns may be nullable)
    df = df.astype("float64")
    # Filter window
    df = df.loc[window_start:window_end]
    # Drop excluded proxy tickers
    drop = [t for t in EXCLUDE_TICKERS if t in df.columns]
    if drop:
        df = df.drop(columns=drop)
    logger.info(
        "CSM panel loaded: %d dates × %d tickers (%s → %s)",
        len(df), df.shape[1], df.index.min().date(), df.index.max().date(),
    )
    return df


def resample_weekly(daily_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily closes → weekly closes (Friday last)."""
    return daily_panel.resample("W-FRI").last()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Signal computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_signal(
    weekly_panel:  pd.DataFrame,
    rebal_date:    pd.Timestamp,
    lookback_weeks: int = LOOKBACK_WEEKS,
    skip_weeks:    int = SKIP_WEEKS,
) -> pd.Series:
    """12-1 cross-sectional momentum signal at a rebalance date.

    For each stock i present at both (rebal_date - lookback_weeks) and
    (rebal_date - skip_weeks):
        mom_raw_i = price_i(t - skip_weeks) / price_i(t - lookback_weeks) - 1

    Returns Series of mom_raw values, ticker-indexed, NaN for stocks
    missing either anchor price. Caller does the cross-sectional rank.
    """
    if rebal_date not in weekly_panel.index:
        # Snap to nearest <= rebal_date
        prior = weekly_panel.index[weekly_panel.index <= rebal_date]
        if len(prior) == 0:
            return pd.Series(dtype=float)
        rebal_date = prior[-1]
    idx_t = weekly_panel.index.get_loc(rebal_date)
    if idx_t < lookback_weeks:
        return pd.Series(dtype=float)

    price_now  = weekly_panel.iloc[idx_t - skip_weeks]
    price_then = weekly_panel.iloc[idx_t - lookback_weeks]
    mom = (price_now / price_then - 1.0)
    # Mask out tickers missing either anchor OR not currently listed at t
    price_t = weekly_panel.iloc[idx_t]
    valid = price_now.notna() & price_then.notna() & price_t.notna()
    return mom.where(valid)


def select_top_decile(
    signal:           pd.Series,
    top_pctile:       float = TOP_DECILE_PCTILE,
    per_name_cap:     float = PER_NAME_CAP,
) -> pd.Series:
    """Cross-sectional rank → top decile → equal-weight (per-name capped).

    Returns Series of weights (ticker → weight), sum = 1.0 on valid names.
    Empty Series if insufficient names.
    """
    valid = signal.dropna()
    if len(valid) < 30:
        return pd.Series(dtype=float)

    # Percentile rank in [0, 1]
    rank = valid.rank(pct=True, method="average")
    top = valid[rank >= top_pctile]
    if len(top) == 0:
        return pd.Series(dtype=float)

    n = len(top)
    # Equal weight, capped per-name
    eq_w = 1.0 / n
    w = pd.Series(min(eq_w, per_name_cap), index=top.index)
    # Renormalize so sum is 1.0 (only matters if cap binds — won't in
    # equal-weight unless n is very small)
    w = w / w.sum()
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────
def build_rebalance_dates(
    weekly_panel: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Find rebalance dates: first valid weekly bar in each calendar month.

    Per spec §2.4: monthly rebalance on first trading day of each month.
    We approximate with first Friday weekly close >= month start.
    """
    dates: list[pd.Timestamp] = []
    weekly_idx = weekly_panel.index
    last_month = None
    for d in weekly_idx:
        ym = (d.year, d.month)
        if ym != last_month:
            dates.append(d)
            last_month = ym
    # Drop the first ones until we have enough lookback
    min_idx = LOOKBACK_WEEKS
    dates = [d for d in dates if weekly_idx.get_loc(d) >= min_idx]
    return dates


def run_csm_backtest(
    panel_path:   str = CRSP_PANEL_PATH,
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
) -> CSMBacktestResult:
    """End-to-end Path V backtest per spec.

    Steps:
      1. Load + resample to weekly
      2. Build rebalance schedule (first weekly bar each month)
      3. At each rebalance: compute signal → top decile → weights
      4. Hold weights through next rebalance
      5. Compute weekly portfolio returns (gross)
      6. Apply TC drag at rebalance weeks
      7. Return aggregate result
    """
    daily = load_crsp_panel(panel_path, window_start, window_end)
    weekly = resample_weekly(daily)
    weekly_returns_per_stock = weekly.pct_change()

    rebal_dates = build_rebalance_dates(weekly)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after lookback exclusion")

    # Position state: ticker → weight, updated at each rebalance
    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)

    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}

    rebal_set = set(rebal_dates)
    weeks = list(weekly.index)
    notes: list[str] = []
    n_skipped = 0

    for i, week in enumerate(weeks):
        # 1) Compute return earned this week by previous-week's positions
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns_per_stock.iloc[i].reindex(current_weights.index)
            # Drop NaN: if stock delisted mid-period, position implicitly closed
            # at last valid value; treat NaN as 0 return for that name
            r_t = r_t.fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
            weekly_tc[week] = 0.0
        else:
            weekly_gross_ret[week] = 0.0
            weekly_tc[week] = 0.0

        # 2) On rebalance days, compute new weights + apply TC drag
        if week in rebal_set:
            signal = compute_signal(weekly, week)
            new_weights = select_top_decile(signal)
            if new_weights.empty:
                n_skipped += 1
                continue
            # Turnover = sum |Δw| across union of tickers
            all_tickers = current_weights.index.union(new_weights.index)
            w_old = current_weights.reindex(all_tickers, fill_value=0.0)
            w_new = new_weights.reindex(all_tickers, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            # TC: turnover × tc_decimal_per_side × 2 sides
            tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
            weekly_tc[week] = tc
            positions_history[week] = new_weights.copy()
            current_weights = new_weights

    gross_series = pd.Series(weekly_gross_ret, name="csm_gross_weekly")
    tc_series    = pd.Series(weekly_tc,        name="csm_tc_weekly")
    net_series   = (gross_series - tc_series).rename("csm_net_weekly")

    # Build positions_history DataFrame
    if positions_history:
        ph_df = pd.DataFrame(positions_history).T.fillna(0.0)
        ph_df.index.name = "rebal_date"
    else:
        ph_df = pd.DataFrame()

    avg_n = (
        np.mean([len(s.dropna()) for s in positions_history.values()])
        if positions_history else 0.0
    )

    # Average turnover excluding first rebal (when prior = empty)
    turnovers = []
    rebal_list = sorted(positions_history.keys())
    for i, d in enumerate(rebal_list):
        if i == 0:
            continue
        prev = positions_history[rebal_list[i - 1]]
        curr = positions_history[d]
        tk = prev.index.union(curr.index)
        po = prev.reindex(tk, fill_value=0)
        nx = curr.reindex(tk, fill_value=0)
        turnovers.append(float((nx - po).abs().sum()))
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0

    if n_skipped:
        notes.append(f"{n_skipped} rebalance(s) skipped due to insufficient names")

    return CSMBacktestResult(
        weekly_returns_gross  = gross_series,
        weekly_returns_net    = net_series,
        weekly_tc_drag        = tc_series,
        positions_history     = ph_df,
        rebalance_dates       = rebal_list,
        n_weeks               = len(weeks),
        n_rebalances          = len(rebal_list),
        avg_n_names_per_rebal = float(avg_n),
        avg_turnover          = avg_to,
        notes                 = notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Save weekly returns parquet for Sprint B replay integration
# ─────────────────────────────────────────────────────────────────────────────
def save_returns_parquet(
    result:    CSMBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_v_csm_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "gross":  result.weekly_returns_gross,
        "tc":     result.weekly_tc_drag,
        "net":    result.weekly_returns_net,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
