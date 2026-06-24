"""
engine/portfolio/short_term_reversal.py — Path Z mean-reversion strategy.

Spec: docs/spec_path_z_short_term_reversal_v1.md
Spec id=73, hash 8845f771 (active).

Implements Jegadeesh 1990 / Lehmann 1990 short-term reversal on
S&P 500 historical universe (CRSP DSF panel).

Algorithm (locked, no parameter tuning post-impl):
  1. Universe: 298 S&P 500 tickers (same CRSP panel as V/W/X)
  2. Resample daily closes to weekly (W-FRI last)
  3. Monthly rebalance: first weekly bar of each month
  4. At each rebalance week t per stock i:
       signal_i(t) = price_i(t) / price_i(t-4w) - 1
                   # 4-week cumulative return, NO skip-month
                   # (NO skip because we WANT the recent move to reverse)
  5. Cross-sectional rank ASCENDING (low return = high desirability)
  6. Bottom decile (rank ≤ 0.10) = long, equal-weight, 3.5% cap
  7. Hold until next monthly rebalance
  8. TC: 10bp per side on rebalance turnover

Note: This is the EXACT OPPOSITE of Path V's signal direction.
Same infrastructure pattern, reversed selection.
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (per spec — DO NOT mutate without amendment)
# ─────────────────────────────────────────────────────────────────────────────
CRSP_PANEL_PATH:    str   = "data/factor_ensemble_singlename/_crsp_dsf_panel.parquet"
WINDOW_START:       str   = "2014-09-12"
WINDOW_END:         str   = "2023-12-29"

LOOKBACK_WEEKS:     int   = 4         # 4-week cumulative return
BOTTOM_PCTILE:      float = 0.10
PER_NAME_CAP:       float = 0.035
TC_DECIMAL_PER_SIDE: float = 10.0 / 10_000.0

EXCLUDE_TICKERS:    frozenset[str] = frozenset({"SPY", "BRK"})


@dataclass(frozen=True)
class STRBacktestResult:
    weekly_returns_gross:  pd.Series
    weekly_returns_net:    pd.Series
    weekly_tc_drag:        pd.Series
    rebalance_dates:       list[pd.Timestamp]
    n_weeks:               int
    n_rebalances:          int
    avg_n_names_per_rebal: float
    avg_turnover:          float
    notes:                 list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as Path V / W / X)
# ─────────────────────────────────────────────────────────────────────────────
def load_crsp_panel(
    panel_path:   str = CRSP_PANEL_PATH,
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
) -> pd.DataFrame:
    df = pd.read_parquet(panel_path).astype("float64").loc[window_start:window_end]
    drop = [t for t in EXCLUDE_TICKERS if t in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("W-FRI").last()


# ─────────────────────────────────────────────────────────────────────────────
# Signal (4-week cumulative return; ascending rank = long losers)
# ─────────────────────────────────────────────────────────────────────────────
def compute_signal(
    weekly:     pd.DataFrame,
    rebal_date: pd.Timestamp,
    lookback:   int = LOOKBACK_WEEKS,
) -> pd.Series:
    """4-week cumulative return per stock; NaN where not 4 weeks of data."""
    if rebal_date not in weekly.index:
        prior = weekly.index[weekly.index <= rebal_date]
        if len(prior) == 0:
            return pd.Series(dtype=float)
        rebal_date = prior[-1]
    idx_t = weekly.index.get_loc(rebal_date)
    if idx_t < lookback:
        return pd.Series(dtype=float)

    p_now  = weekly.iloc[idx_t]
    p_then = weekly.iloc[idx_t - lookback]
    sig = (p_now / p_then) - 1.0
    valid = p_now.notna() & p_then.notna()
    return sig.where(valid)


def select_bottom_decile(
    signal:        pd.Series,
    bottom_pctile: float = BOTTOM_PCTILE,
    per_name_cap:  float = PER_NAME_CAP,
) -> pd.Series:
    """Bottom decile equal-weight (loser stocks = reversal candidates)."""
    valid = signal.dropna()
    if len(valid) < 30:
        return pd.Series(dtype=float)
    rank = valid.rank(pct=True, method="average", ascending=True)
    bottom = valid[rank <= bottom_pctile]
    if len(bottom) == 0:
        return pd.Series(dtype=float)
    n = len(bottom)
    w = pd.Series(min(1.0 / n, per_name_cap), index=bottom.index)
    return w / w.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Monthly rebalance schedule (same as V/W/X)
# ─────────────────────────────────────────────────────────────────────────────
def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    last_month = None
    for i, d in enumerate(weekly.index):
        ym = (d.year, d.month)
        if ym != last_month and i >= LOOKBACK_WEEKS:
            dates.append(d)
            last_month = ym
        elif ym != last_month:
            last_month = ym
    return dates


def run_str_backtest(
    panel_path:   str = CRSP_PANEL_PATH,
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
) -> STRBacktestResult:
    daily = load_crsp_panel(panel_path, window_start, window_end)
    weekly = resample_weekly(daily)
    weekly_returns_per_stock = weekly.pct_change()

    rebal_dates = build_rebalance_dates(weekly)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after warmup")

    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}
    rebal_set = set(rebal_dates)
    weeks = list(weekly.index)
    n_skipped = 0

    for i, week in enumerate(weeks):
        # Return earned this week by prior positions
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns_per_stock.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
            weekly_tc[week] = 0.0
        else:
            weekly_gross_ret[week] = 0.0
            weekly_tc[week] = 0.0

        # On rebal days, compute new weights + TC
        if week in rebal_set:
            sig = compute_signal(weekly, week)
            new_weights = select_bottom_decile(sig)
            if new_weights.empty:
                n_skipped += 1
                continue
            all_tk = current_weights.index.union(new_weights.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_weights.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
            weekly_tc[week] = tc
            positions_history[week] = new_weights.copy()
            current_weights = new_weights

    gross = pd.Series(weekly_gross_ret, name="str_gross")
    tcs   = pd.Series(weekly_tc,        name="str_tc")
    net   = (gross - tcs).rename("str_net")

    avg_n = (float(np.mean([len(s.dropna()) for s in positions_history.values()]))
              if positions_history else 0.0)
    rebal_list = sorted(positions_history.keys())
    turnovers = []
    for i, d in enumerate(rebal_list):
        if i == 0:
            continue
        prev = positions_history[rebal_list[i - 1]]
        curr = positions_history[d]
        tk = prev.index.union(curr.index)
        turnovers.append(
            float((curr.reindex(tk, fill_value=0.0)
                   - prev.reindex(tk, fill_value=0.0)).abs().sum())
        )
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0

    notes = []
    if n_skipped:
        notes.append(f"{n_skipped} rebalance(s) skipped due to insufficient names")

    return STRBacktestResult(
        weekly_returns_gross  = gross,
        weekly_returns_net    = net,
        weekly_tc_drag        = tcs,
        rebalance_dates       = rebal_list,
        n_weeks               = len(weeks),
        n_rebalances          = len(rebal_list),
        avg_n_names_per_rebal = avg_n,
        avg_turnover          = avg_to,
        notes                 = notes,
    )


def save_str_parquet(
    result:    STRBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_z_str_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "gross": result.weekly_returns_gross,
        "tc":    result.weekly_tc_drag,
        "net":   result.weekly_returns_net,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
