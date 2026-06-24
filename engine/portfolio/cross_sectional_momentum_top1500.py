"""
engine/portfolio/cross_sectional_momentum_top1500.py — Path AD impl.

Spec: docs/spec_path_ad_cumulative_momentum_top1500_v3_v1.md
Spec id=78, hash 2cd3bd92025aa0ab66442142028474c911a9f990 (active, v3 alpha class).

Cross-sectional 12-1 cumulative momentum (skip 4w) on point-in-time top-1500
US stocks by market cap. Long-only top decile equal-weight, monthly rebalance.
Same signal formula as Path V (id=69) but on broader universe (2668 permnos
vs 298 S&P 500 historical names).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2014-09-12"
WINDOW_END:   str = "2023-12-29"

LOOKBACK_WEEKS:    int = 52
SKIP_WEEKS:        int = 4
TOP_DECILE_FRAC:   float = 0.10
MIN_NAMES_REQ:     int = 50    # need at least 50 valid signals to rebalance

PER_NAME_CAP_MULT: float = 1.5   # max 1.5x equal-weight per name
PER_NAME_CAP_FLOOR: float = 0.007  # 0.7% floor

TC_BPS_PER_SIDE:    float = 10.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0

PANEL_PATH = "data/factor_ensemble_singlename/_crsp_dsf_top1500_panel.parquet"


@dataclass(frozen=True)
class ADBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int
    avg_turnover:         float
    avg_decile_size:      float
    notes:                list[str] = field(default_factory=list)


def load_panel() -> pd.DataFrame:
    """Load top-1500 weekly W-FRI panel built by top1500_panel_loader."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    p = repo_root / PANEL_PATH
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def compute_signal(weekly: pd.DataFrame, rebal_date: pd.Timestamp) -> pd.Series:
    """12-1 cumulative return (skip 4w) for all stocks at rebal_date."""
    if rebal_date not in weekly.index:
        prior = weekly.index[weekly.index <= rebal_date]
        if len(prior) == 0:
            return pd.Series(dtype=float)
        rebal_date = prior[-1]
    idx_t = weekly.index.get_loc(rebal_date)
    if idx_t < LOOKBACK_WEEKS:
        return pd.Series(dtype=float)

    p_now  = weekly.iloc[idx_t - SKIP_WEEKS]
    p_then = weekly.iloc[idx_t - LOOKBACK_WEEKS]
    sig = p_now / p_then - 1.0
    return sig.where(p_now.notna() & p_then.notna() & (p_now > 0) & (p_then > 0))


def select_top_decile(signal: pd.Series) -> pd.Series:
    """Long-only top decile, equal-weight with cap."""
    valid = signal.dropna()
    if len(valid) < MIN_NAMES_REQ:
        return pd.Series(dtype=float)
    n_keep = max(int(math.ceil(len(valid) * TOP_DECILE_FRAC)), 1)
    top = valid.sort_values(ascending=False).head(n_keep)
    eq_w = 1.0 / n_keep
    cap = max(eq_w * PER_NAME_CAP_MULT, PER_NAME_CAP_FLOOR)
    weights = pd.Series(min(eq_w, cap), index=top.index)
    weights = weights / weights.sum()
    return weights


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    """First weekly bar of each month, post-warmup."""
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


def run_ad_backtest() -> ADBacktestResult:
    weekly = load_panel()
    weekly_returns = weekly.pct_change()

    rebal_dates = build_rebalance_dates(weekly)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after warmup")
    rebal_set = set(rebal_dates)

    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}
    decile_sizes: list[int] = []
    n_skipped = 0

    weeks = list(weekly.index)
    for i, week in enumerate(weeks):
        # Return this week applied to prior positions
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0

        # Rebalance
        if week in rebal_set:
            sig = compute_signal(weekly, week)
            new_w = select_top_decile(sig)
            if new_w.empty:
                n_skipped += 1
                continue
            decile_sizes.append(len(new_w))
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE  # one-sided TC × turnover
            weekly_tc[week] = tc
            positions_history[week] = new_w.copy()
            current_weights = new_w

    gross = pd.Series(weekly_gross_ret, name="ad_gross")
    tcs   = pd.Series(weekly_tc,        name="ad_tc")
    net   = (gross - tcs).rename("ad_net")

    rebal_list = sorted(positions_history.keys())
    turnovers = []
    for i, d in enumerate(rebal_list):
        if i == 0:
            continue
        prev = positions_history[rebal_list[i - 1]]
        curr = positions_history[d]
        tk = prev.index.union(curr.index)
        turnovers.append(float(
            (curr.reindex(tk, fill_value=0.0) - prev.reindex(tk, fill_value=0.0)).abs().sum()
        ))
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0
    avg_decile = float(np.mean(decile_sizes)) if decile_sizes else 0.0

    notes = []
    if n_skipped:
        notes.append(f"{n_skipped} rebalance(s) skipped (insufficient signals)")

    return ADBacktestResult(
        weekly_returns_gross = gross,
        weekly_returns_net   = net,
        weekly_tc_drag       = tcs,
        rebalance_dates      = rebal_list,
        n_weeks              = len(weeks),
        n_rebalances         = len(rebal_list),
        avg_turnover         = avg_to,
        avg_decile_size      = avg_decile,
        notes                = notes,
    )


def save_ad_parquet(
    result:    ADBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ad_top1500_momentum_weekly.parquet",
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
