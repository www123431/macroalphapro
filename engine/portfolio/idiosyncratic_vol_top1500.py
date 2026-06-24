"""
engine/portfolio/idiosyncratic_vol_top1500.py — Path AG impl.

Spec: docs/spec_path_ag_idiosyncratic_vol_top1500_v3_v1.md
Spec id=80, hash b1bd40cb37b5623344c10fd136251ca470b89def.

Ang-Hodrick-Xing-Zhang 2006 low-IVOL premium on top-1500 universe.
Signal: 22-week rolling residual std from SPY single-factor regression.
Long-only BOTTOM decile (low-IVOL), equal-weight, monthly rebalance, 10bp/side TC.
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

IVOL_WINDOW_W:    int = 22
BOT_DECILE_FRAC:  float = 0.10
MIN_NAMES_REQ:    int = 50

PER_NAME_CAP_MULT: float = 1.5
PER_NAME_CAP_FLOOR: float = 0.007

TC_BPS_PER_SIDE:    float = 10.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0

PANEL_PATH = "data/factor_ensemble_singlename/_crsp_dsf_top1500_panel.parquet"


@dataclass(frozen=True)
class AGBacktestResult:
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
    repo_root = Path(__file__).resolve().parent.parent.parent
    p = repo_root / PANEL_PATH
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def load_spy_weekly() -> pd.Series:
    import yfinance as _yf
    df = _yf.download("SPY", start=WINDOW_START, end="2024-01-15",
                       auto_adjust=True, progress=False, multi_level_index=False)
    px = df["Close"] if "Close" in df.columns else df
    px.index = pd.to_datetime(px.index)
    weekly = px.resample("W-FRI").last()
    return weekly.pct_change().dropna().rename("SPY")


def compute_ivol_panel(weekly_panel: pd.DataFrame, spy_weekly: pd.Series) -> pd.DataFrame:
    """For each stock, compute 22-week rolling residual std from SPY single-factor regression.

    Vectorized: for each stock, regress weekly returns on SPY weekly returns
    over rolling 22-week windows. Residual = r_i - (alpha + beta * r_SPY).
    IVOL = std of residuals.
    """
    returns = weekly_panel.pct_change()
    # Align SPY to panel index
    spy_aligned = spy_weekly.reindex(returns.index).fillna(0.0)

    # For each rolling window, compute beta_i + alpha_i + residuals
    # Use closed-form OLS: beta = cov(r_i, r_SPY) / var(r_SPY)
    # IVOL = std(r_i - alpha - beta*r_SPY) over window

    window = IVOL_WINDOW_W
    ivol = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    # Rolling stats on SPY
    spy_mean = spy_aligned.rolling(window, min_periods=window).mean()
    spy_var = spy_aligned.rolling(window, min_periods=window).var()

    for col in returns.columns:
        r = returns[col]
        # Rolling stats per stock
        r_mean = r.rolling(window, min_periods=window).mean()
        cov = ((r - r_mean) * (spy_aligned - spy_mean)).rolling(window, min_periods=window).mean()
        beta = cov / spy_var
        alpha = r_mean - beta * spy_mean
        # Residuals at each point of the rolling window
        # For each point t, residual = r(t) - alpha(t) - beta(t) * spy(t)
        # IVOL = std of residuals in window
        # Computational trick: since we already have alpha + beta rolling, we can compute
        # the in-sample residual variance directly:
        # res_var = r_var - beta^2 * spy_var
        r_var = r.rolling(window, min_periods=window).var()
        res_var = r_var - beta * beta * spy_var
        res_var = res_var.clip(lower=0.0)   # numerical floor
        ivol[col] = np.sqrt(res_var)

    return ivol


def select_low_ivol_decile(ivol_at_t: pd.Series, prices_at_t: pd.Series) -> pd.Series:
    """Pick BOTTOM 10% IVOL (lowest vol) names. Long-only equal-weight."""
    valid = ivol_at_t[(ivol_at_t.notna()) & (ivol_at_t > 0) & prices_at_t.notna() & (prices_at_t > 0)]
    if len(valid) < MIN_NAMES_REQ:
        return pd.Series(dtype=float)
    n_keep = max(int(math.ceil(len(valid) * BOT_DECILE_FRAC)), 1)
    bot = valid.sort_values(ascending=True).head(n_keep)
    eq_w = 1.0 / n_keep
    cap = max(eq_w * PER_NAME_CAP_MULT, PER_NAME_CAP_FLOOR)
    weights = pd.Series(min(eq_w, cap), index=bot.index)
    weights = weights / weights.sum()
    return weights


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    last_month = None
    for i, d in enumerate(weekly.index):
        ym = (d.year, d.month)
        if ym != last_month and i >= IVOL_WINDOW_W:
            dates.append(d)
            last_month = ym
        elif ym != last_month:
            last_month = ym
    return dates


def run_ag_backtest() -> AGBacktestResult:
    weekly = load_panel()
    weekly_returns = weekly.pct_change()
    spy = load_spy_weekly()
    logger.info("Computing IVOL panel (this may take ~30s for 2668 stocks)...")
    ivol_panel = compute_ivol_panel(weekly, spy)
    logger.info("IVOL panel computed.")

    rebal_dates = build_rebalance_dates(weekly)
    rebal_set = set(rebal_dates)

    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}
    decile_sizes: list[int] = []
    n_skipped = 0

    weeks = list(weekly.index)
    for i, week in enumerate(weeks):
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0

        if week in rebal_set:
            ivol_t = ivol_panel.loc[week]
            prices_t = weekly.loc[week]
            new_w = select_low_ivol_decile(ivol_t, prices_t)
            if new_w.empty:
                n_skipped += 1
                continue
            decile_sizes.append(len(new_w))
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE
            weekly_tc[week] = tc
            positions_history[week] = new_w.copy()
            current_weights = new_w

    gross = pd.Series(weekly_gross_ret, name="ag_gross")
    tcs   = pd.Series(weekly_tc, name="ag_tc")
    net   = (gross - tcs).rename("ag_net")

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
        notes.append(f"{n_skipped} rebalance(s) skipped")

    return AGBacktestResult(
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


def save_ag_parquet(
    result:    AGBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ag_ivol_top1500_weekly.parquet",
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
