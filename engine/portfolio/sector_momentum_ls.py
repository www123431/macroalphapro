"""
engine/portfolio/sector_momentum_ls.py — Path AA L/S Sector Momentum.

Spec: docs/spec_path_aa_sector_momentum_ls_v1.md
Spec id=74, hash 4ac52a55 (active, v2 gate framework).

Implements Moskowitz-Grinblatt 1999 industry momentum on 11 SPDR sector ETFs,
with L/S structure (top-3 long / bot-3 short, equal-weight per leg).

Algorithm (locked):
  1. Universe: 11 SPDR sectors with locked pre-coverage substitution
       XLK / XLF / XLE / XLV / XLI / XLY / XLP / XLU / XLB / XLRE / XLC
       XLRE pre-2015-10 → VNQ proxy
       XLC pre-2018-06  → 0.5×XLK + 0.5×XLY weighted blend
  2. Weekly resample to W-FRI close
  3. Monthly rebalance (first weekly bar of each month)
  4. Signal: 12-1 cumulative return (52w lookback, skip 4w)
  5. Cross-sectional rank descending (high mom = long candidate)
  6. Top-3 long (each +1/6) + Bot-3 short (each -1/6)
     gross=1.0, net=0
  7. TC: 4bp per side per rebalance (ETF Tier-1 baseline)

Reuses pattern from cross_sectional_momentum.py / factor_anomalies.py.
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
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_START:       str   = "2014-09-12"
WINDOW_END:         str   = "2023-12-29"

SECTOR_TICKERS:     tuple[str, ...] = (
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "XLRE", "XLC",
)

# Pre-coverage substitution dates + proxies
XLRE_AVAILABLE_FROM = pd.Timestamp("2015-10-08")  # XLRE inception ~2015-10
XLC_AVAILABLE_FROM  = pd.Timestamp("2018-06-19")  # XLC inception ~2018-06
XLRE_PROXY_TICKER   = "VNQ"                       # Vanguard Real Estate
# XLC pre-2018-06: 0.5 × XLK + 0.5 × XLY (constituents split)

LOOKBACK_WEEKS:     int   = 52
SKIP_WEEKS:         int   = 4

N_LONG:             int   = 3
N_SHORT:            int   = 3

TC_BPS_PER_SIDE:    float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class AABacktestResult:
    weekly_returns_gross:  pd.Series
    weekly_returns_net:    pd.Series
    weekly_tc_drag:        pd.Series
    rebalance_dates:       list[pd.Timestamp]
    n_weeks:               int
    n_rebalances:          int
    avg_turnover:          float
    notes:                 list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + pre-coverage substitution
# ─────────────────────────────────────────────────────────────────────────────
def load_sector_panel(
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
) -> pd.DataFrame:
    """Load 11 sector ETFs via yfinance + pre-coverage proxies.

    Returns weekly W-FRI panel of adjusted close prices.
    """
    import yfinance as _yf

    tickers_needed = set(SECTOR_TICKERS) | {XLRE_PROXY_TICKER}
    # XLK + XLY already in SECTOR_TICKERS so XLC proxy uses them
    daily = _yf.download(
        sorted(tickers_needed),
        start=window_start, end=window_end,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if "Close" in daily.columns:
        daily = daily["Close"]
    daily.index = pd.to_datetime(daily.index)

    # Resample to weekly W-FRI close
    weekly = daily.resample("W-FRI").last()

    # XLRE pre-coverage substitution
    if "XLRE" in weekly.columns:
        xlre_naive_start = weekly["XLRE"].first_valid_index()
        if xlre_naive_start is not None and xlre_naive_start > pd.Timestamp(WINDOW_START):
            # For weeks before XLRE has data, fill with VNQ
            pre_xlre = weekly.index < xlre_naive_start
            weekly.loc[pre_xlre, "XLRE"] = weekly.loc[pre_xlre, XLRE_PROXY_TICKER]

    # XLC pre-coverage substitution (0.5 XLK + 0.5 XLY blend)
    if "XLC" in weekly.columns:
        xlc_naive_start = weekly["XLC"].first_valid_index()
        if xlc_naive_start is not None and xlc_naive_start > pd.Timestamp(WINDOW_START):
            pre_xlc = weekly.index < xlc_naive_start
            # Use price-blend; not perfect (no rebasing) but acceptable per spec
            # (Path AA spec §2.1 acknowledges this as locked pre-coverage substitution)
            blend = 0.5 * weekly.loc[pre_xlc, "XLK"] + 0.5 * weekly.loc[pre_xlc, "XLY"]
            weekly.loc[pre_xlc, "XLC"] = blend

    # Return only the 11 sector tickers (drop proxy column)
    out = weekly[list(SECTOR_TICKERS)].dropna(how="all")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Signal + rebalance schedule
# ─────────────────────────────────────────────────────────────────────────────
def compute_signal(
    weekly_panel: pd.DataFrame,
    rebal_date:   pd.Timestamp,
) -> pd.Series:
    """12-1 cumulative return per sector at rebal_date."""
    if rebal_date not in weekly_panel.index:
        prior = weekly_panel.index[weekly_panel.index <= rebal_date]
        if len(prior) == 0:
            return pd.Series(dtype=float)
        rebal_date = prior[-1]
    idx_t = weekly_panel.index.get_loc(rebal_date)
    if idx_t < LOOKBACK_WEEKS:
        return pd.Series(dtype=float)

    p_now  = weekly_panel.iloc[idx_t - SKIP_WEEKS]
    p_then = weekly_panel.iloc[idx_t - LOOKBACK_WEEKS]
    sig = p_now / p_then - 1.0
    return sig.where(p_now.notna() & p_then.notna())


def select_ls_positions(signal: pd.Series) -> pd.Series:
    """Top-3 long + Bot-3 short, equal-weight per leg, gross 1.0 net 0."""
    valid = signal.dropna()
    if len(valid) < N_LONG + N_SHORT:
        return pd.Series(dtype=float)
    sorted_ = valid.sort_values(ascending=False)
    top_3 = sorted_.head(N_LONG).index
    bot_3 = sorted_.tail(N_SHORT).index
    weights = pd.Series(0.0, index=valid.index)
    weights.loc[top_3] = +1.0 / N_LONG / 2.0   # +1/6 each
    weights.loc[bot_3] = -1.0 / N_SHORT / 2.0  # -1/6 each
    return weights


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


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────
def run_aa_backtest() -> AABacktestResult:
    weekly = load_sector_panel()
    weekly_returns_per_sector = weekly.pct_change()

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
            r_t = weekly_returns_per_sector.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
            weekly_tc[week] = 0.0
        else:
            weekly_gross_ret[week] = 0.0
            weekly_tc[week] = 0.0

        # Rebalance
        if week in rebal_set:
            sig = compute_signal(weekly, week)
            new_w = select_ls_positions(sig)
            if new_w.empty:
                n_skipped += 1
                continue
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
            weekly_tc[week] = tc
            positions_history[week] = new_w.copy()
            current_weights = new_w

    gross = pd.Series(weekly_gross_ret, name="aa_gross")
    tcs   = pd.Series(weekly_tc,        name="aa_tc")
    net   = (gross - tcs).rename("aa_net")

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

    notes = []
    if n_skipped:
        notes.append(f"{n_skipped} rebalance(s) skipped (insufficient signals)")

    return AABacktestResult(
        weekly_returns_gross  = gross,
        weekly_returns_net    = net,
        weekly_tc_drag        = tcs,
        rebalance_dates       = rebal_list,
        n_weeks               = len(weeks),
        n_rebalances          = len(rebal_list),
        avg_turnover          = avg_to,
        notes                 = notes,
    )


def save_aa_parquet(
    result:    AABacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_aa_sector_mom_ls_weekly.parquet",
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
