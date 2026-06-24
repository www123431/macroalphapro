"""
engine/d_pead_plus/backtest.py — walk-forward backtest harness.

Spec id=74 §2.7 LOCK:
  - Cross-section rank within quarter by combined score
  - Long: top decile equal-weight; Short: bottom decile equal-weight
  - Dollar-neutral, vol-target 10% ann, max-leverage 2.0, max-name 5%
  - Holding 60 trading days post-rdq entry; quarterly rebalance
  - TC roundtrip 30bp (SS-Tier-1 standing rule)

Mirrors Path D structure exactly for matched A/B test vs D-PEAD baseline.

DOCTRINE: Decision-layer module. ZERO LLM calls.
Enforced by engine.d_pead_plus.doctrine.audit_decision_layer_imports().
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec id=74 §2.7 LOCKED (matches Path D)
HOLDING_PERIOD_TRADING_DAYS: int   = 60
DECILE_LONG:                 int   = 10
DECILE_SHORT:                int   = 1
VOL_TARGET_ANNUAL:           float = 0.10
MAX_LEVERAGE:                float = 2.0
MAX_NAME_WEIGHT:             float = 0.05
TC_BPS_ROUNDTRIP:            float = 30.0     # SS-Tier-1 standing rule
TRADING_DAYS_PER_YEAR:       int   = 252

# Cache
CACHE_DIR = Path("data/d_pead_plus")
BACKTEST_OUTPUT_PATH = CACHE_DIR / "v1_backtest_daily.parquet"


@dataclass(frozen=True)
class BacktestResult:
    """Single strategy backtest output."""
    strategy_name:    str               # 'd_pead_baseline' or 'd_pead_plus'
    daily_returns:    pd.Series         # net of TC, vol-targeted
    n_events:         int               # total firm-quarter events
    n_long_avg:       float             # avg positions in long leg
    n_short_avg:      float             # avg positions in short leg
    ann_return_net:   float
    ann_vol:          float
    sharpe:           float
    max_drawdown:     float
    n_trading_days:   int


def _build_event_holdings(
    panel:            pd.DataFrame,        # cols: permno, rdq, long_flag, short_flag
    daily_returns:    pd.DataFrame,        # date × permno daily returns
    holding_days:     int                  = HOLDING_PERIOD_TRADING_DAYS,
) -> pd.DataFrame:
    """For each event (permno, rdq), build daily position weights.

    Returns DataFrame indexed by date, columns = permno, values = position weight
    (+ for long, − for short, 0 otherwise).

    Each event held for `holding_days` trading days starting from rdq+1 (next trading day).
    """
    if panel.empty or daily_returns.empty:
        return pd.DataFrame()

    trading_days_index = pd.DatetimeIndex(daily_returns.index)
    panel = panel[(panel["long_flag"] == 1) | (panel["short_flag"] == 1)].copy()
    panel["rdq"] = pd.to_datetime(panel["rdq"])

    # Initialize weights df (all zeros)
    permno_cols = sorted(set(panel["permno"].tolist()) & set(daily_returns.columns.tolist()))
    weights = pd.DataFrame(0.0, index=trading_days_index, columns=permno_cols)

    # For each event, set weights over holding window
    for _, row in panel.iterrows():
        permno = int(row["permno"])
        if permno not in weights.columns:
            continue
        rdq = pd.Timestamp(row["rdq"])
        # entry next trading day after rdq
        future_days = trading_days_index[trading_days_index > rdq]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]
        # exit after holding_days trading days
        entry_pos  = trading_days_index.get_loc(entry_date)
        exit_pos   = min(entry_pos + holding_days, len(trading_days_index))
        active_dates = trading_days_index[entry_pos:exit_pos]

        sign = +1.0 if row["long_flag"] == 1 else -1.0
        # Pre-vol-target raw weight: 1/N where N = avg concurrent positions; we'll vol-target later
        # Mark with raw signed indicator first
        for d in active_dates:
            weights.at[d, permno] += sign

    return weights


def _normalize_to_decile_spread(weights: pd.DataFrame) -> pd.DataFrame:
    """Per day: long/short normalized to 1/N_long and 1/N_short equal-weight."""
    import numpy as np
    daily_long_count  = (weights > 0).sum(axis=1).replace(0, 1).values
    daily_short_count = (weights < 0).sum(axis=1).replace(0, 1).values
    long_inv  = (1.0 / daily_long_count )[:, None]   # shape (n_days, 1)
    short_inv = (1.0 / daily_short_count)[:, None]
    w_vals = weights.values
    out_vals = np.where(
        w_vals > 0, np.broadcast_to(long_inv,  w_vals.shape),
        np.where(w_vals < 0, -np.broadcast_to(short_inv, w_vals.shape), 0.0),
    )
    return pd.DataFrame(out_vals, index=weights.index, columns=weights.columns)


def _vol_target_scale(daily_returns_pre: pd.Series, target_vol_ann: float = VOL_TARGET_ANNUAL,
                       max_leverage: float = MAX_LEVERAGE) -> tuple[pd.Series, float]:
    """Apply ex-post vol target — scale daily returns to target ann vol.

    For mirror-D-PEAD comparison, we use the FULL-SAMPLE realized vol
    (Path D convention; assumes ex-ante vol estimation matched between baseline
    and plus). Cap scaling factor at max_leverage.
    """
    realized_vol = daily_returns_pre.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    if realized_vol <= 1e-9:
        return daily_returns_pre, 1.0
    scalar = target_vol_ann / realized_vol
    scalar = min(scalar, max_leverage)
    return daily_returns_pre * scalar, scalar


def _apply_tc(daily_returns: pd.Series, weights: pd.DataFrame,
              tc_bps_roundtrip: float = TC_BPS_ROUNDTRIP) -> pd.Series:
    """Subtract TC at each rebalance event (weight change day)."""
    weight_diff_abs = weights.diff().abs().sum(axis=1).fillna(0)
    # First day: full position cost
    weight_diff_abs.iloc[0] = weights.iloc[0].abs().sum() if len(weights) > 0 else 0
    tc_drag = weight_diff_abs * (tc_bps_roundtrip / 2.0 / 10_000.0)  # one-way bps
    return daily_returns - tc_drag


def run_strategy_backtest(
    panel:           pd.DataFrame,        # cols: permno, rdq, long_flag, short_flag
    daily_returns:   pd.DataFrame,        # date × permno
    strategy_name:   str,
) -> BacktestResult:
    """Mirror-Path-D structure: build positions, daily P&L, vol-target, TC, stats."""
    weights = _build_event_holdings(panel, daily_returns)
    if weights.empty:
        logger.warning("run_strategy_backtest %s: empty weights", strategy_name)
        return BacktestResult(strategy_name=strategy_name, daily_returns=pd.Series(dtype=float),
                               n_events=0, n_long_avg=0.0, n_short_avg=0.0,
                               ann_return_net=float("nan"), ann_vol=float("nan"),
                               sharpe=float("nan"), max_drawdown=float("nan"),
                               n_trading_days=0)

    weights_norm = _normalize_to_decile_spread(weights)
    # Align to daily_returns (intersect tickers)
    common_cols = sorted(set(weights_norm.columns) & set(daily_returns.columns))
    common_idx  = weights_norm.index.intersection(daily_returns.index)
    w = weights_norm.loc[common_idx, common_cols]
    r = daily_returns.loc[common_idx, common_cols]

    # Daily P&L = sum over positions (w[t] × ret[t]); use lagged w to avoid look-ahead
    daily_pnl_pre = (w.shift(1).fillna(0) * r).sum(axis=1)

    # TC drag
    daily_pnl_post_tc = _apply_tc(daily_pnl_pre, w)

    # Vol target
    daily_pnl_voltarget, vol_scalar = _vol_target_scale(daily_pnl_post_tc)

    # Stats
    ann_ret = float(daily_pnl_voltarget.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(daily_pnl_voltarget.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-9 else float("nan")
    cum = (1 + daily_pnl_voltarget.fillna(0)).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())

    n_events = int(((panel["long_flag"] == 1) | (panel["short_flag"] == 1)).sum())
    n_long_avg  = float((w > 0).sum(axis=1).mean())
    n_short_avg = float((w < 0).sum(axis=1).mean())

    logger.info("%s: Sharpe=%.3f ann_ret=%.2f%% vol=%.2f%% maxDD=%.2f%% n_events=%d "
                "n_long_avg=%.0f n_short_avg=%.0f vol_scalar=%.3f",
                strategy_name, sharpe, ann_ret*100, ann_vol*100, max_dd*100,
                n_events, n_long_avg, n_short_avg, vol_scalar)

    return BacktestResult(
        strategy_name    = strategy_name,
        daily_returns    = daily_pnl_voltarget,
        n_events         = n_events,
        n_long_avg       = n_long_avg,
        n_short_avg      = n_short_avg,
        ann_return_net   = ann_ret,
        ann_vol          = ann_vol,
        sharpe           = sharpe,
        max_drawdown     = max_dd,
        n_trading_days   = int(len(daily_pnl_voltarget)),
    )


def save_backtest_daily(result_baseline: BacktestResult, result_plus: BacktestResult) -> None:
    """Save paired daily returns to parquet for downstream Bootstrap CI."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "d_pead_baseline": result_baseline.daily_returns,
        "d_pead_plus":     result_plus.daily_returns,
    })
    df.to_parquet(BACKTEST_OUTPUT_PATH)
    logger.info("Saved paired backtest daily returns to %s (n=%d days)",
                BACKTEST_OUTPUT_PATH, len(df))


def load_backtest_daily() -> Optional[pd.DataFrame]:
    """Load paired backtest daily returns from parquet."""
    if not BACKTEST_OUTPUT_PATH.exists():
        return None
    return pd.read_parquet(BACKTEST_OUTPUT_PATH)


def get_locked_constants() -> dict:
    return {
        "HOLDING_PERIOD_TRADING_DAYS": HOLDING_PERIOD_TRADING_DAYS,
        "DECILE_LONG":                 DECILE_LONG,
        "DECILE_SHORT":                DECILE_SHORT,
        "VOL_TARGET_ANNUAL":           VOL_TARGET_ANNUAL,
        "MAX_LEVERAGE":                MAX_LEVERAGE,
        "MAX_NAME_WEIGHT":             MAX_NAME_WEIGHT,
        "TC_BPS_ROUNDTRIP":            TC_BPS_ROUNDTRIP,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== D-PEAD-Plus Backtest — Locked Constants ===")
    for k, v in get_locked_constants().items():
        print(f"  {k}: {v}")
