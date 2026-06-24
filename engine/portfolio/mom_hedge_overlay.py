"""engine/portfolio/mom_hedge_overlay.py — MOM-factor short hedge sleeve.

Built 2026-05-30 per BARRA Phase 3 finding that the deployed book has
53% of variance from MOM-factor exposure ([[project-barra-phase-chain-
2026-05-30]] L1 factor budget). Two prior anti-MOM candidates (STR and
BAB-lite) BOTH failed H9 orthogonality (PILE_ON, FULLY_ALIGNED_WARN)
because they overlap in the same sector concentrations the book already
holds. The ONLY way to actually reduce MOM-risk is a direct short on a
momentum-factor ETF.

DESIGN — INSTITUTIONAL-HONEST:
  Instrument:  MTUM (iShares MSCI USA Momentum Factor ETF) short
  Sizing:      static 100% short notional vs the hedge sleeve weight
  Rebalance:   monthly (matches D_PEAD / carry / TSMOM cadence)
  Cost model:  borrow_cost_bps_per_yr 35bp + 2bp/side spread
                                       (FIM 2015 ETF range + IB 2024 quotes)
  Cost is NEGATIVE drag on hedge returns.

Purpose (per user 2026-05-30): RISK MANAGEMENT, not Sharpe maximization.
Expected effect on combined book:
  - Reduces 53% MOM-variance contribution
  - Costs ~1.2%/yr MOM premium given up + ~0.4%/yr operational drag
  - Net Sharpe: +0 to +0.05 (limited, but resilience to MOM crashes
    in 2020-Mar style events is the real win)

OUTPUT: monthly return series with cost applied. POSITIVE return =
hedge profitable (= MTUM went down).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Locked institutional cost params per [[feedback-cost-model-rigor-almgren-
# not-scalar-2026-05-30]]:
BORROW_COST_ANNUAL_BPS:  float = 35.0   # MTUM borrow ~30-40bp, mid-range
HALF_SPREAD_BPS:         float = 2.0    # 1-sided
TC_BPS_PER_REBAL:        float = 5.0    # round-trip
ANNUAL_REBALANCES:       int   = 12
TICKER:                  str   = "MTUM"
WINDOW_START:            str   = "2013-04-30"   # MTUM inception 2013-04-16


@dataclass(frozen=True)
class MomHedgeBacktestResult:
    monthly_returns_gross:   pd.Series   # before cost (= -MTUM monthly ret)
    monthly_returns_net:     pd.Series   # after borrow + TC
    monthly_borrow_drag:     pd.Series
    monthly_tc_drag:         pd.Series
    n_months:                int


def fetch_mtum_monthly() -> pd.Series:
    """Fetch MTUM daily close from yfinance, resample to monthly ret."""
    import yfinance as _yf
    df = _yf.download(TICKER, start=WINDOW_START, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance fetch for MTUM returned empty")
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    close.index = pd.to_datetime(close.index)
    monthly_ret = close.resample("ME").last().pct_change().dropna()
    monthly_ret.name = "mtum_ret"
    return monthly_ret


def run_mom_hedge_backtest() -> MomHedgeBacktestResult:
    """Build MTUM short overlay monthly return series with realistic
    institutional cost: borrow fee (continuous) + spread (monthly rebal).
    """
    mtum_ret = fetch_mtum_monthly()
    # Short hedge return: profit when MTUM falls
    gross = (-mtum_ret).rename("hedge_gross")
    # Monthly borrow cost (continuous annual converted to monthly)
    monthly_borrow = (BORROW_COST_ANNUAL_BPS / 10_000.0) / 12.0
    borrow_drag = pd.Series(monthly_borrow, index=gross.index,
                                name="borrow_drag")
    # Monthly rebalance spread cost — 2x half_spread + TC_BPS_PER_REBAL
    rebal_cost_bps = 2.0 * HALF_SPREAD_BPS + TC_BPS_PER_REBAL
    monthly_tc = (rebal_cost_bps / 10_000.0)
    tc_drag = pd.Series(monthly_tc, index=gross.index, name="tc_drag")

    net = (gross - borrow_drag - tc_drag).rename("hedge_net")

    return MomHedgeBacktestResult(
        monthly_returns_gross=gross,
        monthly_returns_net=net,
        monthly_borrow_drag=borrow_drag,
        monthly_tc_drag=tc_drag,
        n_months=int(len(gross)),
    )


def build_mom_hedge_book() -> pd.Series:
    """Adapter compatible with combined_book sleeve builder pattern.
    Returns monthly net return series."""
    r = run_mom_hedge_backtest()
    return r.monthly_returns_net.rename("mom_hedge")
