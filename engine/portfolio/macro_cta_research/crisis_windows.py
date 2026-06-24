"""
Locked crisis window definitions for Path P/Q/S/T/U gate G4 evaluation.

Per spec §2.6 G4 "Crisis-positive ≥ 2 of 3". These 3 windows are locked
across all 5 specs in the horse race — same windows, same gate threshold,
so head-to-head comparison is apples-to-apples.

Selection rationale:
  - 2018-Q4: VIX spike + S&P 500 -19% from peak — first major equity stress
             post-2008. Tests defensive sleeve in equity drawdown.
  - 2020-COVID: VIX hit 80+ · global liquidity crisis · 1-month -34% S&P. Tests
                tail-risk hedge under extreme stress.
  - 2022 full year: Fed tightening · rates trended UP · classic 60/40 broken
                    (TLT -29% same year as SPY -19%) · stress test for
                    "what if rates AND equities both fall."

Dates are NYSE trading days (close-to-close inclusive).
"""
from __future__ import annotations

import datetime
from typing import Final


CRISIS_WINDOWS: Final[dict[str, tuple[datetime.date, datetime.date]]] = {
    "2018_Q4":    (datetime.date(2018, 10,  1), datetime.date(2018, 12, 31)),
    "2020_COVID": (datetime.date(2020,  2, 19), datetime.date(2020,  3, 23)),
    "2022_full":  (datetime.date(2022,  1,  3), datetime.date(2022, 12, 30)),
}


def crisis_window_returns(weekly_returns, crisis_key: str):
    """Extract weekly returns within a given crisis window. Returns pd.Series."""
    if crisis_key not in CRISIS_WINDOWS:
        raise ValueError(f"unknown crisis window: {crisis_key}")
    start, end = CRISIS_WINDOWS[crisis_key]
    mask = (weekly_returns.index.date >= start) & (weekly_returns.index.date <= end)
    return weekly_returns.loc[mask]


def crisis_window_cum_return(weekly_returns, crisis_key: str) -> float:
    """Cumulative return over the crisis window. (1+r).prod() - 1."""
    win = crisis_window_returns(weekly_returns, crisis_key)
    if win.empty:
        return float("nan")
    return float((1.0 + win).prod() - 1.0)


def crisis_positive_count(weekly_returns) -> dict:
    """Returns dict: {crisis_key: cum_return} + 'n_positive' summary."""
    out = {}
    n_pos = 0
    for key in CRISIS_WINDOWS:
        cum = crisis_window_cum_return(weekly_returns, key)
        out[key] = cum
        if cum > 0:
            n_pos += 1
    out["n_positive"] = n_pos
    out["n_total"]    = len(CRISIS_WINDOWS)
    return out
