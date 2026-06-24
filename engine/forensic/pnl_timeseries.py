"""
engine/forensic/pnl_timeseries.py — 30/60/90d trailing P&L per strategy + sleeve.

Read PaperTradeStrategyLog daily aggregate net returns, compute trailing
cumulative P&L for 30/60/90 day windows. Useful for context: is current DD
a sudden spike or continuation of trend?

Auto-gate: each window requires its lookback days. 30d window unlocks day-30,
60d day-60, etc. Day-1 emits all 3 as INSUFFICIENT_DATA.

DOCTRINE: forensic layer, deterministic compute.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_pnl_trailing(
    as_of:    datetime.date,
    windows:  tuple[int, ...] = (30, 60, 90),
) -> dict:
    """Compute trailing cumulative P&L for each window per strategy + sleeve.

    Returns dict with per-window cum_pnl, max_dd, sharpe, status.
    """
    from engine.db_models import PaperTradeStrategyLog, SessionFactory

    max_lookback = max(windows)
    start = as_of - datetime.timedelta(days=int(max_lookback * 1.5))

    s = SessionFactory()
    try:
        rows = (s.query(PaperTradeStrategyLog)
                  .filter(PaperTradeStrategyLog.date >= start)
                  .filter(PaperTradeStrategyLog.date <= as_of)
                  .all())
        if not rows:
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": "no PaperTradeStrategyLog rows yet",
                "have":   0,
                "need":   min(windows),
                "eta_unlock": (as_of + datetime.timedelta(days=min(windows))).isoformat(),
            }
        data = pd.DataFrame([
            {"date": r.date, "strategy_name": r.strategy_name, "sleeve_id": r.sleeve_id,
             "daily_net_return": r.daily_net_return}
            for r in rows if r.daily_net_return is not None
        ])
    finally:
        s.close()

    if data.empty:
        return {
            "status":     "INSUFFICIENT_DATA",
            "reason":     "PaperTradeStrategyLog rows have no daily_net_return yet",
            "have":       0,
            "need":       min(windows),
            "eta_unlock": (as_of + datetime.timedelta(days=min(windows))).isoformat(),
        }

    n_total_days = data["date"].nunique()
    per_window: dict[str, dict] = {}

    for w in windows:
        wstart = as_of - datetime.timedelta(days=int(w * 1.5))
        win_data = data[data["date"] >= wstart]
        n_days = win_data["date"].nunique()
        if n_days < w * 0.7:  # require ~70% coverage
            per_window[f"{w}d"] = {
                "status":     "INSUFFICIENT_DATA",
                "have":       n_days,
                "need":       w,
                "eta_unlock": (as_of + datetime.timedelta(days=w - n_days)).isoformat(),
            }
            continue

        per_strategy: dict[str, dict] = {}
        for strat, sub in win_data.groupby("strategy_name"):
            sub = sub.sort_values("date")
            rets = sub["daily_net_return"].dropna()
            if rets.empty:
                continue
            cum_pnl = float((1.0 + rets).prod() - 1.0)
            # Max drawdown
            cum = (1.0 + rets).cumprod()
            peak = cum.cummax()
            max_dd = float(((cum - peak) / peak).min())
            per_strategy[strat] = {
                "cum_pnl": round(cum_pnl, 6),
                "max_dd":  round(max_dd, 6),
                "n_days":  int(n_days),
            }
        per_window[f"{w}d"] = {
            "status":       "OK",
            "n_days":       n_days,
            "per_strategy": per_strategy,
        }

    any_ok = any(d.get("status") == "OK" for d in per_window.values())
    return {
        "status":         "OK" if any_ok else "INSUFFICIENT_DATA",
        "as_of":          as_of.isoformat(),
        "per_window":     per_window,
        "math_anchor":    "Cumulative geometric return + max drawdown (peak-to-trough)",
    }
