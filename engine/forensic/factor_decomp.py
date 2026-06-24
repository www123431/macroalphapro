"""
engine/forensic/factor_decomp.py — FF5 factor decomposition for trade P&L.

Fama-French 2015 5-factor model: decompose each trade's realized return into
Mkt-Rf / SMB / HML / RMW / CMA exposure + idiosyncratic residual.

Tells investigator "this -15% is mostly Mkt-Rf beta (-12%) + sector (-1%) +
idiosyncratic (-2%)" vs "this -15% is mostly idiosyncratic (signal-related)".

Auto-gate: requires trade_df with realized returns + sufficient yfinance
history for factor proxies (or fallback to Ken French data library).

DOCTRINE: forensic layer, no LLM, no decision feedback.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Proxy ETFs for factors (yfinance-available; not exact Ken French but close)
_FACTOR_PROXIES: dict[str, str] = {
    "Mkt-Rf": "SPY",     # Market proxy
    "SMB":    "IWM",     # Small-cap proxy
    "HML":    "IWN",     # Value proxy (Russell 2000 Value)
    "RMW":    "QUAL",    # Quality proxy
    "CMA":    "MTUM",    # Momentum (proxy for CMA partial)
    "RF":     "BIL",     # 1-3mo T-bill (risk-free proxy)
}


def compute_ff5_decomp(
    trade_df: pd.DataFrame,
    as_of:    datetime.date,
    horizon:  int = 5,
) -> dict:
    """For each trade with realized return, decompose into FF5 factor exposures.

    Note: Uses ETF proxies for FF5 factors (SPY/IWM/IWN/QUAL/MTUM/BIL) since
    full Ken French data needs separate fetch. For institutional rigor, replace
    with direct fetch from Kenneth French Data Library.

    Auto-gate: needs ≥3 trades with realized returns AND yfinance factor data.
    """
    ret_col = f"realized_{horizon}d_return"

    if trade_df.empty or ret_col not in trade_df.columns:
        return {"status": "INSUFFICIENT_DATA", "reason": "no realized returns yet", "have": 0, "need": 3}

    df = trade_df.dropna(subset=[ret_col]).copy()
    if len(df) < 3:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": f"need ≥3 trades with realized returns, have {len(df)}",
            "have":   len(df),
            "need":   3,
        }

    # Fetch factor proxy returns for the same window
    try:
        import yfinance as yf
        proxy_tickers = list(_FACTOR_PROXIES.values())
        start = as_of
        end   = as_of + datetime.timedelta(days=horizon + 7)
        prices = yf.download(proxy_tickers, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"]
        else:
            close = prices[["Close"]].rename(columns={"Close": proxy_tickers[0]})

        # Compute T to T+horizon return per proxy
        factor_returns: dict[str, float] = {}
        for fname, tkr in _FACTOR_PROXIES.items():
            if tkr not in close.columns:
                continue
            series = close[tkr].dropna()
            if len(series) < horizon + 1:
                continue
            entry = float(series.iloc[0])
            exit_ = float(series.iloc[min(horizon, len(series) - 1)])
            if entry > 0:
                factor_returns[fname] = (exit_ - entry) / entry
    except Exception as exc:
        logger.warning("factor_decomp yfinance fetch failed: %s", exc)
        return {"status": "INSUFFICIENT_DATA", "reason": f"factor proxy fetch failed: {exc}",
                "have": len(df), "need": "factor proxy data"}

    if not factor_returns:
        return {"status": "INSUFFICIENT_DATA", "reason": "no factor proxy returns",
                "have": 0, "need": "factor proxy data"}

    mkt_ret = factor_returns.get("Mkt-Rf", 0.0) - factor_returns.get("RF", 0.0)

    # Simple decomposition: market beta ≈ trade return / mkt return (assume β=1 baseline)
    # Full OLS regression requires multi-period observations per ticker — defer to Phase 3
    # Here we provide point-in-time approximate decomp: subtract market exposure from trade return
    n_trades = len(df)
    avg_trade_ret = float(df[ret_col].mean())

    if abs(mkt_ret) > 1e-6:
        approx_market_component = mkt_ret  # assume β=1 (institutional version would regress)
        approx_idio_component   = avg_trade_ret - approx_market_component
    else:
        approx_market_component = 0.0
        approx_idio_component   = avg_trade_ret

    # Per-strategy idiosyncratic share
    per_strategy_idio: dict[str, dict] = {}
    for strat, sub in df.groupby("strategy_name"):
        strat_ret = float((sub["weight"] * sub[ret_col]).sum())
        per_strategy_idio[strat] = {
            "strategy_realized":     round(strat_ret, 6),
            "approx_market_share":   round(approx_market_component * sub["weight"].abs().sum(), 6),
            "approx_idio_residual":  round(strat_ret - approx_market_component * sub["weight"].abs().sum(), 6),
        }

    return {
        "status":            "OK",
        "horizon_days":      horizon,
        "n_trades":          n_trades,
        "factor_returns":    {k: round(v, 6) for k, v in factor_returns.items()},
        "avg_trade_ret":     round(avg_trade_ret, 6),
        "approx_market_component": round(approx_market_component, 6),
        "approx_idio_component":   round(approx_idio_component, 6),
        "per_strategy":      per_strategy_idio,
        "math_anchor":       "Fama-French 2015 FF5; β=1 single-period approximation",
        "limitation":        "Full OLS requires multi-period rolling regression (Phase 3 upgrade)",
    }
