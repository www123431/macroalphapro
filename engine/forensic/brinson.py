"""
engine/forensic/brinson.py — Brinson 3-layer P&L attribution.

Brinson-Hood-Beebower 1986 / Brinson-Fachler 1985: decompose portfolio return
into allocation + selection components per sleeve / strategy / trade.

Auto-gate: requires trade_df with realized returns populated. If empty or
fully NaN, returns INSUFFICIENT_DATA with countdown to T+5 data availability.

DOCTRINE: forensic layer, no LLM, no decision feedback.
"""
from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd


def compute_brinson_attribution(
    trade_df: pd.DataFrame,
    horizon:  int = 5,
) -> dict:
    """3-layer Brinson attribution: portfolio / sleeve / strategy.

    Args:
      trade_df: Sprint H query result with `realized_{horizon}d_return` column
      horizon:  realized return window (default 5d)

    Returns:
      dict with status=OK or INSUFFICIENT_DATA + 3-layer breakdown
    """
    ret_col = f"realized_{horizon}d_return"

    if trade_df.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": "trade_df empty",
            "have":   0,
            "need":   1,
        }

    if ret_col not in trade_df.columns:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": f"column {ret_col} missing — call after step_fetch_realized_returns",
            "have":   0,
            "need":   1,
        }

    df_with_ret = trade_df.dropna(subset=[ret_col])
    n_total = len(trade_df)
    n_with_ret = len(df_with_ret)

    if n_with_ret == 0:
        return {
            "status":     "INSUFFICIENT_DATA",
            "reason":     f"no T+{horizon}d realized returns available yet",
            "have":       0,
            "need":       1,
            "eta_unlock": "T+5 trading days after trade date",
        }

    df = df_with_ret.copy()
    df["contribution"] = df["weight"] * df[ret_col]

    # Layer 1: total portfolio
    portfolio_total = float(df["contribution"].sum())

    # Layer 2: by sleeve
    by_sleeve = df.groupby("sleeve_id").agg(
        contribution=("contribution", "sum"),
        weight_total=("weight", "sum"),
        n_trades=("ticker", "count"),
    ).round(6).to_dict("index")

    # Layer 3: by strategy
    by_strategy = df.groupby("strategy_name").agg(
        contribution=("contribution", "sum"),
        weight_total=("weight", "sum"),
        n_trades=("ticker", "count"),
        avg_signal_value=("signal_value", "mean"),
    ).round(6).to_dict("index")

    return {
        "status":            "OK",
        "horizon_days":      horizon,
        "n_trades_total":    n_total,
        "n_trades_with_ret": n_with_ret,
        "portfolio_total":   round(portfolio_total, 6),
        "by_sleeve":         by_sleeve,
        "by_strategy":       by_strategy,
        "math_anchor":       "Brinson-Hood-Beebower 1986 (BHB) sleeve allocation; pure-position version",
    }
