"""engine/validation/scenario_stress.py — book stress testing on the holdings return panel.

Two complementary, data-grounded views of "what could hurt", beyond the single VaR/ES number:

  1. HISTORICAL replay — relive the available history with TODAY's weights: worst/best cumulative
     1d / 5d / 20d windows (+ the dates), and which positions drove the single worst day. Honest
     about the sample window (it can't include crises outside the panel's span).

  2. MARKET-BETA SHOCK — an instantaneous 1-factor (equity-market) shock: per-holding β to SPY from
     the panel, book P&L under equity −20% / −10% / +10%. Labelled as a 1-factor approximation (no
     rates/credit/vol legs — those need a full factor model, deliberately out of scope).

Reuses the cached holdings panel built by engine.validation.risk_contribution. No look-ahead: all
betas/returns are in-sample historical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SHOCKS = (-0.20, -0.10, 0.10)   # equity-market moves to stress


def compute_scenarios(weights: dict, panel: pd.DataFrame, *, market: str = "SPY") -> dict:
    """Historical worst-window replay + worst-day attribution + market-beta shocks for `weights`."""
    if panel is None or panel.empty:
        return {"available": False, "reason": "no returns panel"}
    cols = [t for t, w in weights.items() if t in panel.columns and abs(w) > 1e-9]
    if len(cols) < 2:
        return {"available": False, "reason": "fewer than 2 holdings have return history"}
    R = panel[cols].fillna(0.0)
    R = R.tail(252 * 2)                               # up to ~2y of daily history
    w = pd.Series({t: weights[t] for t in cols})
    book = R[cols].mul(w, axis=1).sum(axis=1)         # daily book return series

    def _window(k: int, worst: bool) -> dict:
        roll = book.rolling(k).sum().dropna()
        if roll.empty:
            return {"k": k, "ret": None, "end_date": None}
        idx = roll.idxmin() if worst else roll.idxmax()
        return {"k": k, "ret": round(float(roll.loc[idx]), 6), "end_date": str(idx)[:10]}

    worst = {f"{k}d": _window(k, True) for k in (1, 5, 20)}
    best = {f"{k}d": _window(k, False) for k in (1, 5, 20)}

    # worst-day loss attribution: w_i · r_i on the single worst day
    wd = book.idxmin()
    day = R.loc[wd]
    contrib = (day * w).sort_values()                  # most negative first
    attribution = [{"ticker": t, "contrib": round(float(contrib[t]), 6), "ret": round(float(day[t]), 6)}
                   for t in contrib.index[:6]]

    out = {
        "available": True,
        "n_obs": int(len(book)),
        "period": [str(book.index[0])[:10], str(book.index[-1])[:10]],
        "worst": worst, "best": best,
        "worst_day": {"date": str(wd)[:10], "book_ret": round(float(book.loc[wd]), 6), "attribution": attribution},
    }

    # market-beta (1-factor) shocks — only if the market proxy is in the panel
    if market in panel.columns:
        mkt = panel[market].reindex(R.index).fillna(0.0)
        var_m = float(mkt.var())
        if var_m > 1e-12:
            betas = {t: float(np.cov(R[t].values, mkt.values)[0, 1] / var_m) for t in cols}
            book_beta = float(sum(w[t] * betas[t] for t in cols))
            out["market"] = {
                "proxy": market, "book_beta": round(book_beta, 4),
                "shocks": [{"mkt_move": s, "book_pnl": round(book_beta * s, 6)} for s in _SHOCKS],
                "note": "1-factor equity-beta shock; no rates/credit/vol legs.",
            }
    return out
