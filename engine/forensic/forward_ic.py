"""
engine/forensic/forward_ic.py — Forward IC degradation test.

Computes Spearman rank correlation between signal_value (from Sprint H trade
log) and realized_60d_return per strategy. Compares to spec-locked in-sample
IC to detect signal decay.

Auto-gate: requires ≥60 trading days of forward data (per-trade horizon for
D-PEAD is 60d; Path N is 5d; others vary). Min 30 trade-realized pairs for
statistical power.

DOCTRINE: forensic layer; statistical evidence for human review.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MIN_PAIRS_FOR_TEST: int = 30
MIN_HORIZON_FOR_60D_IC: int = 60   # days


# Spec-locked in-sample IC per strategy (from spec verdict files)
# These are approximate — real spec lock values stored in spec verdict JSONs
SPEC_LOCKED_IC: dict[str, float] = {
    "K1_BAB":     0.045,    # BAB IC typical (Frazzini-Pedersen 2018 range)
    "D_PEAD":     0.062,    # PEAD IC (Bernard-Thomas 1989 + Path D verdict)
    "PATH_N":     0.040,    # Reconstitution event IC (small sample)
    "CTA_PQTIX":  0.000,    # No signal — N/A
}


def compute_forward_ic_per_strategy(
    as_of:    datetime.date,
    horizon:  int = 60,
) -> dict:
    """Compute forward IC = Spearman corr(signal_value, realized_60d_return) per strategy.

    Queries Sprint H PaperTradeTradeLog for trades old enough to have realized
    60d returns (trade_date ≤ as_of - 60d). Computes Spearman IC per strategy.
    Compares to spec-locked in-sample IC.

    Returns INSUFFICIENT_DATA if no strategy has ≥30 trade-realized pairs.
    """
    from engine.portfolio.attribution_logger import query_trade_log

    # Trades that could have 60d realized by now: trade_date ≤ as_of - 60d
    cutoff_old = as_of - datetime.timedelta(days=horizon)

    df = query_trade_log(date_end=cutoff_old)
    if df.empty:
        return {
            "status":     "INSUFFICIENT_DATA",
            "reason":     f"no Sprint H trades older than {horizon}d",
            "have":       0,
            "need":       MIN_PAIRS_FOR_TEST,
            "eta_unlock": (as_of + datetime.timedelta(days=horizon + 30)).isoformat(),
        }

    # Fetch realized 60d returns for these trades (per-ticker yfinance)
    # For each (date, ticker), compute return from date to date+60d
    try:
        import yfinance as yf
        unique_pairs = df[["date", "ticker"]].drop_duplicates()
        unique_pairs = unique_pairs[~unique_pairs["ticker"].str.startswith("permno_", na=False)]
        unique_pairs = unique_pairs[unique_pairs["ticker"] != "PQTIX"]
        if unique_pairs.empty:
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": "no real-ticker trades old enough (only permno/PQTIX entries)",
                "have":   0, "need": MIN_PAIRS_FOR_TEST,
            }
    except Exception as exc:
        return {"status": "INSUFFICIENT_DATA", "reason": f"yfinance setup failed: {exc}",
                "have": 0, "need": MIN_PAIRS_FOR_TEST}

    realized_returns: list[dict] = []
    # Group by trade date to batch yfinance calls
    for trade_date, group in unique_pairs.groupby("date"):
        tickers = group["ticker"].tolist()
        start = trade_date
        end   = trade_date + datetime.timedelta(days=horizon + 14)
        try:
            prices = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices["Close"]
            else:
                close = prices[["Close"]].rename(columns={"Close": tickers[0]})
        except Exception as exc:
            logger.warning("forward_ic yfinance fail for %s: %s", trade_date, exc)
            continue

        for tk in tickers:
            if tk not in close.columns:
                continue
            series = close[tk].dropna()
            if len(series) < horizon:
                continue
            entry = float(series.iloc[0])
            exit_ = float(series.iloc[min(horizon, len(series) - 1)])
            if entry > 0:
                realized_returns.append({
                    "date": trade_date,
                    "ticker": tk,
                    "realized_60d_return": (exit_ - entry) / entry,
                })

    if not realized_returns:
        return {"status": "INSUFFICIENT_DATA", "reason": "no yfinance data for 60d realized",
                "have": 0, "need": MIN_PAIRS_FOR_TEST}

    ret_df = pd.DataFrame(realized_returns)
    merged = df.merge(ret_df, on=["date", "ticker"], how="inner")

    if len(merged) < MIN_PAIRS_FOR_TEST:
        return {
            "status":     "INSUFFICIENT_DATA",
            "reason":     f"only {len(merged)} trade-realized pairs available",
            "have":       len(merged),
            "need":       MIN_PAIRS_FOR_TEST,
            "eta_unlock": (as_of + datetime.timedelta(days=30)).isoformat(),
        }

    per_strategy: dict[str, dict] = {}
    for strat, sub in merged.groupby("strategy_name"):
        if len(sub) < MIN_PAIRS_FOR_TEST:
            per_strategy[strat] = {
                "status":  "INSUFFICIENT_DATA",
                "have":    len(sub),
                "need":    MIN_PAIRS_FOR_TEST,
            }
            continue
        # Drop NaNs in signal_value (e.g., CTA)
        sub_valid = sub.dropna(subset=["signal_value", "realized_60d_return"])
        if len(sub_valid) < MIN_PAIRS_FOR_TEST:
            per_strategy[strat] = {
                "status":  "INSUFFICIENT_DATA",
                "reason":  "too many NaN signal_value (likely CTA passive)",
                "have":    len(sub_valid),
                "need":    MIN_PAIRS_FOR_TEST,
            }
            continue
        ic = float(sub_valid["signal_value"].corr(
            sub_valid["realized_60d_return"], method="spearman"
        ))
        spec_ic = SPEC_LOCKED_IC.get(strat, 0.0)
        per_strategy[strat] = {
            "status":           "OK",
            "n_pairs":          len(sub_valid),
            "spearman_ic":      round(ic, 4),
            "spec_locked_ic":   spec_ic,
            "ic_delta":         round(ic - spec_ic, 4),
            "decay_flag":       ic < spec_ic * 0.5,
            "interpretation":   ("DECAY: IC < 50% of spec" if ic < spec_ic * 0.5
                                 else "OK: IC within reasonable range of spec"),
        }

    any_ok = any(d.get("status") == "OK" for d in per_strategy.values())
    return {
        "status":         "OK" if any_ok else "INSUFFICIENT_DATA",
        "as_of":          as_of.isoformat(),
        "horizon_days":   horizon,
        "per_strategy":   per_strategy,
        "math_anchor":    "Spearman rank IC; McLean-Pontiff 2016 post-publication decay framework",
    }
