"""engine/validation/rolling_sharpe.py — is the edge alive, or a relic?

A high full-sample Sharpe can hide the fact that the edge worked in
2014-2018 and has been dead since. McLean-Pontiff (2016) documented
~58% post-publication decay in anomalies; BAB (public since 2014) and
PEAD (public since the 1980s) are prime decay candidates. This module
computes the rolling-window Sharpe trajectory + a first-half / second-
half / recent-window split so the PM can see whether each sleeve is
still earning.

Deterministic, read-only. Operates on the weekly per-strategy returns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecayResult:
    strategy:        str
    full_sharpe:     float
    first_half_sharpe: float
    second_half_sharpe: float
    recent_sharpe:   float          # last `recent_weeks`
    recent_weeks:    int
    decay_ratio:     float          # second_half / first_half (1.0 = no decay)
    verdict:         str


def _ann_sharpe(r: np.ndarray, ppy: int = 52) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(ppy))


def rolling_sharpe(
    returns:  pd.Series,
    window:   int = 104,       # 2 years of weeks
    ppy:      int = 52,
) -> pd.Series:
    """Rolling annualized Sharpe over a trailing ``window`` of weeks."""
    def _f(x):
        x = np.asarray(x, dtype=float)
        sd = x.std(ddof=1)
        return (x.mean() / sd * math.sqrt(ppy)) if sd > 0 else np.nan
    return returns.rolling(window).apply(_f, raw=True)


def decay_split(
    returns:      pd.Series,
    recent_weeks: int = 156,    # last 3 years
    ppy:          int = 52,
) -> DecayResult:
    """Full / first-half / second-half / recent Sharpe + decay verdict."""
    r = returns.dropna()
    name = returns.name or "strategy"
    n = len(r)
    if n < 60:
        return DecayResult(name, float("nan"), float("nan"), float("nan"),
                           float("nan"), recent_weeks, float("nan"),
                           "UNDEFINED (too short)")
    vals = r.values
    mid = n // 2
    full   = _ann_sharpe(vals, ppy)
    first  = _ann_sharpe(vals[:mid], ppy)
    second = _ann_sharpe(vals[mid:], ppy)
    recent = _ann_sharpe(vals[-recent_weeks:], ppy)

    decay = (second / first) if (first and not math.isnan(first) and first != 0) else float("nan")

    # Verdict — lead with the RECENT window (most decision-relevant),
    # use the half-split decay ratio as secondary context. A strategy
    # whose recent window recovered above its 2nd-half is ALIVE even if
    # the 1st-half was exceptional (the half-split would otherwise mis-
    # flag it as DECAYING).
    if math.isnan(recent):
        verdict = "UNDEFINED"
    elif recent <= 0:
        verdict = "DEAD — recent-window Sharpe <= 0"
    elif recent < 0.3:
        verdict = "WEAK recently — recent Sharpe < 0.3"
    elif not math.isnan(decay) and decay < 0.6:
        verdict = "ALIVE but FRONT-LOADED — early period far stronger"
    else:
        verdict = "ALIVE — recent edge intact"

    return DecayResult(
        strategy=name, full_sharpe=full, first_half_sharpe=first,
        second_half_sharpe=second, recent_sharpe=recent,
        recent_weeks=recent_weeks, decay_ratio=decay, verdict=verdict,
    )


def decay_book(
    strat_returns: pd.DataFrame,
    recent_weeks:  int = 156,
    ppy:           int = 52,
) -> dict[str, DecayResult]:
    """Run decay_split for every strategy column."""
    return {c: decay_split(strat_returns[c], recent_weeks, ppy)
            for c in strat_returns.columns}
