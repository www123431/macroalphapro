"""engine.factor_regression.regime — per-regime decomposition of sleeve returns.

For each pre-registered regime window, compute Sharpe / vol / max DD /
mean / hit-rate on the sleeve's returns within that window. The point
is to surface crisis-period behavior that a single full-window Sharpe
hides.

Regime windows here are anchored to the v1 replay data range
(2014-09 → 2023-12). The full GFC 2008 is OUT of sample for this
replay (data starts 2014) — honest limitation, documented.

Standard practice at AQR / Two Sigma / Citadel: every strategy carries
a "crisis behavior" table. This module produces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


# Pre-registered regime windows (calendar-anchored, public market events).
# Stored as (label, start_date, end_date, comment). Both endpoints inclusive.
REGIMES_V1 = [
    ("normal_2014_2017",
     "2014-09-05", "2017-12-29",
     "Post-crisis low-vol bull. Reference baseline."),
    ("2018_volmageddon_Q1",
     "2018-02-01", "2018-04-30",
     "Vol-mageddon (XIV blowup Feb-2018); VIX > 30 briefly."),
    ("2018_q4_drawdown",
     "2018-10-01", "2018-12-31",
     "Q4-2018 risk-off; S&P drew -14% in 3 months."),
    ("2019_recovery",
     "2019-01-01", "2019-12-31",
     "Strong recovery year; equity beta-favorable."),
    ("2020_covid_crash",
     "2020-02-19", "2020-03-31",
     "COVID-19 fastest 30%+ drawdown in S&P history."),
    ("2020_post_covid_rally",
     "2020-04-01", "2020-12-31",
     "Liquidity-fueled rally + Fed intervention."),
    ("2021_bull",
     "2021-01-01", "2021-12-31",
     "Continued bull, growth-stock peak Q4."),
    ("2022_full_year",
     "2022-01-01", "2022-12-30",
     "Inflation shock + Fed hiking cycle; classic 60/40 crisis "
     "(both stocks AND bonds down)."),
    ("2023_recovery",
     "2023-01-01", "2023-12-22",
     "Tech recovery; rate-cycle peaking expectations."),
]


@dataclass(frozen=True)
class RegimeStats:
    label:               str
    start:               str
    end:                 str
    comment:             str
    n_obs:               int
    mean_per_period:     float
    mean_annualized:     float
    vol_per_period:      float
    vol_annualized:      float
    sharpe_annualized:   float
    max_drawdown:        float
    hit_rate:            float   # fraction of positive periods
    cum_return:          float   # 1+r1)(1+r2)... - 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_max_drawdown(returns: pd.Series) -> float:
    """Max drawdown from cumulative-product equity curve. Returns
    a negative number (e.g. -0.15 means -15% peak-to-trough)."""
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def regime_stats(returns: pd.Series, label: str, start: str, end: str,
                 comment: str = "",
                 annualization: int = 52) -> RegimeStats:
    """Stats for ``returns`` sliced to [start, end] inclusive."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sliced = returns.loc[(returns.index >= start_ts) & (returns.index <= end_ts)]
    sliced = sliced.dropna()
    n = len(sliced)

    if n == 0:
        return RegimeStats(
            label=label, start=start, end=end, comment=comment,
            n_obs=0, mean_per_period=0.0, mean_annualized=0.0,
            vol_per_period=0.0, vol_annualized=0.0,
            sharpe_annualized=0.0, max_drawdown=0.0, hit_rate=0.0,
            cum_return=0.0,
        )

    arr = sliced.astype(float).values
    mean_p = float(np.mean(arr))
    vol_p  = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    mean_a = mean_p * annualization
    vol_a  = vol_p * math.sqrt(annualization)
    sharpe = mean_a / vol_a if vol_a > 0 else 0.0
    dd     = _safe_max_drawdown(sliced)
    hit    = float((arr > 0).sum()) / n
    cum    = float(np.prod(1.0 + arr) - 1.0)

    return RegimeStats(
        label=label, start=start, end=end, comment=comment,
        n_obs=n, mean_per_period=mean_p, mean_annualized=mean_a,
        vol_per_period=vol_p, vol_annualized=vol_a,
        sharpe_annualized=sharpe, max_drawdown=dd, hit_rate=hit,
        cum_return=cum,
    )


def run_regime_decomposition(
    returns:        pd.Series,
    regimes:        list[tuple[str, str, str, str]] = None,
    annualization:  int = 52,
) -> list[RegimeStats]:
    """Run the regime grid + return per-regime stats list.

    `regimes`: optional override. Each entry is (label, start, end, comment).
    Default = REGIMES_V1.
    """
    if regimes is None:
        regimes = REGIMES_V1
    return [
        regime_stats(returns, label=lbl, start=s, end=e, comment=c,
                     annualization=annualization)
        for (lbl, s, e, c) in regimes
    ]
