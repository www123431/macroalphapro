"""engine.factor_regression.tc_ablation — TC sensitivity sweep.

Apply varying transaction-cost (TC) drag to a sleeve / book return
series and report Sharpe at each level. The point is to surface
"cost-fragile" alpha — strategies whose Sharpe collapses past, say,
30 bps round-trip become un-investable in retail or mid-size pools.

We do NOT have per-period turnover in the v1 replay parquet. We model
it via per-sleeve **assumed annual one-way turnover** anchored to the
sleeve's rebalance cadence. This is honest practice: any TC ablation
without a real turnover series MUST disclose the turnover model.

Per-sleeve turnover assumptions (one-way, annualized):

  - K1_BAB:    600 %/yr   (monthly rebalance, ~50%/mo)
  - D_PEAD:    400 %/yr   (quarterly event-driven; ~100% per quarter
                          but rolling, so combined ~ 400% annually)
  - PATH_N:    400 %/yr   (monthly carry overlay)
  - CTA_PQTIX: 800 %/yr   (monthly TSMOM, high turnover trend-following)

These are NOT measured — they are anchored to public hedge-fund
literature (e.g. AQR Capital Asset Pricing carry/momentum studies)
and the sleeve's design cadence. A reviewer should treat the
ablation Sharpes here as **upper bounds on the cost sensitivity**;
real per-trade turnover may be higher.

Drag model:
    annual_drag = TC_bps_one_way × annual_turnover_fraction
    per_period_drag = annual_drag / annualization
    r_net = r_gross − per_period_drag
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


SLEEVE_ANNUAL_TURNOVER = {
    "K1_BAB":    6.0,   # 600 %/yr
    "D_PEAD":    4.0,   # 400 %/yr
    "PATH_N":    4.0,   # 400 %/yr
    "CTA_PQTIX": 8.0,   # 800 %/yr
}


@dataclass(frozen=True)
class TCAblationResult:
    tc_bps_one_way:       int
    sleeve_label:         str
    n_obs:                int
    annual_drag_pct:      float   # e.g. 0.33 = 33 bps/yr drag for the book at this TC
    mean_annualized:      float
    vol_annualized:       float
    sharpe_annualized:    float
    cum_return:           float
    max_drawdown:         float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def _stats(returns: pd.Series, annualization: int) -> tuple[float, float, float, float, float]:
    arr = returns.dropna().astype(float).values
    n = len(arr)
    if n < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    mean_a = float(np.mean(arr)) * annualization
    vol_a  = float(np.std(arr, ddof=1)) * math.sqrt(annualization)
    sharpe = mean_a / vol_a if vol_a > 0 else 0.0
    cum    = float(np.prod(1.0 + arr) - 1.0)
    dd     = _max_drawdown(returns)
    return (mean_a, vol_a, sharpe, cum, dd)


def book_effective_turnover(per_sleeve_returns: pd.DataFrame,
                            turnover_map: dict[str, float] = None) -> float:
    """Equal-weight combined book turnover = mean of sleeve turnovers
    (assumes equal weights, which is the v1 replay default).

    If you have sleeve weights that differ from equal, weight here.
    """
    if turnover_map is None:
        turnover_map = SLEEVE_ANNUAL_TURNOVER
    sleeves = list(per_sleeve_returns.columns)
    unknown = [s for s in sleeves if s not in turnover_map]
    if unknown:
        raise ValueError(f"sleeve(s) without turnover assumption: {unknown}; "
                         f"add to turnover_map")
    return float(np.mean([turnover_map[s] for s in sleeves]))


def run_tc_ablation(
    combined_returns:    pd.Series,
    *,
    per_sleeve_returns:  pd.DataFrame = None,
    turnover_annual:     float = None,
    tc_bps_grid:         list[int] = (5, 10, 30, 60, 100),
    annualization:       int = 52,
    sleeve_label:        str = "combined_book",
) -> list[TCAblationResult]:
    """Ablation sweep.

    Either pass per_sleeve_returns (will compute effective turnover from
    SLEEVE_ANNUAL_TURNOVER map) OR pass an explicit turnover_annual.
    """
    if turnover_annual is None and per_sleeve_returns is None:
        raise ValueError("must pass either turnover_annual or per_sleeve_returns")
    if turnover_annual is None:
        turnover_annual = book_effective_turnover(per_sleeve_returns)

    results = []
    for tc_bps in tc_bps_grid:
        annual_drag = (tc_bps / 1e4) * turnover_annual
        per_period_drag = annual_drag / annualization
        r_net = combined_returns - per_period_drag

        mean_a, vol_a, sharpe, cum, dd = _stats(r_net, annualization)
        results.append(TCAblationResult(
            tc_bps_one_way=tc_bps,
            sleeve_label=sleeve_label,
            n_obs=int(combined_returns.dropna().shape[0]),
            annual_drag_pct=float(annual_drag),
            mean_annualized=mean_a,
            vol_annualized=vol_a,
            sharpe_annualized=sharpe,
            cum_return=cum,
            max_drawdown=dd,
        ))
    return results
