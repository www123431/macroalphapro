"""engine/validation/factor_attribution.py — residual-alpha regression.

Regresses each strategy's EXCESS return on the FF5 + UMD factor set.
The regression intercept (annualized) is the residual alpha — the part
of the strategy's return that is NOT explained by cheaply-buyable factor
beta. If the residual alpha is statistically zero, the strategy is not
alpha; it is a (possibly leveraged) repackaging of factors any investor
can buy through ETFs.

Standard errors use Newey-West (HAC) to account for the autocorrelation
that weekly overlapping-signal strategies exhibit — naive OLS t-stats
would be overstated.

Output per strategy:
  alpha_annual    — intercept × 52, in decimal (e.g. 0.03 = 3%/yr)
  alpha_tstat     — Newey-West t-stat on the intercept (|t|>~2 ⇒ sig.)
  r_squared       — fraction of variance explained by factors
  betas           — factor loadings (Mkt-RF, SMB, HML, RMW, CMA, UMD)
  residual_sharpe_annual — alpha / residual vol, annualized: the Sharpe
                  of the part that is NOT factor beta
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]


@dataclass(frozen=True)
class AttributionResult:
    strategy:               str
    n_obs:                  int
    alpha_annual:           float          # intercept × 52 (decimal)
    alpha_tstat:            float          # Newey-West HAC t-stat
    alpha_pvalue:           float
    r_squared:              float
    betas:                  dict           # factor → loading
    residual_sharpe_annual: float
    raw_sharpe_annual:      float          # for comparison
    verdict:                str


def _newey_west_lags(n_obs: int) -> int:
    """Rule-of-thumb HAC lag length: floor(4 (T/100)^(2/9)) (Newey-West
    1994 automatic bandwidth, the common default)."""
    return int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))


def attribute_strategy(
    strat_excess:   pd.Series,
    factors:        pd.DataFrame,
    periods_per_year: int = 52,
) -> AttributionResult:
    """Run the factor regression for ONE strategy's excess returns.

    Args:
      strat_excess:    weekly EXCESS return series (strategy − RF), aligned
                       to ``factors`` index.
      factors:         weekly factor frame containing _FACTOR_COLS.
      periods_per_year: 52.

    Returns AttributionResult.
    """
    import statsmodels.api as sm

    df = pd.concat([strat_excess.rename("y"), factors[_FACTOR_COLS]], axis=1).dropna()
    T = len(df)
    name = strat_excess.name or "strategy"
    if T < 30:
        return AttributionResult(
            strategy=name, n_obs=T, alpha_annual=float("nan"),
            alpha_tstat=float("nan"), alpha_pvalue=float("nan"),
            r_squared=float("nan"), betas={}, residual_sharpe_annual=float("nan"),
            raw_sharpe_annual=float("nan"),
            verdict="UNDEFINED (insufficient overlap)",
        )

    X = sm.add_constant(df[_FACTOR_COLS].values)
    y = df["y"].values
    lags = _newey_west_lags(T)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    intercept_pp = float(model.params[0])
    alpha_annual = intercept_pp * periods_per_year
    alpha_tstat  = float(model.tvalues[0])
    alpha_pval   = float(model.pvalues[0])
    r2           = float(model.rsquared)
    betas        = {c: float(b) for c, b in zip(_FACTOR_COLS, model.params[1:])}

    # Residual Sharpe: alpha relative to the volatility of the regression
    # residual (the idiosyncratic part), annualized.
    resid = model.resid
    resid_vol_pp = float(np.std(resid, ddof=len(_FACTOR_COLS) + 1))
    residual_sharpe = (
        (intercept_pp / resid_vol_pp) * np.sqrt(periods_per_year)
        if resid_vol_pp > 0 else float("nan")
    )

    raw_sharpe = (
        (df["y"].mean() / df["y"].std(ddof=1)) * np.sqrt(periods_per_year)
        if df["y"].std(ddof=1) > 0 else float("nan")
    )

    # Verdict: is there residual alpha after factors?
    if np.isnan(alpha_tstat):
        verdict = "UNDEFINED"
    elif abs(alpha_tstat) >= 2.0 and alpha_annual > 0:
        verdict = "RESIDUAL ALPHA — survives factor decomposition (|t|>=2)"
    elif abs(alpha_tstat) >= 1.65 and alpha_annual > 0:
        verdict = "WEAK residual alpha (|t| 1.65-2.0)"
    else:
        verdict = "NO residual alpha — return is explained by factor beta"

    return AttributionResult(
        strategy=name, n_obs=T, alpha_annual=alpha_annual,
        alpha_tstat=alpha_tstat, alpha_pvalue=alpha_pval, r_squared=r2,
        betas=betas, residual_sharpe_annual=residual_sharpe,
        raw_sharpe_annual=raw_sharpe, verdict=verdict,
    )


def attribute_book(
    strat_returns:  pd.DataFrame,
    factors:        pd.DataFrame,
    rf_col:         str = "RF",
    periods_per_year: int = 52,
) -> dict[str, AttributionResult]:
    """Run factor attribution for every column in ``strat_returns``.

    Subtracts the risk-free rate (factors[rf_col]) from each strategy to
    form excess returns before regressing. Returns {strategy: result}.
    """
    from engine.validation.factor_data import align_returns_to_factors

    aligned_strat, aligned_factors = align_returns_to_factors(
        strat_returns, factors,
    )
    rf = aligned_factors[rf_col] if rf_col in aligned_factors else 0.0

    out: dict[str, AttributionResult] = {}
    for col in aligned_strat.columns:
        excess = (aligned_strat[col] - rf).rename(col)
        out[col] = attribute_strategy(excess, aligned_factors, periods_per_year)
    return out
