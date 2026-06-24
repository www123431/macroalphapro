"""engine.factor_regression.regression — OLS with Newey-West SE.

For each sleeve / book return series, regress excess return on FF5 +
MOM factors:

    r_sleeve - RF  =  α + β_MKT × MKT_RF + β_SMB × SMB + β_HML × HML
                       + β_RMW × RMW + β_CMA × CMA + β_UMD × MOM + ε

Output:
  - α (annualized) + Newey-West HAC t-stat
  - per-factor β + t-stat
  - R² + adjusted R²
  - n_obs

Newey-West correction is mandatory for any quant alpha claim
(see Bailey-LdP "Backtest Overfitting" 2014 + Lo 2002 on hedge fund
return autocorrelation).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


FACTORS_FF5_MOM = ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")


@dataclass(frozen=True)
class FactorRegression:
    sleeve_label:       str
    n_obs:              int
    window_start:       str
    window_end:         str
    annualization:      int       # 12 monthly, 52 weekly, 252 daily

    # Alpha
    alpha_per_period:   float
    alpha_annualized:   float
    alpha_tstat_NW:     float
    alpha_pvalue_NW:    float

    # Betas (one per factor, in FACTORS_FF5_MOM order)
    betas:              dict[str, float]
    beta_tstats_NW:     dict[str, float]
    beta_pvalues_NW:    dict[str, float]

    # Fit
    r_squared:          float
    r_squared_adj:      float
    residual_vol_ann:   float

    # Verdict
    alpha_t_clears_HLZ: bool   # |t_NW| >= 3.0 (Harvey-Liu-Zhu 2016)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _newey_west_se(X: np.ndarray, residuals: np.ndarray,
                   lags: int) -> np.ndarray:
    """Compute Newey-West HAC standard errors for OLS coefficients.

    X: design matrix (n, k) including intercept column.
    residuals: OLS residuals (n,).
    lags: # of autocorrelation lags to include (Bartlett kernel).

    Returns: standard errors (k,) for each coefficient.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)

    # S = sum_i e_i^2 x_i x_i' + sum_l (1 - l/(L+1)) sum_{i=l+1..n}(...)
    S = np.zeros((k, k))
    for i in range(n):
        ei = residuals[i]
        xi = X[i, :].reshape(-1, 1)
        S += (ei ** 2) * (xi @ xi.T)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        for i in range(L, n):
            ei = residuals[i]
            ej = residuals[i - L]
            xi = X[i, :].reshape(-1, 1)
            xj = X[i - L, :].reshape(-1, 1)
            S += w * ei * ej * (xi @ xj.T + xj @ xi.T)
    var = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(var), 0.0))
    return se


def _two_sided_pvalue(t_stat: float, n: int, k: int) -> float:
    """Two-sided p-value for a t-statistic with (n-k) degrees of freedom.
    Approximated with the normal CDF for large n; switches to t for
    small samples."""
    df = max(1, n - k)
    if df >= 30:
        from math import erf, sqrt
        z = abs(t_stat)
        # 2 * (1 - Phi(z))
        return 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    # Small-sample: use scipy if available, else fall back to normal
    try:
        from scipy import stats
        return float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
    except Exception:
        from math import erf, sqrt
        z = abs(t_stat)
        return 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))


def run_ff5_mom_regression(
    sleeve_returns:  pd.Series,
    factor_returns:  pd.DataFrame,
    *,
    sleeve_label:    str = "sleeve",
    annualization:   int = 52,
    nw_lags:         int = 6,
) -> FactorRegression:
    """Run FF5 + MOM regression on a sleeve return series.

    Args:
        sleeve_returns: pd.Series of period returns (decimal), indexed by date.
        factor_returns: pd.DataFrame with FACTORS_FF5_MOM + 'RF' columns.
                        Same frequency as sleeve_returns.
        sleeve_label: human-readable name for the output.
        annualization: # periods per year (52 weekly, 12 monthly, 252 daily).
        nw_lags: Newey-West autocorrelation lags. 6 is reasonable for
                 weekly data; use 12 for monthly low-freq.

    Returns:
        FactorRegression with α, βs, t-stats (NW), R², verdict.
    """
    # Align — inner join on date
    df = pd.DataFrame({"r": sleeve_returns}).join(factor_returns, how="inner")
    if "RF" not in df.columns:
        raise ValueError("factor_returns must include 'RF' column")
    missing_factors = [f for f in FACTORS_FF5_MOM if f not in df.columns]
    if missing_factors:
        raise ValueError(f"factor_returns missing: {missing_factors}")

    df = df.dropna(subset=["r", "RF"] + list(FACTORS_FF5_MOM))
    n = len(df)
    if n < 30:
        raise ValueError(f"insufficient data ({n} rows) for regression")

    # Excess return — force plain float64 (defends against pandas
    # masked Float64 dtype from parquet roundtrip)
    y = np.asarray(df["r"] - df["RF"], dtype=np.float64)
    # Design matrix [1, MKT_RF, SMB, HML, RMW, CMA, MOM]
    X = np.column_stack([
        np.ones(n, dtype=np.float64),
        *[np.asarray(df[f], dtype=np.float64) for f in FACTORS_FF5_MOM],
    ])

    # OLS
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    fitted = X @ beta_hat
    resid = y - fitted

    # SE (Newey-West HAC)
    se = _newey_west_se(X, resid, lags=nw_lags)
    t_stats = beta_hat / np.where(se > 0, se, np.nan)

    # Fit metrics
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = X.shape[1]
    r_sq_adj = 1.0 - (1.0 - r_sq) * (n - 1) / max(1, n - k)
    residual_vol = float(np.std(resid, ddof=1)) * math.sqrt(annualization)

    alpha_per_period = float(beta_hat[0])
    alpha_ann = alpha_per_period * annualization
    alpha_t = float(t_stats[0])
    alpha_p = _two_sided_pvalue(alpha_t, n, k)

    betas       = {f: float(beta_hat[i + 1])   for i, f in enumerate(FACTORS_FF5_MOM)}
    beta_ts     = {f: float(t_stats[i + 1])    for i, f in enumerate(FACTORS_FF5_MOM)}
    beta_ps     = {f: _two_sided_pvalue(t_stats[i + 1], n, k)
                   for i, f in enumerate(FACTORS_FF5_MOM)}

    return FactorRegression(
        sleeve_label=sleeve_label,
        n_obs=n,
        window_start=str(df.index.min().date()) if hasattr(df.index.min(), "date") else str(df.index.min()),
        window_end=str(df.index.max().date()) if hasattr(df.index.max(), "date") else str(df.index.max()),
        annualization=annualization,
        alpha_per_period=alpha_per_period,
        alpha_annualized=alpha_ann,
        alpha_tstat_NW=alpha_t,
        alpha_pvalue_NW=alpha_p,
        betas=betas,
        beta_tstats_NW=beta_ts,
        beta_pvalues_NW=beta_ps,
        r_squared=float(r_sq),
        r_squared_adj=float(r_sq_adj),
        residual_vol_ann=residual_vol,
        alpha_t_clears_HLZ=bool(abs(alpha_t) >= 3.0),
    )
