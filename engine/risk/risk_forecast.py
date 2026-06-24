"""engine/risk/risk_forecast.py — BARRA Phase 4.

Forward-looking portfolio risk forecasting: given current book exposures
+ shrunken factor covariance + EWMA-forecasted specific (idio) risk,
estimate next-period book variance and confidence interval.

This is the institutional next step after Phase 3 (which gives risk
DECOMPOSITION via factor_budget). Phase 4 is risk FORECASTING — what
Aladdin / Axioma BVR / MSCI BarraOne core capability is.

THREE COMPONENTS (each independently testable):

1. ledoit_wolf_shrinkage(factor_returns)
   Sample factor covariance with Ledoit-Wolf 2003 shrinkage toward a
   diagonal target. Mitigates the well-known small-sample bias in
   sample cov estimation (Σ_sample is rank-deficient for n < k factors).

2. ewma_specific_risk(residuals, lambda_=0.97)
   Exponentially-weighted moving-average of squared residuals. RiskMetrics-
   style decay parameter (0.94 for daily, 0.97 for monthly). Forecasts
   next-period idiosyncratic variance.

3. portfolio_risk_forecast(sleeve_returns, weights, factor_returns)
   Combines sleeve betas + shrunk Σ_F + EWMA σ²_idio_i to produce:
     - Forecast book annualized vol
     - Decomposition (factor vs idio share)
     - Bootstrap confidence interval

OUTPUT — RiskForecastReport dataclass:
  forecast_vol_annualized        point estimate
  forecast_ci_95                 (low, high) 95% CI via bootstrap
  factor_vol_forecast / idio_vol_forecast
  per_sleeve_idio_forecast       {sleeve: ewma_idio}
  shrinkage_intensity            Ledoit-Wolf delta (0=no shrink, 1=full)
  n_months_used                  sample size

Use case examples:
- "Given current 5-sleeve config, what's the 1-quarter-ahead vol forecast?"
- "Does the deployed book likely stay within 10% vol-target next year?"
- "Is the regime-conditional config statistically lower-risk than static?"
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.risk.barra_lite import (
    build_factor_returns,
    regress_sleeve_on_factors,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RiskForecastReport:
    forecast_vol_annualized:   float
    forecast_ci_95:            tuple[float, float]     # (low, high)
    factor_vol_forecast:       float
    idio_vol_forecast:         float
    pct_factor:                float
    pct_idio:                  float
    book_exposures:            dict[str, float]
    per_sleeve_idio_forecast:  dict[str, float]
    shrinkage_intensity:       float
    n_months_used:             int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Ledoit-Wolf shrinkage ────────────────────────────────────────────────

def ledoit_wolf_shrinkage(returns: pd.DataFrame,
                              target: str = "diagonal",
                              ) -> tuple[np.ndarray, float]:
    """Shrunken covariance per Ledoit-Wolf 2003.

    Σ_shrunk = δ × F + (1 - δ) × S
    where S is sample cov, F is the target (diagonal of S by default),
    and δ ∈ [0, 1] is the optimal shrinkage intensity estimated from data.

    Args:
      returns: T × k panel of factor returns
      target: "diagonal" (default) or "constant_correlation"

    Returns: (Σ_shrunk, δ)
    """
    R = returns.dropna().values
    n, k = R.shape
    if n <= 1 or k == 0:
        return np.zeros((k, k)), 0.0

    Rm = R - R.mean(axis=0)
    S = (Rm.T @ Rm) / n     # sample covariance (biased estimator)

    if target == "diagonal":
        F = np.diag(np.diag(S))
    elif target == "constant_correlation":
        # average pairwise correlation
        d = np.sqrt(np.diag(S))
        outer = np.outer(d, d)
        outer = np.where(outer > 0, outer, 1e-12)
        corr = S / outer
        offdiag_corr = (corr.sum() - np.trace(corr)) / max(1, k * (k - 1))
        F = offdiag_corr * outer
        np.fill_diagonal(F, np.diag(S))
    else:
        raise ValueError(f"unknown shrinkage target {target!r}")

    # Optimal shrinkage intensity (Ledoit-Wolf 2003 Eq 14 simplified for
    # diagonal target):
    #   δ* = sum_i ≠ j (var(S_ij) - cov(S_ij, F_ij)) / sum_i ≠ j (S_ij - F_ij)²
    # We use the practical estimator from Schäfer-Strimmer 2005:
    pi_hat = 0.0
    for t in range(n):
        rt = Rm[t]
        Y_t = np.outer(rt, rt) - S
        pi_hat += (Y_t ** 2).sum()
    pi_hat /= n

    rho_hat = pi_hat   # diagonal-target simplification
    gamma_hat = ((S - F) ** 2).sum()
    if gamma_hat <= 0:
        delta = 0.0
    else:
        delta = max(0.0, min(1.0, (pi_hat - rho_hat) / (n * gamma_hat) + 1.0))

    Sigma_shrunk = delta * F + (1.0 - delta) * S
    return Sigma_shrunk, float(delta)


# ── EWMA specific risk ───────────────────────────────────────────────────

def ewma_specific_risk(residuals: pd.Series, lambda_: float = 0.97) -> float:
    """EWMA forecast of next-period residual variance.

    σ²_t = λ × σ²_t-1 + (1 - λ) × ε²_t
    Initial: σ²_0 = ε²_0.

    Returns the FORECAST for period T+1 given residuals 1..T.
    """
    r = residuals.dropna().values
    if len(r) == 0:
        return 0.0
    sigma2 = r[0] ** 2
    for t in range(1, len(r)):
        sigma2 = lambda_ * sigma2 + (1.0 - lambda_) * (r[t] ** 2)
    return float(sigma2)


# ── Portfolio risk forecast (main entry) ─────────────────────────────────

def portfolio_risk_forecast(
    sleeve_returns:    dict[str, pd.Series],
    sleeve_weights:    dict[str, float],
    factor_returns:    pd.DataFrame | None = None,
    phase:             int = 3,
    ewma_lambda:       float = 0.97,
    shrinkage_target:  str = "diagonal",
    n_bootstrap:       int = 1000,
    seed:              int = 11,
) -> RiskForecastReport:
    """Forward-looking risk forecast for a multi-sleeve book.

    Method:
      1. Regress each sleeve on factors → betas + residuals.
      2. Estimate shrunk factor cov Σ_F via Ledoit-Wolf.
      3. Per-sleeve EWMA forecast of idio variance.
      4. Book factor variance = B' Σ_F B, B = Σ_i w_i × β_i
      5. Book idio variance = Σ_i w_i² × σ²_idio,i
      6. Forecast total var, annualize.
      7. Bootstrap CI: resample sleeve returns, recompute forecast,
         95% percentile range.
    """
    rng = np.random.default_rng(seed)

    if factor_returns is None:
        factor_returns = build_factor_returns(phase=phase)

    factor_cols = list(factor_returns.columns)
    sleeve_betas: dict[str, dict[str, float]] = {}
    sleeve_resid: dict[str, pd.Series] = {}
    sleeve_n: dict[str, int] = {}

    for name, ret in sleeve_returns.items():
        try:
            rep = regress_sleeve_on_factors(ret, factor_returns,
                                                  sleeve_name=name)
        except ValueError as exc:
            logger.warning("regression failed for %s: %s", name, exc)
            continue
        sleeve_betas[name] = rep.betas
        sleeve_n[name] = rep.n_months

        # Reconstruct residuals: y - α - β × F
        s = ret.copy()
        s.index = pd.to_datetime(s.index)
        s = s.resample("ME").last() if not s.index.equals(
            s.index.to_period("M").to_timestamp("M")) else s
        J = pd.concat([s.rename("y"), factor_returns], axis=1).dropna()
        beta_vec = np.array([rep.betas.get(c, 0.0) for c in factor_cols])
        fitted = (rep.alpha_monthly
                  + (J[factor_cols].values @ beta_vec))
        resid = J["y"].values - fitted
        sleeve_resid[name] = pd.Series(resid, index=J.index)

    if not sleeve_betas:
        raise ValueError("no sleeve regressions succeeded")

    # Step 2: shrunk factor covariance
    Sigma_F, delta = ledoit_wolf_shrinkage(
        factor_returns, target=shrinkage_target,
    )

    # Step 3: per-sleeve EWMA idio variance forecast
    sleeve_idio_var: dict[str, float] = {}
    for name, resid in sleeve_resid.items():
        sleeve_idio_var[name] = ewma_specific_risk(resid, lambda_=ewma_lambda)

    # Step 4-5: book variance components
    B = np.zeros(len(factor_cols))
    for i, c in enumerate(factor_cols):
        for name, betas in sleeve_betas.items():
            B[i] += sleeve_weights.get(name, 0.0) * betas.get(c, 0.0)
    book_factor_var = float(B @ Sigma_F @ B)
    book_idio_var = float(sum(
        sleeve_weights.get(name, 0.0) ** 2 * v
        for name, v in sleeve_idio_var.items()
    ))
    total_var = book_factor_var + book_idio_var
    if total_var <= 0:
        raise ValueError("forecast variance non-positive")

    sqrt12 = float(np.sqrt(12.0))
    forecast_vol_ann = float(np.sqrt(total_var)) * sqrt12
    factor_vol = float(np.sqrt(book_factor_var)) * sqrt12
    idio_vol = float(np.sqrt(book_idio_var)) * sqrt12

    # Step 6: bootstrap CI
    ci = _bootstrap_vol_ci(
        sleeve_returns, sleeve_weights, factor_returns,
        n_bootstrap=n_bootstrap, ewma_lambda=ewma_lambda,
        shrinkage_target=shrinkage_target, rng=rng,
    )

    n_months_med = int(np.median(list(sleeve_n.values())))

    return RiskForecastReport(
        forecast_vol_annualized=forecast_vol_ann,
        forecast_ci_95=ci,
        factor_vol_forecast=factor_vol,
        idio_vol_forecast=idio_vol,
        pct_factor=book_factor_var / total_var,
        pct_idio=book_idio_var / total_var,
        book_exposures={c: float(B[i]) for i, c in enumerate(factor_cols)},
        per_sleeve_idio_forecast={k: float(np.sqrt(v) * sqrt12)
                                      for k, v in sleeve_idio_var.items()},
        shrinkage_intensity=delta,
        n_months_used=n_months_med,
    )


# ── Bootstrap CI helper ─────────────────────────────────────────────────

def _bootstrap_vol_ci(sleeve_returns, sleeve_weights, factor_returns,
                          n_bootstrap, ewma_lambda, shrinkage_target,
                          rng) -> tuple[float, float]:
    """Bootstrap 95% CI on the forecast vol by resampling month-blocks."""
    common_idx = None
    for ret in sleeve_returns.values():
        s = ret.copy()
        s.index = pd.to_datetime(s.index)
        s = s.resample("ME").last()
        if common_idx is None:
            common_idx = s.index
        else:
            common_idx = common_idx.intersection(s.index)
    common_idx = common_idx.intersection(factor_returns.index)
    n = len(common_idx)
    if n < 24:
        return (float("nan"), float("nan"))

    forecasts = []
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(common_idx, size=n, replace=True)
        try:
            sub_sleeves = {
                name: ret.copy().reindex(sample_idx).dropna()
                for name, ret in sleeve_returns.items()
            }
            sub_factors = factor_returns.reindex(sample_idx).dropna()
            r = portfolio_risk_forecast(
                sub_sleeves, sleeve_weights, factor_returns=sub_factors,
                ewma_lambda=ewma_lambda,
                shrinkage_target=shrinkage_target,
                n_bootstrap=0,    # avoid recursive bootstrap
            )
            forecasts.append(r.forecast_vol_annualized)
        except Exception:
            continue
    if not forecasts:
        return (float("nan"), float("nan"))
    arr = np.array(forecasts)
    return (float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))
