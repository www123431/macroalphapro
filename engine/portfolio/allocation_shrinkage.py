"""
engine/portfolio/allocation_shrinkage.py — Bayes-Stein SAA weight optimizer.

Tier-1 audit class A #2 (2026-05-14): improves the 36 / 27 / 27 / 10
fixed SAA by applying academic-grade shrinkage on Sharpe and covariance
before solving for optimal weights.

Why shrinkage matters
---------------------
Sample-Sharpe-based mean-variance optimization is notoriously unstable
(Michaud 1989 "Markowitz Optimization Enigma"). Small estimation error
in μ or Σ produces large weight movements — "error maximization", not
optimization. Two academic fixes applied here:

  Bayes-Stein on Sharpe (Jorion 1986)
    Shrinks per-strategy excess-return estimates toward a common prior
    (here: the cross-sectional weighted mean). Shrinkage intensity is
    a closed-form function of sample size and dispersion. Reduces the
    influence of any single high-Sharpe strategy that may be lucky.

  Ledoit-Wolf shrinkage on covariance (Ledoit-Wolf 2003, 2004)
    Shrinks the sample covariance matrix toward a structured target
    (here: identity scaled by mean variance — single-factor model).
    Closed-form optimal shrinkage intensity. Stabilizes the inverse.

After shrinkage, weights are obtained by solving a constrained
Markowitz program (sum=1, bounded, optional sleeve locks).

Output is intended for review-only — actual SAA change requires
human sign-off via the existing spec amendment workflow. This module
NEVER mutates production state.

References
----------
- Markowitz 1952 "Portfolio Selection"
- James-Stein 1961 (original shrinkage estimator)
- Jorion 1986 "Bayes-Stein Estimation for Portfolio Analysis"
- Ledoit-Wolf 2003 "Improved Estimation of the Covariance Matrix"
- Ledoit-Wolf 2004 "Honey, I Shrunk the Sample Covariance Matrix"
- Michaud 1989 "The Markowitz Optimization Enigma"
- DeMiguel-Garlappi-Uppal 2009 "Optimal vs Naive Diversification"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WEEKLY_RFR_USD = 0.04 / 52.0   # 4% annual risk-free, weekly equivalent

# Current production SAA (per docs/portfolio_deployment_design_2026-05-13.md)
CURRENT_SAA: dict[str, float] = {
    "K1_BAB":    0.36,
    "D_PEAD":    0.27,
    "PATH_N":    0.27,
    "CTA_PQTIX": 0.10,
}

# Sleeve locks (these come from the existing spec doctrine)
#   CTA_PQTIX fixed at 10% per Path O spec id=73 (crisis hedge mandate)
#   etf_l1 (K1) sleeve 36% per deployment_design.md
#   ss_sp500 sleeve 54% (D_PEAD + Path N share this; intra split is the only
#   genuinely free degree of freedom under existing locks)
SLEEVE_LOCKS: dict[str, tuple[float, float]] = {
    "K1_BAB":    (0.36, 0.36),   # fixed
    "CTA_PQTIX": (0.10, 0.10),   # fixed
    # D_PEAD + PATH_N can vary intra-sleeve as long as sum == 0.54
}


# ─────────────────────────────────────────────────────────────────────────────
# Bayes-Stein shrinkage on Sharpe / mean-excess-return
# ─────────────────────────────────────────────────────────────────────────────
def bayes_stein_shrink_mean(
    sample_means: np.ndarray,
    sample_cov:   np.ndarray,
    n_obs:        int,
) -> tuple[np.ndarray, float]:
    """Jorion 1986 Bayes-Stein shrinkage of sample means toward grand mean.

    The shrinkage estimator is:
        μ_BS = (1 - w) * μ_sample + w * μ_grand * 1
    where:
        μ_grand = (1' Σ^-1 μ) / (1' Σ^-1 1)   (precision-weighted mean)
        w = min(1, (N + 2) / ((N + 2) + T * (μ_s - μ_g 1)' Σ^-1 (μ_s - μ_g 1)))
    N = n_strategies, T = n_obs.

    Returns (shrunk_means, shrinkage_intensity).
    """
    N = len(sample_means)
    T = int(n_obs)
    if T <= N or N < 2:
        return sample_means.copy(), 0.0

    # Precision-weighted grand mean
    try:
        cov_inv = np.linalg.pinv(sample_cov)
    except Exception:
        return sample_means.copy(), 0.0
    ones = np.ones(N)
    denom = float(ones @ cov_inv @ ones)
    if denom <= 0:
        return sample_means.copy(), 0.0
    grand_mean = float((ones @ cov_inv @ sample_means) / denom)

    # Shrinkage intensity (Jorion 1986 eq. 11)
    diff = sample_means - grand_mean
    quad = float(diff @ cov_inv @ diff)
    if quad <= 0:
        return np.full(N, grand_mean), 1.0
    w = (N + 2) / ((N + 2) + T * quad)
    w = max(0.0, min(1.0, w))

    shrunk = (1.0 - w) * sample_means + w * grand_mean * ones
    return shrunk, w


# ─────────────────────────────────────────────────────────────────────────────
# Ledoit-Wolf shrinkage on covariance toward single-factor (identity) target
# ─────────────────────────────────────────────────────────────────────────────
def ledoit_wolf_shrink_cov(
    returns: np.ndarray,   # T × N
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 2004 closed-form covariance shrinkage toward identity target.

    Target F = trace(S)/N * I (identity scaled to mean variance).
    Optimal intensity α* derived from sample.

    Returns (shrunk_cov, shrinkage_intensity).
    """
    T, N = returns.shape
    if T < N + 2 or N < 2:
        return np.cov(returns.T, ddof=1), 0.0

    # Sample covariance (MLE — Ledoit-Wolf uses 1/T not 1/(T-1))
    X = returns - returns.mean(axis=0)
    S = (X.T @ X) / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # Variance of S entries (Ledoit-Wolf eq. 12)
    # π_hat = (1/T) Σ_t || x_t x_t' - S ||_F^2 / T
    # Simplified: average squared deviation across (i,j) and t.
    pi_mat = np.zeros((N, N))
    for t in range(T):
        x = X[t:t + 1, :]
        outer = x.T @ x
        pi_mat += (outer - S) ** 2
    pi_mat /= T
    pi_hat = float(pi_mat.sum())

    # Distance between target and sample
    gamma = float(((F - S) ** 2).sum())

    # Optimal intensity
    if gamma <= 0:
        return S, 0.0
    kappa = pi_hat / gamma
    alpha = max(0.0, min(1.0, kappa / T))

    shrunk = alpha * F + (1.0 - alpha) * S
    return shrunk, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Markowitz-style weight solver (with bounds + sleeve locks)
# ─────────────────────────────────────────────────────────────────────────────
def solve_optimal_weights(
    excess_returns: np.ndarray,    # length-N
    covariance:     np.ndarray,    # N × N
    strategies:     list[str],
    risk_aversion:  float = 2.0,
    min_weight:     float = 0.0,   # long-only
    max_weight:     float = 0.60,  # no single strategy > 60%
    sleeve_locks:   Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, float]:
    """Solve max U(w) = w'μ - λ/2 · w'Σw, subject to:
      Σ w_i = 1
      min_weight ≤ w_i ≤ max_weight   (per-strategy bounds)
      sleeve_locks[strat] applied if provided   (overrides per-strat bounds)

    Uses scipy.optimize.minimize SLSQP for robustness (handles bounds +
    equality + per-strategy bounds cleanly).

    Returns dict {strategy_name: weight}.
    """
    from scipy.optimize import minimize  # local import

    N = len(strategies)

    def neg_utility(w):
        port_mu  = float(np.dot(w, excess_returns))
        port_var = float(w @ covariance @ w)
        return -(port_mu - 0.5 * risk_aversion * port_var)

    # Initial guess: equal weight
    x0 = np.full(N, 1.0 / N)

    # Bounds (per-strategy)
    bounds: list[tuple[float, float]] = []
    locks = sleeve_locks or {}
    for s in strategies:
        if s in locks:
            bounds.append(locks[s])
        else:
            bounds.append((min_weight, max_weight))

    # Sum to 1 constraint
    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]

    result = minimize(
        neg_utility, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        # Fallback: equal weight
        return {s: 1.0 / N for s in strategies}

    return {s: float(round(w, 6)) for s, w in zip(strategies, result.x)}


# ─────────────────────────────────────────────────────────────────────────────
# Top-level analysis
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ShrinkageAnalysis:
    """Full Bayes-Stein + Ledoit-Wolf shrinkage analysis result."""
    strategies:           list[str]
    n_weeks:              int

    # Raw (sample) inputs
    sample_means_weekly:  dict[str, float]
    sample_vols_weekly:   dict[str, float]
    sample_sharpe_ann:    dict[str, float]
    sample_covariance:    list[list[float]]

    # Shrunk inputs
    shrunk_means_weekly:  dict[str, float]
    shrunk_sharpe_ann:    dict[str, float]
    shrunk_covariance:    list[list[float]]
    mean_shrinkage_w:     float
    cov_shrinkage_alpha:  float
    grand_mean_weekly:    float

    # Solutions under different constraint sets
    weights_unconstrained: dict[str, float]    # only sum=1, [0, 1] bounds
    weights_with_caps:     dict[str, float]    # + max 60% per strategy
    weights_sleeve_locked: dict[str, float]    # + K1 36% + CTA 10% locked (current SAA structure)
    weights_current_saa:   dict[str, float]    # for comparison

    # Forward Sharpe estimates at each weight set
    forward_sharpe_estimates: dict[str, float]


def run_shrinkage_analysis(
    returns_weekly_path: str | Path = "data/portfolio_replay/v1_per_strategy_returns_weekly.parquet",
    risk_aversion:       float = 2.0,
) -> ShrinkageAnalysis:
    """End-to-end Bayes-Stein + Ledoit-Wolf shrinkage analysis."""
    df = pd.read_parquet(returns_weekly_path)
    # Coerce pandas nullable Float64 columns to plain float64 so numpy.cov works
    df = df.astype("float64").fillna(0.0)
    strategies = list(df.columns)
    n_weeks = len(df)

    # Excess weekly returns (subtract weekly RFR)
    returns_arr = df.values.astype(np.float64)
    excess_weekly = returns_arr - WEEKLY_RFR_USD

    sample_means_w = excess_weekly.mean(axis=0)
    sample_cov     = np.cov(excess_weekly.T, ddof=1)
    sample_vols_w  = np.sqrt(np.diag(sample_cov))

    # Sample Sharpe (annualized via sqrt(52))
    sample_sharpe = sample_means_w / sample_vols_w * math.sqrt(52)

    # Bayes-Stein shrink the means
    shrunk_means_w, mean_w = bayes_stein_shrink_mean(
        sample_means_w, sample_cov, n_obs=n_weeks,
    )

    # Ledoit-Wolf shrink the covariance (using raw weekly returns, not excess —
    # both are equivalent up to a constant shift)
    shrunk_cov, cov_alpha = ledoit_wolf_shrink_cov(excess_weekly)

    shrunk_vols_w = np.sqrt(np.diag(shrunk_cov))
    shrunk_sharpe = shrunk_means_w / shrunk_vols_w * math.sqrt(52)

    # Compute grand mean for reporting
    ones = np.ones(len(strategies))
    cov_inv = np.linalg.pinv(sample_cov)
    grand_mean = float((ones @ cov_inv @ sample_means_w) / float(ones @ cov_inv @ ones))

    # Solve under 3 constraint sets
    w_unc = solve_optimal_weights(
        shrunk_means_w, shrunk_cov, strategies,
        risk_aversion=risk_aversion, max_weight=1.0,
    )
    w_caps = solve_optimal_weights(
        shrunk_means_w, shrunk_cov, strategies,
        risk_aversion=risk_aversion, max_weight=0.60,
    )
    w_locked = solve_optimal_weights(
        shrunk_means_w, shrunk_cov, strategies,
        risk_aversion=risk_aversion, max_weight=0.60,
        sleeve_locks=SLEEVE_LOCKS,
    )

    # Forward Sharpe estimates at each weight set (use shrunk inputs)
    def _port_sharpe(weights: dict[str, float]) -> float:
        w = np.array([weights[s] for s in strategies])
        port_mu = float(np.dot(w, shrunk_means_w))
        port_var = float(w @ shrunk_cov @ w)
        if port_var <= 0:
            return float("nan")
        return port_mu / math.sqrt(port_var) * math.sqrt(52)

    forward_sharpes = {
        "unconstrained": _port_sharpe(w_unc),
        "with_caps":     _port_sharpe(w_caps),
        "sleeve_locked": _port_sharpe(w_locked),
        "current_saa":   _port_sharpe(CURRENT_SAA),
        "equal_weight":  _port_sharpe({s: 0.25 for s in strategies}),
    }

    return ShrinkageAnalysis(
        strategies                = strategies,
        n_weeks                   = n_weeks,
        sample_means_weekly       = {s: float(v) for s, v in zip(strategies, sample_means_w)},
        sample_vols_weekly        = {s: float(v) for s, v in zip(strategies, sample_vols_w)},
        sample_sharpe_ann         = {s: float(v) for s, v in zip(strategies, sample_sharpe)},
        sample_covariance         = sample_cov.tolist(),
        shrunk_means_weekly       = {s: float(v) for s, v in zip(strategies, shrunk_means_w)},
        shrunk_sharpe_ann         = {s: float(v) for s, v in zip(strategies, shrunk_sharpe)},
        shrunk_covariance         = shrunk_cov.tolist(),
        mean_shrinkage_w          = float(mean_w),
        cov_shrinkage_alpha       = float(cov_alpha),
        grand_mean_weekly         = grand_mean,
        weights_unconstrained     = w_unc,
        weights_with_caps         = w_caps,
        weights_sleeve_locked     = w_locked,
        weights_current_saa       = dict(CURRENT_SAA),
        forward_sharpe_estimates  = forward_sharpes,
    )
