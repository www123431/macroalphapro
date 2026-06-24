"""engine/research/bayesian_sharpe_updater.py — SLM Phase 2.5: Bayesian
posterior updating for Sharpe ratio (Layer 1 of the 3-layer validation
framework).

Replaces OBF-as-primary with the academically correct modern approach:
continuous Bayesian updating of P(true Sharpe > threshold | observed data),
naturally sequential, no fixed sample-size requirement, calibrated to
the strategy's pre-deploy honest target.

Academic basis:
  - Geweke (1989) "Exact Predictive Densities for Linear Models"
  - Bauwens & Lubrano (1998) "Bayesian inference in dynamic econometric
    models"
  - López de Prado, Adv FinML Ch 15.2 "The Sharpe Ratio as a Random
    Variable"
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio" — provides
    Var(Sharpe) formula used in the likelihood

Model:
  Prior:        Sharpe_ann ~ Normal(prior_mean, prior_sd²)
  Observation:  SharpeHat_n ~ Normal(Sharpe_ann, σ_obs²)
                σ_obs² = (1 + 0.5 × SharpeHat²) / n_years   (Bailey-LdP)
  Posterior:    Sharpe_ann | data ~ Normal(μ_post, σ_post²)  (conjugate)

Decision:
  ACCEPT   if  P(Sharpe > threshold | data) ≥ accept_posterior_prob
  REJECT   if  P(Sharpe > threshold | data) ≤ reject_posterior_prob
  CONTINUE otherwise

Default thresholds (conservative — calibrated against institutional norms):
  threshold = 0.50           (HLZ floor)
  accept_posterior_prob = 0.80
  reject_posterior_prob = 0.20

Calibration sanity check with PIT SN (Sharpe 2.10 over 123 months):
  After 24 months of similar data, P(Sharpe > 0.5) should approach 1.0
  → ACCEPT correctly. With only 6 months it stays in the indecision
  zone because posterior is still prior-dominated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _stats


class BayesianDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CONTINUE = "CONTINUE"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class BayesianSharpeResult:
    """Output of a single posterior update."""

    n_months: int
    observed_sharpe_ann: float
    prior_mean: float
    prior_sd: float
    posterior_mean: float
    posterior_sd: float
    posterior_prob_above_threshold: float
    threshold: float
    decision: BayesianDecision
    rationale: str


def _observation_variance(
    sharpe_hat: float, n_obs: int, periods_per_year: int = 12,
) -> float:
    """Bailey-LdP standard error of annualized Sharpe estimate.

    Var(SR_ann) = (1 + 0.5 × SR_ann²) / n_years

    T1.2 (2026-06-05): periods_per_year now plumbed through; pre-fix
    hardcoded /12 mis-annualized weekly (2.08×) and daily (5.5×)
    observation panels.
    """
    n_years = n_obs / float(periods_per_year)
    if n_years <= 0:
        return float("inf")
    return (1.0 + 0.5 * sharpe_hat ** 2) / n_years


def bayesian_sharpe_update(
    *,
    sleeve_returns: pd.Series,
    prior_mean: float,
    prior_sd: float = 0.5,
    threshold: float = 0.50,
    accept_posterior_prob: float = 0.80,
    reject_posterior_prob: float = 0.20,
    min_months_for_decision: int = 3,
    periods_per_year: int = 12,
) -> BayesianSharpeResult:
    """Run one Bayesian update step on observed monthly returns.

    Conjugate Normal-Normal: posterior precision = sum of prior + obs
    precisions; posterior mean = precision-weighted average.

    Parameters:
      sleeve_returns:  monthly returns observed during paper trade
      prior_mean:      the honest_deploy_target Sharpe from P-D8 audit
      prior_sd:        prior uncertainty (default 0.5 = moderate)
      threshold:       Sharpe value we test exceedance of (default 0.50,
                       the HLZ floor)
      accept_posterior_prob:  P > this → ACCEPT
      reject_posterior_prob:  P < this → REJECT
      min_months_for_decision: short-circuit return INSUFFICIENT for
                       windows shorter than this
    """
    r = sleeve_returns.dropna()
    n = len(r)
    if n < min_months_for_decision:
        return BayesianSharpeResult(
            n_months=n,
            observed_sharpe_ann=0.0,
            prior_mean=prior_mean, prior_sd=prior_sd,
            posterior_mean=prior_mean, posterior_sd=prior_sd,
            posterior_prob_above_threshold=float(
                1 - _stats.norm.cdf((threshold - prior_mean) / prior_sd)
            ),
            threshold=threshold,
            decision=BayesianDecision.INSUFFICIENT,
            rationale=(
                f"n_months={n} < min_months_for_decision="
                f"{min_months_for_decision}; posterior is still prior-dominated"
            ),
        )

    # Observation likelihood (Bailey-LdP variance)
    mean_r = float(r.mean())
    sd_r = float(r.std(ddof=1))
    if sd_r == 0:
        # Degenerate; cannot compute Sharpe
        return BayesianSharpeResult(
            n_months=n, observed_sharpe_ann=0.0,
            prior_mean=prior_mean, prior_sd=prior_sd,
            posterior_mean=prior_mean, posterior_sd=prior_sd,
            posterior_prob_above_threshold=0.5,
            threshold=threshold,
            decision=BayesianDecision.INSUFFICIENT,
            rationale="zero return volatility — degenerate sample",
        )

    sharpe_hat = (mean_r / sd_r) * math.sqrt(periods_per_year)
    obs_var = _observation_variance(sharpe_hat, n, periods_per_year=periods_per_year)
    obs_sd = math.sqrt(obs_var)

    # Conjugate Normal-Normal update
    prior_var = prior_sd ** 2
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + sharpe_hat / obs_var)
    post_sd = math.sqrt(post_var)

    # P(Sharpe > threshold | data)
    z_threshold = (threshold - post_mean) / post_sd
    p_above = float(1 - _stats.norm.cdf(z_threshold))

    # Decision
    if p_above >= accept_posterior_prob:
        decision = BayesianDecision.ACCEPT
        rationale = (
            f"P(Sharpe > {threshold:.2f} | data) = {p_above:.3f} "
            f"≥ {accept_posterior_prob:.2f}; posterior mean={post_mean:.3f} "
            f"sd={post_sd:.3f} after {n}mo"
        )
    elif p_above <= reject_posterior_prob:
        decision = BayesianDecision.REJECT
        rationale = (
            f"P(Sharpe > {threshold:.2f} | data) = {p_above:.3f} "
            f"≤ {reject_posterior_prob:.2f}; posterior mean={post_mean:.3f} "
            f"sd={post_sd:.3f} after {n}mo"
        )
    else:
        decision = BayesianDecision.CONTINUE
        rationale = (
            f"P(Sharpe > {threshold:.2f} | data) = {p_above:.3f} "
            f"in ({reject_posterior_prob:.2f}, {accept_posterior_prob:.2f}); "
            f"need more data; posterior mean={post_mean:.3f} sd={post_sd:.3f}"
        )

    return BayesianSharpeResult(
        n_months=n,
        observed_sharpe_ann=sharpe_hat,
        prior_mean=prior_mean, prior_sd=prior_sd,
        posterior_mean=post_mean, posterior_sd=post_sd,
        posterior_prob_above_threshold=p_above,
        threshold=threshold,
        decision=decision,
        rationale=rationale,
    )
