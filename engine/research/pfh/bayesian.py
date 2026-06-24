"""engine/research/pfh/bayesian.py — Beta-Binomial posterior with
empirical-Bayes shrinkage and credible intervals.

Model:
    p ~ Beta(α₀, β₀)
    success | p ~ Bernoulli(p)

Per-family posterior:
    p | data ~ Beta(α₀ + n_green_eff, β₀ + n_red_eff)
    n_green_eff = n_green + 0.5 * n_yellow
    n_red_eff   = n_red   + 0.5 * n_yellow

Hyperprior choice (weak-informative, centered on overall base rate):
    α₀ = 1 + base_rate * w
    β₀ = 1 + (1 - base_rate) * w
    w  = "prior strength" — equivalent sample size of the prior

  We default w=4 (≈ 4 prior pseudo-observations, base rate as center).
  With our N=36, this means per-family posterior gets shrunk toward
  base_rate when n_in_family is small; when n_in_family is large the
  data dominates.

Credible intervals via scipy.stats.beta.ppf (5th, 50th, 95th).

DOCTRINE NOTE: this module does NOT update from PFH-suggested outcomes.
The caller (proposer.py) is responsible for filtering the input
LabeledMechanism list to exclude PFH-originated entries when building
the prior. See engine/research/pfh/__init__.py rule 4 (no circular).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from scipy.stats import beta as _beta_dist
    _HAS_SCIPY = True
except ImportError:  # graceful: fall back to a hand-rolled CDF inversion
    _HAS_SCIPY = False


@dataclass
class BetaBinomialPosterior:
    """Posterior summary for one (family, optional sub-cell)."""
    n_green:           int
    n_yellow:          int
    n_red:             int
    alpha_prior:       float
    beta_prior:        float
    alpha_post:        float
    beta_post:         float
    posterior_mean:    float
    credible_05:       float
    credible_50:       float
    credible_95:       float
    # Effective sample sizes used in the update
    n_green_effective: float
    n_red_effective:   float

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _hyperprior_alpha_beta(
    base_rate: float, prior_strength: float = 4.0,
) -> tuple[float, float]:
    """Return (α₀, β₀) for a weak-informative prior centered on base_rate.

    prior_strength = pseudo-observations of weight given to the prior.
    Higher prior_strength = more shrinkage toward base_rate.
    """
    if not (0 < base_rate < 1):
        # Degenerate base rate → use Jeffreys prior (0.5, 0.5)
        return (0.5, 0.5)
    alpha = 1.0 + base_rate * prior_strength
    beta  = 1.0 + (1.0 - base_rate) * prior_strength
    return alpha, beta


def _beta_quantile(alpha: float, beta: float, q: float) -> float:
    """Inverse Beta CDF (PPF). Uses scipy if available; otherwise a
    bisection fallback using a regularized incomplete beta function.

    The fallback is for environments without scipy — it's ~100x slower
    but accurate to 1e-6 which is more than enough for credible
    intervals reported to 3 decimal places.
    """
    if _HAS_SCIPY:
        return float(_beta_dist.ppf(q, alpha, beta))

    # Bisection fallback
    from math import lgamma, log, exp
    def _log_beta(a: float, b: float) -> float:
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    def _betainc(a: float, b: float, x: float) -> float:
        """Regularized incomplete beta via continued fraction (Numerical
        Recipes 6.4). Accurate enough for our quantile needs."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        bt = exp(-_log_beta(a, b) + a * log(x) + b * log(1.0 - x))
        # Lentz's method for the continued fraction
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * _cf(a, b, x) / a
        return 1.0 - bt * _cf(b, a, 1.0 - x) / b

    def _cf(a: float, b: float, x: float, max_iter: int = 200) -> float:
        eps, fpmin = 1e-12, 1e-30
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin: d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin: c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin: d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin: c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    # Bisection over [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _betainc(alpha, beta, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)


def score_candidate(
    *,
    n_green:      int,
    n_yellow:     int,
    n_red:        int,
    base_rate:    float,
    prior_strength: float = 4.0,
) -> BetaBinomialPosterior:
    """Compute the posterior for one (family-or-cell) given counts.

    Args:
      n_green / n_yellow / n_red: observed in the cell
      base_rate:    overall P(GREEN) used to center the prior
      prior_strength: equivalent sample size of the prior (default 4)

    Returns: BetaBinomialPosterior with mean + 5/50/95 credible interval.
    """
    if n_green < 0 or n_yellow < 0 or n_red < 0:
        raise ValueError("counts must be non-negative")

    alpha_0, beta_0 = _hyperprior_alpha_beta(base_rate, prior_strength)
    n_green_eff = n_green + 0.5 * n_yellow
    n_red_eff   = n_red   + 0.5 * n_yellow

    alpha_post = alpha_0 + n_green_eff
    beta_post  = beta_0  + n_red_eff
    posterior_mean = alpha_post / (alpha_post + beta_post)

    q05 = _beta_quantile(alpha_post, beta_post, 0.05)
    q50 = _beta_quantile(alpha_post, beta_post, 0.50)
    q95 = _beta_quantile(alpha_post, beta_post, 0.95)

    return BetaBinomialPosterior(
        n_green=n_green,
        n_yellow=n_yellow,
        n_red=n_red,
        alpha_prior=alpha_0,
        beta_prior=beta_0,
        alpha_post=alpha_post,
        beta_post=beta_post,
        posterior_mean=round(posterior_mean, 4),
        credible_05=round(q05, 4),
        credible_50=round(q50, 4),
        credible_95=round(q95, 4),
        n_green_effective=n_green_eff,
        n_red_effective=n_red_eff,
    )
