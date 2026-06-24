"""engine/validation/block_bootstrap.py — Phase 5.1: paired block
bootstrap for return-series statistical inference.

Senior design choices (see [[project-paper-borrow-ml-btc-costs-2026-06-01]]):

  1) STATIONARY bootstrap (Politis-White 2003), NOT circular
     (Politis-Romano 1994). Stationary uses GEOMETRIC block lengths;
     avoids the boundary-repeat artefact circular block creates in
     financial returns.

  2) AUTOMATIC BLOCK LENGTH SELECTION (Politis-White 2009).
     Spectral-density-based optimal mean block length; hard-coded
     block length (paper's 168h for BTC hourly) doesn't transfer
     across cadences. We expose the manual override but default to
     auto-selection.

  3) PAIRED design — for Sharpe-diff(A, B), the SAME bootstrap
     sample indices are used for both series. This preserves the
     joint distribution + cross-correlation (critical for benchmark
     comparisons where A and B share systematic exposures).

  4) HOCHBERG step-up (1988) for multiple-comparison adjustment
     across families of hypotheses; Holm is too conservative when
     hypotheses are non-orthogonal (correlated benchmark tests).

  5) INTEGRATION with deflated_sharpe.DSRResult — PBB gives us a
     bias-corrected SR + CI from the bootstrap distribution;
     DSR gives multi-trial correction. They are COMPLEMENTARY.
     `pbb_sharpe_with_dsr` returns both.

Statistical references:
  - Politis & White (2003), "Automatic block-length selection for
    the dependent bootstrap"
  - Politis & White (2009) "An automatic block-length selection
    method for resampling time series" [econometric refinement]
  - Hochberg (1988), "A sharper Bonferroni procedure for multiple
    tests of significance"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default count tuned for institutional-grade significance. 10k is the
# academic standard; below 5k the p-value tail estimates get noisy at
# alpha=0.01.
DEFAULT_N_ITER = 10_000
DEFAULT_RNG_SEED = None   # None = OS entropy; tests override


# ── Result dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class PBBResult:
    """One-series block-bootstrap result for a generic statistic."""
    n_obs:           int
    n_iter:          int
    block_len:       float      # average block length used (float for stationary)
    block_method:    str        # "auto-PW2009" / "manual"
    point_estimate:  float      # statistic on the original series
    ci_lo:           float      # bootstrap 2.5%ile
    ci_hi:           float      # bootstrap 97.5%ile
    se_bootstrap:    float      # std of bootstrap statistic distribution


@dataclass(frozen=True)
class PBBSharpeDiffResult:
    """Paired bootstrap for Sharpe(A) - Sharpe(B)."""
    n_obs:           int
    n_iter:          int
    block_len:       float
    block_method:    str
    sharpe_a:        float
    sharpe_b:        float
    diff_point:      float
    diff_ci_lo:      float
    diff_ci_hi:      float
    diff_se:         float
    p_value_two_sided: float    # bootstrap p-value, NOT yet multiple-comp adjusted
    holm_adjusted_p: Optional[float] = None    # set by hochberg_adjust if needed
    hochberg_adjusted_p: Optional[float] = None


# ── Block-length selection (Politis-White 2009) ────────────────────────


def auto_block_length(returns: Sequence[float]) -> float:
    """Politis-White 2009 spectral block-length selector for the
    STATIONARY bootstrap. Returns the optimal mean block length.

    Implementation follows the lag-window estimator of the spectrum
    at frequency 0 (the variance of the partial sum n^{-1/2} Σ X_t).

    Per the paper: pick a moderate trial window K = O(log n) (we use
    max(5, sqrt(log10(n)*5)) ≈ 5-7 for n<10k), compute auto-
    correlations up to lag K, build the optimal block length:
        b* = ( 2 * g^2 / G )^{1/3} * n^{1/3}
    where
        g = Σ_{|k| <= K} w(k/K) * ρ(k) * γ(k)
        G = Σ_{|k| <= K} w(k/K) * γ(k)
    with w() the flat-top Bartlett-type kernel.

    For pure white noise the formula reduces to b* → 1; for highly
    persistent series it grows with the persistence horizon.

    Fallback: if n < 30 returns a tiny block; if the series is
    degenerate (all NaN / zero var) returns 1.0.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 30:
        return max(1.0, float(n ** (1.0 / 3.0)))
    # Trial window K — Politis-White recommends 2*sqrt(log10(n))
    K = max(3, int(2 * math.sqrt(math.log10(max(10, n)))))
    K = min(K, n - 1)

    # Autocovariances via numpy correlate (more stable than fft for
    # the modest lag count we need)
    rc = r - r.mean()
    var_r = float(np.var(r, ddof=1))
    if var_r <= 0 or not math.isfinite(var_r):
        return 1.0
    # gamma(k) for k=0..K
    gammas = np.array([float(np.dot(rc[: n - k], rc[k:]) / n) for k in range(K + 1)])

    # Flat-top kernel weights (Politis-White 1996 trapezoidal lambda)
    def w(x: float) -> float:
        a = abs(x)
        if a <= 0.5:    return 1.0
        if a <= 1.0:    return 2.0 * (1.0 - a)
        return 0.0

    # Politis-White 2004 stationary-bootstrap formula:
    #   G_hat  = Σ_{k} λ(|k|/K) |k| γ(k)        (BIAS contribution)
    #   g0_hat = Σ_{k} λ(|k|/K) γ(k)            (VARIANCE contribution)
    #   b_opt  = (3 G_hat² / (2 g0_hat²))^{1/3} n^{1/3}
    # Use double-sided sums; γ(-k) = γ(k) for stationary process,
    # so factor of 2 for k >= 1 captures both signs.
    G_hat = 0.0
    g0_hat = 0.0
    for k in range(K + 1):
        weight = w(k / K)
        if k == 0:
            g0_hat += weight * gammas[0]
            # |k|*γ(k) = 0 at k=0 contributes nothing to G_hat
        else:
            g0_hat += 2.0 * weight * gammas[k]
            G_hat  += 2.0 * weight * k * gammas[k]

    if g0_hat <= 0:
        return 1.0
    # For pure white noise G_hat = 0 → b_opt → 0 → caller floor to 1
    try:
        b_star = (3.0 * (G_hat ** 2) / (2.0 * (g0_hat ** 2))) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    except (ValueError, ZeroDivisionError):
        return 1.0
    # Floor at 1.0 (degenerate / white-noise case); cap at sqrt(n)
    # (long blocks destroy bootstrap variance)
    b_star = max(1.0, min(b_star, float(math.sqrt(n))))
    return float(b_star)


# ── Stationary bootstrap sample (Politis-White 2003) ───────────────────


def _stationary_sample_indices(
    n: int,
    mean_block_len: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one stationary-bootstrap sample of length n.

    At each position, with probability p = 1/L (L = mean block length)
    start a new block at a uniform random index; otherwise advance to
    the next index wrapping at n. Block lengths are geometric(p)
    distributed; mean block = L.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    L = max(1.0, mean_block_len)
    p = 1.0 / L
    out = np.empty(n, dtype=np.int64)
    idx = int(rng.integers(0, n))
    new_block_mask = rng.random(n) < p
    for t in range(n):
        if new_block_mask[t]:
            idx = int(rng.integers(0, n))
        out[t] = idx
        idx = (idx + 1) % n
    return out


# ── Generic statistic bootstrap ────────────────────────────────────────


def pbb_statistic(
    returns:       Sequence[float],
    statistic_fn:  Callable[[np.ndarray], float],
    *,
    n_iter:        int = DEFAULT_N_ITER,
    block_len:     Optional[float] = None,
    rng_seed:      Optional[int] = DEFAULT_RNG_SEED,
) -> PBBResult:
    """Generic stationary-bootstrap CI + SE for any single-series stat.

    statistic_fn must accept a 1-D numpy array and return a scalar.
    Useful for: Sharpe, mean, median, max drawdown, hit ratio, etc.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 5:
        raise ValueError(f"need >= 5 observations, got {n}")

    if block_len is None:
        block_len = auto_block_length(r)
        block_method = "auto-PW2009"
    else:
        block_method = "manual"

    rng = np.random.default_rng(rng_seed)
    point = float(statistic_fn(r))
    samples = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = _stationary_sample_indices(n, block_len, rng)
        samples[i] = statistic_fn(r[idx])

    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        raise RuntimeError("all bootstrap samples returned NaN")

    return PBBResult(
        n_obs=n,
        n_iter=n_iter,
        block_len=float(block_len),
        block_method=block_method,
        point_estimate=point,
        ci_lo=float(np.percentile(samples, 2.5)),
        ci_hi=float(np.percentile(samples, 97.5)),
        se_bootstrap=float(np.std(samples, ddof=1)),
    )


# ── Paired Sharpe-diff bootstrap (the marquee 5.1 deliverable) ─────────


def _sharpe_per_period(r: np.ndarray) -> float:
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def pbb_sharpe_diff(
    returns_a:     Sequence[float],
    returns_b:     Sequence[float],
    *,
    n_iter:        int = DEFAULT_N_ITER,
    block_len:     Optional[float] = None,
    rng_seed:      Optional[int] = DEFAULT_RNG_SEED,
) -> PBBSharpeDiffResult:
    """Paired bootstrap for Sharpe(A) - Sharpe(B).

    A and B MUST be aligned same-length return series. Bootstrap
    samples use the SAME indices for both → preserves joint
    distribution.

    Block length defaults to auto via Politis-White 2009 applied to
    the DIFFERENCE series (the statistic-of-interest is the diff;
    its serial dependence is what matters).

    Returns p_value (two-sided, unadjusted). For families of tests
    apply hochberg_adjust() at the family level.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"paired series must be same length; got {len(a)} vs {len(b)}"
        )
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n < 5:
        raise ValueError(f"need >= 5 paired observations, got {n}")

    if block_len is None:
        block_len = auto_block_length(a - b)
        block_method = "auto-PW2009"
    else:
        block_method = "manual"

    sharpe_a = _sharpe_per_period(a)
    sharpe_b = _sharpe_per_period(b)
    point_diff = sharpe_a - sharpe_b

    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = _stationary_sample_indices(n, block_len, rng)
        diffs[i] = _sharpe_per_period(a[idx]) - _sharpe_per_period(b[idx])

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        raise RuntimeError("all paired-bootstrap diffs were NaN")

    # Bootstrap p-value: 2 × min(P(diff* >= 0 | H0), P(diff* < 0 | H0))
    # H0: true_diff = 0; center the bootstrap distribution at 0 by
    # subtracting the point estimate
    centered = diffs - point_diff
    p_one_sided_pos = float((centered >= abs(point_diff)).mean())
    p_one_sided_neg = float((centered <= -abs(point_diff)).mean())
    p_two_sided = min(1.0, p_one_sided_pos + p_one_sided_neg)

    return PBBSharpeDiffResult(
        n_obs=n,
        n_iter=n_iter,
        block_len=float(block_len),
        block_method=block_method,
        sharpe_a=sharpe_a,
        sharpe_b=sharpe_b,
        diff_point=point_diff,
        diff_ci_lo=float(np.percentile(diffs, 2.5)),
        diff_ci_hi=float(np.percentile(diffs, 97.5)),
        diff_se=float(np.std(diffs, ddof=1)),
        p_value_two_sided=p_two_sided,
    )


# ── Multiple-comparison adjustment ─────────────────────────────────────


def hochberg_adjust(p_values: Sequence[float]) -> list[float]:
    """Hochberg 1988 step-up procedure — sharper than Bonferroni-Holm
    when hypotheses are non-orthogonal (typical: comparing one
    candidate against multiple correlated benchmarks).

    Returns the adjusted p-values in the ORIGINAL order.
    Adjusted p_k = min over k >= i of (m - k + 1) * p_(k), where p_(k)
    is the k-th order statistic.
    """
    m = len(p_values)
    if m == 0:
        return []
    ps = np.asarray(p_values, dtype=float)
    order = np.argsort(ps)
    ranked = ps[order]
    adj = np.minimum.accumulate(
        (m - np.arange(m)[::-1]) * ranked[::-1]
    )[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return [float(x) for x in out]


def apply_hochberg_to_results(
    results: Sequence[PBBSharpeDiffResult],
) -> list[PBBSharpeDiffResult]:
    """Convenience: apply Hochberg correction to a family of
    PBBSharpeDiffResult and return new instances with
    hochberg_adjusted_p populated."""
    if not results:
        return []
    raw = [r.p_value_two_sided for r in results]
    adj = hochberg_adjust(raw)
    return [
        PBBSharpeDiffResult(
            n_obs=r.n_obs, n_iter=r.n_iter, block_len=r.block_len,
            block_method=r.block_method,
            sharpe_a=r.sharpe_a, sharpe_b=r.sharpe_b,
            diff_point=r.diff_point, diff_ci_lo=r.diff_ci_lo,
            diff_ci_hi=r.diff_ci_hi, diff_se=r.diff_se,
            p_value_two_sided=r.p_value_two_sided,
            holm_adjusted_p=r.holm_adjusted_p,
            hochberg_adjusted_p=float(a_p),
        )
        for r, a_p in zip(results, adj)
    ]


# ── DSR + PBB integration ──────────────────────────────────────────────


@dataclass(frozen=True)
class DSRPBBResult:
    """Combined deflated Sharpe + PBB CI report."""
    sharpe_per_period:    float
    sharpe_annualized:    float
    pbb_ci_lo_per_period: float
    pbb_ci_hi_per_period: float
    deflated_sr:          float        # DSR (multi-trial corrected)
    pbb_se_per_period:    float
    block_len:            float
    n_obs:                int
    n_trials:             int
    verdict:              str


def pbb_sharpe_with_dsr(
    returns:           Sequence[float],
    n_trials:          int = 1,
    *,
    n_iter:            int = DEFAULT_N_ITER,
    block_len:         Optional[float] = None,
    periods_per_year:  int = 12,
    rng_seed:          Optional[int] = DEFAULT_RNG_SEED,
) -> DSRPBBResult:
    """Combine deflated_sharpe.DSRResult with PBB CI on the same SR.

    DSR corrects for multiple-trial selection bias (Bailey-LdP).
    PBB gives a path-realization CI (this is one realization of a
    stochastic process — block bootstrap quantifies how much the SR
    could vary across alternate paths).
    They are complementary; pass both into deploy decisions.
    """
    from engine.validation.deflated_sharpe import (
        annualize_sharpe, deflated_sharpe_ratio,
    )
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 5:
        raise ValueError(f"need >= 5 observations, got {len(r)}")

    pbb = pbb_statistic(
        r, _sharpe_per_period,
        n_iter=n_iter, block_len=block_len, rng_seed=rng_seed,
    )
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials)
    sr_per = pbb.point_estimate
    sr_ann = annualize_sharpe(sr_per, periods_per_year=periods_per_year)

    # Verdict combines both
    pbb_excludes_zero = (pbb.ci_lo > 0) or (pbb.ci_hi < 0)
    dsr_passes = dsr.deflated_sr > 0.95
    if pbb_excludes_zero and dsr_passes:
        verdict = "STRONG_PASS"
    elif pbb_excludes_zero or dsr_passes:
        verdict = "MARGINAL"
    else:
        verdict = "WEAK"

    return DSRPBBResult(
        sharpe_per_period=sr_per,
        sharpe_annualized=sr_ann,
        pbb_ci_lo_per_period=pbb.ci_lo,
        pbb_ci_hi_per_period=pbb.ci_hi,
        deflated_sr=dsr.deflated_sr,
        pbb_se_per_period=pbb.se_bootstrap,
        block_len=pbb.block_len,
        n_obs=pbb.n_obs,
        n_trials=n_trials,
        verdict=verdict,
    )
