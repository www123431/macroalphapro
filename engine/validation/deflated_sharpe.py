"""engine/validation/deflated_sharpe.py — PSR + Deflated Sharpe Ratio.

Implements Bailey & López de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"
(Journal of Portfolio Management).

Why this matters for THIS project: the 5 surviving strategies are the
winners of a 35+ path search (Path A … Path AN). An observed Sharpe of
0.54 on the survivor is inflated by (a) short sample, (b) non-normal
returns, and (c) the multiple-testing selection itself. The Deflated
Sharpe Ratio answers: "given I ran N trials, what is the probability the
TRUE Sharpe of this survivor is greater than zero?"

Three functions, all operating on per-period (here: weekly) returns:

  sharpe_ratio(returns)              — plain per-period SR (not annualized)
  probabilistic_sharpe_ratio(...)   — PSR: P(true SR > benchmark SR)
                                       given sample length + skew + kurtosis
  expected_max_sharpe(n_trials, ..) — SR*_0: expected MAX Sharpe under the
                                       null across N independent trials
  deflated_sharpe_ratio(...)        — DSR = PSR evaluated at benchmark
                                       SR*_0. DSR > 0.95 is the usual
                                       "survives multiple testing" bar.

All math is on the NON-annualized per-period Sharpe (the formulas are
scale-free in the period; annualization is cosmetic and applied only in
reporting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats


_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    """Full deflated-Sharpe diagnostic for one return series."""
    n_obs:              int
    sharpe_per_period:  float    # non-annualized
    sharpe_annualized:  float
    skew:               float
    excess_kurtosis:    float    # kurtosis - 3 (0 for normal)
    n_trials:           int
    var_sr_across_trials: float  # variance of SR estimates across trials
    expected_max_sr:    float    # SR*_0 benchmark (per-period)
    psr_vs_zero:        float    # PSR against SR*=0 (sample-only correction)
    deflated_sr:        float    # PSR against SR*_0 (full correction)
    verdict:            str      # human-readable pass/marginal/fail


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Per-period Sharpe ratio: mean(r) / std(r) (population-style ddof=1).

    Excess-of-risk-free is the caller's responsibility — pass excess
    returns if you want a risk-free-adjusted SR. For weekly strategy
    returns the RF drag is tiny; we report both raw and (in factor
    attribution) excess.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def annualize_sharpe(sr_per_period: float, periods_per_year: int = 52) -> float:
    """Scale a per-period Sharpe to annual: SR_ann = SR_period * sqrt(P)."""
    return sr_per_period * math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    returns:      Sequence[float],
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey-LdP eq. for PSR).

    PSR(SR*) = Φ( (ŜR − SR*) · sqrt(T−1) /
                  sqrt(1 − γ3·ŜR + (γ4−1)/4 · ŜR²) )

    where ŜR is the observed PER-PERIOD Sharpe, T the number of
    observations, γ3 the skewness, γ4 the kurtosis (NOT excess; the
    formula uses raw kurtosis where normal = 3), and Φ the standard
    normal CDF.

    Returns P(true SR > sr_benchmark). Both ŜR and sr_benchmark are
    per-period. A PSR near 1.0 means high confidence the true Sharpe
    beats the benchmark; near 0.5 means coin-flip; below 0.5 means the
    observed edge is probably noise relative to the benchmark.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 3:
        return float("nan")

    sr_hat = sharpe_ratio(r)
    if math.isnan(sr_hat):
        return float("nan")

    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # raw (normal=3)

    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    # Guard: denom can go negative for extreme skew/kurt × high SR.
    if denom <= 0:
        return float("nan")

    z = (sr_hat - sr_benchmark) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(
    n_trials:             int,
    var_sr_across_trials: float,
) -> float:
    """Expected MAXIMUM per-period Sharpe across N independent trials
    under the null (true SR = 0 for all), Bailey-LdP:

      E[max SR_N] ≈ sqrt(V) · [ (1−γ)·Φ⁻¹(1 − 1/N)
                                + γ·Φ⁻¹(1 − 1/(N·e)) ]

    where V = variance of the SR estimates across the N trials, γ the
    Euler-Mascheroni constant, Φ⁻¹ the normal quantile. This is the
    benchmark SR*_0 a survivor must beat to be more than the luckiest
    of N coin flips.

    n_trials <= 1 returns 0 (no selection, nothing to deflate against).
    """
    if n_trials <= 1 or var_sr_across_trials <= 0:
        return 0.0
    sqrt_v = math.sqrt(var_sr_across_trials)
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sqrt_v * ((1.0 - _EULER_MASCHERONI) * q1
                           + _EULER_MASCHERONI * q2))


def _estimate_var_sr_theoretical(returns: Sequence[float]) -> float:
    """Fallback estimate of Var(SR) when the actual cross-trial variance
    is unknown: the asymptotic variance of a single Sharpe estimate,
    Var(ŜR) ≈ (1/(T−1))·(1 + ŜR²/2). This UNDER-states the true
    cross-trial dispersion (which also includes strategy-design
    variation), so DSR computed from it is OPTIMISTIC. The honest input
    is the variance of Sharpe ratios across all N trials actually run —
    pass var_sr_across_trials explicitly when you have it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 3:
        return float("nan")
    sr = sharpe_ratio(r)
    return (1.0 / (T - 1)) * (1.0 + 0.5 * sr ** 2)


def deflated_sharpe_ratio(
    returns:              Sequence[float],
    n_trials:             int,
    var_sr_across_trials: Optional[float] = None,
    periods_per_year:     int = 52,
) -> DSRResult:
    """Full Deflated Sharpe Ratio diagnostic.

    Args:
      returns:              per-period (weekly) return series of the
                            SURVIVING strategy.
      n_trials:             number of independent strategy configurations
                            tried during research (selection breadth).
                            For this project, the Path A…AN search → ~35.
      var_sr_across_trials: variance of the PER-PERIOD Sharpe estimates
                            across those N trials. THE honest input. If
                            None, falls back to a single-strategy
                            theoretical estimate that is optimistic
                            (under-states dispersion) — flagged in the
                            result note.
      periods_per_year:     52 for weekly.

    Returns DSRResult. The headline number is `deflated_sr`:
      >= 0.95  → survives multiple-testing at 95% (institutional bar)
      0.90-0.95→ marginal
      < 0.90   → does NOT clear the bar; the observed edge is plausibly
                 the luckiest of N trials, not true alpha.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)

    sr_pp  = sharpe_ratio(r)
    sr_ann = annualize_sharpe(sr_pp, periods_per_year)
    skew   = float(stats.skew(r, bias=False)) if T >= 3 else float("nan")
    ekurt  = float(stats.kurtosis(r, fisher=True, bias=False)) if T >= 3 else float("nan")

    if var_sr_across_trials is None:
        var_sr = _estimate_var_sr_theoretical(r)
    else:
        var_sr = var_sr_across_trials

    sr_star = expected_max_sharpe(n_trials, var_sr)
    psr0    = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    dsr     = probabilistic_sharpe_ratio(r, sr_benchmark=sr_star)

    if math.isnan(dsr):
        verdict = "UNDEFINED (insufficient / degenerate sample)"
    elif dsr >= 0.95:
        verdict = "PASS — survives multiple-testing at 95%"
    elif dsr >= 0.90:
        verdict = "MARGINAL — 90-95%, treat with caution"
    else:
        verdict = "FAIL — does not clear multiple-testing bar"

    return DSRResult(
        n_obs                = T,
        sharpe_per_period    = sr_pp,
        sharpe_annualized    = sr_ann,
        skew                 = skew,
        excess_kurtosis      = ekurt,
        n_trials             = n_trials,
        var_sr_across_trials = var_sr,
        expected_max_sr      = sr_star,
        psr_vs_zero          = psr0,
        deflated_sr          = dsr,
        verdict              = verdict,
    )


def var_sr_from_trial_sharpes(trial_sharpes: Sequence[float]) -> float:
    """Compute Var(SR across trials) from the actual per-period Sharpe
    ratios of every trial run. THIS is the correct input to
    deflated_sharpe_ratio's var_sr_across_trials when available — the
    honest measure of how much Sharpe varied across the research search.
    """
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 2:
        return float("nan")
    return float(s.var(ddof=1))
