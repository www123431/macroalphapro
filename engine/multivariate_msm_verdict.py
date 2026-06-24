"""
engine/multivariate_msm_verdict.py — Multivariate MSM v1 verdict utilities.

Spec: docs/spec_multivariate_msm_v1.md §3.1 / §3.2 / §3.4

Pure compute functions for the OOS verdict (Sharpe difference, Memmel Z,
Politis-Romano stationary bootstrap with Politis-White auto block size,
descriptive decision rule). Network-free; testable with synthetic returns.

The walk-forward backtest orchestration that USES these functions lives in
scripts/run_multivariate_msm_d6.py (needs FRED + yfinance, runs locally).

Per spec §3.1 honesty disclaimer: 6-year OOS for δ_annual=0.10 is severely
underpowered (~1003 calendar years required for 80% power at ρ=0.6). Verdict
treated as DESCRIPTIVE EFFECT-SIZE ESTIMATION with bootstrap CI, NOT formal
hypothesis test. Memmel Z reported descriptively only.

Boundary invariant: zero LLM imports. All deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec §3.2 locked thresholds (ALL changes require amend_spec(threshold_tweak)+)
_DELTA_SHARPE_PASS_THRESHOLD = 0.10        # spec §3.2 PASS row
_FALLBACK_UNINTERPRETABLE    = 0.50        # spec §3.4 UNINTERPRETABLE tier
_FALLBACK_STRONG_CAVEAT      = 0.25        # spec §3.4 25-50% tier
_FALLBACK_SOFT_CAVEAT        = 0.10        # spec §3.4 10-25% tier

# Sharpe annualization assumes monthly returns (spec lock; Lo 2002 convention)
_OBS_PER_YEAR = 12


# ─────────────────────────────────────────────────────────────────────────────
# Verdict result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MultivariateVerdict:
    """Output of compute_verdict() — full diagnostic snapshot per spec §10."""
    delta_sharpe:               float          # ΔŜ multivariate − univariate (annualized)
    sharpe_multivariate:        float          # annualized Sharpe of multivariate overlay
    sharpe_univariate:          float          # annualized Sharpe of univariate overlay
    bootstrap_ci_lower:         float          # 95% percentile bootstrap CI lower bound for ΔŜ
    bootstrap_ci_upper:         float          # 95% percentile bootstrap CI upper bound
    bootstrap_n_resamples:      int            # 1000 per spec §3.1
    bootstrap_block_size:       int            # Politis-White auto-selected stationary bootstrap block
    memmel_z:                   float          # descriptive only (spec §3.1)
    paired_correlation:         float          # ρ̂ between the two overlay return series
    fallback_rate:              float          # n_multivariate_failed / n_total OOS months (spec §3.4)
    n_oos_months:               int
    decision:                   str            # PASS / MARGINAL_INSUFFICIENT_PRECISION / MARGINAL / FAIL / UNINTERPRETABLE
    achieved_power_descriptive: float          # at observed ρ̂; descriptive only


# ─────────────────────────────────────────────────────────────────────────────
# Pure compute helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlay_returns(
    p_risk_on:    pd.Series,
    base_returns: pd.Series,
) -> pd.Series:
    """Build regime-overlay strategy returns from filtered MSM probability.

    Convention (locked v1):
        overlay_position_t = 2 × p_risk_on_t − 1   ∈ [-1, +1]
        overlay_return_t   = overlay_position_t × base_return_t

    Interpretation: long base when p_risk_on > 0.5, short when < 0.5, zero at 0.5.
    Magnitude scales with conviction.

    Args:
        p_risk_on: filtered MSM probability of risk-on regime, indexed by month-end.
        base_returns: monthly base returns (typically SPY or BAB factor); same index.

    Returns:
        pd.Series of overlay returns, indexed by intersection of inputs (no look-ahead).
    """
    common = p_risk_on.index.intersection(base_returns.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    aligned_p = p_risk_on.loc[common]
    aligned_b = base_returns.loc[common]
    position = 2.0 * aligned_p - 1.0
    return position * aligned_b


def annualized_sharpe(returns: pd.Series, obs_per_year: int = _OBS_PER_YEAR) -> float:
    """Lo 2002 frequency-consistent annualized Sharpe.

    Args:
        returns: monthly returns Series (drops NaN before computing).
        obs_per_year: 12 (monthly) per spec lock.

    Returns:
        (mean / std_ddof1) × √obs_per_year. NaN if std is 0 or insufficient data.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(obs_per_year))


def memmel_z_paired_sharpe_diff(
    returns_a:      pd.Series,
    returns_b:      pd.Series,
    obs_per_year:   int = _OBS_PER_YEAR,
) -> tuple[float, float, float]:
    """Memmel (2003) paired Sharpe-difference Z-statistic (DESCRIPTIVE only per spec §3.1).

    Paired observations on overlapping dates. Variance formula:
        V̂ = 2(1 − ρ̂²) + 0.5(Ŝ₁² + Ŝ₂² − 2 Ŝ₁ Ŝ₂ ρ̂²)

    where Ŝ_i are PER-PERIOD Sharpe values (not annualized).

    Returns:
        (z_stat, paired_corr, V_at_period_freq)

    DESCRIPTIVE ONLY — at 6-year OOS the test is severely underpowered
    (~1003 calendar years required for 80% power at typical ρ=0.6 / δ_annual=0.10);
    do NOT use as significance gate per spec §3.1.
    """
    a = returns_a.dropna()
    b = returns_b.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 12:
        return float("nan"), float("nan"), float("nan")
    a = a.loc[common].astype(float)
    b = b.loc[common].astype(float)

    rho = float(np.corrcoef(a.values, b.values)[0, 1])
    if not np.isfinite(rho):
        return float("nan"), float("nan"), float("nan")

    # Per-period Sharpe (Lo 2002 frequency convention)
    sa_p = float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0 else 0.0
    sb_p = float(b.mean() / b.std(ddof=1)) if b.std(ddof=1) > 0 else 0.0

    V = 2.0 * (1.0 - rho ** 2) + 0.5 * (sa_p ** 2 + sb_p ** 2 - 2.0 * sa_p * sb_p * rho ** 2)
    if V <= 0 or not np.isfinite(V):
        return float("nan"), rho, V

    T = len(common)
    delta_per_period = sa_p - sb_p
    z = delta_per_period / np.sqrt(V / T)
    return float(z), rho, float(V)


def bootstrap_sharpe_diff_ci(
    returns_a:      pd.Series,
    returns_b:      pd.Series,
    n_resamples:    int   = 1000,
    alpha:          float = 0.05,
    obs_per_year:   int   = _OBS_PER_YEAR,
    random_state:   int   = 42,
) -> tuple[float, float, int]:
    """Politis-Romano (1994) stationary bootstrap with Politis-White (2004)
    auto block size selection — paired ΔSharpe 95% percentile CI.

    Per spec §3.1 (primary inference statistic, 1000 resamples).

    Args:
        returns_a, returns_b: paired monthly returns Series; common index used.
        n_resamples: bootstrap iterations; spec §3.1 locks 1000.
        alpha: 1 − confidence level; spec locks 0.05 → 95% CI.
        obs_per_year: 12 (monthly).

    Returns:
        (ci_lower, ci_upper, block_size_used).
    """
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    a = returns_a.dropna()
    b = returns_b.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 12:
        return float("nan"), float("nan"), 0
    a = a.loc[common].astype(float)
    b = b.loc[common].astype(float)

    paired = np.column_stack([a.values, b.values])

    # Politis-White automatic block length on the *paired-difference* series
    diff_series = a.values - b.values
    try:
        opt = optimal_block_length(diff_series)
        # arch returns DataFrame with rows for series; column 'stationary' is for SB
        block = max(1, int(np.ceil(float(opt["stationary"].iloc[0]))))
    except Exception as exc:
        logger.warning("optimal_block_length failed (%s); falling back to block=12", exc)
        block = 12

    rng = np.random.default_rng(random_state)
    sb = StationaryBootstrap(block, paired, seed=int(rng.integers(0, 2**31 - 1)))

    diffs: list[float] = []
    sqrt_freq = float(np.sqrt(obs_per_year))
    for data, _ in sb.bootstrap(n_resamples):
        sample = data[0]
        if sample.shape[0] < 6:
            continue
        ra = sample[:, 0]
        rb = sample[:, 1]
        std_a = float(ra.std(ddof=1))
        std_b = float(rb.std(ddof=1))
        if std_a <= 0 or std_b <= 0:
            continue
        sa = (float(ra.mean()) / std_a) * sqrt_freq
        sb_val = (float(rb.mean()) / std_b) * sqrt_freq
        diffs.append(sa - sb_val)

    if not diffs:
        return float("nan"), float("nan"), block
    arr = np.array(diffs)
    lower = float(np.percentile(arr, alpha / 2.0 * 100.0))
    upper = float(np.percentile(arr, (1.0 - alpha / 2.0) * 100.0))
    return lower, upper, int(block)


def descriptive_achieved_power(
    rho:          float,
    delta_annual: float = _DELTA_SHARPE_PASS_THRESHOLD,
    n_years:      float = 6.0,
    alpha:        float = 0.05,
) -> float:
    """Descriptive achieved power at the observed ρ̂ + spec δ.

    Per spec §3.3 honesty disclaimer: this is for verdict diagnostics, not
    decision input. Power = P(|Z| > z_α/2 | true δ_annual = δ).
    Memmel paired V_annual ≈ 2(1−ρ²); years_required = z² · V / δ².
    """
    from scipy.stats import norm
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    if not np.isfinite(rho):
        return float("nan")
    V = 2.0 * (1.0 - rho ** 2)
    if V <= 0:
        return 1.0
    se = float(np.sqrt(V / n_years))
    z_at_delta = delta_annual / se
    # Power for two-sided test: P(|Z| > z_alpha) ≈ Φ(z_at_delta − z_alpha) + Φ(−z_at_delta − z_alpha)
    return float(norm.cdf(z_at_delta - z_alpha) + norm.cdf(-z_at_delta - z_alpha))


# ─────────────────────────────────────────────────────────────────────────────
# Decision rule (spec §3.2 locked)
# ─────────────────────────────────────────────────────────────────────────────

def apply_decision_rule(
    delta_sharpe:       float,
    bootstrap_ci_lower: float,
    bootstrap_ci_upper: float,
    fallback_rate:      float,
) -> str:
    """Spec §3.2 + §3.4 locked decision rule.

    Returns one of:
        UNINTERPRETABLE                  — fallback rate ≥ 50%
        PASS                             — ΔŜ ≥ 0.10 AND CI lower > 0
        MARGINAL_INSUFFICIENT_PRECISION  — ΔŜ ≥ 0.10 BUT CI crosses zero
        MARGINAL                         — 0 ≤ ΔŜ < 0.10
        FAIL                             — ΔŜ < 0
    """
    if not np.isfinite(fallback_rate):
        fallback_rate = 0.0
    if fallback_rate >= _FALLBACK_UNINTERPRETABLE:
        return "UNINTERPRETABLE"

    if not np.isfinite(delta_sharpe):
        return "FAIL"

    if delta_sharpe >= _DELTA_SHARPE_PASS_THRESHOLD:
        if np.isfinite(bootstrap_ci_lower) and bootstrap_ci_lower > 0:
            return "PASS"
        return "MARGINAL_INSUFFICIENT_PRECISION"
    if delta_sharpe >= 0:
        return "MARGINAL"
    return "FAIL"


def compute_verdict(
    overlay_returns_multivariate: pd.Series,
    overlay_returns_univariate:   pd.Series,
    fallback_rate:                float,
    n_resamples:                  int = 1000,
    alpha:                        float = 0.05,
    random_state:                 int = 42,
) -> MultivariateVerdict:
    """Top-level verdict computation per spec §3.1 + §3.2 + §3.4.

    Args:
        overlay_returns_multivariate: monthly returns of multivariate overlay strategy on OOS.
        overlay_returns_univariate:   same for univariate baseline overlay.
        fallback_rate:                fraction of OOS months where multivariate fell back to
            univariate (computed by walk-forward harness).
        n_resamples / alpha / random_state: bootstrap configuration.

    Returns:
        MultivariateVerdict snapshot ready to write into verdict file template.
    """
    a = overlay_returns_multivariate.dropna()
    b = overlay_returns_univariate.dropna()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]

    s_multi = annualized_sharpe(a)
    s_uni   = annualized_sharpe(b)
    delta   = s_multi - s_uni if (np.isfinite(s_multi) and np.isfinite(s_uni)) else float("nan")

    z, rho, _V = memmel_z_paired_sharpe_diff(a, b)
    ci_lo, ci_up, block = bootstrap_sharpe_diff_ci(
        a, b, n_resamples=n_resamples, alpha=alpha, random_state=random_state,
    )
    power = descriptive_achieved_power(rho, n_years=len(common) / 12.0)

    decision = apply_decision_rule(delta, ci_lo, ci_up, fallback_rate)

    return MultivariateVerdict(
        delta_sharpe               = float(delta),
        sharpe_multivariate        = float(s_multi),
        sharpe_univariate          = float(s_uni),
        bootstrap_ci_lower         = float(ci_lo),
        bootstrap_ci_upper         = float(ci_up),
        bootstrap_n_resamples      = int(n_resamples),
        bootstrap_block_size       = int(block),
        memmel_z                   = float(z),
        paired_correlation         = float(rho),
        fallback_rate              = float(fallback_rate),
        n_oos_months               = int(len(common)),
        decision                   = decision,
        achieved_power_descriptive = float(power),
    )



# ─────────────────────────────────────────────────────────────────────────────
# v2 additions (spec_multivariate_msm_v2.md): ternary overlay + descriptive-only verdict
# ─────────────────────────────────────────────────────────────────────────────

# v2 hysteresis band thresholds (spec_v2 §2.5; locked, mirror engine/regime.py)
_V2_OVERLAY_UPPER = 0.55
_V2_OVERLAY_LOWER = 0.45


def compute_overlay_returns_ternary(
    p_risk_on:    pd.Series,
    base_returns: pd.Series,
    upper:        float = _V2_OVERLAY_UPPER,
    lower:        float = _V2_OVERLAY_LOWER,
) -> pd.Series:
    """Ternary overlay with hysteresis (spec_v2 §2.5; D3 fix).

    position_t = +1   if p_risk_on_t > upper
                 -1   if p_risk_on_t < lower
                  0   otherwise (transition / low conviction)

    Replaces v1 binary 2*p-1 mapping which treats transition months as full short.

    Args:
        p_risk_on: filtered MSM prob of risk-on regime, monthly indexed.
        base_returns: monthly base returns (typically SPY).
        upper / lower: hysteresis cutoffs; spec_v2 locks 0.55 / 0.45.

    Returns:
        pd.Series of overlay returns indexed by intersection.
    """
    if not (0.0 < lower < upper < 1.0):
        raise ValueError(
            f"hysteresis band invalid: lower={lower}, upper={upper}; "
            f"must satisfy 0 < lower < upper < 1"
        )
    common = p_risk_on.index.intersection(base_returns.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    p_aligned = p_risk_on.loc[common]
    b_aligned = base_returns.loc[common]
    position = pd.Series(0.0, index=common)
    position[p_aligned > upper] = +1.0
    position[p_aligned < lower] = -1.0
    return position * b_aligned


@dataclass(frozen=True)
class MultivariateVerdictV2:
    """v2 verdict snapshot per spec_v2 §3.2 — DESCRIPTIVE-only labels (no PASS gate)."""
    delta_sharpe:               float
    sharpe_multivariate:        float
    sharpe_univariate:          float
    bootstrap_ci_lower:         float
    bootstrap_ci_upper:         float
    bootstrap_n_resamples:      int
    bootstrap_block_size:       int
    memmel_z:                   float
    paired_correlation:         float
    fallback_rate:              float
    n_oos_months:               int
    achieved_power_descriptive: float
    # v2 descriptive labels (spec_v2 §3.2)
    ci_lower_above_zero:        bool        # observable: is bootstrap CI lower bound > 0
    ci_lower_above_threshold:   bool        # observable: is CI lower bound ≥ +0.05 (ship-suggesting heuristic)
    decision_label:             str         # DESCRIPTIVE_POSITIVE / DESCRIPTIVE_INSUFFICIENT / DESCRIPTIVE_NEGATIVE / UNINTERPRETABLE


_V2_DESCRIPTIVE_SHIP_THRESHOLD = 0.05   # spec_v2 §3.2: CI lower > +0.05 = ship-suggesting heuristic


def compute_verdict_v2(
    overlay_returns_multivariate: pd.Series,
    overlay_returns_univariate:   pd.Series,
    fallback_rate:                float,
    n_resamples:                  int = 1000,
    alpha:                        float = 0.05,
    random_state:                 int = 42,
) -> MultivariateVerdictV2:
    """v2 verdict per spec_v2 §3.2 — describe-and-let-supervisor-decide framework.

    No PASS/FAIL gate. Decision label is DESCRIPTIVE_*.

    DESCRIPTIVE_POSITIVE        ΔŜ ≥ +0.05 AND CI lower bound > 0  → supervisor may choose to ship
    DESCRIPTIVE_INSUFFICIENT    other non-negative cases           → not enough evidence to ship
    DESCRIPTIVE_NEGATIVE        ΔŜ < 0                             → 9th falsification chain entry
    UNINTERPRETABLE             fallback_rate ≥ 50%                → spec §3.4
    """
    a = overlay_returns_multivariate.dropna()
    b = overlay_returns_univariate.dropna()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]

    s_multi = annualized_sharpe(a)
    s_uni   = annualized_sharpe(b)
    delta   = s_multi - s_uni if (np.isfinite(s_multi) and np.isfinite(s_uni)) else float("nan")

    z, rho, _V = memmel_z_paired_sharpe_diff(a, b)
    ci_lo, ci_up, block = bootstrap_sharpe_diff_ci(
        a, b, n_resamples=n_resamples, alpha=alpha, random_state=random_state,
    )
    power = descriptive_achieved_power(rho, n_years=len(common) / 12.0)

    # Determine descriptive label
    if not np.isfinite(fallback_rate):
        fallback_rate = 0.0
    if fallback_rate >= 0.50:
        label = "UNINTERPRETABLE"
    elif not np.isfinite(delta) or delta < 0:
        label = "DESCRIPTIVE_NEGATIVE"
    elif delta >= _V2_DESCRIPTIVE_SHIP_THRESHOLD and np.isfinite(ci_lo) and ci_lo > 0:
        label = "DESCRIPTIVE_POSITIVE"
    else:
        label = "DESCRIPTIVE_INSUFFICIENT"

    ci_above_zero      = bool(np.isfinite(ci_lo) and ci_lo > 0.0)
    ci_above_threshold = bool(np.isfinite(ci_lo) and ci_lo >= _V2_DESCRIPTIVE_SHIP_THRESHOLD)

    return MultivariateVerdictV2(
        delta_sharpe               = float(delta),
        sharpe_multivariate        = float(s_multi),
        sharpe_univariate          = float(s_uni),
        bootstrap_ci_lower         = float(ci_lo),
        bootstrap_ci_upper         = float(ci_up),
        bootstrap_n_resamples      = int(n_resamples),
        bootstrap_block_size       = int(block),
        memmel_z                   = float(z),
        paired_correlation         = float(rho),
        fallback_rate              = float(fallback_rate),
        n_oos_months               = int(len(common)),
        achieved_power_descriptive = float(power),
        ci_lower_above_zero        = ci_above_zero,
        ci_lower_above_threshold   = ci_above_threshold,
        decision_label             = label,
    )


__all__ = [
    "MultivariateVerdict",
    "compute_overlay_returns",
    "annualized_sharpe",
    "memmel_z_paired_sharpe_diff",
    "bootstrap_sharpe_diff_ci",
    "descriptive_achieved_power",
    "apply_decision_rule",
    "compute_verdict",
    # v2 additions
    "MultivariateVerdictV2",
    "compute_overlay_returns_ternary",
    "compute_verdict_v2",
]
