"""
engine/factor_lab/power.py — Pre-test power analysis for Sharpe ratio difference.

Spec: docs/spec_factor_lab.md §3 (power analysis)
Boundary invariant: zero LLM imports — pure deterministic math.

Academic anchors
----------------
1. Lo (2002) "The Statistics of Sharpe Ratios", FAJ 58(4):36-52.
   For i.i.d. returns, Var(Ŝ_period) ≈ (1 + 0.5·S_period²) / T  (S and T at
   the same frequency).

2. Memmel (2003) "Performance Hypothesis Testing with the Sharpe Ratio",
   Finance Letters 1:21-23. Paired-strategy correction; this module uses
   the conservative *independent-samples* upper bound (yields larger n_req
   than Memmel paired) to avoid false REGISTERED.

3. Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
   2nd ed., Lawrence Erlbaum. Generic two-sample power formula:
       n_req = ((z_{α/2} + z_β)² · (Var_A + Var_B)) / δ²

Frequency convention (CRITICAL — read before using)
----------------------------------------------------
**User inputs**:  ANNUALIZED Sharpe ratios (project + industry convention).
                  baseline_sharpe = 1.0 means "Sharpe of 1 per year".
                  expected_sharpe_lift = 0.5 means "+0.5 annual Sharpe".

**Function returns**:  number of OBSERVATION PERIODS required.
                       Default observations_per_year=12 → returns months.
                       Pass observations_per_year=252 → returns trading days.
                       Pass observations_per_year=52  → returns weeks.

**Internal math**:  We convert annualized → per-period inside the function:
    S_period = S_annual / sqrt(observations_per_year)
    δ_period = δ_annual / sqrt(observations_per_year)
Then apply Lo's formula at the per-period frequency. This avoids the
classic frequency-mixing error (using annualized S in a per-period variance
formula yields meaningless n_req values).

Why this module exists
----------------------
P3c COT-conditional BAB pre-reg test (2026-05-07) FAILED with verdict
UNDERPOWERED — n_extreme=18, BHY-p=0.43. The system spent compute on a
test that *could not have produced FDR-passing evidence even under the
strongest hypothesized lift*. This module makes that detection happen at
register time, before the BHY runner is invoked.

Worked example — P3c retrospective:
    lift_annual = +1.38, baseline_annual = 1.0, n_available = 18 months
    → S_A_monthly = 0.289, δ_monthly = 0.398, S_B_monthly = 0.687
    → Var_A = 1.042, Var_B = 1.236
    → n_req = (1.96+0.84)² · (1.042+1.236) / 0.398² ≈ 113 months
    → 18 vs 113 = 6.3× shortfall, achieved power ≈ 0.20
    → BLOCKED with quantitative reason
"""
from __future__ import annotations

import math

from scipy import stats  # noqa — deterministic stats only, not LLM

from engine.factor_lab.types import FactorState, PowerCheckResult


# ── Spec-locked thresholds (per docs/spec_factor_lab.md §3.2) ────────────────
# These are statistical convention values from Cohen 1988 / standard practice.
# Changing them requires amend_spec(kind='threshold_tweak') + spec_factor_lab.md
# update — Tier R rule_spec_hash_vs_code_drift will catch silent edits.
_DEFAULT_TARGET_POWER       = 0.80    # Cohen 1988 convention; β = 0.20
_DEFAULT_TARGET_ALPHA       = 0.05    # two-sided
_DEFAULT_OBS_PER_YEAR       = 12      # monthly (project default frequency)

# Input validation bounds (defense against degenerate / p-hacking calls)
_MIN_POWER, _MAX_POWER = 0.50, 0.99
_MIN_ALPHA, _MAX_ALPHA = 0.001, 0.20
_MIN_OBS_PER_YEAR      = 4    # quarterly is the coarsest reasonable frequency
_MAX_OBS_PER_YEAR      = 365  # daily upper bound


def _validate_inputs(
    expected_sharpe_lift: float,
    baseline_sharpe:      float,
    target_power:         float,
    target_alpha:         float,
    observations_per_year: int,
) -> None:
    """Shared input validation (raises ValueError on degenerate inputs)."""
    if expected_sharpe_lift <= 0:
        raise ValueError(
            f"expected_sharpe_lift must be > 0 (H1 is directional, testing "
            f"for improvement); got {expected_sharpe_lift}. Negative lift "
            f"is not a meaningful research hypothesis for factor swap."
        )
    if not math.isfinite(baseline_sharpe):
        raise ValueError(f"baseline_sharpe must be finite; got {baseline_sharpe}")
    if not (_MIN_POWER < target_power < _MAX_POWER):
        raise ValueError(
            f"target_power must be in ({_MIN_POWER}, {_MAX_POWER}); "
            f"got {target_power}. Cohen 1988 convention is 0.80; values "
            f"≤ 0.50 indicate degenerate test, ≥ 0.99 indicates n→∞."
        )
    if not (_MIN_ALPHA < target_alpha < _MAX_ALPHA):
        raise ValueError(
            f"target_alpha must be in ({_MIN_ALPHA}, {_MAX_ALPHA}); "
            f"got {target_alpha}. Standard convention is 0.05."
        )
    if not (_MIN_OBS_PER_YEAR <= observations_per_year <= _MAX_OBS_PER_YEAR):
        raise ValueError(
            f"observations_per_year must be in "
            f"[{_MIN_OBS_PER_YEAR}, {_MAX_OBS_PER_YEAR}]; got "
            f"{observations_per_year}. Typical values: 12 (monthly), "
            f"52 (weekly), 252 (trading-day daily)."
        )


def required_sample_size_sharpe_diff(
    expected_sharpe_lift:  float,
    baseline_sharpe:       float,
    target_power:          float = _DEFAULT_TARGET_POWER,
    target_alpha:          float = _DEFAULT_TARGET_ALPHA,
    observations_per_year: int   = _DEFAULT_OBS_PER_YEAR,
) -> int:
    """Compute minimum observation periods to detect a Sharpe lift δ.

    Method: Lo (2002) per-period variance + Cohen (1988) two-sample power
    formula with independent-samples upper bound (Memmel 2003 conservative).

    Args:
        expected_sharpe_lift: ANNUALIZED δ = candidate Sharpe − baseline.
            Must be strictly positive (H1 is directional).
        baseline_sharpe: ANNUALIZED current production_signal Sharpe.
        target_power: 1 − β. Default 0.80 per Cohen convention.
        target_alpha: Type I error rate (two-sided). Default 0.05.
        observations_per_year: Sampling frequency. 12 (monthly), 52
            (weekly), 252 (daily). Determines the unit of the return value.

    Returns:
        Minimum observation periods (months by default). Always rounded up.

    Raises:
        ValueError: any input out of statistical convention bounds.
    """
    _validate_inputs(
        expected_sharpe_lift, baseline_sharpe,
        target_power, target_alpha, observations_per_year,
    )

    # ── Annualized → per-period conversion (Lo 2002 frequency-consistency) ──
    # S_period = S_annual / sqrt(observations_per_year)
    sqrt_freq    = math.sqrt(observations_per_year)
    s_a_period   = baseline_sharpe / sqrt_freq
    delta_period = expected_sharpe_lift / sqrt_freq
    s_b_period   = s_a_period + delta_period

    # ── Lo 2002 per-period variance ──────────────────────────────────────────
    # Var(Ŝ_period) per unit time ≈ 1 + 0.5·S_period²
    var_a = 1.0 + 0.5 * s_a_period * s_a_period
    var_b = 1.0 + 0.5 * s_b_period * s_b_period

    # ── Cohen 1988 two-sample power formula ──────────────────────────────────
    z_alpha = stats.norm.ppf(1.0 - target_alpha / 2.0)
    z_beta  = stats.norm.ppf(target_power)

    n_req = ((z_alpha + z_beta) ** 2 * (var_a + var_b)) / (delta_period ** 2)
    return int(math.ceil(n_req))


def achieved_power_at_n(
    expected_sharpe_lift:  float,
    baseline_sharpe:       float,
    n_available:           int,
    target_alpha:          float = _DEFAULT_TARGET_ALPHA,
    observations_per_year: int   = _DEFAULT_OBS_PER_YEAR,
) -> float:
    """Inverse: given fixed n periods, return achievable power at lift δ.

    n_available is in OBSERVATION PERIODS (matches observations_per_year unit).
    Returns power in [0, 1].
    """
    if n_available <= 0:
        return 0.0
    if expected_sharpe_lift <= 0 or not math.isfinite(baseline_sharpe):
        return 0.0
    if not (_MIN_ALPHA < target_alpha < _MAX_ALPHA):
        return 0.0
    if not (_MIN_OBS_PER_YEAR <= observations_per_year <= _MAX_OBS_PER_YEAR):
        return 0.0

    sqrt_freq    = math.sqrt(observations_per_year)
    s_a_period   = baseline_sharpe / sqrt_freq
    delta_period = expected_sharpe_lift / sqrt_freq
    s_b_period   = s_a_period + delta_period
    var_a        = 1.0 + 0.5 * s_a_period * s_a_period
    var_b        = 1.0 + 0.5 * s_b_period * s_b_period

    z_alpha = stats.norm.ppf(1.0 - target_alpha / 2.0)
    # Solve for power given n: rearrange Cohen formula
    # δ_period = (z_α/2 + z_β) · sqrt((Var_A + Var_B) / n)
    # ⇒ z_β = δ_period · sqrt(n / (Var_A + Var_B)) − z_α/2
    z_beta = delta_period * math.sqrt(n_available / (var_a + var_b)) - z_alpha
    return float(stats.norm.cdf(z_beta))


def power_check(
    *,
    expected_sharpe_lift:  float,
    baseline_sharpe:       float,
    n_available:           int,
    target_power:          float = _DEFAULT_TARGET_POWER,
    target_alpha:          float = _DEFAULT_TARGET_ALPHA,
    observations_per_year: int   = _DEFAULT_OBS_PER_YEAR,
) -> PowerCheckResult:
    """Decide whether candidate has sufficient sample size to register.

    Returns PowerCheckResult with decision ∈ {REGISTERED, BLOCKED_UNDERPOWERED}.
    Caller is responsible for executing the state transition + writing
    spec_amendment_log entry.

    Spec ref: docs/spec_factor_lab.md §3.3.
    """
    n_req = required_sample_size_sharpe_diff(
        expected_sharpe_lift  = expected_sharpe_lift,
        baseline_sharpe       = baseline_sharpe,
        target_power          = target_power,
        target_alpha          = target_alpha,
        observations_per_year = observations_per_year,
    )
    achieved = achieved_power_at_n(
        expected_sharpe_lift  = expected_sharpe_lift,
        baseline_sharpe       = baseline_sharpe,
        n_available           = n_available,
        target_alpha          = target_alpha,
        observations_per_year = observations_per_year,
    )

    if n_available >= n_req:
        return PowerCheckResult(
            decision                      = FactorState.REGISTERED,
            n_required                    = n_req,
            n_available                   = n_available,
            achieved_power_at_n_available = achieved,
            expected_sharpe_lift          = expected_sharpe_lift,
            baseline_sharpe               = baseline_sharpe,
            target_power                  = target_power,
            target_alpha                  = target_alpha,
            reason                        = (
                f"n_available={n_available} ≥ n_required={n_req} periods "
                f"(obs/year={observations_per_year}) at "
                f"power={target_power:.0%} for annualized lift "
                f"={expected_sharpe_lift:+.2f}. "
                f"Achieved power at current n: {achieved:.2f}."
            ),
        )

    # BLOCKED — explain how short
    shortfall_factor = n_req / max(n_available, 1)
    # Solve for the min lift detectable at n_available + target_power (informative)
    # In annualized units (multiply per-period min lift by sqrt(obs_per_year))
    sqrt_freq = math.sqrt(observations_per_year)
    s_a_period = baseline_sharpe / sqrt_freq
    var_a_period = 1.0 + 0.5 * s_a_period * s_a_period
    z_alpha = stats.norm.ppf(1.0 - target_alpha / 2.0)
    z_beta  = stats.norm.ppf(target_power)
    # Var_B ≈ Var_A — fine for small δ; conservative-direction approximation
    min_lift_period   = (z_alpha + z_beta) * math.sqrt(2.0 * var_a_period / n_available)
    min_lift_annual   = min_lift_period * sqrt_freq
    return PowerCheckResult(
        decision                      = FactorState.BLOCKED_UNDERPOWERED,
        n_required                    = n_req,
        n_available                   = n_available,
        achieved_power_at_n_available = achieved,
        expected_sharpe_lift          = expected_sharpe_lift,
        baseline_sharpe               = baseline_sharpe,
        target_power                  = target_power,
        target_alpha                  = target_alpha,
        reason                        = (
            f"n_available={n_available} < n_required={n_req} periods "
            f"(obs/year={observations_per_year}, achieved power "
            f"{achieved:.2f}). Increase sample by {shortfall_factor:.1f}× "
            f"or reduce target lift from {expected_sharpe_lift:+.2f} "
            f"(annualized) to ≤{min_lift_annual:+.2f}."
        ),
    )
