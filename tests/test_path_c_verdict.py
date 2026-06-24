"""
tests/test_path_c_verdict.py — Sprint 5 verdict aggregation + decision tests.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §3 + §九
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.path_c import NW_LAG_TRADING_DAYS_LOCKED, TC_BPS_ROUNDTRIP_LOCKED
from engine.path_c.pead_backtest import WalkForwardPeadResult
from engine.path_c.verdict import (
    PASS_SHARPE_THRESHOLD, PASS_NW_T_THRESHOLD,
    MARGINAL_SHARPE_THRESHOLD, MARGINAL_NW_T_THRESHOLD,
    TRADING_DAYS_PER_YEAR, BOOTSTRAP_RESAMPLES_LOCKED,
    PeadVerdict,
    compute_annualized_sharpe,
    compute_nw_t_stat,
    compute_max_drawdown,
    compute_cumulative_return,
    compute_bhy_fdr_passes,
    nw_t_to_two_sided_p_value,
    classify_decision,
    build_pead_verdict,
    persist_verdict,
    HONEST_DISCLOSE_LOCKED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked decision-gate constants (spec §3.2 + §六)
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_thresholds_match_spec():
    assert PASS_SHARPE_THRESHOLD == 0.50          # spec §3.2
    assert PASS_NW_T_THRESHOLD == 2.00            # spec §3.2
    assert MARGINAL_SHARPE_THRESHOLD == 0.30      # spec §3.2
    assert MARGINAL_NW_T_THRESHOLD == 1.50        # spec §3.2
    assert TRADING_DAYS_PER_YEAR == 252           # daily PEAD frequency
    assert BOOTSTRAP_RESAMPLES_LOCKED == 1000     # spec §六


# ─────────────────────────────────────────────────────────────────────────────
# compute_annualized_sharpe
# ─────────────────────────────────────────────────────────────────────────────

def test_sharpe_zero_variance_returns_nan():
    """Exact-zero variance (e.g., all zeros) → NaN Sharpe."""
    r = pd.Series(np.zeros(250))
    assert np.isnan(compute_annualized_sharpe(r))


def test_sharpe_iid_normal_finite_positive():
    """IID normal returns with positive mean → finite positive Sharpe."""
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(loc=0.0005, scale=0.01, size=1000))
    sharpe = compute_annualized_sharpe(r)
    assert np.isfinite(sharpe)
    assert sharpe > 0


def test_sharpe_handles_nan_input():
    r = pd.Series([0.001, np.nan, 0.002])
    sharpe = compute_annualized_sharpe(r)
    assert np.isfinite(sharpe) or np.isnan(sharpe)  # both acceptable for 2-point series


def test_sharpe_too_short_returns_nan():
    assert np.isnan(compute_annualized_sharpe(pd.Series([])))
    assert np.isnan(compute_annualized_sharpe(pd.Series([0.001])))


# ─────────────────────────────────────────────────────────────────────────────
# compute_nw_t_stat
# ─────────────────────────────────────────────────────────────────────────────

def test_nw_t_lag_zero_equals_standard_t():
    """NW with lag=0 = (sample mean / SE), where SE = std_ddof0 / sqrt(T).
    Note: standard t-stat uses std_ddof1; NW lag=0 uses biased var (γ_0 = pop var)."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, size=500))
    nw_t = compute_nw_t_stat(r, lag=0)
    # Manual: γ_0 / T denominator, where γ_0 is pop variance
    xbar = r.mean()
    dev = r - xbar
    gamma_0 = (dev ** 2).sum() / len(r)
    expected_t = xbar / np.sqrt(gamma_0 / len(r))
    assert nw_t == pytest.approx(expected_t, rel=1e-6)


def test_nw_t_lag_60_finite_for_realistic_series():
    """NW lag=60 on a 1000-obs series returns finite t."""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.005, size=1000))
    nw_t = compute_nw_t_stat(r, lag=60)
    assert np.isfinite(nw_t)


def test_nw_t_uses_locked_lag_by_default():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.005, size=500))
    default = compute_nw_t_stat(r)
    explicit = compute_nw_t_stat(r, lag=NW_LAG_TRADING_DAYS_LOCKED)
    assert default == pytest.approx(explicit)


def test_nw_t_empty_returns_nan():
    assert np.isnan(compute_nw_t_stat(pd.Series([])))


def test_nw_t_zero_variance_returns_nan():
    """Exact-zero series → γ_0 = 0 → SE = 0 → NaN t."""
    r = pd.Series(np.zeros(100))
    assert np.isnan(compute_nw_t_stat(r, lag=10))


def test_nw_t_negative_lag_returns_nan():
    """Negative lag is invalid."""
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, size=100))
    assert np.isnan(compute_nw_t_stat(r, lag=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Max drawdown + cumulative return
# ─────────────────────────────────────────────────────────────────────────────

def test_max_drawdown_no_loss_zero():
    """Monotone-increasing NAV → 0 drawdown."""
    r = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert compute_max_drawdown(r) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known_case():
    """Returns +10%, -20%, +5% → NAV 1.0 → 1.1 → 0.88 → 0.924, peak=1.1, trough=0.88, DD=-0.20."""
    r = pd.Series([0.10, -0.20, 0.05])
    dd = compute_max_drawdown(r)
    assert dd == pytest.approx(-0.20, abs=1e-9)


def test_cumulative_return_known_case():
    r = pd.Series([0.10, -0.20, 0.05])
    # (1.1 × 0.8 × 1.05) - 1 = 0.924 - 1 = -0.076
    assert compute_cumulative_return(r) == pytest.approx(-0.076, abs=1e-9)


def test_max_drawdown_empty_returns_nan():
    assert np.isnan(compute_max_drawdown(pd.Series([])))


# ─────────────────────────────────────────────────────────────────────────────
# BHY-FDR conservative bound
# ─────────────────────────────────────────────────────────────────────────────

def test_bhy_fdr_n1_equals_raw_alpha():
    """For n_trials=1, BHY threshold = α (H_1 = 1)."""
    # threshold = 0.05 / (1 × 1) = 0.05
    assert compute_bhy_fdr_passes(p_value=0.04, effective_n_trials=1, alpha=0.05) is True
    assert compute_bhy_fdr_passes(p_value=0.06, effective_n_trials=1, alpha=0.05) is False


def test_bhy_fdr_n21_stricter_than_raw_alpha():
    """For n=21 (project current), BHY threshold ≈ 0.05 / (21 × 3.65) ≈ 0.00065."""
    # Raw α=0.05 would pass, BHY does not
    assert compute_bhy_fdr_passes(p_value=0.04, effective_n_trials=21) is False
    # p = 0.0005 should pass BHY
    assert compute_bhy_fdr_passes(p_value=0.0005, effective_n_trials=21) is True


def test_bhy_fdr_zero_trials_returns_false():
    """n=0 is degenerate; safe default = False."""
    assert compute_bhy_fdr_passes(p_value=0.001, effective_n_trials=0) is False


def test_bhy_fdr_invalid_pvalue_returns_false():
    assert compute_bhy_fdr_passes(p_value=-0.01, effective_n_trials=5) is False
    assert compute_bhy_fdr_passes(p_value=1.5, effective_n_trials=5) is False


def test_nw_t_to_p_value_large_t_tiny_p():
    """|t| = 4 → p ≈ 6.3e-5."""
    p = nw_t_to_two_sided_p_value(4.0)
    assert p < 1e-4


def test_nw_t_to_p_value_zero_t_one():
    """t = 0 → p = 1."""
    p = nw_t_to_two_sided_p_value(0.0)
    assert p == pytest.approx(1.0)


def test_nw_t_to_p_value_nan_returns_nan():
    assert np.isnan(nw_t_to_two_sided_p_value(float("nan")))


# ─────────────────────────────────────────────────────────────────────────────
# classify_decision (spec §3.3 logic)
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_pass_high_gates_only():
    """BHY demoted 2026-05-12: PASS = Sharpe≥0.5 + NW t≥2.0 (industry-grade)."""
    assert classify_decision(sharpe_net=0.55, nw_t=2.10) == "PASS"


def test_classify_pass_when_bhy_irrelevant():
    """BHY demoted: PASS unchanged whether bhy_passes is True or False."""
    assert classify_decision(sharpe_net=0.55, nw_t=2.10, bhy_passes=False) == "PASS"
    assert classify_decision(sharpe_net=0.55, nw_t=2.10, bhy_passes=True) == "PASS"


def test_classify_marginal_below_pass_threshold():
    """Sharpe in [0.3, 0.5) + t in [1.5, 2.0) → MARGINAL regardless of BHY."""
    assert classify_decision(sharpe_net=0.40, nw_t=1.80, bhy_passes=False) == "MARGINAL"


def test_classify_low_threshold_with_bhy_stays_marginal():
    """BHY demoted 2026-05-12: low-threshold gates stay MARGINAL even if BHY passes
    (BHY no longer participates in gating logic)."""
    assert classify_decision(sharpe_net=0.40, nw_t=1.80, bhy_passes=True) == "MARGINAL"


def test_classify_fail_sharpe_too_low():
    assert classify_decision(sharpe_net=0.25, nw_t=2.50) == "FAIL"


def test_classify_fail_nw_t_too_low():
    assert classify_decision(sharpe_net=0.55, nw_t=1.20) == "FAIL"


def test_classify_fail_nan_inputs():
    assert classify_decision(sharpe_net=float("nan"), nw_t=2.0) == "FAIL"
    assert classify_decision(sharpe_net=0.5, nw_t=float("nan")) == "FAIL"


def test_classify_at_exact_boundary_pass():
    """Boundary: Sharpe = 0.50 exact + t = 2.00 exact → PASS (BHY irrelevant)."""
    assert classify_decision(sharpe_net=0.50, nw_t=2.00) == "PASS"


def test_classify_at_marginal_boundary():
    """Boundary: Sharpe = 0.30 + t = 1.50 → MARGINAL."""
    assert classify_decision(sharpe_net=0.30, nw_t=1.50, bhy_passes=False) == "MARGINAL"
    assert classify_decision(sharpe_net=0.30, nw_t=1.50, bhy_passes=True) == "MARGINAL"


# ─────────────────────────────────────────────────────────────────────────────
# build_pead_verdict end-to-end (uses Sprint 4 result + Sprint 5 stats)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_wf_result(
    n_days: int = 500,
    mean: float = 0.0001,
    std: float = 0.005,
    seed: int = 42,
) -> WalkForwardPeadResult:
    """Build a minimal WalkForwardPeadResult with synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2014-01-01", periods=n_days)
    rng_long  = rng.normal(loc=mean, scale=std, size=n_days)
    rng_short = rng.normal(loc=-mean, scale=std, size=n_days)
    daily = pd.DataFrame({
        "r_long":           rng_long,
        "r_short":          rng_short,
        "r_long_short":     rng_long - rng_short,
        "n_long":           [10] * n_days,
        "n_short":          [10] * n_days,
    }, index=pd.DatetimeIndex(dates, name="date"))
    daily["r_long_short_net"] = daily["r_long_short"] - 0.00001  # tiny drag
    return WalkForwardPeadResult(
        daily_returns=daily,
        n_quarters_processed=40,
        n_firm_quarters_active=800,
        annual_turnover_estimate=4.2,
        tc_bps_roundtrip=TC_BPS_ROUNDTRIP_LOCKED,
        window_start=datetime.date(2014, 1, 1),
        window_end=datetime.date(2023, 12, 31),
    )


def test_build_verdict_schema_complete():
    """Verdict dataclass has all spec §九 fields populated."""
    wf = _make_synthetic_wf_result()
    signal = pd.DataFrame({"leg": ["long"] * 800, "fiscal_yearq": ["2014Q1"] * 800})
    v = build_pead_verdict(
        wf, signal,
        spec_hash="abcd1234",
        effective_n_trials=21,
    )
    assert isinstance(v, PeadVerdict)
    assert v.decision in {"PASS", "MARGINAL", "FAIL"}
    assert v.spec_hash == "abcd1234"
    assert v.wave == "C1"   # default for PEAD (id=57)
    assert v.universe_source == "crsp_vintage_top200"
    assert v.nw_lag == NW_LAG_TRADING_DAYS_LOCKED
    assert v.tc_bps_roundtrip == TC_BPS_ROUNDTRIP_LOCKED
    assert v.effective_n_trials_at_verdict == 21
    assert len(v.honest_disclose) == len(HONEST_DISCLOSE_LOCKED)
    assert v.n_daily_observations == 500


def test_build_verdict_accepts_custom_wave_and_universe():
    """Path C Labor (id=58) overrides wave='C-labor' + universe_source."""
    wf = _make_synthetic_wf_result()
    signal = pd.DataFrame({"leg": ["long"] * 800, "fiscal_yearq": ["2014Q1"] * 800})
    v = build_pead_verdict(
        wf, signal,
        spec_hash="labor1234",
        effective_n_trials=22,
        wave="C-labor",
        universe_source="crsp_vintage_top200_revelio_linked",
    )
    assert v.wave == "C-labor"
    assert v.universe_source == "crsp_vintage_top200_revelio_linked"
    # Other locked fields independent of wave/universe overrides
    assert v.effective_n_trials_at_verdict == 22


def test_build_verdict_with_empty_walk_forward():
    """Empty walk-forward result → FAIL verdict with NaN stats."""
    wf = WalkForwardPeadResult(
        daily_returns=pd.DataFrame(),
        n_quarters_processed=0,
        n_firm_quarters_active=0,
        annual_turnover_estimate=0.0,
        tc_bps_roundtrip=TC_BPS_ROUNDTRIP_LOCKED,
        window_start=datetime.date(2014, 1, 1),
        window_end=datetime.date(2023, 12, 31),
    )
    v = build_pead_verdict(
        wf, pd.DataFrame(),
        spec_hash="empty_hash",
        effective_n_trials=21,
    )
    assert v.decision == "FAIL"
    assert v.n_daily_observations == 0


def test_build_verdict_decision_consistent_with_classifier():
    """Verdict.decision matches classify_decision(sharpe_net, nw_t, bhy_passes)."""
    wf = _make_synthetic_wf_result(n_days=300, mean=0.001, std=0.005, seed=7)
    signal = pd.DataFrame({"leg": ["long"] * 800, "fiscal_yearq": ["2014Q1"] * 800})
    v = build_pead_verdict(
        wf, signal,
        spec_hash="check_hash",
        effective_n_trials=21,
    )
    # Re-classify from the verdict's own stats
    expected = classify_decision(v.sharpe_net, v.nw_t_stat, v.bhy_fdr_passes)
    assert v.decision == expected


# ─────────────────────────────────────────────────────────────────────────────
# persist_verdict
# ─────────────────────────────────────────────────────────────────────────────

def test_persist_verdict_roundtrip(tmp_path):
    wf = _make_synthetic_wf_result()
    signal = pd.DataFrame({"leg": ["long"] * 800, "fiscal_yearq": ["2014Q1"] * 800})
    v = build_pead_verdict(
        wf, signal,
        spec_hash="rt_hash",
        effective_n_trials=21,
    )
    path = tmp_path / "v1_pead_10y_verdict.json"
    persist_verdict(v, path)
    assert path.exists()

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Spec §九 required fields
    required_fields = {
        "decision", "spec_hash", "spec_path", "run_at", "wave",
        "window_start", "window_end", "universe_source",
        "n_quarters", "n_firm_quarters_used", "n_firm_quarters_excluded",
        "n_daily_observations",
        "sharpe_gross", "sharpe_net", "nw_t_stat", "nw_lag",
        "bootstrap_ci_lower", "bootstrap_ci_upper",
        "bhy_fdr_passes", "effective_n_trials_at_verdict",
        "cumulative_return", "max_drawdown",
        "long_only_sharpe", "short_only_sharpe",
        "annual_turnover", "tc_bps_roundtrip", "tc_drag_annualized",
        "fallback_rate_per_quarter", "honest_disclose",
    }
    assert required_fields.issubset(set(data.keys()))


def test_persist_verdict_creates_parent_dirs(tmp_path):
    """persist_verdict mkdir -p on nested target."""
    wf = _make_synthetic_wf_result()
    signal = pd.DataFrame({"leg": ["long"] * 800, "fiscal_yearq": ["2014Q1"] * 800})
    v = build_pead_verdict(
        wf, signal,
        spec_hash="dir_hash",
        effective_n_trials=21,
    )
    path = tmp_path / "subdir1" / "subdir2" / "v1.json"
    persist_verdict(v, path)
    assert path.exists()
