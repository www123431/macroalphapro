"""tests/test_validation_alpha.py — Phase 1 alpha-validation harness tests.

Covers the deterministic math:
  - sharpe_ratio
  - probabilistic_sharpe_ratio (PSR) properties
  - expected_max_sharpe (multiple-testing benchmark) properties
  - deflated_sharpe_ratio (DSR <= PSR; verdict logic)
  - factor_attribution recovers known beta + alpha on synthetic data

Network-dependent Ken French fetch is NOT tested here (it's an external
data pull); factor_attribution is tested against a synthetic factor
frame so the regression math is verified deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# sharpe_ratio
# ──────────────────────────────────────────────────────────────────────────────
def test_sharpe_ratio_basic():
    from engine.validation.deflated_sharpe import sharpe_ratio
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    sr = sharpe_ratio(r)
    # mean/std ≈ 0.001/0.01 = 0.1, within sampling noise
    assert 0.0 < sr < 0.3


def test_sharpe_ratio_degenerate():
    from engine.validation.deflated_sharpe import sharpe_ratio
    assert np.isnan(sharpe_ratio([1.0]))           # too short
    assert np.isnan(sharpe_ratio([0.5, 0.5, 0.5])) # zero variance


# ──────────────────────────────────────────────────────────────────────────────
# probabilistic_sharpe_ratio
# ──────────────────────────────────────────────────────────────────────────────
def test_psr_high_for_strong_positive_series():
    from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio
    rng = np.random.default_rng(1)
    # Strong, long positive-Sharpe series → PSR vs 0 near 1
    r = rng.normal(0.002, 0.01, 500)
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert psr > 0.95


def test_psr_near_half_for_zero_edge():
    from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, 500)
    r = r - r.mean()                 # exact zero sample mean → SR_hat = 0
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert psr == pytest.approx(0.5, abs=0.02)   # exactly coin-flip at SR=0


def test_psr_monotonic_in_benchmark():
    """Higher benchmark Sharpe → lower PSR (harder to beat)."""
    from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio
    rng = np.random.default_rng(3)
    r = rng.normal(0.0015, 0.01, 500)
    psr_low  = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    psr_high = probabilistic_sharpe_ratio(r, sr_benchmark=0.10)
    assert psr_low > psr_high


# ──────────────────────────────────────────────────────────────────────────────
# expected_max_sharpe
# ──────────────────────────────────────────────────────────────────────────────
def test_expected_max_sharpe_increases_with_trials():
    from engine.validation.deflated_sharpe import expected_max_sharpe
    v = 0.04
    e5   = expected_max_sharpe(5, v)
    e35  = expected_max_sharpe(35, v)
    e100 = expected_max_sharpe(100, v)
    assert e5 < e35 < e100   # more trials → higher expected max under null


def test_expected_max_sharpe_zero_for_single_trial():
    from engine.validation.deflated_sharpe import expected_max_sharpe
    assert expected_max_sharpe(1, 0.04) == 0.0
    assert expected_max_sharpe(35, 0.0) == 0.0   # zero variance → zero


# ──────────────────────────────────────────────────────────────────────────────
# deflated_sharpe_ratio
# ──────────────────────────────────────────────────────────────────────────────
def test_dsr_never_exceeds_psr_vs_zero():
    """Deflation can only LOWER confidence: DSR <= PSR(vs 0)."""
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    rng = np.random.default_rng(4)
    r = rng.normal(0.0015, 0.01, 485)
    res = deflated_sharpe_ratio(r, n_trials=35)
    assert res.deflated_sr <= res.psr_vs_zero + 1e-9


def test_dsr_more_trials_lowers_confidence():
    """More research trials → harder to clear → lower DSR."""
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    rng = np.random.default_rng(5)
    r = rng.normal(0.0015, 0.01, 485)
    dsr_5   = deflated_sharpe_ratio(r, n_trials=5).deflated_sr
    dsr_100 = deflated_sharpe_ratio(r, n_trials=100).deflated_sr
    assert dsr_100 <= dsr_5


def test_dsr_verdict_thresholds():
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    # Very strong, long series with few trials → should PASS
    rng = np.random.default_rng(6)
    r = rng.normal(0.003, 0.008, 800)
    res = deflated_sharpe_ratio(r, n_trials=2)
    assert res.deflated_sr > 0.95
    assert "PASS" in res.verdict


def test_var_sr_from_trial_sharpes():
    from engine.validation.deflated_sharpe import var_sr_from_trial_sharpes
    sharpes = [0.1, 0.05, 0.15, 0.0, 0.2]
    v = var_sr_from_trial_sharpes(sharpes)
    assert v == pytest.approx(np.var(sharpes, ddof=1))


# ──────────────────────────────────────────────────────────────────────────────
# factor_attribution (synthetic, no network)
# ──────────────────────────────────────────────────────────────────────────────
def _synthetic_factors(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    df = pd.DataFrame({
        "Mkt-RF": rng.normal(0.001, 0.02, n),
        "SMB":    rng.normal(0.0, 0.01, n),
        "HML":    rng.normal(0.0, 0.01, n),
        "RMW":    rng.normal(0.0, 0.01, n),
        "CMA":    rng.normal(0.0, 0.01, n),
        "UMD":    rng.normal(0.0, 0.012, n),
        "RF":     np.full(n, 0.0001),
    }, index=idx)
    return df


def test_attribution_recovers_pure_factor_beta_zero_alpha():
    """Return = 1.5*Mkt + noise, no intercept → beta_Mkt≈1.5, alpha≈0,
    NO residual alpha verdict."""
    from engine.validation.factor_attribution import attribute_strategy
    f = _synthetic_factors()
    rng = np.random.default_rng(8)
    y = 1.5 * f["Mkt-RF"] + rng.normal(0.0, 0.003, len(f))
    res = attribute_strategy(y.rename("pure_beta"), f)
    assert abs(res.betas["Mkt-RF"] - 1.5) < 0.1
    assert abs(res.alpha_annual) < 0.03           # near zero
    assert "NO residual alpha" in res.verdict or abs(res.alpha_tstat) < 2.0


def test_attribution_detects_real_alpha():
    """Return = constant alpha + small factor + noise → significant
    positive intercept, RESIDUAL ALPHA verdict."""
    from engine.validation.factor_attribution import attribute_strategy
    f = _synthetic_factors()
    rng = np.random.default_rng(9)
    # 0.002/wk alpha ≈ 10%/yr, tiny factor loading, low noise → strong t
    y = 0.002 + 0.1 * f["Mkt-RF"] + rng.normal(0.0, 0.004, len(f))
    res = attribute_strategy(y.rename("real_alpha"), f)
    assert res.alpha_annual > 0.05
    assert abs(res.alpha_tstat) >= 2.0
    assert "RESIDUAL ALPHA" in res.verdict


def test_attribution_insufficient_overlap_is_undefined():
    from engine.validation.factor_attribution import attribute_strategy
    f = _synthetic_factors(n=10)
    y = f["Mkt-RF"] * 1.0
    res = attribute_strategy(y.rename("short"), f)
    assert "UNDEFINED" in res.verdict


# ──────────────────────────────────────────────────────────────────────────────
# rolling_sharpe / decay
# ──────────────────────────────────────────────────────────────────────────────
def test_decay_detects_dead_recent_window():
    """Strong early, dead-flat recent → DEAD/WEAK verdict."""
    from engine.validation.rolling_sharpe import decay_split
    rng = np.random.default_rng(11)
    early  = rng.normal(0.003, 0.01, 250)    # strong
    recent = rng.normal(-0.0005, 0.01, 250)  # negative
    s = pd.Series(np.concatenate([early, recent]),
                  index=pd.date_range("2015-01-02", periods=500, freq="W-FRI"),
                  name="decayer")
    res = decay_split(s, recent_weeks=156)
    assert res.recent_sharpe < res.first_half_sharpe
    assert "DEAD" in res.verdict or "WEAK" in res.verdict


def test_decay_alive_when_recent_strong():
    from engine.validation.rolling_sharpe import decay_split
    rng = np.random.default_rng(12)
    r = rng.normal(0.002, 0.01, 500)
    s = pd.Series(r, index=pd.date_range("2015-01-02", periods=500, freq="W-FRI"),
                  name="alive")
    res = decay_split(s, recent_weeks=156)
    assert "ALIVE" in res.verdict


def test_rolling_sharpe_window_length():
    from engine.validation.rolling_sharpe import rolling_sharpe
    rng = np.random.default_rng(13)
    s = pd.Series(rng.normal(0.001, 0.01, 300),
                  index=pd.date_range("2015-01-02", periods=300, freq="W-FRI"))
    rs = rolling_sharpe(s, window=104)
    # First 103 are NaN (incomplete window), rest defined
    assert rs.iloc[:103].isna().all()
    assert rs.iloc[104:].notna().all()


# ──────────────────────────────────────────────────────────────────────────────
# cost_stress
# ──────────────────────────────────────────────────────────────────────────────
def test_cost_stress_event_breakeven():
    """Break-even cost = gross_mean * 10000 / roundtrips. With mean 0.0069
    and 1 round-trip, break-even ≈ 69bp."""
    from engine.validation.cost_stress import cost_stress_event
    rng = np.random.default_rng(14)
    ev = rng.normal(0.0069, 0.0495, 238)
    res = cost_stress_event(ev, events_per_year=24.0, realistic_cost_bps=30.0)
    assert 50 < res.breakeven_cost_bps < 90   # near 69 modulo sampling
    # net Sharpe must decrease monotonically with cost
    sr_vals = [res.net_sharpe_at[c] for c in sorted(res.net_sharpe_at)]
    assert all(sr_vals[i] >= sr_vals[i+1] for i in range(len(sr_vals)-1))


def test_cost_stress_event_dies_at_high_cost():
    """A thin edge dies once cost exceeds break-even."""
    from engine.validation.cost_stress import cost_stress_event
    rng = np.random.default_rng(15)
    ev = rng.normal(0.0010, 0.02, 200)   # thin edge, ~10bp/event
    res = cost_stress_event(ev, events_per_year=24.0, realistic_cost_bps=30.0)
    # realistic 30bp > ~10bp break-even → net mean negative → not survive
    assert res.survives is False


def test_cost_stress_period_breakeven():
    from engine.validation.cost_stress import cost_stress_period
    rng = np.random.default_rng(16)
    r = rng.normal(0.0015, 0.01, 485)    # ~7.8%/yr gross
    res = cost_stress_period(r, annual_turnover=6.0, realistic_cost_bps=15.0)
    assert res.breakeven_cost_bps > 0
    assert 0 in res.net_sharpe_at
    # zero-cost net Sharpe == gross Sharpe
    assert res.net_sharpe_at[0] == pytest.approx(res.gross_ann_sharpe, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# aqr_factors — weekly→monthly + cached BAB parse (network-gated)
# ──────────────────────────────────────────────────────────────────────────────
def test_weekly_to_monthly_compounds():
    from engine.validation.aqr_factors import weekly_to_monthly
    # 4 weeks of +1% compound to ~ (1.01)^4 - 1 within the month
    idx = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
    s = pd.Series([0.01] * 8, index=idx, name="w")
    m = weekly_to_monthly(s)
    # January has ~4-5 Fridays; monthly return should be > single-week
    assert (m > 0.01).all()


def test_aqr_bab_parse_from_cache_if_present():
    """If the AQR xlsx is cached locally, the parser must extract a USA
    monthly BAB Series. Skips cleanly if neither cache nor network is
    available (offline CI)."""
    from pathlib import Path
    if not Path("data/cache/aqr_bab_monthly.xlsx").exists() \
            and not Path("data/cache/aqr_bab_usa_monthly.parquet").exists():
        pytest.skip("AQR BAB not cached and no network — skip")
    from engine.validation.aqr_factors import load_bab_usa_monthly
    s = load_bab_usa_monthly()
    assert s.name == "BAB"
    assert len(s) > 100              # decades of monthly data
    assert s.abs().max() < 1.0       # decimal monthly returns, sane range


# ──────────────────────────────────────────────────────────────────────────────
# diversification: effective bets + insurance contribution
# ──────────────────────────────────────────────────────────────────────────────
def test_effective_bets_uncorrelated_equals_n():
    """5 independent series → effective bets ≈ 5."""
    from engine.validation.diversification import effective_number_of_bets
    rng = np.random.default_rng(20)
    X = rng.normal(0, 1, (1000, 5))
    corr = np.corrcoef(X, rowvar=False)
    enb = effective_number_of_bets(corr)
    assert 4.5 < enb <= 5.0


def test_effective_bets_identical_equals_one():
    """5 identical series → effective bets ≈ 1."""
    from engine.validation.diversification import effective_number_of_bets
    rng = np.random.default_rng(21)
    base = rng.normal(0, 1, 1000)
    X = np.column_stack([base + rng.normal(0, 1e-6, 1000) for _ in range(5)])
    corr = np.corrcoef(X, rowvar=False)
    enb = effective_number_of_bets(corr)
    assert enb < 1.5


def test_analyze_diversification_reports_pair_and_verdict():
    from engine.validation.diversification import analyze_diversification
    rng = np.random.default_rng(22)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    df = pd.DataFrame({
        "A": rng.normal(0.001, 0.01, 300),
        "B": rng.normal(0.001, 0.01, 300),
        "C": rng.normal(0.001, 0.01, 300),
    }, index=idx)
    res = analyze_diversification(df)
    assert res.n_strategies == 3
    assert len(res.max_pair) == 3
    assert res.effective_bets > 0


def test_insurance_contribution_runs_and_classifies():
    """Insurance contribution must produce a verdict for each insurance
    sleeve present, comparing book with vs without it."""
    from engine.validation.diversification import (
        insurance_contribution, DEFAULT_BOOK_WEIGHTS,
    )
    rng = np.random.default_rng(23)
    idx = pd.date_range("2015-01-02", periods=400, freq="W-FRI")
    cols = list(DEFAULT_BOOK_WEIGHTS.keys())
    df = pd.DataFrame(
        {c: rng.normal(0.001, 0.012, 400) for c in cols}, index=idx,
    )
    out = insurance_contribution(df)
    # Both insurance sleeves present → both get a verdict
    assert "CTA_PQTIX" in out
    assert "AC_proxy_AB_2014_23" in out
    for c in out.values():
        assert isinstance(c.verdict, str) and len(c.verdict) > 0
        # full vs without are real numbers
        assert not np.isnan(c.full_sharpe)


def test_max_drawdown_known_path():
    """A series that goes +10% then -50% has MaxDD = -50%."""
    from engine.validation.diversification import _max_drawdown
    r = np.array([0.10, -0.50, 0.05])
    dd = _max_drawdown(r)
    assert dd == pytest.approx(-0.50, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# after_cost — gross → net conversion + cost lowers deflated Sharpe
# ──────────────────────────────────────────────────────────────────────────────
def test_apply_cost_subtracts_uniform_drag():
    from engine.validation.after_cost import apply_cost
    s = pd.Series([0.01] * 52)
    net = apply_cost(s, annual_drag=0.52, ppy=52)   # 1%/wk drag
    assert net.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_net_audit_cost_lowers_deflated_sr():
    """Net deflated Sharpe must be <= gross for any strategy with cost."""
    from engine.validation.after_cost import net_audit
    rng = np.random.default_rng(30)
    idx = pd.date_range("2015-01-02", periods=485, freq="W-FRI")
    df = pd.DataFrame({
        "K1_BAB":              rng.normal(0.0008, 0.007, 485),
        "D_PEAD":              rng.normal(0.0018, 0.014, 485),
        "PATH_N":              rng.normal(0.0026, 0.026, 485),
        "CTA_PQTIX":           rng.normal(0.0009, 0.015, 485),
        "AC_proxy_AB_2014_23": rng.normal(0.0008, 0.016, 485),
    }, index=idx)
    res = net_audit(df, n_trials=35)
    for name, r in res.items():
        # cost can only lower (or equal, for zero-cost CTA) the deflated SR
        assert r.net_deflated_sr_base <= r.gross_deflated_sr + 1e-9
        # high cost <= base cost
        assert r.net_deflated_sr_high <= r.net_deflated_sr_base + 1e-9


def test_net_audit_zero_cost_strategy_unchanged():
    """CTA (mutual fund, 0 round-trip bp) → net == gross deflated SR."""
    from engine.validation.after_cost import net_audit
    rng = np.random.default_rng(31)
    idx = pd.date_range("2015-01-02", periods=485, freq="W-FRI")
    df = pd.DataFrame({"CTA_PQTIX": rng.normal(0.001, 0.015, 485)}, index=idx)
    res = net_audit(df, n_trials=35)
    r = res["CTA_PQTIX"]
    assert r.net_deflated_sr_base == pytest.approx(r.gross_deflated_sr, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# dpead_tilt — long/short tilt conditioning + split-sample robustness
# ──────────────────────────────────────────────────────────────────────────────
def test_combined_tilt_construction():
    from engine.validation.dpead_tilt import combined
    idx = pd.date_range("2015-01-02", periods=10, freq="B")
    rl = pd.Series([0.02] * 10, index=idx)
    rs = pd.Series([0.01] * 10, index=idx)
    # w=1 → 0.02-0.01=0.01; w=0 → 0.02; w=0.5 → 0.015
    assert combined(rl, rs, 1.0).iloc[0] == pytest.approx(0.01)
    assert combined(rl, rs, 0.0).iloc[0] == pytest.approx(0.02)
    assert combined(rl, rs, 0.5).iloc[0] == pytest.approx(0.015)


def test_split_sample_detects_robust_improvement():
    """Construct a case where a low short_weight is genuinely better in
    BOTH halves → split-sample must call it ROBUST."""
    from engine.validation.dpead_tilt import split_sample_robustness
    rng = np.random.default_rng(40)
    n = 1000
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    # Long leg: clean positive edge. Short leg: pure noise (no edge) → the
    # full short leg (w=1) only ADDS variance, so lower w should win in
    # both halves.
    rl = pd.Series(rng.normal(0.0010, 0.010, n), index=idx)
    rs = pd.Series(rng.normal(0.0000, 0.012, n), index=idx)
    res = split_sample_robustness(rl, rs)
    # train-optimal should be a low short weight, and it should hold OOS
    assert res.train_optimal_w <= 0.5
    assert res.test_improvement > -0.05   # not worse OOS
    assert "ROBUST" in res.verdict or "MILD" in res.verdict


def test_split_sample_flags_overfit_when_no_real_edge():
    """If both legs are pure noise, no tilt should robustly beat baseline
    OOS — verdict should not falsely claim ROBUST with a big margin."""
    from engine.validation.dpead_tilt import split_sample_robustness
    rng = np.random.default_rng(41)
    n = 1000
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    rl = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    rs = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    res = split_sample_robustness(rl, rs)
    # OOS improvement should be small (no genuine structure to exploit)
    assert abs(res.test_improvement) < 0.6


def test_tilt_metric_recovers_beta(_synth=None):
    """tilt_metric must recover a known market beta on synthetic data."""
    from engine.validation.dpead_tilt import tilt_metric
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    mkt = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    rf  = pd.Series(np.full(n, 0.00001), index=idx)
    combo = (0.8 * mkt + rng.normal(0.0002, 0.003, n)).rename("c")
    tm = tilt_metric(combo, mkt, rf)
    assert abs(tm.market_beta - 0.8) < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# dpead_events — reconstruction self-validation guard
# ──────────────────────────────────────────────────────────────────────────────
def test_reconstruction_guard_flags_inverted_sign():
    """If long-high-SUE minus short-low-SUE CAR is NEGATIVE (contradicting
    the known-profitable production strategy), the guard must mark the
    reconstruction UNRELIABLE."""
    from engine.validation.dpead_events import validate_reconstruction
    rng = np.random.default_rng(50)
    n = 1000
    # Construct events where LOW sue has HIGHER car (inverted vs PEAD)
    sue = rng.normal(0, 1, n)
    car = -0.02 * sue + rng.normal(0, 0.05, n)   # negative slope = inverted
    ev = pd.DataFrame({"sue": sue, "car": car})
    chk = validate_reconstruction(ev, n_total_events=n)
    assert chk.sign_matches_production is False
    assert chk.reliable is False
    assert "BROKEN" in chk.note


def test_reconstruction_guard_passes_correct_sign_and_coverage():
    """Correct PEAD sign (high SUE → high CAR) + full coverage → reliable."""
    from engine.validation.dpead_events import validate_reconstruction
    rng = np.random.default_rng(51)
    n = 1000
    sue = rng.normal(0, 1, n)
    car = 0.02 * sue + rng.normal(0, 0.03, n)    # positive slope = correct PEAD
    ev = pd.DataFrame({"sue": sue, "car": car})
    chk = validate_reconstruction(ev, n_total_events=n)
    assert chk.sign_matches_production is True
    assert chk.reliable is True


def test_reconstruction_guard_flags_low_coverage():
    """Correct sign but tiny coverage → still not reliable."""
    from engine.validation.dpead_events import validate_reconstruction
    rng = np.random.default_rng(52)
    n = 200
    sue = rng.normal(0, 1, n)
    car = 0.02 * sue + rng.normal(0, 0.03, n)
    ev = pd.DataFrame({"sue": sue, "car": car})
    # claim 10000 total events → 2% coverage
    chk = validate_reconstruction(ev, n_total_events=10000)
    assert chk.reliable is False
    assert "COVERAGE" in chk.note.upper()


# ──────────────────────────────────────────────────────────────────────────────
# alpha_factory — universe-aware go/no-go gate
# ──────────────────────────────────────────────────────────────────────────────
def test_factory_green_for_strong_alpha():
    """A strong, low-trial, alive series with benchmark='none' (raw mean
    alpha) → GREEN."""
    from engine.validation.alpha_factory import CandidateSpec, screen_candidate
    rng = np.random.default_rng(60)
    idx = pd.date_range("2015-01-02", periods=500, freq="W-FRI")
    r = pd.Series(rng.normal(0.0030, 0.008, 500), index=idx)   # strong edge
    spec = CandidateSpec("strong", r, frequency="weekly", n_trials=2,
                         benchmark="none", cost_class="mutual_fund")
    v = screen_candidate(spec)
    assert v.light == "GREEN"
    assert v.net_deflated_sr >= 0.90


def test_factory_red_for_noise():
    """Pure noise → RED (no edge survives multiple-testing)."""
    from engine.validation.alpha_factory import CandidateSpec, screen_candidate
    rng = np.random.default_rng(61)
    idx = pd.date_range("2015-01-02", periods=500, freq="W-FRI")
    r = pd.Series(rng.normal(0.0001, 0.012, 500), index=idx)
    spec = CandidateSpec("noise", r, frequency="weekly", n_trials=35,
                         benchmark="none", cost_class="ss_large", annual_turnover=6.0)
    v = screen_candidate(spec)
    assert v.light == "RED"


def test_factory_cost_lowers_net_dsr():
    """A non-zero cost class must lower net deflated SR below gross."""
    from engine.validation.alpha_factory import CandidateSpec, screen_candidate
    rng = np.random.default_rng(62)
    idx = pd.date_range("2015-01-02", periods=485, freq="W-FRI")
    r = pd.Series(rng.normal(0.0016, 0.01, 485), index=idx)
    spec = CandidateSpec("c", r, frequency="weekly", n_trials=10,
                         benchmark="none", cost_class="ss_small", annual_turnover=8.0)
    v = screen_candidate(spec)
    assert v.net_deflated_sr <= v.deflated_sr + 1e-9


def test_factory_already_net_skips_cost():
    """already_net=True → net deflated SR equals gross."""
    from engine.validation.alpha_factory import CandidateSpec, screen_candidate
    rng = np.random.default_rng(63)
    idx = pd.date_range("2015-01-02", periods=485, freq="W-FRI")
    r = pd.Series(rng.normal(0.0016, 0.01, 485), index=idx)
    spec = CandidateSpec("net", r, frequency="weekly", n_trials=10,
                         benchmark="none", cost_class="ss_large",
                         annual_turnover=8.0, already_net=True)
    v = screen_candidate(spec)
    assert v.net_deflated_sr == pytest.approx(v.deflated_sr, abs=1e-9)


def test_factory_frequency_annualization():
    """Frequency must drive annualization (daily vs weekly give different
    annualized residual alpha for the same per-period mean)."""
    from engine.validation.alpha_factory import VALID_FREQ
    assert VALID_FREQ["daily"] == 252
    assert VALID_FREQ["weekly"] == 52
    assert VALID_FREQ["monthly"] == 12


def test_returns_hash_stable_and_distinct():
    """Same series → same fingerprint; a changed value → different."""
    from engine.validation.alpha_factory import _returns_hash
    idx = pd.date_range("2015-01-02", periods=50, freq="W-FRI")
    r1 = pd.Series(np.linspace(0.001, 0.002, 50), index=idx)
    r2 = r1.copy()
    assert _returns_hash(r1) == _returns_hash(r2)
    r2.iloc[0] += 0.01
    assert _returns_hash(r1) != _returns_hash(r2)


def test_gate_appends_to_ledger(tmp_path, monkeypatch):
    """gate() must record every run to the verdict ledger."""
    import json
    from engine.validation import alpha_factory as af
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(af, "_LEDGER", led)
    rng = np.random.default_rng(70)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    r = pd.Series(rng.normal(0.0016, 0.01, 300), index=idx)
    spec = af.CandidateSpec("led_test", r, frequency="weekly", n_trials=5,
                            benchmark="none", cost_class="mutual_fund")
    af.gate(spec)
    assert led.exists()
    recs = [json.loads(l) for l in led.read_text().splitlines() if l.strip()]
    assert len(recs) == 1
    assert recs[0]["name"] == "led_test"
    assert recs[0]["light"] in ("GREEN", "YELLOW", "RED")


def test_gate_flags_rescreen_under_changed_assumptions(tmp_path, monkeypatch):
    """Re-screening the SAME series with a different n_trials must be flagged
    in the verdict reasons — p-hacking-by-rerun made visible."""
    from engine.validation import alpha_factory as af
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(af, "_LEDGER", led)
    rng = np.random.default_rng(71)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    r = pd.Series(rng.normal(0.0016, 0.01, 300), index=idx)
    af.gate(af.CandidateSpec("x", r, frequency="weekly", n_trials=35,
                             benchmark="none", cost_class="mutual_fund"))
    v2 = af.gate(af.CandidateSpec("x", r, frequency="weekly", n_trials=2,
                                  benchmark="none", cost_class="mutual_fund"))
    assert any("RE-SCREENED" in reason for reason in v2.reasons)


def test_gate_no_flag_for_distinct_series(tmp_path, monkeypatch):
    """Different return series under different assumptions must NOT trip the
    re-screen flag (it keys on the series fingerprint, not the name)."""
    from engine.validation import alpha_factory as af
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(af, "_LEDGER", led)
    rng = np.random.default_rng(72)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    af.gate(af.CandidateSpec("a", pd.Series(rng.normal(0.0016, 0.01, 300), index=idx),
                             frequency="weekly", n_trials=35, benchmark="none",
                             cost_class="mutual_fund"))
    v2 = af.gate(af.CandidateSpec("a", pd.Series(rng.normal(0.0016, 0.01, 300), index=idx),
                                  frequency="weekly", n_trials=2, benchmark="none",
                                  cost_class="mutual_fund"))
    assert not any("RE-SCREENED" in reason for reason in v2.reasons)


def test_render_table_one_row_per_candidate():
    """render_table emits a header + one row per verdict."""
    from engine.validation.alpha_factory import CandidateSpec, screen_candidate, render_table
    rng = np.random.default_rng(73)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    vs = [screen_candidate(CandidateSpec(f"c{i}", pd.Series(rng.normal(0.0016, 0.01, 300), index=idx),
                                         frequency="weekly", n_trials=5, benchmark="none",
                                         cost_class="mutual_fund")) for i in range(3)]
    txt = render_table(vs)
    body = [l for l in txt.splitlines() if l and not l.startswith("-") and "CANDIDATE" not in l]
    assert len(body) == 3
