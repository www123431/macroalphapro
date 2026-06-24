"""Tests for the strict-gate production redesign — 3 commits in one file.

Per [[project-gate-production-redesign-2026-05-30]]:
  1. HAC lags via cov_type='HAC' kwarg
  2. n_trials default 1 (kill ledger semantics)
  3. profile= kwarg + GATE_PROFILE per-template constants
  + doctrine red-lines (HLZ_T / DEFLSR_MIN / MAX_BOOK_CORR immutable)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── HAC lags (Commit 1) ────────────────────────────────────────────────────

def test_ols_plain_vs_hac_returns_different_t_for_autocorrelated_residuals():
    """With overlapping holdings → positive residual autocorrelation,
    plain OLS overstates t-stat; HAC corrects it."""
    from engine.research.pipeline import _ols_alpha_t
    np.random.seed(42)
    n = 240
    # Construct AR(3) residuals — multi-month-hold style
    noise = np.random.randn(n)
    ar = np.zeros(n)
    for i in range(3, n):
        ar[i] = 0.5 * ar[i-1] + 0.3 * ar[i-2] + 0.2 * ar[i-3] + noise[i]
    factor = np.random.randn(n)
    y = pd.Series(0.005 + 0.1 * factor + ar * 0.01,
                  index=pd.date_range("2010-01-01", periods=n, freq="ME"))
    X = pd.DataFrame({"f": factor}, index=y.index)
    _, t_plain, _ = _ols_alpha_t(y, X, hac_lags=0)
    _, t_hac, _ = _ols_alpha_t(y, X, hac_lags=12)
    # HAC t-stat should be SMALLER in absolute value (proper inflation correction)
    assert abs(t_hac) < abs(t_plain), \
        f"HAC ({t_hac:.3f}) should be smaller than plain OLS ({t_plain:.3f})"


def test_hac_lags_zero_matches_plain_ols():
    """hac_lags=0 must reproduce plain OLS exactly (back-compat)."""
    from engine.research.pipeline import _ols_alpha_t
    np.random.seed(43)
    n = 100
    y = pd.Series(np.random.randn(n) * 0.05 + 0.01,
                  index=pd.date_range("2010-01-01", periods=n, freq="ME"))
    X = pd.DataFrame({"f": np.random.randn(n)}, index=y.index)
    a0, t0, r0 = _ols_alpha_t(y, X, hac_lags=0)
    a_omit, t_omit, r_omit = _ols_alpha_t(y, X)    # default kwarg
    assert a0 == a_omit
    assert t0 == t_omit
    assert r0 == r_omit


# ── n_trials production semantics (Commit 2) ─────────────────────────────

def test_run_gate_default_n_trials_is_one_not_ledger():
    """Default n_trials=1, NOT ledger_n_trials() + 1."""
    from engine.research.pipeline import run_gate
    np.random.seed(44)
    series = pd.Series(np.random.randn(120) * 0.05 + 0.005,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="default_n_trials_smoke",
                          pead_control=False, log=False)
    assert verdict["n_trials"] == 1, \
        f"default n_trials should be 1 (production), got {verdict['n_trials']}"


def test_explicit_n_trials_overrides_default():
    from engine.research.pipeline import run_gate
    np.random.seed(45)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="explicit_n_trials",
                          n_trials=15, pead_control=False, log=False)
    assert verdict["n_trials"] == 15


def test_n_trials_one_vs_hundred_produces_different_dsr():
    """Sanity: deflation actually responds to n_trials when caller
    declares it. Use a borderline-significant series (Sharpe ~0.7) so
    DSR is in the sensitive range, not saturated at 1.0."""
    from engine.research.pipeline import run_gate
    np.random.seed(60)
    series = pd.Series(np.random.randn(120) * 0.05 + 0.008,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    v1 = run_gate(series, name="n_trials_low",
                     n_trials=1, pead_control=False, log=False)
    v100 = run_gate(series, name="n_trials_hi",
                       n_trials=100, pead_control=False, log=False)
    assert v100["deflated_sr"] < v1["deflated_sr"], \
        f"n_trials=100 should deflate harder than n_trials=1; got {v100['deflated_sr']} vs {v1['deflated_sr']}"


# ── GATE_PROFILE (Commit 3) ──────────────────────────────────────────────

def test_profile_overrides_hac_lags():
    from engine.research.pipeline import run_gate
    np.random.seed(47)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="profile_hac",
                          profile={"hac_lags": 18},
                          pead_control=False, log=False)
    assert verdict["hac_lags"] == 18


def test_profile_overrides_pead_control():
    from engine.research.pipeline import run_gate
    np.random.seed(48)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    # profile says pead_control=False — should bypass PEAD residualization.
    # Signal: corr_with_book is None when pead_control=False (no PEAD series
    # to correlate against), regardless of alpha_t_ff5umd_pead (which is
    # set to t_ff in the False branch for back-compat ledger schema).
    verdict = run_gate(series, name="profile_pead",
                          profile={"pead_control": False}, log=False)
    assert verdict.get("corr_with_book") is None or pd.isna(
        verdict.get("corr_with_book")), \
        f"pead_control=False should yield corr_with_book=None, got {verdict.get('corr_with_book')}"


def test_profile_n_trials_base_applied_when_kwarg_default():
    from engine.research.pipeline import run_gate
    np.random.seed(49)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="profile_n_trials",
                          profile={"n_trials_base": 30},
                          pead_control=False, log=False)
    assert verdict["n_trials"] == 30


def test_explicit_n_trials_wins_over_profile():
    """Explicit n_trials kwarg overrides profile n_trials_base."""
    from engine.research.pipeline import run_gate
    np.random.seed(50)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="explicit_wins",
                          n_trials=5,
                          profile={"n_trials_base": 30},
                          pead_control=False, log=False)
    assert verdict["n_trials"] == 5


def test_profile_unknown_keys_ignored_with_warning():
    """Unknown profile keys should be logged but NOT raise."""
    from engine.research.pipeline import run_gate
    np.random.seed(51)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="unknown_key",
                          profile={"hac_lags": 6,
                                    "made_up_key": 999,
                                    "another_one": "garbage"},
                          pead_control=False, log=False)
    # Still ran; doctrine still held
    assert verdict["available"] is True
    assert verdict["hac_lags"] == 6


# ── Doctrine red lines ────────────────────────────────────────────────────

def test_doctrine_red_lines_are_module_level_constants():
    """HLZ_T / DEFLSR_MIN / MAX_BOOK_CORR are immutable global bars."""
    from engine.research import pipeline
    assert pipeline.HLZ_T == 3.0
    assert pipeline.DEFLSR_MIN >= 0.9
    assert pipeline.MAX_BOOK_CORR <= 0.5


def test_profile_cannot_modify_doctrine_red_lines():
    """STRICT: even if profile contains a key matching a red line,
    it must not affect actual behavior. _ALLOWED_PROFILE_KEYS
    whitelist enforces this."""
    from engine.research.pipeline import _ALLOWED_PROFILE_KEYS
    assert "HLZ_T" not in _ALLOWED_PROFILE_KEYS
    assert "hlz_t" not in _ALLOWED_PROFILE_KEYS
    assert "DEFLSR_MIN" not in _ALLOWED_PROFILE_KEYS
    assert "deflsr_min" not in _ALLOWED_PROFILE_KEYS
    assert "MAX_BOOK_CORR" not in _ALLOWED_PROFILE_KEYS
    assert "max_book_corr" not in _ALLOWED_PROFILE_KEYS


def test_run_gate_runtime_asserts_doctrine():
    """Even if someone monkeypatches the constants at runtime, the
    run_gate function asserts them before executing. The function
    must abort, not return a verdict computed against a tampered bar."""
    from engine.research import pipeline
    np.random.seed(52)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    # Save and tamper
    orig = pipeline.HLZ_T
    pipeline.HLZ_T = 2.0    # try to lower the bar
    try:
        with pytest.raises(AssertionError):
            pipeline.run_gate(series, name="tampered",
                                 pead_control=False, log=False)
    finally:
        pipeline.HLZ_T = orig


# ── Per-template GATE_PROFILE constants ──────────────────────────────────

def test_all_templates_declare_gate_profile():
    """Every template module must declare a GATE_PROFILE constant with
    at minimum the 4 core keys. Additional whitelisted keys (e.g.
    oos_split for sparse-event templates) are permitted."""
    from engine.research.pipeline import _ALLOWED_PROFILE_KEYS
    from engine.research.templates import (
        equity_xsmom, factor_quartile, cross_asset_tsmom,
        event_study, dispersion, term_structure,
    )
    required_keys = {"hac_lags", "cost_bps_default", "pead_control",
                       "n_trials_base"}
    for mod in (equity_xsmom, factor_quartile, cross_asset_tsmom,
                event_study, dispersion, term_structure):
        assert hasattr(mod, "GATE_PROFILE"), \
            f"{mod.__name__} missing GATE_PROFILE"
        assert isinstance(mod.GATE_PROFILE, dict)
        keys = set(mod.GATE_PROFILE.keys())
        # Required keys must be present
        assert required_keys.issubset(keys), \
            f"{mod.__name__} GATE_PROFILE missing required keys: " \
            f"{required_keys - keys}"
        # Any extra keys must be in the whitelist
        extra = keys - required_keys
        assert extra.issubset(_ALLOWED_PROFILE_KEYS), \
            f"{mod.__name__} GATE_PROFILE has non-whitelisted keys: " \
            f"{extra - _ALLOWED_PROFILE_KEYS}"


def test_gate_profiles_reflect_economic_logic():
    """HAC lags ≈ hold-period autocorrelation; equity → pead_control True;
    cross-asset (futures/rates/fx) → pead_control False."""
    from engine.research.templates import (
        equity_xsmom, event_study, cross_asset_tsmom, term_structure,
    )
    # Equity monthly rebal → HAC ~6
    assert equity_xsmom.GATE_PROFILE["hac_lags"] == 6
    assert equity_xsmom.GATE_PROFILE["pead_control"] is True
    # Event-driven multi-month hold → HAC ~12
    assert event_study.GATE_PROFILE["hac_lags"] == 12
    assert event_study.GATE_PROFILE["pead_control"] is True
    # 12-month TSMOM hold → HAC ~18
    assert cross_asset_tsmom.GATE_PROFILE["hac_lags"] == 18
    assert cross_asset_tsmom.GATE_PROFILE["pead_control"] is False
    # Cross-asset curve → pead_control False
    assert term_structure.GATE_PROFILE["pead_control"] is False


# ── ledger entry enrichment ──────────────────────────────────────────────

def test_ledger_entry_records_hac_lags():
    """Gate run output should record hac_lags so later analysis can
    tell which runs used HAC and which used plain OLS."""
    from engine.research.pipeline import run_gate
    np.random.seed(53)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="hac_recorded", hac_lags=12,
                          pead_control=False, log=False)
    assert "hac_lags" in verdict
    assert verdict["hac_lags"] == 12
