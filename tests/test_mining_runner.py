"""
tests/test_mining_runner.py — F-LAB-E4 unit tests.

Coverage:
  - Locked verdict thresholds + verdict classifier
  - Portfolio weight construction (z-weighted long-short, expected_sign)
  - NW t-stat computation (synthetic data with known properties)
  - Walk-forward integration with real factor (IVOL on synthetic panel)
  - Artifact persistence (tmp_path isolation, no production data write)
  - 0-LLM-imports invariant
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.factor_lab import mining_runner
from engine.factor_lab.mining_runner import (
    GROSS_EXPOSURE_LOCKED,
    T_DIRECTIONAL_LOCKED,
    T_PROMOTABLE_LOCKED,
    TC_BPS_LOCKED,
    VERDICT_DIRECTIONAL_AGAINST,
    VERDICT_DIRECTIONAL_POSITIVE,
    VERDICT_NOISE,
    VERDICT_PROMOTABLE,
    MiningResult,
    _classify_verdict,
    _compute_nw_t_stat,
    _compute_portfolio_weights,
    _compute_realized_return,
    _compute_tc_drag,
    _generate_monthend_dates,
    run_mining_session,
)


# ── Locked constants ────────────────────────────────────────────────────────
def test_locked_thresholds_match_design() -> None:
    assert T_PROMOTABLE_LOCKED == 2.5
    assert T_DIRECTIONAL_LOCKED == 1.65
    assert TC_BPS_LOCKED == 12.0
    assert GROSS_EXPOSURE_LOCKED == 1.0


# ── Verdict classification ─────────────────────────────────────────────────
def test_verdict_promotable() -> None:
    assert _classify_verdict(nw_t_abs=3.0, sign_match=True) == VERDICT_PROMOTABLE


def test_verdict_directional_positive() -> None:
    assert _classify_verdict(nw_t_abs=2.0, sign_match=True) == VERDICT_DIRECTIONAL_POSITIVE


def test_verdict_noise() -> None:
    assert _classify_verdict(nw_t_abs=1.0, sign_match=True) == VERDICT_NOISE


def test_verdict_directional_against_overrides_t_stat() -> None:
    """Sign mismatch → directional_against_prior regardless of |t|."""
    assert _classify_verdict(nw_t_abs=4.0, sign_match=False) == VERDICT_DIRECTIONAL_AGAINST
    assert _classify_verdict(nw_t_abs=0.5, sign_match=False) == VERDICT_DIRECTIONAL_AGAINST


# ── Portfolio weight construction ───────────────────────────────────────────
def test_portfolio_weights_zero_sum_after_zscore() -> None:
    """Cross-section z-score has mean ≈ 0 → weights sum ≈ 0 (long-short balanced)."""
    z = pd.Series({"A": -1.0, "B": -0.5, "C": 0.0, "D": 0.5, "E": 1.0})
    w = _compute_portfolio_weights(z, expected_sign=+1)
    assert abs(w.sum()) < 1e-9   # symmetric z → zero net


def test_portfolio_weights_gross_one() -> None:
    """Gross exposure = sum(|w|) ≈ 1.0 (no leverage)."""
    z = pd.Series({"A": -1.0, "B": -0.5, "C": 0.5, "D": 1.0})
    w = _compute_portfolio_weights(z, expected_sign=+1)
    assert abs(w.abs().sum() - 1.0) < 1e-9


def test_portfolio_weights_expected_sign_negative_inverts() -> None:
    """expected_sign=-1 → portfolio is reverse direction (short high z, long low z)."""
    z = pd.Series({"A": -1.0, "B": 1.0})
    w_pos = _compute_portfolio_weights(z, expected_sign=+1)
    w_neg = _compute_portfolio_weights(z, expected_sign=-1)
    pd.testing.assert_series_equal(w_pos, -w_neg)


def test_portfolio_weights_drops_nan_and_zero() -> None:
    z = pd.Series({"A": np.nan, "B": 0.0, "C": 1.0, "D": -1.0})
    w = _compute_portfolio_weights(z, expected_sign=+1)
    assert "A" not in w.index
    assert "B" not in w.index
    assert set(w.index) == {"C", "D"}


def test_portfolio_weights_all_nan_returns_empty() -> None:
    z = pd.Series({"A": np.nan, "B": np.nan})
    w = _compute_portfolio_weights(z, expected_sign=+1)
    assert w.empty


# ── NW t-stat ───────────────────────────────────────────────────────────────
def test_nw_t_stat_zero_for_zero_mean_sample() -> None:
    arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    assert _compute_nw_t_stat(arr) == 0.0


def test_nw_t_stat_high_for_strong_signal() -> None:
    """Strong positive signal with low noise → large |t|."""
    rng = np.random.default_rng(42)
    arr = rng.normal(loc=0.05, scale=0.01, size=100)   # high mean, low std
    t = _compute_nw_t_stat(arr)
    assert abs(t) > 10.0   # very strong


def test_nw_t_stat_near_zero_for_pure_noise() -> None:
    """Pure noise (mean=0) → |t| typically < 2."""
    rng = np.random.default_rng(1)
    arr = rng.normal(loc=0.0, scale=0.01, size=100)
    t = _compute_nw_t_stat(arr)
    assert abs(t) < 3.0   # could be anywhere up to ~3 due to sampling


def test_nw_t_stat_handles_n_below_2() -> None:
    assert _compute_nw_t_stat(np.array([])) == 0.0
    assert _compute_nw_t_stat(np.array([0.05])) == 0.0


def test_nw_t_stat_handles_zero_variance() -> None:
    arr = np.array([0.05, 0.05, 0.05, 0.05])  # constant → zero variance
    assert _compute_nw_t_stat(arr) == 0.0


# ── TC drag ────────────────────────────────────────────────────────────────
def test_tc_drag_first_period_uses_full_turnover() -> None:
    weights_new = pd.Series({"A": 0.4, "B": -0.4, "C": 0.2})
    tc = _compute_tc_drag(weights_new, weights_prev=None, bps_roundtrip=12.0)
    expected_turnover = 1.0  # gross
    expected_tc = expected_turnover * (12.0 / 10000.0)
    assert abs(tc - expected_tc) < 1e-12


def test_tc_drag_zero_when_weights_unchanged() -> None:
    w = pd.Series({"A": 0.5, "B": -0.5})
    tc = _compute_tc_drag(w, weights_prev=w, bps_roundtrip=12.0)
    assert tc == 0.0


def test_tc_drag_proportional_to_turnover() -> None:
    w_new  = pd.Series({"A": 0.5, "B": -0.5})
    w_prev = pd.Series({"A": 0.0, "B": -1.0})
    # diff = (0.5, 0.5) → turnover = 0.5 × 1.0 = 0.5 (one-way)
    tc = _compute_tc_drag(w_new, w_prev, bps_roundtrip=12.0)
    assert abs(tc - 0.5 * (12.0 / 10000.0)) < 1e-12


# ── Realized return ────────────────────────────────────────────────────────
def test_realized_return_simple() -> None:
    dates = pd.date_range("2024-01-01", "2024-02-29", freq="B")
    panel = pd.DataFrame({
        "A": np.linspace(100.0, 110.0, len(dates)),  # +10%
        "B": np.linspace(100.0,  95.0, len(dates)),  # -5%
    }, index=dates)
    weights = pd.Series({"A": 0.5, "B": -0.5})
    r = _compute_realized_return(weights, panel,
                                 datetime.date(2024, 1, 1),
                                 datetime.date(2024, 2, 29))
    # Long A (+10%) × 0.5 + Short B (-5%) × -0.5 = 0.05 - (-0.025) = 0.075
    assert abs(r - 0.075) < 0.01   # tolerance for date snapping


def test_realized_return_skips_missing_ticker() -> None:
    dates = pd.date_range("2024-01-01", "2024-02-29", freq="B")
    panel = pd.DataFrame({
        "A": np.linspace(100.0, 110.0, len(dates)),
    }, index=dates)
    weights = pd.Series({"A": 0.5, "MISSING": 0.5})
    r = _compute_realized_return(weights, panel,
                                 datetime.date(2024, 1, 1),
                                 datetime.date(2024, 2, 29))
    # Only A contributes
    assert r > 0


# ── Month-end date generation ──────────────────────────────────────────────
def test_generate_monthend_dates() -> None:
    dates = _generate_monthend_dates(
        datetime.date(2024, 1, 15), datetime.date(2024, 4, 10),
    )
    # Should include Jan/Feb/Mar last day-of-month (within range)
    assert datetime.date(2024, 1, 31) in dates
    assert datetime.date(2024, 2, 29) in dates
    assert datetime.date(2024, 3, 31) in dates


def test_generate_monthend_dates_empty_for_short_range() -> None:
    dates = _generate_monthend_dates(
        datetime.date(2024, 1, 15), datetime.date(2024, 1, 25),
    )
    # No month-end falls in this short range
    assert dates == []


# ── End-to-end mini-walk-forward(no real factor; stubbed signal_fn)──────
def _stub_signal_fn(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Deterministic z-score: rank tickers alphabetically."""
    n = len(universe)
    if n < 5:
        return pd.Series(np.nan, index=universe, dtype=float)
    sorted_u = sorted(universe)
    raw = pd.Series(
        {t: float(i - n / 2) for i, t in enumerate(sorted_u)},
        dtype=float,
    )
    valid = raw.dropna()
    return (raw - valid.mean()) / valid.std(ddof=1)


def test_run_mining_session_unknown_factor_raises() -> None:
    panel = pd.DataFrame()
    with pytest.raises(KeyError, match="not in FACTOR_REGISTRY_SINGLENAME"):
        run_mining_session(
            factor_id="nonexistent_factor",
            universe_at_date_fn=lambda d: ["A"],
            panel=panel,
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 12, 31),
        )


def test_run_mining_session_with_registered_factor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path:    Path,
) -> None:
    """End-to-end on registered IVOL factor with synthetic panel."""
    # Redirect artifact paths to tmp
    monkeypatch.setattr(mining_runner, "_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(mining_runner, "_decisions_dir", lambda: tmp_path / "decisions")

    # Build synthetic panel covering 2 years for IVOL (needs SPY benchmark)
    dates = pd.bdate_range("2022-01-01", "2024-01-31", freq="B")
    rng = np.random.default_rng(7)
    universe = [f"T{i:02d}" for i in range(15)]
    panel = pd.DataFrame(index=dates, dtype=float)
    panel["SPY"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    for t in universe:
        scale = rng.uniform(0.005, 0.025)
        panel[t] = 100.0 * np.exp(np.cumsum(rng.normal(0.0, scale, len(dates))))

    result = run_mining_session(
        factor_id="ivol_singlestock",
        universe_at_date_fn=lambda d: universe,
        panel=panel,
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 12, 31),
        persist_artifacts=True,
    )

    assert isinstance(result, MiningResult)
    assert result.factor_id == "ivol_singlestock"
    assert result.expected_sign == -1
    assert result.n_periods >= 5   # ~12 months of monthly rebalances minus last incomplete
    assert result.verdict in {
        VERDICT_PROMOTABLE,
        VERDICT_DIRECTIONAL_POSITIVE,
        VERDICT_DIRECTIONAL_AGAINST,
        VERDICT_NOISE,
    }

    # Artifacts persisted
    json_files = list((tmp_path / "data").glob("ivol_singlestock_*.json"))
    md_files   = list((tmp_path / "decisions").glob("factor_mining_ivol_singlestock_*.md"))
    assert len(json_files) == 1
    assert len(md_files)   == 1

    # JSON has expected schema
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["factor_id"] == "ivol_singlestock"
    assert "verdict" in payload
    assert "monthly_returns_net" in payload

    # Markdown has key sections
    md = md_files[0].read_text(encoding="utf-8")
    assert "Tier 1 Mining Verdict" in md
    assert "NOT a production claim" in md
    assert result.factor_id in md


def test_run_mining_session_invalid_dates() -> None:
    panel = pd.DataFrame({"A": [100.0]}, index=[pd.Timestamp("2024-01-01")])
    with pytest.raises(ValueError, match="must be < end_date"):
        run_mining_session(
            factor_id="ivol_singlestock",
            universe_at_date_fn=lambda d: ["A"],
            panel=panel,
            start_date=datetime.date(2024, 12, 31),
            end_date=datetime.date(2024, 1, 1),
        )


def test_run_mining_session_empty_panel_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_mining_session(
            factor_id="ivol_singlestock",
            universe_at_date_fn=lambda d: ["A"],
            panel=pd.DataFrame(),
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 12, 31),
        )


# ── Boundary invariant — no LLM imports ─────────────────────────────────────
def test_module_has_no_llm_imports() -> None:
    src = open(mining_runner.__file__, encoding="utf-8").read()
    forbidden = ["google.generativeai", "google.genai",
                 "from engine.deepseek_client", "from engine.key_pool"]
    for pattern in forbidden:
        assert pattern not in src, (
            f"mining_runner violates 0-LLM-imports invariant: found {pattern!r}"
        )
