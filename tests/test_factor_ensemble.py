"""
tests/test_factor_ensemble.py — Sprint Week 2 ensemble combiner tests.

Spec: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5) §2.3
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine import factor_ensemble as fe
from engine.factor_ensemble import (
    ENSEMBLE_FACTORS,
    N_FACTORS,
    _cross_section_z_score,
    _nan_aware_factor_average,
    compute_ensemble_signal,
    compute_cross_factor_correlation,
    compute_per_factor_coverage,
    compute_factor_risk_contribution,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (per spec §2.2)
# ─────────────────────────────────────────────────────────────────────────────


def test_ensemble_factors_locked():
    """Spec §2.2 — 4 factors v1 locked."""
    assert ENSEMBLE_FACTORS == ("tsmom", "carry_equity", "quality", "bab")
    assert N_FACTORS == 4


# ─────────────────────────────────────────────────────────────────────────────
# Cross-section z-score (Step 2 vol-parity)
# ─────────────────────────────────────────────────────────────────────────────


def test_z_score_basic_arithmetic():
    """z = (x - mean) / std (population std, ddof=0)."""
    raw = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean = 3, std (ddof=0) = sqrt(((-2)²+(-1)²+0²+1²+2²)/5) = sqrt(2)
    z = _cross_section_z_score(raw)
    expected = (raw - 3.0) / np.sqrt(2.0)
    pd.testing.assert_series_equal(z, expected)


def test_z_score_preserves_nan():
    """NaN inputs → NaN outputs (per spec §2.3 NaN protocol)."""
    raw = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    z = _cross_section_z_score(raw)
    assert pd.isna(z.iloc[2])
    assert not pd.isna(z.iloc[0])


def test_z_score_insufficient_data_returns_all_nan():
    """len(valid) < 2 → all-NaN output."""
    raw = pd.Series([1.0, np.nan, np.nan])  # only 1 valid
    z = _cross_section_z_score(raw)
    assert z.isna().all()


def test_z_score_zero_variance_returns_zeros():
    """std < eps → all-zero (no cross-section information)."""
    raw = pd.Series([2.5, 2.5, 2.5, 2.5])
    z = _cross_section_z_score(raw)
    assert (z == 0.0).all()


def test_z_score_zero_variance_with_nan():
    """std=0 + some NaN → 0 for valid, NaN for invalid."""
    raw = pd.Series([2.5, 2.5, np.nan])
    z = _cross_section_z_score(raw)
    assert z.iloc[0] == 0.0
    assert z.iloc[1] == 0.0
    assert pd.isna(z.iloc[2])


def test_z_score_empty_series():
    z = _cross_section_z_score(pd.Series(dtype=float))
    assert z.empty


def test_z_score_normalizes_scale():
    """Different absolute scales → same z-score after normalization."""
    raw_small = pd.Series([0.001, 0.002, 0.003])
    raw_big   = pd.Series([1000, 2000, 3000])
    z_small = _cross_section_z_score(raw_small)
    z_big   = _cross_section_z_score(raw_big)
    pd.testing.assert_series_equal(z_small, z_big, check_exact=False)


# ─────────────────────────────────────────────────────────────────────────────
# NaN-aware factor average (Step 1 + Step 3)
# ─────────────────────────────────────────────────────────────────────────────


def test_nan_aware_average_all_factors_present():
    """All 4 factors valid for ticker → mean of 4 z-scores."""
    z = {
        "tsmom":        pd.Series({"QQQ": 1.0, "XLF": -1.0}),
        "carry_equity": pd.Series({"QQQ": 0.5, "XLF": 0.5}),
        "quality":      pd.Series({"QQQ": -0.5, "XLF": 1.5}),
        "bab":          pd.Series({"QQQ": 0.0, "XLF": -2.0}),
    }
    result = _nan_aware_factor_average(z, universe=["QQQ", "XLF"])
    # QQQ: (1.0 + 0.5 + (-0.5) + 0.0) / 4 = 0.25
    # XLF: (-1.0 + 0.5 + 1.5 + (-2.0)) / 4 = -0.25
    assert result["QQQ"] == pytest.approx(0.25)
    assert result["XLF"] == pytest.approx(-0.25)


def test_nan_aware_average_some_factors_missing():
    """NaN per-ticker → only available factors contribute."""
    z = {
        "tsmom":        pd.Series({"QQQ": 1.0, "GLD": 0.5}),
        "carry_equity": pd.Series({"QQQ": 0.5, "GLD": np.nan}),  # GLD non-equity → NaN
        "quality":      pd.Series({"QQQ": -0.5, "GLD": np.nan}),  # GLD non-equity → NaN
        "bab":          pd.Series({"QQQ": 0.0, "GLD": -1.0}),
    }
    result = _nan_aware_factor_average(z, universe=["QQQ", "GLD"])
    # QQQ: 4 factors, mean = (1+0.5-0.5+0)/4 = 0.25
    # GLD: 2 factors (TSMOM + BAB only), mean = (0.5+(-1.0))/2 = -0.25
    assert result["QQQ"] == pytest.approx(0.25)
    assert result["GLD"] == pytest.approx(-0.25)


def test_nan_aware_average_all_factors_nan_returns_zero():
    """All factors NaN for ticker → 0 fallback (neutral, no trade)."""
    z = {
        "tsmom":        pd.Series({"DEAD": np.nan}),
        "carry_equity": pd.Series({"DEAD": np.nan}),
        "quality":      pd.Series({"DEAD": np.nan}),
        "bab":          pd.Series({"DEAD": np.nan}),
    }
    result = _nan_aware_factor_average(z, universe=["DEAD"])
    assert result["DEAD"] == 0.0


def test_nan_aware_average_ticker_not_in_factors():
    """Ticker absent from any factor series → 0 fallback."""
    z = {
        "tsmom":        pd.Series({"QQQ": 1.0}),
        "carry_equity": pd.Series({"QQQ": 0.5}),
        "quality":      pd.Series({"QQQ": -0.5}),
        "bab":          pd.Series({"QQQ": 0.0}),
    }
    result = _nan_aware_factor_average(z, universe=["QQQ", "PHANTOM"])
    assert result["QQQ"] == pytest.approx(0.25)
    assert result["PHANTOM"] == 0.0  # not in any factor series


# ─────────────────────────────────────────────────────────────────────────────
# compute_ensemble_signal — top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────


def _mock_factor_signals_4factor():
    """Build mock raw signals for 4 factors over universe of 5 ETFs."""
    return {
        "tsmom":        pd.Series({"QQQ": 1, "XLF": 1, "XLE": -1, "GLD": 1, "TLT": 1}),
        "carry_equity": pd.Series({"QQQ": 0.005, "XLF": 0.020, "XLE": 0.030, "GLD": np.nan, "TLT": np.nan}),
        "quality":      pd.Series({"QQQ": 0.5, "XLF": -0.3, "XLE": -0.2, "GLD": np.nan, "TLT": np.nan}),
        "bab":          pd.Series({"QQQ": -1, "XLF": 0, "XLE": -1, "GLD": 1, "TLT": 1}),
    }


def test_compute_ensemble_signal_rejects_non_date():
    with pytest.raises(TypeError):
        compute_ensemble_signal(
            as_of="2026-05-31", universe=["QQQ"], asset_classes={"QQQ": "equity_sector"},
        )


def test_compute_ensemble_signal_requires_asset_classes():
    with pytest.raises(ValueError, match="asset_classes"):
        compute_ensemble_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ"],
            asset_classes=None,
        )


def test_compute_ensemble_signal_empty_universe():
    result = compute_ensemble_signal(
        as_of=datetime.date(2026, 5, 31),
        universe=[],
        asset_classes={},
    )
    assert result.empty


def test_compute_ensemble_signal_full_pipeline_mocked():
    """End-to-end: mock all 4 factor signals → vol-parity z-score → average."""
    universe = ["QQQ", "XLF", "XLE", "GLD", "TLT"]
    asset_classes = {
        "QQQ": "equity_sector",
        "XLF": "equity_sector",
        "XLE": "equity_sector",
        "GLD": "commodity",
        "TLT": "fixed_income",
    }
    mock_signals = _mock_factor_signals_4factor()

    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        return_value=mock_signals,
    ):
        result = compute_ensemble_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    assert isinstance(result, pd.Series)
    assert set(result.index) == set(universe)
    # All values finite (NaN-aware → 0 fallback)
    assert result.notna().all()


def test_compute_ensemble_signal_factor_failure_resilient():
    """If a factor fails, ensemble still produces output via remaining factors."""
    universe = ["QQQ", "XLF"]
    asset_classes = {"QQQ": "equity_sector", "XLF": "equity_sector"}

    # Quality fails (raises Exception)
    def fake_compute_all_signals(*args, **kwargs):
        return {
            "tsmom":        pd.Series({"QQQ": 1, "XLF": -1}),
            "carry_equity": pd.Series({"QQQ": 0.01, "XLF": 0.02}),
            "quality":      pd.Series(np.nan, index=universe),  # all NaN
            "bab":          pd.Series({"QQQ": 1, "XLF": 0}),
        }

    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        side_effect=fake_compute_all_signals,
    ):
        result = compute_ensemble_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    # Should still produce signal (3 factors valid, 1 NaN)
    assert result["QQQ"] != 0  # got non-trivial signal from valid factors
    assert result["XLF"] != 0


def test_compute_ensemble_signal_handles_factor_module_exception():
    """If a factor module raises, _compute_all_factor_signals returns all-NaN for it."""
    universe = ["QQQ"]
    asset_classes = {"QQQ": "equity_sector"}

    with patch("engine.factors.compute_tsmom_signal", side_effect=Exception("test fail")), \
         patch("engine.factors.compute_carry_equity_signal", return_value=pd.Series({"QQQ": 0.02})), \
         patch("engine.factors.compute_quality_signal", return_value=pd.Series({"QQQ": 0.5})), \
         patch("engine.factors.compute_bab_signal", return_value=pd.Series({"QQQ": 1})):
        result = compute_ensemble_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    # tsmom failed → all-NaN; ensemble averages remaining 3 factors' z-scores
    # Since universe is 1 ticker, z-score insufficient (len < 2) → all NaN → 0 fallback
    assert result["QQQ"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-factor correlation diagnostic (Validation Gate 2)
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_factor_correlation_returns_4x4_matrix():
    universe = ["QQQ", "XLF", "XLE", "GLD", "TLT", "EWG", "USMV"]
    asset_classes = {t: "equity_sector" for t in universe}
    mock_signals = {
        "tsmom":        pd.Series({t: i for i, t in enumerate(universe)}),
        "carry_equity": pd.Series({t: i * 0.5 for i, t in enumerate(universe)}),
        "quality":      pd.Series({t: i * 0.3 for i, t in enumerate(universe)}),
        "bab":          pd.Series({t: -i * 0.2 for i, t in enumerate(universe)}),
    }
    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        return_value=mock_signals,
    ):
        corr = compute_cross_factor_correlation(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    assert corr.shape == (4, 4)
    # Self-correlation = 1.0 on diagonal
    for f in ENSEMBLE_FACTORS:
        assert corr.loc[f, f] == pytest.approx(1.0)


def test_cross_factor_correlation_perfectly_collinear():
    """Two factors perfectly correlated → ρ = 1.0; symmetric matrix."""
    universe = ["A", "B", "C", "D"]
    asset_classes = {t: "equity_sector" for t in universe}
    mock_signals = {
        "tsmom":        pd.Series({"A": 1, "B": 2, "C": 3, "D": 4}),
        "carry_equity": pd.Series({"A": 1, "B": 2, "C": 3, "D": 4}),  # identical to TSMOM
        "quality":      pd.Series({"A": 4, "B": 3, "C": 2, "D": 1}),  # anti-correlated
        "bab":          pd.Series({"A": 0, "B": 1, "C": 0, "D": 1}),
    }
    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        return_value=mock_signals,
    ):
        corr = compute_cross_factor_correlation(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    assert corr.loc["tsmom", "carry_equity"] == pytest.approx(1.0)
    assert corr.loc["tsmom", "quality"] == pytest.approx(-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-factor coverage diagnostic (verdict template transparency)
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_per_factor_coverage():
    universe = ["QQQ", "XLF", "GLD", "TLT", "VXX"]
    asset_classes = {
        "QQQ": "equity_sector", "XLF": "equity_sector",
        "GLD": "commodity", "TLT": "fixed_income", "VXX": "volatility",
    }
    mock_signals = {
        "tsmom":        pd.Series({t: 1 for t in universe}),  # all 5
        "carry_equity": pd.Series({"QQQ": 0.01, "XLF": 0.02, "GLD": np.nan,
                                    "TLT": np.nan, "VXX": np.nan}),  # 2 only
        "quality":      pd.Series({"QQQ": 0.5, "XLF": -0.5, "GLD": np.nan,
                                    "TLT": np.nan, "VXX": np.nan}),  # 2 only
        "bab":          pd.Series({t: 1 for t in universe}),  # all 5
    }
    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        return_value=mock_signals,
    ):
        cov = compute_per_factor_coverage(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    assert cov["tsmom"]["n_total"] == 5
    assert cov["tsmom"]["n_valid"] == 5
    assert cov["tsmom"]["coverage_pct"] == 100.0

    assert cov["carry_equity"]["n_valid"] == 2
    assert cov["carry_equity"]["n_nan"] == 3
    assert cov["carry_equity"]["coverage_pct"] == 40.0

    assert cov["bab"]["coverage_pct"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Vol-parity risk contribution diagnostic (§五 Gate 6)
# ─────────────────────────────────────────────────────────────────────────────


def test_factor_risk_contribution_after_z_score_approximately_equal():
    """
    After z-score normalization, each factor's cross-section variance should be ~1.0.
    Risk share should be ~equal (1/N) when all factors have similar coverage.
    """
    universe = ["A", "B", "C", "D", "E", "F", "G", "H"]
    asset_classes = {t: "equity_sector" for t in universe}
    rng = np.random.default_rng(42)
    mock_signals = {
        f: pd.Series(rng.normal(0, scale, size=len(universe)), index=universe)
        for f, scale in zip(
            ENSEMBLE_FACTORS,
            [1.0, 0.05, 1.5, 1.0],  # different raw scales
        )
    }
    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        return_value=mock_signals,
    ):
        contributions = compute_factor_risk_contribution(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    # After z-score normalization, each factor cross-section variance is ~1.0
    # → risk share each ~25%
    for f in ENSEMBLE_FACTORS:
        assert contributions[f] == pytest.approx(0.25, abs=0.05)
    # Sum to 1.0
    assert sum(contributions.values()) == pytest.approx(1.0, abs=1e-6)


def test_factor_risk_contribution_unequal_when_one_factor_all_nan():
    """One factor all-NaN → 0 risk share; remaining factors share 100%."""
    universe = ["A", "B", "C", "D", "E"]
    asset_classes = {t: "equity_sector" for t in universe}
    mock_signals = {
        "tsmom":        pd.Series([1, 2, 3, 4, 5], index=universe, dtype=float),
        "carry_equity": pd.Series([2, 4, 6, 8, 10], index=universe, dtype=float),
        "quality":      pd.Series([1, 2, 3, 4, 5], index=universe, dtype=float),
        "bab":          pd.Series(np.nan, index=universe, dtype=float),  # all NaN
    }
    with patch(
        "engine.factor_ensemble._compute_all_factor_signals",
        return_value=mock_signals,
    ):
        contributions = compute_factor_risk_contribution(
            as_of=datetime.date(2026, 5, 31),
            universe=universe,
            asset_classes=asset_classes,
        )

    assert contributions["bab"] == 0.0
    assert sum(contributions.values()) == pytest.approx(1.0, abs=1e-6)
