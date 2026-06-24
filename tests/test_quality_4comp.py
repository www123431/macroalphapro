"""
tests/test_quality_4comp.py — Unit tests for engine.factors_singlename.quality_4comp
(W-B-4, 2026-05-10).

Coverage scope:
  - Locked constants match spec (4 sub-components, equal weights, min-5 gate)
  - Each sub-component mock z-score has correct shape + determinism
  - Composite is mean-zero, std-one re-z-score
  - 3-of-4-sub tolerance (NaN propagation)
  - Auto-fallback to mock when WRDS unavailable
  - Real-path stub raises pre-activation
  - API parity with value_pe + dividend_yield
"""
from __future__ import annotations

import datetime
import inspect

import numpy as np
import pandas as pd
import pytest

from engine.factors_singlename import quality_4comp
from engine.factors_singlename.quality_4comp import (
    MIN_UNIVERSE_FOR_ZSCORE_LOCKED,
    SUB_COMPONENTS_LOCKED,
    SUB_WEIGHTS_LOCKED,
    _COMPUSTAT_FIELDS_BY_SUB,
    _mock_subscore,
    compute_quality_singlestock_signal,
)


# ── Locked constants ────────────────────────────────────────────────────────
def test_locked_sub_components_per_afp_2019() -> None:
    """AFP 2019 locks exactly 4 sub-components, equal-weighted."""
    assert SUB_COMPONENTS_LOCKED == ("profitability", "growth", "safety", "payout")
    assert SUB_WEIGHTS_LOCKED == (0.25, 0.25, 0.25, 0.25)
    assert sum(SUB_WEIGHTS_LOCKED) == 1.0
    assert MIN_UNIVERSE_FOR_ZSCORE_LOCKED == 5


def test_compustat_fields_documented_for_all_subs() -> None:
    """Every locked sub must have its Compustat field list documented for
    activation guidance."""
    for sub in SUB_COMPONENTS_LOCKED:
        assert sub in _COMPUSTAT_FIELDS_BY_SUB
        assert len(_COMPUSTAT_FIELDS_BY_SUB[sub]) > 0


# ── Helper: tiny panel ──────────────────────────────────────────────────────
def _make_panel(tickers: list[str], as_of: datetime.date, n_days: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=n_days, freq="B")
    return pd.DataFrame(
        {t: 100.0 + i * 5.0 for i, t in enumerate(tickers)},
        index=dates,
    )


# ── Sub-component mock behavior ─────────────────────────────────────────────
def test_mock_subscore_produces_zscore_series() -> None:
    universe = [f"T{i:02d}" for i in range(10)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    z = _mock_subscore("profitability", datetime.date(2024, 6, 28), universe, panel)
    valid = z.dropna()
    assert len(valid) == 10
    assert abs(valid.mean()) < 1e-9       # exact zero by re-z-score
    assert abs(valid.std(ddof=1) - 1.0) < 1e-9


def test_mock_subscores_are_independent_across_subs() -> None:
    """4 sub-components must use distinct seeds (not collinear)."""
    universe = [f"T{i:02d}" for i in range(10)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    z_prof = _mock_subscore("profitability", datetime.date(2024, 6, 28), universe, panel)
    z_grow = _mock_subscore("growth", datetime.date(2024, 6, 28), universe, panel)
    z_safe = _mock_subscore("safety", datetime.date(2024, 6, 28), universe, panel)
    z_payo = _mock_subscore("payout", datetime.date(2024, 6, 28), universe, panel)
    # Cross-correlations should be ~0 (small sample noise OK)
    corr_pg = z_prof.corr(z_grow)
    corr_ps = z_prof.corr(z_safe)
    corr_pp = z_prof.corr(z_payo)
    assert abs(corr_pg) < 0.7  # generous bound, mostly to rule out perfect ±1
    assert abs(corr_ps) < 0.7
    assert abs(corr_pp) < 0.7
    # And not literally equal
    assert not z_prof.equals(z_grow)
    assert not z_prof.equals(z_safe)
    assert not z_prof.equals(z_payo)


def test_mock_subscore_is_deterministic() -> None:
    universe = [f"T{i:02d}" for i in range(10)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    z1 = _mock_subscore("safety", datetime.date(2024, 6, 28), universe, panel)
    z2 = _mock_subscore("safety", datetime.date(2024, 6, 28), universe, panel)
    pd.testing.assert_series_equal(z1, z2)


def test_mock_subscore_returns_nan_below_min_universe() -> None:
    universe = ["A", "B"]  # only 2 < 5
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    z = _mock_subscore("profitability", datetime.date(2024, 6, 28), universe, panel)
    assert z.isna().all()


# ── Composite signal behavior ───────────────────────────────────────────────
def test_composite_signal_is_zscore() -> None:
    universe = [f"T{i:02d}" for i in range(15)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel, mock_mode=True,
    )
    valid = sig.dropna()
    assert len(valid) == 15
    assert abs(valid.mean()) < 1e-9
    assert abs(valid.std(ddof=1) - 1.0) < 1e-9


def test_composite_signal_deterministic() -> None:
    universe = [f"T{i:02d}" for i in range(10)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    s1 = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel, mock_mode=True,
    )
    s2 = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel, mock_mode=True,
    )
    pd.testing.assert_series_equal(s1, s2)


def test_composite_different_as_of_gives_different_signals() -> None:
    universe = [f"T{i:02d}" for i in range(10)]
    panel_jun = _make_panel(universe, datetime.date(2024, 6, 28))
    panel_sep = _make_panel(universe, datetime.date(2024, 9, 30))
    s_jun = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel_jun, mock_mode=True,
    )
    s_sep = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 9, 30), universe=universe, panel=panel_sep, mock_mode=True,
    )
    assert not s_jun.equals(s_sep)


# ── NaN propagation + edge cases ────────────────────────────────────────────
def test_missing_ticker_returns_nan_in_composite() -> None:
    universe = ["AAPL", "MSFT", "GOOG", "MISSING", "NVDA", "META", "T07"]
    panel = _make_panel(["AAPL", "MSFT", "GOOG", "NVDA", "META", "T07"],
                         datetime.date(2024, 6, 28))
    sig = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel, mock_mode=True,
    )
    assert pd.isna(sig["MISSING"])
    assert sig.dropna().shape[0] == 6


def test_min_universe_gate_returns_all_nan_below_5() -> None:
    universe = ["A", "B", "C", "D"]   # only 4 < 5
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel, mock_mode=True,
    )
    assert sig.isna().all()


def test_empty_universe_returns_empty() -> None:
    sig = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=[], panel=_make_panel(["X"], datetime.date(2024, 6, 28)),
        mock_mode=True,
    )
    assert sig.empty


def test_empty_panel_returns_all_nan() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    sig = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe,
        panel=pd.DataFrame(), mock_mode=True,
    )
    assert sig.isna().all()


def test_invalid_as_of_type_raises() -> None:
    with pytest.raises(TypeError, match="as_of must be datetime.date"):
        compute_quality_singlestock_signal(
            as_of="2024-06-28",  # type: ignore
            universe=["A"], panel=pd.DataFrame(), mock_mode=True,
        )


# ── Auto-mode routing ───────────────────────────────────────────────────────
def test_auto_mode_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.universe_singlename import crsp_loader as cl
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)
    universe = [f"T{i:02d}" for i in range(8)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_quality_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel,
    )
    assert sig.notna().all()


# ── Real-path stub error handling ───────────────────────────────────────────
def test_real_mode_no_wrds_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.universe_singlename import crsp_loader as cl
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)
    universe = [f"T{i:02d}" for i in range(8)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    with pytest.raises(RuntimeError, match="WRDS not configured"):
        compute_quality_singlestock_signal(
            as_of=datetime.date(2024, 6, 28),
            universe=universe, panel=panel, mock_mode=False,
        )


def test_real_mode_no_longer_stubbed_post_activation() -> None:
    """Post-2026-05-11 (Wave B activation): real Compustat Quality 4-comp
    path is implemented. Guards against stub regression."""
    from engine.factors_singlename.quality_4comp import _real_quality_signal
    import inspect
    src = inspect.getsource(_real_quality_signal)
    assert "NotImplementedError" not in src, (
        "_real_quality_signal regressed to stub state. Post-Wave-B-activation, "
        "this function must be implemented."
    )


# ── API parity with Wave A (dividend_yield) and W-B-3 (value_pe) ────────────
def test_api_signature_parity_with_dividend_yield_and_value_pe() -> None:
    """quality_4comp must be drop-in replaceable in the factor ensemble."""
    from engine.factors_singlename.dividend_yield import (
        compute_dividend_yield_singlestock_signal,
    )
    from engine.factors_singlename.value_pe import (
        compute_value_pe_singlestock_signal,
    )
    a_sig = inspect.signature(compute_dividend_yield_singlestock_signal)
    b_sig = inspect.signature(compute_value_pe_singlestock_signal)
    q_sig = inspect.signature(compute_quality_singlestock_signal)
    common = {"as_of", "universe", "asset_classes", "panel"}
    for name, s in [("dividend_yield", a_sig), ("value_pe", b_sig), ("quality", q_sig)]:
        assert common.issubset(set(s.parameters)), f"{name} missing common args: {s}"
