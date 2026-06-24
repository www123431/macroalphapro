"""
tests/test_value_pe.py — Unit tests for engine.factors_singlename.value_pe
(W-B-3, 2026-05-10).

Coverage scope:
  - Mock-mode E/P synthesis (deterministic, reproducible)
  - Cross-section z-score (mean ~0, std ~1, NaN propagation)
  - Auto-fallback to mock when WRDS unavailable
  - Real-path stub raises pre-activation
  - API parity with dividend_yield (Wave A counterpart)
"""
from __future__ import annotations

import datetime
import inspect
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from engine.factors_singlename import value_pe
from engine.factors_singlename.value_pe import (
    EARNINGS_LOOKBACK_QUARTERS_LOCKED,
    MIN_UNIVERSE_FOR_ZSCORE_LOCKED,
    compute_value_pe_singlestock_signal,
)


# ── Locked constants ────────────────────────────────────────────────────────
def test_locked_constants_match_spec() -> None:
    """Spec §2.2 Wave B locks 4-quarter (TTM) lookback + 5-min cross-section."""
    assert EARNINGS_LOOKBACK_QUARTERS_LOCKED == 4
    assert MIN_UNIVERSE_FOR_ZSCORE_LOCKED == 5


# ── Helper: synthesise a panel for tests ────────────────────────────────────
def _make_panel(
    tickers: list[str],
    as_of:   datetime.date,
    n_days:  int = 60,
) -> pd.DataFrame:
    """Tiny synthetic panel (constant prices for simplicity)."""
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=n_days, freq="B")
    return pd.DataFrame(
        {t: 100.0 + i * 5.0 for i, t in enumerate(tickers)},
        index=dates,
    )


# ── Mock mode happy path ────────────────────────────────────────────────────
def test_mock_mode_returns_zscore_series_with_correct_index() -> None:
    universe = ["AAPL", "MSFT", "GOOG", "NVDA", "META", "TSLA"]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe,
        panel=panel,
        mock_mode=True,
    )
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == set(universe)
    # Z-scores should have mean ≈ 0
    assert abs(sig.mean()) < 1.0
    # All non-NaN given panel covers all tickers
    assert sig.notna().all()


def test_mock_mode_is_deterministic() -> None:
    """Same (universe, as_of) inputs → same z-scores across runs."""
    universe = ["AAPL", "MSFT", "GOOG", "NVDA", "META"]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig1 = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel, mock_mode=True,
    )
    sig2 = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel, mock_mode=True,
    )
    pd.testing.assert_series_equal(sig1, sig2)


def test_mock_mode_different_as_of_gives_different_signals() -> None:
    """Same universe, different as_of → independent signals (seed varies)."""
    universe = ["AAPL", "MSFT", "GOOG", "NVDA", "META"]
    panel_jun = _make_panel(universe, datetime.date(2024, 6, 28))
    panel_sep = _make_panel(universe, datetime.date(2024, 9, 30))
    sig_jun = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel_jun, mock_mode=True,
    )
    sig_sep = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 9, 30),
        universe=universe, panel=panel_sep, mock_mode=True,
    )
    # Should not be identical
    assert not sig_jun.equals(sig_sep)


# ── Cross-section z-score correctness ───────────────────────────────────────
def test_zscore_mean_zero_std_one_for_balanced_universe() -> None:
    """With 10 tickers, z-scores should have ~ mean 0, std 1."""
    universe = [f"T{i:02d}" for i in range(10)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel, mock_mode=True,
    )
    # With sample std (ddof=1) and n=10, mean and std should be computable
    valid = sig.dropna()
    assert len(valid) == 10
    # Sample z-score has mean 0 by construction
    assert abs(valid.mean()) < 1e-9
    # Sample std with ddof=1 is exactly 1 by construction (z = (x-mean)/std)
    assert abs(valid.std(ddof=1) - 1.0) < 1e-9


def test_min_universe_gate_returns_all_nan_below_5() -> None:
    """< 5 valid tickers → all-NaN (mirror dividend_yield gate)."""
    universe = ["A", "B", "C"]   # only 3 < 5
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel, mock_mode=True,
    )
    assert sig.isna().all()


# ── NaN propagation ─────────────────────────────────────────────────────────
def test_ticker_missing_from_panel_returns_nan() -> None:
    """If a ticker isn't in panel.columns, its z-score is NaN."""
    universe = ["AAPL", "MSFT", "GOOG", "MISSING", "NVDA", "META"]
    panel = _make_panel(["AAPL", "MSFT", "GOOG", "NVDA", "META"],
                         datetime.date(2024, 6, 28))
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel, mock_mode=True,
    )
    assert pd.isna(sig["MISSING"])
    assert sig.dropna().shape[0] == 5  # other 5 valid


def test_empty_universe_returns_empty_series() -> None:
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=[], panel=_make_panel(["X"], datetime.date(2024, 6, 28)),
        mock_mode=True,
    )
    assert sig.empty


def test_empty_panel_returns_all_nan() -> None:
    universe = ["AAPL", "MSFT", "GOOG", "NVDA", "META"]
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=pd.DataFrame(), mock_mode=True,
    )
    assert sig.isna().all()


def test_invalid_as_of_type_raises() -> None:
    with pytest.raises(TypeError, match="as_of must be datetime.date"):
        compute_value_pe_singlestock_signal(
            as_of="2024-06-28",  # type: ignore
            universe=["A"], panel=pd.DataFrame(), mock_mode=True,
        )


# ── Auto-mode routing ───────────────────────────────────────────────────────
def test_auto_mode_falls_back_to_mock_when_wrds_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.universe_singlename import crsp_loader as cl
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)
    universe = [f"T{i:02d}" for i in range(6)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_value_pe_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel,
        # mock_mode unspecified → auto-detect
    )
    assert sig.notna().all()  # mock fallback ran successfully


# ── Real-path stub error handling ───────────────────────────────────────────
def test_real_mode_with_no_wrds_raises_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.universe_singlename import crsp_loader as cl
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)
    universe = [f"T{i:02d}" for i in range(6)]
    panel = _make_panel(universe, datetime.date(2024, 6, 28))
    with pytest.raises(RuntimeError, match="WRDS not configured"):
        compute_value_pe_singlestock_signal(
            as_of=datetime.date(2024, 6, 28),
            universe=universe, panel=panel, mock_mode=False,
        )


def test_real_mode_no_longer_stubbed_post_activation() -> None:
    """Post-2026-05-11 (Wave B activation): real Compustat E/P path is implemented.
    Guards against stub regression. Live WRDS integration covered via smoke
    scripts, not unit tests (keeps test suite offline-capable)."""
    from engine.factors_singlename.value_pe import _real_value_pe_signal
    import inspect
    src = inspect.getsource(_real_value_pe_signal)
    assert "NotImplementedError" not in src, (
        "_real_value_pe_signal regressed to stub state. Post-Wave-B-activation, "
        "this function must be implemented."
    )


# ── API parity with dividend_yield (Wave A counterpart) ─────────────────────
def test_api_signature_parity_with_dividend_yield() -> None:
    """value_pe must be drop-in replaceable with dividend_yield in the
    factor ensemble — required positional + keyword args must align."""
    from engine.factors_singlename.dividend_yield import (
        compute_dividend_yield_singlestock_signal,
    )
    a_sig = inspect.signature(compute_dividend_yield_singlestock_signal)
    b_sig = inspect.signature(compute_value_pe_singlestock_signal)
    common = {"as_of", "universe", "asset_classes", "panel"}
    assert common.issubset(set(a_sig.parameters))
    assert common.issubset(set(b_sig.parameters))
