"""
tests/test_strev.py — unit tests for engine.factors_singlename.strev (F-LAB-E3).

Coverage:
  - Locked constants
  - Math correctness (controlled past-return panels)
  - Cross-section z-score properties
  - NaN propagation
  - API parity with Wave A factors
  - Registration into FACTOR_REGISTRY_SINGLENAME at module load
"""
from __future__ import annotations

import datetime
import inspect

import numpy as np
import pandas as pd
import pytest

from engine.factors_singlename import strev
from engine.factors_singlename.strev import (
    MIN_UNIVERSE_FOR_ZSCORE_LOCKED,
    STREV_LOOKBACK_DAYS_LOCKED,
    STREV_MIN_OBS_RATIO_LOCKED,
    compute_strev_singlestock_signal,
)


# ── Locked constants ────────────────────────────────────────────────────────
def test_locked_constants_match_jegadeesh_1990() -> None:
    assert STREV_LOOKBACK_DAYS_LOCKED == 21    # 1 trading month
    assert STREV_MIN_OBS_RATIO_LOCKED == 0.5
    assert MIN_UNIVERSE_FOR_ZSCORE_LOCKED == 5


# ── Helper: panel with controlled cumulative returns ────────────────────────
def _make_panel_with_known_returns(
    ticker_to_total_return: dict[str, float],
    as_of: datetime.date,
    n_days: int = 45,
) -> pd.DataFrame:
    """Build a panel where each ticker has a known cumulative return over the
    last ~21 trading days (linear price path for simplicity)."""
    dates = pd.bdate_range(end=pd.Timestamp(as_of) - pd.Timedelta(days=1), periods=n_days, freq="B")
    panel = pd.DataFrame(index=dates, dtype=float)
    for ticker, total_ret in ticker_to_total_return.items():
        # Linear path: start_price = 100, end_price = 100 * (1 + total_ret)
        end_price = 100.0 * (1.0 + total_ret)
        panel[ticker] = np.linspace(100.0, end_price, n_days)
    return panel


# ── Happy path ──────────────────────────────────────────────────────────────
def test_strev_returns_zscore_correct_index() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    panel = _make_panel_with_known_returns(
        {t: 0.05 for t in universe}, datetime.date(2024, 6, 28),
    )
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel,
    )
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == set(universe)
    # All same return → all same raw → after z-score all NaN (std=0)
    # (or all 0 — depending on impl; current impl returns NaN when std≈0)
    # Actually: if all 8 tickers have identical raw return, std = 0 → all NaN
    assert sig.isna().all()


def test_strev_zscore_n10_sample_properties() -> None:
    """10 tickers with varied past returns → z-score mean=0 std=1."""
    rng = np.random.default_rng(11)
    universe = [f"T{i:02d}" for i in range(10)]
    returns = rng.uniform(-0.10, +0.10, 10)
    panel = _make_panel_with_known_returns(
        dict(zip(universe, returns)), datetime.date(2024, 6, 28),
    )
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    valid = sig.dropna()
    assert len(valid) == 10
    assert abs(valid.mean()) < 1e-9
    assert abs(valid.std(ddof=1) - 1.0) < 1e-9


def test_strev_deterministic() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    rng = np.random.default_rng(33)
    returns = rng.uniform(-0.05, +0.05, 8)
    panel = _make_panel_with_known_returns(
        dict(zip(universe, returns)), datetime.date(2024, 6, 28),
    )
    s1 = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    s2 = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    pd.testing.assert_series_equal(s1, s2)


# ── Math correctness ────────────────────────────────────────────────────────
def test_strev_highest_past_return_has_highest_zscore() -> None:
    """Ticker with highest 1-month return should have the highest z-score
    (raw factor IS past return; expected_sign=-1 in registry handles direction)."""
    universe = ["LOSER", "MID1", "MID2", "MID3", "MID4", "WINNER"]
    panel = _make_panel_with_known_returns({
        "LOSER":  -0.10,
        "MID1":   -0.02,
        "MID2":    0.00,
        "MID3":   +0.01,
        "MID4":   +0.03,
        "WINNER": +0.15,
    }, datetime.date(2024, 6, 28))
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert sig["WINNER"] == sig.max()
    assert sig["LOSER"]  == sig.min()
    # WINNER raw return > LOSER raw return → z-score > 0 vs < 0
    assert sig["WINNER"] > 0
    assert sig["LOSER"]  < 0


def test_strev_median_return_has_smallest_abs_zscore() -> None:
    """Ticker with the median past return should have z-score closer to 0
    than the extremes — true regardless of multiplicative path asymmetry.

    Uses 7 tickers with monotonically increasing returns so the middle
    ticker (T03) has the median raw return; its abs z-score must be
    strictly less than both extremes (T00 and T06).
    """
    universe = [f"T{i:02d}" for i in range(7)]
    returns = {f"T{i:02d}": (-0.06 + 0.02 * i) for i in range(7)}
    panel = _make_panel_with_known_returns(returns, datetime.date(2024, 6, 28))
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert abs(sig["T03"]) < abs(sig["T00"])
    assert abs(sig["T03"]) < abs(sig["T06"])


# ── NaN propagation ─────────────────────────────────────────────────────────
def test_missing_ticker_returns_nan() -> None:
    universe = [f"T{i:02d}" for i in range(8)] + ["MISSING"]
    panel = _make_panel_with_known_returns(
        {f"T{i:02d}": 0.01 * (i - 3) for i in range(8)}, datetime.date(2024, 6, 28),
    )
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert pd.isna(sig["MISSING"])
    assert sig.dropna().shape[0] == 8


def test_min_universe_gate_below_5() -> None:
    universe = ["A", "B", "C"]   # only 3 < 5
    panel = _make_panel_with_known_returns(
        {t: 0.01 * i for i, t in enumerate(universe)}, datetime.date(2024, 6, 28),
    )
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert sig.isna().all()


def test_empty_universe_returns_empty() -> None:
    panel = _make_panel_with_known_returns({"X": 0.01}, datetime.date(2024, 6, 28))
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=[], panel=panel,
    )
    assert sig.empty


def test_empty_panel_returns_all_nan() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    sig = compute_strev_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=pd.DataFrame(),
    )
    assert sig.isna().all()


def test_invalid_as_of_type() -> None:
    with pytest.raises(TypeError, match="as_of must be datetime.date"):
        compute_strev_singlestock_signal(
            as_of="2024-06-28",  # type: ignore
            universe=["A"], panel=pd.DataFrame(),
        )


# ── API parity ──────────────────────────────────────────────────────────────
def test_api_parity_with_wave_a_factors() -> None:
    from engine.factors_singlename.dividend_yield import (
        compute_dividend_yield_singlestock_signal,
    )
    from engine.factors_singlename.bab import compute_bab_singlestock_signal
    from engine.factors_singlename.ivol import compute_ivol_singlestock_signal

    common = {"as_of", "universe", "asset_classes", "panel"}
    for name, fn in [
        ("dividend_yield", compute_dividend_yield_singlestock_signal),
        ("bab",            compute_bab_singlestock_signal),
        ("ivol",           compute_ivol_singlestock_signal),
        ("strev",          compute_strev_singlestock_signal),
    ]:
        sig = inspect.signature(fn)
        assert common.issubset(set(sig.parameters)), f"{name}: {sig}"


# ── Registration ────────────────────────────────────────────────────────────
def test_strev_registered_in_singlename_factor_library() -> None:
    from engine.factor_library_singlename import get_factor
    spec = get_factor("strev_singlestock")
    assert spec.factor_id == "strev_singlestock"
    assert spec.asset_class == "equity_singlename"
    assert spec.expected_sign == -1
    assert spec.signal_fn is compute_strev_singlestock_signal
    assert "Jegadeesh" in spec.citation


def test_strev_factor_id_in_list_factors() -> None:
    from engine.factor_library_singlename import list_factors
    assert "strev_singlestock" in list_factors()


def test_both_tier1_factors_registered() -> None:
    from engine.factor_library_singlename import list_factors
    factors = list_factors()
    assert "ivol_singlestock" in factors
    assert "strev_singlestock" in factors
