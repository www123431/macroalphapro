"""tests/test_fetch_fx_spot_g10.py — G10 FX spot fetcher tests.

Offline tests for the fetcher infrastructure. Network calls NOT
tested here (integration only). Tests cover:
  - _to_month_end_close resampling
  - FRED_FX_SERIES structural contract
  - direct vs indirect quote convention table
  - integration on real parquet when cached
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import fetch_fx_spot_g10 as fx  # noqa: E402


def test_to_month_end_close_takes_last_valid_per_month():
    daily = pd.Series(
        [110.0, 111.5, 112.0, 113.0, 113.5],
        index=pd.to_datetime([
            "2024-01-02", "2024-01-15", "2024-01-31",
            "2024-02-01", "2024-02-29",
        ]),
    )
    monthly = fx._to_month_end_close(daily)
    assert len(monthly) == 2
    assert monthly.loc["2024-01-31"] == 112.0
    assert monthly.loc["2024-02-29"] == 113.5


def test_to_month_end_close_handles_holiday_nans():
    daily = pd.Series(
        [110.0, np.nan, 113.5],
        index=pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-29"]),
    )
    monthly = fx._to_month_end_close(daily)
    assert len(monthly) == 2
    assert monthly.loc["2024-01-31"] == 110.0  # NaN dropped; last valid wins
    assert monthly.loc["2024-02-29"] == 113.5


def test_g10_series_count_and_currencies():
    """G10 = 10 currencies. Anti-regression for accidental drop."""
    series = fx.FRED_FX_SERIES
    assert len(series) == 10
    currencies = {ccy for _, ccy, _ in series}
    assert currencies == {
        "JPY", "EUR", "GBP", "CHF", "CAD",
        "AUD", "NZD", "SEK", "NOK", "DKK",
    }


def test_quote_convention_table_consistent():
    """Direct quote (is_direct=True) means series IS already USD per
    FCY. Indirect (False) means FCY per USD. EUR / GBP / AUD / NZD
    use DEXUS_ prefix (direct from US); others use DEX_US which is
    indirect.

    This test pins the convention table — if someone flips a flag
    by mistake, log returns get inverted and HML_FX construction
    later would silently use sign-flipped FX moves."""
    convention = {(sid, ccy): is_direct
                    for sid, ccy, is_direct in fx.FRED_FX_SERIES}
    # DEXUS_ series are DIRECT (USD per FCY)
    assert convention[("DEXUSEU", "EUR")] is True
    assert convention[("DEXUSUK", "GBP")] is True
    assert convention[("DEXUSAL", "AUD")] is True
    assert convention[("DEXUSNZ", "NZD")] is True
    # DEX_US series are INDIRECT (FCY per USD)
    assert convention[("DEXJPUS", "JPY")] is False
    assert convention[("DEXSZUS", "CHF")] is False
    assert convention[("DEXCAUS", "CAD")] is False
    assert convention[("DEXSDUS", "SEK")] is False
    assert convention[("DEXNOUS", "NOK")] is False
    assert convention[("DEXDNUS", "DKK")] is False


# ────────────────────────────────────────────────────────────────────
# Integration: real cached parquet
# ────────────────────────────────────────────────────────────────────
def test_real_fx_parquet_shape_if_present():
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "fx_spot_g10_monthly.parquet")
    if not p.exists():
        pytest.skip("FX parquet not cached — run fetcher")
    df = pd.read_parquet(p)
    # 10 spot columns + 10 logret columns + date = 21 columns
    expected_spot = {f"spot_{ccy}_per_USD" for _, ccy, _
                       in fx.FRED_FX_SERIES}
    expected_logret = {f"logret_{ccy}" for _, ccy, _ in fx.FRED_FX_SERIES}
    expected = {"date"} | expected_spot | expected_logret
    assert set(df.columns) == expected
    # EUR start 1999-01 → first valid month with log return is 1999-02
    assert df["date"].iloc[0] >= pd.Timestamp("1999-01-31")
    assert len(df) > 300   # 25+ years of monthly
    # Log returns sanity: monthly FX vol typically 2-4%
    for c in expected_logret:
        v = df[c].std()
        assert 0.015 < v < 0.05, f"{c} log return vol {v} outside [1.5%, 5%]"
    # No NaN in production data (inner-joined)
    for c in df.columns:
        if c == "date":
            continue
        assert df[c].notna().all(), f"{c} has NaN"


def test_real_fx_parquet_currency_strength_sanity():
    """Validate quote direction normalization: CHF should have
    POSITIVE long-run mean log return vs USD (safe haven). JPY
    should be ~zero to slightly negative (BoJ regime). These are
    well-known empirical patterns; if they're sign-flipped, the
    convention table is bugged."""
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "fx_spot_g10_monthly.parquet")
    if not p.exists():
        pytest.skip("FX parquet not cached")
    df = pd.read_parquet(p)
    # CHF over 1999-2026: well-documented safe-haven appreciation
    chf_mean = df["logret_CHF"].mean()
    assert chf_mean > 0, (
        f"CHF mean log return {chf_mean} should be positive over "
        f"1999-2026 (safe-haven). If negative, quote convention "
        f"DEXSZUS=indirect flag is wrong (sign-flipped)."
    )
    # JPY over 1999-2026: BoJ low-rate; carry funding currency;
    # should be SMALL (near zero, slightly negative)
    jpy_mean = df["logret_JPY"].mean()
    assert -0.005 < jpy_mean < 0.005, (
        f"JPY mean log return {jpy_mean} outside [-0.5%, +0.5%] mo "
        f"band — sanity check on quote convention failed."
    )
