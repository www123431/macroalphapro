"""tests/test_fetch_g10_short_rates.py — G10 short rate fetcher tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import fetch_g10_short_rates as rates  # noqa: E402


def test_to_month_end_shifts_to_month_end():
    """FRED IR3TIB has month-start dates; we shift to month-end."""
    raw = pd.Series(
        [2.5, 2.6, 2.7],
        index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
    )
    out = rates._to_month_end(raw)
    assert list(out.index) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-31"),
    ]


def test_g10_series_count_includes_usd():
    """11 series total: G10 + USD itself (USD is the base)."""
    assert len(rates.FRED_RATE_SERIES) == 11
    ccys = {ccy for _, ccy in rates.FRED_RATE_SERIES}
    assert ccys == {
        "USD", "EUR", "JPY", "GBP", "CHF", "CAD",
        "AUD", "NZD", "SEK", "NOK", "DKK",
    }


def test_real_rates_parquet_shape_if_present():
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "g10_short_rates_monthly.parquet")
    if not p.exists():
        pytest.skip("rates parquet not cached — run fetcher")
    df = pd.read_parquet(p)
    # 11 rate columns + 10 rdiff columns + date = 22
    assert len(df.columns) == 22
    rate_cols = [c for c in df.columns if c.startswith("rate_")]
    rdiff_cols = [c for c in df.columns if c.startswith("rdiff_")]
    assert len(rate_cols) == 11
    assert len(rdiff_cols) == 10   # USD itself omitted (= 0 by def)
    # No NaN (inner-joined)
    for c in df.columns:
        if c == "date":
            continue
        assert df[c].notna().all(), f"{c} has NaN"


def test_real_rates_textbook_carry_sort():
    """Sanity: AUD/NZD should have POSITIVE mean rate differential
    vs USD (high-yield carry destinations). JPY/CHF should be
    NEGATIVE (funding currencies). This is the canonical
    Lustig-Roussanov-Verdelhan 2011 / Menkhoff et al. 2012 sort."""
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "g10_short_rates_monthly.parquet")
    if not p.exists():
        pytest.skip("rates parquet not cached")
    df = pd.read_parquet(p)
    # Carry destinations (long leg)
    assert df["rdiff_NZD_pct"].mean() > 1.0  # textbook NZD high yield
    assert df["rdiff_AUD_pct"].mean() > 1.0  # textbook AUD high yield
    # Funding currencies (short leg)
    assert df["rdiff_JPY_pct"].mean() < -1.0  # textbook JPY low yield
    assert df["rdiff_CHF_pct"].mean() < -1.0  # textbook CHF low yield
    # NZD should be the highest carry; CHF should be the lowest (typically)
    rdiff_cols = [c for c in df.columns if c.startswith("rdiff_")]
    means = {c.replace("rdiff_", "").replace("_pct", ""): df[c].mean()
                for c in rdiff_cols}
    sorted_by_carry = sorted(means.items(), key=lambda kv: -kv[1])
    high_carry = [c for c, _ in sorted_by_carry[:3]]
    low_carry  = [c for c, _ in sorted_by_carry[-3:]]
    assert "NZD" in high_carry  # well-documented carry destination
    assert "JPY" in low_carry   # well-documented funding currency
    assert "CHF" in low_carry   # well-documented funding currency


def test_rate_differentials_are_consistent_with_rates():
    """For each currency != USD, rdiff = rate_ccy - rate_USD by
    construction. Anti-regression for bug in differential computation."""
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "g10_short_rates_monthly.parquet")
    if not p.exists():
        pytest.skip("rates parquet not cached")
    df = pd.read_parquet(p)
    usd = df["rate_USD_pct"]
    for c in df.columns:
        if not c.startswith("rdiff_"):
            continue
        ccy = c.replace("rdiff_", "").replace("_pct", "")
        rate_col = f"rate_{ccy}_pct"
        if rate_col not in df.columns:
            continue
        # Within tiny floating-point tolerance
        diff = (df[rate_col] - usd) - df[c]
        assert diff.abs().max() < 1e-9, (
            f"rdiff_{ccy} != rate_{ccy} - rate_USD (max diff {diff.abs().max()})"
        )
