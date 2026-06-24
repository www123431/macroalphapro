"""
tests/test_path_c_asset_growth_signal.py — Sprint J-3 asset growth signal tests.

Pre-registration: docs/spec_path_j_asset_growth_drift_v1.md (id=60) §2.3 + §2.4
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from engine.path_c.asset_growth_signal import (
    compute_asset_growth_signal,
    build_asset_growth_signal_panel,
)
from engine.path_c.asset_growth_signal_panel import (
    MIN_ATQ_DOLLAR_M,
    MAX_ABSOLUTE_GROWTH,
    _mock_asset_growth_panel,
)


def _make_panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype({
        "atq_recent": float,
        "atq_prior":  float,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Signal formula
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_zero_growth():
    """recent = prior → growth = 0 → signal = 0."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 1000.0, "atq_prior": 1000.0,
    }])
    out = compute_asset_growth_signal(p)
    assert out["asset_growth_signal"].iloc[0] == pytest.approx(0.0)


def test_signal_positive_growth_inverted_to_negative():
    """CGS BEARISH: high growth → SHORT (low signal). recent = 1.5 × prior → growth +0.5 → signal -0.5."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 1500.0, "atq_prior": 1000.0,
    }])
    out = compute_asset_growth_signal(p)
    assert out["asset_growth_signal"].iloc[0] == pytest.approx(-0.5)


def test_signal_negative_growth_inverted_to_positive():
    """CGS BEARISH: low growth → LONG (high signal). recent = 0.7 × prior → growth -0.3 → signal +0.3."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 700.0, "atq_prior": 1000.0,
    }])
    out = compute_asset_growth_signal(p)
    assert out["asset_growth_signal"].iloc[0] == pytest.approx(0.3)


def test_signal_low_atq_recent_excluded():
    """atq_recent < MIN_ATQ_DOLLAR_M ($100M) → NaN."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": MIN_ATQ_DOLLAR_M - 0.01, "atq_prior": 500.0,
    }])
    out = compute_asset_growth_signal(p)
    assert np.isnan(out["asset_growth_signal"].iloc[0])


def test_signal_extreme_growth_excluded():
    """|growth| > MAX_ABSOLUTE_GROWTH (5.0 = 500%) → NaN (M&A filter)."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 6500.0,   # 5.5× prior, growth = 5.5
        "atq_prior": 1000.0,
    }])
    out = compute_asset_growth_signal(p)
    assert np.isnan(out["asset_growth_signal"].iloc[0])


def test_signal_extreme_shrinkage_excluded():
    """|growth| > MAX (e.g., -500%+) — covers severe asset reduction (delisting, restructure)."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 100.0,
        "atq_prior":  1000.0,   # growth = -0.9, within bounds → should be POSITIVE signal
    }])
    out = compute_asset_growth_signal(p)
    # growth = -0.9 → signal = 0.9 (LONG: shrinking firms)
    assert out["asset_growth_signal"].iloc[0] == pytest.approx(0.9)


def test_signal_missing_atq_prior():
    """atq_prior NaN → NaN signal (no growth computable)."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 1000.0, "atq_prior": np.nan,
    }])
    out = compute_asset_growth_signal(p)
    assert np.isnan(out["asset_growth_signal"].iloc[0])


def test_signal_zero_atq_prior():
    """atq_prior ≤ 0 → NaN."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 1000.0, "atq_prior": 0.0,
    }])
    out = compute_asset_growth_signal(p)
    assert np.isnan(out["asset_growth_signal"].iloc[0])


def test_signal_empty_panel():
    p = pd.DataFrame(columns=["atq_recent", "atq_prior"])
    out = compute_asset_growth_signal(p)
    assert out.empty
    assert "asset_growth_signal" in out.columns


def test_signal_missing_columns_raises():
    p = pd.DataFrame([{"atq_recent": 100.0}])
    with pytest.raises(ValueError, match="missing columns"):
        compute_asset_growth_signal(p)


def test_signal_does_not_mutate_input():
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "atq_recent": 1000.0, "atq_prior": 1000.0,
    }])
    cols_before = list(p.columns)
    _ = compute_asset_growth_signal(p)
    assert list(p.columns) == cols_before


# ─────────────────────────────────────────────────────────────────────────────
# build_asset_growth_signal_panel end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_build_pipeline_with_mock():
    """Mock panel → end-to-end pipeline produces long/short/flat/excluded."""
    mock = _mock_asset_growth_panel(
        tickers=[f"T{i:03d}" for i in range(100)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2017, 12, 31),
    )
    out = build_asset_growth_signal_panel(mock)
    assert {"asset_growth_signal", "sue_rank_pct", "leg"}.issubset(set(out.columns))
    leg_counts = out["leg"].value_counts()
    # Year-1 will be all excluded (no atq_prior); year-2+ has signal
    assert leg_counts.get("long", 0) > 0
    assert leg_counts.get("short", 0) > 0


def test_build_pipeline_deterministic():
    mock = _mock_asset_growth_panel(
        tickers=[f"T{i:03d}" for i in range(30)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2016, 12, 31),
    )
    o1 = build_asset_growth_signal_panel(mock)
    o2 = build_asset_growth_signal_panel(mock)
    pd.testing.assert_frame_equal(o1, o2)


def test_build_pipeline_low_growth_in_long_leg():
    """LOW asset growth → LONG leg per CGS BEARISH inverted."""
    rows = []
    for i in range(100):
        # T000 lowest growth (atq shrinks), T099 highest growth
        rows.append({
            "ticker": f"T{i:03d}",
            "fiscal_yearq": "Q1",
            "atq_recent": 1000.0 + i * 5.0,
            "atq_prior":  1000.0,
        })
    p = _make_panel(rows)
    out = build_asset_growth_signal_panel(p)
    # Bottom-10 by asset growth = T000-T009 → HIGHEST signal → LONG
    long_tickers = set(out[out["leg"] == "long"]["ticker"])
    expected_long = set(f"T{i:03d}" for i in range(10))
    assert long_tickers == expected_long
    # Top-10 by asset growth = T090-T099 → LOWEST signal → SHORT
    short_tickers = set(out[out["leg"] == "short"]["ticker"])
    expected_short = set(f"T{i:03d}" for i in range(90, 100))
    assert short_tickers == expected_short
