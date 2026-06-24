"""
tests/test_path_c_labor_signal.py — Sprint G3 labor signal composition tests.

Pre-registration: docs/spec_path_c_labor_signal_drift_v1.md (id=58) §2.3 + §2.4
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from engine.path_c.labor_signal import (
    LAYOFF_WEIGHT_LOCKED,
    compute_labor_signal,
    build_labor_signal_panel,
)
from engine.path_c.labor_signal_panel import (
    MIN_L6_POSTINGS_REQUIRED,
    MIN_B12_POSTINGS_REQUIRED,
    _mock_labor_panel,
)


def _make_panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype({
        "l6_postings_count":  int,
        "b12_postings_count": int,
        "layoff_flag":        int,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_layoff_weight():
    assert LAYOFF_WEIGHT_LOCKED == 0.5   # spec §2.3


# ─────────────────────────────────────────────────────────────────────────────
# compute_labor_signal formula correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_zero_growth_no_layoff_is_zero():
    """L6 = B12/2 (no growth) and no layoff → signal = 0."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 50, "b12_postings_count": 100, "layoff_flag": 0,
    }])
    out = compute_labor_signal(p)
    assert out["labor_signal"].iloc[0] == pytest.approx(0.0)


def test_signal_positive_growth():
    """L6 = B12 (2x baseline) → growth = 1.0 (doubled hiring)."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 100, "b12_postings_count": 100, "layoff_flag": 0,
    }])
    out = compute_labor_signal(p)
    # L6=100, B12/2=50 → (100-50)/50 = 1.0
    assert out["labor_signal"].iloc[0] == pytest.approx(1.0)


def test_signal_negative_growth():
    """L6 = B12/4 (halved hiring) → growth = -0.5."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 25, "b12_postings_count": 100, "layoff_flag": 0,
    }])
    out = compute_labor_signal(p)
    # L6=25, B12/2=50 → (25-50)/50 = -0.5
    assert out["labor_signal"].iloc[0] == pytest.approx(-0.5)


def test_signal_layoff_penalizes_by_half():
    """layoff_flag = 1 subtracts 0.5 from growth."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 100, "b12_postings_count": 100, "layoff_flag": 1,
    }])
    out = compute_labor_signal(p)
    # growth = 1.0 then - 0.5 layoff = 0.5
    assert out["labor_signal"].iloc[0] == pytest.approx(0.5)


def test_signal_thin_l6_returns_nan():
    """L6 < MIN_L6_POSTINGS_REQUIRED → NaN."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": MIN_L6_POSTINGS_REQUIRED - 1,
        "b12_postings_count": 50, "layoff_flag": 0,
    }])
    out = compute_labor_signal(p)
    assert np.isnan(out["labor_signal"].iloc[0])


def test_signal_thin_b12_returns_nan():
    """B12 < MIN_B12_POSTINGS_REQUIRED → NaN."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 50,
        "b12_postings_count": MIN_B12_POSTINGS_REQUIRED - 1, "layoff_flag": 0,
    }])
    out = compute_labor_signal(p)
    assert np.isnan(out["labor_signal"].iloc[0])


def test_signal_empty_panel():
    p = pd.DataFrame(columns=["l6_postings_count", "b12_postings_count", "layoff_flag"])
    out = compute_labor_signal(p)
    assert out.empty
    assert "labor_signal" in out.columns


def test_signal_missing_columns_raises():
    p = pd.DataFrame([{"l6_postings_count": 50, "b12_postings_count": 50}])
    with pytest.raises(ValueError, match="missing columns"):
        compute_labor_signal(p)


def test_signal_does_not_mutate_input():
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 50, "b12_postings_count": 100, "layoff_flag": 0,
    }])
    cols_before = list(p.columns)
    _ = compute_labor_signal(p)
    assert list(p.columns) == cols_before


def test_signal_custom_layoff_weight():
    """Override default layoff_weight."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "l6_postings_count": 100, "b12_postings_count": 100, "layoff_flag": 1,
    }])
    out = compute_labor_signal(p, layoff_weight=1.0)
    # growth = 1.0, layoff penalty = 1.0 × 1 = 1.0 → signal = 0.0
    assert out["labor_signal"].iloc[0] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# build_labor_signal_panel (end-to-end with mock panel)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_pipeline_with_mock_panel_produces_legs():
    """G2 mock panel → G3 build_labor_signal_panel produces long/short/flat legs."""
    mock = _mock_labor_panel(
        tickers=[f"T{i:03d}" for i in range(100)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
    )
    out = build_labor_signal_panel(mock)
    assert {"labor_signal", "sue_rank_pct", "leg"}.issubset(set(out.columns))
    leg_counts = out["leg"].value_counts()
    # With 100 firms / quarter, ~10 long + ~10 short + ~80 flat
    assert leg_counts.get("long", 0) > 0
    assert leg_counts.get("short", 0) > 0
    assert leg_counts.get("flat", 0) > 0


def test_build_pipeline_deterministic():
    """Same input → same output."""
    mock = _mock_labor_panel(
        tickers=[f"T{i:03d}" for i in range(30)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 6, 30),
    )
    o1 = build_labor_signal_panel(mock)
    o2 = build_labor_signal_panel(mock)
    pd.testing.assert_frame_equal(o1, o2)


def test_build_pipeline_excludes_thin_coverage_firms():
    """Firms with very low postings → NaN signal → leg=excluded."""
    rows = []
    for i in range(20):
        # Half normal, half thin
        if i < 10:
            l6, b12 = 100, 200
        else:
            l6, b12 = 2, 5   # below thresholds
        rows.append({
            "ticker": f"T{i:03d}",
            "fiscal_yearq": "2014Q1",
            "l6_postings_count": l6,
            "b12_postings_count": b12,
            "layoff_flag": 0,
        })
    p = _make_panel(rows)
    out = build_labor_signal_panel(p)
    excluded = out[out["leg"] == "excluded"]
    assert len(excluded) == 10
    assert all(out[out["ticker"].str.startswith("T01")]["leg"] == "excluded")


def test_build_pipeline_long_top_decile_high_growth():
    """Firms with highest growth get long leg."""
    # 100 firms with monotone increasing L6
    rows = []
    for i in range(100):
        rows.append({
            "ticker": f"T{i:03d}",
            "fiscal_yearq": "Q1",
            "l6_postings_count": 100 + i,   # T099 has highest growth
            "b12_postings_count": 100,
            "layoff_flag": 0,
        })
    p = _make_panel(rows)
    out = build_labor_signal_panel(p)
    # Top 10 by labor_signal (T090..T099) → long
    top_10_tickers = set(f"T{i:03d}" for i in range(90, 100))
    long_tickers = set(out[out["leg"] == "long"]["ticker"])
    assert top_10_tickers == long_tickers
