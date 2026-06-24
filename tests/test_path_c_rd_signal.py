"""
tests/test_path_c_rd_signal.py — Sprint I-3 R&D signal composition tests.

Pre-registration: docs/spec_path_i_rd_premium_drift_v1.md (id=59) §2.3 + §2.4
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from engine.path_c.rd_signal import (
    INTENSITY_WEIGHT_SCALE,
    compute_rd_signal,
    build_rd_signal_panel,
)
from engine.path_c.rd_signal_panel import (
    R_AND_D_MIN_DISCLOSED_QUARTERS,
    R_AND_D_MIN_DOLLAR_M,
    _mock_rd_panel,
)


def _make_panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype({
        "r_and_d_4q_recent":  float,
        "r_and_d_4q_prior":   float,
        "n_quarters_recent":  int,
        "n_quarters_prior":   int,
        "atq":                float,
    })


def test_locked_intensity_weight_scale():
    assert INTENSITY_WEIGHT_SCALE == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_rd_signal formula
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_no_growth_positive_intensity():
    """recent = prior (no growth) → rd_growth = 0 → signal = 0 regardless of intensity."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 100.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4, "n_quarters_prior": 4,
        "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    assert out["rd_signal"].iloc[0] == pytest.approx(0.0)


def test_signal_positive_growth():
    """recent 2x prior → rd_growth = 1.0; signal positive."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 200.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4, "n_quarters_prior": 4,
        "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    # rd_growth = 1.0; intensity = 200/10000 = 0.02; weight = log(1+2)=1.0986
    expected = 1.0 * np.log1p(0.02 * 100.0)
    assert out["rd_signal"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_signal_negative_growth():
    """recent = 0.5 × prior → rd_growth = -0.5."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 50.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4, "n_quarters_prior": 4,
        "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    # rd_growth = -0.5, intensity = 50/10000 = 0.005, weight = log(1+0.5)=0.405
    expected = -0.5 * np.log1p(0.005 * 100.0)
    assert out["rd_signal"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_signal_intensity_weight_higher_for_rd_heavy_firms():
    """Two firms same growth, different intensity → R&D-heavy firm has larger |signal|."""
    p = _make_panel([
        {"ticker": "TECH", "fiscal_yearq": "Q1",
         "r_and_d_4q_recent": 1000.0, "r_and_d_4q_prior": 500.0,  # 100% growth, intensity 10%
         "n_quarters_recent": 4, "n_quarters_prior": 4, "atq": 10000.0},
        {"ticker": "OLD", "fiscal_yearq": "Q1",
         "r_and_d_4q_recent": 10.0, "r_and_d_4q_prior": 5.0,  # 100% growth, intensity 0.1%
         "n_quarters_recent": 4, "n_quarters_prior": 4, "atq": 10000.0},
    ])
    out = compute_rd_signal(p)
    tech_sig = out[out["ticker"] == "TECH"]["rd_signal"].iloc[0]
    old_sig  = out[out["ticker"] == "OLD"]["rd_signal"].iloc[0]
    assert tech_sig > old_sig   # R&D-heavy firm signal stronger at same growth rate


def test_signal_thin_recent_quarters():
    """n_quarters_recent < MIN_DISCLOSED → NaN."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 100.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": R_AND_D_MIN_DISCLOSED_QUARTERS - 1,
        "n_quarters_prior": 4, "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    assert np.isnan(out["rd_signal"].iloc[0])


def test_signal_thin_prior_quarters():
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 100.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4,
        "n_quarters_prior": R_AND_D_MIN_DISCLOSED_QUARTERS - 1,
        "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    assert np.isnan(out["rd_signal"].iloc[0])


def test_signal_low_dollar_recent():
    """r_and_d_4q_recent < $1M → NaN."""
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": R_AND_D_MIN_DOLLAR_M - 0.01,
        "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4, "n_quarters_prior": 4, "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    assert np.isnan(out["rd_signal"].iloc[0])


def test_signal_low_dollar_prior():
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 100.0,
        "r_and_d_4q_prior": R_AND_D_MIN_DOLLAR_M - 0.01,
        "n_quarters_recent": 4, "n_quarters_prior": 4, "atq": 10000.0,
    }])
    out = compute_rd_signal(p)
    assert np.isnan(out["rd_signal"].iloc[0])


def test_signal_invalid_atq():
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 100.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4, "n_quarters_prior": 4, "atq": 0.0,
    }])
    out = compute_rd_signal(p)
    assert np.isnan(out["rd_signal"].iloc[0])


def test_signal_empty_panel():
    p = pd.DataFrame(columns=[
        "r_and_d_4q_recent", "r_and_d_4q_prior",
        "n_quarters_recent", "n_quarters_prior", "atq",
    ])
    out = compute_rd_signal(p)
    assert out.empty
    assert "rd_signal" in out.columns


def test_signal_missing_columns_raises():
    p = pd.DataFrame([{"r_and_d_4q_recent": 100.0}])
    with pytest.raises(ValueError, match="missing columns"):
        compute_rd_signal(p)


def test_signal_does_not_mutate_input():
    p = _make_panel([{
        "ticker": "A", "fiscal_yearq": "Q1",
        "r_and_d_4q_recent": 100.0, "r_and_d_4q_prior": 100.0,
        "n_quarters_recent": 4, "n_quarters_prior": 4, "atq": 10000.0,
    }])
    cols_before = list(p.columns)
    _ = compute_rd_signal(p)
    assert list(p.columns) == cols_before


# ─────────────────────────────────────────────────────────────────────────────
# build_rd_signal_panel end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_build_pipeline_with_mock_panel():
    mock = _mock_rd_panel(
        tickers=[f"T{i:03d}" for i in range(100)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
    )
    out = build_rd_signal_panel(mock)
    assert {"rd_signal", "sue_rank_pct", "leg"}.issubset(set(out.columns))
    leg_counts = out["leg"].value_counts()
    assert leg_counts.get("long", 0) > 0
    assert leg_counts.get("short", 0) > 0
    assert leg_counts.get("flat", 0) > 0


def test_build_pipeline_deterministic():
    mock = _mock_rd_panel(
        tickers=[f"T{i:03d}" for i in range(30)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 6, 30),
    )
    o1 = build_rd_signal_panel(mock)
    o2 = build_rd_signal_panel(mock)
    pd.testing.assert_frame_equal(o1, o2)


def test_build_pipeline_high_growth_in_long_leg():
    """Firms with highest R&D growth get long leg."""
    rows = []
    for i in range(100):
        # T099 has highest growth
        rows.append({
            "ticker": f"T{i:03d}",
            "fiscal_yearq": "Q1",
            "r_and_d_4q_recent": 100.0 + i * 10.0,  # growth from 0 to +900%
            "r_and_d_4q_prior": 100.0,
            "n_quarters_recent": 4, "n_quarters_prior": 4,
            "atq": 10000.0,
        })
    p = _make_panel(rows)
    out = build_rd_signal_panel(p)
    top_10 = set(f"T{i:03d}" for i in range(90, 100))
    long_tickers = set(out[out["leg"] == "long"]["ticker"])
    assert top_10 == long_tickers
