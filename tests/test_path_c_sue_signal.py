"""
tests/test_path_c_sue_signal.py — Sprint 3 SUE signal + decile-leg tests.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §2.3 + §2.4

Surface:
  - SUE formula correctness (= (actual - consensus_median) / dispersion)
  - NaN propagation through dispersion=0 / insufficient_analysts
  - Cross-sectional rank within quarter (multi-quarter independence)
  - Tie-break by ticker (deterministic per spec §N6)
  - Decile leg assignment (top 10% = long, bottom 10% = short, middle = flat)
  - Boundary INCLUSIVE per spec §N6
  - One-shot wrapper produces consistent end-to-end output
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from engine.path_c import (
    DECILE_LONG_THRESHOLD,
    DECILE_SHORT_THRESHOLD,
    MIN_ANALYSTS_REQUIRED,
)
from engine.path_c.sue_signal import (
    compute_sue,
    rank_within_quarter,
    assign_decile_legs,
    build_sue_signal_panel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic firm-quarter panel matching Sprint 2 schema
# ─────────────────────────────────────────────────────────────────────────────

def _make_panel(rows: list[dict]) -> pd.DataFrame:
    """Build minimal firm-quarter panel with required columns."""
    return pd.DataFrame(rows).astype({
        "actual_eps":           float,
        "consensus_median":     float,
        "consensus_dispersion": float,
        "n_analysts":           int,
    })


# ─────────────────────────────────────────────────────────────────────────────
# compute_sue: formula correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_sue_basic_positive_surprise():
    """SUE > 0 when actual exceeds consensus."""
    panel = _make_panel([{
        "ticker_ibes": "AAPL", "fiscal_yearq": "2014Q1",
        "actual_eps": 1.50, "consensus_median": 1.20, "consensus_dispersion": 0.10,
        "n_analysts": 10,
    }])
    out = compute_sue(panel)
    # SUE = (1.50 - 1.20) / 0.10 = 3.0
    assert out["sue"].iloc[0] == pytest.approx(3.0)


def test_compute_sue_basic_negative_surprise():
    """SUE < 0 when actual misses consensus."""
    panel = _make_panel([{
        "ticker_ibes": "MSFT", "fiscal_yearq": "2014Q1",
        "actual_eps": 0.80, "consensus_median": 1.20, "consensus_dispersion": 0.20,
        "n_analysts": 8,
    }])
    out = compute_sue(panel)
    # SUE = (0.80 - 1.20) / 0.20 = -2.0
    assert out["sue"].iloc[0] == pytest.approx(-2.0)


def test_compute_sue_zero_dispersion_returns_nan():
    """dispersion=0 should NaN-out SUE (Sprint 2 already filters, but defensive)."""
    panel = _make_panel([{
        "ticker_ibes": "GOOG", "fiscal_yearq": "2014Q1",
        "actual_eps": 1.50, "consensus_median": 1.20, "consensus_dispersion": 0.0,
        "n_analysts": 10,
    }])
    out = compute_sue(panel)
    assert np.isnan(out["sue"].iloc[0])


def test_compute_sue_insufficient_analysts_returns_nan():
    """n_analysts < MIN_ANALYSTS_REQUIRED → NaN."""
    panel = _make_panel([{
        "ticker_ibes": "TSLA", "fiscal_yearq": "2014Q1",
        "actual_eps": 1.50, "consensus_median": 1.20, "consensus_dispersion": 0.10,
        "n_analysts": MIN_ANALYSTS_REQUIRED - 1,  # = 1, below threshold
    }])
    out = compute_sue(panel)
    assert np.isnan(out["sue"].iloc[0])


def test_compute_sue_at_min_analysts_threshold_returns_value():
    """n_analysts EXACTLY at MIN_ANALYSTS_REQUIRED → valid SUE."""
    panel = _make_panel([{
        "ticker_ibes": "NVDA", "fiscal_yearq": "2014Q1",
        "actual_eps": 1.50, "consensus_median": 1.20, "consensus_dispersion": 0.15,
        "n_analysts": MIN_ANALYSTS_REQUIRED,  # = 2, AT threshold
    }])
    out = compute_sue(panel)
    assert not np.isnan(out["sue"].iloc[0])


def test_compute_sue_empty_panel():
    """Empty panel returns empty panel (no crash)."""
    panel = pd.DataFrame(columns=["actual_eps", "consensus_median", "consensus_dispersion", "n_analysts"])
    out = compute_sue(panel)
    assert out.empty


def test_compute_sue_missing_columns_raises():
    """Panel missing required schema raises ValueError."""
    panel = pd.DataFrame([{"actual_eps": 1.0, "consensus_median": 0.8}])
    with pytest.raises(ValueError, match="missing columns"):
        compute_sue(panel)


def test_compute_sue_does_not_mutate_input():
    """compute_sue returns a copy; input untouched."""
    panel = _make_panel([{
        "ticker_ibes": "AAPL", "fiscal_yearq": "2014Q1",
        "actual_eps": 1.50, "consensus_median": 1.20, "consensus_dispersion": 0.10,
        "n_analysts": 10,
    }])
    cols_before = list(panel.columns)
    _ = compute_sue(panel)
    assert list(panel.columns) == cols_before


# ─────────────────────────────────────────────────────────────────────────────
# rank_within_quarter: cross-sectional ranking
# ─────────────────────────────────────────────────────────────────────────────

def _build_cross_section(n_firms: int, quarter: str = "2014Q1") -> pd.DataFrame:
    """Build a cross-section with n_firms firms, SUE values from -n/2..n/2."""
    rows = []
    for i in range(n_firms):
        sue_value = float(i) - (n_firms / 2.0)  # linear range, distinct
        rows.append({
            "ticker_ibes": f"T{i:03d}",
            "fiscal_yearq": quarter,
            "actual_eps": 1.0 + sue_value * 0.1,
            "consensus_median": 1.0,
            "consensus_dispersion": 0.1,
            "n_analysts": 5,
        })
    return _make_panel(rows)


def test_rank_within_quarter_monotonic():
    """Higher SUE → higher rank percentile within same quarter."""
    panel = _build_cross_section(n_firms=100)
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    # Bottom firm (T000, lowest SUE) has lowest rank pct
    bot_pct = p2[p2["ticker_ibes"] == "T000"]["sue_rank_pct"].iloc[0]
    top_pct = p2[p2["ticker_ibes"] == "T099"]["sue_rank_pct"].iloc[0]
    assert bot_pct < top_pct
    # Mid-rank empirical CDF: top firm gets (n - 0.5) / n = 0.995 for n=100
    assert top_pct == pytest.approx(0.995)
    assert bot_pct == pytest.approx(0.005)


def test_rank_within_quarter_two_quarters_independent():
    """Rankings within Q1 and Q2 are computed independently."""
    p_q1 = _build_cross_section(n_firms=50, quarter="2014Q1")
    p_q2 = _build_cross_section(n_firms=50, quarter="2014Q2")
    # Make Q2 SUE values all higher than Q1's
    p_q2["actual_eps"] = p_q2["actual_eps"] + 100.0
    panel = pd.concat([p_q1, p_q2], ignore_index=True)

    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)

    # T000 in Q1 still has rank near 0 (lowest in Q1 cross-section)
    q1_t000 = p2[(p2["ticker_ibes"] == "T000") & (p2["fiscal_yearq"] == "2014Q1")]["sue_rank_pct"].iloc[0]
    # T000 in Q2 also has rank near 0 (lowest in Q2 cross-section), even though
    # absolute SUE is much higher than Q1's
    q2_t000 = p2[(p2["ticker_ibes"] == "T000") & (p2["fiscal_yearq"] == "2014Q2")]["sue_rank_pct"].iloc[0]
    assert q1_t000 < 0.05
    assert q2_t000 < 0.05  # Independent ranking — within-quarter only


def test_rank_within_quarter_tiebreak_deterministic():
    """Ties in SUE broken by ticker ascending — same input → same ranks."""
    rows = []
    for i, ticker in enumerate(["ZZZ", "AAA", "MMM"]):
        rows.append({
            "ticker_ibes": ticker, "fiscal_yearq": "2014Q1",
            "actual_eps": 1.1, "consensus_median": 1.0, "consensus_dispersion": 0.1,
            "n_analysts": 5,
        })
    panel = _make_panel(rows)
    p1 = compute_sue(panel)
    p2a = rank_within_quarter(p1)
    p2b = rank_within_quarter(p1)
    # Run twice → identical rank assignment
    pd.testing.assert_series_equal(
        p2a.set_index("ticker_ibes")["sue_rank_pct"],
        p2b.set_index("ticker_ibes")["sue_rank_pct"],
    )
    # Ties broken by ticker asc: AAA gets lowest rank, ZZZ highest
    aaa_pct = p2a[p2a["ticker_ibes"] == "AAA"]["sue_rank_pct"].iloc[0]
    zzz_pct = p2a[p2a["ticker_ibes"] == "ZZZ"]["sue_rank_pct"].iloc[0]
    assert aaa_pct < zzz_pct


def test_rank_within_quarter_too_few_firms_nan():
    """Single-firm quarter has NaN rank (cannot rank cross-section of 1)."""
    panel = _make_panel([{
        "ticker_ibes": "T001", "fiscal_yearq": "2014Q1",
        "actual_eps": 1.5, "consensus_median": 1.0, "consensus_dispersion": 0.1,
        "n_analysts": 5,
    }])
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    assert np.isnan(p2["sue_rank_pct"].iloc[0])


def test_rank_within_quarter_excludes_nan_sue():
    """Firms with NaN SUE don't affect other firms' rank denominator."""
    panel = _make_panel([
        {"ticker_ibes": "AAA", "fiscal_yearq": "Q1",
         "actual_eps": 1.0, "consensus_median": 1.0, "consensus_dispersion": 0.0,  # NaN SUE
         "n_analysts": 5},
        {"ticker_ibes": "BBB", "fiscal_yearq": "Q1",
         "actual_eps": 1.5, "consensus_median": 1.0, "consensus_dispersion": 0.1,
         "n_analysts": 5},
        {"ticker_ibes": "CCC", "fiscal_yearq": "Q1",
         "actual_eps": 0.5, "consensus_median": 1.0, "consensus_dispersion": 0.1,
         "n_analysts": 5},
    ])
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    # AAA has NaN rank (excluded); BBB and CCC ranked over n=2
    aaa = p2[p2["ticker_ibes"] == "AAA"]["sue_rank_pct"].iloc[0]
    bbb = p2[p2["ticker_ibes"] == "BBB"]["sue_rank_pct"].iloc[0]
    ccc = p2[p2["ticker_ibes"] == "CCC"]["sue_rank_pct"].iloc[0]
    assert np.isnan(aaa)
    # Mid-rank: 2 valid firms → (rank - 0.5) / 2 → ranks 0.25, 0.75
    assert bbb == pytest.approx(0.75)  # higher SUE
    assert ccc == pytest.approx(0.25)  # lower SUE


def test_rank_within_quarter_empty_panel():
    p = pd.DataFrame(columns=["actual_eps", "consensus_median", "consensus_dispersion",
                              "n_analysts", "ticker_ibes", "fiscal_yearq", "sue"])
    out = rank_within_quarter(p)
    assert "sue_rank_pct" in out.columns


def test_rank_within_quarter_missing_sue_column_raises():
    panel = _make_panel([{
        "ticker_ibes": "AAPL", "fiscal_yearq": "Q1",
        "actual_eps": 1.5, "consensus_median": 1.0, "consensus_dispersion": 0.1,
        "n_analysts": 5,
    }])
    with pytest.raises(ValueError, match="run compute_sue first"):
        rank_within_quarter(panel)


# ─────────────────────────────────────────────────────────────────────────────
# assign_decile_legs: top/bottom decile assignment
# ─────────────────────────────────────────────────────────────────────────────

def test_assign_decile_legs_top_decile_long():
    """Firms in top 10% by SUE → leg = 'long'."""
    panel = _build_cross_section(n_firms=100)
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3 = assign_decile_legs(p2)
    # Top 10 firms (T090..T099) should be long
    top_10 = p3[p3["ticker_ibes"].isin([f"T{i:03d}" for i in range(90, 100)])]
    assert (top_10["leg"] == "long").all()


def test_assign_decile_legs_bottom_decile_short():
    """Firms in bottom 10% by SUE → leg = 'short'."""
    panel = _build_cross_section(n_firms=100)
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3 = assign_decile_legs(p2)
    # Bottom 10 firms (T000..T009) should be short
    bot_10 = p3[p3["ticker_ibes"].isin([f"T{i:03d}" for i in range(0, 10)])]
    assert (bot_10["leg"] == "short").all()


def test_assign_decile_legs_middle_80_flat():
    """Firms in middle 80% → leg = 'flat'."""
    panel = _build_cross_section(n_firms=100)
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3 = assign_decile_legs(p2)
    mid = p3[p3["ticker_ibes"].isin([f"T{i:03d}" for i in range(10, 90)])]
    assert (mid["leg"] == "flat").all()


def test_assign_decile_legs_boundary_inclusive():
    """Spec §N6: firms exactly at 10/90 percentile included in long/short leg."""
    panel = _build_cross_section(n_firms=10)  # 10 firms → percentiles 0.1, 0.2, ..., 1.0
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3 = assign_decile_legs(p2)
    # T000 has rank 1/10 = 0.10 (≤ 0.10 short threshold)
    t000 = p3[p3["ticker_ibes"] == "T000"]["leg"].iloc[0]
    # T009 has rank 10/10 = 1.00 (≥ 0.90 long threshold)
    t009 = p3[p3["ticker_ibes"] == "T009"]["leg"].iloc[0]
    assert t000 == "short"   # boundary at 0.10 inclusive
    assert t009 == "long"    # boundary at 0.90 inclusive (rank 1.0 ≥ 0.90)


def test_assign_decile_legs_excluded_for_nan_rank():
    """NaN rank → leg = 'excluded'."""
    panel = _make_panel([{
        "ticker_ibes": "AAPL", "fiscal_yearq": "Q1",
        "actual_eps": 1.5, "consensus_median": 1.0, "consensus_dispersion": 0.0,
        "n_analysts": 5,
    }])
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3 = assign_decile_legs(p2)
    assert p3["leg"].iloc[0] == "excluded"


def test_assign_decile_legs_bad_thresholds_raise():
    panel = _make_panel([{
        "ticker_ibes": "A", "fiscal_yearq": "Q1", "actual_eps": 1.0,
        "consensus_median": 1.0, "consensus_dispersion": 0.1, "n_analysts": 5,
    }])
    p = compute_sue(panel)
    p = rank_within_quarter(p)
    # short > long
    with pytest.raises(ValueError):
        assign_decile_legs(p, long_threshold=0.20, short_threshold=0.80)
    # short = long
    with pytest.raises(ValueError):
        assign_decile_legs(p, long_threshold=0.50, short_threshold=0.50)
    # long ≥ 1
    with pytest.raises(ValueError):
        assign_decile_legs(p, long_threshold=1.10, short_threshold=0.10)


def test_assign_decile_legs_missing_rank_column_raises():
    panel = _make_panel([{
        "ticker_ibes": "A", "fiscal_yearq": "Q1", "actual_eps": 1.0,
        "consensus_median": 1.0, "consensus_dispersion": 0.1, "n_analysts": 5,
    }])
    p = compute_sue(panel)
    with pytest.raises(ValueError, match="run rank_within_quarter first"):
        assign_decile_legs(p)


def test_assign_decile_legs_uses_locked_thresholds_by_default():
    """Default thresholds = DECILE_LONG/SHORT_THRESHOLD per __init__.py (spec §六)."""
    panel = _build_cross_section(n_firms=100)
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3_default  = assign_decile_legs(p2)
    p3_explicit = assign_decile_legs(p2, long_threshold=DECILE_LONG_THRESHOLD,
                                          short_threshold=DECILE_SHORT_THRESHOLD)
    pd.testing.assert_series_equal(p3_default["leg"], p3_explicit["leg"])


# ─────────────────────────────────────────────────────────────────────────────
# build_sue_signal_panel: one-shot pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_build_sue_signal_panel_end_to_end():
    """One-shot wrapper produces full sue + rank + leg columns."""
    panel = _build_cross_section(n_firms=100)
    out = build_sue_signal_panel(panel)
    assert {"sue", "sue_rank_pct", "leg"}.issubset(set(out.columns))
    # 10% long + 10% short + 80% flat
    leg_counts = out["leg"].value_counts()
    assert leg_counts.get("long", 0) == 10
    assert leg_counts.get("short", 0) == 10
    assert leg_counts.get("flat", 0) == 80


def test_build_sue_signal_panel_deterministic():
    """Same input → same output (no RNG, no order-dependence)."""
    panel = _build_cross_section(n_firms=50)
    o1 = build_sue_signal_panel(panel)
    o2 = build_sue_signal_panel(panel)
    pd.testing.assert_frame_equal(o1, o2)


def test_build_sue_signal_panel_uses_mock_panel_from_sprint2():
    """Integration: Sprint 2 mock panel → Sprint 3 produces non-trivial legs."""
    from engine.path_c.earnings_panel import _mock_earnings_panel
    sprint2_panel = _mock_earnings_panel(
        tickers=[f"T{i:03d}" for i in range(100)],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
    )
    out = build_sue_signal_panel(sprint2_panel)
    # Should produce at least some long and short rows
    leg_counts = out["leg"].value_counts()
    assert leg_counts.get("long", 0) > 0
    assert leg_counts.get("short", 0) > 0
    # Total legs should sum to total rows
    assert leg_counts.sum() == len(out)
