"""bt-flex-1 (2026-06-11) tests for OOS in-paper / post-paper / full-sample triple.

Covers:
  * window parser handles 'YYYY-MM' and 'YYYY-MM-DD' forms
  * segment stats decline N<min and degenerate std
  * triple computes correctly for: paper-window-fully-inside / paper-ends-mid /
    paper-fully-before-data / paper-fully-after-data / non-overlap
  * severity classification at band boundaries
  * narrative includes the expected phrases per case
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd
import pytest

from engine.research import oos_triple as ot


# ── Window parser ───────────────────────────────────────────────────


def test_parse_window_year_month():
    s, e = ot._parse_window("1990-01:2010-12")
    assert s == _dt.date(1990, 1, 1)
    assert e == _dt.date(2010, 12, 31)


def test_parse_window_year_month_day():
    s, e = ot._parse_window("1990-01-15:2010-06-30")
    assert s == _dt.date(1990, 1, 15)
    assert e == _dt.date(2010, 6, 30)


def test_parse_window_rejects_missing_colon():
    with pytest.raises(ValueError):
        ot._parse_window("1990-01-2010-12")


# ── Intersection ────────────────────────────────────────────────────


def test_intersect_overlap():
    a = (_dt.date(2000, 1, 1), _dt.date(2010, 12, 31))
    b = (_dt.date(2005, 1, 1), _dt.date(2015, 12, 31))
    assert ot._intersect(a, b) == (_dt.date(2005, 1, 1), _dt.date(2010, 12, 31))


def test_intersect_no_overlap():
    a = (_dt.date(2000, 1, 1), _dt.date(2004, 12, 31))
    b = (_dt.date(2005, 1, 1), _dt.date(2010, 12, 31))
    assert ot._intersect(a, b) is None


# ── Segment stats ───────────────────────────────────────────────────


def _make_monthly_pnl(start: str, end: str, mean: float, std: float, seed: int = 7):
    """Build a monthly PnL series with controllable mean/std."""
    idx = pd.date_range(start, end, freq="M")
    rng = np.random.default_rng(seed)
    vals = rng.normal(mean, std, size=len(idx))
    return pd.Series(vals, index=idx)


def test_segment_stats_below_min_returns_none():
    # Only 12 months — below MIN_MONTHS_PER_SEGMENT=24
    s = _make_monthly_pnl("2020-01", "2020-12", 0.01, 0.02)
    assert ot._compute_segment_stats(s) is None


def test_segment_stats_basic_sharpe():
    s = _make_monthly_pnl("2010-01", "2020-12", mean=0.01, std=0.04, seed=42)
    stats = ot._compute_segment_stats(s)
    assert stats is not None
    assert stats.n_months >= 24
    # Sharpe should be roughly 0.01 / 0.04 * sqrt(12) ≈ 0.87 (noisy)
    assert 0.3 < stats.sharpe_ann < 1.5
    assert stats.start_ym.startswith("2010-")
    assert stats.end_ym.startswith("2020-")


def test_segment_stats_zero_std_returns_none():
    idx = pd.date_range("2010-01", "2015-12", freq="M")
    s = pd.Series([0.01] * len(idx), index=idx)  # constant — std == 0
    assert ot._compute_segment_stats(s) is None


# ── Severity classification ─────────────────────────────────────────


def test_severity_bands():
    assert ot._classify_severity(0.0) == "none"
    assert ot._classify_severity(-0.10) == "none"
    assert ot._classify_severity(-0.20) == "none"        # boundary
    assert ot._classify_severity(-0.25) == "mild"
    assert ot._classify_severity(-0.40) == "mild"        # boundary
    assert ot._classify_severity(-0.55) == "severe"
    assert ot._classify_severity(-0.70) == "severe"      # boundary
    assert ot._classify_severity(-0.85) == "broken"
    assert ot._classify_severity(-2.00) == "broken"


# ── Triple end-to-end ───────────────────────────────────────────────


def test_triple_paper_window_inside_data():
    # Data: 1990-2025. Paper: 1990-2010. Post-paper: 2011-2025.
    idx_in = pd.date_range("1990-01", "2010-12", freq="M")
    idx_post = pd.date_range("2011-01", "2025-12", freq="M")
    idx = idx_in.append(idx_post)
    rng = np.random.default_rng(123)
    # In-paper higher Sharpe than post — emulate decay.
    in_paper_returns  = rng.normal(0.012, 0.04, size=len(idx_in))
    post_paper_returns = rng.normal(0.004, 0.04, size=len(idx_post))
    values = np.concatenate([in_paper_returns, post_paper_returns])
    pnl = pd.Series(values, index=idx)
    df = pd.DataFrame({"pnl_net_13bp": pnl})

    result = ot.compute_oos_triple(
        df,
        full_window  = "1990-01:2025-12",
        paper_window = "1990-01:2010-12",
    )
    assert result is not None
    assert result.in_paper is not None
    assert result.post_paper is not None
    assert result.in_paper.n_months >= 12 * 21 - 1
    assert result.post_paper.n_months >= 12 * 15 - 1
    # Expect decay (post Sharpe < in Sharpe given lower mean)
    assert result.decay_pct is not None
    assert result.decay_pct < 0.0
    assert result.severity in ("mild", "severe", "broken")
    assert "Sharpe" in result.narrative
    assert "Decay" in result.narrative or "decay" in result.narrative


def test_triple_paper_window_predates_data():
    # Data: 2010-2025. Paper: 1990-2005. No overlap → in_paper None,
    # only post_paper computable.
    pnl = _make_monthly_pnl("2010-01", "2025-12", 0.008, 0.04)
    df = pd.DataFrame({"pnl_net_13bp": pnl})
    result = ot.compute_oos_triple(
        df,
        full_window  = "2010-01:2025-12",
        paper_window = "1990-01:2005-12",
    )
    assert result is not None
    assert result.in_paper is None
    assert result.post_paper is not None
    assert result.severity == "inconclusive"
    assert "cannot be tested" in result.narrative


def test_triple_paper_window_covers_all_data():
    # Data: 2010-2020. Paper: 1990-2025 (covers everything → no post-paper).
    pnl = _make_monthly_pnl("2010-01", "2020-12", 0.008, 0.04)
    df = pd.DataFrame({"pnl_net_13bp": pnl})
    result = ot.compute_oos_triple(
        df,
        full_window  = "2010-01:2020-12",
        paper_window = "1990-01:2025-12",
    )
    assert result is not None
    assert result.in_paper is not None
    assert result.post_paper is None
    assert result.severity == "inconclusive"
    assert "decay cannot be tested" in result.narrative or "<24mo" in result.narrative


def test_triple_no_paper_column_returns_none():
    pnl = _make_monthly_pnl("2010-01", "2020-12", 0.008, 0.04)
    df = pd.DataFrame({"some_other_col": pnl})
    result = ot.compute_oos_triple(
        df,
        full_window  = "2010-01:2020-12",
        paper_window = "2010-01:2015-12",
        pnl_column   = "pnl_net_13bp",
    )
    assert result is None


def test_triple_none_df_returns_none():
    result = ot.compute_oos_triple(None, full_window="2010-01:2020-12", paper_window="2010-01:2015-12")
    assert result is None


def test_triple_no_decay_when_signal_dominates_noise():
    # Deterministic signal-dominated series: each month = mean + small jitter.
    # In/post Sharpe should be nearly identical → severity 'none'.
    idx = pd.date_range("1990-01", "2025-12", freq="M")
    rng = np.random.default_rng(7)
    # Mean 0.01, tiny std → Sharpe annualized ~70 (degenerate but
    # validates decay_pct ≈ 0).
    vals = 0.01 + rng.normal(0.0, 0.0005, size=len(idx))
    pnl = pd.Series(vals, index=idx)
    df = pd.DataFrame({"pnl_net_13bp": pnl})
    result = ot.compute_oos_triple(
        df,
        full_window  = "1990-01:2025-12",
        paper_window = "1990-01:2010-12",
    )
    assert result is not None
    assert result.decay_pct is not None
    # |decay_pct| should be small — both segments have identical DGP
    assert abs(result.decay_pct) < 0.20
    assert result.severity == "none"


def test_triple_to_dict_serializable():
    pnl = _make_monthly_pnl("1990-01", "2025-12", 0.01, 0.04, seed=11)
    df = pd.DataFrame({"pnl_net_13bp": pnl})
    result = ot.compute_oos_triple(
        df,
        full_window  = "1990-01:2025-12",
        paper_window = "1990-01:2010-12",
    )
    import json
    d = result.to_dict()
    s = json.dumps(d, default=str)
    assert "in_paper" in s
    assert "decay_pct" in s
    assert "severity" in s
