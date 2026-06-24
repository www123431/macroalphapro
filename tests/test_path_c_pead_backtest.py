"""
tests/test_path_c_pead_backtest.py — Sprint 4 walk-forward + aggregation tests.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §2.4 + §2.6

Surface:
  - Trading-day arithmetic (60 trading days post-rdq, skips day 0)
  - Position window computation
  - Daily L-S aggregation (synthetic 2-firm panel)
  - Multi-firm cross-section aggregation
  - Position lifecycle (firm enters rdq+1, exits rdq+60)
  - Turnover + TC drag math
  - Walk-forward orchestrator end-to-end (mock)
  - Per-quarter checkpoint write/read
  - Edge cases: empty panel, single quarter, no active firms
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.path_c import (
    HOLD_TRADING_DAYS_LOCKED,
    TC_BPS_ROUNDTRIP_LOCKED,
)
from engine.path_c.pead_backtest import (
    trading_day_after,
    compute_position_windows,
    compute_daily_long_short_returns,
    compute_annual_turnover,
    apply_tc_drag,
    write_quarter_checkpoint,
    read_quarter_checkpoints,
    run_walk_forward_pead,
    persist_walk_forward_result,
    WalkForwardPeadResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Trading day arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def test_trading_day_after_skips_weekends():
    """Friday + 1 trading day = next Monday."""
    fri = datetime.date(2014, 1, 3)  # Fri
    mon = trading_day_after(fri, 1)
    assert mon == datetime.date(2014, 1, 6)  # Mon


def test_trading_day_after_60_days_is_about_3_months():
    """60 trading days ≈ 12 weeks ≈ 84 calendar days from a Monday."""
    mon = datetime.date(2014, 1, 6)  # Mon
    result = trading_day_after(mon, 60)
    delta = (result - mon).days
    # 60 bdays ≈ 84 calendar days (12 weeks × 7 = 84)
    assert 83 <= delta <= 91  # allow ±1 week slack for holiday alignment


def test_trading_day_after_n_zero_raises():
    with pytest.raises(ValueError, match="n_days must be"):
        trading_day_after(datetime.date(2014, 1, 1), 0)


def test_trading_day_after_negative_raises():
    with pytest.raises(ValueError, match="n_days must be"):
        trading_day_after(datetime.date(2014, 1, 1), -1)


# ─────────────────────────────────────────────────────────────────────────────
# Position windows
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal_panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_compute_position_windows_for_long_short():
    """Long + short rows get window_start/end; flat rows get NaT."""
    panel = _make_signal_panel([
        {"ticker_ibes": "AAPL", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
        {"ticker_ibes": "MSFT", "rdq": datetime.date(2014, 2, 5), "leg": "short",
         "fiscal_yearq": "2014Q1"},
        {"ticker_ibes": "GOOG", "rdq": datetime.date(2014, 2, 7), "leg": "flat",
         "fiscal_yearq": "2014Q1"},
    ])
    out = compute_position_windows(panel)
    aapl = out[out["ticker_ibes"] == "AAPL"].iloc[0]
    assert aapl["window_start"] == datetime.date(2014, 2, 4)  # day after Feb 3 (Mon)
    msft = out[out["ticker_ibes"] == "MSFT"].iloc[0]
    assert msft["window_start"] == datetime.date(2014, 2, 6)
    goog = out[out["ticker_ibes"] == "GOOG"].iloc[0]
    assert pd.isna(goog["window_start"])


def test_compute_position_windows_hold_days_locked():
    """window_end - window_start ≈ HOLD_TRADING_DAYS_LOCKED - 1 trading days."""
    panel = _make_signal_panel([{
        "ticker_ibes": "AAPL", "rdq": datetime.date(2014, 2, 3), "leg": "long",
        "fiscal_yearq": "2014Q1",
    }])
    out = compute_position_windows(panel)
    row = out.iloc[0]
    diff_bdays = len(pd.bdate_range(start=row["window_start"], end=row["window_end"]))
    assert diff_bdays == HOLD_TRADING_DAYS_LOCKED


def test_compute_position_windows_empty_panel():
    out = compute_position_windows(pd.DataFrame())
    assert out.empty
    assert "window_start" in out.columns
    assert "window_end"   in out.columns


# ─────────────────────────────────────────────────────────────────────────────
# Daily long-short aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _make_returns_panel(tickers, start_date, end_date, daily_ret_const):
    """Build a returns panel with constant daily return per ticker."""
    dates = pd.bdate_range(start=start_date, end=end_date)
    if isinstance(daily_ret_const, dict):
        cols = {t: pd.Series(daily_ret_const.get(t, 0.0), index=dates) for t in tickers}
    else:
        cols = {t: pd.Series(daily_ret_const, index=dates) for t in tickers}
    df = pd.DataFrame(cols)
    df.index.name = "date"
    return df


def test_daily_ls_single_long_single_short_constant_returns():
    """Long firm returns +0.001/day, short firm returns -0.001/day → L-S = +0.002."""
    signal = _make_signal_panel([
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
        {"ticker_ibes": "S1", "rdq": datetime.date(2014, 2, 3), "leg": "short",
         "fiscal_yearq": "2014Q1"},
    ])
    returns = _make_returns_panel(
        ["L1", "S1"],
        datetime.date(2014, 2, 1),
        datetime.date(2014, 5, 31),
        daily_ret_const={"L1": 0.001, "S1": -0.001},
    )
    daily = compute_daily_long_short_returns(signal, returns)
    assert not daily.empty
    # Pick a mid-window day
    mid = pd.Timestamp(datetime.date(2014, 3, 15))
    if mid not in daily.index:
        mid = daily.index[len(daily) // 2]
    assert daily.loc[mid, "r_long"]  == pytest.approx(0.001, abs=1e-9)
    assert daily.loc[mid, "r_short"] == pytest.approx(-0.001, abs=1e-9)
    assert daily.loc[mid, "r_long_short"] == pytest.approx(0.002, abs=1e-9)
    assert daily.loc[mid, "n_long"]  == 1
    assert daily.loc[mid, "n_short"] == 1


def test_daily_ls_multi_firm_cross_section_mean():
    """3 long firms each +0.01 → r_long = mean = 0.01."""
    signal = _make_signal_panel([
        {"ticker_ibes": f"L{i}", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"} for i in range(3)
    ] + [
        {"ticker_ibes": f"S{i}", "rdq": datetime.date(2014, 2, 3), "leg": "short",
         "fiscal_yearq": "2014Q1"} for i in range(3)
    ])
    returns = _make_returns_panel(
        [f"L{i}" for i in range(3)] + [f"S{i}" for i in range(3)],
        datetime.date(2014, 2, 1),
        datetime.date(2014, 5, 31),
        daily_ret_const=0.01,
    )
    # Override short legs to be -0.01
    for i in range(3):
        returns[f"S{i}"] = -0.01

    daily = compute_daily_long_short_returns(signal, returns)
    mid = daily.index[len(daily) // 2]
    assert daily.loc[mid, "r_long"]  == pytest.approx(0.01, abs=1e-9)
    assert daily.loc[mid, "r_short"] == pytest.approx(-0.01, abs=1e-9)
    assert daily.loc[mid, "r_long_short"] == pytest.approx(0.02, abs=1e-9)
    assert daily.loc[mid, "n_long"]  == 3
    assert daily.loc[mid, "n_short"] == 3


def test_daily_ls_position_lifecycle_60day():
    """Position only contributes to daily L-S between rdq+1 and rdq+60 trading days."""
    rdq = datetime.date(2014, 2, 3)
    signal = _make_signal_panel([{
        "ticker_ibes": "L1", "rdq": rdq, "leg": "long", "fiscal_yearq": "2014Q1",
    }])
    returns = _make_returns_panel(
        ["L1"], datetime.date(2014, 2, 1), datetime.date(2014, 8, 31),
        daily_ret_const=0.01,
    )
    daily = compute_daily_long_short_returns(signal, returns)

    # On rdq day itself (Feb 3 = Mon), position not yet active → not in daily index OR n_long=0
    rdq_ts = pd.Timestamp(rdq)
    if rdq_ts in daily.index:
        assert daily.loc[rdq_ts, "n_long"] == 0

    # Window start Feb 4 (Tue) → active
    ws_ts = pd.Timestamp(trading_day_after(rdq, 1))
    assert daily.loc[ws_ts, "n_long"] == 1

    # Window end (rdq + 60 trading days)
    we_date = trading_day_after(rdq, 60)
    we_ts = pd.Timestamp(we_date)
    assert daily.loc[we_ts, "n_long"] == 1

    # Past window_end → no longer active
    past_we = trading_day_after(we_date, 1)
    past_ts = pd.Timestamp(past_we)
    if past_ts in daily.index:
        assert daily.loc[past_ts, "n_long"] == 0


def test_daily_ls_empty_signal_panel():
    """Empty signal_panel → empty daily DataFrame with correct columns."""
    signal = pd.DataFrame(columns=["ticker_ibes", "rdq", "leg", "fiscal_yearq"])
    returns = _make_returns_panel(
        ["A"], datetime.date(2014, 2, 1), datetime.date(2014, 5, 31), 0.001,
    )
    daily = compute_daily_long_short_returns(signal, returns)
    assert daily.empty
    assert set(daily.columns) == {"r_long", "r_short", "r_long_short", "n_long", "n_short"}


def test_daily_ls_empty_returns_panel():
    """Empty returns panel → empty daily output."""
    signal = _make_signal_panel([{
        "ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
        "fiscal_yearq": "2014Q1",
    }])
    returns = pd.DataFrame()
    daily = compute_daily_long_short_returns(signal, returns)
    assert daily.empty


def test_daily_ls_flat_legs_ignored():
    """Flat-leg firms don't contribute to portfolio at all."""
    signal = _make_signal_panel([
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
        {"ticker_ibes": "F1", "rdq": datetime.date(2014, 2, 3), "leg": "flat",
         "fiscal_yearq": "2014Q1"},
    ])
    returns = _make_returns_panel(
        ["L1", "F1"], datetime.date(2014, 2, 1), datetime.date(2014, 5, 31), 0.01,
    )
    daily = compute_daily_long_short_returns(signal, returns)
    mid = daily.index[len(daily) // 2]
    # n_long = 1 (just L1, F1 excluded)
    assert daily.loc[mid, "n_long"] == 1


def test_daily_ls_same_firm_overlap_dedupe():
    """Rigor audit fix 2026-05-12 finding D: same firm in long leg from 2
    consecutive quarters with overlapping holds gets equal-weight (not 2x).

    Scenario: firm L1 announces Q1 + Q2 both top-decile, hold windows overlap
    for ~17 trading days. Without dedupe, mean() counts L1 twice in those days.
    With dedupe, L1 counted once.
    """
    signal = _make_signal_panel([
        # Q1 announce: rdq = 2014-02-03 Mon, hold to ~2014-04-30 (60 bdays)
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
        # Q2 announce: rdq = 2014-03-31 Mon (early/overlapping with Q1 hold)
        # Hold extends to ~2014-06-23. Overlap with Q1 hold: ~2014-04-01 to 2014-04-30 ≈ 22 bdays
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 3, 31), "leg": "long",
         "fiscal_yearq": "2014Q2"},
        # Independent firm in long leg
        {"ticker_ibes": "L2", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
    ])
    returns = _make_returns_panel(
        ["L1", "L2"], datetime.date(2014, 2, 1), datetime.date(2014, 8, 31),
        daily_ret_const={"L1": 0.020, "L2": 0.005},  # L1 has 4x L2's return
    )
    daily = compute_daily_long_short_returns(signal, returns)
    # In overlap day (say 2014-04-15), both L1 entries + L2 in book.
    # After dedupe: only one (L1, 2014-04-15, long) + (L2, 2014-04-15, long).
    # mean = (0.020 + 0.005) / 2 = 0.0125 (not 0.015 = (0.020+0.020+0.005)/3 which
    # would be the buggy double-counted value).
    overlap_day = pd.Timestamp(datetime.date(2014, 4, 15))
    if overlap_day in daily.index:
        assert daily.loc[overlap_day, "r_long"] == pytest.approx(0.0125, abs=1e-9)
        # n_long = 2 distinct firms (not 3 rows)
        assert daily.loc[overlap_day, "n_long"] == 2


def test_daily_ls_coverage_warning_logged(caplog):
    """Rigor audit fix 2026-05-12 finding A: warn when returns_panel ends
    before position windows extend (end-of-window drift truncation)."""
    import logging
    signal = _make_signal_panel([
        # Announces late in returns window — hold extends past
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 6, 15), "leg": "long",
         "fiscal_yearq": "2014Q2"},
    ])
    # Returns only cover until 2014-07-31 — but hold needs to 2014-09-08 (60 bdays)
    returns = _make_returns_panel(
        ["L1"], datetime.date(2014, 5, 1), datetime.date(2014, 7, 31), 0.001,
    )
    with caplog.at_level(logging.WARNING, logger="engine.path_c.pead_backtest"):
        compute_daily_long_short_returns(signal, returns)
    # Warning should mention truncation
    assert any("truncated" in r.message.lower() or "extending" in r.message.lower()
               for r in caplog.records), \
        f"Expected truncation warning; got: {[r.message for r in caplog.records]}"


def test_daily_ls_missing_required_columns_raises():
    """Missing ticker_ibes / rdq / leg → ValueError."""
    bad_signal = pd.DataFrame([{"ticker_ibes": "X", "rdq": datetime.date(2014, 2, 3), "leg": "long"}])
    bad_signal = bad_signal.drop(columns=["rdq"])
    with pytest.raises(ValueError, match="missing"):
        compute_daily_long_short_returns(bad_signal, _make_returns_panel(["X"], datetime.date(2014, 2, 1), datetime.date(2014, 5, 31), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Turnover + TC drag
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_annual_turnover_60d_hold():
    """60-day hold → ~4.2 roundtrips/year (252/60)."""
    daily = pd.DataFrame()  # not used by structural estimator
    turnover = compute_annual_turnover(daily, hold_trading_days=60)
    assert turnover == pytest.approx(252.0 / 60.0)


def test_compute_annual_turnover_zero_hold_returns_zero():
    daily = pd.DataFrame()
    assert compute_annual_turnover(daily, hold_trading_days=0) == 0.0


def test_apply_tc_drag_reduces_gross_returns():
    """Net daily returns < gross by uniform drag."""
    gross = pd.Series([0.001, 0.002, -0.001], index=pd.bdate_range(start="2014-02-01", periods=3))
    net = apply_tc_drag(gross, annual_turnover=4.2, tc_bps_roundtrip=30.0)
    expected_drag = (30 / 10_000) * 4.2 / 252
    for i in range(3):
        assert net.iloc[i] == pytest.approx(gross.iloc[i] - expected_drag, abs=1e-12)


def test_apply_tc_drag_zero_turnover_zero_drag():
    gross = pd.Series([0.001, 0.002], index=pd.bdate_range(start="2014-02-01", periods=2))
    net = apply_tc_drag(gross, annual_turnover=0.0, tc_bps_roundtrip=30.0)
    pd.testing.assert_series_equal(net, gross, check_exact=False, rtol=1e-12)


def test_apply_tc_drag_uses_locked_bps_by_default():
    """Default tc_bps_roundtrip = TC_BPS_ROUNDTRIP_LOCKED per spec §六."""
    gross = pd.Series([0.001], index=pd.bdate_range(start="2014-02-01", periods=1))
    net_default  = apply_tc_drag(gross, annual_turnover=4.2)
    net_explicit = apply_tc_drag(gross, annual_turnover=4.2, tc_bps_roundtrip=TC_BPS_ROUNDTRIP_LOCKED)
    pd.testing.assert_series_equal(net_default, net_explicit)


# ─────────────────────────────────────────────────────────────────────────────
# Per-quarter checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_write_read_roundtrip(tmp_path):
    write_quarter_checkpoint("run_x", "2014Q1", 10, 10, 5, base_dir=tmp_path)
    write_quarter_checkpoint("run_x", "2014Q2", 12, 11, 3, base_dir=tmp_path)
    records = read_quarter_checkpoints("run_x", base_dir=tmp_path)
    assert len(records) == 2
    assert records[0]["quarter"] == "2014Q1"
    assert records[0]["n_long"]  == 10
    assert records[1]["quarter"] == "2014Q2"


def test_checkpoint_read_nonexistent_returns_empty(tmp_path):
    records = read_quarter_checkpoints("no_such_run", base_dir=tmp_path)
    assert records == []


def test_checkpoint_handles_malformed_jsonl_line(tmp_path):
    write_quarter_checkpoint("run_y", "2014Q1", 10, 10, 5, base_dir=tmp_path)
    # Corrupt the file
    from engine.path_c.pead_backtest import _checkpoint_path
    path = _checkpoint_path("run_y", tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("NOT JSON\n")
    write_quarter_checkpoint("run_y", "2014Q2", 12, 11, 3, base_dir=tmp_path)
    records = read_quarter_checkpoints("run_y", base_dir=tmp_path)
    # 2 valid records + 1 skipped malformed line
    assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def test_run_walk_forward_pead_end_to_end(tmp_path):
    """Two-firm 1-quarter happy path."""
    signal = _make_signal_panel([
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
        {"ticker_ibes": "S1", "rdq": datetime.date(2014, 2, 3), "leg": "short",
         "fiscal_yearq": "2014Q1"},
    ])
    returns = _make_returns_panel(
        ["L1", "S1"],
        datetime.date(2014, 2, 1),
        datetime.date(2014, 5, 31),
        daily_ret_const={"L1": 0.002, "S1": -0.001},
    )
    result = run_walk_forward_pead(
        signal_panel=signal,
        returns_panel=returns,
        window_start=datetime.date(2014, 1, 1),
        window_end=datetime.date(2014, 12, 31),
        checkpoint_run_id="test_e2e",
        checkpoint_base_dir=tmp_path,
    )
    assert isinstance(result, WalkForwardPeadResult)
    assert result.n_quarters_processed == 1
    assert result.n_firm_quarters_active == 2
    assert result.annual_turnover_estimate == pytest.approx(252.0 / 60.0)
    # daily_returns has the net column
    assert "r_long_short_net" in result.daily_returns.columns
    # Checkpoint file written
    cp_path = tmp_path / "test_e2e.jsonl"
    assert cp_path.exists()
    cp_records = read_quarter_checkpoints("test_e2e", base_dir=tmp_path)
    assert len(cp_records) == 1
    assert cp_records[0]["quarter"] == "2014Q1"


def test_run_walk_forward_pead_empty_signal_returns_empty():
    result = run_walk_forward_pead(
        signal_panel=pd.DataFrame(),
        returns_panel=_make_returns_panel(["A"], datetime.date(2014, 2, 1), datetime.date(2014, 5, 31), 0.0),
        window_start=datetime.date(2014, 1, 1),
        window_end=datetime.date(2014, 12, 31),
    )
    assert result.daily_returns.empty
    assert result.n_quarters_processed == 0
    assert result.n_firm_quarters_active == 0


def test_run_walk_forward_pead_locked_constants_used():
    """Default hold_trading_days and tc_bps_roundtrip match spec §六."""
    signal = _make_signal_panel([
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
    ])
    returns = _make_returns_panel(["L1"], datetime.date(2014, 2, 1), datetime.date(2014, 5, 31), 0.001)
    result = run_walk_forward_pead(
        signal_panel=signal,
        returns_panel=returns,
        window_start=datetime.date(2014, 1, 1),
        window_end=datetime.date(2014, 12, 31),
    )
    assert result.tc_bps_roundtrip == TC_BPS_ROUNDTRIP_LOCKED


def test_persist_walk_forward_result(tmp_path):
    """Walk-forward output persisted to parquet."""
    signal = _make_signal_panel([
        {"ticker_ibes": "L1", "rdq": datetime.date(2014, 2, 3), "leg": "long",
         "fiscal_yearq": "2014Q1"},
        {"ticker_ibes": "S1", "rdq": datetime.date(2014, 2, 3), "leg": "short",
         "fiscal_yearq": "2014Q1"},
    ])
    returns = _make_returns_panel(["L1", "S1"], datetime.date(2014, 2, 1), datetime.date(2014, 5, 31), 0.001)
    result = run_walk_forward_pead(
        signal_panel=signal,
        returns_panel=returns,
        window_start=datetime.date(2014, 1, 1),
        window_end=datetime.date(2014, 12, 31),
    )
    parquet_path = tmp_path / "walk_forward_pead.parquet"
    persist_walk_forward_result(result, parquet_path=parquet_path)
    assert parquet_path.exists()
    # Read back
    df = pd.read_parquet(parquet_path)
    assert "r_long_short" in df.columns
    assert "r_long_short_net" in df.columns
