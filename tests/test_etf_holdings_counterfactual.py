"""
tests/test_etf_holdings_counterfactual.py — Sprint Week 5 counterfactual tests.

Spec: docs/spec_etf_holdings_llm_risk_monitor.md (id=49) §2.9 + §3.1 L1
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from engine import etf_holdings_counterfactual as ehrc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_portfolio_weights(weights_dict: dict[str, float]):
    """Create mock PortfolioWeights-like object."""
    obj = MagicMock()
    obj.weights = pd.Series(weights_dict, name="weight")
    obj.warnings = []
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# _portfolio_weights_to_dict
# ─────────────────────────────────────────────────────────────────────────────


def test_portfolio_weights_to_dict_from_series():
    pw = _make_portfolio_weights({"QQQ": 0.20, "XLF": 0.15, "GLD": 0.10})
    result = ehrc._portfolio_weights_to_dict(pw)
    assert result == {"QQQ": 0.20, "XLF": 0.15, "GLD": 0.10}


def test_portfolio_weights_to_dict_filters_zero():
    pw = _make_portfolio_weights({"QQQ": 0.20, "XLF": 0.0, "GLD": 0.10})
    result = ehrc._portfolio_weights_to_dict(pw)
    assert "XLF" not in result  # zero weight filtered
    assert result == {"QQQ": 0.20, "GLD": 0.10}


def test_portfolio_weights_to_dict_handles_none():
    assert ehrc._portfolio_weights_to_dict(None) == {}


def test_portfolio_weights_to_dict_handles_dict_attr():
    obj = MagicMock()
    obj.weights = {"QQQ": 0.20, "XLF": 0.10}
    obj.warnings = []
    assert ehrc._portfolio_weights_to_dict(obj) == {"QQQ": 0.20, "XLF": 0.10}


# ─────────────────────────────────────────────────────────────────────────────
# compute_dual_track_snapshot — mocked construct_portfolio
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_dual_track_snapshot_no_caps_active():
    """Both tracks identical when no caps active → zero capped_etfs."""
    pw = _make_portfolio_weights({"QQQ": 0.20, "XLF": 0.15})

    with patch("engine.portfolio.construct_portfolio") as mock_cp:
        mock_cp.return_value = pw  # same return for both calls
        snapshot = ehrc.compute_dual_track_snapshot(
            as_of=datetime.date(2026, 5, 31),
            signal_df=pd.DataFrame(),
            regime=None,
        )

    assert snapshot["status"] == "ok"
    assert snapshot["track_a_weights"] == snapshot["track_b_weights"]
    assert snapshot["n_capped"] == 0


def test_compute_dual_track_snapshot_caps_active():
    """Track A has cap, Track B doesn't → capped_etfs identifies QQQ."""
    pw_a = _make_portfolio_weights({"QQQ": 0.15, "XLF": 0.20})  # QQQ capped
    pw_b = _make_portfolio_weights({"QQQ": 0.25, "XLF": 0.20})  # no cap

    def cp_side_effect(**kwargs):
        if kwargs.get("_disable_etf_holdings_caps", False):
            return pw_b  # Track B
        return pw_a  # Track A

    with patch("engine.portfolio.construct_portfolio", side_effect=cp_side_effect):
        snapshot = ehrc.compute_dual_track_snapshot(
            as_of=datetime.date(2026, 5, 31),
            signal_df=pd.DataFrame(),
            regime=None,
        )

    assert snapshot["status"] == "ok"
    assert "QQQ" in snapshot["capped_etfs"]
    assert "XLF" not in snapshot["capped_etfs"]
    assert snapshot["n_capped"] == 1
    assert snapshot["track_a_weights"]["QQQ"] == 0.15
    assert snapshot["track_b_weights"]["QQQ"] == 0.25


def test_compute_dual_track_snapshot_construct_portfolio_failure():
    """Returns error status if construct_portfolio raises."""
    with patch("engine.portfolio.construct_portfolio", side_effect=Exception("fail")):
        snapshot = ehrc.compute_dual_track_snapshot(
            as_of=datetime.date(2026, 5, 31),
            signal_df=pd.DataFrame(),
            regime=None,
        )
    assert snapshot["status"] == "error"


# ─────────────────────────────────────────────────────────────────────────────
# persist_dual_track_snapshot
# ─────────────────────────────────────────────────────────────────────────────


def test_persist_dual_track_snapshot_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    snapshot = {
        "status":          "ok",
        "snapshot_date":   "2026-05-31",
        "track_a_weights": {"QQQ": 0.15, "XLF": 0.20},
        "track_b_weights": {"QQQ": 0.25, "XLF": 0.20},
        "capped_etfs":     ["QQQ"],
        "n_capped":        1,
    }
    assert ehrc.persist_dual_track_snapshot(snapshot) is True
    df = pd.read_parquet(tmp_path / "snapshots.parquet")
    assert len(df) == 2  # 2 ETFs
    assert "QQQ" in df["etf"].values


def test_persist_dual_track_snapshot_idempotent_re_run(tmp_path, monkeypatch):
    """Running same snapshot date twice → drops old, keeps new."""
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    snapshot1 = {
        "status":          "ok",
        "snapshot_date":   "2026-05-31",
        "track_a_weights": {"QQQ": 0.15},
        "track_b_weights": {"QQQ": 0.25},
        "capped_etfs":     ["QQQ"],
        "n_capped":        1,
    }
    ehrc.persist_dual_track_snapshot(snapshot1)
    snapshot2 = {
        "status":          "ok",
        "snapshot_date":   "2026-05-31",  # same date
        "track_a_weights": {"QQQ": 0.10},  # different
        "track_b_weights": {"QQQ": 0.30},
        "capped_etfs":     ["QQQ"],
        "n_capped":        1,
    }
    ehrc.persist_dual_track_snapshot(snapshot2)
    df = pd.read_parquet(tmp_path / "snapshots.parquet")
    assert len(df) == 1  # not 2 (idempotent — replaces)
    assert df.iloc[0]["track_a_weight"] == 0.10  # latest


def test_persist_dual_track_snapshot_skip_non_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    snapshot = {"status": "error", "error": "test"}
    assert ehrc.persist_dual_track_snapshot(snapshot) is False
    assert not (tmp_path / "snapshots.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# get_latest_dual_track_snapshot
# ─────────────────────────────────────────────────────────────────────────────


def test_get_latest_dual_track_snapshot_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    assert ehrc.get_latest_dual_track_snapshot(datetime.date(2026, 6, 1)) is None


def test_get_latest_dual_track_snapshot_returns_latest_before_as_of(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    # Persist 2 snapshots
    s1 = {
        "status": "ok", "snapshot_date": "2026-04-30",
        "track_a_weights": {"QQQ": 0.10}, "track_b_weights": {"QQQ": 0.25},
        "capped_etfs": ["QQQ"], "n_capped": 1,
    }
    s2 = {
        "status": "ok", "snapshot_date": "2026-05-31",
        "track_a_weights": {"QQQ": 0.15}, "track_b_weights": {"QQQ": 0.25},
        "capped_etfs": ["QQQ"], "n_capped": 1,
    }
    ehrc.persist_dual_track_snapshot(s1)
    ehrc.persist_dual_track_snapshot(s2)

    # Query as of June 1 → should return May 31 snapshot
    result = ehrc.get_latest_dual_track_snapshot(datetime.date(2026, 6, 1))
    assert result is not None
    assert result["snapshot_date"] == "2026-05-31"
    assert result["track_a_weights"]["QQQ"] == 0.15

    # Query as of May 1 → should return April 30 snapshot
    result_may1 = ehrc.get_latest_dual_track_snapshot(datetime.date(2026, 5, 1))
    assert result_may1["snapshot_date"] == "2026-04-30"
    assert result_may1["track_a_weights"]["QQQ"] == 0.10


def test_get_latest_dual_track_snapshot_no_match_too_early(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    s = {
        "status": "ok", "snapshot_date": "2026-05-31",
        "track_a_weights": {"QQQ": 0.15}, "track_b_weights": {"QQQ": 0.25},
        "capped_etfs": ["QQQ"], "n_capped": 1,
    }
    ehrc.persist_dual_track_snapshot(s)
    # Query before snapshot exists
    assert ehrc.get_latest_dual_track_snapshot(datetime.date(2026, 4, 30)) is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_daily_pnl_delta
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_daily_pnl_delta_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._DUAL_TRACK_SNAPSHOTS_PATH",
                        tmp_path / "snapshots.parquet")
    result = ehrc.compute_daily_pnl_delta(datetime.date(2026, 6, 1))
    assert result["status"] == "no_snapshot"
    assert result["delta"] == 0.0


def test_compute_daily_pnl_delta_skipped_pre_snapshot():
    """If as_of <= snapshot_date, status=skipped_pre_snapshot."""
    snapshot = {
        "snapshot_date":   "2026-05-31",
        "track_a_weights": {"QQQ": 0.15}, "track_b_weights": {"QQQ": 0.25},
        "capped_etfs":     ["QQQ"],
    }
    result = ehrc.compute_daily_pnl_delta(
        datetime.date(2026, 5, 31), snapshot=snapshot,
    )
    assert result["status"] == "skipped_pre_snapshot"


def test_compute_daily_pnl_delta_correct_attribution():
    """Delta = Σ (weight_diff × return). Verify exact attribution."""
    snapshot = {
        "snapshot_date":   "2026-05-31",
        "track_a_weights": {"QQQ": 0.15, "XLF": 0.20},  # QQQ capped
        "track_b_weights": {"QQQ": 0.25, "XLF": 0.20},  # no cap
        "capped_etfs":     ["QQQ"],
    }
    # Mock returns: QQQ +1%, XLF +0.5%
    with patch("engine.etf_holdings_counterfactual.fetch_etf_returns_for_date") as mock_fetch:
        mock_fetch.return_value = {"QQQ": 0.01, "XLF": 0.005}
        result = ehrc.compute_daily_pnl_delta(
            datetime.date(2026, 6, 2), snapshot=snapshot,
        )

    assert result["status"] == "ok"
    # Track A: 0.15×0.01 + 0.20×0.005 = 0.0015 + 0.001 = 0.0025
    # Track B: 0.25×0.01 + 0.20×0.005 = 0.0025 + 0.001 = 0.0035
    # Delta: A - B = -0.001 (Track A capped → less exposure to QQQ rally → underperforms)
    assert result["track_a_pnl"] == pytest.approx(0.0025, abs=1e-9)
    assert result["track_b_pnl"] == pytest.approx(0.0035, abs=1e-9)
    assert result["delta"] == pytest.approx(-0.001, abs=1e-9)
    assert result["n_diff_etfs"] == 1  # only QQQ differs


def test_compute_daily_pnl_delta_qqq_drop_cap_helps():
    """If capped ETF DROPS, cap reduction = positive delta (less loss)."""
    snapshot = {
        "snapshot_date":   "2026-05-31",
        "track_a_weights": {"QQQ": 0.15},  # capped
        "track_b_weights": {"QQQ": 0.25},  # no cap
        "capped_etfs":     ["QQQ"],
    }
    with patch("engine.etf_holdings_counterfactual.fetch_etf_returns_for_date") as mock_fetch:
        mock_fetch.return_value = {"QQQ": -0.05}  # QQQ -5%

        result = ehrc.compute_daily_pnl_delta(
            datetime.date(2026, 6, 2), snapshot=snapshot,
        )

    # Track A: 0.15 × -0.05 = -0.0075
    # Track B: 0.25 × -0.05 = -0.0125
    # Delta: A - B = -0.0075 - (-0.0125) = +0.005 (cap helped!)
    assert result["delta"] == pytest.approx(0.005, abs=1e-9)


def test_compute_daily_pnl_delta_missing_returns_default_zero():
    snapshot = {
        "snapshot_date":   "2026-05-31",
        "track_a_weights": {"QQQ": 0.15, "EWS": 0.05},
        "track_b_weights": {"QQQ": 0.25, "EWS": 0.05},
        "capped_etfs":     ["QQQ"],
    }
    with patch("engine.etf_holdings_counterfactual.fetch_etf_returns_for_date") as mock_fetch:
        mock_fetch.return_value = {"QQQ": 0.01}  # EWS missing

        result = ehrc.compute_daily_pnl_delta(
            datetime.date(2026, 6, 2), snapshot=snapshot,
        )

    # EWS missing → contribution 0; only QQQ counted
    assert result["track_a_pnl"] == pytest.approx(0.15 * 0.01, abs=1e-9)
    assert result["track_b_pnl"] == pytest.approx(0.25 * 0.01, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# persist_daily_pnl_delta
# ─────────────────────────────────────────────────────────────────────────────


def test_persist_daily_pnl_delta_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._COUNTERFACTUAL_PNL_PATH",
                        tmp_path / "cf_pnl.parquet")
    record = {
        "status":               "ok",
        "date":                 "2026-06-02",
        "snapshot_date":        "2026-05-31",
        "track_a_pnl":          0.0025,
        "track_b_pnl":          0.0035,
        "delta":                -0.001,
        "n_diff_etfs":          1,
        "capped_etfs":          ["QQQ"],
        "n_etfs_with_returns":  2,
    }
    assert ehrc.persist_daily_pnl_delta(record) is True

    df = pd.read_parquet(tmp_path / "cf_pnl.parquet")
    assert len(df) == 1
    assert df.iloc[0]["delta"] == pytest.approx(-0.001)
    assert df.iloc[0]["capped_etfs"] == "QQQ"


def test_persist_daily_pnl_delta_idempotent_same_date(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._COUNTERFACTUAL_PNL_PATH",
                        tmp_path / "cf_pnl.parquet")
    record1 = {
        "status": "ok", "date": "2026-06-02",
        "snapshot_date": "2026-05-31", "track_a_pnl": 0.001, "track_b_pnl": 0.002,
        "delta": -0.001, "n_diff_etfs": 1, "capped_etfs": ["QQQ"],
        "n_etfs_with_returns": 2,
    }
    record2 = {**record1, "delta": 0.05}  # same date, different delta
    ehrc.persist_daily_pnl_delta(record1)
    ehrc.persist_daily_pnl_delta(record2)
    df = pd.read_parquet(tmp_path / "cf_pnl.parquet")
    assert len(df) == 1  # idempotent
    assert df.iloc[0]["delta"] == pytest.approx(0.05)  # latest wins


# ─────────────────────────────────────────────────────────────────────────────
# compute_cumulative_metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_cumulative_metrics_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._COUNTERFACTUAL_PNL_PATH",
                        tmp_path / "cf_pnl.parquet")
    result = ehrc.compute_cumulative_metrics()
    assert result["status"] == "empty"


def test_compute_cumulative_metrics_basic(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._COUNTERFACTUAL_PNL_PATH",
                        tmp_path / "cf_pnl.parquet")
    # 10 days of synthetic delta data
    deltas = [0.001, -0.0005, 0.002, -0.001, 0.0015, 0.0008, -0.0003, 0.0012, 0.0007, -0.0009]
    rows = [
        {
            "status":               "ok",
            "date":                 f"2026-06-{i+1:02d}",
            "snapshot_date":        "2026-05-31",
            "track_a_pnl":          d + 0.001,
            "track_b_pnl":          0.001,
            "delta":                d,
            "n_diff_etfs":          1,
            "capped_etfs":          "QQQ",
            "n_etfs_with_returns":  2,
        }
        for i, d in enumerate(deltas)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "cf_pnl.parquet", index=False)

    metrics = ehrc.compute_cumulative_metrics()
    assert metrics["status"] == "ok"
    assert metrics["n_obs"] == 10
    assert metrics["n_active_days"] == 10  # all have capped_etfs="QQQ"
    assert metrics["cumulative_delta"] == pytest.approx(sum(deltas), abs=1e-9)
    assert metrics["delta_sharpe_annualized"] != 0  # should compute non-zero
    assert "delta_max_drawdown" in metrics


def test_compute_cumulative_metrics_excludes_non_ok_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_counterfactual._COUNTERFACTUAL_PNL_PATH",
                        tmp_path / "cf_pnl.parquet")
    rows = [
        {"status": "ok", "date": "2026-06-01", "snapshot_date": "2026-05-31",
         "track_a_pnl": 0.001, "track_b_pnl": 0.002, "delta": -0.001,
         "n_diff_etfs": 1, "capped_etfs": "QQQ", "n_etfs_with_returns": 2},
        {"status": "skipped_pre_snapshot", "date": "2026-05-31", "snapshot_date": "2026-05-31",
         "track_a_pnl": 0, "track_b_pnl": 0, "delta": 0,
         "n_diff_etfs": 0, "capped_etfs": "", "n_etfs_with_returns": 0},
        {"status": "no_snapshot", "date": "2026-04-30", "snapshot_date": None,
         "track_a_pnl": 0, "track_b_pnl": 0, "delta": 0,
         "n_diff_etfs": 0, "capped_etfs": "", "n_etfs_with_returns": 0},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "cf_pnl.parquet", index=False)

    metrics = ehrc.compute_cumulative_metrics()
    assert metrics["n_obs"] == 1  # only "ok" status counted
    assert metrics["cumulative_delta"] == pytest.approx(-0.001, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# fetch_etf_returns_for_date — minimal smoke test
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_etf_returns_empty_tickers():
    assert ehrc.fetch_etf_returns_for_date(datetime.date(2026, 5, 31), []) == {}
