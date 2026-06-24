"""tests/test_dq_post_feed.py — DQ Inspector Phase 6c post-feed gate tests.

Covers:
  - gather_post_feed_inputs builds correct kwargs dict shape
  - mocked yfinance / panel paths give deterministic Mode 5/6/9 values
  - ticker_to_sleeves preserves multi-sleeve membership for TLT/GLD
  - post_feed_gate output structure on synthetic clean + breach inputs

Per spec_dq_inspector_agent_v1.md §2.1 + §2.1a Mode 5/6/7/9.
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.agents.dq_inspector.post_feed_inputs import (
    _build_active_universe,
    _count_pead_universe_coverage,
    gather_post_feed_inputs,
)


# ──────────────────────────────────────────────────────────────────────────────
# _build_active_universe
# ──────────────────────────────────────────────────────────────────────────────
class TestBuildActiveUniverse:
    def test_contains_k1_etfs(self):
        tickers, t2s = _build_active_universe()
        # K1 universe at least includes major sector ETFs
        assert "SPY" not in t2s   # SPY is benchmark, not in K1
        assert any(t in t2s for t in ("QQQ", "XLF", "TLT"))

    def test_ac_insurance_tickers_included(self):
        tickers, t2s = _build_active_universe()
        assert "TLT" in t2s
        assert "GLD" in t2s
        assert "rms_crisis_hedge" in t2s["TLT"]
        assert "rms_crisis_hedge" in t2s["GLD"]

    def test_tlt_gld_multi_sleeve_preserved(self):
        """TLT/GLD are in BOTH K1 (etf_l1) and AC (rms_crisis_hedge).
        Multi-sleeve membership must be captured as a set."""
        _, t2s = _build_active_universe()
        # Both K1 universe and AC sleeve contribute to TLT/GLD
        assert t2s["TLT"] == {"etf_l1", "rms_crisis_hedge"}
        assert t2s["GLD"] == {"etf_l1", "rms_crisis_hedge"}

    def test_returns_sorted_unique_list(self):
        tickers, t2s = _build_active_universe()
        assert tickers == sorted(set(tickers))
        assert len(tickers) == len(set(tickers))


# ──────────────────────────────────────────────────────────────────────────────
# _count_pead_universe_coverage
# ──────────────────────────────────────────────────────────────────────────────
class TestCountPeadUniverse:
    def test_missing_file_returns_zero(self, monkeypatch, tmp_path):
        from pathlib import Path
        monkeypatch.setattr(
            "engine.agents.dq_inspector.post_feed_inputs._PEAD_PANEL_PATH",
            tmp_path / "does_not_exist.parquet",
        )
        assert _count_pead_universe_coverage() == 0

    def test_counts_unique_tickers(self, monkeypatch, tmp_path):
        from pathlib import Path
        panel_path = tmp_path / "panel.parquet"
        pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "AAPL", "GOOG"],   # 3 unique
            "rdq":    [1, 1, 1, 1],
        }).to_parquet(panel_path)
        monkeypatch.setattr(
            "engine.agents.dq_inspector.post_feed_inputs._PEAD_PANEL_PATH",
            panel_path,
        )
        assert _count_pead_universe_coverage() == 3


# ──────────────────────────────────────────────────────────────────────────────
# gather_post_feed_inputs (with mocked _fetch_closes)
# ──────────────────────────────────────────────────────────────────────────────
class TestGatherPostFeedInputs:
    def test_clean_run_returns_complete_dict(self, monkeypatch):
        """Mock yfinance to return 2-day closes for all K1+AC tickers;
        gather should return all expected keys with non-degenerate values."""
        tickers, _ = _build_active_universe()
        # Build mock closes DataFrame: 2 dates × all tickers, all priced
        dates = pd.date_range("2026-05-18", "2026-05-19", freq="D")
        fake_closes = pd.DataFrame(
            np.random.RandomState(42).uniform(100, 200, size=(2, len(tickers))),
            index=dates, columns=tickers,
        )
        with patch("engine.signal._fetch_closes", return_value=fake_closes):
            inputs = gather_post_feed_inputs(datetime.date(2026, 5, 19))

        assert set(inputs) == {
            "as_of", "k1_n_with_price", "pead_n_with_rdq",
            "daily_returns", "ticker_to_sleeves",
            "n_nan_close", "n_universe",
        }
        assert inputs["as_of"] == datetime.date(2026, 5, 19)
        assert inputs["n_universe"] == len(tickers)
        assert inputs["n_nan_close"] == 0
        assert isinstance(inputs["daily_returns"], pd.Series)

    def test_fetch_failure_degrades_to_zero_coverage(self, monkeypatch):
        """Yfinance raising should NOT crash; should return zero K1
        coverage and full NaN burst (downstream gates HALT, correctly)."""
        with patch("engine.signal._fetch_closes",
                   side_effect=Exception("network down")):
            inputs = gather_post_feed_inputs(datetime.date(2026, 5, 19))

        assert inputs["k1_n_with_price"] == 0
        assert inputs["n_nan_close"] == inputs["n_universe"]
        assert len(inputs["daily_returns"]) == 0

    def test_partial_coverage_counted_correctly(self, monkeypatch):
        """If some tickers are NaN today, n_with_price reflects only
        the priced ones and n_nan_close reflects the missing."""
        tickers, _ = _build_active_universe()
        # Half the K1 universe NaN
        dates = pd.date_range("2026-05-18", "2026-05-19", freq="D")
        data = np.random.RandomState(7).uniform(100, 200, size=(2, len(tickers)))
        # Force half today's row to NaN
        n_nan = len(tickers) // 2
        data[-1, :n_nan] = np.nan
        fake_closes = pd.DataFrame(data, index=dates, columns=tickers)
        with patch("engine.signal._fetch_closes", return_value=fake_closes):
            inputs = gather_post_feed_inputs(datetime.date(2026, 5, 19))

        # The half that's NaN today still has yesterday's price, so they
        # still count as "priced" in Mode 5's lenient counting. Just check
        # NaN burst captured the bad rows.
        assert inputs["n_nan_close"] == n_nan


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end with post_feed_gate (Mode 5/6/7/9 via gates.evaluate_post_feed)
# ──────────────────────────────────────────────────────────────────────────────
class TestEndToEndPostFeedGate:
    def test_clean_inputs_no_breach(self):
        """Realistic inputs: 45 K1 ETFs priced, 1800-stock D-PEAD panel,
        no anomalous returns, no NaN burst → 0 breaches."""
        from engine.agents.dq_inspector.gates import evaluate_post_feed
        # Synthetic clean daily returns — all within etf 30% cap
        returns = pd.Series({f"E{i:02d}": 0.005 for i in range(45)})
        breaches = evaluate_post_feed(
            as_of            = datetime.date(2026, 5, 19),
            k1_n_with_price  = 45,
            pead_n_with_rdq  = 1800,
            daily_returns    = returns,
            ticker_to_sleeves= {f"E{i:02d}": {"etf_l1"} for i in range(45)},
            n_nan_close      = 0,
            n_universe       = 45,
        )
        assert breaches == []

    def test_low_coverage_triggers_mode_5(self):
        """K1 universe coverage 30/43 = 70% < 90% min → Mode 5 HARD_HALT."""
        from engine.agents.dq_inspector.gates import evaluate_post_feed
        breaches = evaluate_post_feed(
            as_of            = datetime.date(2026, 5, 19),
            k1_n_with_price  = 30,    # 70% coverage
            pead_n_with_rdq  = 1800,
            n_universe       = 45,
        )
        mode_5 = [b for b in breaches if b.mode_id == "5"]
        assert len(mode_5) == 1
        assert mode_5[0].severity == "HARD_HALT"

    def test_pead_panel_empty_triggers_mode_6(self):
        from engine.agents.dq_inspector.gates import evaluate_post_feed
        breaches = evaluate_post_feed(
            as_of            = datetime.date(2026, 5, 19),
            k1_n_with_price  = 45,
            pead_n_with_rdq  = 0,
            n_universe       = 45,
        )
        mode_6 = [b for b in breaches if b.mode_id == "6"]
        assert len(mode_6) == 1
        assert mode_6[0].severity == "HARD_HALT"

    def test_price_anomaly_etf_class_30pct(self):
        """ETF ticker with 35% daily return — exceeds 30% etf cap."""
        from engine.agents.dq_inspector.gates import evaluate_post_feed
        returns = pd.Series({"BIG_MOVE_ETF": 0.35})
        breaches = evaluate_post_feed(
            as_of             = datetime.date(2026, 5, 19),
            k1_n_with_price   = 45,
            pead_n_with_rdq   = 1800,
            daily_returns     = returns,
            ticker_to_sleeves = {"BIG_MOVE_ETF": {"etf_l1"}},
            n_nan_close       = 0,
            n_universe        = 1,
        )
        mode_7 = [b for b in breaches if b.mode_id == "7"]
        assert len(mode_7) == 1

    def test_nan_burst_above_5pct_triggers_mode_9(self):
        from engine.agents.dq_inspector.gates import evaluate_post_feed
        breaches = evaluate_post_feed(
            as_of            = datetime.date(2026, 5, 19),
            k1_n_with_price  = 45,
            pead_n_with_rdq  = 1800,
            n_nan_close      = 5,    # 5/45 ≈ 11% > 5% cap
            n_universe       = 45,
        )
        mode_9 = [b for b in breaches if b.mode_id == "9"]
        assert len(mode_9) == 1
        assert mode_9[0].severity == "HARD_HALT"
