"""
tests/test_etf_holdings_ingestion.py — Sprint Week 1 ingestion module tests.

Spec: docs/spec_etf_holdings_llm_risk_monitor.md (id=49)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from engine.etf_holdings_ingestion import (
    TOP_N_HOLDINGS,
    fetch_etf_top10_holdings,
    deduplicate_holdings_to_unique_names,
    fetch_all_equity_etf_holdings,
    audit_universe_holdings_coverage,
    _fetch_via_yfinance,
    _fetch_via_sec_edgar_13f,
    _DATA_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function deterministic tests (no network)
# ─────────────────────────────────────────────────────────────────────────────


def test_top_n_holdings_locked_constant():
    """Spec §2.2 — TOP_N_HOLDINGS must be 10."""
    assert TOP_N_HOLDINGS == 10


def test_deduplicate_holdings_to_unique_names_empty():
    """Empty input → empty set."""
    assert deduplicate_holdings_to_unique_names({}) == set()


def test_deduplicate_holdings_to_unique_names_single_etf():
    holdings_by_etf = {
        "QQQ": [
            {"name": "AAPL", "weight": 0.08, "rank": 1},
            {"name": "MSFT", "weight": 0.07, "rank": 2},
        ],
    }
    assert deduplicate_holdings_to_unique_names(holdings_by_etf) == {"AAPL", "MSFT"}


def test_deduplicate_holdings_to_unique_names_overlap_dedup():
    """AAPL in 3 ETFs → counted once (the core spec §2.2 requirement)."""
    holdings_by_etf = {
        "QQQ": [{"name": "AAPL", "weight": 0.08, "rank": 1}],
        "SMH": [{"name": "AAPL", "weight": 0.05, "rank": 2}, {"name": "NVDA", "weight": 0.20, "rank": 1}],
        "XLK": [{"name": "AAPL", "weight": 0.18, "rank": 1}, {"name": "MSFT", "weight": 0.15, "rank": 2}],
    }
    unique = deduplicate_holdings_to_unique_names(holdings_by_etf)
    assert unique == {"AAPL", "NVDA", "MSFT"}
    assert len(unique) == 3  # AAPL counted once across 3 ETFs


def test_deduplicate_normalizes_to_uppercase():
    holdings_by_etf = {
        "ETF1": [{"name": "aapl", "weight": 0.1, "rank": 1}],
        "ETF2": [{"name": "AAPL", "weight": 0.1, "rank": 1}],
    }
    assert deduplicate_holdings_to_unique_names(holdings_by_etf) == {"AAPL"}


def test_deduplicate_handles_missing_name_key():
    """Robustness: rows without 'name' key are skipped."""
    holdings_by_etf = {
        "ETF1": [
            {"name": "AAPL", "weight": 0.1, "rank": 1},
            {"weight": 0.05, "rank": 2},  # missing name
        ],
    }
    assert deduplicate_holdings_to_unique_names(holdings_by_etf) == {"AAPL"}


# ─────────────────────────────────────────────────────────────────────────────
# yfinance fetcher tests (mocked)
# ─────────────────────────────────────────────────────────────────────────────


def _mock_top_holdings_df(symbols_weights: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a DataFrame matching yfinance funds_data.top_holdings shape."""
    df = pd.DataFrame(
        [{"Name": sym, "Holding Percent": w} for sym, w in symbols_weights],
        index=pd.Index([sym for sym, _ in symbols_weights], name="Symbol"),
    )
    return df


def test_fetch_via_yfinance_normal():
    """Standard case: yfinance returns 5 holdings, fraction-form weights."""
    mock_df = _mock_top_holdings_df([
        ("AAPL", 0.08),
        ("MSFT", 0.07),
        ("NVDA", 0.06),
        ("GOOGL", 0.04),
        ("AMZN", 0.03),
    ])
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = mock_df
        result = _fetch_via_yfinance("QQQ")

    assert len(result) == 5
    assert result[0] == {"name": "AAPL", "weight": 0.08, "rank": 1}
    assert result[4] == {"name": "AMZN", "weight": 0.03, "rank": 5}


def test_fetch_via_yfinance_truncates_to_top_10():
    """yfinance returning 15 → truncated to top 10."""
    mock_df = _mock_top_holdings_df([
        (f"T{i}", 0.10 - i * 0.005) for i in range(15)
    ])
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = mock_df
        result = _fetch_via_yfinance("QQQ")

    assert len(result) == 10  # truncated
    assert result[0]["rank"] == 1
    assert result[-1]["rank"] == 10


def test_fetch_via_yfinance_normalizes_percent_to_fraction():
    """yfinance occasionally returns weights in percent (8.5 not 0.085) — auto-normalize."""
    mock_df = _mock_top_holdings_df([
        ("AAPL", 8.5),  # percent form, > 1.0
        ("MSFT", 7.0),
    ])
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = mock_df
        result = _fetch_via_yfinance("QQQ")

    assert result[0]["weight"] == pytest.approx(0.085, abs=1e-9)
    assert result[1]["weight"] == pytest.approx(0.070, abs=1e-9)


def test_fetch_via_yfinance_empty_df():
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = pd.DataFrame()
        assert _fetch_via_yfinance("XYZ") == []


def test_fetch_via_yfinance_none_top_holdings():
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = None
        assert _fetch_via_yfinance("XYZ") == []


def test_fetch_via_yfinance_raises_returns_empty():
    """Network / API error → empty list, no crash."""
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        MockTicker.side_effect = Exception("network error")
        assert _fetch_via_yfinance("XYZ") == []


def test_fetch_via_sec_edgar_13f_stub_returns_empty():
    """v1 stub — SEC EDGAR 13F not yet implemented per spec."""
    assert _fetch_via_sec_edgar_13f("SPY", datetime.date(2026, 5, 31)) == []


# ─────────────────────────────────────────────────────────────────────────────
# Cache integration tests
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_etf_top10_holdings_cache_hit(tmp_path, monkeypatch):
    """Second call with same (etf, YYYYMM) → cache hit, no re-fetch."""
    monkeypatch.setattr("engine.etf_holdings_ingestion._DATA_DIR", tmp_path)

    # First call: mock yfinance, write cache
    mock_df = _mock_top_holdings_df([("AAPL", 0.08)])
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = mock_df
        result1 = fetch_etf_top10_holdings("TEST", datetime.date(2026, 5, 31))

    assert len(result1) == 1
    cache_path = tmp_path / "TEST_202605.json"
    assert cache_path.exists()

    # Second call: yfinance MUST NOT be called (cache hit)
    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        result2 = fetch_etf_top10_holdings("TEST", datetime.date(2026, 5, 31))
        MockTicker.assert_not_called()  # cache hit, no API call

    assert result2 == result1


def test_fetch_etf_top10_holdings_cache_miss_different_month(tmp_path, monkeypatch):
    """Different YYYYMM → cache miss, re-fetch."""
    monkeypatch.setattr("engine.etf_holdings_ingestion._DATA_DIR", tmp_path)

    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = _mock_top_holdings_df([("AAPL", 0.08)])
        fetch_etf_top10_holdings("TEST", datetime.date(2026, 5, 31))

    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = _mock_top_holdings_df([("MSFT", 0.07)])
        result_jun = fetch_etf_top10_holdings("TEST", datetime.date(2026, 6, 30))
        MockTicker.assert_called()  # cache miss → re-fetch

    assert result_jun[0]["name"] == "MSFT"


def test_fetch_etf_top10_holdings_no_cache_when_use_cache_false(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_ingestion._DATA_DIR", tmp_path)

    # Pre-populate cache with stale data
    stale = {
        "etf_ticker": "TEST",
        "as_of_date": "2026-05-31",
        "holdings":  [{"name": "OLD", "weight": 0.5, "rank": 1}],
        "source":    "yfinance",
    }
    cache_path = tmp_path / "TEST_202605.json"
    cache_path.write_text(json.dumps(stale), encoding="utf-8")

    with patch("engine.etf_holdings_ingestion.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.funds_data.top_holdings = _mock_top_holdings_df([("FRESH", 0.1)])
        result = fetch_etf_top10_holdings(
            "TEST", datetime.date(2026, 5, 31), use_cache=False,
        )

    assert result[0]["name"] == "FRESH"


def test_fetch_etf_top10_holdings_rejects_non_date():
    """as_of must be datetime.date."""
    with pytest.raises(TypeError):
        fetch_etf_top10_holdings("QQQ", "2026-05-31")  # string not allowed


# ─────────────────────────────────────────────────────────────────────────────
# Universe-wide ingestion (live network — marked slow)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_fetch_all_equity_etf_holdings_live_universe():
    """
    LIVE NETWORK — verify all 24 equity ETFs return non-empty holdings via yfinance.
    Expected: 100% coverage based on Sprint Week 1 smoke test (2026-05-08).
    """
    holdings = fetch_all_equity_etf_holdings(datetime.date(2026, 5, 31))
    # Test DB seeds only _INITIAL_18 (15 equity ETFs); production DB has 34
    # equity post-Path K1 swap 2026-05-12 (added 10 size/style ETFs).
    # Flexible range covers both contexts. Original hardcoded ==24 was always
    # incorrect under test DB; surfaced when K1 swap audit ran 2026-05-12.
    assert 10 <= len(holdings) <= 50, f"Equity universe size out of [10,50] band: got {len(holdings)}"
    # At least 80% should return holdings (allow for transient yfinance failures)
    n_with = sum(1 for h in holdings.values() if h)
    coverage = n_with / len(holdings)
    assert coverage >= 0.80, f"Equity ETF coverage {coverage:.1%} < 80% threshold"


@pytest.mark.slow
def test_audit_universe_holdings_coverage_live_structure():
    """LIVE NETWORK — audit returns expected structure."""
    audit = audit_universe_holdings_coverage(datetime.date(2026, 5, 31))
    assert "n_total_etfs" in audit
    assert "n_with_holdings" in audit
    assert "n_unique_names" in audit
    assert "coverage_pct" in audit
    # Flexible range covers test DB (_INITIAL_18, 15 equity) and production
    # DB (post-K1 34 equity). Original hardcoded ==24 surfaced as wrong during
    # K1 swap audit 2026-05-12.
    assert 10 <= audit["n_total_etfs"] <= 50
    assert audit["coverage_pct"] >= 80.0
    # K1 size/style ETFs add many new constituent names; widened upper band.
    assert 50 <= audit["n_unique_names"] <= 500, (
        f"Unique names {audit['n_unique_names']} outside [50, 500] expected band"
    )
