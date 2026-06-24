"""Tests for the 4 non-WRDS fetchers (Phase 6b non-WRDS).

Uses mocked HTTP / mocked yfinance / mocked fredapi to avoid hitting real
APIs in CI. Real-world smoke is gated by env REAL_NETWORK_SMOKE=true and
not run by default.
"""
from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest

from engine.data.fetchers import api_edgar, api_fred, api_yfinance, scraper_wikipedia
from engine.data.orchestrator import ProbeResult


# ── api_yfinance ────────────────────────────────────────────────────────

def test_yfinance_probe_calls_ticker_history(monkeypatch):
    """Probe should call yf.Ticker('SPY').history(period='1d')."""
    called = {}

    class _FakeTicker:
        def __init__(self, sym):
            called["symbol"] = sym
        def history(self, **kw):
            called["history_kw"] = kw
            return pd.DataFrame({"Close": [400.0]},
                                  index=[pd.Timestamp("2024-01-01")])

    fake_yf = mock.MagicMock()
    fake_yf.Ticker.side_effect = _FakeTicker

    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    result = api_yfinance.probe("2024-01-01", "2024-01-31")
    assert result.available is True
    assert called["symbol"] == "SPY"


def test_yfinance_probe_empty_means_unavailable(monkeypatch):
    fake_yf = mock.MagicMock()
    fake_yf.Ticker.return_value.history.return_value = pd.DataFrame()
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    result = api_yfinance.probe("2024-01-01", "2024-01-31")
    assert result.available is False
    assert "empty" in (result.error or "").lower()


def test_yfinance_fetch_equity_daily_normalizes_columns(monkeypatch):
    """Fetcher returns canonical date/ticker/prc/ret columns."""
    fake_yf = mock.MagicMock()
    idx = pd.date_range("2024-01-01", periods=3)
    idx.name = "Date"    # yfinance sets this; required for reset_index
    fake_df = pd.DataFrame({
        "Open":  [100, 101, 102],
        "High":  [101, 102, 103],
        "Low":   [99, 100, 101],
        "Close": [100, 101, 102],
        "Volume": [1000, 1100, 1200],
    }, index=idx)
    fake_yf.download.return_value = fake_df
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    df = api_yfinance.fetch_equity_daily("2024-01-01", "2024-01-04",
                                            tickers=["TEST"])
    assert "date" in df.columns
    assert "ticker" in df.columns
    assert "prc" in df.columns
    assert "ret" in df.columns


# ── api_fred ────────────────────────────────────────────────────────────

def test_fred_probe_without_key_returns_auth_missing(monkeypatch):
    monkeypatch.setattr(api_fred, "get_secret", lambda k: None)
    result = api_fred.probe("2020-01-01", "2020-12-31")
    assert result.available is False
    assert result.error_class == "auth_missing"


def test_fred_probe_with_key_calls_fred(monkeypatch):
    monkeypatch.setattr(api_fred, "get_secret", lambda k: "fake_key")
    fake_fred_module = mock.MagicMock()
    fake_fred = mock.MagicMock()
    fake_series = pd.Series({pd.Timestamp("2020-01-01"): 3.5})
    fake_fred.get_series.return_value = fake_series
    fake_fred_module.Fred.return_value = fake_fred
    monkeypatch.setitem(__import__("sys").modules, "fredapi", fake_fred_module)
    result = api_fred.probe("2020-01-01", "2020-12-31")
    assert result.available is True


def test_fred_fetch_series_returns_canonical(monkeypatch):
    monkeypatch.setattr(api_fred, "get_secret", lambda k: "fake_key")
    fake_fred_module = mock.MagicMock()
    fake_fred = mock.MagicMock()
    fake_series = pd.Series(
        [3.5, 3.6, 3.4],
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    fake_fred.get_series.return_value = fake_series
    fake_fred_module.Fred.return_value = fake_fred
    monkeypatch.setitem(__import__("sys").modules, "fredapi", fake_fred_module)

    df = api_fred.fetch_series("2020-01-01", "2020-12-31", series_id="UNRATE")
    assert list(df.columns) == ["date", "series_id", "value"]
    assert (df["series_id"] == "UNRATE").all()
    assert len(df) == 3


# ── api_edgar ───────────────────────────────────────────────────────────

def test_edgar_probe_http_403_means_access_denied(monkeypatch):
    """EDGAR returns 403 if User-Agent is wrong. Should report access_denied."""
    class _FakeResp:
        status_code = 403
        def raise_for_status(self): pass

    fake_session = mock.MagicMock()
    fake_session.get.return_value = _FakeResp()
    monkeypatch.setattr(api_edgar, "http_session", lambda **kw: fake_session)
    result = api_edgar.probe("2024-01-01", "2024-01-01")
    assert result.available is False
    assert result.error_class == "access_denied"


def test_edgar_probe_http_429_means_rate_limited(monkeypatch):
    class _FakeResp:
        status_code = 429
        def raise_for_status(self): pass

    fake_session = mock.MagicMock()
    fake_session.get.return_value = _FakeResp()
    monkeypatch.setattr(api_edgar, "http_session", lambda **kw: fake_session)
    result = api_edgar.probe("2024-01-01", "2024-01-01")
    assert result.available is False
    assert result.error_class == "rate_limited"


def test_edgar_probe_200_means_available(monkeypatch):
    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass

    fake_session = mock.MagicMock()
    fake_session.get.return_value = _FakeResp()
    monkeypatch.setattr(api_edgar, "http_session", lambda **kw: fake_session)
    result = api_edgar.probe("2024-01-01", "2024-01-01")
    assert result.available is True


def test_edgar_fetch_8k_returns_canonical_columns(monkeypatch):
    """Mock EDGAR JSON response; verify column parsing."""
    fake_payload = {
        "hits": {
            "hits": [
                {"_id": "0001-23-456", "_source": {
                    "file_date": "2024-01-15",
                    "ciks": ["0000320193"],
                    "adsh": "0001-23-456",
                    "form": "8-K",
                }},
                {"_id": "0001-23-457", "_source": {
                    "file_date": "2024-01-16",
                    "ciks": ["0000789019"],
                    "adsh": "0001-23-457",
                    "form": "8-K",
                }},
            ]
        }
    }

    class _FakeResp:
        status_code = 200
        def json(self): return fake_payload
        def raise_for_status(self): pass

    fake_session = mock.MagicMock()
    fake_session.get.return_value = _FakeResp()
    monkeypatch.setattr(api_edgar, "http_session", lambda **kw: fake_session)
    df = api_edgar.fetch_8k_meta("2024-01-01", "2024-01-31", max_results=2)
    assert list(df.columns) == ["filing_date", "cik", "accession_no",
                                  "filing_type", "link"]
    assert len(df) == 2


# ── scraper_wikipedia ───────────────────────────────────────────────────

def test_wikipedia_probe_200_html(monkeypatch):
    class _FakeResp:
        status_code = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        def raise_for_status(self): pass

    fake_session = mock.MagicMock()
    fake_session.head.return_value = _FakeResp()
    monkeypatch.setattr(scraper_wikipedia, "http_session", lambda **kw: fake_session)
    result = scraper_wikipedia.probe("2024-01-01", "2024-01-31")
    assert result.available is True


def test_wikipedia_probe_non_html_means_schema_unknown(monkeypatch):
    class _FakeResp:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        def raise_for_status(self): pass

    fake_session = mock.MagicMock()
    fake_session.head.return_value = _FakeResp()
    monkeypatch.setattr(scraper_wikipedia, "http_session", lambda **kw: fake_session)
    result = scraper_wikipedia.probe("2024-01-01", "2024-01-31")
    assert result.available is False
    assert result.error_class == "schema_unknown"


def test_wikipedia_layer1_canonical_table():
    """Layer 1 should find a well-structured constituents table."""
    html = """
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
      <tr><td>MSFT</td><td>Microsoft Corp.</td><td>Information Technology</td></tr>
    </table>
    """
    df = scraper_wikipedia._layer1_pd_read_html(html)
    assert df is not None
    assert "ticker" in df.columns
    assert "AAPL" in df["ticker"].values


def test_wikipedia_layer2_semantic_table():
    """Layer 2 should find a table without the canonical id."""
    html = """
    <table class="new-style-table">
      <tr><th>Stock Symbol</th><th>Company Name</th><th>Sector</th></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td><td>Tech</td></tr>
    </table>
    """
    df = scraper_wikipedia._layer2_semantic_bs(html)
    assert df is not None
    assert "ticker" in df.columns


def test_wikipedia_layer3_disabled_without_env():
    """Layer 3 should return None when env var not set."""
    import os
    os.environ.pop("LAYER3_LLM_RESCUE", None)
    df = scraper_wikipedia._layer3_llm_rescue("<html>...</html>")
    assert df is None


def test_wikipedia_normalize_handles_variants():
    """Column normalizer handles common variants."""
    tbl = pd.DataFrame({
        "Stock Symbol": ["AAPL", "MSFT"],
        "Company Name": ["Apple", "Microsoft"],
        "GICS Sector":  ["Tech", "Tech"],
    })
    out = scraper_wikipedia._normalize_columns(tbl)
    assert list(out.columns) == ["ticker", "name", "sector", "date_added"]


def test_wikipedia_normalize_handles_missing_ticker():
    """If no ticker-like column, returns empty."""
    tbl = pd.DataFrame({"name": ["A"], "sector": ["X"]})
    out = scraper_wikipedia._normalize_columns(tbl)
    assert out.empty


# ── _common utilities ───────────────────────────────────────────────────

def test_common_to_utc_dates():
    from engine.data.fetchers._common import to_utc_dates
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-15"]))
    out = to_utc_dates(s)
    assert out.dt.tz is None    # stripped to naive UTC
    assert out.iloc[0].year == 2024


def test_common_replace_sentinels():
    from engine.data.fetchers._common import replace_sentinel_values
    df = pd.DataFrame({"x": [1.0, 9999.99, 2.0, -9999.0]})
    out = replace_sentinel_values(df)
    assert pd.isna(out["x"].iloc[1])
    assert pd.isna(out["x"].iloc[3])
    assert out["x"].iloc[0] == 1.0
