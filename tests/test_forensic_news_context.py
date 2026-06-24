"""
Sprint H follow-up tests — engine/forensic/news_context.py

Spec: docs/spec_forensic_news_context_v1.md
"""
from __future__ import annotations

import datetime
import json
import os
from unittest.mock import patch

import pytest

# conftest.py uses engine.memory.Base for create_all; engine.db_models has its
# OWN Base. Ensure db_models tables exist in test DB before any test runs.
@pytest.fixture(scope="session", autouse=True)
def _ensure_db_models_tables_exist():
    from engine.db_models import Base as DBBase
    from engine.memory import engine as memory_engine
    DBBase.metadata.create_all(memory_engine)
    yield


from engine.forensic.news_context import (
    ForensicNewsSummary,
    _build_headlines_text,
    fetch_news_window,
    investigate_trade,
)


# ───────────────────────────────────────────────────────────────────────────
# Test 1 — dataclass roundtrip + markdown rendering
# ───────────────────────────────────────────────────────────────────────────

def test_forensic_news_summary_dataclass():
    """ForensicNewsSummary frozen dataclass + to_markdown/to_json work."""
    s = ForensicNewsSummary(
        date                  = datetime.date(2026, 6, 15),
        ticker                = "NVDA",
        strategy_name         = "D_PEAD",
        signal_value          = 2.31,
        weight                = 0.04,
        realized_return       = -0.154,
        expected_horizon_days = 60,
        date_window_start     = datetime.date(2026, 6, 10),
        date_window_end       = datetime.date(2026, 6, 20),
        n_articles            = 12,
        n_sources             = 3,
        material_events       = ("EU antitrust probe announced 6/13",),
        macro_context         = "Tech sector under selling pressure",
        sentiment_assessment  = "Sharply bearish post-announcement",
        signal_alignment      = "Signal was strong long but exogenous shock dominated",
        key_quotes            = ('"EU launches formal antitrust probe into NVIDIA AI chips" — Reuters',),
        forensic_verdict      = "case_c",
        cost_usd              = 0.0012,
        llm_model             = "gemini-2.5-flash",
        llm_latency_ms        = 1830,
        extracted_at_utc      = datetime.datetime(2026, 6, 20, 10, 30),
    )

    # Frozen
    with pytest.raises(dataclasses_error()):
        s.ticker = "META"  # type: ignore[misc]

    md = s.to_markdown()
    assert "NVDA" in md
    assert "case_c" in md
    assert "Exogenous Shock" in md
    assert "EU antitrust probe" in md
    assert "$0.0012" in md

    j = s.to_json()
    parsed = json.loads(j)
    assert parsed["date"]                == "2026-06-15"
    assert parsed["forensic_verdict"]    == "case_c"
    assert parsed["material_events"]     == ["EU antitrust probe announced 6/13"]


def dataclasses_error():
    """Return the exception class raised when mutating frozen dataclass."""
    return dataclasses.FrozenInstanceError  # type: ignore[name-defined]


import dataclasses  # noqa: E402  (after dataclasses_error definition)


# ───────────────────────────────────────────────────────────────────────────
# Test 2 — _build_headlines_text formatting
# ───────────────────────────────────────────────────────────────────────────

def test_build_headlines_text_empty_and_populated():
    """Empty headlines → placeholder; populated → numbered list."""
    empty = _build_headlines_text([])
    assert "No news headlines" in empty

    text = _build_headlines_text([
        {"title": "Earnings beat", "source": "Reuters", "published": "2026-06-15"},
        {"title": "FDA approval", "source": "BBG", "published": "2026-06-14", "sentiment_label": "Bullish"},
    ])
    assert "1." in text and "Earnings beat" in text and "Reuters" in text
    assert "2." in text and "FDA approval" in text and "[Bullish]" in text


# ───────────────────────────────────────────────────────────────────────────
# Test 3 — fetch_news_window date window calculation
# ───────────────────────────────────────────────────────────────────────────

def test_fetch_news_window_returns_window_dates(monkeypatch):
    """Even with mocked empty AV + RSS, window_start/end are computed correctly."""
    from engine.forensic import news_context as nc
    from engine.news import NewsPerceiver

    # Mock historical AV + RSS to return empty
    def fake_hist_av(ticker, ws, we, av_key, n_max=30):
        return []
    def fake_rss(self, ticker, sector_name, n=4):
        return []
    monkeypatch.setattr(nc, "_fetch_alpha_vantage_historical", fake_hist_av)
    monkeypatch.setattr(NewsPerceiver, "_fetch_rss", fake_rss)

    headlines, ws, we, n_sources = fetch_news_window(
        "AAPL", datetime.date(2026, 6, 15), window_days=5,
    )
    assert ws == datetime.date(2026, 6, 10)
    assert we == datetime.date(2026, 6, 20)
    assert headlines == []
    assert n_sources == 0


def test_fetch_alpha_vantage_historical_no_key_returns_empty():
    """Without AV key, helper returns empty list (no API call)."""
    from engine.forensic.news_context import _fetch_alpha_vantage_historical
    result = _fetch_alpha_vantage_historical(
        "AAPL",
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 10),
        av_key="",  # empty key
    )
    assert result == []


def test_fetch_alpha_vantage_historical_filters_window(monkeypatch):
    """Defense-in-depth: drops headlines outside the date window even if AV returns broader range."""
    import json
    from engine.forensic.news_context import _fetch_alpha_vantage_historical

    fake_feed = {
        "feed": [
            {"title": "in window", "time_published": "20240105T120000",
             "source": "Reuters", "overall_sentiment_label": "Neutral",
             "overall_sentiment_score": 0.1},
            {"title": "outside window", "time_published": "20240201T120000",
             "source": "Reuters", "overall_sentiment_label": "Neutral",
             "overall_sentiment_score": 0.0},
        ]
    }

    class FakeResp:
        def json(self): return fake_feed
    def fake_get(url, timeout=15):
        return FakeResp()

    monkeypatch.setattr("requests.get", fake_get)

    result = _fetch_alpha_vantage_historical(
        "AAPL",
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 10),
        av_key="fake_key",
    )
    titles = [h["title"] for h in result]
    assert "in window" in titles
    assert "outside window" not in titles


# ───────────────────────────────────────────────────────────────────────────
# Test 4 — smoke E2E (real LLM call; gated on Vertex ADC)
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    os.environ.get("FORENSIC_NEWS_SMOKE_E2E") != "1",
    reason="E2E smoke requires FORENSIC_NEWS_SMOKE_E2E=1 + Vertex ADC + AV_KEY",
)
def test_smoke_investigate_trade_e2e():
    """Real LLM call on AAPL recent date. Validates the full pipeline.

    Skipped by default to keep CI fast / avoid cost. Enable explicitly via
    FORENSIC_NEWS_SMOKE_E2E=1 environment variable.
    """
    summary = investigate_trade(
        date                  = datetime.date.today() - datetime.timedelta(days=2),
        ticker                = "AAPL",
        signal_value          = 1.0,
        weight                = 0.03,
        realized_return       = -0.05,
        strategy_name         = "D_PEAD",
        expected_horizon_days = 60,
    )
    assert summary.ticker == "AAPL"
    assert summary.forensic_verdict in ("case_a", "case_b", "case_c")
    assert summary.cost_usd > 0
    assert summary.llm_latency_ms > 0
    assert len(summary.to_markdown()) > 100
