"""Phase 8b tests: NBER + Tier-1 RSS + multi-source dispatcher."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from engine.data import source_health
from engine.research.discovery import (
    nber_fetcher,
    tier1_rss_fetcher,
    multi_source_dispatch,
)


@pytest.fixture(autouse=True)
def isolated_health(tmp_path, monkeypatch):
    monkeypatch.setattr(source_health, "HEALTH_FILE",
                          tmp_path / "source_health.json")
    yield


# ── NBER fetcher ────────────────────────────────────────────────────────

SAMPLE_NBER_RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>NBER</title>
    <item>
      <title>Test Finance Paper</title>
      <link>https://www.nber.org/papers/w32999</link>
      <description>This paper studies factor premia.</description>
      <pubDate>Mon, 15 Jan 2024 00:00:00 GMT</pubDate>
      <dc:creator>Alice Author</dc:creator>
    </item>
  </channel>
</rss>"""


def test_nber_rss_parses(monkeypatch):
    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            class _R:
                status_code = 200
                content = SAMPLE_NBER_RSS
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(nber_fetcher, "POLITE_DELAY_SEC", 0)
    df = nber_fetcher.fetch_nber_recent()
    assert not df.empty
    assert df.iloc[0]["nber_id"] == "w32999"
    assert "Finance" in df.iloc[0]["title"]


def test_nber_skips_when_unhealthy():
    source_health.mark_failure("nber_rss", "rate_limited", "test")
    df = nber_fetcher.fetch_nber_recent()
    assert df.empty


def test_nber_marks_unhealthy_on_429(monkeypatch):
    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            class _R:
                status_code = 429
                content = b""
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(nber_fetcher, "POLITE_DELAY_SEC", 0)
    df = nber_fetcher.fetch_nber_recent(skip_health_check=True)
    assert df.empty
    healthy, _ = source_health.is_healthy("nber_rss")
    assert healthy is False


def test_nber_api_filters_jel():
    """API parser includes paper when JEL matches target."""
    item = {
        "paper": "32999", "title": "Title",
        "abstract": "abs", "authors": ["Alice"],
        "jel_codes": ["G12", "C10"],
        "public_date": "2024-01-15",
        "pdf_url": None, "url": None,
    }
    # G12 starts with G1 → matches if target includes G12 prefix-style
    row = nber_fetcher._parse_api_item(item, "2024-01-01", "2024-12-31",
                                          target_jel_codes=("G12",))
    assert row is not None
    assert row["nber_id"] == "w32999"


def test_nber_api_filters_out_non_jel():
    item = {
        "paper": "32999", "title": "Title", "abstract": "abs",
        "authors": ["Alice"], "jel_codes": ["C10", "D40"],
        "public_date": "2024-01-15",
    }
    row = nber_fetcher._parse_api_item(item, "2024-01-01", "2024-12-31",
                                          target_jel_codes=("G10",))
    assert row is None


# ── Tier-1 RSS aggregator ───────────────────────────────────────────────

SAMPLE_RSS2 = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>JFE</title>
    <item>
      <title>A Journal Paper</title>
      <link>https://example.com/paper1</link>
      <description>Description of paper.</description>
      <pubDate>Mon, 15 Jan 2024 00:00:00 GMT</pubDate>
      <dc:creator>Bob Author</dc:creator>
    </item>
  </channel>
</rss>"""


def test_tier1_rss_loads_default_feeds(monkeypatch, tmp_path):
    """If config file missing, falls back to defaults."""
    monkeypatch.setattr(tier1_rss_fetcher, "FEED_CONFIG", tmp_path / "missing.yaml")
    feeds = tier1_rss_fetcher.load_feed_config()
    assert len(feeds) >= 1
    assert all("id" in f and "url" in f for f in feeds)


def test_tier1_rss_loads_yaml(monkeypatch, tmp_path):
    config = tmp_path / "feeds.yaml"
    config.write_text("""
feeds:
  - id: test_feed
    name: Test
    url: https://example.com/feed
    publisher: Test
""", encoding="utf-8")
    monkeypatch.setattr(tier1_rss_fetcher, "FEED_CONFIG", config)
    feeds = tier1_rss_fetcher.load_feed_config()
    assert len(feeds) == 1
    assert feeds[0]["id"] == "test_feed"


def test_tier1_rss_parses_single_feed(monkeypatch):
    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            class _R:
                status_code = 200
                content = SAMPLE_RSS2
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(tier1_rss_fetcher, "POLITE_DELAY_SEC", 0)
    feeds = [{"id": "test_feed", "name": "Test", "url": "x"}]
    df = tier1_rss_fetcher.fetch_tier1_rss(feeds=feeds)
    assert not df.empty
    assert "Journal Paper" in df.iloc[0]["title"]
    assert df.iloc[0]["journal"] == "Test"


def test_tier1_per_feed_health_independent(monkeypatch):
    """Marking one feed unhealthy doesn't block others."""
    source_health.mark_failure("tier1_rss_feed_a", "rate_limited", "test")
    # Feed B should still be attempted

    captured = []
    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            captured.append(url)
            class _R:
                status_code = 200
                content = SAMPLE_RSS2
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(tier1_rss_fetcher, "POLITE_DELAY_SEC", 0)
    feeds = [
        {"id": "feed_a", "name": "Feed A", "url": "x"},
        {"id": "feed_b", "name": "Feed B", "url": "y"},
    ]
    tier1_rss_fetcher.fetch_tier1_rss(feeds=feeds)
    # feed_a skipped due to health; feed_b attempted
    assert all("y" in u for u in captured)


# ── multi_source_dispatch ───────────────────────────────────────────────

def test_dedup_across_sources_high_overlap_removed():
    df = pd.DataFrame([
        {"source": "arxiv", "source_id": "a", "title": "Cross Sectional Momentum Returns",
         "authors": "x", "abstract": "y", "categories": "",
         "submitted_date": None, "updated_date": None,
         "pdf_url": None, "abs_url": None},
        {"source": "nber",  "source_id": "b", "title": "Cross Sectional Momentum Returns",
         "authors": "x", "abstract": "y", "categories": "",
         "submitted_date": None, "updated_date": None,
         "pdf_url": None, "abs_url": None},
    ])
    out = multi_source_dispatch._dedup_across_sources(df, threshold=0.85)
    assert len(out) == 1


def test_dedup_across_sources_distinct_kept():
    df = pd.DataFrame([
        {"source": "arxiv", "source_id": "a", "title": "Foo Mechanism Returns",
         "authors": "x", "abstract": "y", "categories": "",
         "submitted_date": None, "updated_date": None,
         "pdf_url": None, "abs_url": None},
        {"source": "nber",  "source_id": "b", "title": "Bar Quality Premium",
         "authors": "x", "abstract": "y", "categories": "",
         "submitted_date": None, "updated_date": None,
         "pdf_url": None, "abs_url": None},
    ])
    out = multi_source_dispatch._dedup_across_sources(df, threshold=0.85)
    assert len(out) == 2


def test_normalize_paper_row_handles_arxiv_schema():
    arxiv_row = {
        "arxiv_id": "2401.x", "title": "T",
        "authors": "A", "abstract": "Ab", "categories": "q-fin.PR",
        "submitted_date": "2024-01-01", "updated_date": None,
        "pdf_url": "p", "abs_url": "a",
    }
    out = multi_source_dispatch._normalize_paper_row("arxiv", arxiv_row)
    assert out["source"] == "arxiv"
    assert out["source_id"] == "2401.x"


def test_normalize_paper_row_handles_nber_schema():
    nber_row = {
        "nber_id": "w32999", "title": "T",
        "authors": "A", "abstract": "Ab", "categories": "G12",
        "submitted_date": "2024-01-01", "updated_date": None,
        "pdf_url": "p", "abs_url": "a",
    }
    out = multi_source_dispatch._normalize_paper_row("nber", nber_row)
    assert out["source"] == "nber"
    assert out["source_id"] == "w32999"


def test_fetch_new_flow_returns_empty_when_all_fail(monkeypatch):
    """When all sources fail, dispatcher returns empty (no exception)."""
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(multi_source_dispatch, "POLITE_INTER_SOURCE_DELAY", 0)

    from engine.research.discovery import arxiv_qfin_fetcher
    from engine.research.discovery import nber_fetcher
    from engine.research.discovery import tier1_rss_fetcher
    from engine.research.discovery import openalex_fetcher
    from engine.research.discovery import crossref_fetcher
    monkeypatch.setattr(arxiv_qfin_fetcher, "fetch_qfin_with_fallback",
                          lambda *args, **kw: pd.DataFrame())
    monkeypatch.setattr(nber_fetcher, "fetch_nber_recent",
                          lambda *args, **kw: pd.DataFrame())
    monkeypatch.setattr(tier1_rss_fetcher, "fetch_tier1_rss",
                          lambda *args, **kw: pd.DataFrame())
    monkeypatch.setattr(openalex_fetcher, "fetch_cross_disciplinary",
                          lambda *args, **kw: pd.DataFrame())
    monkeypatch.setattr(crossref_fetcher, "fetch_crossref_recent",
                          lambda *args, **kw: pd.DataFrame())

    df = multi_source_dispatch.fetch_new_flow()
    assert df.empty


def test_fetch_new_flow_one_source_succeeds(monkeypatch):
    """Other sources fail; one succeeds → return that one's data."""
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(multi_source_dispatch, "POLITE_INTER_SOURCE_DELAY", 0)

    from engine.research.discovery import arxiv_qfin_fetcher
    from engine.research.discovery import nber_fetcher
    from engine.research.discovery import tier1_rss_fetcher
    monkeypatch.setattr(arxiv_qfin_fetcher, "fetch_qfin_with_fallback",
                          lambda *args, **kw: pd.DataFrame([
                              {"arxiv_id": "x", "title": "T", "authors": "a",
                                "abstract": "b", "categories": "",
                                "submitted_date": "2024-01-01",
                                "updated_date": None, "pdf_url": None,
                                "abs_url": None}
                          ]))
    monkeypatch.setattr(nber_fetcher, "fetch_nber_recent",
                          lambda *args, **kw: pd.DataFrame())
    monkeypatch.setattr(tier1_rss_fetcher, "fetch_tier1_rss",
                          lambda *args, **kw: pd.DataFrame())

    df = multi_source_dispatch.fetch_new_flow()
    assert not df.empty
    assert df.iloc[0]["source"] == "arxiv"


def test_fetch_historical_backfill_year_loop(monkeypatch):
    """Backfill iterates year-by-year backwards."""
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(multi_source_dispatch, "POLITE_INTER_SOURCE_DELAY", 0)

    arxiv_calls = []
    nber_calls = []
    from engine.research.discovery import arxiv_qfin_fetcher
    from engine.research.discovery import nber_fetcher
    def _mock_arxiv(start, end, max_results=500):
        arxiv_calls.append((start, end))
        return pd.DataFrame()
    def _mock_nber(start, end, max_results=500):
        nber_calls.append((start, end))
        return pd.DataFrame()
    monkeypatch.setattr(arxiv_qfin_fetcher, "fetch_qfin_papers", _mock_arxiv)
    monkeypatch.setattr(nber_fetcher, "fetch_nber_api", _mock_nber)

    multi_source_dispatch.fetch_historical_backfill(
        "2020-01-01", "2023-12-31",
        sources=["arxiv", "nber"],   # explicit (default is now arxiv only)
    )
    # Should call each source for each year in range
    arxiv_years = {c[0][:4] for c in arxiv_calls}
    nber_years = {c[0][:4] for c in nber_calls}
    assert arxiv_years == {"2020", "2021", "2022", "2023"}
    assert nber_years == {"2020", "2021", "2022", "2023"}
