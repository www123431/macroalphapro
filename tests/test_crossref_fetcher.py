"""Tests for engine.research.discovery.crossref_fetcher (senior #3 fallback)."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from engine.research.discovery import crossref_fetcher as cr


@pytest.fixture(autouse=True)
def _clear_venues_cache():
    cr._VENUES_CACHE = None
    yield
    cr._VENUES_CACHE = None


# ── Venue loading ─────────────────────────────────────────────────────────

def test_load_venues_only_returns_those_with_issn():
    """Venues must have ISSN to be Crossref-fetchable."""
    venues = cr.load_venues()
    assert len(venues) > 0
    for sid, cfg in venues.items():
        assert "issn" in cfg, f"venue {sid} missing ISSN"


def test_get_venues_with_issn_flattens_to_list():
    venues = cr.get_venues_with_issn()
    assert isinstance(venues, list)
    assert len(venues) > 0
    for v in venues:
        for k in ("issn", "name", "short", "category"):
            assert k in v


def test_critical_finance_review_preserves_routing():
    """CFR's graveyard_routing flag must propagate through to crossref
    side too, so a fresh CFR replication paper still auto-routes."""
    venues = cr.get_venues_with_issn()
    cfr = next((v for v in venues if v["short"] == "CFR"), None)
    assert cfr is not None
    assert cfr["graveyard_routing"] == "auto_negative_evidence"


# ── Record normalization ──────────────────────────────────────────────────

@pytest.fixture
def sample_crossref_item():
    return {
        "DOI": "10.1287/mnsc.2024.1234",
        "title": ["Supply Chain Disruptions and Equity Returns"],
        "author": [
            {"given": "Alice", "family": "Smith",
              "affiliation": [{"name": "MIT Sloan"}]},
            {"given": "Bob", "family": "Jones",
              "affiliation": [{"name": "Wharton"}]},
        ],
        "issued": {"date-parts": [[2024, 6, 15]]},
        "container-title": ["Management Science"],
        "abstract": "<jats:p>We study how supply chain disruptions predict returns.</jats:p>",
    }


def test_normalize_extracts_all_fields(sample_crossref_item):
    venue_cfg = {
        "name": "Management Science",
        "short": "MS",
        "category": "operations",
        "credibility_tier": 0.70,
    }
    row = cr._normalize_crossref_record(sample_crossref_item, venue_cfg)
    assert row["source"] == "crossref"
    assert row["source_id"] == "10.1287/mnsc.2024.1234"
    assert row["title"] == "Supply Chain Disruptions and Equity Returns"
    assert "Smith, Alice" in row["authors"]
    assert "Jones, Bob" in row["authors"]
    assert row["venue"] == "Management Science"
    assert row["venue_category"] == "operations"
    assert row["credibility_tier_hint"] == 0.70
    assert row["submitted_date"] == "2024-06-15"
    assert row["doi"] == "10.1287/mnsc.2024.1234"
    assert row["abs_url"] == "https://doi.org/10.1287/mnsc.2024.1234"


def test_normalize_strips_jats_xml_from_abstract(sample_crossref_item):
    row = cr._normalize_crossref_record(sample_crossref_item, {})
    # JATS tags <jats:p> should be stripped
    assert "<jats:p>" not in row["abstract"]
    assert "supply chain" in row["abstract"].lower()


def test_normalize_handles_partial_date(sample_crossref_item):
    """Crossref sometimes returns only [YYYY] or [YYYY, M]."""
    sample_crossref_item["issued"]["date-parts"] = [[2024]]
    row = cr._normalize_crossref_record(sample_crossref_item, {})
    assert row["submitted_date"] == "2024-01-01"


def test_normalize_handles_month_only(sample_crossref_item):
    sample_crossref_item["issued"]["date-parts"] = [[2024, 6]]
    row = cr._normalize_crossref_record(sample_crossref_item, {})
    assert row["submitted_date"] == "2024-06-01"


def test_normalize_prefers_published_online_over_issued(sample_crossref_item):
    sample_crossref_item["published-online"] = {"date-parts": [[2024, 1, 10]]}
    # issued says June 15, published-online says Jan 10 → use earliest (Jan 10)
    row = cr._normalize_crossref_record(sample_crossref_item, {})
    assert row["submitted_date"] == "2024-01-10"


def test_normalize_returns_none_for_missing_doi():
    item = {
        "title": ["A title"],
        "issued": {"date-parts": [[2024, 1, 1]]},
    }
    assert cr._normalize_crossref_record(item, {}) is None


def test_normalize_returns_none_for_missing_title():
    item = {
        "DOI": "10.1234/x",
        "title": [],
        "issued": {"date-parts": [[2024, 1, 1]]},
    }
    assert cr._normalize_crossref_record(item, {}) is None


def test_normalize_cfr_preserves_graveyard_routing(sample_crossref_item):
    """CFR papers via Crossref must still carry routing flag."""
    cfg = {"category": "replication", "credibility_tier": 0.85,
             "graveyard_routing": "auto_negative_evidence"}
    row = cr._normalize_crossref_record(sample_crossref_item, cfg)
    assert row["graveyard_routing"] == "auto_negative_evidence"


# ── Single-venue fetch (mocked HTTP) ──────────────────────────────────────

def test_fetch_crossref_venue_happy_path(sample_crossref_item):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"items": [sample_crossref_item], "next-cursor": None}
    }
    with mock.patch("requests.Session.get", return_value=mock_response):
        with mock.patch("time.sleep"):
            df = cr.fetch_crossref_venue(
                "0025-1909", "2024-06-01", "2024-06-30",
                max_results=10, skip_health_check=True,
            )
    assert len(df) == 1
    assert df.iloc[0]["doi"] == "10.1287/mnsc.2024.1234"


def test_fetch_crossref_venue_handles_429(monkeypatch):
    mock_response = mock.MagicMock()
    mock_response.status_code = 429
    monkeypatch.setattr("requests.Session.get", lambda *a, **kw: mock_response)
    monkeypatch.setattr("time.sleep", lambda x: None)
    from engine.data import source_health
    with mock.patch.object(source_health, "mark_failure") as marker:
        df = cr.fetch_crossref_venue(
            "0025-1909", "2024-06-01", "2024-06-30",
            skip_health_check=True,
        )
    assert df.empty
    marker.assert_called()
    assert marker.call_args[0][1] == "rate_limited"


# ── Multi-venue fetch (mocked) ────────────────────────────────────────────

def test_fetch_crossref_recent_iterates_all_venues_with_issn(monkeypatch):
    call_log = []
    def _mock(issn, *a, **kw):
        call_log.append(issn)
        return pd.DataFrame()
    monkeypatch.setattr(cr, "fetch_crossref_venue", _mock)
    cr.fetch_crossref_recent(days_back=14, skip_health_check=True)
    venues = cr.get_venues_with_issn()
    expected_issns = {v["issn"] for v in venues}
    assert set(call_log) == expected_issns


def test_fetch_crossref_recent_respects_category_filter(monkeypatch):
    call_log = []
    def _mock(issn, *a, **kw):
        call_log.append(issn)
        return pd.DataFrame()
    monkeypatch.setattr(cr, "fetch_crossref_venue", _mock)
    cr.fetch_crossref_recent(days_back=14,
                                  category_filter=("replication",),
                                  skip_health_check=True)
    # Only CFR matches replication category
    assert len(call_log) == 1
    assert call_log[0] == "2164-5744"     # CFR ISSN


def test_fetch_crossref_recent_skips_failed_venue(monkeypatch):
    def _mock(issn, *a, **kw):
        if issn == "0025-1909":
            raise RuntimeError("simulated")
        return pd.DataFrame()
    monkeypatch.setattr(cr, "fetch_crossref_venue", _mock)
    # Should not raise
    df = cr.fetch_crossref_recent(days_back=14, skip_health_check=True)
    assert isinstance(df, pd.DataFrame)
