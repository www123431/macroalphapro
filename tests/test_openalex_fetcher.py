"""Tests for engine.research.discovery.openalex_fetcher (senior roadmap #3)."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from engine.research.discovery import openalex_fetcher as oa


# ── Venue loader ──────────────────────────────────────────────────────────

def test_load_venues_returns_dict():
    """Sanity: YAML loads as venue map."""
    oa._VENUES_CACHE = None
    venues = oa.load_venues()
    assert isinstance(venues, dict)
    assert len(venues) > 0
    # Curated venues exist
    assert "S160506855" in venues   # TAR
    assert "S4210226497" in venues  # Critical Finance Review


def test_critical_finance_review_routes_to_graveyard():
    """CFR must carry graveyard_routing flag."""
    oa._VENUES_CACHE = None
    venues = oa.load_venues()
    cfr = venues.get("S4210226497")
    assert cfr is not None
    assert cfr.get("graveyard_routing") == "auto_negative_evidence"


# ── Inverted-index abstract decoder ───────────────────────────────────────

def test_abstract_from_inverted_index_reconstructs():
    """Encoded as position → word. Decode preserves order."""
    inv = {"the": [0, 4], "quick": [1], "brown": [2], "fox": [3, 5]}
    result = oa._abstract_from_inverted_index(inv)
    assert result == "the quick brown fox the fox"


def test_abstract_from_inverted_index_empty():
    assert oa._abstract_from_inverted_index(None) == ""
    assert oa._abstract_from_inverted_index({}) == ""


# ── Record normalization ──────────────────────────────────────────────────

@pytest.fixture
def sample_openalex_item():
    return {
        "id": "https://openalex.org/W123456",
        "title": "Earnings Quality and Future Returns",
        "abstract_inverted_index": {"earnings": [0], "quality": [1], "predicts": [2]},
        "authorships": [
            {"author": {"display_name": "Alice Smith"},
             "institutions": [{"display_name": "Chicago Booth"}]},
            {"author": {"display_name": "Bob Jones"},
             "institutions": [{"display_name": "Wharton"}]},
        ],
        "primary_location": {
            "source": {"display_name": "The Accounting Review"},
        },
        "publication_date": "2024-03-15",
        "doi": "https://doi.org/10.1111/abc.123",
        "cited_by_count": 42,
    }


def test_normalize_extracts_all_fields(sample_openalex_item):
    venue_cfg = {"category": "accounting", "credibility_tier": 0.80}
    row = oa._normalize_openalex_record(sample_openalex_item, venue_cfg)
    assert row["source"] == "openalex"
    assert row["source_id"] == "W123456"
    assert row["title"] == "Earnings Quality and Future Returns"
    assert row["abstract"] == "earnings quality predicts"
    assert "Alice Smith" in row["authors"]
    assert "Bob Jones" in row["authors"]
    assert row["venue"] == "The Accounting Review"
    assert row["venue_category"] == "accounting"
    assert row["credibility_tier_hint"] == 0.80
    assert row["submitted_date"] == "2024-03-15"
    assert row["doi"] == "10.1111/abc.123"
    assert row["citation_count"] == 42


def test_normalize_strips_doi_url_prefix(sample_openalex_item):
    sample_openalex_item["doi"] = "https://doi.org/10.1111/test.456"
    row = oa._normalize_openalex_record(sample_openalex_item, {})
    assert row["doi"] == "10.1111/test.456"


def test_normalize_handles_cfr_routing():
    """Critical Finance Review venue config carries graveyard_routing."""
    cfg = {
        "category": "replication",
        "credibility_tier": 0.85,
        "graveyard_routing": "auto_negative_evidence",
    }
    item = {
        "id": "https://openalex.org/W999",
        "title": "Replication: The Volatility Effect Failed",
        "abstract_inverted_index": {"failed": [0]},
        "primary_location": {"source": {"display_name": "Critical Finance Review"}},
        "publication_date": "2024-01-01",
        "doi": None,
        "cited_by_count": 5,
    }
    row = oa._normalize_openalex_record(item, cfg)
    assert row["graveyard_routing"] == "auto_negative_evidence"


def test_normalize_returns_none_for_missing_title():
    item = {
        "id": "https://openalex.org/W123",
        "title": "",      # missing
        "primary_location": {"source": {"display_name": "X"}},
        "publication_date": "2024-01-01",
    }
    assert oa._normalize_openalex_record(item, {}) is None


def test_normalize_returns_none_for_missing_id():
    item = {
        "id": "",
        "title": "A title",
        "publication_date": "2024-01-01",
    }
    assert oa._normalize_openalex_record(item, {}) is None


# ── Single-venue fetch (mocked) ───────────────────────────────────────────

def test_fetch_openalex_venue_mocked_returns_papers(sample_openalex_item):
    """Mock requests.get → confirm fetch returns DataFrame with expected
    shape + content."""
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [sample_openalex_item],
        "meta": {"next_cursor": None},
    }
    with mock.patch("requests.Session.get", return_value=mock_response):
        with mock.patch("time.sleep"):
            df = oa.fetch_openalex_venue(
                "S160506855", "2024-01-01", "2024-12-31",
                max_results=10, skip_health_check=True,
            )
    assert len(df) == 1
    assert df.iloc[0]["source"] == "openalex"


def test_fetch_openalex_venue_handles_429(monkeypatch):
    """Rate limit → graceful empty return + health update."""
    mock_response = mock.MagicMock()
    mock_response.status_code = 429
    monkeypatch.setattr("requests.Session.get", lambda *a, **kw: mock_response)
    monkeypatch.setattr("time.sleep", lambda x: None)
    from engine.data import source_health
    with mock.patch.object(source_health, "mark_failure") as marker:
        df = oa.fetch_openalex_venue(
            "S160506855", "2024-01-01", "2024-12-31",
            skip_health_check=True,
        )
    assert df.empty
    marker.assert_called()
    args = marker.call_args[0]
    assert args[1] == "rate_limited"


def test_fetch_openalex_venue_handles_network_error(monkeypatch):
    def _raise(*a, **kw):
        raise ConnectionError("simulated")
    monkeypatch.setattr("requests.Session.get", _raise)
    monkeypatch.setattr("time.sleep", lambda x: None)
    from engine.data import source_health
    with mock.patch.object(source_health, "mark_failure") as marker:
        df = oa.fetch_openalex_venue(
            "S160506855", "2024-01-01", "2024-12-31",
            skip_health_check=True,
        )
    assert df.empty


# ── Multi-venue fetch (mocked) ────────────────────────────────────────────

def test_fetch_cross_disciplinary_iterates_all_venues(monkeypatch):
    """Mock fetch_openalex_venue → confirm fetch_cross_disciplinary
    calls it once per curated venue."""
    call_log = []
    def _mock_fetch(source_id, *a, **kw):
        call_log.append(source_id)
        return pd.DataFrame()
    monkeypatch.setattr(oa, "fetch_openalex_venue", _mock_fetch)
    oa._VENUES_CACHE = None
    oa.fetch_cross_disciplinary("2024-01-01", "2024-12-31",
                                    max_results_per_venue=10,
                                    skip_health_check=True)
    assert len(call_log) == len(oa.load_venues())


def test_fetch_cross_disciplinary_respects_category_filter(monkeypatch):
    """category_filter=('replication',) → only CFR queried."""
    call_log = []
    def _mock_fetch(source_id, *a, **kw):
        call_log.append(source_id)
        return pd.DataFrame()
    monkeypatch.setattr(oa, "fetch_openalex_venue", _mock_fetch)
    oa._VENUES_CACHE = None
    oa.fetch_cross_disciplinary("2024-01-01", "2024-12-31",
                                    category_filter=("replication",))
    # Only Critical Finance Review (replication category)
    assert call_log == ["S4210226497"]


def test_fetch_cross_disciplinary_skips_failed_venue(monkeypatch):
    """One venue raising should NOT abort the rest."""
    def _mock_fetch(source_id, *a, **kw):
        if source_id == "S160506855":
            raise RuntimeError("simulated venue failure")
        return pd.DataFrame()
    monkeypatch.setattr(oa, "fetch_openalex_venue", _mock_fetch)
    oa._VENUES_CACHE = None
    # Should not raise
    df = oa.fetch_cross_disciplinary("2024-01-01", "2024-12-31",
                                          skip_health_check=True)
    assert isinstance(df, pd.DataFrame)


# ── Discovery pipeline auto-routing ───────────────────────────────────────

def test_discovery_pipeline_auto_routes_cfr_paper():
    """A paper with graveyard_routing='auto_negative_evidence' must
    short-circuit BEFORE LLM extraction + before credibility scorer.

    This is the Critical Finance Review automatic-graveyard-feed
    promised by the venue YAML routing flag."""
    from engine.research.discovery import discovery_pipeline
    paper = {
        "arxiv_id":         "W999",
        "source_id":        "W999",
        "title":            "Replication: The Volatility Effect Failed",
        "abstract":         "We show the volatility anomaly does not survive.",
        "venue":            "Critical Finance Review",
        "graveyard_routing": "auto_negative_evidence",
    }
    out = discovery_pipeline.process_paper(paper, use_llm=False)
    assert out["stage"] == "graveyard_auto_route"
    assert out["verdict"] == "route_to_negative_evidence"
    assert "Critical Finance Review" in out["reason"]
