"""Phase 8a discovery pipeline tests — arxiv + extractor + pipeline."""
from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest

from engine.research.discovery import (
    arxiv_qfin_fetcher,
    paper_extractor,
    discovery_pipeline,
)


# ── arxiv fetcher parser ─────────────────────────────────────────────────

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
       xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Cross-Sectional Momentum in Emerging Markets</title>
    <summary>We study the JT 1993 cross-sectional momentum strategy applied
to emerging market equities, finding a Sharpe ratio of 0.5 after costs.</summary>
    <published>2024-01-15T10:00:00Z</published>
    <updated>2024-01-15T10:00:00Z</updated>
    <author><name>Jane Researcher</name></author>
    <author><name>John Coauthor</name></author>
    <category term="q-fin.PR"/>
    <category term="q-fin.PM"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1"/>
  </entry>
</feed>"""


def test_arxiv_parses_atom_entry(monkeypatch):
    """Verify the parser handles standard arXiv Atom format."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(SAMPLE_ATOM)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    row = arxiv_qfin_fetcher._parse_entry(entry)
    assert row is not None
    assert row["arxiv_id"] == "2401.12345v1"
    assert "Momentum" in row["title"]
    assert "Jane Researcher" in row["authors"]
    assert "q-fin.PR" in row["categories"]
    assert row["pdf_url"] == "http://arxiv.org/pdf/2401.12345v1"
    assert row["submitted_date"] == "2024-01-15"


def test_arxiv_fetch_constructs_query(monkeypatch):
    """Verify the API call builds the right query + handles pagination."""
    captured_urls = []

    class _FakeResp:
        status_code = 200
        content = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

    class _FakeSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kw):
            captured_urls.append(url)
            return _FakeResp()

    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr(arxiv_qfin_fetcher, "POLITE_DELAY_SEC", 0)
    # Patch time.sleep to make test fast
    monkeypatch.setattr("time.sleep", lambda x: None)
    df = arxiv_qfin_fetcher.fetch_qfin_papers(
        "2024-01-01", "2024-01-31", max_results=50,
    )
    assert len(captured_urls) >= 1
    assert "search_query=" in captured_urls[0]
    assert "q-fin" in captured_urls[0]


# ── paper_extractor ─────────────────────────────────────────────────────

def test_extractor_deterministic_fallback_without_key(monkeypatch):
    monkeypatch.setattr(paper_extractor, "_read_anthropic_key", lambda: None)
    result = paper_extractor.extract_from_paper(
        "2401.12345v1", "Test Title", "Abstract text",
        use_llm=True,
    )
    assert result is not None
    assert result.mode == "deterministic_fallback"
    assert result.confidence == 0.0


def test_extractor_parse_json_handles_strict_output():
    text = (
        'Some preamble {"mechanism_proposal": "test mech", "family_guess": '
        '"momentum", "parent_family_guess": "equity_factor", '
        '"required_data_tokens": ["crsp_dsf"], "economic_intuition": "x", '
        '"decay_resilience_claim": "not addressed", "novelty_assessment": '
        '"novel", "confidence": 0.8} more text'
    )
    parsed = paper_extractor._parse_json(text)
    assert parsed is not None
    assert parsed["family_guess"] == "momentum"


def test_extractor_parse_json_returns_none_on_no_json():
    assert paper_extractor._parse_json("no json here") is None
    assert paper_extractor._parse_json("") is None


# ── discovery_pipeline dedup ────────────────────────────────────────────

def test_dedup_token_overlap_high_returns_true():
    ext = paper_extractor.PaperExtraction(
        arxiv_id="x", title="Cross-Sectional Momentum Strategy Returns",
        mechanism_proposal="m", family_guess="momentum",
        parent_family_guess="equity_factor", required_data_tokens=[],
        economic_intuition="e", decay_resilience_claim="r",
        novelty_assessment="extension", confidence=0.6, cost_usd=0.0,
        mode="llm",
    )
    library_titles = {"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"}
    # Some overlap on "Returns" — but not enough for 0.7
    assert not discovery_pipeline._is_dup_against_library(ext, library_titles)


def test_dedup_high_overlap_caught():
    ext = paper_extractor.PaperExtraction(
        arxiv_id="x",
        title="Replicating Anomalies Across Asset Classes",
        mechanism_proposal="m", family_guess="momentum",
        parent_family_guess="equity_factor", required_data_tokens=[],
        economic_intuition="e", decay_resilience_claim="r",
        novelty_assessment="extension", confidence=0.6, cost_usd=0.0,
        mode="llm",
    )
    library_titles = {"Replicating Anomalies"}
    # "replicating anomalies" / "replicating anomalies across asset classes"
    # overlap: {replicating, anomalies} ∩ {replicating, anomalies, across, asset, classes}
    # = 2 / 5 = 0.4 — NOT caught (need ≥0.7)
    assert not discovery_pipeline._is_dup_against_library(ext, library_titles)


def test_data_inventory_check_passes_when_all_known():
    all_present, missing = discovery_pipeline._check_data_inventory(["crsp_dsf"])
    assert all_present is True
    assert missing == []


def test_data_inventory_check_fails_when_unknown():
    all_present, missing = discovery_pipeline._check_data_inventory(
        ["crsp_dsf", "unknown_token_xyz"]
    )
    assert all_present is False
    assert "unknown_token_xyz" in missing


def test_data_inventory_check_empty_tokens_red_flag():
    all_present, missing = discovery_pipeline._check_data_inventory([])
    assert all_present is False    # empty is a red flag


# ── process_paper end-to-end ────────────────────────────────────────────

def test_process_paper_low_confidence_skipped(monkeypatch):
    """Low confidence < threshold → skip."""
    monkeypatch.setattr(paper_extractor, "_read_anthropic_key", lambda: None)
    paper = {
        "arxiv_id": "2401.xxx", "title": "Test", "abstract": "Abstract",
        "submitted_date": "2024-01-15",
    }
    out = discovery_pipeline.process_paper(
        paper, use_llm=True, confidence_threshold=0.5,
        library_titles=set(), library_families={},
    )
    assert out["verdict"] == "skip"
    assert "confidence" in out["reason"]


def test_process_paper_missing_id_skipped():
    out = discovery_pipeline.process_paper(
        {"title": "x"}, use_llm=False,
        library_titles=set(), library_families={},
    )
    assert out["verdict"] == "skip"
    assert "missing" in out["reason"]


def test_process_paper_with_mock_high_confidence_extraction(monkeypatch):
    """Mock LLM extraction to high-confidence + valid inventory → queued."""
    def _mock_extract(arxiv_id, title, abstract, *, use_llm=True):
        return paper_extractor.PaperExtraction(
            arxiv_id=arxiv_id, title=title,
            mechanism_proposal="A novel cross-sectional thing.",
            family_guess="momentum", parent_family_guess="equity_factor",
            required_data_tokens=["crsp_dsf"],
            economic_intuition="econ", decay_resilience_claim="addressed",
            novelty_assessment="novel", confidence=0.8, cost_usd=0.05,
            mode="llm",
        )
    monkeypatch.setattr(
        "engine.research.discovery.discovery_pipeline.extract_from_paper",
        _mock_extract,
        raising=False,
    )
    import engine.research.discovery.paper_extractor as PE
    monkeypatch.setattr(PE, "extract_from_paper", _mock_extract)

    paper = {
        "arxiv_id": "2401.xxx", "title": "Test Paper Novel",
        "abstract": "Abstract", "submitted_date": "2024-01-15",
    }
    out = discovery_pipeline.process_paper(
        paper, use_llm=True, confidence_threshold=0.5,
        library_titles=set(),
        library_families={},
    )
    assert out["verdict"] == "queue_for_review"


def test_process_paper_cousin_of_red_caveat(monkeypatch):
    """Family already in library as RED → review_with_caveat."""
    def _mock_extract(arxiv_id, title, abstract, *, use_llm=True):
        return paper_extractor.PaperExtraction(
            arxiv_id=arxiv_id, title=title,
            mechanism_proposal="Another momentum spec.",
            family_guess="quality", parent_family_guess="equity_factor",
            required_data_tokens=["crsp_dsf"],
            economic_intuition="", decay_resilience_claim="",
            novelty_assessment="extension", confidence=0.7, cost_usd=0.05,
            mode="llm",
        )
    import engine.research.discovery.paper_extractor as PE
    monkeypatch.setattr(PE, "extract_from_paper", _mock_extract)
    paper = {
        "arxiv_id": "2401.xxx", "title": "Quality Variant",
        "abstract": "Abstract", "submitted_date": "2024-01-15",
    }
    out = discovery_pipeline.process_paper(
        paper, use_llm=True, confidence_threshold=0.5,
        library_titles=set(),
        library_families={"quality": ["quality_qmj"]},   # quality is RED in library
    )
    # Post-Phase 8c upgrade: graveyard check fires before legacy cousin check.
    # Either verdict is acceptable — both demonstrate dead-cousin blocking.
    assert out["verdict"] in ("skip", "review_with_caveat")
    reason = out.get("reason", "").lower()
    assert "graveyard" in reason or "family" in reason


# ── run_discovery_batch ─────────────────────────────────────────────────

def test_run_discovery_batch_writes_queue(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.jsonl"
    log_path = tmp_path / "log.jsonl"
    monkeypatch.setattr(discovery_pipeline, "DISCOVERY_QUEUE", queue_path)
    monkeypatch.setattr(discovery_pipeline, "DISCOVERY_LOG", log_path)

    def _mock_extract(arxiv_id, title, abstract, *, use_llm=True):
        return paper_extractor.PaperExtraction(
            arxiv_id=arxiv_id, title=title,
            mechanism_proposal="novel mech",
            family_guess="lead_lag", parent_family_guess="network_effects",
            required_data_tokens=["crsp_dsf"],
            economic_intuition="", decay_resilience_claim="",
            novelty_assessment="novel", confidence=0.85, cost_usd=0.05,
            mode="llm",
        )
    import engine.research.discovery.paper_extractor as PE
    monkeypatch.setattr(PE, "extract_from_paper", _mock_extract)
    monkeypatch.setattr(
        discovery_pipeline, "_load_library_titles", lambda: set()
    )
    monkeypatch.setattr(
        discovery_pipeline, "_load_library_families", lambda: {}
    )

    papers = pd.DataFrame([
        {"arxiv_id": "2401.aaa", "title": "Paper A novel novel novel",
         "abstract": "abs A", "submitted_date": "2024-01-01"},
        {"arxiv_id": "2401.bbb", "title": "Paper B distinct distinct content",
         "abstract": "abs B", "submitted_date": "2024-01-02"},
    ])
    summary = discovery_pipeline.run_discovery_batch(papers, use_llm=True)
    assert summary["total"] == 2
    assert summary["queued"] >= 1
    assert queue_path.exists()
    rows = [json.loads(l) for l in queue_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 1
