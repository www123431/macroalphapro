"""tests/test_papers_curator_parser.py — arxiv Atom parser fixture test.

The crawler's network path can't be smoke-tested deterministically
(arxiv rate-limits + DNS flakiness from CN). This test proves the
PARSER end-to-end with a fixture Atom XML — if a future arxiv API
shape change breaks parsing, this test catches it before it ships.
"""
from __future__ import annotations

import xml.etree.ElementTree as _ET


_FIXTURE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
       xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>q-fin recent submissions</title>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-01-15T18:00:00Z</updated>
    <published>2024-01-10T09:00:00Z</published>
    <title>
      A Multi-Factor   Model for Cryptocurrency Returns
    </title>
    <summary>
      We propose a five-factor model for the cross-section of
      cryptocurrency returns. The factors are MKT, SMB, MOM, ILLIQ,
      and NETWORK. We document significant alpha decay post-2021.
    </summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Quant</name></author>
    <category term="q-fin.PR" />
    <category term="q-fin.ST" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.99999</id>
    <published>2024-02-20T12:00:00Z</published>
    <title>Term Structure of Volatility Risk Premia</title>
    <summary>We construct VRP across maturities.</summary>
    <author><name>Carol Economist</name></author>
    <category term="q-fin.RM" />
  </entry>
</feed>
"""


def test_parse_arxiv_entry_extracts_canonical_fields():
    from engine.agents.papers_curator.crawler import _parse_arxiv_entry, _ARXIV_NS
    root = _ET.fromstring(_FIXTURE_ATOM)
    entries = root.findall("atom:entry", _ARXIV_NS)
    assert len(entries) == 2

    c0 = _parse_arxiv_entry(entries[0], "2024-01-16T00:00:00Z")
    assert c0 is not None
    assert c0.source == "arxiv"
    # version suffix must be stripped
    assert c0.source_id == "2401.12345"
    # multi-line + multi-whitespace title normalized to single spaces
    assert c0.title == "A Multi-Factor Model for Cryptocurrency Returns"
    # multi-paragraph abstract collapsed
    assert "five-factor model" in c0.abstract
    assert "  " not in c0.abstract     # no double-spaces left
    assert c0.authors == ("Alice Researcher", "Bob Quant")
    assert c0.categories == ("q-fin.PR", "q-fin.ST")
    assert c0.abs_url == "http://arxiv.org/abs/2401.12345v2"
    # pdf url convention
    assert c0.pdf_url == "http://arxiv.org/pdf/2401.12345v2.pdf"
    assert c0.published_ts == "2024-01-10T09:00:00Z"
    assert c0.fetched_ts == "2024-01-16T00:00:00Z"


def test_parse_arxiv_entry_handles_minimal_record():
    """Second fixture entry has only one author + one category + no
    updated tag. Parser must still produce a valid candidate."""
    from engine.agents.papers_curator.crawler import _parse_arxiv_entry, _ARXIV_NS
    root = _ET.fromstring(_FIXTURE_ATOM)
    entries = root.findall("atom:entry", _ARXIV_NS)

    c1 = _parse_arxiv_entry(entries[1], "2024-02-21T00:00:00Z")
    assert c1 is not None
    assert c1.source_id == "2402.99999"
    assert c1.authors == ("Carol Economist",)
    assert c1.categories == ("q-fin.RM",)
    assert c1.title == "Term Structure of Volatility Risk Premia"


def test_parse_arxiv_entry_returns_none_on_missing_id():
    """Malformed entries must drop, not crash the whole crawl."""
    from engine.agents.papers_curator.crawler import _parse_arxiv_entry, _ARXIV_NS
    bad = _ET.fromstring("""
      <entry xmlns="http://www.w3.org/2005/Atom">
        <title>orphan with no id</title>
      </entry>
    """)
    # Must not raise; returns None
    result = _parse_arxiv_entry(bad, "2024-01-01T00:00:00Z")
    # source_id derived from id - "" -> ""  → falsy check returns None
    assert result is None


def test_save_new_candidates_dedups_against_cache(tmp_path, monkeypatch):
    """Re-saving the same (source, source_id) is a no-op."""
    import engine.agents.papers_curator.store as store_mod
    from engine.agents.papers_curator.crawler import PaperCandidate

    monkeypatch.setattr(store_mod, "CACHE_PATH", tmp_path / "cache.jsonl")

    c = PaperCandidate(
        source       = "arxiv",
        source_id    = "2401.12345",
        title        = "test",
        authors      = ("A",),
        abstract     = "x",
        abs_url      = "http://arxiv.org/abs/2401.12345",
        pdf_url      = "http://arxiv.org/pdf/2401.12345.pdf",
        published_ts = "2024-01-10T00:00:00Z",
        categories   = ("q-fin.PR",),
        fetched_ts   = "2024-01-11T00:00:00Z",
    )
    n1 = store_mod.save_new_candidates([c])
    assert n1 == 1
    # Re-save the same — should be 0 new
    n2 = store_mod.save_new_candidates([c])
    assert n2 == 0
    # Same source_id, different source — should be new (different key)
    c2 = PaperCandidate(**{**c.__dict__, "source": "nber"})
    n3 = store_mod.save_new_candidates([c2])
    assert n3 == 1
