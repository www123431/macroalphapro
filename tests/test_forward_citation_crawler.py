"""tests/test_forward_citation_crawler.py — Stage A piece 4.

Tests the seed-load + crawl orchestration. SS calls are stubbed so
tests don't hit the network. State persistence + dedup + skip-recent
all exercised.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _write_seeds(p: Path, seeds: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"schema_version": 1, "seeds": seeds}, f)


def _fake_ss_paper(paper_id, title, year=2024, venue="JF",
                    authors=("A.", "B.")):
    return SimpleNamespace(
        paper_id      = paper_id,
        title         = title,
        year          = year,
        venue         = venue,
        abstract      = "abs",
        url           = f"https://ss/{paper_id}",
        author_names  = authors,
    )


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect store CACHE_PATH to tmp file so save_new_candidates is
    safe in tests."""
    from engine.agents.papers_curator import store
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setattr(store, "CACHE_PATH", cache)
    return cache


# ────────────────────────────────────────────────────────────────────
# load_seeds
# ────────────────────────────────────────────────────────────────────
def test_load_seeds_parses_doi_arxiv_paper_id(tmp_path):
    from engine.agents.papers_curator import forward_citation_crawler as fc
    p = tmp_path / "seeds.yaml"
    _write_seeds(p, [
        {"slug": "s1", "kind": "sleeve_seed", "doi": "10.1/abc"},
        {"slug": "s2", "kind": "methodology_seed", "arxiv": "2606.x"},
        {"slug": "s3", "kind": "red_verdict_seed", "paper_id": "ssp1"},
    ])
    seeds = fc.load_seeds(path=p)
    assert len(seeds) == 3
    by_slug = {s.slug: s for s in seeds}
    assert by_slug["s1"].doi == "10.1/abc"
    assert by_slug["s2"].arxiv_id == "2606.x"
    assert by_slug["s3"].paper_id == "ssp1"


def test_load_seeds_drops_unresolvable(tmp_path):
    """A seed with no doi/arxiv/paper_id is silently dropped (defensive
    against yaml typos)."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    p = tmp_path / "seeds.yaml"
    _write_seeds(p, [
        {"slug": "good", "kind": "sleeve_seed", "doi": "10.1/x"},
        {"slug": "bad",  "kind": "sleeve_seed"},   # no id at all
    ])
    seeds = fc.load_seeds(path=p)
    assert [s.slug for s in seeds] == ["good"]


def test_load_seeds_missing_file_returns_empty(tmp_path):
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds = fc.load_seeds(path=tmp_path / "does_not_exist.yaml")
    assert seeds == ()


# ────────────────────────────────────────────────────────────────────
# Seed → SS paper_id resolution priority
# ────────────────────────────────────────────────────────────────────
def test_resolve_priority_paper_id_first(monkeypatch):
    """paper_id wins over doi (no SS call needed)."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    monkeypatch.setattr(fc, "lookup_paper_by_doi",
                          lambda d: pytest.fail("should not call"))
    seed = fc.ForwardSeed(slug="s", kind="sleeve_seed",
                            paper_id="ssp1", doi="10.1/x")
    assert fc._resolve_seed_paper_id(seed) == "ssp1"


def test_resolve_doi_calls_ss(monkeypatch):
    from engine.agents.papers_curator import forward_citation_crawler as fc
    calls = []
    def fake_doi(doi):
        calls.append(doi)
        return _fake_ss_paper("ss_doi_resolved", "x")
    monkeypatch.setattr(fc, "lookup_paper_by_doi", fake_doi)
    seed = fc.ForwardSeed(slug="s", kind="sleeve_seed", doi="10.1/y")
    assert fc._resolve_seed_paper_id(seed) == "ss_doi_resolved"
    assert calls == ["10.1/y"]


def test_resolve_arxiv_fallback_when_doi_missing(monkeypatch):
    from engine.agents.papers_curator import forward_citation_crawler as fc
    monkeypatch.setattr(fc, "lookup_paper_by_arxiv",
                          lambda a: _fake_ss_paper("ss_arxiv_x", "x"))
    seed = fc.ForwardSeed(slug="s", kind="sleeve_seed", arxiv_id="2606.x")
    assert fc._resolve_seed_paper_id(seed) == "ss_arxiv_x"


def test_resolve_returns_none_when_doi_lookup_fails(monkeypatch):
    from engine.agents.papers_curator import forward_citation_crawler as fc
    monkeypatch.setattr(fc, "lookup_paper_by_doi", lambda d: None)
    monkeypatch.setattr(fc, "search_paper_by_title", lambda q, **kw: ())
    seed = fc.ForwardSeed(slug="s", kind="sleeve_seed", doi="10.1/missing")
    assert fc._resolve_seed_paper_id(seed) is None


# ────────────────────────────────────────────────────────────────────
# Title-search fallback + strict match — piece 4 follow-up
# ────────────────────────────────────────────────────────────────────
def test_parse_slug_standard_pattern():
    from engine.agents.papers_curator import forward_citation_crawler as fc
    assert fc._parse_slug("koijen_moskowitz_pedersen_vrugt_2018_jfe") == (
        ("koijen", "moskowitz", "pedersen", "vrugt"), 2018,
    )
    assert fc._parse_slug("harvey_liu_zhu_2016_rfs") == (
        ("harvey", "liu", "zhu"), 2016,
    )
    # Malformed — year first → ((),None)
    assert fc._parse_slug("2018_something") == ((), None)
    assert fc._parse_slug("no_year_here") == ((), None)


def test_build_title_search_query():
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seed = fc.ForwardSeed(
        slug="koijen_moskowitz_pedersen_vrugt_2018_jfe",
        kind="sleeve_seed", title_hint="Carry",
    )
    q = fc._build_title_search_query(seed)
    assert "koijen" in q
    assert "vrugt" in q
    assert "2018" in q
    assert "Carry" in q


def test_strict_match_accepts_canonical(monkeypatch):
    """Title hint present + 3/3 authors present + year exact → accept."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    r = _fake_ss_paper(
        "ssp_hlz", "...and the Cross-Section of Expected Returns",
        year=2016, venue="RFS",
        authors=("Campbell R. Harvey", "Yan Liu", "Heqing Zhu"),
    )
    assert fc._strict_match(
        r,
        slug_authors=("harvey", "liu", "zhu"),
        slug_year=2016,
        title_hint="Cross-Section of Expected Returns",
    )


def test_strict_match_rejects_wrong_year(monkeypatch):
    """Title + authors match but year is +6 → reject (catches the
    Blitz 2011 vs Huij 2017 follow-up false positive)."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    r = _fake_ss_paper(
        "ss_huij_2017",
        "Residual Momentum and Reversal Strategies Revisited",
        year=2017, authors=("J. Huij",),
    )
    assert not fc._strict_match(
        r,
        slug_authors=("blitz", "huij", "martens"),
        slug_year=2011,
        title_hint="Residual momentum",
    )


def test_strict_match_rejects_missing_title_hint():
    from engine.agents.papers_curator import forward_citation_crawler as fc
    r = _fake_ss_paper(
        "ssp_other", "Some Completely Different Topic",
        year=2018, authors=("Harvey", "Liu", "Zhu"),
    )
    assert not fc._strict_match(
        r,
        slug_authors=("harvey", "liu", "zhu"),
        slug_year=2018,
        title_hint="Cross-Section of Expected Returns",
    )


def test_strict_match_rejects_one_author_coincidence():
    """Title hint matches + only 1 of 3 slug authors present → reject.
    Catches the 'someone else also published a Carry paper' case."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    r = _fake_ss_paper(
        "ssp_lonely_carry", "Carry trades in emerging markets",
        year=2018, authors=("Pedersen", "Other", "Different"),
    )
    assert not fc._strict_match(
        r,
        slug_authors=("koijen", "moskowitz", "pedersen", "vrugt"),
        slug_year=2018,
        title_hint="Carry",
    )


def test_resolve_uses_title_search_when_doi_misses(monkeypatch):
    """DOI lookup returns None → fall back to title search → strict
    match accepts → return SS paper_id."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    monkeypatch.setattr(fc, "lookup_paper_by_doi", lambda d: None)
    monkeypatch.setattr(fc, "lookup_paper_by_arxiv", lambda a: None)

    monkeypatch.setattr(fc, "search_paper_by_title", lambda q, **kw: (
        _fake_ss_paper(
            "ss_resolved_via_title",
            "...and the Cross-Section of Expected Returns",
            year=2016,
            authors=("Campbell R. Harvey", "Yan Liu", "Heqing Zhu"),
        ),
    ))

    seed = fc.ForwardSeed(
        slug="harvey_liu_zhu_2016_rfs",
        kind="methodology_seed",
        doi="10.1093/rfs/hhv059",   # DOI present but lookup will miss
        title_hint="Cross-Section of Expected Returns",
    )
    assert fc._resolve_seed_paper_id(seed) == "ss_resolved_via_title"


def test_resolve_title_search_fuzzy_match_rejected(monkeypatch):
    """SS returns a fuzzy follow-up paper; strict match rejects → None.
    No fuzzy contamination of the seed set."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    monkeypatch.setattr(fc, "lookup_paper_by_doi", lambda d: None)
    monkeypatch.setattr(fc, "search_paper_by_title", lambda q, **kw: (
        _fake_ss_paper(
            "ss_huij_2017",
            "Residual Momentum and Reversal Strategies Revisited",
            year=2017, authors=("J. Huij",),
        ),
    ))
    seed = fc.ForwardSeed(
        slug="blitz_huij_martens_2011_jempfin",
        kind="sleeve_seed",
        doi="10.1016/j.jempfin.2011.04.005",
        title_hint="Residual momentum",
    )
    assert fc._resolve_seed_paper_id(seed) is None


def test_resolve_skips_title_search_when_no_hint(monkeypatch):
    """No title_hint → don't even call search_paper_by_title."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    monkeypatch.setattr(fc, "lookup_paper_by_doi", lambda d: None)
    monkeypatch.setattr(fc, "search_paper_by_title",
        lambda q, **kw: pytest.fail("should not be called"))
    seed = fc.ForwardSeed(slug="s_2024_jf", kind="sleeve_seed",
                            doi="10.1/missing")
    assert fc._resolve_seed_paper_id(seed) is None


# ────────────────────────────────────────────────────────────────────
# Full crawl orchestration
# ────────────────────────────────────────────────────────────────────
def test_crawl_basic_flow(tmp_path, monkeypatch, tmp_cache):
    """Two seeds → SS resolves both → forward_citations returns 3
    citers each → 6 candidates persist → state file gets two slugs."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "alpha", "kind": "sleeve_seed", "doi": "10.1/a"},
        {"slug": "beta",  "kind": "methodology_seed", "doi": "10.1/b"},
    ])

    monkeypatch.setattr(fc, "lookup_paper_by_doi",
        lambda d: _fake_ss_paper(f"ss_{d[-1]}", f"seed-{d}"))

    citers_returned = []
    def fake_forward(pid, *, limit, min_year):
        # Each seed yields 3 unique citer papers
        out = tuple(_fake_ss_paper(f"{pid}_c{i}", f"cite{i}")
                     for i in range(3))
        citers_returned.append((pid, len(out)))
        return out
    monkeypatch.setattr(fc, "forward_citations", fake_forward)

    result = fc.crawl_forward_citations(
        max_per_seed      = 10,
        lookback_years    = 3,
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )

    assert result["n_seeds_total"] == 2
    assert result["n_seeds_crawled"] == 2
    assert result["n_seeds_skipped"] == 0
    assert result["n_citations_fetched"] == 6
    assert result["n_citations_new"] == 6
    assert result["unresolved_seeds"] == []
    assert result["errors"] == []
    # Both seeds got recorded in state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state.keys()) == {"alpha", "beta"}


def test_crawl_skips_recently_crawled(tmp_path, monkeypatch, tmp_cache):
    """A seed with last_crawled_ts within skip-window is bypassed."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "alpha", "kind": "sleeve_seed", "doi": "10.1/a"},
    ])
    # Pre-seed state so the seed appears 'recently crawled'
    from datetime import datetime
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    state_path.write_text(json.dumps({"alpha": now_iso}),
                           encoding="utf-8")

    monkeypatch.setattr(fc, "lookup_paper_by_doi",
        lambda d: pytest.fail("should not call when skipped"))
    monkeypatch.setattr(fc, "forward_citations",
        lambda *a, **kw: pytest.fail("should not call when skipped"))

    result = fc.crawl_forward_citations(
        max_per_seed      = 10,
        skip_recent_hours = 24,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_seeds_crawled"] == 0
    assert result["n_seeds_skipped"] == 1


def test_crawl_force_overrides_skip(tmp_path, monkeypatch, tmp_cache):
    """skip_recent_hours=0 (CLI --force) ignores last_crawled_ts."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "alpha", "kind": "sleeve_seed", "doi": "10.1/a"},
    ])
    from datetime import datetime
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    state_path.write_text(json.dumps({"alpha": now_iso}),
                           encoding="utf-8")

    monkeypatch.setattr(fc, "lookup_paper_by_doi",
        lambda d: _fake_ss_paper("p1", "x"))
    monkeypatch.setattr(fc, "forward_citations",
        lambda pid, *, limit, min_year:
            (_fake_ss_paper("c1", "cite1"),))

    result = fc.crawl_forward_citations(
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_seeds_crawled"] == 1
    assert result["n_citations_fetched"] == 1


def test_crawl_handles_unresolved_seed(tmp_path, monkeypatch, tmp_cache):
    """A seed whose DOI can't be resolved → unresolved_seeds list,
    no crash, other seeds still processed."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "good", "kind": "sleeve_seed", "doi": "10.1/good"},
        {"slug": "bad",  "kind": "sleeve_seed", "doi": "10.1/missing"},
    ])

    def fake_doi(d):
        return (_fake_ss_paper("ss_good", "x") if d == "10.1/good"
                  else None)
    monkeypatch.setattr(fc, "lookup_paper_by_doi", fake_doi)
    monkeypatch.setattr(fc, "forward_citations",
        lambda *a, **kw: (_fake_ss_paper("c1", "y"),))

    result = fc.crawl_forward_citations(
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_seeds_crawled"] == 1
    assert result["n_seeds_skipped"] == 1
    assert result["unresolved_seeds"] == ["bad"]


def test_crawl_dedup_drops_existing_paper_ids(tmp_path, monkeypatch,
                                                  tmp_cache):
    """If a citer's source_id is already in cache, it doesn't get
    re-written. n_citations_fetched counts the SS response; new counts
    after dedup."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    # Pre-seed cache with one (source, source_id) the citer will reuse
    tmp_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache.write_text(json.dumps({
        "source":       "semantic_scholar",
        "source_id":    "duplicate_pid",
        "title":        "already there",
        "authors":      [],
        "abstract":     "",
        "abs_url":      "",
        "pdf_url":      "",
        "published_ts": "",
        "categories":   [],
        "fetched_ts":   "",
    }) + "\n", encoding="utf-8")

    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "alpha", "kind": "sleeve_seed", "doi": "10.1/a"},
    ])
    monkeypatch.setattr(fc, "lookup_paper_by_doi",
        lambda d: _fake_ss_paper("ssp", "x"))
    monkeypatch.setattr(fc, "forward_citations",
        lambda *a, **kw: (
            _fake_ss_paper("duplicate_pid", "dup"),
            _fake_ss_paper("fresh_pid", "fresh"),
        ))

    result = fc.crawl_forward_citations(
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_citations_fetched"] == 2
    assert result["n_citations_new"] == 1


def test_crawl_handles_ss_error_continues(tmp_path, monkeypatch,
                                              tmp_cache):
    """SS forward_citations raising for one seed shouldn't crash the
    whole run; error gets recorded, other seeds proceed."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "alpha", "kind": "sleeve_seed", "doi": "10.1/a"},
        {"slug": "beta",  "kind": "sleeve_seed", "doi": "10.1/b"},
    ])
    monkeypatch.setattr(fc, "lookup_paper_by_doi",
        lambda d: _fake_ss_paper(f"ss_{d[-1]}", "x"))

    def fake_fwd(pid, *, limit, min_year):
        if "_a" in pid:
            raise RuntimeError("simulated SS failure")
        return (_fake_ss_paper("ok_pid", "y"),)
    monkeypatch.setattr(fc, "forward_citations", fake_fwd)

    result = fc.crawl_forward_citations(
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_seeds_crawled"] == 1     # only beta counted
    assert any("alpha" in e for e in result["errors"])
    assert result["n_citations_new"] == 1


def test_paper_candidate_tagged_with_seed_slug(monkeypatch):
    """Categories include 'forward_citation' + 'seed:<slug>' so
    downstream attribution can trace each candidate to its seed."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    ss = _fake_ss_paper("p1", "T", year=2025, venue="JFE")
    pc = fc._to_paper_candidate(ss, fetched_ts="2026-06-07T00:00:00Z",
                                  seed_slug="my_seed")
    assert "forward_citation" in pc.categories
    assert "seed:my_seed" in pc.categories
    assert "JFE" in pc.categories
    assert pc.source == "semantic_scholar"
    assert pc.source_id == "p1"


# ────────────────────────────────────────────────────────────────────
# Empty / degenerate paths
# ────────────────────────────────────────────────────────────────────
def test_crawl_empty_seeds(tmp_path, tmp_cache):
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    _write_seeds(seeds_path, [])
    result = fc.crawl_forward_citations(
        seeds_path = seeds_path,
        state_path = tmp_path / "state.json",
    )
    assert result["n_seeds_total"] == 0
    assert result["n_citations_new"] == 0


# ────────────────────────────────────────────────────────────────────
# Outage detection — added 2026-06-07 after failure-surface walk
# found silent failure on total network outage / bad SS key.
# ────────────────────────────────────────────────────────────────────
def test_crawl_emits_outage_signal_when_all_seeds_unresolved(
    tmp_path, tmp_cache, monkeypatch,
):
    """All seeds fail to resolve (SS down) → outage_suspected error
    on result so chief_of_staff sees the upstream failure."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "s1", "kind": "sleeve_seed", "doi": "10.1/a"},
        {"slug": "s2", "kind": "methodology_seed", "doi": "10.1/b"},
    ])
    monkeypatch.setattr(fc, "lookup_paper_by_doi", lambda d: None)
    monkeypatch.setattr(fc, "search_paper_by_title", lambda q, **kw: ())

    result = fc.crawl_forward_citations(
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_seeds_crawled"] == 0
    assert len(result["unresolved_seeds"]) == 2
    assert any(e.startswith("outage_suspected:")
                 for e in result["errors"])


def test_crawl_no_outage_signal_when_partial_success(
    tmp_path, tmp_cache, monkeypatch,
):
    """One seed resolves, one doesn't → no outage signal — it's a
    REAL per-seed unresolved situation, not an upstream outage."""
    from engine.agents.papers_curator import forward_citation_crawler as fc
    seeds_path = tmp_path / "seeds.yaml"
    state_path = tmp_path / "state.json"
    _write_seeds(seeds_path, [
        {"slug": "good", "kind": "sleeve_seed", "doi": "10.1/good"},
        {"slug": "bad",  "kind": "sleeve_seed", "doi": "10.1/missing"},
    ])
    monkeypatch.setattr(fc, "lookup_paper_by_doi", lambda d: (
        _fake_ss_paper("ss_good", "x") if d == "10.1/good" else None))
    monkeypatch.setattr(fc, "search_paper_by_title", lambda q, **kw: ())
    monkeypatch.setattr(fc, "forward_citations",
        lambda *a, **kw: (_fake_ss_paper("c1", "y"),))

    result = fc.crawl_forward_citations(
        skip_recent_hours = 0,
        seeds_path        = seeds_path,
        state_path        = state_path,
    )
    assert result["n_seeds_crawled"] == 1
    assert result["unresolved_seeds"] == ["bad"]
    assert not any(e.startswith("outage_suspected:")
                     for e in result["errors"])
