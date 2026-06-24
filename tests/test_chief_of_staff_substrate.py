"""tests/test_chief_of_staff_substrate.py — Stage A piece 7a.

Tests the weekly substrate orchestrator. Each of the 5 crawlers is
mocked at its module entry point so tests are offline + deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _patch_arxiv(monkeypatch, *, candidates=(), persist_returns=0,
                  fetch_raises=None):
    """Patch crawl_arxiv_qfin + save_new_candidates."""
    from engine.agents.papers_curator import crawler as c
    from engine.agents.papers_curator import store as st
    if fetch_raises:
        monkeypatch.setattr(
            c, "crawl_arxiv_qfin",
            lambda **kw: (_ for _ in ()).throw(fetch_raises),
        )
    else:
        monkeypatch.setattr(c, "crawl_arxiv_qfin",
                              lambda **kw: list(candidates))
    monkeypatch.setattr(st, "save_new_candidates",
                          lambda cs: persist_returns)


def _patch_module_fn(monkeypatch, module_path, fn_name, replacement):
    import importlib
    mod = importlib.import_module(module_path)
    monkeypatch.setattr(mod, fn_name, replacement)


# ────────────────────────────────────────────────────────────────────
# All-sources happy path
# ────────────────────────────────────────────────────────────────────
def test_all_sources_happy_path(monkeypatch, tmp_path):
    """All 5 crawlers succeed → result aggregates totals + persists
    to disk."""
    from engine.agents.chief_of_staff import substrate

    _patch_arxiv(monkeypatch, candidates=["c1", "c2"], persist_returns=2)
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.nber_rss_crawler",
        "crawl_and_persist_nber",
        lambda: {"source": "nber", "n_fetched": 35, "n_new": 20,
                  "errors": []})
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.ssrn_crossref_crawler",
        "crawl_and_persist_ssrn",
        lambda **kw: {"source": "ssrn", "n_fetched": 50, "n_new": 30,
                       "errors": []})
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.watchlist_crawler",
        "crawl_watchlist",
        lambda **kw: {"n_papers_fetched": 12, "n_papers_new": 5,
                       "errors": []})
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.forward_citation_crawler",
        "crawl_forward_citations",
        lambda **kw: {"n_citations_fetched": 80, "n_citations_new": 60,
                       "errors": []})

    result = substrate.run_weekly_substrate(persist_dir=tmp_path)
    assert not result.dry_run
    assert result.enabled_sources == substrate.ALL_SOURCES
    # 2 + 35 + 50 + 12 + 80 = 179 fetched
    assert result.total_fetched == 179
    # 2 + 20 + 30 + 5 + 60 = 117 new
    assert result.total_new == 117
    assert result.errors == []

    # Persisted to disk
    out_files = list(tmp_path.glob("*.json"))
    assert len(out_files) == 1
    saved = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert saved["total_fetched"] == 179
    assert saved["total_new"] == 117
    assert saved["enabled_sources"] == list(substrate.ALL_SOURCES)


# ────────────────────────────────────────────────────────────────────
# Source isolation — one crawler raising shouldn't kill the others
# ────────────────────────────────────────────────────────────────────
def test_one_source_raises_others_proceed(monkeypatch, tmp_path):
    from engine.agents.chief_of_staff import substrate

    # arxiv raises; others succeed
    _patch_arxiv(monkeypatch,
                  fetch_raises=RuntimeError("simulated arxiv down"))
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.nber_rss_crawler",
        "crawl_and_persist_nber",
        lambda: {"source": "nber", "n_fetched": 35, "n_new": 35,
                  "errors": []})
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.ssrn_crossref_crawler",
        "crawl_and_persist_ssrn",
        lambda **kw: {"source": "ssrn", "n_fetched": 0, "n_new": 0,
                       "errors": []})
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.watchlist_crawler",
        "crawl_watchlist",
        lambda **kw: {"n_papers_fetched": 0, "n_papers_new": 0,
                       "errors": []})
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.forward_citation_crawler",
        "crawl_forward_citations",
        lambda **kw: {"n_citations_fetched": 0, "n_citations_new": 0,
                       "errors": []})

    result = substrate.run_weekly_substrate(persist_dir=tmp_path)
    # arxiv contributed nothing; nber still did
    assert result.total_fetched == 35
    assert result.total_new == 35
    # Arxiv error propagated to top-level errors
    assert any(e.startswith("arxiv:") for e in result.errors)


def test_source_returning_errors_records_them(monkeypatch, tmp_path):
    """A crawler that returns a result dict with errors[] should have
    those propagated to the top-level errors list."""
    from engine.agents.chief_of_staff import substrate

    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.nber_rss_crawler",
        "crawl_and_persist_nber",
        lambda: {"source": "nber", "n_fetched": 0, "n_new": 0,
                  "errors": ["feed bozo"]})
    for path, fn in [
        ("engine.agents.papers_curator.ssrn_crossref_crawler",
         "crawl_and_persist_ssrn"),
        ("engine.agents.papers_curator.watchlist_crawler",
         "crawl_watchlist"),
        ("engine.agents.papers_curator.forward_citation_crawler",
         "crawl_forward_citations"),
    ]:
        _patch_module_fn(monkeypatch, path, fn,
            lambda **kw: {"n_fetched": 0, "n_new": 0,
                           "n_papers_fetched": 0, "n_papers_new": 0,
                           "n_citations_fetched": 0,
                           "n_citations_new": 0, "errors": []})

    result = substrate.run_weekly_substrate(persist_dir=tmp_path)
    assert any("nber:" in e and "feed bozo" in e
                 for e in result.errors)


# ────────────────────────────────────────────────────────────────────
# enabled_sources filter
# ────────────────────────────────────────────────────────────────────
def test_disabled_source_not_invoked(monkeypatch, tmp_path):
    """If 'nber' is omitted, _run_nber must not be called."""
    from engine.agents.chief_of_staff import substrate

    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)
    # nber crawler should NOT be called
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.nber_rss_crawler",
        "crawl_and_persist_nber",
        lambda: pytest.fail("nber should not be called"))
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.ssrn_crossref_crawler",
        "crawl_and_persist_ssrn",
        lambda **kw: pytest.fail("ssrn should not be called"))
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.watchlist_crawler",
        "crawl_watchlist",
        lambda **kw: pytest.fail("watchlist should not be called"))
    _patch_module_fn(monkeypatch,
        "engine.agents.papers_curator.forward_citation_crawler",
        "crawl_forward_citations",
        lambda **kw: pytest.fail("forward_citations should not be called"))

    result = substrate.run_weekly_substrate(
        enabled_sources = ("arxiv",),
        persist_dir     = tmp_path,
    )
    assert result.enabled_sources == ("arxiv",)
    # nber/ssrn/watchlist/forward result dicts stay empty
    assert result.nber_result == {}
    assert result.ssrn_result == {}
    assert result.watchlist_result == {}
    assert result.forward_citation_result == {}


def test_unknown_source_recorded_as_error(monkeypatch, tmp_path):
    from engine.agents.chief_of_staff import substrate

    # Patch all crawlers to no-op (they won't be called anyway since
    # the only enabled source is bogus)
    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)

    result = substrate.run_weekly_substrate(
        enabled_sources = ("bogus_source",),
        persist_dir     = tmp_path,
    )
    assert any("bogus_source" in e for e in result.errors)


# ────────────────────────────────────────────────────────────────────
# dry_run path — no crawler called, no disk write
# ────────────────────────────────────────────────────────────────────
def test_dry_run_skips_all_crawlers(monkeypatch, tmp_path):
    from engine.agents.chief_of_staff import substrate

    # If ANY crawler runs, the test fails
    for path, fn in [
        ("engine.agents.papers_curator.crawler",
         "crawl_arxiv_qfin"),
        ("engine.agents.papers_curator.nber_rss_crawler",
         "crawl_and_persist_nber"),
        ("engine.agents.papers_curator.ssrn_crossref_crawler",
         "crawl_and_persist_ssrn"),
        ("engine.agents.papers_curator.watchlist_crawler",
         "crawl_watchlist"),
        ("engine.agents.papers_curator.forward_citation_crawler",
         "crawl_forward_citations"),
    ]:
        _patch_module_fn(monkeypatch, path, fn,
            lambda *a, **kw: pytest.fail(f"dry_run called {fn}"))

    result = substrate.run_weekly_substrate(dry_run=True,
                                              persist_dir=tmp_path)
    assert result.dry_run is True
    assert result.total_fetched == 0
    assert result.total_new == 0
    # No file written
    assert list(tmp_path.glob("*.json")) == []


# ────────────────────────────────────────────────────────────────────
# _extract_counts roll-up helper
# ────────────────────────────────────────────────────────────────────
def test_extract_counts_handles_all_naming_conventions():
    from engine.agents.chief_of_staff import substrate
    assert substrate._extract_counts(
        {"n_fetched": 5, "n_new": 2}) == (5, 2)
    assert substrate._extract_counts(
        {"n_papers_fetched": 12, "n_papers_new": 4}) == (12, 4)
    assert substrate._extract_counts(
        {"n_citations_fetched": 80, "n_citations_new": 60}) == (80, 60)


def test_extract_counts_returns_zero_on_empty():
    from engine.agents.chief_of_staff import substrate
    assert substrate._extract_counts({}) == (0, 0)
    assert substrate._extract_counts({"errors": ["x"]}) == (0, 0)


def test_extract_counts_handles_none_value():
    """When a crawler reports an int field as None (defensive)."""
    from engine.agents.chief_of_staff import substrate
    assert substrate._extract_counts(
        {"n_fetched": None, "n_new": None}) == (0, 0)


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────
def test_persist_filename_is_run_date(monkeypatch, tmp_path):
    """The persisted file is named <run_date>.json."""
    from engine.agents.chief_of_staff import substrate
    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)
    for path, fn in [
        ("engine.agents.papers_curator.nber_rss_crawler",
         "crawl_and_persist_nber"),
        ("engine.agents.papers_curator.ssrn_crossref_crawler",
         "crawl_and_persist_ssrn"),
        ("engine.agents.papers_curator.watchlist_crawler",
         "crawl_watchlist"),
        ("engine.agents.papers_curator.forward_citation_crawler",
         "crawl_forward_citations"),
    ]:
        _patch_module_fn(monkeypatch, path, fn,
            lambda **kw: {"n_fetched": 0, "n_new": 0,
                           "n_papers_fetched": 0, "n_papers_new": 0,
                           "n_citations_fetched": 0,
                           "n_citations_new": 0, "errors": []})

    result = substrate.run_weekly_substrate(persist_dir=tmp_path)
    out_file = tmp_path / f"{result.run_date}.json"
    assert out_file.exists()


def test_persist_writes_history_jsonl_alongside_snapshot(
    monkeypatch, tmp_path,
):
    """Each persist writes BOTH <run_date>.json (latest snapshot,
    overwritten) AND _history.jsonl (append-only audit log) so
    same-day re-runs don't erase prior runs' audit data."""
    from engine.agents.chief_of_staff import substrate
    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)
    for path, fn in [
        ("engine.agents.papers_curator.nber_rss_crawler",
         "crawl_and_persist_nber"),
        ("engine.agents.papers_curator.ssrn_crossref_crawler",
         "crawl_and_persist_ssrn"),
        ("engine.agents.papers_curator.watchlist_crawler",
         "crawl_watchlist"),
        ("engine.agents.papers_curator.forward_citation_crawler",
         "crawl_forward_citations"),
    ]:
        _patch_module_fn(monkeypatch, path, fn,
            lambda **kw: {"n_fetched": 0, "n_new": 0,
                           "n_papers_fetched": 0, "n_papers_new": 0,
                           "n_citations_fetched": 0,
                           "n_citations_new": 0, "errors": []})

    # Run TWICE on the same day → snapshot overwritten,
    # history has 2 lines
    substrate.run_weekly_substrate(persist_dir=tmp_path)
    substrate.run_weekly_substrate(persist_dir=tmp_path)

    snaps = list(tmp_path.glob("*.json"))
    assert len(snaps) == 1, "snapshot file should be exactly 1 (same day)"

    history = tmp_path / "_history.jsonl"
    assert history.is_file(), "_history.jsonl must exist"
    lines = [l for l in
              history.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    assert len(lines) == 2, "_history must accumulate one line per run"
    # Each line must parse as valid JSON
    for l in lines:
        assert json.loads(l)["run_date"]


def test_persist_history_append_failure_is_non_fatal(
    monkeypatch, tmp_path,
):
    """If _history.jsonl can't be written (locked / read-only), the
    snapshot still goes through and the run reports no error."""
    from engine.agents.chief_of_staff import substrate
    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)
    for path, fn in [
        ("engine.agents.papers_curator.nber_rss_crawler",
         "crawl_and_persist_nber"),
        ("engine.agents.papers_curator.ssrn_crossref_crawler",
         "crawl_and_persist_ssrn"),
        ("engine.agents.papers_curator.watchlist_crawler",
         "crawl_watchlist"),
        ("engine.agents.papers_curator.forward_citation_crawler",
         "crawl_forward_citations"),
    ]:
        _patch_module_fn(monkeypatch, path, fn,
            lambda **kw: {"n_fetched": 0, "n_new": 0,
                           "n_papers_fetched": 0, "n_papers_new": 0,
                           "n_citations_fetched": 0,
                           "n_citations_new": 0, "errors": []})

    # Pre-create history as a directory so .open('a') fails
    (tmp_path / "_history.jsonl").mkdir()

    result = substrate.run_weekly_substrate(persist_dir=tmp_path)
    # Snapshot file still written
    snaps = [p for p in tmp_path.glob("*.json")
              if p.name != "_history.jsonl"]
    assert len(snaps) == 1
    # History write failed silently — no hard error on the result
    assert result.errors == []


def test_persist_failure_does_not_crash(monkeypatch, tmp_path):
    """If disk write fails (e.g. read-only fs), the in-memory result
    is still returned + persist error is appended."""
    from engine.agents.chief_of_staff import substrate
    _patch_arxiv(monkeypatch, candidates=[], persist_returns=0)
    for path, fn in [
        ("engine.agents.papers_curator.nber_rss_crawler",
         "crawl_and_persist_nber"),
        ("engine.agents.papers_curator.ssrn_crossref_crawler",
         "crawl_and_persist_ssrn"),
        ("engine.agents.papers_curator.watchlist_crawler",
         "crawl_watchlist"),
        ("engine.agents.papers_curator.forward_citation_crawler",
         "crawl_forward_citations"),
    ]:
        _patch_module_fn(monkeypatch, path, fn,
            lambda **kw: {"n_fetched": 0, "n_new": 0,
                           "n_papers_fetched": 0, "n_papers_new": 0,
                           "n_citations_fetched": 0,
                           "n_citations_new": 0, "errors": []})

    # Force _persist to raise
    monkeypatch.setattr(substrate, "_persist",
        lambda r, **kw: (_ for _ in ()).throw(OSError("disk full")))

    result = substrate.run_weekly_substrate(persist_dir=tmp_path)
    assert any("persist:" in e for e in result.errors)
