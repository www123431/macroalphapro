"""Tests for engine.research.discovery.credibility_scorer — senior #1."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest

from engine.research.discovery import credibility_scorer as cs
from engine.research.discovery.credibility_scorer import (
    CredibilityScore, DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, PaperMetadata,
    score_paper, explain_paper,
)


@pytest.fixture(autouse=True)
def _reset_venue_cache():
    """Force re-load of venue map so monkeypatching VENUE_TIER_MAP_PATH works."""
    cs._VENUE_MAP = None
    yield
    cs._VENUE_MAP = None


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect ledger paths to tmp + bypass graveyard for unit isolation."""
    monkeypatch.setattr(cs, "AUTHOR_TRACK_PATH",
                          tmp_path / "author_track.jsonl")
    # Patch graveyard to a controlled stub for novelty tests
    def _stub_check(candidate, **kw):
        from engine.research.graveyard import GraveyardMatch
        return GraveyardMatch(
            matched=False, signals_matched=[], matched_entries=[],
            overall_confidence=0.0, recommendation="allow",
            explanation="no match", cousin_count_in_family=0, elevated=False,
        )
    monkeypatch.setattr(
        "engine.research.graveyard.check_against_graveyard", _stub_check,
    )
    return tmp_path


# ── Venue tier ────────────────────────────────────────────────────────────

def test_jf_gets_top_tier(patched_paths):
    score = score_paper(PaperMetadata(title="X", venue="JF"))
    assert score.features["venue_tier"] >= 0.95


def test_unknown_venue_gets_default(patched_paths):
    score = score_paper(PaperMetadata(title="X", venue="ObscureJournal"))
    assert score.features["venue_tier"] == pytest.approx(0.4, abs=0.05)


def test_substring_match_for_venue(patched_paths):
    """venue='JF 2024 vol 80' should still match 'jf'."""
    score = score_paper(PaperMetadata(title="X", venue="JF 2024 vol 80"))
    assert score.features["venue_tier"] >= 0.9


def test_arxiv_tier_is_low(patched_paths):
    score = score_paper(PaperMetadata(title="X", venue="arxiv"))
    assert score.features["venue_tier"] <= 0.55


def test_critical_finance_review_is_tier1(patched_paths):
    """Replication venue should rank with JFE — critical signal for graveyard."""
    score = score_paper(PaperMetadata(title="X", venue="Critical Finance Review"))
    assert score.features["venue_tier"] >= 0.8


# ── First-author track ───────────────────────────────────────────────────

def test_no_author_returns_neutral(patched_paths):
    score = score_paper(PaperMetadata(title="X"))
    assert score.features["first_author_track"] == pytest.approx(0.4, abs=0.05)


def test_top_dept_author_gets_higher_prior(patched_paths):
    no_aff = score_paper(PaperMetadata(
        title="X", authors="Smith, John", affiliations=""))
    top_dept = score_paper(PaperMetadata(
        title="X", authors="Smith, John", affiliations="Chicago Booth"))
    assert top_dept.features["first_author_track"] > no_aff.features["first_author_track"]


def test_author_track_updates_posterior(patched_paths):
    cs.update_author_track("smith, john", "pass")
    cs.update_author_track("smith, john", "pass")
    cs.update_author_track("smith, john", "pass")
    score = score_paper(PaperMetadata(title="X", authors="Smith, John"))
    # 3 passes raises posterior above neutral
    assert score.features["first_author_track"] > 0.3


def test_author_track_fails_lower_posterior(patched_paths):
    for _ in range(5):
        cs.update_author_track("jones, jane", "fail")
    score = score_paper(PaperMetadata(title="X", authors="Jones, Jane",
                                          affiliations="Chicago Booth"))
    # 5 fails + top-dept prior (3,7) → α=3, β=12 → mean 0.20
    assert score.features["first_author_track"] < 0.3


# ── Sample window ────────────────────────────────────────────────────────

def test_pre_1990_sample_scores_high(patched_paths):
    score = score_paper(PaperMetadata(
        title="X", abstract="We study returns from 1965 to 2020 across CRSP."))
    assert score.features["sample_window"] >= 0.8


def test_post_2010_sample_scores_low(patched_paths):
    score = score_paper(PaperMetadata(
        title="X", abstract="Sample period 2015-2023 of major ETFs."))
    assert score.features["sample_window"] <= 0.3


def test_missing_sample_neutral(patched_paths):
    score = score_paper(PaperMetadata(
        title="X", abstract="No date mentioned at all."))
    assert score.features["sample_window"] == pytest.approx(0.5, abs=0.05)


def test_long_panel_scores_well(patched_paths):
    score = score_paper(PaperMetadata(
        title="X", abstract="Spanning 1985–2024 in CRSP."))
    assert score.features["sample_window"] >= 0.6


# ── Mechanism novelty (via graveyard) ────────────────────────────────────

def test_novel_paper_scores_high(patched_paths):
    """Graveyard stub returns allow → novelty high."""
    score = score_paper(PaperMetadata(
        title="Brand new mechanism via cosmic rays"))
    assert score.features["mechanism_novelty"] >= 0.7


def test_graveyard_block_crashes_novelty(monkeypatch, patched_paths):
    """Override stub to return block → novelty floor."""
    from engine.research.graveyard import GraveyardMatch
    monkeypatch.setattr(
        "engine.research.graveyard.check_against_graveyard",
        lambda c, **kw: GraveyardMatch(
            matched=True, signals_matched=["family_match"], matched_entries=[],
            overall_confidence=0.9, recommendation="block",
            explanation="family_match on bond_xsmom",
            cousin_count_in_family=2, elevated=False,
        ),
    )
    score = score_paper(PaperMetadata(title="Yet another momentum study"))
    assert score.features["mechanism_novelty"] <= 0.2


# ── Cite count ───────────────────────────────────────────────────────────

def test_no_date_neutral_cite(patched_paths):
    score = score_paper(PaperMetadata(title="X", submitted_date=None))
    assert score.features["cite_count_age_adj"] == pytest.approx(0.4, abs=0.05)


def test_recent_paper_neutral_cite(patched_paths):
    recent = (date.today() - timedelta(days=180)).isoformat()
    score = score_paper(PaperMetadata(
        title="X", submitted_date=recent, doi="10.1234/x"))
    # <2yr → neutral regardless of fetch_remote
    assert score.features["cite_count_age_adj"] == pytest.approx(0.4, abs=0.05)


def test_old_paper_with_no_remote_neutral(patched_paths):
    old = (date.today() - timedelta(days=4 * 365)).isoformat()
    score = score_paper(PaperMetadata(
        title="X", submitted_date=old, doi="10.1234/x"),
        fetch_cite_count=False)
    assert score.features["cite_count_age_adj"] == pytest.approx(0.4, abs=0.05)


def test_old_paper_high_cite_scores_well(monkeypatch, patched_paths):
    """When fetch_cite_count=True and crossref returns 500 cites in 5yr → high."""
    monkeypatch.setattr(cs, "_crossref_cite_count", lambda doi: 500)
    old = (date.today() - timedelta(days=5 * 365)).isoformat()
    score = score_paper(PaperMetadata(
        title="X", submitted_date=old, doi="10.1234/x"),
        fetch_cite_count=True)
    assert score.features["cite_count_age_adj"] >= 0.6


# ── Total score + threshold ───────────────────────────────────────────────

def test_total_score_within_unit_interval(patched_paths):
    score = score_paper(PaperMetadata(
        title="X", venue="JF", authors="Smith, A",
        affiliations="Chicago Booth",
        abstract="1980-2024 panel", submitted_date="2018-01-01",
    ))
    assert 0.0 <= score.score <= 1.0


def test_high_quality_paper_passes_filter(patched_paths):
    score = score_paper(PaperMetadata(
        title="A truly novel structural arbitrage signal",
        venue="JF", authors="Smith, A",
        affiliations="Chicago Booth",
        abstract="Panel from 1965 to 2020 across CRSP."))
    assert score.passes_filter is True
    assert score.score >= DEFAULT_THRESHOLD


def test_low_quality_paper_fails_filter(monkeypatch, patched_paths):
    """Low-quality paper = weak venue + post-2010 sample + graveyard hit."""
    from engine.research.graveyard import GraveyardMatch
    monkeypatch.setattr(
        "engine.research.graveyard.check_against_graveyard",
        lambda c, **kw: GraveyardMatch(
            matched=True, signals_matched=["family_match"], matched_entries=[],
            overall_confidence=0.8, recommendation="block",
            explanation="dead family", cousin_count_in_family=2,
            elevated=False,
        ),
    )
    score = score_paper(PaperMetadata(
        title="X", venue="SSRN", authors="",
        abstract="Sample 2018-2023 of post-IPO returns."))
    assert score.passes_filter is False


def test_weights_must_sum_to_one(patched_paths):
    with pytest.raises(ValueError):
        score_paper(PaperMetadata(title="X"),
                       weights={"venue_tier": 0.5, "first_author_track": 0.5,
                                  "sample_window": 0.5,
                                  "mechanism_novelty": 0.5,
                                  "cite_count_age_adj": 0.5})


def test_threshold_override_works(patched_paths):
    score = score_paper(PaperMetadata(title="X"), threshold=0.9)
    assert score.threshold == 0.9


# ── Auditability (STRICT RED LINE) ────────────────────────────────────────

def test_score_dict_contains_full_audit_trail(patched_paths):
    score = score_paper(PaperMetadata(title="X", venue="JF"))
    d = score.to_dict()
    assert "features" in d
    assert "feature_explanations" in d
    assert "passes_filter" in d
    # All 5 features must be present + explained
    for f in DEFAULT_WEIGHTS:
        assert f in d["features"]
        assert f in d["feature_explanations"]
        assert len(d["feature_explanations"][f]) > 0


def test_score_dict_does_not_contain_verdict(patched_paths):
    """STRICT RED LINE: scorer outputs advisory, never a verdict label."""
    score = score_paper(PaperMetadata(title="X"))
    d = score.to_dict()
    forbidden = ("verdict", "decision", "recommendation",
                  "should_run", "auto_deploy")
    keys = " ".join(d.keys()).lower()
    for f in forbidden:
        assert f not in keys, f"forbidden field {f!r} in scorer output"


# ── update_author_track input validation ──────────────────────────────────

def test_update_author_track_validates_outcome(patched_paths):
    with pytest.raises(ValueError):
        cs.update_author_track("anyone", "neutral")


def test_update_author_track_appends_jsonl(patched_paths):
    cs.update_author_track("doe, jane", "pass")
    cs.update_author_track("doe, jane", "fail")
    contents = (patched_paths / "author_track.jsonl").read_text(encoding="utf-8")
    lines = contents.strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["author"] == "doe, jane"
    assert rec1["outcome"] == "pass"


# ── explain_paper helper ─────────────────────────────────────────────────

def test_explain_paper_returns_readable_text(patched_paths):
    out = explain_paper(PaperMetadata(title="X", venue="JF"))
    assert "Paper:" in out
    assert "venue_tier" in out
    assert "TOTAL" in out
    assert "PASSES FILTER" in out
