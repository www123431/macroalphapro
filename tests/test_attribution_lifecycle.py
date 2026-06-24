"""tests/test_attribution_lifecycle.py — Layer 4 piece 3b.

Tests the read-time JOIN that produces CandidateLifecycle records and
the four aggregate rollups.

Strategy: mock _load_hypotheses / _load_verdicts / _load_resolutions
to inject the exact shape we want to verify. The JOIN logic + final
state ladder is the unit under test, not the underlying store schemas.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


# ────────────────────────────────────────────────────────────────────
# Helpers — build fake Hypothesis-shaped namespace + seed events file
# ────────────────────────────────────────────────────────────────────
def _hyp(hypothesis_id, claim, created_ts="2026-06-07T00:00:00Z",
         synthesizes_paper_ids=(), citation_quality=None, version=1,
         extraction_method_value="llm_synthesis",
         mechanism_family_value="behavioral"):
    """Build a SimpleNamespace matching the Hypothesis fields the
    lifecycle code reads. Avoids the full schema construction cost."""
    return SimpleNamespace(
        hypothesis_id          = hypothesis_id,
        version                = version,
        created_ts             = created_ts,
        claim                  = claim,
        synthesizes_paper_ids  = synthesizes_paper_ids,
        citation_quality       = citation_quality,
        extraction_method      = SimpleNamespace(value=extraction_method_value),
        mechanism_family       = SimpleNamespace(value=mechanism_family_value),
    )


def _seed_events(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _seed_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture(autouse=True)
def _clear_helper_caches():
    """Each test starts with cleared lru_caches in helpers.py."""
    from engine.agents.attribution import helpers
    helpers.clear_caches()
    yield
    helpers.clear_caches()


@pytest.fixture
def lc_module(tmp_path, monkeypatch):
    """Redirect lifecycle.py's store paths + helpers cache to tmp."""
    from engine.agents.attribution import lifecycle, helpers

    events_path      = tmp_path / "events.jsonl"
    verdicts_path    = tmp_path / "verdicts.jsonl"
    resolutions_path = tmp_path / "resolutions.jsonl"
    cache_path       = tmp_path / "cache.jsonl"
    watchlist_path   = tmp_path / "watchlist.yaml"

    monkeypatch.setattr(lifecycle, "_EVENTS_PATH", events_path)
    monkeypatch.setattr(lifecycle, "_VERDICTS_PATH", verdicts_path)
    monkeypatch.setattr(lifecycle, "_RESOLUTIONS_PATH", resolutions_path)
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache_path)

    from engine.agents.papers_curator import watchlist as wl_mod
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", watchlist_path)

    helpers.clear_caches()

    return SimpleNamespace(
        lifecycle = lifecycle,
        events    = events_path,
        verdicts  = verdicts_path,
        resolutions = resolutions_path,
        cache     = cache_path,
        watchlist = watchlist_path,
    )


def _mock_hypotheses(monkeypatch, lifecycle, hyps):
    """Replace _load_hypotheses with a function that returns {id: h}."""
    by_id = {h.hypothesis_id: h for h in hyps}
    monkeypatch.setattr(lifecycle, "_load_hypotheses", lambda: by_id)


# ────────────────────────────────────────────────────────────────────
# get_candidate_lifecycle — pre-B-review path
# ────────────────────────────────────────────────────────────────────
def test_lifecycle_pre_b_review(lc_module, monkeypatch):
    """Brand-new candidate with no B verdict yet → final_state =
    PRE_B_REVIEW."""
    lifecycle = lc_module.lifecycle
    h = _hyp("h1", "carry trade alpha decays after publication")
    _mock_hypotheses(monkeypatch, lifecycle, [h])

    lc = lifecycle.get_candidate_lifecycle("h1")
    assert lc is not None
    assert lc.final_state == lifecycle.FINAL_STATE_PRE_B_REVIEW
    assert lc.b_verdict_type is None
    assert lc.principal_decision is None
    assert lc.strict_gate_verdict is None


def test_lifecycle_unknown_id_returns_none(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    _mock_hypotheses(monkeypatch, lifecycle, [])
    assert lifecycle.get_candidate_lifecycle("never_seen") is None


# ────────────────────────────────────────────────────────────────────
# get_candidate_lifecycle — B-approved / rejected / amendment ladder
# ────────────────────────────────────────────────────────────────────
def test_lifecycle_b_rejected(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    h = _hyp("h2", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_jsonl(lc_module.verdicts, [{
        "hypothesis_id": "h2", "verdict_type": "REJECT",
        "confidence": 0.85, "review_ts": "2026-06-07T01:00:00Z",
    }])

    lc = lifecycle.get_candidate_lifecycle("h2")
    assert lc.final_state == lifecycle.FINAL_STATE_B_REJECTED
    assert lc.b_verdict_type == "REJECT"
    assert lc.b_confidence == 0.85


def test_lifecycle_b_approved_then_principal_approved(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    h = _hyp("h3", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_jsonl(lc_module.verdicts, [{
        "hypothesis_id": "h3", "verdict_type": "APPROVE_FOR_PIPELINE",
        "confidence": 0.72, "review_ts": "2026-06-07T01:00:00Z",
    }])
    _seed_jsonl(lc_module.resolutions, [{
        "hypothesis_id": "h3", "decision": "approved",
        "resolved_ts": "2026-06-07T02:00:00Z",
    }])

    lc = lifecycle.get_candidate_lifecycle("h3")
    assert lc.final_state == lifecycle.FINAL_STATE_PRINCIPAL_APPROVED
    assert lc.b_verdict_type == "APPROVE_FOR_PIPELINE"
    assert lc.principal_decision == "approved"


def test_lifecycle_amendment_state(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    h = _hyp("h4", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_jsonl(lc_module.verdicts, [{
        "hypothesis_id": "h4",
        "verdict_type": "DOCTRINE_AMENDMENT_NEEDED",
        "confidence": 0.6, "review_ts": "2026-06-07T01:00:00Z",
    }])
    lc = lifecycle.get_candidate_lifecycle("h4")
    assert lc.final_state == lifecycle.FINAL_STATE_B_AMENDMENT


def test_lifecycle_uses_latest_verdict(lc_module, monkeypatch):
    """Multiple verdicts → use most-recent review_ts."""
    lifecycle = lc_module.lifecycle
    h = _hyp("h5", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_jsonl(lc_module.verdicts, [
        {"hypothesis_id": "h5", "verdict_type": "REJECT",
         "confidence": 0.5, "review_ts": "2026-06-07T01:00:00Z"},
        {"hypothesis_id": "h5", "verdict_type": "APPROVE_FOR_PIPELINE",
         "confidence": 0.8, "review_ts": "2026-06-07T03:00:00Z"},
    ])
    lc = lifecycle.get_candidate_lifecycle("h5")
    assert lc.b_verdict_type == "APPROVE_FOR_PIPELINE"


# ────────────────────────────────────────────────────────────────────
# Strict-gate JOIN via events.jsonl
# ────────────────────────────────────────────────────────────────────
def test_lifecycle_skipped_pre_compute(lc_module, monkeypatch):
    """candidate_skipped_pre_compute event → SKIPPED final state."""
    lifecycle = lc_module.lifecycle
    h = _hyp("h6", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_events(lc_module.events, [{
        "event_type": "candidate_skipped_pre_compute",
        "subject_id": "auto_aaaaaaaa1111",
        "metrics": {"source_hypothesis_id": "h6"},
        "ts": "2026-06-07T05:00:00Z",
    }])
    lc = lifecycle.get_candidate_lifecycle("h6")
    assert lc.strict_gate_subject_id == "auto_aaaaaaaa1111"
    assert lc.strict_gate_verdict == "SKIPPED"
    assert lc.final_state == lifecycle.FINAL_STATE_SKIPPED_PRE_COMPUTE


def test_lifecycle_green_via_factor_verdict(lc_module, monkeypatch):
    """pipeline_started → factor_verdict_filed GREEN."""
    lifecycle = lc_module.lifecycle
    h = _hyp("h7", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_events(lc_module.events, [
        {"event_type": "candidate_pipeline_started",
         "subject_id": "auto_aaaaaaaa2222",
         "metrics": {"source_hypothesis_id": "h7"},
         "ts": "2026-06-07T05:00:00Z"},
        {"event_type": "factor_verdict_filed",
         "subject_id": "auto_aaaaaaaa2222",
         "verdict": "GREEN",
         "metrics": {"score": 5},
         "ts": "2026-06-07T06:00:00Z"},
    ])
    lc = lifecycle.get_candidate_lifecycle("h7")
    assert lc.strict_gate_verdict == "GREEN"
    assert lc.strict_gate_score == 5
    assert lc.final_state == lifecycle.FINAL_STATE_GREEN


def test_lifecycle_red_via_factor_verdict(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    h = _hyp("h8", "x")
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_events(lc_module.events, [
        {"event_type": "candidate_pipeline_started",
         "subject_id": "auto_aaaaaaaa3333",
         "metrics": {"source_hypothesis_id": "h8"},
         "ts": "2026-06-07T05:00:00Z"},
        {"event_type": "factor_verdict_filed",
         "subject_id": "auto_aaaaaaaa3333",
         "verdict": "RED",
         "metrics": {"score": 1},
         "ts": "2026-06-07T06:00:00Z"},
    ])
    lc = lifecycle.get_candidate_lifecycle("h8")
    assert lc.strict_gate_verdict == "RED"
    assert lc.final_state == lifecycle.FINAL_STATE_RED


# ────────────────────────────────────────────────────────────────────
# Source / watchlist attribution into the lifecycle record
# ────────────────────────────────────────────────────────────────────
def test_lifecycle_attaches_sources_and_watchlist_authors(
    lc_module, monkeypatch,
):
    """Hypothesis cites 2 papers; one is watchlisted, one isn't."""
    from engine.agents.papers_curator.watchlist import add_author

    lifecycle = lc_module.lifecycle
    add_author("Tim Bollerslev", rationale="VRP",
                path=lc_module.watchlist)

    _seed_jsonl(lc_module.cache, [
        {"source": "semantic_scholar", "source_id": "ssp1",
         "title": "x", "authors": ["Tim Bollerslev", "Coauthor"]},
        {"source": "arxiv", "source_id": "2606.x",
         "title": "y", "authors": ["Random Researcher"]},
    ])

    h = _hyp("h9", "VRP carry hybrid",
             synthesizes_paper_ids=("semantic_scholar/ssp1",
                                      "arxiv/2606.x"))
    _mock_hypotheses(monkeypatch, lifecycle, [h])

    lc = lifecycle.get_candidate_lifecycle("h9")
    assert lc.cited_paper_sources == {
        "semantic_scholar/ssp1": "semantic_scholar",
        "arxiv/2606.x":          "arxiv",
    }
    assert lc.cited_watchlist_authors == ("Tim Bollerslev",)


# ────────────────────────────────────────────────────────────────────
# Doctrine snippet ids recovered from synthesis event
# ────────────────────────────────────────────────────────────────────
def test_lifecycle_recovers_doctrine_snippet_ids(lc_module, monkeypatch):
    """A candidate's lifecycle should carry the doctrine_snippet_ids
    A was looking at when it wrote the candidate. Match is by claim
    substring (synthesis event does NOT carry hypothesis_id directly)."""
    lifecycle = lc_module.lifecycle
    h = _hyp("h10",
             "carry trade post-publication decay confirmed by McLean")
    _mock_hypotheses(monkeypatch, lifecycle, [h])

    _seed_events(lc_module.events, [{
        "event_type": "papers_curator_synthesis_run",
        "ts": "2026-06-07T05:00:00Z",
        "metrics": {
            "doctrine_snippet_ids": [
                "feedback_loop_is_robustness_doctrine_2026-05-31",
                "project_position_weighting_precision_queued_2026-06-02",
            ],
            "candidates_summary": [{
                "claim": "carry trade post-publication decay "
                          "confirmed by McLean",
                "expected_outcome_prior": "moderate_GREEN",
            }],
        },
    }])

    lc = lifecycle.get_candidate_lifecycle("h10")
    assert lc.doctrine_snippet_ids == (
        "feedback_loop_is_robustness_doctrine_2026-05-31",
        "project_position_weighting_precision_queued_2026-06-02",
    )
    assert lc.a_expected_outcome_prior == "moderate_GREEN"


# ────────────────────────────────────────────────────────────────────
# Aggregates — empty data path
# ────────────────────────────────────────────────────────────────────
def test_all_aggregates_empty_when_no_hypotheses(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    _mock_hypotheses(monkeypatch, lifecycle, [])
    assert lifecycle.aggregate_by_author() == ()
    assert lifecycle.aggregate_by_source() == ()
    assert lifecycle.aggregate_by_doctrine_snippet() == ()
    assert lifecycle.calibration_a_confidence() == ()


# ────────────────────────────────────────────────────────────────────
# Aggregates — by_source counts a single candidate once per source
# ────────────────────────────────────────────────────────────────────
def test_aggregate_by_source_counts_each_source_once_per_candidate(
    lc_module, monkeypatch,
):
    """A candidate citing 2 arxiv papers + 1 SS paper counts as 1
    arxiv-cited + 1 SS-cited, NOT 2+1."""
    lifecycle = lc_module.lifecycle
    _seed_jsonl(lc_module.cache, [
        {"source": "arxiv", "source_id": "a1",
         "title": "x", "authors": ["A"]},
        {"source": "arxiv", "source_id": "a2",
         "title": "y", "authors": ["B"]},
        {"source": "semantic_scholar", "source_id": "s1",
         "title": "z", "authors": ["C"]},
    ])
    h = _hyp("hX", "claim",
             synthesizes_paper_ids=("arxiv/a1", "arxiv/a2",
                                      "semantic_scholar/s1"))
    _mock_hypotheses(monkeypatch, lifecycle, [h])

    rows = lifecycle.aggregate_by_source()
    by_src = {r.source: r for r in rows}
    assert by_src["arxiv"].n_candidates_cited == 1
    assert by_src["semantic_scholar"].n_candidates_cited == 1


def test_aggregate_by_source_promotes_through_gates(lc_module, monkeypatch):
    """A GREEN candidate should count toward n_green for every source
    it cited."""
    lifecycle = lc_module.lifecycle
    _seed_jsonl(lc_module.cache, [
        {"source": "arxiv", "source_id": "a1",
         "title": "x", "authors": ["A"]},
    ])
    h = _hyp("hY", "claim",
             synthesizes_paper_ids=("arxiv/a1",))
    _mock_hypotheses(monkeypatch, lifecycle, [h])
    _seed_jsonl(lc_module.verdicts, [{
        "hypothesis_id": "hY", "verdict_type": "APPROVE_FOR_PIPELINE",
        "confidence": 0.8, "review_ts": "2026-06-07T01:00:00Z",
    }])
    _seed_jsonl(lc_module.resolutions, [{
        "hypothesis_id": "hY", "decision": "approved",
        "resolved_ts": "2026-06-07T02:00:00Z",
    }])
    _seed_events(lc_module.events, [
        {"event_type": "candidate_pipeline_started",
         "subject_id": "auto_abc", "ts": "2026-06-07T03:00:00Z",
         "metrics": {"source_hypothesis_id": "hY"}},
        {"event_type": "factor_verdict_filed",
         "subject_id": "auto_abc", "verdict": "GREEN",
         "metrics": {"score": 5}, "ts": "2026-06-07T04:00:00Z"},
    ])

    rows = lifecycle.aggregate_by_source()
    arxiv_row = next(r for r in rows if r.source == "arxiv")
    assert arxiv_row.n_candidates_cited == 1
    assert arxiv_row.n_b_approved == 1
    assert arxiv_row.n_principal_approved == 1
    assert arxiv_row.n_strict_gate_run == 1
    assert arxiv_row.n_green == 1
    assert arxiv_row.conversion_rate_to_green == 1.0


# ────────────────────────────────────────────────────────────────────
# Aggregates — by_author requires watchlist intersection
# ────────────────────────────────────────────────────────────────────
def test_aggregate_by_author_only_includes_watchlist_members(
    lc_module, monkeypatch,
):
    from engine.agents.papers_curator.watchlist import add_author

    lifecycle = lc_module.lifecycle
    add_author("Tim Bollerslev", rationale="VRP",
                path=lc_module.watchlist)
    _seed_jsonl(lc_module.cache, [
        {"source": "ss", "source_id": "p1",
         "title": "x", "authors": ["Tim Bollerslev"]},
        {"source": "ss", "source_id": "p2",
         "title": "y", "authors": ["Outsider Author"]},
    ])
    h1 = _hyp("hA", "claim", synthesizes_paper_ids=("ss/p1",))
    h2 = _hyp("hB", "claim", synthesizes_paper_ids=("ss/p2",))
    _mock_hypotheses(monkeypatch, lifecycle, [h1, h2])

    rows = lifecycle.aggregate_by_author()
    names = {r.author_name for r in rows}
    assert names == {"Tim Bollerslev"}
    bol = next(r for r in rows if r.author_name == "Tim Bollerslev")
    assert bol.n_candidates_cited == 1


# ────────────────────────────────────────────────────────────────────
# Aggregates — calibration partitions by tier
# ────────────────────────────────────────────────────────────────────
def test_calibration_partitions_by_tier(lc_module, monkeypatch):
    lifecycle = lc_module.lifecycle
    h1 = _hyp("hC", "alpha A is robust over crisis")
    h2 = _hyp("hD", "alpha B is moderate")
    _mock_hypotheses(monkeypatch, lifecycle, [h1, h2])
    _seed_events(lc_module.events, [
        {"event_type": "papers_curator_synthesis_run",
         "ts": "2026-06-07T05:00:00Z",
         "metrics": {
             "doctrine_snippet_ids": [],
             "candidates_summary": [
                 {"claim": "alpha A is robust over crisis",
                  "expected_outcome_prior": "strong_GREEN"},
                 {"claim": "alpha B is moderate",
                  "expected_outcome_prior": "moderate_GREEN"},
             ],
         }},
    ])

    rows = lifecycle.calibration_a_confidence()
    by_tier = {r.a_predicted_tier: r for r in rows}
    assert "strong_GREEN" in by_tier
    assert "moderate_GREEN" in by_tier
    assert by_tier["strong_GREEN"].n_candidates == 1
    assert by_tier["moderate_GREEN"].n_candidates == 1
    # No strict-gate run yet → actual_green_rate = 0
    assert by_tier["strong_GREEN"].actual_green_rate == 0.0
