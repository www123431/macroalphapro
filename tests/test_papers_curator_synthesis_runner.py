"""tests/test_papers_curator_synthesis_runner.py — Phase 2.0 step 5a.

Orchestrator tests. Mocks the LLM call + redirects gatherer paths to
tmp_path so no real I/O or LLM cost. Covers:

  - Happy path: gather → synthesize → write
  - Dry-run skips writer
  - Empty candidates path (LLM returned []) — valid, no errors
  - Snapshot summary fields populated
  - Errors caught + recorded, never raises
  - Custom tags propagated to written rows
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


def _ts(days_ago: int = 0) -> str:
    return (_dt.datetime.utcnow()
            - _dt.timedelta(days=days_ago)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_substrate(tmp_path: Path, monkeypatch):
    """Build a minimal substrate the gatherer will read from."""
    from engine.agents.papers_curator import synthesis_context as sc
    papers = tmp_path / "papers_curator"
    library = tmp_path / "library"
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(sc, "_PAPERS_DIR", papers)
    monkeypatch.setattr(sc, "_LIBRARY_DIR", library)
    monkeypatch.setattr(sc, "_EVENTS_PATH", events)

    papers.mkdir(parents=True, exist_ok=True)
    (papers / "cache.jsonl").write_text(
        json.dumps({"source": "arxiv", "source_id": "p1",
                     "title": "EM bond carry", "fetched_ts": _ts(2)}) + "\n",
        encoding="utf-8")
    (papers / "summaries.jsonl").write_text(
        json.dumps({"source": "arxiv", "source_id": "p1",
                     "thesis": "carry persists",
                     "recommended_action": "INGEST",
                     "summarized_ts": _ts(2)}) + "\n",
        encoding="utf-8")
    library.mkdir(parents=True, exist_ok=True)
    (library / "carry.yaml").write_text(
        "id: cross_asset_carry\nfamily: CARRY\nstatus_in_our_book: DEPLOYED\n",
        encoding="utf-8")
    events.write_text(
        json.dumps({"event_id": "ev1", "event_type": "factor_verdict_filed",
                     "subject_id": "x", "ts": _ts(3)}) + "\n",
        encoding="utf-8")


def _valid_candidate_dict():
    return {
        "claim": "QMJ on EM sovereign bonds delivers risk-adjusted excess",
        "mechanism_family": "carry",
        "mechanism_subtype": "qmj_em_sovereign",
        "predicted_direction": "positive",
        "predicted_magnitude": "Sharpe 0.5+ OOS",
        "required_data": ["EM sovereign bond returns"],
        "test_methodology": "long-short decile sort",
        "synthesizes_paper_ids": ["arxiv/p1"],
        "synthesizes_event_ids": ["ev1"],
        "addresses_decay_in": None,
        "cochrane_frame": "risk",
        "novelty_vs_known": "extension_to_em_sov",
        "estimated_n_trials_in_family": 5,
        "graveyard_conflicts": [],
        "doctrine_conflicts": [],
        "expected_outcome_prior": "marginal_per_HXZ",
    }


def _mock_llm(monkeypatch, tool_input):
    """Force run_synthesis's underlying llm_call to return a canned
    structured payload. Patches synthesis.llm_call (top-level import)
    so the synthesis module returns our pre-baked candidates."""
    from engine.agents.papers_curator import synthesis as sm
    from engine.llm.call import LLMCallResult, ToolCall

    def _fake_call(**kw):
        return LLMCallResult(
            text="",
            tool_calls=(ToolCall(id="tc", name="emit_synthesis",
                                  input=tool_input),),
            stop_reason="tool_use", model="claude-sonnet-4-6",
            provider="anthropic", cost_usd=0.08, latency_ms=4200,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr(sm, "llm_call", _fake_call)


def _mock_emit(monkeypatch, event_id: str = "ev-test-uuid"):
    """Phase 2.0 step 4c: runner calls emit.papers_curator_synthesis_run
    at the end. Without mocking, tests would write real events to
    data/research_store/events.jsonl AND require the subject to be
    registered. Mock returns a sentinel event_id."""
    from engine.research_store import emit
    captured = {}
    def _fake_emit(**kw):
        captured.update(kw)
        return event_id
    monkeypatch.setattr(emit, "papers_curator_synthesis_run", _fake_emit)
    return captured


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────
def test_runner_full_pipeline_persists_candidates(tmp_path, monkeypatch):
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    hyp_path = tmp_path / "hypotheses.jsonl"
    result = run_synthesis_pipeline(hypotheses_path=hyp_path)

    assert result["dry_run"] is False
    assert result["snapshot"]["recent_summaries"] == 1
    assert result["snapshot"]["deployed_sleeves"] == 1
    assert result["snapshot"]["recent_events"] == 1
    assert result["n_candidates"] == 1
    assert result["n_written"] == 1
    assert len(result["written_hypothesis_ids"]) == 1
    assert result["errors"] == []
    assert result["event_id"] == "ev-test-uuid"

    # Disk row check
    rows = [json.loads(ln) for ln in hyp_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 1
    assert rows[0]["extraction_method"] == "llm_synthesis"


def test_runner_candidates_include_generation_metadata(tmp_path, monkeypatch):
    """The metadata dropped by the WRITER (cochrane_frame etc.) must
    still appear in the runner's candidates list — that's how the UI
    will render the preview + step 4c will emit them as events."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    c = result["candidates"][0]
    assert c["cochrane_frame"] == "risk"
    assert c["expected_outcome_prior"] == "marginal_per_HXZ"
    assert c["novelty_vs_known"] == "extension_to_em_sov"
    assert c["estimated_n_trials_in_family"] == 5


# ─────────────────────────────────────────────────────────────────────
# Dry-run
# ─────────────────────────────────────────────────────────────────────
def test_runner_dry_run_skips_writer(tmp_path, monkeypatch):
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    hyp_path = tmp_path / "hypotheses.jsonl"
    result = run_synthesis_pipeline(dry_run=True, hypotheses_path=hyp_path)
    assert result["dry_run"] is True
    assert result["n_candidates"] == 1
    assert result["n_written"] == 0
    assert result["written_hypothesis_ids"] == []
    assert not hyp_path.exists()


# ─────────────────────────────────────────────────────────────────────
# Empty / edge
# ─────────────────────────────────────────────────────────────────────
def test_runner_empty_candidates_no_error(tmp_path, monkeypatch):
    """LLM returning [] is valid (prefer empty over weak). Not an error."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": []})
    _mock_emit(monkeypatch)
    hyp_path = tmp_path / "hypotheses.jsonl"
    result = run_synthesis_pipeline(hypotheses_path=hyp_path)
    assert result["n_candidates"] == 0
    assert result["n_written"] == 0
    assert result["errors"] == []
    assert not hyp_path.exists()


def test_runner_empty_substrate_still_runs(tmp_path, monkeypatch):
    """No papers / no sleeves / no events — the pipeline must still
    complete cleanly (LLM gets sparse input + likely returns [])."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    from engine.agents.papers_curator import synthesis_context as sc
    monkeypatch.setattr(sc, "_PAPERS_DIR", tmp_path / "p")
    monkeypatch.setattr(sc, "_LIBRARY_DIR", tmp_path / "l")
    monkeypatch.setattr(sc, "_EVENTS_PATH", tmp_path / "e.jsonl")
    _mock_llm(monkeypatch, {"candidates": []})
    _mock_emit(monkeypatch)
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    assert result["snapshot"]["recent_summaries"] == 0
    assert result["snapshot"]["deployed_sleeves"] == 0
    assert result["snapshot"]["recent_events"] == 0
    assert result["errors"] == []


# ─────────────────────────────────────────────────────────────────────
# Error paths — never raise, always return structured errors
# ─────────────────────────────────────────────────────────────────────
def test_runner_gather_failure_recorded_not_raised(tmp_path, monkeypatch):
    """If the gatherer blows up, runner returns a structured error
    without raising — important for the API endpoint."""
    from engine.agents.papers_curator import synthesis_runner as sr
    def _boom(**kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(sr, "build_synthesis_input", _boom)
    result = sr.run_synthesis_pipeline()
    assert result["n_candidates"] == 0
    assert any("gather" in e for e in result["errors"])


def test_runner_synthesize_failure_recorded_not_raised(tmp_path, monkeypatch):
    from engine.agents.papers_curator import synthesis_runner as sr
    _seed_substrate(tmp_path, monkeypatch)
    def _boom(*a, **kw):
        raise RuntimeError("LLM rate limit")
    monkeypatch.setattr(sr, "run_synthesis", _boom)
    result = sr.run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    assert result["n_candidates"] == 0
    assert any("synthesize" in e for e in result["errors"])


# ─────────────────────────────────────────────────────────────────────
# Tagging
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# Phase 2.2b: citation verifier integration
# ─────────────────────────────────────────────────────────────────────
def _mock_citation_verifier(monkeypatch, *, raises=False,
                              checks_per_claim=None,
                              aggregate_returns=None):
    """Patch verify_citations + aggregate_citation_quality in the
    runner's _enrich_with_citation_checks call path."""
    from engine.agents.papers_curator import citation_verifier as cv

    if raises:
        def _fake_verify(**kw):
            raise RuntimeError("verifier exploded")
    else:
        def _fake_verify(**kw):
            return checks_per_claim if checks_per_claim is not None else ()

    def _fake_agg(checks):
        if aggregate_returns is not None:
            return aggregate_returns
        return {
            "n_papers_cited":      len(checks or ()),
            "n_resolved":          len(checks or ()),
            "n_unresolved":        0,
            "mean_confidence":     0.8,
            "min_confidence":      0.8,
            "any_unresolved":      False,
            "low_confidence_flag": False,
        }

    monkeypatch.setattr(cv, "verify_citations", _fake_verify)
    monkeypatch.setattr(cv, "aggregate_citation_quality", _fake_agg)


def test_runner_enriches_candidates_with_citation_quality(tmp_path, monkeypatch):
    """After run_synthesis returns candidates, each gets enriched with
    citation_verifications + citation_quality from the verifier."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    _mock_citation_verifier(monkeypatch)
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    assert result["n_candidates"] == 1
    c = result["candidates"][0]
    # citation_quality field present on the runner's candidate payload
    assert "citation_quality" in c
    assert c["citation_quality"]["mean_confidence"] == 0.8
    assert c["citation_quality"]["low_confidence_flag"] is False


def test_runner_low_confidence_flag_propagates(tmp_path, monkeypatch):
    """When the verifier flags a candidate as low_confidence (e.g.
    hallucinated citation), the flag MUST make it onto the candidate
    so downstream (B's prompt, audit event) can act on it."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    _mock_citation_verifier(monkeypatch,
        aggregate_returns={
            "n_papers_cited":      1,
            "n_resolved":          0,
            "n_unresolved":        1,
            "mean_confidence":     0.0,
            "min_confidence":      0.0,
            "any_unresolved":      True,
            "low_confidence_flag": True,
        })
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    c = result["candidates"][0]
    assert c["citation_quality"]["low_confidence_flag"] is True
    assert c["citation_quality"]["any_unresolved"] is True


def test_runner_verifier_failure_doesnt_drop_candidate(tmp_path, monkeypatch):
    """A verifier exception on ONE candidate must NOT cause that
    candidate to be dropped — better to write an un-verified row
    than to silently lose A's output. The error is recorded."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    _mock_citation_verifier(monkeypatch, raises=True)
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    assert result["n_candidates"] == 1
    assert any("citation" in e for e in result["errors"])
    # Candidate written despite verifier failure
    assert result["n_written"] == 1


def test_runner_no_candidates_no_verifier_call(tmp_path, monkeypatch):
    """Empty candidates list → don't fire verifier at all (cost
    discipline: no LLM calls when there's nothing to verify)."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": []})
    _mock_emit(monkeypatch)
    # If verifier IS called, this would raise — but it shouldn't be
    _mock_citation_verifier(monkeypatch, raises=True)
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")
    assert result["n_candidates"] == 0
    assert result["errors"] == []   # no verifier error → never called


def test_runner_extra_tags_propagate_to_written_rows(tmp_path, monkeypatch):
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    _mock_emit(monkeypatch)
    hyp_path = tmp_path / "hypotheses.jsonl"
    run_synthesis_pipeline(
        hypotheses_path = hyp_path,
        extra_tags      = ("session:cos-2026-06-06",),
    )
    row = json.loads(hyp_path.read_text(encoding="utf-8").strip())
    assert "synthesis" in row["tags"]
    assert "session:cos-2026-06-06" in row["tags"]


# ─────────────────────────────────────────────────────────────────────
# Step 4c: emit audit event
# ─────────────────────────────────────────────────────────────────────
def test_runner_emits_event_with_full_payload(tmp_path, monkeypatch):
    """Verify the emit call gets the right arguments — snapshot,
    candidates, dry_run flag, etc."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    captured = _mock_emit(monkeypatch, event_id="ev-payload-test")
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")

    assert result["event_id"] == "ev-payload-test"
    assert captured["n_candidates"] == 1
    assert captured["n_written"] == 1
    assert captured["dry_run"] is False
    assert captured["snapshot"]["recent_summaries"] == 1
    assert len(captured["candidates"]) == 1
    assert captured["candidates"][0]["cochrane_frame"] == "risk"
    assert captured["errors"] == []


def test_runner_emits_event_on_dry_run_too(tmp_path, monkeypatch):
    """dry_run must STILL emit — the audit trail wants to know A ran
    (with what snapshot, what was proposed in preview)."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    captured = _mock_emit(monkeypatch)
    result = run_synthesis_pipeline(dry_run=True, hypotheses_path=tmp_path / "h.jsonl")

    assert result["event_id"] is not None
    assert captured["dry_run"] is True
    assert captured["n_written"] == 0     # nothing persisted on dry-run


def test_runner_emits_event_on_empty_candidates(tmp_path, monkeypatch):
    """Honest-empty is data — 'A ran this week, returned 0' is exactly
    what the orchestrator needs to see in the audit trail."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": []})
    captured = _mock_emit(monkeypatch)
    result = run_synthesis_pipeline(hypotheses_path=tmp_path / "h.jsonl")

    assert result["event_id"] is not None
    assert captured["n_candidates"] == 0
    assert captured["candidates"] == []


def test_runner_emit_failure_recorded_not_raised(tmp_path, monkeypatch):
    """If emit blows up (e.g. registry temp issue), the run still
    succeeds; emit error is recorded but doesn't kill the result.
    Important: the candidates ARE already written to hypotheses.jsonl
    before this point — emit happens AFTER write."""
    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    from engine.research_store import emit as emit_mod
    _seed_substrate(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, {"candidates": [_valid_candidate_dict()]})
    def _broken_emit(**kw):
        raise RuntimeError("registry unreachable")
    monkeypatch.setattr(emit_mod, "papers_curator_synthesis_run", _broken_emit)
    hyp_path = tmp_path / "h.jsonl"
    result = run_synthesis_pipeline(hypotheses_path=hyp_path)

    # Write still succeeded
    assert result["n_written"] == 1
    assert hyp_path.exists()
    # Emit failure recorded
    assert result["event_id"] is None
    assert any("emit" in e for e in result["errors"])
