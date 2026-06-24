"""tests/test_strengthener_runner.py — Phase 2.0 step 11b.

Runner tests. Mocks review LLM call + uses tmp_path so no real I/O.
Covers:
  - Selects only PROPOSED + LLM_SYNTHESIS rows; ignores others
  - Skips hypotheses already in verdicts.jsonl (idempotency)
  - Persists verdicts to verdicts.jsonl
  - dry_run skips persistence
  - max_hypotheses caps the batch
  - One bad review doesn't kill the batch (per-hypothesis fail-safe)
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────
def _synthesis_hypothesis(hypothesis_id: str, family: str = "CARRY"):
    from engine.research_store.hypothesis import Hypothesis
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    from engine.research_store.red_lessons.mechanism_families import MechanismFamily
    return Hypothesis(
        hypothesis_id        = hypothesis_id,
        source_paper_id      = "",
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = (),
        verbatim_quotes      = (),
        claim                = f"EM sov carry candidate {hypothesis_id}",
        mechanism_family     = MechanismFamily(family),
        mechanism_subtype    = "qmj_em_sovereign",
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = "Sharpe 0.5+",
        required_data        = ("EM sov bond returns",),
        test_methodology     = "long-short decile",
        extraction_method    = ExtractionMethod.LLM_SYNTHESIS,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = "2026-06-06T12:00:00Z",
        updated_ts           = "2026-06-06T12:00:00Z",
        created_by           = "test",
        tags                 = ("synthesis",),
        synthesizes_paper_ids = ("arxiv/p1",),
        synthesizes_event_ids = ("ev1",),
        addresses_decay_in    = None,
    )


def _paper_rooted_hypothesis(hypothesis_id: str):
    """A paper-rooted (LLM_EXTRACT) hypothesis — runner should IGNORE."""
    from engine.research_store.hypothesis import Hypothesis, VerbatimQuote
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    from engine.research_store.red_lessons.mechanism_families import MechanismFamily
    return Hypothesis(
        hypothesis_id        = hypothesis_id,
        source_paper_id      = "paper-1",
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = ("chunk-1",),
        verbatim_quotes      = (
            VerbatimQuote(chunk_id="chunk-1",
                            quote_text="A verbatim quote at least 20 chars"),
            VerbatimQuote(chunk_id="chunk-1",
                            quote_text="Another verbatim quote at least 20 chars"),
        ),
        claim                = "Paper-rooted candidate",
        mechanism_family     = MechanismFamily.MOMENTUM,
        mechanism_subtype    = "12_1",
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = "Sharpe 0.4+",
        required_data        = ("US equity returns",),
        test_methodology     = "decile sort",
        extraction_method    = ExtractionMethod.LLM_EXTRACT,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = "2026-06-05T12:00:00Z",
        updated_ts           = "2026-06-05T12:00:00Z",
        created_by           = "test",
        tags                 = (),
    )


def _make_reject_verdict_for(hyp_id):
    from engine.agents.strengthener.review import StrengthenerVerdict, VerdictType
    return StrengthenerVerdict(
        hypothesis_id              = hyp_id,
        verdict_type               = VerdictType.REJECT,
        one_line_summary           = "Too similar to deployed carry",
        confidence                 = 0.8,
        reasoning                  = "overlap with cross_asset_carry",
        similar_to_deployed        = "cross_asset_carry",
        replaces_decaying          = None,
        blocking_doctrine_id       = None,
        proposed_amendment_summary = None,
        recommended_pipeline_action= None,
        risk_flags                 = ("overlap",),
        review_ts                  = "2026-06-06T13:00:00Z",
        model                      = "claude-sonnet-4-6",
    )


def _seed_hypotheses(tmp_path: Path, hyps: list) -> Path:
    """Write hypotheses to a fresh jsonl. Uses save_hypothesis with
    skip_cross_checks=True so synthesis rows (empty source_paper_id)
    don't fail registry resolution."""
    from engine.research_store.hypothesis.store import save_hypothesis
    p = tmp_path / "hypotheses.jsonl"
    for h in hyps:
        save_hypothesis(h, path=p, skip_cross_checks=True)
    return p


def _patch_context(monkeypatch):
    """Stub out the heavy gatherers (deployed_sleeves / family_verdicts /
    doctrine_snippets) so tests don't read real library yamls /
    events.jsonl / doctrine_chroma."""
    from engine.agents.strengthener import runner as rm
    monkeypatch.setattr(rm, "_load_deployed_sleeves", lambda: ())
    monkeypatch.setattr(rm, "_load_family_verdicts", lambda f, **kw: ())
    # Tier-2 Q3: signature is now (family, claim_hint="", top_k=3)
    monkeypatch.setattr(rm, "_load_doctrine_snippets",
                          lambda *, family, claim_hint="", top_k=3: ())


def _patch_review(monkeypatch, verdict_factory=None, raise_for=None):
    """verdict_factory: callable(StrengthenerInput) → StrengthenerVerdict | None.
    raise_for: hypothesis_id that should raise when reviewed."""
    from engine.agents.strengthener import runner as rm
    calls: list = []
    def _fake_review(si):
        calls.append(si.hypothesis.hypothesis_id)
        if raise_for and si.hypothesis.hypothesis_id == raise_for:
            raise RuntimeError("review bug")
        if verdict_factory is None:
            return _make_reject_verdict_for(si.hypothesis.hypothesis_id)
        return verdict_factory(si)
    monkeypatch.setattr(rm, "run_strengthener_review", _fake_review)
    return calls


# ─────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────
def test_selects_only_proposed_llm_synthesis_rows(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    calls = _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path, [
        _synthesis_hypothesis("syn1"),
        _synthesis_hypothesis("syn2"),
        _paper_rooted_hypothesis("paper1"),   # MUST be ignored
    ])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(
        hypotheses_path = hyps_path,
        verdicts_path   = verdicts_path,
    )
    assert result["n_candidates"] == 2
    assert result["n_reviewed"] == 2
    assert set(calls) == {"syn1", "syn2"}


def test_ignores_already_reviewed(tmp_path, monkeypatch):
    """Re-running the pipeline with the same input must NOT re-review
    hypotheses that already appear in verdicts.jsonl."""
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    calls = _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path, [
        _synthesis_hypothesis("syn1"),
        _synthesis_hypothesis("syn2"),
    ])
    verdicts_path = tmp_path / "verdicts.jsonl"
    # First pass — reviews both
    run_strengthener_pipeline(hypotheses_path=hyps_path,
                                verdicts_path=verdicts_path)
    assert len(calls) == 2
    # Second pass — re-runs, but both already reviewed → no calls
    result = run_strengthener_pipeline(hypotheses_path=hyps_path,
                                          verdicts_path=verdicts_path)
    assert result["n_candidates"] == 0
    assert result["n_reviewed"] == 0
    assert len(calls) == 2   # no additional review calls


def test_picks_up_new_proposed_rows_on_rerun(tmp_path, monkeypatch):
    """First pass reviews syn1; then we add syn2; second pass picks
    up only syn2."""
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    calls = _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path, [_synthesis_hypothesis("syn1")])
    verdicts_path = tmp_path / "verdicts.jsonl"
    run_strengthener_pipeline(hypotheses_path=hyps_path,
                                verdicts_path=verdicts_path)
    # Add a second synthesis row
    from engine.research_store.hypothesis.store import save_hypothesis
    save_hypothesis(_synthesis_hypothesis("syn2"), path=hyps_path,
                     skip_cross_checks=True)
    result = run_strengthener_pipeline(hypotheses_path=hyps_path,
                                          verdicts_path=verdicts_path)
    assert result["n_candidates"] == 1
    assert result["n_reviewed"] == 1
    assert calls[-1] == "syn2"


# ─────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────
def test_verdicts_persisted_to_jsonl(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path,
        [_synthesis_hypothesis(f"syn{i}") for i in range(3)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(hypotheses_path=hyps_path,
                                          verdicts_path=verdicts_path)
    assert result["n_persisted"] == 3
    rows = [json.loads(ln) for ln in verdicts_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 3
    assert all(r["verdict_type"] == "REJECT" for r in rows)
    assert {r["hypothesis_id"] for r in rows} == {"syn0", "syn1", "syn2"}


def test_dry_run_does_not_persist(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path, [_synthesis_hypothesis("syn1")])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(
        dry_run         = True,
        hypotheses_path = hyps_path,
        verdicts_path   = verdicts_path,
    )
    assert result["n_reviewed"] == 1
    assert result["n_persisted"] == 0
    assert not verdicts_path.exists()


# ─────────────────────────────────────────────────────────────────────
# Cost gate
# ─────────────────────────────────────────────────────────────────────
def test_max_hypotheses_caps_batch(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    calls = _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path,
        [_synthesis_hypothesis(f"syn{i}") for i in range(20)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(
        max_hypotheses  = 5,
        hypotheses_path = hyps_path,
        verdicts_path   = verdicts_path,
    )
    assert result["n_candidates"] == 20   # all detected
    assert result["n_reviewed"]   == 5    # but only 5 reviewed
    assert len(calls) == 5


# ─────────────────────────────────────────────────────────────────────
# Per-hypothesis fail-safe
# ─────────────────────────────────────────────────────────────────────
def test_review_exception_doesnt_kill_batch(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    _patch_review(monkeypatch, raise_for="syn2")
    hyps_path = _seed_hypotheses(tmp_path, [
        _synthesis_hypothesis("syn1"),
        _synthesis_hypothesis("syn2"),
        _synthesis_hypothesis("syn3"),
    ])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(hypotheses_path=hyps_path,
                                          verdicts_path=verdicts_path)
    assert result["n_reviewed"] == 2   # syn1 + syn3
    assert any("syn2" in e for e in result["errors"])
    rows = [json.loads(ln) for ln in verdicts_path.read_text(encoding="utf-8").strip().split("\n")]
    assert {r["hypothesis_id"] for r in rows} == {"syn1", "syn3"}


def test_review_returning_none_recorded_as_error(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    _patch_review(monkeypatch, verdict_factory=lambda si: None)
    hyps_path = _seed_hypotheses(tmp_path, [_synthesis_hypothesis("syn1")])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(hypotheses_path=hyps_path,
                                          verdicts_path=verdicts_path)
    assert result["n_reviewed"] == 0
    assert result["n_persisted"] == 0
    assert any("returned None" in e for e in result["errors"])


# ─────────────────────────────────────────────────────────────────────
# Tier-2 Q3: doctrine_snippets retrieval
# ─────────────────────────────────────────────────────────────────────
def test_doctrine_load_blank_family_and_claim_returns_empty(monkeypatch):
    """Cost discipline: blank inputs → don't fire chroma."""
    from engine.agents.strengthener import runner as rm
    out = rm._load_doctrine_snippets(family="", claim_hint="")
    assert out == ()


def test_doctrine_load_passes_family_and_claim_to_query(monkeypatch):
    """Topic anchor MUST combine family + claim so doctrine retrieval
    is candidate-specific (not just family-bucket)."""
    from engine.agents.strengthener import runner as rm
    captured = {}
    def _fake_query(topic, **kw):
        captured["topic"] = topic
        captured["top_k"] = kw.get("top_k")
        class _H:
            name = "feedback-x"
            description = "desc"
            snippet = "body"
            distance = 0.1
        return (_H(),)
    from engine.agents.papers_curator import doctrine_index as di
    monkeypatch.setattr(di, "query_doctrine", _fake_query)

    out = rm._load_doctrine_snippets(
        family     = "CARRY",
        claim_hint = "EM sovereign QMJ on bonds",
    )
    assert len(out) == 1
    assert "CARRY" in captured["topic"]
    assert "EM sovereign QMJ" in captured["topic"]


def test_doctrine_load_chroma_failure_returns_empty(monkeypatch):
    """Chroma down → B degrades to no-doctrine, same as pre-tier-2."""
    from engine.agents.strengthener import runner as rm
    from engine.agents.papers_curator import doctrine_index as di
    def _fake_query(topic, **kw):
        raise RuntimeError("chroma down")
    monkeypatch.setattr(di, "query_doctrine", _fake_query)
    assert rm._load_doctrine_snippets(family="CARRY",
                                        claim_hint="x") == ()


def test_doctrine_load_adapts_hit_to_DoctrineContextRef(monkeypatch):
    """doctrine_index returns DoctrineHit (different shape); the
    adapter must produce strengthener.DoctrineContextRef with the
    expected fields B's prompt reads."""
    from engine.agents.strengthener import runner as rm
    from engine.agents.papers_curator import doctrine_index as di
    class _H:
        name = "feedback-cross-asset-2026"
        description = "Cross-asset breadth focus"
        snippet = "12+ RED categories, do not pursue equity single-name."
        distance = 0.05
        file_path = "/x.md"
    monkeypatch.setattr(di, "query_doctrine",
                          lambda topic, **kw: (_H(),))
    out = rm._load_doctrine_snippets(family="EARNINGS_DRIFT",
                                       claim_hint="PEAD variant")
    assert len(out) == 1
    ref = out[0]
    assert ref.memory_file_id == "feedback-cross-asset-2026"
    assert ref.headline == "Cross-asset breadth focus"
    assert "12+ RED" in ref.snippet
    assert "distance=" in ref.relevance_note


def test_build_input_for_passes_claim_to_doctrine_load(monkeypatch):
    """The end-to-end wiring: build_input_for must pass the hypothesis
    claim into the doctrine query, not just the family. This is the
    Tier-2 quality leap — candidate-specific doctrine retrieval."""
    from engine.agents.strengthener import runner as rm
    _patch_context(monkeypatch)
    captured = {}
    def _fake_load(*, family, claim_hint="", top_k=3):
        captured["family"] = family
        captured["claim_hint"] = claim_hint
        return ()
    monkeypatch.setattr(rm, "_load_doctrine_snippets", _fake_load)

    h = _synthesis_hypothesis("syn-test", family="CARRY")
    si = rm.build_input_for(h)
    assert captured["family"] == "CARRY"
    # The claim from _synthesis_hypothesis is "EM sov carry candidate syn-test"
    assert "EM sov carry" in captured["claim_hint"]


# ─────────────────────────────────────────────────────────────────────
# Empty world
# ─────────────────────────────────────────────────────────────────────
def test_no_candidates_returns_clean_zero(tmp_path, monkeypatch):
    from engine.agents.strengthener.runner import run_strengthener_pipeline
    _patch_context(monkeypatch)
    calls = _patch_review(monkeypatch)
    hyps_path = _seed_hypotheses(tmp_path, [])
    verdicts_path = tmp_path / "verdicts.jsonl"
    result = run_strengthener_pipeline(hypotheses_path=hyps_path,
                                          verdicts_path=verdicts_path)
    assert result["n_candidates"] == 0
    assert result["n_reviewed"] == 0
    assert result["n_persisted"] == 0
    assert result["errors"] == []
    assert calls == []
