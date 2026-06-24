"""tests/test_emit_papers_curator_synthesis_run.py — Phase 2.0 step 4c.

Direct tests for the emit helper itself — verifies verdict logic,
metrics payload shape, candidates_summary trimming, tag wiring.

Uses monkeypatch on store.append + sessions module to avoid writing
real events. registry is real (`papers_curator` subject was registered
once at step 4c land-time; if missing, test registers it).
"""
from __future__ import annotations

from engine.research_store import emit
from engine.research_store.schema import EventType, SubjectType, Verdict


# Capture instead of writing — the inner _emit() calls store.append
# (and _publish_to_bus); we intercept the append.
def _patch_append(monkeypatch):
    captured: list = []
    def _fake_append(event):
        captured.append(event)
    from engine.research_store import store
    monkeypatch.setattr(store, "append", _fake_append)
    # Silence bus publish (best-effort, doesn't affect logic)
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "_publish_to_bus", lambda ev: None)
    return captured


def _ensure_subject_registered():
    """Idempotent — runs at module load to make sure the subject
    exists even on a fresh clone that hasn't run step 4c migration."""
    from engine.research_store import registry
    try:
        registry.require("papers_curator")
    except Exception:
        registry.register_subject(
            subject_id="papers_curator",
            subject_type=SubjectType.agent_run,
            description="Employee A papers curator (test guard)",
            created_by="test_emit_papers_curator_synthesis_run",
        )


_ensure_subject_registered()


SNAPSHOT = {
    "snapshot_ts":        "2026-06-06T13:00:00Z",
    "recent_summaries":   6,
    "deployed_sleeves":   5,
    "recent_events":      40,
    "doctrine_snippets":  0,
}


def _rich_candidate():
    return {
        "claim": "EM sovereign carry refresh delivers Sharpe 0.6 OOS",
        "mechanism_family": "carry",
        "mechanism_subtype": "qmj_em_sovereign",
        "predicted_direction": "positive",
        "predicted_magnitude": "Sharpe 0.5+ OOS",
        "synthesizes_paper_ids": ["arxiv/p1"],
        "synthesizes_event_ids": ["ev1"],
        "addresses_decay_in": None,
        "cochrane_frame": "risk",
        "novelty_vs_known": "extension_to_em_sov",
        "expected_outcome_prior": "marginal_per_HXZ",
        "graveyard_conflicts": [],
        "doctrine_conflicts": [],
    }


# ─────────────────────────────────────────────────────────────────────
# Event shape
# ─────────────────────────────────────────────────────────────────────
def test_emit_writes_correct_event_type_and_subject(monkeypatch):
    captured = _patch_append(monkeypatch)
    eid = emit.papers_curator_synthesis_run(
        n_candidates=1, n_written=1, snapshot=SNAPSHOT,
        candidates=[_rich_candidate()],
    )
    assert eid
    assert len(captured) == 1
    ev = captured[0]
    assert ev.event_type == EventType.papers_curator_synthesis_run
    assert ev.subject_id == "papers_curator"
    assert ev.subject_type == SubjectType.agent_run


def test_emit_metrics_carry_snapshot_and_counts(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=2, n_written=1, snapshot=SNAPSHOT,
        candidates=[_rich_candidate(), _rich_candidate()],
        errors=[],
    )
    m = captured[0].metrics
    assert m["n_candidates"] == 2
    assert m["n_written"] == 1
    assert m["n_dropped_by_writer"] == 1
    assert m["dry_run"] is False
    assert m["snapshot_papers"] == 6
    assert m["snapshot_sleeves"] == 5
    assert m["snapshot_events"] == 40
    assert m["snapshot_doctrine"] == 0


def test_emit_candidates_summary_keeps_audit_fields(monkeypatch):
    """The summary list must keep the metadata the writer drops
    (cochrane_frame / novelty / prior / conflicts) — that's the whole
    point of the audit event."""
    captured = _patch_append(monkeypatch)
    cand = _rich_candidate()
    cand["graveyard_conflicts"] = ["project-cross-asset-breadth-2026-05-28"]
    cand["doctrine_conflicts"]  = ["feedback-piece-by-piece-not-batch-2026-06-05"]
    emit.papers_curator_synthesis_run(
        n_candidates=1, n_written=1, snapshot=SNAPSHOT,
        candidates=[cand],
    )
    s = captured[0].metrics["candidates_summary"][0]
    assert s["cochrane_frame"] == "risk"
    assert s["novelty_vs_known"] == "extension_to_em_sov"
    assert s["expected_outcome_prior"] == "marginal_per_HXZ"
    assert "cross-asset-breadth" in s["graveyard_conflicts"][0]
    assert "piece-by-piece" in s["doctrine_conflicts"][0]
    assert s["n_papers"] == 1
    assert s["n_events"] == 1


# ─────────────────────────────────────────────────────────────────────
# Verdict logic
# ─────────────────────────────────────────────────────────────────────
def test_verdict_neutral_when_zero_candidates(monkeypatch):
    """Honest-empty is NEUTRAL, not GREEN — there's no positive result
    to report, but it's not a failure either."""
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=0, n_written=0, snapshot=SNAPSHOT, candidates=[],
    )
    assert captured[0].verdict == Verdict.NEUTRAL


def test_verdict_green_on_successful_persist(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=2, n_written=2, snapshot=SNAPSHOT,
        candidates=[_rich_candidate(), _rich_candidate()],
    )
    assert captured[0].verdict == Verdict.GREEN


def test_verdict_marginal_on_errors(monkeypatch):
    """Partial-success path: candidates were produced but write/emit
    flagged an error. Surface as MARGINAL so the UI shows a warning
    but doesn't bury the result."""
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=2, n_written=1, snapshot=SNAPSHOT,
        candidates=[_rich_candidate(), _rich_candidate()],
        errors=["write: validation failed on candidate 2"],
    )
    assert captured[0].verdict == Verdict.MARGINAL
    assert captured[0].metrics["errors_count"] == 1
    assert "validation failed" in captured[0].metrics["errors_sample"][0]


# ─────────────────────────────────────────────────────────────────────
# dry_run + tags
# ─────────────────────────────────────────────────────────────────────
def test_dry_run_tag_propagates(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=1, n_written=0, snapshot=SNAPSHOT,
        candidates=[_rich_candidate()], dry_run=True,
    )
    ev = captured[0]
    assert "synthesis" in ev.tags
    assert "dry_run" in ev.tags
    assert ev.metrics["dry_run"] is True


def test_non_dry_run_no_dry_run_tag(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=1, n_written=1, snapshot=SNAPSHOT,
        candidates=[_rich_candidate()],
    )
    ev = captured[0]
    assert "synthesis" in ev.tags
    assert "dry_run" not in ev.tags


# ─────────────────────────────────────────────────────────────────────
# Summary string
# ─────────────────────────────────────────────────────────────────────
def test_summary_string_human_readable(monkeypatch):
    captured = _patch_append(monkeypatch)
    emit.papers_curator_synthesis_run(
        n_candidates=2, n_written=1, snapshot=SNAPSHOT,
        candidates=[_rich_candidate(), _rich_candidate()],
        errors=["write: bad"],
    )
    s = captured[0].summary
    assert "2 candidate" in s
    assert "1 written" in s
    assert "error" in s
