"""Tests for engine.research.embeddings — vector RAG over local ledgers.

Strategy: write fake ledger files into a tempdir, monkeypatch
embeddings._REPO_ROOT and embeddings._INDEX_DIR to point inside it, then
exercise build_index + search.

Model loading is slow on cold start (~3s). Tests share the singleton
via the embeddings._model cache; first call pays once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_ledger_root(tmp_path, monkeypatch):
    """Build a fake research/feature_store tree, monkeypatch
    embeddings constants to point at it. Yields the tmp root."""
    from engine.research import embeddings as E

    research = tmp_path / "data" / "research"
    fs = tmp_path / "data" / "feature_store" / "_computed"
    research.mkdir(parents=True)
    fs.mkdir(parents=True)

    # Decay history — engineered for the synonym test:
    # mom_hedge has terrible trailing Sharpe -1.6, crisis_hedge near zero,
    # tsmom_book healthy. A "degraded performance" query should rank
    # mom_hedge / crisis_hedge above tsmom_book.
    decay_rows = [
        {"sleeve": "tsmom_book",   "library_id": "L1", "audit_date": "2026-05-01",
         "trailing_sharpe":  1.10, "alert_level": "OK"},
        {"sleeve": "mom_hedge",    "library_id": "L2", "audit_date": "2026-05-01",
         "trailing_sharpe": -1.62, "alert_level": "HARD",
         "recommendation": "decommission — strategy losing money"},
        {"sleeve": "crisis_hedge", "library_id": "L3", "audit_date": "2026-05-01",
         "trailing_sharpe":  0.04, "alert_level": "WARN"},
    ]
    with (research / "decay_sentinel_history.jsonl").open("w", encoding="utf-8") as f:
        for r in decay_rows:
            f.write(json.dumps(r) + "\n")

    # Two council runs — one APPROVE on carry, one REJECT on news novelty.
    council_rows = [
        {"run_id": "abc123", "ts": "2026-04-15T10:00:00",
         "consensus": "APPROVE", "stage": "candidate_review",
         "proposal": {"title": "Commodity roll-yield carry sleeve",
                      "family": "carry"},
         "rationale": "Net Sharpe 0.66 with deflated SR 0.998. Approve for paper."},
        {"run_id": "def456", "ts": "2026-04-20T10:00:00",
         "consensus": "REJECT",
         "proposal": {"title": "8-K body novelty signal",
                      "family": "news_attention"},
         "rationale": "Adjacent to two RED graveyard entries. Reject."},
    ]
    with (research / "council_runs.jsonl").open("w", encoding="utf-8") as f:
        for r in council_rows:
            f.write(json.dumps(r) + "\n")

    # Empty stubs so build_all doesn't crash on missing files
    (research / "l4_iterations.jsonl").write_text("", encoding="utf-8")
    (research / "pfh_suggestions.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(E, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(E, "_INDEX_DIR", research / "_embedding_index")
    yield tmp_path


def test_encode_returns_unit_normalized_384d(tmp_ledger_root):
    from engine.research import embeddings as E
    v = E.encode("hello world")
    assert v.shape == (384,)
    assert v.dtype == np.float32
    np.testing.assert_allclose(float((v * v).sum()), 1.0, atol=1e-4)


def test_encode_batch_shape(tmp_ledger_root):
    from engine.research import embeddings as E
    vs = E.encode(["alpha", "beta", "gamma"])
    assert vs.shape == (3, 384)
    # Each row is unit-normalized
    norms = np.linalg.norm(vs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_build_index_decay_creates_npz(tmp_ledger_root):
    from engine.research import embeddings as E
    summary = E.build_index("decay_audits")
    assert summary["ledger"] == "decay_audits"
    assert summary["n_rows_total"] == 3
    assert summary["n_new_embedded"] == 3
    assert summary["n_existing_kept"] == 0

    idx_file = E._INDEX_DIR / "decay_audits.npz"
    assert idx_file.is_file()


def test_build_index_incremental_keeps_existing(tmp_ledger_root):
    """Second build reuses existing rows (n_existing_kept == 3, no
    new embeddings)."""
    from engine.research import embeddings as E
    E.build_index("decay_audits")
    summary = E.build_index("decay_audits")
    assert summary["n_existing_kept"] == 3
    assert summary["n_new_embedded"] == 0


def test_search_decay_synonym_match(tmp_ledger_root):
    """The synonym promise: a "degraded performance" query should
    surface mom_hedge / crisis_hedge above tsmom_book even though the
    query contains neither "trailing" nor "Sharpe"."""
    from engine.research import embeddings as E
    E.build_index("decay_audits")
    hits = E.search("decay_audits", "which sleeve is underperforming badly", top_k=3)
    assert len(hits) == 3
    # Top hit should be mom_hedge or crisis_hedge — both are problematic
    # sleeves and semantically closer to "underperforming" than the
    # healthy tsmom_book.
    top_sleeve = hits[0]["sleeve"]
    assert top_sleeve in {"mom_hedge", "crisis_hedge"}, \
        f"expected mom_hedge or crisis_hedge as top hit, got {top_sleeve}"
    # All hits carry a semantic score for retrieval_mode probing
    assert all("_semantic_score" in h for h in hits)


def test_search_council_concept_match(tmp_ledger_root):
    """A council query about "roll yield" should rank the carry approve
    higher than the unrelated news_attention reject — even though the
    council snippet doesn't contain "roll" or "yield"."""
    from engine.research import embeddings as E
    E.build_index("council_runs")
    hits = E.search("council_runs", "what did the council say about roll yield", top_k=2)
    assert len(hits) == 2
    assert hits[0]["consensus"] == "APPROVE"
    assert hits[0]["proposal_family"] == "carry"


def test_search_missing_index_returns_empty(tmp_ledger_root):
    from engine.research import embeddings as E
    # Don't build any index, just search
    hits = E.search("decay_audits", "anything", top_k=3)
    assert hits == []


def test_index_status_reports_built_ledgers(tmp_ledger_root):
    from engine.research import embeddings as E
    E.build_index("decay_audits")
    status = E.index_status()
    assert status["decay_audits"]["indexed"] is True
    assert status["decay_audits"]["n_rows"] == 3
    assert status["decay_audits"]["embed_dim"] == 384
    assert status["council_runs"]["indexed"] is False
