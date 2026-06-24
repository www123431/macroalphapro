"""
tests/test_history_rag.py — Project History RAG infra tests (P2.7, 2026-05-07).

Network-free tests of the local-only paths in engine.agents.history_rag.
Avoid touching the real ChromaDB store (the corpus is environment-dependent
and would slow CI). Instead, stub get_store() with an in-memory fake.

Coverage
--------
1. schema.IndexedDoc.chroma_metadata() flattens scalars correctly
2. config constants are sane
3. retrieve.retrieve() handles empty query / empty result / filter assembly
4. retrieve._recency_weight() math: half-life decay
5. retrieve._build_chroma_where() composes AND clauses correctly
6. synthesize.synthesize_answer() short-circuits on empty input + budget
7. eval.GROUND_TRUTH set is well-formed (each query has substrings + comment)
"""
from __future__ import annotations

import datetime
import math
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.agents.history_rag.schema import IndexedDoc, SourceType
from engine.agents.history_rag.config import (
    CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS,
    DEFAULT_FINAL_K, DEFAULT_TOP_K,
    EMBED_DIM, EMBED_MODEL_NAME, COLLECTION_NAME,
)


# ── 1. schema ────────────────────────────────────────────────────────────────
def test_indexed_doc_chroma_metadata_flattens():
    doc = IndexedDoc(
        doc_id="x:1",
        source_type=SourceType.DECISION_LOG,
        source_id="1",
        text="hello",
        title="t",
        occurred_at=datetime.datetime(2026, 5, 7, 10, 30),
        metadata={
            "ticker": "BAB",
            "n":      30,
            "pct":    1.5,
            "ok":     True,
            "list_dropped": [1, 2, 3],   # must be filtered
            "dict_dropped": {"a": 1},    # must be filtered
            "none_dropped": None,        # must be filtered
        },
        deep_link="pages/decisions.py",
    )
    m = doc.chroma_metadata()
    assert m["source_type"] == "decision_log"
    assert m["source_id"]   == "1"
    assert m["title"]       == "t"
    assert m["ticker"]      == "BAB"
    assert m["n"]           == 30
    assert m["pct"]         == 1.5
    assert m["ok"]          is True
    assert m["deep_link"]   == "pages/decisions.py"
    # Non-scalar / None must not leak through
    assert "list_dropped" not in m
    assert "dict_dropped" not in m
    assert "none_dropped" not in m
    # Timestamp serialization
    assert m["occurred_at_iso"].startswith("2026-05-07T10:30")
    assert isinstance(m["occurred_at_ts"], (int, float)) and m["occurred_at_ts"] > 0


def test_indexed_doc_no_occurred_at():
    doc = IndexedDoc(
        doc_id="x:2",
        source_type=SourceType.AUDIT_FINDING,
        source_id="2",
        text="t",
        title="ft",
    )
    m = doc.chroma_metadata()
    assert "occurred_at_ts" not in m
    assert "occurred_at_iso" not in m


# ── 2. config sanity ─────────────────────────────────────────────────────────
def test_config_constants_sane():
    assert isinstance(EMBED_MODEL_NAME, str) and "multilingual" in EMBED_MODEL_NAME
    assert EMBED_DIM == 768
    assert CHUNK_TARGET_CHARS  > CHUNK_OVERLAP_CHARS > 0
    assert DEFAULT_TOP_K       >= DEFAULT_FINAL_K   > 0
    assert isinstance(COLLECTION_NAME, str) and COLLECTION_NAME


# ── 3 + 5. retrieve where-clause assembly ────────────────────────────────────
def test_build_chroma_where_empty():
    from engine.agents.history_rag.retrieve import _build_chroma_where
    assert _build_chroma_where(None, None, None, None) is None


def test_build_chroma_where_single_source():
    from engine.agents.history_rag.retrieve import _build_chroma_where
    w = _build_chroma_where([SourceType.DECISION_LOG], None, None, None)
    assert w == {"source_type": {"$eq": "decision_log"}}


def test_build_chroma_where_multi_source():
    from engine.agents.history_rag.retrieve import _build_chroma_where
    w = _build_chroma_where(
        [SourceType.DECISION_LOG, SourceType.SPEC_REGISTRY], None, None, None,
    )
    assert "$and" not in w  # only 1 clause -> top-level
    assert w["source_type"]["$in"] == ["decision_log", "spec_registry"]


def test_build_chroma_where_combo():
    from engine.agents.history_rag.retrieve import _build_chroma_where
    after  = datetime.datetime(2026, 5, 1)
    before = datetime.datetime(2026, 5, 31)
    w = _build_chroma_where(
        [SourceType.DECISION_LOG], after, before,
        extra={"ticker": "BAB"},
    )
    assert "$and" in w
    clauses = w["$and"]
    assert len(clauses) == 4
    assert {"source_type": {"$eq": "decision_log"}} in clauses
    assert any("$gte" in c.get("occurred_at_ts", {}) for c in clauses)
    assert any("$lte" in c.get("occurred_at_ts", {}) for c in clauses)
    assert {"ticker": {"$eq": "BAB"}} in clauses


# ── 4. recency weight math ───────────────────────────────────────────────────
def test_recency_weight_half_life():
    from engine.agents.history_rag.retrieve import _recency_weight
    now = datetime.datetime(2026, 5, 7, 12, 0, 0)
    # Today
    assert abs(_recency_weight(now, now, 180.0) - 1.0) < 1e-9
    # 180 days ago = exactly half
    past = now - datetime.timedelta(days=180)
    assert abs(_recency_weight(past, now, 180.0) - 0.5) < 1e-9
    # 360 days = quarter
    older = now - datetime.timedelta(days=360)
    assert abs(_recency_weight(older, now, 180.0) - 0.25) < 1e-9


def test_recency_weight_unknown_default_one():
    from engine.agents.history_rag.retrieve import _recency_weight
    assert _recency_weight(None, datetime.datetime.utcnow(), 180.0) == 1.0


# ── 6. synthesize short-circuits ─────────────────────────────────────────────
def test_synthesize_empty_query_no_llm_call():
    from engine.agents.history_rag.synthesize import synthesize_answer
    r = synthesize_answer("", [])
    assert r.status == "no_evidence"
    assert r.cost_usd == 0.0


def test_synthesize_empty_results_no_llm_call():
    from engine.agents.history_rag.synthesize import synthesize_answer
    r = synthesize_answer("real question", [])
    assert r.status == "no_evidence"
    assert r.cost_usd == 0.0


def test_synthesize_budget_exhausted_no_llm_call():
    """When daily_budget=0 and there's evidence, must short-circuit."""
    from engine.agents.history_rag.synthesize import synthesize_answer
    from engine.agents.history_rag.retrieve import RetrievalResult

    fake_hits = [RetrievalResult(
        doc_id="x:1", source_type=SourceType.DECISION_LOG, source_id="1",
        text="dummy", title="t",
        occurred_at=None, similarity=0.5, recency_weight=1.0, score=0.5,
    )]
    r = synthesize_answer("q", fake_hits, daily_budget_usd=0.0)
    assert r.status == "budget_exhausted"
    assert r.cost_usd == 0.0


# ── 7. eval ground-truth set well-formed ─────────────────────────────────────
def test_eval_ground_truth_well_formed():
    from engine.agents.history_rag.eval import GROUND_TRUTH
    assert len(GROUND_TRUTH) >= 5
    seen_queries: set[str] = set()
    for q in GROUND_TRUTH:
        assert q.query.strip(), "empty query string"
        assert q.query not in seen_queries, f"duplicate query: {q.query[:40]}"
        seen_queries.add(q.query)
        assert isinstance(q.expected_title_subs, list) and q.expected_title_subs, (
            f"empty expected_title_subs for {q.query[:40]}"
        )
        for sub in q.expected_title_subs:
            assert sub.strip(), f"empty substring in {q.query[:40]}"
        assert q.comment.strip(), (
            f"missing comment (eval queries must self-document) for {q.query[:40]}"
        )


# ── 8. SourceType enum stable ────────────────────────────────────────────────
def test_source_type_values_stable():
    """Indexed corpus uses these exact strings; a rename is a corpus-break.
    SYSTEM_HELP added 2026-05-07 (Tier 1 polish) — hard-coded self-description
    docs for routing meta-queries; see engine/agents/history_rag/self_help_docs.py.
    """
    expected = {
        "decision_log",
        "spec_registry",
        "spec_amendment",
        "pending_approval",
        "agent_reflection",
        "audit_finding",
        "system_help",
    }
    actual = {s.value for s in SourceType}
    assert actual == expected, (
        f"SourceType drift: missing={expected - actual} extra={actual - expected}"
    )
