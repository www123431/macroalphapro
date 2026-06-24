"""
Lightweight evaluation harness for the Project History RAG.

Defines a small fixed set of "ground-truth" queries paired with the
specific source docs that *should* appear in the top-K retrieval
results. Running ``python -m engine.agents.history_rag.eval`` (or
``scripts/run_history_rag_eval.py``) reports:

  - per-query: top-K hit recall vs expected source IDs
  - aggregate: mean recall@5, n_pass / n_total

This is **not** a research-grade eval — the corpus is too small and the
queries are project-specific. The point is to **detect drift**: if
someone changes the embedding model / chunking / filter logic and
recall@5 silently drops from 0.8 to 0.3, the harness catches it.

Adding new queries: append to ``GROUND_TRUTH`` below. Each entry is
``{query, expected_doc_substrings, must_include_source_type, comment}``.
A query "passes" if ≥1 hit in top-K matches *any* expected substring
in the title field (substring match — robust to chunking).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from engine.agents.history_rag.retrieve import retrieve
from engine.agents.history_rag.schema import SourceType

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    query:                  str
    expected_title_subs:    list[str]    # any one of these in result.title = pass
    must_include_src:       SourceType | None = None
    comment:                str          = ""


# Ground-truth set (small, project-specific). Treat as data fixture.
GROUND_TRUTH: list[EvalQuery] = [
    EvalQuery(
        query="为什么 ship BAB 策略 literature-conditional ship rule",
        expected_title_subs=[
            "spec_b_plus_mass_fdr_search",
            "BAB",
            "literature",
        ],
        must_include_src=SourceType.SPEC_AMENDMENT,
        comment="Anchor query. literature-conditional rule was introduced as a "
                "spec amendment to b_plus_mass_fdr_search.md, so a top hit must "
                "be either that spec or its amendment.",
    ),
    EvalQuery(
        query="HARKing detection R1 R4 pre-registration enforcement",
        expected_title_subs=[
            "pre_registration",
            "preregistration",
            "harking",
            "spec_pre_reg",
        ],
        must_include_src=SourceType.SPEC_REGISTRY,
        comment="HARKing R1-R4 detection lives in the pre-registration "
                "enforcement spec; that spec must be findable.",
    ),
    EvalQuery(
        query="amendment ledger spec_hash drift",
        expected_title_subs=[
            "spec_",
            "amendment",
            "preregistration",
        ],
        comment="Generic query about amendment infrastructure — at least one "
                "spec or spec_amendment doc should rank in top-K.",
    ),
    EvalQuery(
        query="regime overlay disabled REGIME_SCALE risk-off",
        expected_title_subs=[
            "regime",
            "risk_control",
            "PA #",
        ],
        comment="Regime overlay disabling history is in PA risk_control rows "
                "and decision_log narrative. Either source is acceptable.",
    ),
    EvalQuery(
        query="auto audit Tier R 11 critical rules layer 0 deterministic",
        expected_title_subs=[
            "auto_audit",
            "audit",
            "Tier R",
            "spec_audit",
        ],
        comment="Tier R 3-layer architecture is documented in spec_registry "
                "rows + amendment entries on auto_audit_proposer.py.",
    ),
]


@dataclass
class EvalResult:
    query:        str
    n_hits:       int
    pass_:        bool        # True if any hit's title contains one expected substring
    matched_sub:  str | None  # which substring matched (for diagnosis)
    top_titles:   list[str]
    top_scores:   list[float]
    src_check:    bool        # must_include_src constraint satisfied
    comment:      str


def _query_matches(hits, expected_subs, must_include_src):
    """Return (pass, matched_sub_or_None, src_constraint_ok)."""
    matched = None
    for h in hits:
        for sub in expected_subs:
            if sub.lower() in (h.title or "").lower():
                matched = sub
                break
        if matched:
            break
    src_ok = True
    if must_include_src is not None:
        src_ok = any(h.source_type == must_include_src for h in hits)
    return (matched is not None and src_ok, matched, src_ok)


def run_eval(top_k: int = 5) -> dict[str, Any]:
    """Run the full ground-truth eval. Returns aggregate + per-query results.

    Caller can inspect ``per_query`` for diagnosis or use ``aggregate``
    for a single PASS/FAIL gate.
    """
    per_query: list[EvalResult] = []
    n_pass = 0

    for q in GROUND_TRUTH:
        hits = retrieve(q.query, top_k=top_k)
        ok, matched, src_ok = _query_matches(hits, q.expected_title_subs, q.must_include_src)
        if ok:
            n_pass += 1
        per_query.append(EvalResult(
            query=q.query,
            n_hits=len(hits),
            pass_=ok,
            matched_sub=matched,
            top_titles=[h.title for h in hits[:top_k]],
            top_scores=[round(h.score, 3) for h in hits[:top_k]],
            src_check=src_ok,
            comment=q.comment,
        ))

    n_total = len(GROUND_TRUTH)
    return {
        "aggregate": {
            "n_total":   n_total,
            "n_pass":    n_pass,
            "recall_at_k": round(n_pass / n_total, 3) if n_total else 0.0,
            "top_k":     top_k,
        },
        "per_query": [
            {
                "query":        r.query,
                "pass":         r.pass_,
                "matched_sub":  r.matched_sub,
                "src_check":    r.src_check,
                "n_hits":       r.n_hits,
                "top_titles":   r.top_titles,
                "top_scores":   r.top_scores,
                "comment":      r.comment,
            } for r in per_query
        ],
    }


def main() -> int:
    """CLI entry point. Exits 0 if recall@K ≥ 0.6, 1 otherwise.

    The 0.6 threshold is intentionally lenient — small corpus + strict
    title-substring matching means even a working RAG sometimes misses
    one query. Drift detection still works: a regression to 0.3 fires.
    """
    import sys
    out = run_eval()
    print(json.dumps(out, indent=2, ensure_ascii=False))
    threshold = 0.6
    return 0 if out["aggregate"]["recall_at_k"] >= threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
