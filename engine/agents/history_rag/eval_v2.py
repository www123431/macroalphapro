"""engine.agents.history_rag.eval_v2 — proper IR eval for the project RAG.

Existing eval.py was a drift-detector with 5 ground-truth queries and
Recall@5 only. This module upgrades that to a research-grade eval with:

  - 20-30 ground-truth queries covering ALL SourceTypes
  - Standard IR metrics: Recall@K + Precision@K + MRR + nDCG@K
  - Adversarial queries (out-of-domain / should-not-retrieve)
  - Per-source-type breakdown
  - Latency p50/p95 per query

Run:
    python -m engine.agents.history_rag.eval_v2
    python -m engine.agents.history_rag.eval_v2 --k 10
    python -m engine.agents.history_rag.eval_v2 --json out.json

The eval is project-specific (corpus is small) but it's the right SHAPE
of an eval — what differentiates a "I have a RAG" claim from "I have
a RAG and here's its measured Recall@5 / MRR / nDCG@5".
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import math
import statistics
import time
from typing import Any

from engine.agents.history_rag.retrieve import retrieve
from engine.agents.history_rag.schema import SourceType

logger = logging.getLogger(__name__)


# ── Ground truth ────────────────────────────────────────────────


@_dc.dataclass
class EvalQueryV2:
    query:                  str
    expected_title_subs:    list[str]                # substring match in any returned title
    must_include_src:       SourceType | None = None
    category:               str = "general"          # spec / amendment / agent / audit / system / adversarial
    comment:                str = ""


GROUND_TRUTH: list[EvalQueryV2] = [
    # ── SPEC_REGISTRY / SPEC_AMENDMENT (specs are the densest source) ────
    EvalQueryV2(
        query="为什么 ship BAB 策略 literature-conditional ship rule",
        expected_title_subs=["spec_b_plus_mass_fdr_search", "BAB", "literature"],
        must_include_src=SourceType.SPEC_AMENDMENT,
        category="spec_amendment",
    ),
    EvalQueryV2(
        query="HARKing detection R1 R4 pre-registration enforcement",
        expected_title_subs=["pre_registration", "preregistration", "harking"],
        must_include_src=SourceType.SPEC_REGISTRY,
        category="spec_registry",
    ),
    EvalQueryV2(
        query="amendment ledger spec_hash drift",
        expected_title_subs=["spec_", "amendment", "preregistration"],
        category="spec_amendment",
    ),
    EvalQueryV2(
        query="auto audit Tier R 11 critical rules layer 0 deterministic",
        expected_title_subs=["auto_audit", "audit", "Tier R", "spec_audit"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="combined book vol target risk parity sleeve allocation",
        expected_title_subs=["combined_book", "vol_target", "risk_parity",
                              "sleeve", "spec_portfolio"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="commodity carry KMPV cross-asset everywhere",
        expected_title_subs=["carry", "commodity", "KMPV", "kmpv",
                              "cross_asset", "Koijen"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="MSM multivariate Markov regime switching",
        expected_title_subs=["msm", "MSM", "markov", "regime", "multivariate"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="data quality inspector pre batch post feed mode 1",
        expected_title_subs=["dq_inspector", "data_quality", "dq inspector",
                              "spec_dq"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="risk manager 13 modes pre trade post trade gate",
        expected_title_subs=["risk_manager", "risk manager", "spec_risk",
                              "pre_trade", "post_trade"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="capacity simulator AUM Pastor Stambaugh Berk Green",
        expected_title_subs=["capacity", "pastor", "berk", "stambaugh",
                              "spec_capacity"],
        category="spec_registry",
    ),

    # ── DECISION_LOG / AGENT_REFLECTION ──────────────────────────
    EvalQueryV2(
        query="regime overlay disabled REGIME_SCALE risk-off",
        expected_title_subs=["regime", "risk_control", "PA #", "decision"],
        category="decision_log",
    ),
    EvalQueryV2(
        query="Phase A v3 weighting ablation 1/N retained DGU 2009",
        expected_title_subs=["phase_a", "weighting", "1/N", "DGU",
                              "DeMiguel", "ablation"],
        category="decision_log",
    ),
    EvalQueryV2(
        query="path C earnings PEAD signal labor",
        expected_title_subs=["path_c", "pead", "earnings", "labor"],
        category="decision_log",
    ),

    # ── AUDIT_FINDING ─────────────────────────────────────────────
    EvalQueryV2(
        query="forensic ticker check anomaly z-score sigma",
        expected_title_subs=["forensic", "anomaly", "sentinel", "ticker"],
        category="audit_finding",
    ),

    # ── SYSTEM_HELP (meta-queries) ────────────────────────────────
    EvalQueryV2(
        query="what can the supervisor do how do I use it",
        expected_title_subs=["supervisor", "self_help", "help", "guide"],
        category="system_help",
    ),
    EvalQueryV2(
        query="how to navigate cockpit dashboard explain",
        expected_title_subs=["cockpit", "navigate", "dashboard", "help", "self"],
        category="system_help",
    ),

    # ── Hard / multi-anchor queries ──────────────────────────────
    EvalQueryV2(
        query="cousin spec graveyard family overlap warning",
        expected_title_subs=["cousin", "graveyard", "family", "overlap"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="ETF holdings monthly LLM risk monitor",
        expected_title_subs=["etf_holdings", "ETF Holdings",
                              "spec_etf", "monthly"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="point in time PIT look ahead bias audit",
        expected_title_subs=["pit", "PIT", "point_in_time", "look_ahead",
                              "look-ahead"],
        category="spec_registry",
    ),
    EvalQueryV2(
        query="council critique multi agent Devil's Advocate constrained evidence",
        expected_title_subs=["council", "devils_advocate", "critique",
                              "spec_council", "constrained"],
        category="spec_registry",
    ),

    # ── ADVERSARIAL (should NOT retrieve high-relevance — out of domain) ──
    EvalQueryV2(
        query="best restaurant in tokyo for sushi",
        expected_title_subs=[],     # nothing should match
        category="adversarial",
        comment="Out-of-domain query — top hit score should be low (<0.3 ideal).",
    ),
    EvalQueryV2(
        query="how to train a neural network for image classification",
        expected_title_subs=[],
        category="adversarial",
        comment="Adjacent-domain (ML general) but not project — should not return high-confidence project docs.",
    ),
]


# ── Per-query result ─────────────────────────────────────────────


@_dc.dataclass
class HitJudgement:
    rank:        int           # 1-based
    title:       str
    score:       float
    is_relevant: bool          # title-substring match against expected


@_dc.dataclass
class EvalResultV2:
    query:           str
    category:        str
    n_hits:          int
    judgements:      list[HitJudgement]
    src_check_ok:    bool      # must_include_src constraint satisfied
    elapsed_ms:      float
    top_score:       float     # top-1 cosine score (for adversarial threshold)
    comment:         str

    # Derived per-query metrics
    recall_at_1:     float
    recall_at_3:     float
    recall_at_5:     float
    recall_at_10:    float
    precision_at_5:  float
    rr:              float     # reciprocal rank of FIRST relevant hit (0 if none)
    ndcg_at_5:       float

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["judgements"] = [_dc.asdict(j) for j in self.judgements]
        return d


# ── Metric computation ──────────────────────────────────────────


def _is_relevant(title: str, text: str, expected_subs: list[str]) -> bool:
    """Check title OR body-text for any expected substring.

    Original eval.py only checked title — too narrow. spec_registry
    titles are mostly bare filenames (e.g. "Spec #49 docs/spec_etf_
    holdings_llm_risk_monitor.md") which don't contain semantic
    keywords like "data_quality". Checking body text covers cases
    where the relevant doc IS retrieved but its title doesn't match.
    """
    if not expected_subs:
        return False
    haystack = ((title or "") + " " + (text or "")).lower()
    return any(s.lower() in haystack for s in expected_subs)


def _ndcg_at_k(rels: list[int], k: int) -> float:
    """Standard nDCG@K on binary relevance. Ideal DCG = sum(1 / log2(i+1))
    for i = 1..min(K, n_relevant_at_all)."""
    if not rels:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    n_rel = min(sum(rels), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_rel))
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_query(q: EvalQueryV2, top_k: int = 10) -> EvalResultV2:
    t0 = time.perf_counter()
    hits = retrieve(q.query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    judgements = [
        HitJudgement(
            rank=i + 1,
            title=h.title,
            score=float(h.score),
            is_relevant=_is_relevant(h.title, h.text, q.expected_title_subs),
        )
        for i, h in enumerate(hits)
    ]
    rels = [1 if j.is_relevant else 0 for j in judgements]

    # Source-type constraint
    src_ok = True
    if q.must_include_src is not None:
        src_ok = any(h.source_type == q.must_include_src for h in hits)

    # Binary "any hit in top K" recall (since we use substring relevance)
    def _any_rel_in_top(k: int) -> float:
        return 1.0 if any(rels[:k]) else 0.0
    recall_1  = _any_rel_in_top(1)
    recall_3  = _any_rel_in_top(3)
    recall_5  = _any_rel_in_top(5)
    recall_10 = _any_rel_in_top(10)

    # Precision@5 (fraction of top-5 hits that are relevant)
    precision_5 = sum(rels[:5]) / 5.0 if rels else 0.0

    # Reciprocal rank of first relevant
    rr = 0.0
    for i, r in enumerate(rels):
        if r:
            rr = 1.0 / (i + 1)
            break

    ndcg5 = _ndcg_at_k(rels, k=5)

    return EvalResultV2(
        query=q.query,
        category=q.category,
        n_hits=len(hits),
        judgements=judgements,
        src_check_ok=src_ok,
        elapsed_ms=round(elapsed_ms, 1),
        top_score=judgements[0].score if judgements else 0.0,
        comment=q.comment,
        recall_at_1=recall_1,
        recall_at_3=recall_3,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        precision_at_5=round(precision_5, 3),
        rr=round(rr, 4),
        ndcg_at_5=round(ndcg5, 4),
    )


def run_eval(top_k: int = 10) -> dict[str, Any]:
    in_domain  = [q for q in GROUND_TRUTH if q.category != "adversarial"]
    adv        = [q for q in GROUND_TRUTH if q.category == "adversarial"]
    in_results = [evaluate_query(q, top_k=top_k) for q in in_domain]
    adv_results = [evaluate_query(q, top_k=top_k) for q in adv]

    # Aggregate
    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    agg = {
        "n_queries":          len(in_results),
        "recall_at_1":        _mean([r.recall_at_1  for r in in_results]),
        "recall_at_3":        _mean([r.recall_at_3  for r in in_results]),
        "recall_at_5":        _mean([r.recall_at_5  for r in in_results]),
        "recall_at_10":       _mean([r.recall_at_10 for r in in_results]),
        "precision_at_5":     _mean([r.precision_at_5 for r in in_results]),
        "mrr":                _mean([r.rr for r in in_results]),
        "ndcg_at_5":          _mean([r.ndcg_at_5 for r in in_results]),
        "src_constraint_ok":  _mean([1.0 if r.src_check_ok else 0.0 for r in in_results]),
        "latency_ms_p50":     round(statistics.median([r.elapsed_ms for r in in_results]), 1) if in_results else 0,
        "latency_ms_p95":     round(statistics.quantiles([r.elapsed_ms for r in in_results], n=20)[18], 1) if len(in_results) >= 20 else round(max([r.elapsed_ms for r in in_results]), 1),
    }

    # Per-category breakdown (in-domain)
    by_cat: dict[str, list[EvalResultV2]] = {}
    for r in in_results:
        by_cat.setdefault(r.category, []).append(r)
    by_category = {
        cat: {
            "n": len(rs),
            "recall_at_5": _mean([r.recall_at_5 for r in rs]),
            "mrr":         _mean([r.rr         for r in rs]),
            "ndcg_at_5":   _mean([r.ndcg_at_5  for r in rs]),
        } for cat, rs in by_cat.items()
    }

    # Adversarial: top score should be LOW (we want NOT to confidently retrieve)
    adv_summary = {
        "n":                  len(adv_results),
        "mean_top_score":     _mean([r.top_score for r in adv_results]),
        "max_top_score":      max([r.top_score for r in adv_results], default=0.0),
        "ideal_top_score_lt": 0.30,    # if mean top-score > 0.30, retriever is overconfident
    }

    return {
        "aggregate":    agg,
        "by_category":  by_category,
        "adversarial":  adv_summary,
        "per_query":    [r.to_dict() for r in (in_results + adv_results)],
    }


# ── CLI ──────────────────────────────────────────────────────────


def _print_summary(out: dict) -> None:
    agg = out["aggregate"]
    print(f"\n=== Project History RAG · eval v2 ===")
    print(f"  n_queries         = {agg['n_queries']}")
    print(f"  Recall@1          = {agg['recall_at_1']:.3f}")
    print(f"  Recall@3          = {agg['recall_at_3']:.3f}")
    print(f"  Recall@5          = {agg['recall_at_5']:.3f}")
    print(f"  Recall@10         = {agg['recall_at_10']:.3f}")
    print(f"  Precision@5       = {agg['precision_at_5']:.3f}")
    print(f"  MRR               = {agg['mrr']:.3f}")
    print(f"  nDCG@5            = {agg['ndcg_at_5']:.3f}")
    print(f"  src_constraint_ok = {agg['src_constraint_ok']:.3f}")
    print(f"  latency p50 / p95 = {agg['latency_ms_p50']} / {agg['latency_ms_p95']} ms")
    print()
    print("By category:")
    for cat, m in out["by_category"].items():
        print(f"  {cat:<18s} n={m['n']:2d}  R@5={m['recall_at_5']:.2f}  MRR={m['mrr']:.2f}  nDCG@5={m['ndcg_at_5']:.2f}")
    adv = out["adversarial"]
    print()
    print(f"Adversarial (n={adv['n']}): mean top-1 score = {adv['mean_top_score']:.3f}  "
          f"(ideal < {adv['ideal_top_score_lt']:.2f}; overconfident if > 0.5)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="top-K to retrieve")
    ap.add_argument("--json", type=str, default=None, help="path to write full json report")
    args = ap.parse_args()

    out = run_eval(top_k=args.k)
    _print_summary(out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\nFull report written to {args.json}")
    # Exit non-zero if recall@5 < 0.6 (drift detector)
    return 0 if out["aggregate"]["recall_at_5"] >= 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
