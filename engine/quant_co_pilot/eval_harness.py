"""
engine/quant_co_pilot/eval_harness.py — Tool 1 case study eval harness.

Pre-registration: docs/spec_quant_co_pilot_decision_lineage_v1.md (id=53) §3

Per spec:
  - 10 representative queries locked pre-launch (EASY 3 / MEDIUM 4 / HARD 3)
  - 3 runs per query for stability check (citation overlap rate)
  - Manual gold-standard answers comparison (precision / recall)
  - Output: data/quant_co_pilot/decision_lineage_eval.json + per-query traces

PASS criteria (locked, spec §3.3):
  - Mean citation recall ≥ 0.7
  - Mean citation precision ≥ 0.5
  - Stability rate ≥ 0.8 (8/10 queries with ≥90% citation overlap across 3 runs)
  - Hallucination count ≤ 2 / 30 runs
  - Mean cost ≤ $0.05/query
  - Mean latency ≤ 30s/query

Wave A note: gold standard manual answers must be written before this harness runs
(reserved as case-study evidence per spec §3.4).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import statistics
from pathlib import Path
from typing import Optional

from engine.quant_co_pilot.base import TraceResult, Citation
from engine.quant_co_pilot.decision_lineage import DecisionLineageAgent

logger = logging.getLogger(__name__)

# Locked per spec §3.1 — 10 queries (3+4+3)
EVAL_QUERIES_LOCKED: tuple[tuple[str, str], ...] = (
    # (difficulty, query)
    ("EASY",   "PRODUCTION_SIGNAL 当前是什么?"),
    ("EASY",   "EFFECTIVE_N_TRIALS 现在多少?"),
    ("EASY",   "spec id=50 有几个 amendments?"),

    ("MEDIUM", "为什么 PRODUCTION_SIGNAL=ql01_bab 不是 tsmom?"),
    ("MEDIUM", "factor_ensemble v1 (id=50) verdict 数字是什么? 为什么 DESCRIPTIVE_POSITIVE?"),
    ("MEDIUM", "factor_ensemble v2 robust (id=51) 跟 v1 的差别是什么? v2 数字为什么 substantively FAIL?"),
    ("MEDIUM", "Quality factor 为什么 SPEC_LOCK_DATE 设 2026-05-09?"),

    ("HARD",   "项目 falsification chain 有几条 + 各是 reject 因为什么? + v1/v2/Wave A 在 chain 中位置?"),
    ("HARD",   "factor_ensemble v1 ETF → v2 robust → Wave A single-stock 的 absolute Sharpe trajectory 是什么? 单股扩展贡献了多少 lift? 仍打不过哪个 baseline?"),
    ("HARD",   "Stage 2 Wave A 第一次跑 ensemble Sharpe -0.08 后来变 +0.25,中间 debug 了什么?"),
)

# Spec §3.3 PASS criteria
PASS_CRITERIA_LOCKED: dict = {
    "mean_citation_recall_min":     0.70,
    "mean_citation_precision_min":  0.50,
    "stability_rate_min":           0.80,   # 8/10 queries ≥ 90% citation overlap
    "stability_overlap_threshold":  0.90,
    "hallucination_count_max":      2,      # / 30 runs (10 queries × 3)
    "cost_usd_max":                 0.05,
    "latency_ms_max":               30000,
}

N_RUNS_PER_QUERY: int = 3   # spec §2.5

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "quant_co_pilot"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_EVAL_OUT_PATH = _DATA_DIR / "decision_lineage_eval.json"
_TRACES_DIR = _DATA_DIR / "decision_lineage_traces"
_TRACES_DIR.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class QueryEvalResult:
    """Per-query eval result across N_RUNS_PER_QUERY runs."""
    query:                     str
    difficulty:                str
    runs:                      list[dict]   # each: {final_answer, citations, cost_usd, latency_ms, abort_reason}
    citation_overlap_rate:     float        # fraction of critical citations shared across 3 runs
    is_stable:                 bool         # overlap >= 0.90
    hallucination_count:       int          # # citations marked unverified across 3 runs
    mean_cost_usd:             float
    mean_latency_ms:           float


@dataclasses.dataclass
class EvalReport:
    """Aggregate eval report."""
    eval_timestamp:           str
    n_queries:                int
    n_runs_per_query:         int
    total_cost_usd:           float
    total_runs:               int
    per_query_results:        list[QueryEvalResult]
    mean_citation_recall:     Optional[float]   # vs gold; None if no gold provided
    mean_citation_precision:  Optional[float]
    stability_rate:           float
    total_hallucinations:     int
    mean_cost_usd:            float
    mean_latency_ms:          float
    overall_decision:         str   # PASS / FAIL with reason


def _citation_set(citations: list[Citation]) -> set[tuple[str, str]]:
    """Return set of (pattern, raw_match) for stable comparison."""
    return {(c.pattern, c.raw_match) for c in citations}


def _compute_overlap_rate(citation_sets: list[set]) -> float:
    """Jaccard-style overlap across N citation sets.

    Returns: |intersection of all| / |union of all|. 1.0 = perfect agreement.
    """
    if not citation_sets:
        return 0.0
    union = set().union(*citation_sets)
    if not union:
        return 1.0  # all empty = trivially identical
    intersection = set(citation_sets[0])
    for s in citation_sets[1:]:
        intersection = intersection & s
    return len(intersection) / len(union)


def run_eval(
    queries:           Optional[list[tuple[str, str]]] = None,
    n_runs_per_query:  int = N_RUNS_PER_QUERY,
    gold_answers:      Optional[dict[str, dict]] = None,
    persist:           bool = True,
) -> EvalReport:
    """Run case study eval per spec §3.

    Args:
        queries:          override locked set (default: EVAL_QUERIES_LOCKED)
        n_runs_per_query: stability check runs (default 3)
        gold_answers:     {query: {citations: [...], canonical_facts: [...]}}; if None, citation_recall/precision = None
        persist:          write to data/quant_co_pilot/decision_lineage_eval.json
    """
    queries_to_run = queries or list(EVAL_QUERIES_LOCKED)
    agent = DecisionLineageAgent()
    per_query_results: list[QueryEvalResult] = []
    total_cost = 0.0

    for q_idx, (difficulty, query) in enumerate(queries_to_run):
        logger.info("eval [%d/%d] (%s): %s", q_idx + 1, len(queries_to_run), difficulty, query[:80])
        run_results = []
        citation_sets = []
        hallucination_count = 0
        for run_idx in range(n_runs_per_query):
            try:
                trace: TraceResult = agent.answer(query)
            except Exception as exc:
                logger.error("query %d run %d crashed: %s", q_idx + 1, run_idx + 1, exc)
                run_results.append({
                    "run_idx":      run_idx,
                    "final_answer": "",
                    "citations":    [],
                    "cost_usd":     0.0,
                    "latency_ms":   0,
                    "abort_reason": f"crashed: {exc!s}",
                })
                citation_sets.append(set())
                continue

            run_results.append({
                "run_idx":           run_idx,
                "final_answer":      trace.final_answer,
                "annotated_answer":  trace.annotated_answer,
                "citations":         [
                    {"pattern": c.pattern, "raw_match": c.raw_match,
                     "verified": c.verified, "verify_msg": c.verify_msg}
                    for c in trace.citations
                ],
                "n_steps":           len(trace.steps),
                "cost_usd":          trace.cost_usd,
                "latency_ms":        trace.latency_ms,
                "abort_reason":      trace.abort_reason,
            })
            citation_sets.append(_citation_set(trace.citations))
            hallucination_count += sum(1 for c in trace.citations if not c.verified)
            total_cost += trace.cost_usd

            # Persist individual trace for audit
            if persist:
                trace_path = _TRACES_DIR / f"q{q_idx + 1:02d}_r{run_idx + 1}.json"
                try:
                    trace_path.write_text(
                        json.dumps(run_results[-1], ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.warning("trace persist failed: %s", exc)

        overlap = _compute_overlap_rate(citation_sets)
        is_stable = overlap >= PASS_CRITERIA_LOCKED["stability_overlap_threshold"]
        mean_cost = statistics.mean(r["cost_usd"] for r in run_results) if run_results else 0.0
        mean_lat = statistics.mean(r["latency_ms"] for r in run_results) if run_results else 0.0

        per_query_results.append(QueryEvalResult(
            query=query,
            difficulty=difficulty,
            runs=run_results,
            citation_overlap_rate=overlap,
            is_stable=is_stable,
            hallucination_count=hallucination_count,
            mean_cost_usd=mean_cost,
            mean_latency_ms=mean_lat,
        ))

    # Aggregate
    n_total_runs = len(queries_to_run) * n_runs_per_query
    total_hallucinations = sum(q.hallucination_count for q in per_query_results)
    n_stable = sum(1 for q in per_query_results if q.is_stable)
    stability_rate = n_stable / len(per_query_results) if per_query_results else 0.0
    mean_cost = total_cost / n_total_runs if n_total_runs > 0 else 0.0
    all_latencies = [r["latency_ms"] for q in per_query_results for r in q.runs]
    mean_lat = statistics.mean(all_latencies) if all_latencies else 0.0

    # Gold-standard precision/recall (if provided)
    mean_recall = mean_precision = None
    if gold_answers:
        recalls, precisions = [], []
        for q in per_query_results:
            gold = gold_answers.get(q.query, {})
            gold_cites = set(tuple(c) for c in gold.get("citations", []))
            if not gold_cites:
                continue
            for run in q.runs:
                run_cites = {(c["pattern"], c["raw_match"]) for c in run["citations"]}
                if gold_cites:
                    recalls.append(len(run_cites & gold_cites) / len(gold_cites))
                if run_cites:
                    precisions.append(len(run_cites & gold_cites) / len(run_cites))
        mean_recall = statistics.mean(recalls) if recalls else None
        mean_precision = statistics.mean(precisions) if precisions else None

    # Decision
    fail_reasons = []
    if mean_recall is not None and mean_recall < PASS_CRITERIA_LOCKED["mean_citation_recall_min"]:
        fail_reasons.append(f"recall {mean_recall:.2f} < {PASS_CRITERIA_LOCKED['mean_citation_recall_min']}")
    if mean_precision is not None and mean_precision < PASS_CRITERIA_LOCKED["mean_citation_precision_min"]:
        fail_reasons.append(f"precision {mean_precision:.2f} < {PASS_CRITERIA_LOCKED['mean_citation_precision_min']}")
    if stability_rate < PASS_CRITERIA_LOCKED["stability_rate_min"]:
        fail_reasons.append(f"stability {stability_rate:.2f} < {PASS_CRITERIA_LOCKED['stability_rate_min']}")
    if total_hallucinations > PASS_CRITERIA_LOCKED["hallucination_count_max"]:
        fail_reasons.append(f"hallucinations {total_hallucinations} > {PASS_CRITERIA_LOCKED['hallucination_count_max']}")
    if mean_cost > PASS_CRITERIA_LOCKED["cost_usd_max"]:
        fail_reasons.append(f"cost ${mean_cost:.4f} > ${PASS_CRITERIA_LOCKED['cost_usd_max']}")
    if mean_lat > PASS_CRITERIA_LOCKED["latency_ms_max"]:
        fail_reasons.append(f"latency {mean_lat:.0f}ms > {PASS_CRITERIA_LOCKED['latency_ms_max']}ms")

    overall = "PASS" if not fail_reasons else "FAIL: " + "; ".join(fail_reasons)
    if mean_recall is None and mean_precision is None and not fail_reasons:
        overall = "PASS_NO_GOLD (gold answers not provided; precision/recall skipped)"

    report = EvalReport(
        eval_timestamp=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        n_queries=len(queries_to_run),
        n_runs_per_query=n_runs_per_query,
        total_cost_usd=total_cost,
        total_runs=n_total_runs,
        per_query_results=per_query_results,
        mean_citation_recall=mean_recall,
        mean_citation_precision=mean_precision,
        stability_rate=stability_rate,
        total_hallucinations=total_hallucinations,
        mean_cost_usd=mean_cost,
        mean_latency_ms=mean_lat,
        overall_decision=overall,
    )

    if persist:
        try:
            payload = {
                "eval_timestamp":          report.eval_timestamp,
                "n_queries":               report.n_queries,
                "n_runs_per_query":        report.n_runs_per_query,
                "total_runs":              report.total_runs,
                "total_cost_usd":          round(report.total_cost_usd, 4),
                "mean_cost_usd":           round(report.mean_cost_usd, 4),
                "mean_latency_ms":         round(report.mean_latency_ms),
                "stability_rate":          round(report.stability_rate, 3),
                "total_hallucinations":    report.total_hallucinations,
                "mean_citation_recall":    round(report.mean_citation_recall, 3) if report.mean_citation_recall is not None else None,
                "mean_citation_precision": round(report.mean_citation_precision, 3) if report.mean_citation_precision is not None else None,
                "overall_decision":        report.overall_decision,
                "pass_criteria_locked":    PASS_CRITERIA_LOCKED,
                "per_query_summary": [
                    {
                        "difficulty":            q.difficulty,
                        "query":                 q.query,
                        "citation_overlap_rate": round(q.citation_overlap_rate, 3),
                        "is_stable":             q.is_stable,
                        "hallucination_count":   q.hallucination_count,
                        "mean_cost_usd":         round(q.mean_cost_usd, 4),
                        "mean_latency_ms":       round(q.mean_latency_ms),
                    }
                    for q in report.per_query_results
                ],
            }
            _EVAL_OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("eval report persisted to %s", _EVAL_OUT_PATH)
        except Exception as exc:
            logger.warning("eval persist failed: %s", exc)

    return report
