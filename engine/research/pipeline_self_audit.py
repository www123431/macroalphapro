"""engine/research/pipeline_self_audit.py — Phase 1 P1b of loop robustness.

The candidate_pipeline itself could silently break/drift over time:
  - Step impl changes without test updating
  - Library YAML schema shift breaks downstream
  - Cached data file gets corrupted
  - LLM Devil's Advocate returns unexpected verdict

This module runs the pipeline on KNOWN-OUTCOME sleeves and asserts the
verdict matches baseline. Run periodically (weekly) as cron — if
output diverges from baseline, ALERT.

KNOWN BASELINES (locked at commit 3ed9a05 / dogfood 2026-05-31):
  mom_hedge as insurance role        → HARD_REJECT (insurance hyp fails regime)
  PIT sector-neutral D_PEAD as       → PROMOTE_AS_REPLACEMENT
    alpha_seeker
  crisis_hedge as diversifier        → BORDERLINE_REVIEW or PROMOTE_AS_REPLACEMENT

OUTPUT:
  data/research/pipeline_self_audit.jsonl     each run's results

Per [[feedback-loop-is-robustness-doctrine-2026-05-31]] Phase 1 P1b.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = REPO_ROOT / "data" / "research" / "pipeline_self_audit.jsonl"

# Known sleeves + expected verdicts (locked at 2026-05-31 dogfood).
KNOWN_BASELINES = [
    {
        "returns_cache":    "data/cache/_mom_hedge_monthly.parquet",
        "proposal_name":    "mom_hedge_smoke",
        "proposed_role":    "insurance",
        "expected_decision_in": {"HARD_REJECT"},
        "expected_relation_in": {"UNKNOWN", "ADDITION", "REPLACEMENT"},
        "expected_steps_must_pass": {"H10_evaluate_candidate", "data_quality"},
        "expected_steps_must_fail": {"regime_stratified_BARRA"},
        "note": "Insurance hypothesis fails: STRESS α < NORMAL α (mom_hedge "
                "failure mode from 2026-05-31)",
    },
    {
        "returns_cache":    "data/cache/_dpead_sn_pit_monthly.parquet",
        "proposal_name":    "dpead_sn_pit_smoke",
        "proposed_role":    "alpha_seeker",
        "expected_decision_in": {"PROMOTE_AS_REPLACEMENT", "BORDERLINE_REVIEW"},
        "expected_relation_in": {"REPLACEMENT", "UNKNOWN"},
        "expected_steps_must_pass": {"H10_evaluate_candidate", "data_quality",
                                       "regime_stratified_BARRA"},
        "note": "PIT sector-neutral D_PEAD: high correlation with equity "
                "sleeve (corr ~+0.71), should be classified REPLACEMENT.",
    },
    {
        "returns_cache":    "data/cache/_crisis_hedge_monthly.parquet",
        "proposal_name":    "crisis_hedge_smoke",
        "proposed_role":    "diversifier",
        "expected_decision_in": {"BORDERLINE_REVIEW", "PROMOTE_AS_REPLACEMENT",
                                  "SOFT_REJECT"},
        "expected_relation_in": {"UNKNOWN", "ADDITION", "REPLACEMENT"},
        "expected_steps_must_pass": {"H10_evaluate_candidate", "data_quality"},
        "note": "TLT/GLD crisis hedge as diversifier role.",
    },
]


def _check_baseline(baseline: dict, phase: int = 3) -> dict:
    """Run pipeline on a baseline sleeve, check verdict matches expectation."""
    from engine.research.candidate_pipeline import run_candidate_pipeline

    cache_path = REPO_ROOT / baseline["returns_cache"]
    if not cache_path.exists():
        return {
            "baseline":       baseline["proposal_name"],
            "status":         "SKIP",
            "reason":         f"cache missing: {cache_path}",
            "actual_decision": None,
        }
    try:
        df = pd.read_parquet(cache_path)
        col = df.columns[0]
        returns = df[col]
        returns.index = pd.to_datetime(returns.index)
    except Exception as exc:
        return {
            "baseline":       baseline["proposal_name"],
            "status":         "ERROR",
            "reason":         f"cache load failed: {exc}",
            "actual_decision": None,
        }

    try:
        report = run_candidate_pipeline(
            returns,
            proposal_name=baseline["proposal_name"],
            proposed_role=baseline["proposed_role"],
            phase=phase,
        )
    except Exception as exc:
        return {
            "baseline":       baseline["proposal_name"],
            "status":         "ERROR",
            "reason":         f"pipeline failed: {exc}",
            "actual_decision": None,
        }

    failures = []
    if report.final_decision not in baseline["expected_decision_in"]:
        failures.append(
            f"final_decision {report.final_decision!r} not in "
            f"expected {baseline['expected_decision_in']}"
        )
    if report.candidate_relation not in baseline["expected_relation_in"]:
        failures.append(
            f"candidate_relation {report.candidate_relation!r} not in "
            f"expected {baseline['expected_relation_in']}"
        )
    # Step status checks
    step_status_map = {s.step_name: s.status for s in report.step_results}
    for step_name in baseline.get("expected_steps_must_pass", []):
        actual = step_status_map.get(step_name)
        if actual != "PASS":
            failures.append(
                f"step {step_name!r} expected PASS, got {actual!r}"
            )
    for step_name in baseline.get("expected_steps_must_fail", []):
        actual = step_status_map.get(step_name)
        if actual != "FAIL":
            failures.append(
                f"step {step_name!r} expected FAIL, got {actual!r}"
            )

    return {
        "baseline":          baseline["proposal_name"],
        "status":            "PASS" if not failures else "FAIL",
        "actual_decision":   report.final_decision,
        "actual_relation":   report.candidate_relation,
        "step_statuses":     step_status_map,
        "failures":          failures,
        "expected":          {
            "decision_in":  list(baseline["expected_decision_in"]),
            "relation_in":  list(baseline["expected_relation_in"]),
        },
    }


def run_self_audit(phase: int = 3) -> dict:
    """Run pipeline self-audit on all baselines."""
    audit_date = datetime.date.today().isoformat()
    results = []
    for baseline in KNOWN_BASELINES:
        res = _check_baseline(baseline, phase=phase)
        res["audit_date"] = audit_date
        results.append(res)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_error = sum(1 for r in results if r["status"] == "ERROR")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")

    summary = {
        "audit_date":   audit_date,
        "n_total":      len(results),
        "n_pass":       n_pass,
        "n_fail":       n_fail,
        "n_error":      n_error,
        "n_skip":       n_skip,
        "all_pass":     n_fail == 0 and n_error == 0,
        "results":      results,
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    summary = run_self_audit(phase=args.phase)

    print(f"[pipeline_self_audit] {summary['audit_date']} "
          f"phase={args.phase} {summary['n_total']} baselines")
    print(f"  PASS:  {summary['n_pass']}")
    print(f"  FAIL:  {summary['n_fail']}")
    print(f"  ERROR: {summary['n_error']}")
    print(f"  SKIP:  {summary['n_skip']}")
    print()
    for r in summary["results"]:
        flag = "✓" if r["status"] == "PASS" else "✗" if r["status"] == "FAIL" else "?"
        flag = flag.encode("ascii", "replace").decode("ascii")
        print(f"  [{r['status']:<5}] {r['baseline']:<30}  "
              f"actual_decision={r.get('actual_decision', 'n/a')}")
        if r["status"] == "FAIL":
            for fail in r.get("failures", []):
                print(f"      - {fail}")
        elif r["status"] in ("ERROR", "SKIP"):
            print(f"      - {r.get('reason', '')}")
    print()
    if not summary["all_pass"]:
        print(f"  PIPELINE DRIFT DETECTED — review failures.")
        print(f"  History: {HISTORY_PATH.relative_to(REPO_ROOT)}")
        return 1
    print(f"  All baselines passing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
