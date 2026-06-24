"""scripts/test_pipeline_v2_parity.py — verify v2 LangGraph pipeline
produces same final_decision as v1 on PIT SN + LTR.

Phase A.1 acceptance test: if v2.final_decision == v1.final_decision,
the LangGraph refactor is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import engine.research.sleeves  # noqa: F401

from engine.research.candidate_pipeline import run_candidate_pipeline as run_v1
from engine.research.candidate_pipeline_v2 import run_candidate_pipeline_v2 as run_v2


def test_case(label, series_path, mechanism_id=None, proposal_dict=None):
    print(f"\n{'=' * 90}")
    print(f" PARITY TEST: {label}")
    print(f"{'=' * 90}")
    s = pd.read_parquet(series_path).iloc[:, 0]
    s.index = pd.to_datetime(s.index)

    print(f"\n  Running v1...")
    r1 = run_v1(
        candidate_returns=s, proposal_name=label,
        proposed_role="alpha_seeker", mechanism_id=mechanism_id,
        proposal_dict=proposal_dict, phase=3,
    )
    print(f"  v1 final_decision:       {r1.final_decision}")
    print(f"  v1 n_step_results:       {len(r1.step_results)}")
    print(f"  v1 candidate_relation:   {r1.candidate_relation}")

    print(f"\n  Running v2 (LangGraph)...")
    r2 = run_v2(
        candidate_returns=s, proposal_name=label,
        proposed_role="alpha_seeker", mechanism_id=mechanism_id,
        proposal_dict=proposal_dict, phase=3,
    )
    print(f"  v2 final_decision:       {r2.final_decision}")
    print(f"  v2 n_step_results:       {len(r2.step_results)}")
    print(f"  v2 candidate_relation:   {r2.candidate_relation}")

    match = r1.final_decision == r2.final_decision
    print(f"\n  PARITY: {'PASS' if match else 'FAIL'} "
          f"({'verdicts match' if match else 'verdicts differ!'})")
    if not match:
        print(f"    v1 rationale: {r1.rationale[:200]}")
        print(f"    v2 rationale: {r2.rationale[:200]}")
    return match


def main() -> int:
    results = []

    # Test 1: PIT SN (real PIT SN deploy candidate)
    results.append(test_case(
        label="post_earnings_drift_pit_sn",
        series_path="data/cache/_dpead_sn_pit_monthly.parquet",
        mechanism_id="post_earnings_drift",
        proposal_dict={
            "family":         "earnings_underreaction",
            "parent_family":  "equity_factor",
            "required_data":  ["SUE_panel", "quarterly_eps", "ret_60d"],
            "economics_text": "PIT FF12 within-sector D_PEAD.",
        },
    ))

    # Test 2: LTR (the one we just validated through v1 pipeline)
    results.append(test_case(
        label="LTR_long_history",
        series_path="data/cache/_ltr_monthly_long.parquet",
        mechanism_id="long_term_reversal",
        proposal_dict={
            "family":         "long_term_reversal",
            "parent_family":  "equity_factor",
            "required_data":  ["crsp_msf", "monthly_returns"],
            "economics_text": "De Bondt-Thaler 1985 long-term reversal.",
            "post_pub_decay": {
                "post_2020_replications": [
                    {"paper_id": "hou_xue_zhang_2020_rfs",
                     "delta_range_estimate": [-0.20, 0.00]},
                ],
            },
        },
    ))

    print(f"\n{'=' * 90}")
    print(f" SUMMARY: {sum(results)}/{len(results)} parity tests PASS")
    print(f"{'=' * 90}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
