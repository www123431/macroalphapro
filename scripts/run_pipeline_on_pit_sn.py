"""scripts/run_pipeline_on_pit_sn.py — exercise full
candidate_pipeline on PIT SN to find loop gaps via diff vs manual.

Per user direction 2026-05-31: "loop 应该能自驱 senior 级工作; 双线
是为找 gap, 不是补 gap". This script:
  1. Runs the 14-step pipeline on PIT SN
  2. Prints each step's verdict + diagnostic
  3. Compares to manual analysis from earlier sessions:
     - alpha t (Phase 3): expected 9.65
     - cosine with parent D_PEAD: expected 0.78
     - honest deploy Sharpe (P-D8): expected 1.38
     - meta-decision: expected PROMOTE_AS_REPLACEMENT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import engine.research.sleeves  # noqa: F401

from engine.research.candidate_pipeline import run_candidate_pipeline


def main() -> int:
    # Load PIT SN returns via the registered sleeve
    from engine.research.sleeve_registry import get_sleeve
    sleeve = get_sleeve("post_earnings_drift_pit_sn")
    pit_sn = sleeve.returns()

    print("=" * 95)
    print(" CANDIDATE_PIPELINE on PIT SN (full 14-step audit + manual diff)")
    print("=" * 95)
    print(f"  Input: {sleeve.strategy_id}")
    print(f"  n_months: {len(pit_sn)}")
    print(f"  date range: {pit_sn.index.min().date()} → {pit_sn.index.max().date()}")
    print(f"  gross Sharpe: {(pit_sn.mean()*12)/(pit_sn.std()*(12**0.5)):.3f}")

    print(f"\n  Running run_candidate_pipeline...")
    report = run_candidate_pipeline(
        candidate_returns=pit_sn,
        proposal_name="post_earnings_drift_pit_sn",
        proposed_role="alpha_seeker",
        mechanism_id="post_earnings_drift",  # link to graveyard / library
        parent_returns_path="data/cache/_dpead_recon_base.parquet",
        phase=3,
    )

    print(f"\n  Pipeline complete. {len(report.step_results)} steps executed.")
    print(f"\n{'='*95}")
    print(f"  STEP-BY-STEP RESULTS")
    print(f"{'='*95}\n")
    for i, step in enumerate(report.step_results, 1):
        d = step.to_dict()
        status_icon = {"PASS": "[OK]", "WARN": "[WARN]", "FAIL": "[FAIL]",
                       "SKIP": "[SKIP]", "INFO": "[INFO]"}.get(d["status"], "[?]")
        print(f"  Step {i:2d} {status_icon:<6} {d.get('step_name','<no name>'):<40}")
        if d.get("verdict"):
            v = d['verdict']
            if len(v) > 100:
                v = v[:97] + "..."
            print(f"            {v}")
        kf = d.get("key_findings", {})
        if kf:
            keys = list(kf.keys())[:6]
            print(f"            key_findings: {keys}")

    print(f"\n{'='*95}")
    print(f"  META-DECISION")
    print(f"{'='*95}")
    meta = report.to_dict()
    print(f"  final_decision:   {meta.get('final_decision', '<missing>')}")
    rat = meta.get('rationale', '<missing>') or '<missing>'
    print(f"  rationale:        {rat[:250] if isinstance(rat, str) else rat}")
    print(f"  role_used:        {meta.get('role_used', '<missing>')}")
    print(f"  role_was_inferred: {meta.get('role_was_inferred', '<missing>')}")
    print(f"  candidate_relation: {meta.get('candidate_relation', '<missing>')}")
    mc_sleeve = meta.get('most_correlated_sleeve')
    mc_val = meta.get('most_correlated_value')
    if mc_sleeve and mc_val is not None:
        print(f"  most_correlated:  {mc_sleeve} (cosine {mc_val:+.3f})")

    print(f"\n{'='*95}")
    print(f"  MANUAL-vs-LOOP DIFF (gaps in loop ability)")
    print(f"{'='*95}")
    expected = {
        "Phase 3 alpha t": 9.65,
        "Cosine w/ parent D_PEAD": 0.78,
        "Honest deploy Sharpe (P-D8)": 1.38,
        "Meta decision": "PROMOTE_AS_REPLACEMENT",
    }
    print(f"  Manual expectations (from session earlier today):")
    for k, v in expected.items():
        print(f"    {k:<35} {v}")

    print(f"\n  Save full report to data/research/pit_sn_pipeline_report.json")
    Path("data/research").mkdir(parents=True, exist_ok=True)
    Path("data/research/pit_sn_pipeline_report.json").write_text(
        json.dumps(meta, default=str, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
