"""
scripts/run_auto_audit_proposals.py — Cron entry for Layer 1 LLM proposer (R-1.C, 2026-05-06).

Picks up OPEN audit findings without a proposal, generates LLM remediation
draft for each (capped at engine.config.R_PROPOSAL_CAP_PER_RUN), persists
to AuditProposal table.

Recommended cadence:
  • Daily, ~30 minutes after `run_auto_audit.py --scope critical` completes.
  • This decouples LLM API latency / failure from the rule sweep.

Cap + budget:
  • R_PROPOSAL_CAP_PER_RUN bounds per-run burst (35 by default 2026-05-06).
  • R_COST_BUDGET_USD bounds annual cumulative spend ($50 by default).
  • Excess findings receive a placeholder proposal row with
    generation_status='deferred_quota'; next run picks them up.

Usage:
  python scripts/run_auto_audit_proposals.py
  python scripts/run_auto_audit_proposals.py --cap 5         # smoke-test
  python scripts/run_auto_audit_proposals.py --finding-id 7  # single finding
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cap", type=int, default=None,
                        help="Override R_PROPOSAL_CAP_PER_RUN for this invocation")
    parser.add_argument("--finding-id", type=int, default=None,
                        help="Process a single finding (skip OPEN-loop selection)")
    args = parser.parse_args()

    from engine.auto_audit_proposer import (
        generate_proposal, generate_proposals_for_open_findings, get_cost_status,
    )

    if args.finding_id is not None:
        result = generate_proposal(args.finding_id)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("generation_status") == "success" else 1

    summary = generate_proposals_for_open_findings(cap=args.cap)
    cost = get_cost_status()
    summary["budget_total_usd"]    = cost["budget_usd"]
    summary["budget_used_usd"]     = cost["total_usd"]
    summary["budget_fraction"]     = round(cost["fraction"], 4)
    summary["lifetime_calls"]      = cost["calls"]

    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
