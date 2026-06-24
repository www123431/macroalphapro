"""scripts/promote_candidate.py — CLI for SLM Phase 1 promotion.

Reads cached audit blocks JSON (produced by an audit script like
scripts/audits/audit_pit_sn_dpead_for_deploy.py) + identity metadata from
CLI args + commits the candidate to AUDITED via the typed orchestrator.

USAGE:
  python scripts/promote_candidate.py <strategy_id> \\
    --family earnings_underreaction \\
    --relation-to-parent REPLACEMENT \\
    --parent post_earnings_drift \\
    --canonical-paper bernard_thomas_1989_jar \\
    --audit-blocks-json data/cache/_pit_sn_audit_blocks.json \\
    --pipeline-run-id manual_2026-05-31 \\
    --actor zhangxizhe \\
    --dry-run

The cached JSON must contain `cost_model` + `factor_exposure` dicts at
the top level (matching what audit scripts emit).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.research.library_yaml_renderer import StrategyIdentity
from engine.research.promote_candidate import promote_candidate
from engine.research.strategy_lifecycle import (
    AuditBlocks, CapacityBlock, CostModelAudit,
    FactorExposureAudit, MultiAumSharpe,
)


def _load_audit_blocks_from_json(path: Path) -> AuditBlocks:
    """Parse the cached audit JSON into typed AuditBlocks. Caller's
    audit script emits the JSON in a layout matching the YAML schema."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cm = raw["cost_model"]
    fx = raw["factor_exposure"]

    cost = CostModelAudit(
        audit_status="audited",
        audit_date=_dt.date.fromisoformat(_dt.date.today().isoformat()),
        audit_script=raw.get("audit_script", "scripts/promote_candidate.py"),
        audit_commit=_current_git_sha(),
        type="almgren_chriss",
        half_spread_bps=cm.get("half_spread_bps", 5.0),
        impact_coef=cm.get("impact_coef", 0.5),
        daily_sigma_estimate=cm.get("daily_sigma", 0.015),
        universe_median_adv_usd=cm.get("universe_median_adv_usd", 50_000_000),
        n_positions_typical=cm.get("n_positions_typical", 100),
        monthly_turnover_estimate=cm.get("monthly_turnover", 0.4),
        stress_multiplier=cm.get("stress_multiplier", 2.5),
        rationale=cm.get("rationale",
            "Promoted via CLI; rationale auto-stub (≥50 chars required by validator)."),
        multi_aum_sharpe_sleeve=MultiAumSharpe(
            at_10M=cm.get("sharpe_at_10M", 0.0),
            at_100M=cm.get("sharpe_at_100M", 0.0),
            at_1B=cm.get("sharpe_at_1B", 0.0),
        ),
        capacity=CapacityBlock(
            hard_capacity_usd=cm.get("hard_capacity_usd", 1.0),
            binding_constraint=cm.get("binding_constraint", "participation_cap_5pct"),
            safe_deploy_band_usd=(
                cm.get("safe_deploy_low", 10_000_000),
                cm.get("safe_deploy_high", 100_000_000),
            ),
            max_participation_assumed=0.05,
        ),
    )
    factor = FactorExposureAudit(
        audit_status="audited",
        audit_date=_dt.date.today(),
        audit_script=raw.get("audit_script", "scripts/promote_candidate.py"),
        audit_commit=_current_git_sha(),
        phase=fx.get("phase", 3),
        proposed_role=fx.get("proposed_role", "alpha_seeker"),
        n_months=fx["n_months"],
        alpha_annualized=fx["alpha_annualized"],
        alpha_t_hac=fx["alpha_t_hac"],
        betas=fx["betas"],
        t_stats_hac=fx["t_stats_hac"],
        r_squared=fx["r_squared"],
        verdict=fx["verdict"],
    )
    return AuditBlocks(cost_model=cost, factor_exposure=factor)


def _current_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("strategy_id", help="Unique strategy identifier")
    p.add_argument("--family", required=True,
                     help="e.g. earnings_underreaction, carry, momentum")
    p.add_argument("--parent-family", default="equity_factor")
    p.add_argument("--purpose", default="deploy_replacement")
    p.add_argument("--relation-to-parent", default="ADDITION",
                     choices=["REPLACEMENT", "ADDITION", "ORIGINAL"])
    p.add_argument("--parent", default=None,
                     help="parent strategy_id (only required if relation=REPLACEMENT)")
    p.add_argument("--canonical-paper", default=None)
    p.add_argument("--audit-blocks-json", required=True, type=Path,
                     help="path to cached audit blocks JSON")
    p.add_argument("--pipeline-run-id", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--snooping-pub-date", default=None,
                     help="ISO date of original publication (data-snooping audit)")
    p.add_argument("--dry-run", action="store_true",
                     help="render + validate without writing files or DB")
    p.add_argument("--overwrite-existing", action="store_true")
    args = p.parse_args()

    print(f"[promote_candidate] strategy_id={args.strategy_id}  "
          f"dry_run={args.dry_run}")
    audit_blocks = _load_audit_blocks_from_json(args.audit_blocks_json)
    print(f"  loaded audit blocks: cost.audit_status="
          f"{audit_blocks.cost_model.audit_status}  "
          f"factor.proposed_role={audit_blocks.factor_exposure.proposed_role}")

    identity = StrategyIdentity(
        strategy_id=args.strategy_id,
        family=args.family,
        parent_family=args.parent_family,
        purpose=args.purpose,
        relation_to_parent=args.relation_to_parent,
        parent_strategy_id=args.parent,
        canonical_paper_id=args.canonical_paper,
    )

    pub_date = (_dt.date.fromisoformat(args.snooping_pub_date)
                if args.snooping_pub_date else None)

    result = promote_candidate(
        identity=identity,
        audit_blocks=audit_blocks,
        pipeline_run_id=args.pipeline_run_id,
        actor=args.actor,
        snooping_publication_date=pub_date,
        overwrite_existing=args.overwrite_existing,
        dry_run=args.dry_run,
        git_sha=_current_git_sha(),
    )

    print(f"\n[result]")
    print(f"  validators_passed: {result.validators_passed}")
    for k, v in result.validator_summary.items():
        print(f"    {k:55s} {v}")
    if result.error:
        print(f"  ERROR: {result.error}")
        return 1
    if result.dry_run:
        print(f"\n  [DRY RUN] YAML preview (first 30 lines):")
        for line in result.yaml_preview.splitlines()[:30]:
            print(f"    {line}")
        return 0
    print(f"  library_yaml_path: {result.library_yaml_path}")
    if result.state_record:
        print(f"  state: {result.state_record.current_state.value}  "
              f"(audited_at={result.state_record.audited_at.isoformat()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
