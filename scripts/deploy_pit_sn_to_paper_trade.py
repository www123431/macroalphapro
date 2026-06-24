"""scripts/deploy_pit_sn_to_paper_trade.py — REAL deploy of PIT SN
D_PEAD via the Strategy Lifecycle Manager.

This is the first production use of SLM: it walks PIT SN through the
3-gate flow (PROMOTE / WIRE / ALLOCATE) using the actual production
state store, not a test DB. After this script runs:

  - data/strategy_lifecycle.db contains a real row for PIT SN
  - State is PAPER_TRADE with paper_trade_started=now
  - paper_trade_monitor will tick on it monthly going forward
  - 24-month OBF boundary clock is running

The PIT SN library YAML already exists (created manually 2026-05-31
when designing the deploy artifact); we do NOT re-render it. We just
register the sleeve in the state store and transition it through gates.

REQUIRES:
  - data/research/mechanism_library/post_earnings_drift_pit_sn.yaml
  - engine.research.sleeves.post_earnings_drift_pit_sn registered

USAGE:
  python scripts/deploy_pit_sn_to_paper_trade.py            # live
  python scripts/deploy_pit_sn_to_paper_trade.py --dry-run  # no DB writes
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine.research.sleeves  # noqa: F401  (triggers registration)

from engine.research.sleeve_registry import get_sleeve
from engine.research.strategy_lifecycle import (
    GateNotMetError, InvalidTransitionError, StrategyState,
)
from engine.research.strategy_state_store import (
    create_strategy, get_strategy, get_transition_history, transition,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                     help="print plan without DB writes")
    p.add_argument("--actor", default="zhangxizhe",
                     help="signoff actor for APPROVED transition")
    p.add_argument("--pipeline-run-id",
                     default="manual_2026-05-31_pit_sn_audit",
                     help="reference to the candidate_pipeline run")
    args = p.parse_args()

    sid = "post_earnings_drift_pit_sn"
    git_sha = _git_sha()

    print("=" * 90)
    print(f" REAL DEPLOY: {sid} → PAPER_TRADE via SLM (live state store)")
    print("=" * 90)
    print(f"  actor: {args.actor}")
    print(f"  pipeline_run_id: {args.pipeline_run_id}")
    print(f"  git_sha: {git_sha}")
    print(f"  dry_run: {args.dry_run}")

    # Resolve sleeve
    sleeve = get_sleeve(sid)
    print(f"\n  Resolved sleeve: {sleeve.strategy_id}")
    print(f"  Library YAML:    {sleeve.library_yaml_path.name}")
    audit = sleeve.audit_blocks()
    print(f"  Role:            {audit.factor_exposure.proposed_role}")
    print(f"  Alpha t (Phase 3): {audit.factor_exposure.alpha_t_hac:+.2f}")
    print(f"  Sharpe @ $10M:   {audit.cost_model.multi_aum_sharpe_sleeve.at_10M:.3f}")

    # Check whether already deployed (idempotency)
    try:
        existing = get_strategy(sid)
        print(f"\n  [WARNING] STRATEGY ALREADY IN STATE STORE")
        print(f"  Current state: {existing.current_state.value}")
        print(f"  Created at:    {existing.created_at.isoformat()}")
        print(f"  Aborting to prevent duplicate transitions. If you want to")
        print(f"  re-deploy, delete the existing row first (manual SQL).")
        return 1
    except KeyError:
        pass  # fresh deploy — continue

    if args.dry_run:
        print(f"\n  [DRY RUN] Would walk: PROPOSED → AUDITED → APPROVED → PAPER_TRADE")
        print(f"            Would mark paper_trade_started = now")
        print(f"            Would emit 4 transition rows in state_transitions")
        return 0

    # ── LIVE DEPLOY ────────────────────────────────────────────────────
    print(f"\n  [LIVE] Creating PROPOSED row...")
    rec = create_strategy(
        strategy_id=sid,
        library_yaml_path=str(sleeve.library_yaml_path),
        candidate_pipeline_run_id=args.pipeline_run_id,
        notes=(
            "PIT FF12 within-sector D_PEAD. Real deploy 2026-05-31. "
            "Audit blocks: cost_model + factor_exposure (Phase 3) audited; "
            "alpha t=9.65, honest deploy Sharpe 1.38 per P-D8. "
            "Paper trade clock starts now; 24mo OBF + Bayesian + DeflSR via "
            "engine.research.paper_trade_monitor."
        ),
        actor="slm_deploy_script",
    )
    print(f"    state={rec.current_state.value}  proposed_at={rec.proposed_at.date()}")

    print(f"\n  [LIVE] Transitioning PROPOSED → AUDITED (pipeline evidence)...")
    rec = transition(
        strategy_id=sid,
        to_state=StrategyState.AUDITED,
        actor="candidate_pipeline",
        reason=("14-step pipeline PROMOTE_AS_REPLACEMENT (cosine 0.78 with parent "
                "post_earnings_drift). DA approval via DeepSeek V4 Pro. "
                "P-D8 honest deploy 1.38."),
        has_candidate_pipeline_run=True,
        git_sha=git_sha,
    )
    print(f"    state={rec.current_state.value}  audited_at={rec.audited_at.date()}")

    print(f"\n  [LIVE] Transitioning AUDITED → APPROVED (human signoff by {args.actor})...")
    rec = transition(
        strategy_id=sid,
        to_state=StrategyState.APPROVED,
        actor=args.actor,
        reason=("Manual approval after reviewing audit blocks + DA verdict. "
                "Honest deploy Sharpe 1.38 > 1.0 floor; alpha t 9.65 strongly "
                "above HLZ 3.0; family-aware DeflSR 0.86 (n_trials=7). "
                "PROMOTE gate cleared."),
        has_human_approval=True,
        git_sha=git_sha,
    )
    print(f"    approved_by={rec.approved_by}  approved_at={rec.approved_at.date()}")

    print(f"\n  [LIVE] Transitioning APPROVED → PAPER_TRADE (WIRE gate)...")
    rec = transition(
        strategy_id=sid,
        to_state=StrategyState.PAPER_TRADE,
        actor="slm_deploy_script",
        reason=("WIRE gate: sleeve registered in engine.research.sleeves; "
                "paper trade clock starts; no real capital. 24-month observation "
                "window + 3-layer monthly tick via paper_trade_monitor."),
        git_sha=git_sha,
    )
    print(f"    state={rec.current_state.value}  paper_trade_started={rec.paper_trade_started.date()}")

    # Print audit trail
    print(f"\n  Transition history:")
    history = get_transition_history(sid)
    for h in history:
        from_s = h.from_state.value if h.from_state else "NULL"
        print(f"    {h.transition_at.isoformat()}  "
              f"{from_s:>12} → {h.to_state.value:<14} "
              f"by {h.actor} (git={h.git_sha or '-'})")

    print(f"\n  [DONE] PIT SN now in PAPER_TRADE. Paper trade clock is RUNNING.")
    print(f"         Run scripts/slm_paper_trade_tick.py monthly to evaluate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
