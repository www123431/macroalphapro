"""scripts/deploy_put_spread_decommission_mom_hedge.py — coordinated
SLM transition: put_spread → PAPER_TRADE + mom_hedge → RAMP_DOWN.

Per Path C verdict 2026-05-31 (commit 38ad27d):
  - put_spread sleeve replaces mom_hedge_overlay
  - mom_hedge had cumulative -92% return over 13yr, broken hedge
  - put_spread has -0.04%/yr drag + +7.24 crisis-yield ratio

ATOMIC actions (in order):
  1. put_spread: PROPOSED → AUDITED (pipeline run evidence)
  2. put_spread: AUDITED → APPROVED (human signoff)
  3. put_spread: APPROVED → PAPER_TRADE (WIRE gate; 24mo clock starts)
  4. mom_hedge_overlay: BACKFILL as LIVE state (was never in SLM before)
  5. mom_hedge_overlay: LIVE → RAMP_DOWN (manual decommission, human approval)

Both transitions write to Merkle ledger so the decision trail is
cryptographically auditable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine.research.sleeves  # noqa: F401

from engine.research.sleeve_registry import get_sleeve
from engine.research.strategy_lifecycle import StrategyState
from engine.research.strategy_state_store import (
    create_strategy, get_strategy, get_transition_history, transition,
)

PUT_SPREAD_ID = "tail_hedge_put_spread"
MOM_HEDGE_ID = "mom_hedge_overlay"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    git_sha = _git_sha()
    actor = "zhangxizhe"

    print("=" * 90)
    print(" SLM transition: put_spread → PAPER_TRADE + mom_hedge → RAMP_DOWN")
    print("=" * 90)
    print(f"  actor: {actor}  git: {git_sha}")

    # ─── Resolve put_spread sleeve (verify registration) ────────────
    sleeve = get_sleeve(PUT_SPREAD_ID)
    audit = sleeve.audit_blocks()
    print(f"\n  [verify put_spread]")
    print(f"    strategy_id: {sleeve.strategy_id}")
    print(f"    yaml:        {sleeve.library_yaml_path.name}")
    print(f"    role:        {audit.factor_exposure.proposed_role}")
    print(f"    crisis-yield ratio (from analysis): +7.24")

    # ─── Phase 1: put_spread PROPOSED → PAPER_TRADE ─────────────────
    try:
        existing = get_strategy(PUT_SPREAD_ID)
        print(f"\n  [SKIP] put_spread already in state store as {existing.current_state.value}")
    except KeyError:
        print(f"\n  [LIVE] Creating put_spread PROPOSED row...")
        rec = create_strategy(
            strategy_id=PUT_SPREAD_ID,
            library_yaml_path=str(sleeve.library_yaml_path),
            candidate_pipeline_run_id="path_c_audit_2026-05-31",
            parent_strategy_id=MOM_HEDGE_ID,
            notes=(
                "Put-spread SPX tail hedge. Replaces mom_hedge_overlay. "
                "Path C verdict 2/3 strict pass; crisis-yield ratio +7.24 "
                "vs mom_hedge -1.28."
            ),
            actor="slm_deploy_script",
        )
        print(f"    state={rec.current_state.value}")

        for to_state, reason in [
            (StrategyState.AUDITED,
             "Path C audit pass (commit 38ad27d); 2/3 strict criteria + 1 negligibly off"),
            (StrategyState.APPROVED,
             "Manual approval — drag -0.04%/yr beats mom_hedge -14.56%/yr by 350x"),
            (StrategyState.PAPER_TRADE,
             "WIRE gate: sleeve registered; 24mo paper-trade clock starts"),
        ]:
            kwargs = {"has_candidate_pipeline_run": True} if to_state == StrategyState.AUDITED else {}
            if to_state == StrategyState.APPROVED:
                kwargs["has_human_approval"] = True
            rec = transition(
                strategy_id=PUT_SPREAD_ID,
                to_state=to_state,
                actor=actor if to_state == StrategyState.APPROVED else "slm_deploy_script",
                reason=reason,
                git_sha=git_sha,
                **kwargs,
            )
            print(f"  → {to_state.value:<14}")
        print(f"    final state: {rec.current_state.value}  "
              f"paper_trade_started: {rec.paper_trade_started.date()}")

    # ─── Phase 2: mom_hedge backfill LIVE + decommission to RAMP_DOWN ──
    print(f"\n  [mom_hedge_overlay decommission]")
    try:
        existing_mh = get_strategy(MOM_HEDGE_ID)
        print(f"    already in state store as {existing_mh.current_state.value}")
    except KeyError:
        print(f"    backfilling as LIVE (was in production but pre-SLM)...")
        rec_mh = create_strategy(
            strategy_id=MOM_HEDGE_ID,
            initial_state=StrategyState.LIVE,
            library_yaml_path=str(
                Path("data/research/mechanism_library/mom_hedge_overlay.yaml").resolve()
            ),
            notes=(
                "BACKFILL: mom_hedge_overlay was deployed in combined_book "
                "before SLM existed. Backfilled here so decommissioning "
                "follows lifecycle process."
            ),
            actor="slm_deploy_script",
        )
        print(f"    backfilled state: {rec_mh.current_state.value}")

    # Transition LIVE → RAMP_DOWN (manual decommission via human approval)
    rec_mh = transition(
        strategy_id=MOM_HEDGE_ID,
        to_state=StrategyState.RAMP_DOWN,
        actor=actor,
        reason=(
            "MANUAL DECOMMISSION: mom_hedge_overlay shown BROKEN per "
            "Path C analysis (commit 38ad27d). Cumulative -92% over 13yr; "
            "LOSES in SPX crisis months (-1.51% mean). "
            "Replaced by tail_hedge_put_spread (drag -0.04%/yr, "
            "crisis-yield +7.24)."
        ),
        has_human_approval=True,
        git_sha=git_sha,
    )
    print(f"    transitioned: LIVE → {rec_mh.current_state.value}")

    # ─── Audit trail ─────────────────────────────────────────────────
    print(f"\n  [audit trail — both sleeves]")
    for sid in [PUT_SPREAD_ID, MOM_HEDGE_ID]:
        hist = get_transition_history(sid)
        print(f"\n  {sid} ({len(hist)} transitions):")
        for h in hist:
            fr = h.from_state.value if h.from_state else "NULL"
            print(f"    {h.transition_at.isoformat()}  "
                  f"{fr:>10} → {h.to_state.value:<12} by {h.actor}")

    print(f"\n  [DONE] put_spread PAPER_TRADE clock running; mom_hedge RAMP_DOWN")
    print(f"         Run scripts/slm_verify_ledger.py to confirm Merkle integrity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
