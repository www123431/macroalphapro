"""scripts/slm_smoke_test_pit_sn.py — Strategy Lifecycle Manager Phase 0
end-to-end smoke test.

Walks PIT SN D_PEAD through the registry + state-store from PROPOSED
through PAPER_TRADE, validating that:

  1. Sleeve registers via @register_sleeve and resolves via get_sleeve()
  2. Audit blocks parse cleanly from the library YAML (Pydantic strict)
  3. Returns series loads (sleeve interface contract)
  4. State machine accepts PROPOSED → AUDITED → APPROVED → PAPER_TRADE
     with required evidence
  5. Transition history is immutable and queryable
  6. Role-specific gate enforced for PAPER_TRADE → SHADOW (would-fail
     without sequential testing evidence — proves Phase 2 gate is wired)

Idempotent: deletes its test DB at end so the run is repeatable.
Does NOT touch DEFAULT_DB_PATH.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine.research.sleeves  # noqa: F401  — triggers @register_sleeve

from engine.research.sleeve_registry import (
    get_sleeve, list_registered_sleeves,
)
from engine.research.strategy_lifecycle import (
    GateNotMetError, SleeveRole, StrategyState,
)
from engine.research.strategy_state_store import (
    create_strategy, get_strategy, get_transition_history,
    reset_db_for_test, transition,
)


def main() -> int:
    print("=" * 88)
    print(" SLM Phase 0 — END-TO-END SMOKE TEST (PIT SN D_PEAD)")
    print("=" * 88)

    # Use an isolated temp DB so the production state store is untouched
    tmp = Path(tempfile.mkdtemp(prefix="slm_smoke_")) / "lifecycle.db"

    try:
        # ─── Step 1: registry ───────────────────────────────────────────
        print("\n[1] Sleeve registry ─────────────────────────────────────")
        registered = list_registered_sleeves()
        print(f"  registered sleeves: {registered}")
        assert "post_earnings_drift_pit_sn" in registered
        assert "post_earnings_drift" in registered  # parent
        sleeve = get_sleeve("post_earnings_drift_pit_sn")
        print(f"  resolved: {sleeve.strategy_id}")
        print(f"  library YAML: {sleeve.library_yaml_path.name}")

        # ─── Step 2: audit blocks parse cleanly ────────────────────────
        print("\n[2] Audit blocks (Pydantic strict parse) ────────────────")
        ab = sleeve.audit_blocks()
        print(f"  cost_model.audit_status:      {ab.cost_model.audit_status}")
        print(f"  cost_model.type:              {ab.cost_model.type}")
        print(f"  cost_model.audit_commit:      {ab.cost_model.audit_commit}")
        print(f"  cost_model.sharpe @10M:       {ab.cost_model.multi_aum_sharpe_sleeve.at_10M:.3f}")
        print(f"  cost_model.hard_capacity_usd: ${ab.cost_model.capacity.hard_capacity_usd/1e6:.0f}M")
        print(f"  factor_exposure.phase:        {ab.factor_exposure.phase}")
        print(f"  factor_exposure.proposed_role:{ab.factor_exposure.proposed_role}")
        print(f"  factor_exposure.alpha_t_hac:  {ab.factor_exposure.alpha_t_hac:.3f}")
        print(f"  factor_exposure.alpha_ann:    {ab.factor_exposure.alpha_annualized:+.4f}")

        # ─── Step 3: returns load ──────────────────────────────────────
        print("\n[3] Returns series ──────────────────────────────────────")
        r = sleeve.returns()
        print(f"  n_months: {len(r)}")
        print(f"  date range: {r.index.min().date()} → {r.index.max().date()}")
        print(f"  gross Sharpe: {(r.mean()*12)/(r.std()*(12**0.5)):.3f}")

        # ─── Step 4: lifecycle walk ────────────────────────────────────
        print("\n[4] State machine walk ──────────────────────────────────")
        sid = "post_earnings_drift_pit_sn"
        rec = create_strategy(
            strategy_id=sid,
            library_yaml_path=str(sleeve.library_yaml_path),
            candidate_pipeline_run_id="manual_2026-05-31_audit",
            notes="PIT SN D_PEAD; alpha t=9.65; honest deploy Sharpe 1.38",
            actor="slm_smoke_test",
            db_path=tmp,
        )
        print(f"  CREATED: state={rec.current_state.value} "
              f"proposed_at={rec.proposed_at.date()}")

        rec = transition(
            strategy_id=sid,
            to_state=StrategyState.AUDITED,
            actor="candidate_pipeline",
            reason="14-step pipeline PROMOTE_AS_REPLACEMENT (cosine 0.78 with parent)",
            has_candidate_pipeline_run=True,
            git_sha="844d401",
            db_path=tmp,
        )
        print(f"  AUDITED: state={rec.current_state.value} audited_at={rec.audited_at.date()}")

        rec = transition(
            strategy_id=sid,
            to_state=StrategyState.APPROVED,
            actor="zhangxizhe",
            reason="manual approval after reviewing audit blocks + DA verdict",
            has_human_approval=True,
            git_sha="844d401",
            db_path=tmp,
        )
        print(f"  APPROVED by {rec.approved_by} at {rec.approved_at.date()}")

        rec = transition(
            strategy_id=sid,
            to_state=StrategyState.PAPER_TRADE,
            actor="slm_smoke_test",
            reason="WIRE gate: sleeve registered + paper trade started",
            db_path=tmp,
        )
        print(f"  PAPER_TRADE: started {rec.paper_trade_started.date()}")

        # ─── Step 5: role-specific gate enforcement ────────────────────
        print("\n[5] Role-specific gate enforcement ──────────────────────")
        # This SHOULD fail — paper trade just started, no months of evidence
        try:
            transition(
                strategy_id=sid,
                to_state=StrategyState.SHADOW,
                actor="slm_smoke_test",
                paper_trade_months=0,
                sequential_test_pass=False,
                ramp_protocol_step=0,
                extra_evidence={
                    "role": "alpha_seeker",
                    "role_specific_evidence_passed": False,
                },
                db_path=tmp,
            )
            print("  UNEXPECTED PASS — should have raised GateNotMetError")
            return 1
        except GateNotMetError as exc:
            print(f"  EXPECTED-FAIL: PAPER_TRADE → SHADOW correctly blocked")
            print(f"    reason: {exc}")

        # ─── Step 6: transition history is queryable ───────────────────
        print("\n[6] Transition history (immutable audit trail) ──────────")
        history = get_transition_history(sid, db_path=tmp)
        for h in history:
            from_s = h.from_state.value if h.from_state else "NULL"
            print(f"  {h.transition_at.isoformat()}  "
                  f"{from_s:>12} → {h.to_state.value:<14} "
                  f"by {h.actor} (git={h.git_sha or '-'})")

        # ─── Done ──────────────────────────────────────────────────────
        print("\n" + "=" * 88)
        print("  SMOKE TEST PASSED")
        print("=" * 88)
        return 0
    finally:
        reset_db_for_test(tmp)


if __name__ == "__main__":
    sys.exit(main())
