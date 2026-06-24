"""scripts/slm_phase2_smoke_paper_trade.py — SLM Phase 2 end-to-end
smoke test: simulate PIT SN going through 6 months of paper trade with
sequential O'Brien-Fleming boundary evaluation.

Walks through:
  1. Show the planned OBF boundary table BEFORE any data is observed
     (this is what gets pre-registered)
  2. For each simulated month, compute the role-specific metric +
     boundary decision
  3. Print the running decision trace + final outcome
  4. Demonstrate role-specific dispatch: same returns, different roles
     yield different decisions

Uses real PIT SN monthly returns from data/cache/_dpead_sn_pit_monthly.parquet
to simulate a realistic 6-month observation window. Idempotent: writes
nothing to state store; uses a temp DB.
"""
from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import engine.research.sleeves  # noqa: F401

from engine.research.role_specific_metric_eval import (
    evaluate_role_specific_metric,
)
from engine.research.sequential_testing import (
    OBrienFlemingBoundary, SequentialDecision,
    default_obf_boundary_paper_trade,
)
from engine.research.sleeve_registry import get_sleeve
from engine.research.strategy_lifecycle import SleeveRole


def main() -> int:
    print("=" * 90)
    print(" SLM Phase 2 — END-TO-END SMOKE (simulate 6mo paper trade for PIT SN)")
    print("=" * 90)

    # ─── Step 1: pre-register the boundary ─────────────────────────────
    print("\n[1] PRE-REGISTERED O'Brien-Fleming boundary (paper-trade default)")
    boundary = default_obf_boundary_paper_trade()
    print(f"  total_months={boundary.total_months}  "
          f"alpha_two_sided={boundary.alpha_two_sided}  "
          f"min_first_look={boundary.min_months_before_first_look}")
    print(f"\n  Pre-registered critical t at each look:")
    print(f"    {'month':>6}  {'critical_t (ACCEPT if observed_t >= this)':>50}")
    for m, crit in boundary.planned_boundary_table():
        print(f"    {m:>6d}  {crit:>50.3f}")

    # ─── Step 2: load PIT SN real returns ───────────────────────────────
    print("\n[2] Real PIT SN monthly returns (simulating 6-month paper trade)")
    sleeve = get_sleeve("post_earnings_drift_pit_sn")
    full = sleeve.returns()
    print(f"  full series: {len(full)} months "
          f"({full.index.min().date()} → {full.index.max().date()})")
    # Take the last 6 months as the simulated paper-trade window
    pt_window = full.tail(6)
    print(f"  paper-trade window (simulated):")
    for d, v in pt_window.items():
        print(f"    {d.date()}  return={v:+.4f}")

    # ─── Step 3: monthly tick simulation ────────────────────────────────
    print("\n[3] Monthly tick simulation — alpha_seeker role")
    print(f"  {'month':>5}  {'metric':>15}  {'t-stat':>8}  "
          f"{'critical':>9}  {'decision':>10}  {'rationale':>50}")
    print(f"  {'-'*5}  {'-'*15}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*50}")
    final_decision = None
    for m in range(1, len(pt_window) + 1):
        window = pt_window.iloc[:m]
        metric = evaluate_role_specific_metric(
            role=SleeveRole.ALPHA_SEEKER, sleeve_returns=window,
        )
        result = boundary.decide(observed_t=metric.t_stat, m=m)
        print(f"  {m:>5d}  {metric.metric_value:>+15.3f}  "
              f"{metric.t_stat:>+8.3f}  "
              f"{result.upper_critical_t:>9.3f}  "
              f"{result.decision.value:>10s}  "
              f"{result.rationale[:50]:>50s}")
        if result.decision in (SequentialDecision.ACCEPT, SequentialDecision.REJECT):
            final_decision = result.decision
            break
    if final_decision is None:
        final_decision = result.decision  # last value

    print(f"\n  FINAL DECISION: {final_decision.value}")

    # ─── Step 4: role dispatch demonstration ────────────────────────────
    print("\n[4] Same returns — different role yields different decision")
    role_tests = [
        ("alpha_seeker", SleeveRole.ALPHA_SEEKER, {}),
        ("risk_premium_harvester", SleeveRole.RISK_PREMIUM_HARVESTER, {}),
    ]
    full_window = pt_window
    for label, role, ctx in role_tests:
        m = evaluate_role_specific_metric(
            role=role, sleeve_returns=full_window, **ctx,
        )
        b_result = boundary.decide(observed_t=m.t_stat, m=len(full_window))
        print(f"  role={label:>30s}  "
              f"metric={m.metric_name:>20s}={m.metric_value:+.3f}  "
              f"t={m.t_stat:+.3f}  decision={b_result.decision.value}")

    # ─── Step 5: insurance role — needs risk_source ─────────────────────
    print("\n[5] Insurance role — needs risk_source_returns")
    np.random.seed(7)
    fake_risk = pd.Series(np.random.normal(0.01, 0.04, len(full_window)),
                          index=full_window.index)
    fake_insurance_returns = (-0.6 * fake_risk +
                              pd.Series(np.random.normal(0, 0.005, len(full_window)),
                                        index=full_window.index))
    m_ins = evaluate_role_specific_metric(
        role=SleeveRole.INSURANCE, sleeve_returns=fake_insurance_returns,
        risk_source_returns=fake_risk,
    )
    b_ins = boundary.decide(observed_t=m_ins.t_stat, m=len(full_window))
    print(f"  hedge_beta = {m_ins.metric_value:+.3f}  "
          f"(passes threshold ≤ -0.30: {m_ins.evidence_passed})  "
          f"t-stat={m_ins.t_stat:+.3f}  decision={b_ins.decision.value}")

    print("\n" + "=" * 90)
    print(" SMOKE TEST COMPLETE")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
