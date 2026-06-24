"""scripts/slm_phase2_5_smoke_three_layer.py — Phase 2.5 end-to-end
smoke test: PIT SN through corrected 3-layer validation framework
(Bayesian + DeflSR + OBF) on the institutional-standard 24-month
window.

Demonstrates:
  1. Layer 1 Bayesian — posterior P(Sharpe > 0.50) updating
  2. Layer 2 DeflSR — multiple-testing-adjusted significance
  3. Layer 3 OBF — pre-registered frequentist sanity check
  4. Composite voting with reject-blocking asymmetry
  5. Side-by-side: real PIT SN 24mo vs synthesized 24mo from prior
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import engine.research.sleeves  # noqa: F401

from engine.research.sequential_testing import (
    default_obf_boundary_paper_trade, default_obf_boundary_shadow,
)
from engine.research.sleeve_registry import get_sleeve
from engine.research.three_layer_validator import (
    ThreeLayerDecision, evaluate_three_layer,
)


def main() -> int:
    print("=" * 95)
    print(" SLM Phase 2.5 — 3-LAYER VALIDATOR SMOKE (PIT SN, 24mo window, post-critique fix)")
    print("=" * 95)

    sleeve = get_sleeve("post_earnings_drift_pit_sn")
    full = sleeve.returns()
    print(f"\n  Full PIT SN series: n={len(full)} months "
          f"({full.index.min().date()} → {full.index.max().date()})")

    # ─── Scenario 1: real PIT SN last 24mo ──────────────────────────────
    print("\n[1] REAL PIT SN — last 24 months window")
    pt_24mo = full.tail(24)
    print(f"  Sharpe (annualized): {pt_24mo.mean()*12/(pt_24mo.std()*(12**0.5)):.3f}")

    boundary = default_obf_boundary_paper_trade()
    print(f"\n  Pre-registered OBF boundary (24mo, alpha=0.05, "
          f"first-look month 12):")
    table = boundary.planned_boundary_table()
    print(f"    {'month':>6}  {'crit t':>8}  {'crit Sharpe (impl)':>20}")
    for m, t in [table[0], table[len(table)//2], table[-1]]:  # first, mid, last
        crit_sharpe = t * (12**0.5) / (m**0.5)
        print(f"    {m:>6d}  {t:>8.3f}  {crit_sharpe:>20.3f}")

    print(f"\n  Running 3-layer validator at month 24 (family-aware n_trials)...")
    from engine.research.family_trial_counter import explain_count
    fc = explain_count("earnings_underreaction")
    print(f"  family='earnings_underreaction': library_entries={fc['library_entries']}, "
          f"exploration_buffer={fc['exploration_buffer']} ({fc['buffer_source']}), "
          f"computed_n_trials={fc['computed_n_trials']}")
    result = evaluate_three_layer(
        sleeve_returns=pt_24mo,
        prior_mean_sharpe=1.38,                # P-D8 honest target
        family="earnings_underreaction",       # AUTO-resolves n_trials
        obf_boundary=boundary,
        obf_month=24,
    )
    print(f"\n  ── Layer 1 (Bayesian posterior) ──────────────────────────")
    b = result.layer1_bayesian
    print(f"    observed Sharpe:    {b.observed_sharpe_ann:+.3f}")
    print(f"    prior:              N({b.prior_mean:+.2f}, {b.prior_sd:.2f})")
    print(f"    posterior:          N({b.posterior_mean:+.3f}, "
          f"{b.posterior_sd:.3f})")
    print(f"    P(Sharpe > {b.threshold:.2f} | data) = "
          f"{b.posterior_prob_above_threshold:.3f}")
    print(f"    DECISION:           {b.decision.value}")

    print(f"\n  ── Layer 2 (Deflated Sharpe, Bailey-LdP) ─────────────────")
    d = result.layer2_deflated_sr
    print(f"    Sharpe (annualized): {d.sharpe_annualized:+.3f}")
    print(f"    skew / excess kurt:  {d.skew:+.3f} / {d.excess_kurtosis:+.3f}")
    print(f"    n_trials (search):   {d.n_trials}")
    print(f"    expected max SR:     {d.expected_max_sr:+.3f}")
    print(f"    DeflSR:              {d.deflated_sr:.3f}")
    print(f"    VERDICT:             {d.verdict}")

    print(f"\n  ── Layer 3 (OBF, Layer-3 sanity check) ───────────────────")
    o = result.layer3_obf
    if o is not None:
        print(f"    observed t-stat:    {o.observed_t:+.3f}")
        print(f"    upper crit t (m={o.month}): {o.upper_critical_t:.3f}")
        print(f"    DECISION:           {o.decision.value}")
    else:
        print(f"    SKIPPED")

    print(f"\n  ── COMPOSITE VOTING ──────────────────────────────────────")
    print(f"    layer1_vote: {result.layer1_vote}")
    print(f"    layer2_vote: {result.layer2_vote}")
    print(f"    layer3_vote: {result.layer3_vote}")
    print(f"    FINAL:       {result.final_decision.value}  "
          f"(evidence_passed={result.evidence_passed})")

    # ─── Scenario 2: FULL 123mo for comparison ─────────────────────────
    print("\n[2] FULL 123mo PIT SN (control — what we'd see with full data)")
    full_result = evaluate_three_layer(
        sleeve_returns=full,
        prior_mean_sharpe=1.38,
        family="earnings_underreaction",
        obf_boundary=default_obf_boundary_shadow(),  # 36mo boundary
        obf_month=36,
    )
    print(f"  Layer 1 posterior_mean: {full_result.layer1_bayesian.posterior_mean:+.3f}  "
          f"P>0.5: {full_result.layer1_bayesian.posterior_prob_above_threshold:.3f}  "
          f"→ {full_result.layer1_vote}")
    print(f"  Layer 2 DeflSR: {full_result.layer2_deflated_sr.deflated_sr:.3f}  "
          f"→ {full_result.layer2_vote}")
    print(f"  Layer 3 OBF (m=36): {full_result.layer3_obf.decision.value if full_result.layer3_obf else 'N/A'}  "
          f"→ {full_result.layer3_vote}")
    print(f"  FINAL: {full_result.final_decision.value}")

    # ─── Scenario 3: synthesized weak signal for contrast ──────────────
    print("\n[3] SYNTHESIZED WEAK SIGNAL (true Sharpe ≈ 0.3) — should NOT accept")
    rng = np.random.default_rng(7)
    n = 24
    weak = pd.Series(
        rng.normal(0.003, 0.025, n),  # implied annualized Sharpe ~ 0.4
        index=pd.date_range("2022-01-31", periods=n, freq="ME"),
    )
    weak_result = evaluate_three_layer(
        sleeve_returns=weak,
        prior_mean_sharpe=1.38,           # we EXPECTED 1.38 but got weak
        family="earnings_underreaction",
        obf_boundary=boundary,
        obf_month=24,
    )
    print(f"  Layer 1: {weak_result.layer1_vote} "
          f"(P>0.5: {weak_result.layer1_bayesian.posterior_prob_above_threshold:.3f})")
    print(f"  Layer 2: {weak_result.layer2_vote} "
          f"(DeflSR: {weak_result.layer2_deflated_sr.deflated_sr:.3f})")
    print(f"  Layer 3: {weak_result.layer3_vote}")
    print(f"  FINAL: {weak_result.final_decision.value}")

    print("\n" + "=" * 95)
    print(" SMOKE TEST COMPLETE")
    print("=" * 95)
    return 0


if __name__ == "__main__":
    sys.exit(main())
