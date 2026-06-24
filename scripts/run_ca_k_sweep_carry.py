"""scripts/run_ca_k_sweep_carry.py — Phase 5.7 multi-asset rework.

Real PBB-validated k-sweep on cross_asset_carry using per-contract
(signal, returns, position) panels via build_carry_contract_panels()
+ apply_ca_filter_to_panel() — the multi-asset fix for the
abstraction defect caught in 2026-06-01 post-audit
([[project-multi-asset-ca-filter-gap-2026-06-01]]).

Run:
  python -m scripts.run_ca_k_sweep_carry
"""
from __future__ import annotations

import sys

import pandas as pd

from engine.portfolio.carry_sleeve import build_carry_contract_panels
from engine.portfolio.execution_filter import apply_ca_filter_to_panel
from engine.validation.filter_counterfactual import evaluate_k_sweep


TCOST = 0.0008  # 8 bp round trip per cross_asset_carry.yaml
K_VALUES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def main() -> int:
    print("Building per-contract carry panels from real history...")
    signal_panel, returns_panel, position_panel = build_carry_contract_panels()
    print(f"  → {signal_panel.shape[0]} months × {signal_panel.shape[1]} contracts "
          f"({signal_panel.index.min().date()} → {signal_panel.index.max().date()})")

    # Diagnose the BASELINE (k=0, always trade) — this is the "naive
    # rebalance every month" backtest
    print("\nBaseline diagnostics (k=0 — always trade):")
    base_rets, base_diag = apply_ca_filter_to_panel(
        sleeve_id="cross_asset_carry",
        signal_panel=signal_panel,
        returns_panel=returns_panel,
        target_position_panel=position_panel,
        tcost_round_trip=TCOST,
        k=0.0,
    )
    for k_, v in base_diag.items():
        print(f"  {k_:>22s}  {v}")
    print(f"  baseline Sharpe (monthly): {base_rets.mean() / base_rets.std():.4f}")
    print(f"  baseline annualized:       "
          f"{base_rets.mean() * 12:+.2%} / vol {base_rets.std() * (12 ** 0.5):.2%}")

    def factory(k: float) -> tuple[pd.Series, pd.Series]:
        baseline_rets, _ = apply_ca_filter_to_panel(
            sleeve_id="cross_asset_carry",
            signal_panel=signal_panel, returns_panel=returns_panel,
            target_position_panel=position_panel,
            tcost_round_trip=TCOST, k=0.0,
        )
        filtered_rets, _ = apply_ca_filter_to_panel(
            sleeve_id="cross_asset_carry",
            signal_panel=signal_panel, returns_panel=returns_panel,
            target_position_panel=position_panel,
            tcost_round_trip=TCOST, k=k,
        )
        return baseline_rets, filtered_rets

    print(f"\nRunning PBB k-sweep ({len(K_VALUES)} k values, 3000 iters each)...")
    results = evaluate_k_sweep(
        sleeve_name="cross_asset_carry",
        counterfactual_factory=factory,
        k_values=K_VALUES,
        n_iter=3000,
        rng_seed=42,
    )

    # Per-k diagnostics — turnover reduction is the structural KPI
    print("\nPer-k diagnostics:")
    for k in K_VALUES:
        _, diag = apply_ca_filter_to_panel(
            sleeve_id="cross_asset_carry",
            signal_panel=signal_panel, returns_panel=returns_panel,
            target_position_panel=position_panel,
            tcost_round_trip=TCOST, k=k,
        )
        print(f"  k={k:>4.1f}  trade_rate={diag['trade_rate_pct']:5.1f}%  "
              f"monthly_turnover={diag['monthly_turnover']:6.3f}  "
              f"annual={diag['annual_turnover']:6.3f}")

    print("\nResults (Hochberg-adjusted p):")
    print(f"  {'k':>4s}  {'verdict':>14s}  {'diff':>9s}  {'CI':>22s}  "
          f"{'p':>6s}  {'turnover_red%':>13s}")
    for r in results:
        k_str = r.filter_descriptor.replace("CA filter k=", "")
        ci_str = f"[{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]"
        print(f"  {k_str:>4s}  {r.verdict:>14s}  {r.sharpe_diff:+9.4f}  "
              f"{ci_str:>22s}  {r.p_value:>6.3f}  "
              f"{(r.turnover_reduction_pct or 0):>12.1f}%")

    deploys = [r for r in results if r.verdict == "DEPLOY"]
    if not deploys:
        print("\nVerdict: NO k surviving Hochberg DEPLOY. "
              "Keep paper_default k=2.0 in YAML.")
        return 0
    best = min(
        deploys,
        key=lambda r: float(r.filter_descriptor.replace("CA filter k=", "")),
    )
    chosen_k = float(best.filter_descriptor.replace("CA filter k=", ""))
    print(f"\nVerdict: SMALLEST k surviving DEPLOY = {chosen_k}")
    print(f"  Sharpe diff: {best.sharpe_diff:+.4f} "
          f"(CI [{best.diff_ci_lo:+.4f}, {best.diff_ci_hi:+.4f}])")
    print(f"  PBB p (Hochberg-adjusted): {best.p_value:.4f}")
    print(f"  Turnover reduction:        "
          f"{best.turnover_reduction_pct or 0:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
