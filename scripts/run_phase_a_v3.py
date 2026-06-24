"""scripts/run_phase_a_v3.py — Phase A v3 rigorous ablation CLI driver.

Runs the 4 × 5 × CPCV (N=6, k=2) grid → metrics battery + PBO + paired
bootstrap → promotion gate → MCC approvals for winners (if any).

Usage:
    python scripts/run_phase_a_v3.py
    python scripts/run_phase_a_v3.py --no-mcc
    python scripts/run_phase_a_v3.py --signals sue_z abnormal_sue
    python scripts/run_phase_a_v3.py --n-splits 8 --k 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.research.ablation import runner


def fmt(v, w: int = 7, prec: int = 3):
    if v is None:
        return "  --   "
    try:
        if isinstance(v, float) and not float("inf") > abs(v) > -float("inf"):
            return f"{'  NaN  ':>{w}}"
        return f"{v:>{w}.{prec}f}"
    except Exception:
        return str(v).ljust(w)


def print_gate(gate):
    print()
    print("=" * 110)
    print("PROMOTION GATE v3 SUMMARY (RIGOROUS)")
    print("  Bars: median OOS lift ≥ +0.10 · PBO < 0.50 · deflSR ≥ 0.90 · bootstrap p < 0.05")
    print("=" * 110)
    print(f"  {'signal':<20} {'weighting':<22} {'OOS Sh':>8} {'lift':>7} {'PBO':>6} "
          f"{'deflSR':>7} {'boot p':>7} {'n_paths':>8} {'winner':>8}")
    print("  " + "-" * 100)
    for _, row in gate.iterrows():
        flag = "WINNER" if row["winner"] else ""
        print(f"  {row['signal']:<20} {row['weighting']:<22}"
              f"  {fmt(row['median_oos_sharpe'])} {fmt(row['lift_vs_equal'])}"
              f" {fmt(row.get('pbo'), 6)} {fmt(row['mean_deflated_sr'])}"
              f" {fmt(row['bootstrap_p'], 7, 4)} {int(row['n_paths']):>8}  {flag:>8}")


def create_mcc_for_winners(gate, out, rt_cost_bps: float) -> list[str]:
    from engine.governance.approval_ledger import create_request
    winners = gate[gate["winner"] == True]
    if winners.empty:
        return []
    rids = []
    for _, row in winners.iterrows():
        rid = create_request(
            request_type="weight_method_change",
            title=f"Phase A v3 winner · {row['signal']} × {row['weighting']}",
            summary=(
                f"Phase A v3 rigorous ablation. Signal={row['signal']}, "
                f"weighting={row['weighting']} achieves median OOS Sharpe "
                f"{row['median_oos_sharpe']:.3f} vs equal-weight baseline "
                f"(lift {row['lift_vs_equal']:+.3f}). PBO={row['pbo']:.3f} "
                f"(< 0.50 bar → IS performance generalizes). Mean deflated SR "
                f"{row['mean_deflated_sr']:.3f} (n_trials={out['n_trials']}). "
                f"Cross-path bootstrap p={row['bootstrap_p']:.4f}. "
                f"CPCV N=6, k=2 → {row['n_paths']} backtest paths. "
                f"Sector-neutral L/S decile within GICS gsector, vol-target 10% "
                f"per leg, RT cost {rt_cost_bps}bps × turnover both legs, "
                f"market-cap floor $500M, max single weight 10%."
            ),
            proposed_payload={
                "sleeve":           "equity_book",
                "signal":           row["signal"],
                "weighting_method": row["weighting"],
                "sector_neutral":   True,
                "vol_target_ann":   0.10,
                "mcap_floor_usd":   500_000_000,
                "rt_cost_bps":      rt_cost_bps,
                "max_single_weight": 0.10,
            },
            current_state={
                "sleeve":           "equity_book",
                "signal":           "sue_z",
                "weighting_method": "equal",
            },
            evidence_pack={
                "median_oos_sharpe":  float(row["median_oos_sharpe"]),
                "lift_vs_equal":      float(row["lift_vs_equal"]),
                "pbo":                float(row["pbo"]),
                "mean_deflated_sr":   float(row["mean_deflated_sr"]),
                "bootstrap_p":        float(row["bootstrap_p"]),
                "n_paths":            int(row["n_paths"]),
                "n_trials_grid":      out["n_trials"],
                "construction":       "v3 rigorous: CPCV + sector-neutral + vol-target + costs + Newey-West + PBO",
                "family":             "weight_method_change",
            },
            created_by="scripts/run_phase_a_v3.py",
        )
        rids.append(rid)
    return rids


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase A v3 rigorous ablation")
    ap.add_argument("--signals",     nargs="+", help="signal definitions to test (default: all)")
    ap.add_argument("--weightings",  nargs="+", help="weighting methods to test (default: all)")
    ap.add_argument("--n-splits",    type=int, default=6, help="CPCV N splits")
    ap.add_argument("--k",           type=int, default=2, help="CPCV k test groups per split")
    ap.add_argument("--no-sector-neutral", action="store_true")
    ap.add_argument("--rt-bps",      type=float, default=30.0, help="round-trip cost in bps")
    ap.add_argument("--no-mcc",      action="store_true", help="don't create MCC approvals")
    args = ap.parse_args()

    out = runner.run_grid(
        signals=args.signals,
        weightings=args.weightings,
        n_splits=args.n_splits,
        k_test_groups=args.k,
        sector_neutral=(not args.no_sector_neutral),
        rt_cost_bps=args.rt_bps,
    )
    gate = runner.apply_promotion_gate(out)
    print_gate(gate)

    out_dir = runner.write_outputs(out, gate)
    print(f"\nOutputs → {out_dir}")

    if not args.no_mcc:
        rids = create_mcc_for_winners(gate, out, rt_cost_bps=args.rt_bps)
        if rids:
            print(f"\n=== MCC Gateway ===")
            for rid in rids:
                print(f"  Approval created: {rid}")
        else:
            print(f"\nNo cells cleared the v3 promotion gate.")
            print(f"This is a VALID scientific result — DeMiguel-Garlappi-Uppal 2009 1/N defense holds.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
