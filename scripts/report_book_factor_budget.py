"""scripts/report_book_factor_budget.py — produce the institutional-
standard factor risk budget for the live deployed book.

The output answers: of our 10% annualized book volatility, what
fraction is from MOM exposure, sector tilts, idiosyncratic alpha, etc.

Per Aladdin / Axioma / MSCI BPM standard. This is L1 of the post-
Phase-3 improvement layer.

USAGE:
  python scripts/report_book_factor_budget.py [--phase {1,2,3}]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine.portfolio.combined_book import (
    DEFAULT_CARRY_RISK_WEIGHT,
    DEFAULT_TSMOM_RISK_WEIGHT,
    build_carry_book,
    build_equity_book,
    build_tsmom_book,
)
from engine.risk.factor_budget import compute_factor_budget


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, default=3, choices=[1, 2, 3])
    args = p.parse_args()

    sleeve_returns = {
        "equity_book":  build_equity_book(),
        "carry_book":   build_carry_book(),
        "tsmom_book":   build_tsmom_book(),
    }
    sleeve_weights = {
        "equity_book": 1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT,
        "carry_book":  DEFAULT_CARRY_RISK_WEIGHT,
        "tsmom_book":  DEFAULT_TSMOM_RISK_WEIGHT,
    }

    print("=" * 90)
    print(" BOOK FACTOR RISK BUDGET  (BARRA-equivalent Phase {})".format(args.phase))
    print("=" * 90)
    print(f"Sleeve weights: " +
          " / ".join(f"{k}={v:.0%}" for k, v in sleeve_weights.items()))
    print()
    print(" NOTE: vol shown below is the NATURAL blend vol (no vol-target overlay).")
    print(" Deployed book is scaled to 10% target via build_combined_book(book_vol_")
    print(" target=0.10) — that's a uniform linear scaling so % breakdowns are")
    print(" invariant. The 53%-MOM / 34%-idio / etc. shares apply to either.")
    print()

    report = compute_factor_budget(sleeve_returns, sleeve_weights, phase=args.phase)

    print("-" * 90)
    print(" BOOK RISK DECOMPOSITION")
    print("-" * 90)
    print(f"  Book annualized vol:    {report.book_vol_annualized:>8.3%}")
    print(f"    Factor (systematic):  {report.factor_vol_annualized:>8.3%}  "
          f"({report.pct_factor:.1%} of variance)")
    print(f"    Idiosyncratic:        {report.idio_vol_annualized:>8.3%}  "
          f"({report.pct_idio:.1%} of variance)")
    print(f"  N months used:          {report.n_months_used}")
    print()

    print("-" * 90)
    print(" TOP 5 FACTORS BY VARIANCE CONTRIBUTION")
    print("-" * 90)
    print(f"  {'rank':>4}  {'factor':<12}  {'book β':>10}  {'% of total risk':>16}")
    for i, (name, pct) in enumerate(report.top_5_factors_by_risk, 1):
        beta = report.factor_exposures.get(name, 0.0)
        print(f"  {i:>4}  {name:<12}  {beta:>+10.3f}  {pct:>15.1%}")
    print()

    print("-" * 90)
    print(" FULL FACTOR-EXPOSURE PROFILE (sorted by absolute book β)")
    print("-" * 90)
    sorted_exposures = sorted(report.factor_exposures.items(),
                                  key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  {'factor':<12}  {'book β':>10}  {'% of risk':>10}  {'(sign)':>8}")
    for name, beta in sorted_exposures:
        pct = report.factor_var_contrib_pct.get(name, 0.0)
        sign = "+" if pct >= 0 else "-"
        print(f"  {name:<12}  {beta:>+10.3f}  {pct:>+9.1%}  {sign:>8}")
    print()

    print("-" * 90)
    print(" PER-SLEEVE IDIOSYNCRATIC RISK CONTRIBUTION")
    print("-" * 90)
    print(f"  {'sleeve':<14}  {'weight':>8}  {'idio % of total risk':>20}")
    for name, pct in sorted(report.sleeve_idio_contrib_pct.items(),
                                 key=lambda kv: kv[1], reverse=True):
        w = sleeve_weights.get(name, 0.0)
        print(f"  {name:<14}  {w:>7.0%}   {pct:>19.1%}")
    print()

    print("=" * 90)
    print(" INSTITUTIONAL HONEST READ")
    print("=" * 90)
    factor_pct = report.pct_factor * 100
    idio_pct = report.pct_idio * 100
    top1_name, top1_pct = report.top_5_factors_by_risk[0]
    print(f"  Of the book's {report.book_vol_annualized*100:.1f}% annualized vol,")
    print(f"  {factor_pct:.0f}% is systematic factor risk and "
          f"{idio_pct:.0f}% is idiosyncratic (alpha-bearing).")
    print(f"  The single largest factor exposure is {top1_name} contributing "
          f"{top1_pct*100:.1f}% of total risk.")
    print()
    print(f"  IMPROVEMENT OPPORTUNITIES:")
    # Identify large factor concentrations as targets
    large_factors = [(n, p) for n, p in report.factor_var_contrib_pct.items()
                        if abs(p) >= 0.05]
    if large_factors:
        print(f"  Factors contributing >= 5% of book risk:")
        for n, p in sorted(large_factors, key=lambda kv: abs(kv[1]),
                              reverse=True)[:5]:
            print(f"    - {n:<12}: {p:>+6.1%} of risk - candidates that REDUCE "
                  f"this exposure ADD diversification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
