"""scripts/report_book_risk_forecast.py — institutional forward-
looking risk forecast for the deployed 5-sleeve book.

BARRA Phase 4. Aladdin / Axioma BVR equivalent capability.

Output: forecast annualized vol + 95% bootstrap CI + factor/idio split
+ Ledoit-Wolf shrinkage intensity used.

USAGE:
  python scripts/report_book_risk_forecast.py [--phase {1,2,3}] [--lambda 0.97]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine.portfolio.combined_book import (
    DEFAULT_CARRY_RISK_WEIGHT,
    DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
    DEFAULT_MOM_HEDGE_RISK_WEIGHT,
    DEFAULT_TSMOM_RISK_WEIGHT,
    build_carry_book,
    build_crisis_hedge_book,
    build_equity_book,
    build_mom_hedge_book,
    build_tsmom_book,
)
from engine.risk.risk_forecast import portfolio_risk_forecast


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--lambda", dest="ewma_lambda", type=float, default=0.97,
                     help="EWMA decay for specific risk (RiskMetrics monthly default 0.97)")
    p.add_argument("--n-bootstrap", type=int, default=500)
    args = p.parse_args()

    sleeves = {
        "equity_book":  build_equity_book(),
        "carry_book":   build_carry_book(),
        "tsmom_book":   build_tsmom_book(),
        "crisis_hedge": build_crisis_hedge_book(),
        "mom_hedge":    build_mom_hedge_book(),
    }
    eq_w = (1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
              - DEFAULT_CRISIS_HEDGE_RISK_WEIGHT - DEFAULT_MOM_HEDGE_RISK_WEIGHT)
    weights = {
        "equity_book": eq_w,
        "carry_book":  DEFAULT_CARRY_RISK_WEIGHT,
        "tsmom_book":  DEFAULT_TSMOM_RISK_WEIGHT,
        "crisis_hedge": DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
        "mom_hedge":    DEFAULT_MOM_HEDGE_RISK_WEIGHT,
    }

    print("=" * 90)
    print(" BOOK RISK FORECAST — BARRA Phase 4 (forward-looking)")
    print(f" Method: Ledoit-Wolf shrinkage + EWMA(lambda={args.ewma_lambda}) "
          f"specific risk")
    print("=" * 90)
    print(f"Sleeve weights: " +
          " / ".join(f"{k}={v:.0%}" for k, v in weights.items()))
    print()

    r = portfolio_risk_forecast(
        sleeves, weights, phase=args.phase,
        ewma_lambda=args.ewma_lambda,
        n_bootstrap=args.n_bootstrap,
    )

    print("-" * 90)
    print(" FORECAST POINT ESTIMATE")
    print("-" * 90)
    print(f"  forecast ann vol:           {r.forecast_vol_annualized:>8.3%}")
    print(f"  95% bootstrap CI:           "
          f"({r.forecast_ci_95[0]:.3%}, {r.forecast_ci_95[1]:.3%})")
    print(f"    CI half-width:            "
          f"{(r.forecast_ci_95[1] - r.forecast_ci_95[0]) / 2:>+.3%}")
    print()
    print(f"  factor (systematic) vol:    {r.factor_vol_forecast:>8.3%}  "
          f"({r.pct_factor:.1%} of total var)")
    print(f"  idio vol:                   {r.idio_vol_forecast:>8.3%}  "
          f"({r.pct_idio:.1%} of total var)")
    print(f"  Ledoit-Wolf shrinkage δ:    {r.shrinkage_intensity:>+.3f}  "
          f"(0=sample cov, 1=full shrink)")
    print(f"  n_months used:              {r.n_months_used}")
    print()

    print("-" * 90)
    print(" PER-SLEEVE FORECAST IDIO VOL (annualized)")
    print("-" * 90)
    print(f"  {'sleeve':<14}  {'weight':>8}  {'ewma idio vol':>13}")
    for nm, ivol in sorted(r.per_sleeve_idio_forecast.items(),
                                key=lambda kv: kv[1], reverse=True):
        w = weights.get(nm, 0.0)
        print(f"  {nm:<14}  {w:>7.0%}   {ivol:>12.3%}")
    print()

    print("-" * 90)
    print(" TOP 5 BOOK FACTOR EXPOSURES")
    print("-" * 90)
    sorted_exp = sorted(r.book_exposures.items(),
                            key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  {'factor':<10}  {'book β':>10}")
    for nm, b in sorted_exp[:5]:
        print(f"  {nm:<10}  {b:>+10.3f}")
    print()

    print("=" * 90)
    print(" INSTITUTIONAL READ")
    print("=" * 90)
    vol_pct = r.forecast_vol_annualized * 100
    ci_low = r.forecast_ci_95[0] * 100
    ci_high = r.forecast_ci_95[1] * 100
    print(f"  Point forecast: book carries {vol_pct:.1f}% expected annualized vol.")
    print(f"  95% confidence: [{ci_low:.1f}%, {ci_high:.1f}%].")
    print(f"  Factor share {r.pct_factor*100:.0f}% / idio share {r.pct_idio*100:.0f}%.")
    print(f"  Shrinkage δ={r.shrinkage_intensity:.2f} indicates "
          f"{'mostly sample-cov' if r.shrinkage_intensity < 0.3 else 'meaningful shrinkage applied'}.")
    print()
    print("  KEY USES:")
    print("  - VAR scaling: 99% 1-month VAR ≈ 2.33 × monthly vol × $AUM")
    print("  - Vol-target check: forecast within target band?")
    print("  - Risk attribution: which factors drive next-period risk?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
