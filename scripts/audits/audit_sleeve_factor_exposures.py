"""scripts/audit_sleeve_factor_exposures.py — apply BARRA-lite A.1
factor exposure check to live deployed sleeves.

Answers the senior-quant questions for each sleeve:
  Q1. Is D_PEAD just momentum in disguise?
  Q2. Do carry / TSMOM have near-zero equity factor exposure?
  Q3. What's each sleeve's alpha after MKT/SMB/MOM control?

USAGE:
  python scripts/audit_sleeve_factor_exposures.py
  python scripts/audit_sleeve_factor_exposures.py --rebuild-factors
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine.risk.barra_lite import (
    build_factor_returns,
    regress_sleeve_on_factors,
)

CACHE_FACTORS_BY_PHASE = {
    1: Path("data/cache/_barra_lite_factors.parquet"),
    2: Path("data/cache/_barra_lite_factors_phase2.parquet"),
    3: Path("data/cache/_barra_lite_factors_phase3.parquet"),
}

FACTOR_LIST_BY_PHASE = {
    1: "MKT/SMB/MOM",
    2: "MKT/SMB/MOM/HML/QMJ",
    3: "MKT/SMB/MOM/HML/QMJ + 11 GICS sectors",
}


def get_factors(rebuild: bool = False, phase: int = 1) -> pd.DataFrame:
    cache = CACHE_FACTORS_BY_PHASE.get(phase, CACHE_FACTORS_BY_PHASE[1])
    if not rebuild and cache.exists():
        return pd.read_parquet(cache)
    factor_list = FACTOR_LIST_BY_PHASE.get(phase, "MKT/SMB/MOM")
    extras = "+ Compustat" if phase >= 2 else ""
    if phase >= 3:
        extras += " + GICS"
    print(f"[factors] building Phase-{phase} ({factor_list}) from CRSP "
          f"{extras} cache...")
    f = build_factor_returns(phase=phase)
    f.to_parquet(cache)
    print(f"[factors] built {len(f)} months {f.index.min().date()} -> "
          f"{f.index.max().date()}; {len(f.columns)} factors; cached to {cache}")
    return f


def load_sleeve_returns() -> dict[str, pd.Series]:
    from engine.portfolio.combined_book import (
        build_equity_book, build_carry_book, build_tsmom_book,
    )
    return {
        "equity_book (D_PEAD + revision)": build_equity_book(),
        "carry_book (4-leg cross-asset)":   build_carry_book(),
        "tsmom_book (5-leg cross-asset)":   build_tsmom_book(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild-factors", action="store_true",
                     help="re-construct factor returns from cache")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                     help="1=MKT/SMB/MOM, 2=+HML/QMJ, 3=+11 GICS sectors")
    args = p.parse_args()

    print("=" * 90)
    print(" SLEEVE FACTOR EXPOSURE AUDIT — BARRA-lite A.1 (MKT / SMB / MOM)")
    print("=" * 90)
    print()

    factors = get_factors(rebuild=args.rebuild_factors, phase=args.phase)
    print(f"Phase {args.phase} factors: {len(factors)} months "
          f"{factors.index.min().date()} -> {factors.index.max().date()}")
    means_line = "  ".join(
        f"{c}={factors[c].mean()*12:+.2%}" for c in factors.columns
    )
    print(f"Factor mean returns (annualized): {means_line}")
    print()

    sleeves = load_sleeve_returns()
    reports = {}
    for name, sr in sleeves.items():
        try:
            r = regress_sleeve_on_factors(sr, factors, sleeve_name=name)
            reports[name] = r
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            continue

    print("-" * 90)
    print(" PER-SLEEVE EXPOSURE TABLE")
    print("-" * 90)
    factor_cols = list(factors.columns)
    head_cells = [f"{'sleeve':<35}", f"{'n':>4}",
                     f"{'alpha/yr':>10}", f"{'aα t':>6}"]
    for c in factor_cols:
        head_cells.append(f"{'b_'+c:>8}")
        head_cells.append(f"{'t'+c:>4}")
    head_cells.append(f"{'R^2':>6}")
    hdr = "  " + " ".join(head_cells)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in reports.items():
        cells = [f"{name:<35}", f"{r.n_months:>4d}",
                    f"{r.alpha_annualized:>+9.3%}", f"{r.alpha_t_hac:>+6.2f}"]
        for c in factor_cols:
            cells.append(f"{r.betas.get(c, 0):>+8.3f}")
            cells.append(f"{r.t_stats_hac.get(c, 0):>+4.2f}")
        cells.append(f"{r.r_squared:>6.3f}")
        print("  " + " ".join(cells))
    print()

    print("-" * 90)
    print(" VERDICT PER SLEEVE")
    print("-" * 90)
    for name, r in reports.items():
        print(f"  {name}:")
        print(f"    {r.verdict}")
    print()

    print("=" * 90)
    print(" SENIOR-QUANT QUESTIONS ANSWERED")
    print("=" * 90)
    eq_name = next((n for n in reports if "equity" in n), None)
    cy_name = next((n for n in reports if "carry" in n), None)
    ts_name = next((n for n in reports if "tsmom" in n), None)

    if eq_name:
        eq = reports[eq_name]
        mom_t = eq.t_stats_hac["MOM"]
        if abs(mom_t) < 2.0:
            print(f"  Q1. Is D_PEAD just momentum?  NO. "
                  f"MOM beta = {eq.betas['MOM']:+.3f} (|t|={abs(mom_t):.2f} insignificant)")
        else:
            print(f"  Q1. Is D_PEAD just momentum?  YES — beware. "
                  f"MOM beta = {eq.betas['MOM']:+.3f} (|t|={abs(mom_t):.2f} significant)")
        if eq.alpha_t_hac >= 2.0:
            print(f"      Alpha after 3-factor control: "
                  f"{eq.alpha_annualized:+.2%}/yr t={eq.alpha_t_hac:.2f} — "
                  f"genuine residual.")
        else:
            print(f"      Alpha t={eq.alpha_t_hac:.2f} — "
                  f"residual not significant after control.")

    if cy_name:
        cy = reports[cy_name]
        all_t = max(abs(cy.t_stats_hac[k]) for k in ["MKT", "SMB", "MOM"])
        if all_t < 2.0:
            print(f"  Q2a. Carry sleeve equity-factor-orthogonal?  YES. "
                  f"all factor |t| < 2.0 (max={all_t:.2f})")
        else:
            print(f"  Q2a. Carry sleeve equity-factor-orthogonal?  NO — "
                  f"max |t|={all_t:.2f} on factor(s); inspect")

    if ts_name:
        ts = reports[ts_name]
        all_t = max(abs(ts.t_stats_hac[k]) for k in ["MKT", "SMB", "MOM"])
        if all_t < 2.0:
            print(f"  Q2b. TSMOM sleeve equity-factor-orthogonal?  YES. "
                  f"all factor |t| < 2.0 (max={all_t:.2f})")
        else:
            print(f"  Q2b. TSMOM sleeve equity-factor-orthogonal?  NO — "
                  f"max |t|={all_t:.2f}; check eq-index futures contribution")

    print()
    print(" CAVEATS:")
    print("  -A.1 scope only (MKT/SMB/MOM). A.2 adds HML/QMJ + 11 sectors after")
    print("   Compustat fundamentals + GICS sector fetcher wiring.")
    print("  -Universe: CRSP top-1500 by point-in-time market_cap_at_q from PEAD panel.")
    print("  -Factor returns are equal-weighted, monthly. SMB and MOM are L/S construction.")
    print("  -HAC SE Newey-West with 6 lags. Monthly observations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
