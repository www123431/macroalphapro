"""scripts/audit_book_cost_capacity.py — backtest live deployed book
through the new Almgren-Chriss cost model + capacity_report.

What this answers: "用了新精准成本模型后我们策略的 Sharpe 还剩多少?"

INPUT: live deployed config from engine.portfolio.combined_book
  - equity 70%  = D_PEAD + analyst revision (RT_EQ = 30bp flat)
  - carry  25%  = 4-leg cross-asset carry  (RT_CY = 12bp flat)
  - tsmom   5%  = 5-leg cross-asset TSMOM  (RT_TS = 12bp flat)

METHOD: each sleeve has gross-return time series. The OLD cost model
is a scalar bps haircut applied per month. The NEW Almgren-Chriss
cost depends on (per-name spread, sigma, participation = trade_$/ADV);
since the sleeves are post-aggregation single series and don't expose
per-name weights, we run a STYLIZED-universe sweep:

  D_PEAD:    300 names (top/bottom decile of top-1500), median ADV $50M,
             sigma 1.5%/day, half-spread 5bp, monthly turnover 40%
  Revision:  similar but slightly larger names, half-spread 4bp
  Carry:     44 futures legs (24 cmdty + 9 FX + 4 US-rt + 7 G10-bnd),
             ADV $500M-$2B, sigma 1%/day, half-spread 2bp, turnover 15%
  TSMOM:     48 futures legs, same liquidity, turnover 25%

For each {sleeve, AUM} pair we compute:
  net_sharpe = gross_sharpe_excl_cost − cost_in_sharpe_units(AUM)
  cost_bps_per_year = turnover_per_yr × Almgren_round_trip_bps(AUM)
  cost_in_sharpe_units = (cost_bps_per_yr / 10000) / vol_annual
  hard_capacity_usd = floor over names of (ADV × max_participation / weight)
  half_life_aum_usd = AUM where impact equals 50% of gross alpha

OUTPUT: a side-by-side table OLD_Sharpe vs NEW_Sharpe @ deploy AUM,
plus a "what changes with AUM" capacity curve.

USAGE:
  python scripts/audit_book_cost_capacity.py
  python scripts/audit_book_cost_capacity.py --aum-scan 10M,50M,100M,500M,1B
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine.portfolio.combined_book import (
    DEFAULT_BOOK_VOL_TARGET,
    DEFAULT_CARRY_RISK_WEIGHT,
    DEFAULT_TARGET_VOL,
    DEFAULT_TSMOM_RISK_WEIGHT,
    RT_CY,
    RT_EQ,
    RT_TS,
    blend_three,
    book_stats,
    build_carry_book,
    build_equity_book,
    build_tsmom_book,
    scale_to_book_vol,
    voltarget,
)
from engine.research.cost_model import almgren_chriss_cost


# -- Stylized universe characteristics per sleeve --------------------------
# Sources: D_PEAD code uses top-1500 by market cap (engine/portfolio/
# dpead_recon.py UNIVERSE_TOP_N=1500, decile=0.1 → ~150 per leg).
# Futures legs from build_carry_book / build_tsmom_book inspection.
# ADV / spread numbers from Frazzini-Israel-Moskowitz 2015 + standard
# CME liquidity stats (futures end of 2024).

@dataclass
class SleeveUniverse:
    name: str
    n_positions: int          # total long+short positions
    median_adv_usd: float     # typical ADV per name
    daily_sigma: float        # typical daily vol per name
    half_spread_bps: float    # one-side spread
    impact_coef: float        # Almgren sqrt-impact multiplier
    monthly_turnover: float   # fraction of book rebalanced per month
    old_cost_bps_per_side: float  # what RT_xx represents
    annualized_vol: float     # for sharpe-units conversion (filled later)


SLEEVE_UNIVERSE = {
    "equity": SleeveUniverse(
        name="equity (D_PEAD + analyst revision)",
        n_positions=300,           # 150 long + 150 short
        median_adv_usd=50_000_000, # top-1500 median ADV
        daily_sigma=0.015,         # 1.5% daily
        half_spread_bps=5.0,
        impact_coef=0.5,
        monthly_turnover=0.40,     # high post-earnings
        old_cost_bps_per_side=30.0,
        annualized_vol=0.0,        # filled below
    ),
    "carry": SleeveUniverse(
        name="carry (4-leg cross-asset)",
        n_positions=44,            # 24 cmdty + 9 FX + 4 US rt + 7 G10 bnd
        median_adv_usd=1_000_000_000,
        daily_sigma=0.010,
        half_spread_bps=2.0,
        impact_coef=0.3,           # futures impact softer than equity
        monthly_turnover=0.15,     # slow carry signal
        old_cost_bps_per_side=12.0,
        annualized_vol=0.0,
    ),
    "tsmom": SleeveUniverse(
        name="tsmom (5-leg cross-asset)",
        n_positions=48,            # 24+9+4+7+4 eq idx
        median_adv_usd=1_000_000_000,
        daily_sigma=0.012,
        half_spread_bps=2.0,
        impact_coef=0.3,
        monthly_turnover=0.25,     # faster than carry
        old_cost_bps_per_side=12.0,
        annualized_vol=0.0,
    ),
}


# -- Cost computation per AUM ----------------------------------------------

# Default stress multiplier — applied to BOTH half-spread and impact-coef in
# stress mode. Conservative bracket per Frazzini-Israel-Moskowitz 2015 (2008
# realized factor cost ~2-3x normal) + Almgren et al 2005 (illiquidity event
# impact rises ~3x). Used for the --stress audit pass.
DEFAULT_STRESS_MULTIPLIER = 2.5


def per_trade_bps_at_aum(u: SleeveUniverse, aum_usd: float,
                              stress_multiplier: float = 1.0) -> float:
    """Round-trip cost per trade in bps at given AUM.

    Per-position dollar = (aum * monthly_turnover) / n_positions
    participation = per_position_$ / median_ADV
    Almgren bps = 2 * (half_spread + impact_coef * sigma * sqrt(participation) * 10000)

    stress_multiplier (>=1): multiplies BOTH half-spread and impact-coef.
    Use 1.0 for normal regime, 2.5 (default) for stress regime per
    FIM 2015 + Almgren 2005 stress observations. The whole bracketed
    cost rises; spreads widen AND impact slope steepens simultaneously
    in real 2008 / 2020-Mar stress events.
    """
    if aum_usd <= 0:
        return 0.0
    per_position_dollar = (aum_usd * u.monthly_turnover) / max(1, u.n_positions)
    participation = per_position_dollar / max(1.0, u.median_adv_usd)
    eff_spread_bps = u.half_spread_bps * stress_multiplier
    eff_impact_coef = u.impact_coef * stress_multiplier
    impact_bps = eff_impact_coef * u.daily_sigma * np.sqrt(participation) * 10_000.0
    return 2.0 * (eff_spread_bps + impact_bps)


def annual_cost_drag_bps(u: SleeveUniverse, aum_usd: float,
                              stress_multiplier: float = 1.0) -> float:
    """Total annual cost drag in bps at the given AUM and regime."""
    rt_bps = per_trade_bps_at_aum(u, aum_usd, stress_multiplier=stress_multiplier)
    return rt_bps * u.monthly_turnover * 12.0


def hard_capacity_per_sleeve(u: SleeveUniverse, max_participation: float = 0.05) -> float:
    """Equal-weighted: capacity = ADV × max_participation × n_positions / monthly_turnover."""
    return (u.median_adv_usd * max_participation * u.n_positions) / max(1e-9, u.monthly_turnover)


# -- Apply NEW cost model to live gross series -----------------------------

def rebuild_gross_sleeve_returns():
    """Reverse-engineer the GROSS (pre-cost) monthly returns for each sleeve
    by adding back the OLD cost haircut that combined_book.py applies."""
    from engine.validation.analyst_revision import build_revision_sleeve_buffered

    # equity: D_PEAD recon-base is daily gross; revision is monthly gross.
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]
    d.index = pd.to_datetime(d.index)
    dp_m_gross = ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1).rename("dp_gross")

    rev_gross, rev_turn = build_revision_sleeve_buffered(
        q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5,
    )

    # carry: build_carry_book deducts 4×RT_CY/10000/12 from carry_gross.
    from engine.portfolio.carry_sleeve import risk_parity_combine
    from engine.validation.crossasset_carry import (
        build_commodity_carry_ls, build_fx_carry, build_rates_carry, build_rates_xc_carry,
    )
    carry_legs = {
        "cmdty":    build_commodity_carry_ls(),
        "fx":       build_fx_carry()[2],
        "rates_us": build_rates_carry()[2],
        "rates_xc": build_rates_xc_carry()[2],
    }
    carry_g = risk_parity_combine(carry_legs).rename("carry_gross")

    # tsmom
    from engine.validation.crossasset_tsmom import build_tsmom_sleeve_returns
    tsmom_g = build_tsmom_sleeve_returns().rename("tsmom_gross")

    return {
        "equity_dp_gross": dp_m_gross,
        "equity_rev_gross": rev_gross,
        "equity_rev_turn": rev_turn,
        "carry_gross": carry_g,
        "tsmom_gross": tsmom_g,
    }


def apply_new_cost(gross_returns: pd.Series, u: SleeveUniverse,
                       aum_usd: float, stress_multiplier: float = 1.0) -> pd.Series:
    """Apply NEW Almgren-Chriss cost as a constant monthly haircut."""
    monthly_cost_bps = per_trade_bps_at_aum(
        u, aum_usd, stress_multiplier=stress_multiplier) * u.monthly_turnover
    return gross_returns - monthly_cost_bps / 10_000.0


def build_book_with_new_cost(gross: dict, aum_usd: float,
                                  stress_multiplier: float = 1.0) -> tuple[pd.Series, dict]:
    """Reconstruct the 3-sleeve book with NEW Almgren cost at given AUM.
    stress_multiplier (1.0 = normal regime, 2.5 = stress) scales both spread
    and impact-coef simultaneously.
    Returns (book_series, per_sleeve_stats_dict)."""
    eq_u = SLEEVE_UNIVERSE["equity"]
    cy_u = SLEEVE_UNIVERSE["carry"]
    ts_u = SLEEVE_UNIVERSE["tsmom"]

    # equity book = vol-inv-weighted (D_PEAD + revision), each with NEW cost
    dp_net = apply_new_cost(gross["equity_dp_gross"], eq_u, aum_usd,
                                stress_multiplier=stress_multiplier).rename("dp")
    eq_rt_new = per_trade_bps_at_aum(eq_u, aum_usd, stress_multiplier=stress_multiplier)
    rev_net = (gross["equity_rev_gross"]
                - gross["equity_rev_turn"] * eq_rt_new / 10_000.0 / 12.0).rename("rev")
    E = pd.concat([dp_net, rev_net], axis=1).dropna()
    vdp = E["dp"].rolling(12).std().shift(1)
    vre = E["rev"].rolling(12).std().shift(1)
    w = (1 / vdp) / (1 / vdp + 1 / vre)
    eq_book = (w * E["dp"] + (1 - w) * E["rev"]).dropna().rename("equity_book")

    cy_book = apply_new_cost(gross["carry_gross"], cy_u, aum_usd,
                                  stress_multiplier=stress_multiplier).rename("carry")
    ts_book = apply_new_cost(gross["tsmom_gross"], ts_u, aum_usd,
                                  stress_multiplier=stress_multiplier).rename("tsmom")

    # vol-target each leg then blend at deployed risk weights
    eq_vt = voltarget(eq_book, DEFAULT_TARGET_VOL)
    cy_vt = voltarget(cy_book, DEFAULT_TARGET_VOL)
    ts_vt = voltarget(ts_book, DEFAULT_TARGET_VOL)
    book = blend_three(eq_vt, cy_vt, ts_vt,
                          DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_TSMOM_RISK_WEIGHT)
    book = scale_to_book_vol(book, DEFAULT_BOOK_VOL_TARGET)

    per_sleeve = {
        "equity": book_stats(eq_book),
        "carry":  book_stats(cy_book),
        "tsmom":  book_stats(ts_book),
        "book":   book_stats(book),
    }
    return book, per_sleeve


# -- Reporting ------------------------------------------------------------

def format_aum(x: float) -> str:
    if x >= 1e9:
        return f"${x/1e9:.1f}B"
    if x >= 1e6:
        return f"${x/1e6:.0f}M"
    return f"${x:,.0f}"


def parse_aum(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    if s.endswith("M"):
        return float(s[:-1]) * 1e6
    if s.endswith("K"):
        return float(s[:-1]) * 1e3
    return float(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aum-scan", default="10M,50M,100M,500M,1B",
                     help="comma-separated AUM values")
    p.add_argument("--max-participation", type=float, default=0.05)
    p.add_argument("--stress-multiplier", type=float, default=DEFAULT_STRESS_MULTIPLIER,
                     help="cost multiplier applied to half-spread AND impact-coef in "
                            "the stress regime (default 2.5 per FIM 2015 + Almgren 2005)")
    p.add_argument("--no-stress", action="store_true",
                     help="skip the stress-regime sweep (normal regime only)")
    args = p.parse_args()

    aums = [parse_aum(s) for s in args.aum_scan.split(",")]

    print("=" * 90)
    print(" LIVE BOOK COST + CAPACITY AUDIT")
    print(" (Almgren-Chriss vs scalar-bps, deployed 3-mechanism config)")
    print("=" * 90)
    print()
    print(f"Deployed config: equity {1-DEFAULT_CARRY_RISK_WEIGHT-DEFAULT_TSMOM_RISK_WEIGHT:.0%} / "
          f"carry {DEFAULT_CARRY_RISK_WEIGHT:.0%} / tsmom {DEFAULT_TSMOM_RISK_WEIGHT:.0%}; "
          f"book vol-target {DEFAULT_BOOK_VOL_TARGET:.0%}")
    print(f"Source: engine.portfolio.combined_book (spec 77 §11+§12 amendments, locked)")
    print()

    # Stylized universe parameters used
    print("-" * 90)
    print(" Stylized universe assumptions per sleeve")
    print("-" * 90)
    print(f"  {'sleeve':<8} {'#pos':>5} {'ADV/name':>10} {'sigma/day':>7} {'h-sprd':>7} "
          f"{'turn/mo':>8} {'OLD bps':>9}")
    for k, u in SLEEVE_UNIVERSE.items():
        print(f"  {k:<8} {u.n_positions:>5d} {format_aum(u.median_adv_usd):>10} "
              f"{u.daily_sigma:>7.1%} {u.half_spread_bps:>6.1f}bp {u.monthly_turnover:>7.0%} "
              f"{u.old_cost_bps_per_side:>7.1f}bp")
    print()

    # OLD cost (currently deployed scalar-bps model) baseline
    print("-" * 90)
    print(" BASELINE — current scalar-bps deployment")
    print("-" * 90)
    eq = build_equity_book()
    cy = build_carry_book()
    ts = build_tsmom_book()
    from engine.portfolio.combined_book import build_combined_book
    book_old = build_combined_book(book_vol_target=DEFAULT_BOOK_VOL_TARGET)
    s_eq, s_cy, s_ts, s_book = (book_stats(eq), book_stats(cy),
                                    book_stats(ts), book_stats(book_old))
    print(f"  equity book:  Sharpe={s_eq['sharpe']:.3f}  ann={s_eq['ann']:>+.3%}  "
          f"vol={s_eq['vol']:.3%}  maxDD={s_eq['maxdd']:>+.3%}  n={s_eq['n']}")
    print(f"  carry book:   Sharpe={s_cy['sharpe']:.3f}  ann={s_cy['ann']:>+.3%}  "
          f"vol={s_cy['vol']:.3%}  maxDD={s_cy['maxdd']:>+.3%}  n={s_cy['n']}")
    print(f"  tsmom book:   Sharpe={s_ts['sharpe']:.3f}  ann={s_ts['ann']:>+.3%}  "
          f"vol={s_ts['vol']:.3%}  maxDD={s_ts['maxdd']:>+.3%}  n={s_ts['n']}")
    print(f"  COMBINED:     Sharpe={s_book['sharpe']:.3f}  ann={s_book['ann']:>+.3%}  "
          f"vol={s_book['vol']:.3%}  maxDD={s_book['maxdd']:>+.3%}  n={s_book['n']}")
    print()

    # Rebuild gross + apply NEW Almgren cost per AUM
    print("-" * 90)
    print(" NEW Almgren-Chriss cost — AUM sweep (deploy-honest Sharpe per AUM)")
    print("-" * 90)
    gross = rebuild_gross_sleeve_returns()

    header = f"  {'AUM':>8} | {'eq Sharpe':>10} {'cy Sharpe':>10} {'ts Sharpe':>10} " \
             f"| {'BOOK Sharpe':>11} {'BOOK ann':>9} | {'eq cost bp/y':>13} {'cy bp/y':>9} {'ts bp/y':>9}"
    print(header)
    print(f"  {'-' * 8} | {'-' * 10} {'-' * 10} {'-' * 10} "
          f"| {'-' * 11} {'-' * 9} | {'-' * 13} {'-' * 9} {'-' * 9}")

    rows_for_summary = []
    for aum in aums:
        book_new, stats = build_book_with_new_cost(gross, aum)
        cost_eq = annual_cost_drag_bps(SLEEVE_UNIVERSE["equity"], aum)
        cost_cy = annual_cost_drag_bps(SLEEVE_UNIVERSE["carry"], aum)
        cost_ts = annual_cost_drag_bps(SLEEVE_UNIVERSE["tsmom"], aum)
        print(f"  {format_aum(aum):>8} | "
              f"{stats['equity']['sharpe']:>10.3f} {stats['carry']['sharpe']:>10.3f} "
              f"{stats['tsmom']['sharpe']:>10.3f} | "
              f"{stats['book']['sharpe']:>11.3f} {stats['book']['ann']:>+9.3%} | "
              f"{cost_eq:>11.1f}bp {cost_cy:>7.1f}bp {cost_ts:>7.1f}bp")
        rows_for_summary.append((aum, stats['book']['sharpe'], stats['book']['ann']))
    print()

    # ── Stress regime sweep ─────────────────────────────────────────────
    stress_rows = []
    if not args.no_stress:
        print("-" * 90)
        print(f" STRESS REGIME (half-spread + impact-coef x {args.stress_multiplier:.1f}) — "
              f"FIM 2015 + Almgren 2005 stress bracket")
        print("-" * 90)
        print(header)
        print(f"  {'-' * 8} | {'-' * 10} {'-' * 10} {'-' * 10} "
              f"| {'-' * 11} {'-' * 9} | {'-' * 13} {'-' * 9} {'-' * 9}")
        for aum in aums:
            _, stats_s = build_book_with_new_cost(
                gross, aum, stress_multiplier=args.stress_multiplier)
            cost_eq_s = annual_cost_drag_bps(
                SLEEVE_UNIVERSE["equity"], aum, stress_multiplier=args.stress_multiplier)
            cost_cy_s = annual_cost_drag_bps(
                SLEEVE_UNIVERSE["carry"], aum, stress_multiplier=args.stress_multiplier)
            cost_ts_s = annual_cost_drag_bps(
                SLEEVE_UNIVERSE["tsmom"], aum, stress_multiplier=args.stress_multiplier)
            print(f"  {format_aum(aum):>8} | "
                  f"{stats_s['equity']['sharpe']:>10.3f} {stats_s['carry']['sharpe']:>10.3f} "
                  f"{stats_s['tsmom']['sharpe']:>10.3f} | "
                  f"{stats_s['book']['sharpe']:>11.3f} {stats_s['book']['ann']:>+9.3%} | "
                  f"{cost_eq_s:>11.1f}bp {cost_cy_s:>7.1f}bp {cost_ts_s:>7.1f}bp")
            stress_rows.append((aum, stats_s['book']['sharpe'], stats_s['book']['ann']))
        print()

    # Hard capacity per sleeve
    print("-" * 90)
    print(f" HARD CAPACITY (max participation = {args.max_participation:.0%}, equal-weight stylized)")
    print("-" * 90)
    for k, u in SLEEVE_UNIVERSE.items():
        hc = hard_capacity_per_sleeve(u, args.max_participation)
        print(f"  {k:<8}: {format_aum(hc):>10}  "
              f"(={u.n_positions} pos × {format_aum(u.median_adv_usd)} ADV × "
              f"{args.max_participation:.0%} / {u.monthly_turnover:.0%} mo turn)")
    print()

    # Summary table — OLD vs NEW-normal vs NEW-stress
    print("=" * 90)
    print(" SUMMARY — OLD vs NEW (normal) vs NEW (stress) Sharpe by AUM")
    print("=" * 90)
    print(f"  OLD baseline (scalar bps):       Sharpe = {s_book['sharpe']:.3f}   ann = {s_book['ann']:+.3%}")
    print()
    if stress_rows:
        print(f"  {'AUM':>6}     {'NEW normal':>12}   {'NEW stress':>12}   {'OLD vs NEW-stress':>20}")
        print(f"  {'-' * 6}     {'-' * 12}   {'-' * 12}   {'-' * 20}")
        for (aum, sh_n, _), (_, sh_s, _) in zip(rows_for_summary, stress_rows):
            old_vs_stress = s_book['sharpe'] - sh_s
            verdict = "OLD safer" if old_vs_stress > 0 else "NEW still wins"
            print(f"  {format_aum(aum):>6}     {sh_n:>12.3f}   {sh_s:>12.3f}   "
                  f"{old_vs_stress:>+10.3f}    [{verdict}]")
        print()
        # Cross-over AUM detection: where does OLD scalar start outperforming
        # NEW-stress?
        crossover = None
        for (aum, sh_n, _), (_, sh_s, _) in zip(rows_for_summary, stress_rows):
            if s_book['sharpe'] > sh_s and crossover is None:
                crossover = aum
        if crossover:
            print(f"  CROSS-OVER: OLD scalar {s_book['sharpe']:.3f} starts beating "
                  f"NEW-stress at AUM >= {format_aum(crossover)}.")
            print(f"  Interpretation: above this AUM, the 30bp scalar buffer pays off")
            print(f"  in 2008/2020-style stress regimes because Almgren impact rises 2-3x.")
        else:
            print(f"  NO CROSS-OVER in scanned range: NEW-stress beats OLD at every "
                  f"AUM up to {format_aum(aums[-1])}.")
            print(f"  Interpretation: even in stress, Almgren-modeled cost stays below")
            print(f"  the 30bp scalar at deploy-band AUM; the scalar is over-conservative.")
    else:
        print(f"  -- NEW Almgren-Chriss --")
        for aum, sh, an in rows_for_summary:
            delta = sh - s_book['sharpe']
            sign = "+" if delta >= 0 else ""
            print(f"  AUM {format_aum(aum):>6}:    Sharpe = {sh:.3f}   ann = {an:+.3%}   "
                  f"delta = {sign}{delta:.3f}")
    print()
    print(" Caveats:")
    print("  -Stylized universe (no per-name weights_panel) — see SLEEVE_UNIVERSE for assumptions.")
    print("  -OLD model is a flat scalar haircut; NEW scales with sqrtAUM.")
    print("  -Sleeve gross returns include their original signal-level cost handling EXCEPT")
    print("    the cost subtraction (which we re-apply with the new model).")
    print("  -Honest haircut requires per-name weights — TODO upgrade dpead_recon to emit weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
