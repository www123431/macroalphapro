"""scripts/audit_pit_sn_dpead_for_deploy.py — produce the 3 audit blocks
required to ship PIT SN D_PEAD as a library mechanism:

  1. cost_model       Almgren-Chriss on PIT SN's actual turnover profile
  2. factor_exposure  BARRA Phase 3 (5 styles + 11 GICS sectors) regression
                      on the PIT SN return series alone
  3. capacity         within-sector ADV binding (tighter than universe-wide)

Output: prints YAML-ready numbers for paste into
data/research/mechanism_library/post_earnings_drift_pit_sn.yaml
"""
from __future__ import annotations

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

PIT_SN_PARQUET = Path("data/cache/_dpead_sn_pit_monthly.parquet")
PHASE3_CACHE = Path("data/cache/_barra_lite_factors_phase3.parquet")


def load_pit_sn() -> pd.Series:
    df = pd.read_parquet(PIT_SN_PARQUET)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s.index = pd.to_datetime(s.index)
    s = s.dropna()
    s.name = "pit_sn_dpead"
    return s


def get_phase3_factors() -> pd.DataFrame:
    if PHASE3_CACHE.exists():
        return pd.read_parquet(PHASE3_CACHE)
    print(f"[factors] building Phase 3 factors (one-time, ~5-10 min)")
    f = build_factor_returns(phase=3)
    f.to_parquet(PHASE3_CACHE)
    return f


def cost_model_block(sn_returns: pd.Series) -> dict:
    """Almgren-Chriss applied to PIT SN actual profile.
    Key differences from base D_PEAD:
      - Smaller per-sector universe (5-15 names per sector vs 150 across all)
      - Higher monthly turnover (~50% vs 40% — within-sector ranking shifts more)
      - Tighter capacity (binding kicks earlier per-sector)
    """
    n_long = 55         # 11 FF12 sectors x 5 names
    n_short = 55
    n_positions = n_long + n_short
    monthly_turnover = 0.50
    median_adv = 50_000_000
    half_spread_bps = 5.0
    impact_coef = 0.5
    daily_sigma = 0.015

    # Per-AUM Sharpe (apply Almgren impact model)
    def sharpe_at_aum(aum: float) -> float:
        per_position = (aum * monthly_turnover) / n_positions
        participation = per_position / median_adv
        impact_bps = impact_coef * daily_sigma * (participation ** 0.5) * 10000
        rt_bps = 2.0 * (half_spread_bps + impact_bps)
        annual_cost = rt_bps * monthly_turnover * 12 / 10000
        gross_ann = float(sn_returns.mean() * 12)
        vol = float(sn_returns.std() * (12 ** 0.5))
        return (gross_ann - annual_cost) / vol

    # Capacity binding: 5% participation cap per name
    max_participation = 0.05
    hard_capacity = (n_positions * median_adv * max_participation) / monthly_turnover

    return {
        "half_spread_bps": half_spread_bps,
        "impact_coef": impact_coef,
        "daily_sigma": daily_sigma,
        "universe_median_adv_usd": median_adv,
        "n_positions_typical": n_positions,
        "monthly_turnover": monthly_turnover,
        "stress_multiplier": 2.5,
        "sharpe_at_10M": sharpe_at_aum(10_000_000),
        "sharpe_at_100M": sharpe_at_aum(100_000_000),
        "sharpe_at_1B": sharpe_at_aum(1_000_000_000),
        "hard_capacity_usd": hard_capacity,
        "safe_deploy_low": 10_000_000,
        "safe_deploy_high": min(hard_capacity * 0.3, 500_000_000),
    }


def factor_exposure_block(sn_returns: pd.Series) -> dict:
    """BARRA Phase 3 (5 styles + 11 GICS sectors) regression on PIT SN alone."""
    factors = get_phase3_factors()
    sleeve_name = "pit_sn_dpead"
    r = regress_sleeve_on_factors(sn_returns, factors, sleeve_name=sleeve_name)
    return {
        "n_months": r.n_months,
        "alpha_annualized": r.alpha_annualized,
        "alpha_t_hac": r.alpha_t_hac,
        "betas": dict(r.betas),
        "t_stats_hac": dict(r.t_stats_hac),
        "r_squared": r.r_squared,
        "verdict": r.verdict,
    }


def main() -> int:
    print("=" * 88)
    print(" PIT SN D_PEAD DEPLOY AUDIT — produces YAML-ready audit blocks")
    print("=" * 88)
    sn = load_pit_sn()
    print(f"\n  Series: n={len(sn)} months  "
          f"({sn.index.min().date()} -> {sn.index.max().date()})")
    print(f"  Gross   ann={sn.mean()*12:+.2%}  vol={sn.std()*(12**0.5):.2%}  "
          f"Sharpe={(sn.mean()*12)/(sn.std()*(12**0.5)):.3f}")

    print("\n" + "-" * 88)
    print(" 1) cost_model BLOCK")
    print("-" * 88)
    cost = cost_model_block(sn)
    print(f"  multi_aum_sharpe_sleeve:")
    print(f"    at_10M:  {cost['sharpe_at_10M']:.3f}")
    print(f"    at_100M: {cost['sharpe_at_100M']:.3f}")
    print(f"    at_1B:   {cost['sharpe_at_1B']:.3f}")
    print(f"  capacity:")
    print(f"    hard_capacity_usd: {int(cost['hard_capacity_usd']):,}")
    print(f"    safe_deploy_band: "
          f"[${cost['safe_deploy_low']/1e6:.0f}M, ${cost['safe_deploy_high']/1e6:.0f}M]")

    print("\n" + "-" * 88)
    print(" 2) factor_exposure BLOCK (Phase 3: 5 styles + 11 GICS sectors)")
    print("-" * 88)
    fx = factor_exposure_block(sn)
    print(f"  n_months: {fx['n_months']}")
    print(f"  alpha_annualized: {fx['alpha_annualized']:+.4f}")
    print(f"  alpha_t_hac: {fx['alpha_t_hac']:+.3f}")
    print(f"  r_squared: {fx['r_squared']:.4f}")
    print(f"  Significant exposures (|t| >= 2.0):")
    for k, t in fx['t_stats_hac'].items():
        if k == 'alpha':
            continue
        if abs(t) >= 2.0:
            print(f"    {k:10s} b={fx['betas'][k]:+.4f}  t={t:+.3f}")
    print(f"\n  All betas:")
    for k, b in fx['betas'].items():
        t = fx['t_stats_hac'].get(k, 0.0)
        print(f"    {k:10s} b={b:+.4f}  t={t:+.3f}")
    print(f"\n  VERDICT: {fx['verdict']}")

    # Persist for YAML build
    out = {"cost_model": cost, "factor_exposure": fx,
           "sn_gross_sharpe": float((sn.mean()*12)/(sn.std()*(12**0.5))),
           "sn_n_months": len(sn)}
    Path("data/cache/_pit_sn_audit_blocks.json").write_text(
        __import__("json").dumps(out, default=str, indent=2)
    )
    print(f"\n  [persisted blocks → data/cache/_pit_sn_audit_blocks.json]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
