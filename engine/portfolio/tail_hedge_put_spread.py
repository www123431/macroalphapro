"""engine/portfolio/tail_hedge_put_spread.py — Path C: SPX put-spread
tail hedge sleeve.

Strategy:
  At each month-end, buy a 30-day SPX put at delta=-25 (~5% OTM) and
  sell a 30-day put at delta=-10 (~10% OTM). Hold to expiration. Roll
  to new spread at next month-end. Notional is 5% of book NAV.

Academic basis:
  Israelov (2017) "Pathetic Protection" (AQR) — put SPREADS reduce
    long-put carrying cost (negative theta drag) by ~60% while
    preserving protection in the 5%-15% drawdown band.
  Bondarenko (2014) "Why Are Put Options So Expensive?" — outright
    OTM puts trade rich; spreads finance the long leg via the
    over-rich short leg.
  Israelov-Nielsen (2015) "Still Not Cheap" — naked OTM puts run
    2-4% annual drag, ruling them out for sustained hedging.

Methodology (deliberately simple for sanity check):
  - Enter at month-end (last trading day with skew data)
  - Hold to expiry (~30 calendar days, approximately next month-end)
  - Payoff at expiry uses observed SPX spot at expiry date
  - P&L = expiry_payoff - entry_cost
  - Normalize to monthly return on 5% notional basis
  - Roll to next spread at expiry date

Comparison target: engine.portfolio.mom_hedge_overlay (MTUM-short
current insurance sleeve).

Pre-commit acceptance criteria (Path C doctrine):
  D1. crisis-period (>= 5% SPX drawdown) PnL > mom_hedge in same period
  D2. annualized drag <= mom_hedge drag (currently ~-0.50% pa)
  D3. cosine with book lower than mom_hedge cosine
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.research.options.bs_pricer import bs_price
from engine.research.options.skew_surface import SkewSurfaceLoader

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_PATH = REPO_ROOT / "data" / "cache" / "_tail_hedge_put_spread_monthly.parquet"

# Tunable parameters
SPX_SECID = 108105
LONG_DELTA = -25
SHORT_DELTA = -10
MATURITY_DAYS = 30
NOTIONAL_PCT = 0.05    # 5% of book NAV
RISK_FREE = 0.03       # approximate USD risk-free over the window
DIVIDEND_YIELD = 0.018 # SPX historical dividend yield
TRANSACTION_COST_BP = 5.0    # one-way exec friction per leg, bps of notional


def build_put_spread_monthly_returns(
    long_delta: int = LONG_DELTA,
    short_delta: int = SHORT_DELTA,
    maturity_days: int = MATURITY_DAYS,
    notional_pct: float = NOTIONAL_PCT,
) -> pd.Series:
    """Build monthly return series for the put-spread hedge sleeve.

    Each row = monthly P&L of the put spread normalized to book NAV.
    """
    loader = SkewSurfaceLoader.from_cache()
    full = loader._load_full()

    # Get ATM strike+IV time series for spot approximation
    atm = full[(full["secid"] == SPX_SECID) & (full["cp_flag"] == "C")
               & (full["delta"] == 50) & (full["days"] == maturity_days)].copy()
    atm = atm.sort_values("date").set_index("date")
    # Spot approximation: strike at delta=50 is ~1-2% above spot for 30d
    # ATM call (forward adjustment). For backtest fidelity we use the
    # observed strike as spot proxy and accept small bias.
    atm["spot_approx"] = atm["impl_strike"]

    # OTM put surfaces
    put_long = full[(full["secid"] == SPX_SECID) & (full["cp_flag"] == "P")
                    & (full["delta"] == long_delta)
                    & (full["days"] == maturity_days)].copy()
    put_long = put_long.sort_values("date").set_index("date")[
        ["impl_strike", "impl_volatility"]
    ].rename(columns={"impl_strike": "K_long", "impl_volatility": "iv_long"})

    put_short = full[(full["secid"] == SPX_SECID) & (full["cp_flag"] == "P")
                     & (full["delta"] == short_delta)
                     & (full["days"] == maturity_days)].copy()
    put_short = put_short.sort_values("date").set_index("date")[
        ["impl_strike", "impl_volatility"]
    ].rename(columns={"impl_strike": "K_short", "impl_volatility": "iv_short"})

    # Join: each entry date has spot + 2 strike+IV pairs
    joined = atm[["spot_approx"]].join(put_long, how="inner") \
                                  .join(put_short, how="inner")
    joined = joined.dropna()
    if joined.empty:
        raise RuntimeError("no overlap in spot/long/short data")

    # Iterate over MONTH-END dates as entry points
    joined["month_end"] = joined.index.to_period("M").to_timestamp("M")
    # Last trading day per month as entry
    entries = joined.groupby("month_end").tail(1)
    entries = entries[entries.index >= entries.index.min()]

    rows = []
    T = maturity_days / 365.0
    for entry_date, row in entries.iterrows():
        S0 = float(row["spot_approx"])
        K_long = float(row["K_long"])
        K_short = float(row["K_short"])
        iv_long = float(row["iv_long"])
        iv_short = float(row["iv_short"])
        # Entry cost (BS prices the puts at trade time)
        try:
            price_long = bs_price(S=S0, K=K_long, T=T, r=RISK_FREE,
                                  q=DIVIDEND_YIELD, sigma=iv_long, cp="P")
            price_short = bs_price(S=S0, K=K_short, T=T, r=RISK_FREE,
                                   q=DIVIDEND_YIELD, sigma=iv_short, cp="P")
        except Exception as exc:
            logger.warning(f"BS price fail at {entry_date.date()}: {exc}")
            continue
        spread_cost = price_long - price_short

        # Expiry date ≈ entry + maturity_days (find nearest trading day)
        target_expiry = entry_date + pd.Timedelta(days=maturity_days)
        # Find first date in joined index >= target_expiry
        future = joined.index[joined.index >= target_expiry]
        if len(future) == 0:
            continue
        expiry_date = future[0]
        S_T = float(joined.loc[expiry_date, "spot_approx"])

        # Expiry payoff
        long_payoff = max(K_long - S_T, 0.0)
        short_payoff = -max(K_short - S_T, 0.0)
        spread_payoff = long_payoff + short_payoff

        # Transaction cost — 2 sides × cost per round-trip ≈ 4 × COST_BP/10000
        tx_cost = 4.0 * TRANSACTION_COST_BP / 10000.0 * spread_cost

        # P&L per $1 of spread cost
        pnl_per_dollar = (spread_payoff - spread_cost - tx_cost) / max(S0, 1e-6)

        # Notional-scaled monthly return: position size = notional_pct of NAV
        # Each $1 of spread cost protects $K_long worth of S&P; we hold
        # notional_pct * NAV of underlying-equivalent protection. Conversion:
        # spread_pct_of_spot = spread_cost / S0
        # n_spreads = (notional_pct * NAV) / S0
        # monthly_pnl_book_pct = n_spreads * spread_payoff_per_contract / NAV
        #                       = (notional_pct / S0) * spread_payoff
        spread_payoff_pct = (spread_payoff - spread_cost - tx_cost) / S0
        monthly_book_return = notional_pct * spread_payoff_pct

        rows.append({
            "month_end": entry_date,
            "spot_entry": S0,
            "spot_expiry": S_T,
            "spx_return": (S_T - S0) / S0,
            "K_long": K_long,
            "K_short": K_short,
            "iv_long": iv_long,
            "iv_short": iv_short,
            "spread_cost_$": spread_cost,
            "spread_payoff_$": spread_payoff,
            "net_pnl_$": spread_payoff - spread_cost - tx_cost,
            "net_pnl_pct_spot": spread_payoff_pct,
            "monthly_book_return": monthly_book_return,
        })

    df = pd.DataFrame(rows).set_index("month_end").sort_index()
    monthly = df["monthly_book_return"].rename("put_spread_hedge")
    df.to_parquet(_OUT_PATH.with_suffix(".diag.parquet"))
    monthly.to_frame("put_spread_hedge").to_parquet(_OUT_PATH)
    return monthly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = build_put_spread_monthly_returns()
    print("=" * 70)
    print(f" Put-spread tail hedge: {len(m)} months")
    print(f"   ann return: {m.mean()*12:+.4f}  ann vol: {m.std()*(12**0.5):.4f}")
    print(f"   Sharpe:     {(m.mean()*12)/(m.std()*(12**0.5)):+.3f}")
    print(f"   max month:  {m.max():+.4f}  min month: {m.min():+.4f}")
    print(f"   win rate:   {(m > 0).mean():.1%}")
    print(f"   saved to:   {_OUT_PATH}")
