"""
scripts/scout_etf_flow_reversal.py — Ben-David-Franzoni-Moussawi 2018 Flow Reversal scout.

Tests whether ETF fund flow creates temporary price pressure that mean-reverts:
  - Hypothesis: top decile inflow ETFs underperform in next 5d (temporary buying pressure)
                bottom decile outflow ETFs outperform (temporary selling pressure)
  - Strategy: weekly L/S — long bottom flow decile, short top flow decile, hold 5d

Universe: top-N most-liquid ETFs by avg (shares × NAV) over 2014-2023
Data: etfg_fund_flow (daily flow + NAV + shares_outstanding) via WRDS academic
Window: 2014-2023 (10y)

If raw Sharpe > 0.3, worth full Path N spec lock.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


def sharpe(x, p=252):
    if len(x) < 30 or x.std(ddof=1) == 0: return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(p))


def nw_t(x, lag=60):
    v = x.dropna().values
    n = len(v)
    if n < lag + 2: return float("nan")
    mu = v.mean(); e = v - mu
    s = (e * e).sum() / n
    for k in range(1, lag + 1):
        gk = (e[k:] * e[:-k]).sum() / n
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else 0.0


def max_dd(x):
    c = (1 + x).cumprod(); rm = c.cummax()
    return float(((c - rm) / rm).min())


def summarize(name, x):
    x = x.dropna()
    if len(x) < 30:
        print(f"  {name}: too few obs ({len(x)})")
        return
    sh = sharpe(x); nw = nw_t(x, 60)
    ret_ann = float(x.mean() * 252) * 100
    vol_ann = float(x.std(ddof=1) * np.sqrt(252)) * 100
    cum = float((1 + x).cumprod().iloc[-1] - 1) * 100
    dd = max_dd(x) * 100
    pre = sharpe(x.loc[:"2019-12-31"])
    cov = sharpe(x.loc["2020-01-01":"2021-12-31"])
    pos = sharpe(x.loc["2022-01-01":])
    all_pos = all(s > 0 for s in [pre, cov, pos] if not np.isnan(s))
    gate = "PASS" if sh >= 0.4 and nw >= 1.8 else "FAIL"
    print(f"  {name}")
    print(f"    Sharpe={sh:+.3f}  NW t={nw:+.3f}  ann_ret={ret_ann:+5.1f}%  vol={vol_ann:5.1f}%  cum={cum:+6.0f}%  DD={dd:+.0f}%")
    print(f"    Sub-period: Pre {pre:+.3f}  COVID {cov:+.3f}  Post {pos:+.3f}  AllPos={all_pos}  Gate={gate}")


def main():
    from engine.universe_singlename.crsp_loader import _open_wrds_connection
    print("Querying etfg_fund_flow for 2014-2023 daily flows...")

    conn = _open_wrds_connection()
    try:
        sql = """
        SELECT as_of_date, composite_ticker, shares_outstanding, nav, fundflow
        FROM etfg_fund_flow.fund_flow
        WHERE as_of_date BETWEEN '2013-01-01' AND '2023-12-31'
        ORDER BY composite_ticker, as_of_date
        """
        df = conn.raw_sql(sql, date_cols=['as_of_date'])
    finally:
        conn.close()

    print(f"Fetched {len(df):,} rows × {df['composite_ticker'].nunique():,} unique tickers")
    df = df.dropna(subset=['composite_ticker', 'nav', 'shares_outstanding'])
    df = df[df['nav'] > 0]

    # Compute AUM = shares × NAV; pick top-N liquid ETFs by avg AUM
    df['aum'] = df['shares_outstanding'] * df['nav']
    avg_aum = df.groupby('composite_ticker')['aum'].mean().sort_values(ascending=False)
    print(f"\nTop 10 by avg AUM:")
    print(avg_aum.head(10).to_string())

    # Universe: top-100 (excludes tiny ETFs with noisy flow)
    UNIVERSE_SIZE = 100
    universe = avg_aum.head(UNIVERSE_SIZE).index.tolist()
    print(f"\nUniverse: top-{UNIVERSE_SIZE} by avg AUM")

    df_u = df[df['composite_ticker'].isin(universe)].copy()
    print(f"Universe data: {len(df_u):,} rows")

    # Pivot to wide format: index=date, cols=ticker
    nav_panel  = df_u.pivot_table(index='as_of_date', columns='composite_ticker',
                                    values='nav', aggfunc='first')
    flow_panel = df_u.pivot_table(index='as_of_date', columns='composite_ticker',
                                    values='fundflow', aggfunc='first')
    aum_panel  = df_u.pivot_table(index='as_of_date', columns='composite_ticker',
                                    values='aum', aggfunc='first')

    nav_panel = nav_panel.sort_index().ffill()
    flow_panel = flow_panel.sort_index().fillna(0)
    aum_panel = aum_panel.sort_index().ffill()
    print(f"NAV panel: {nav_panel.shape[0]} dates × {nav_panel.shape[1]} tickers")

    # Trim to spec window
    nav_panel = nav_panel.loc['2014-01-01':'2023-12-31']
    flow_panel = flow_panel.loc['2014-01-01':'2023-12-31']
    aum_panel = aum_panel.loc['2014-01-01':'2023-12-31']

    # Daily returns from NAV
    daily_ret = nav_panel.pct_change()

    # Flow signal: 5-day rolling flow as fraction of AUM (relative flow magnitude)
    rolling_flow_5d = flow_panel.rolling(window=5, min_periods=3).sum()
    rolling_aum_5d  = aum_panel.rolling(window=5, min_periods=3).mean()
    rel_flow_5d = rolling_flow_5d / rolling_aum_5d  # NaN if AUM 0

    print(f"\nRelative flow signal computed (5d cum flow / 5d avg AUM)")

    # Strategy: weekly rebalance (every Friday)
    # At rebalance, rank ETFs by 5d flow
    # Long bottom decile (outflow), short top decile (inflow)
    # Hold 5 trading days

    # Resample to weekly Fridays for rebalance
    rebal_dates = []
    for d in daily_ret.index:
        if d.weekday() == 4:  # Friday
            rebal_dates.append(d)
    rebal_dates = pd.DatetimeIndex(rebal_dates)
    print(f"Weekly rebalance dates: {len(rebal_dates)}")

    # L-S strategy
    strategy_ret = pd.Series(0.0, index=daily_ret.index)
    long_size_avg = []
    short_size_avg = []
    n_valid_rebals = 0

    for i, rd in enumerate(rebal_dates):
        if rd not in rel_flow_5d.index:
            continue
        sig = rel_flow_5d.loc[rd].dropna()
        if len(sig) < 20:  # need enough tickers for decile
            continue
        # Decile threshold
        n_decile = max(3, len(sig) // 10)
        sorted_sig = sig.sort_values()
        # bottom decile (most outflow, NEGATIVE flow) → LONG
        longs = sorted_sig.head(n_decile).index.tolist()
        # top decile (most inflow, POSITIVE flow) → SHORT
        shorts = sorted_sig.tail(n_decile).index.tolist()

        long_size_avg.append(len(longs))
        short_size_avg.append(len(shorts))
        n_valid_rebals += 1

        # Hold 5 trading days
        if i + 1 < len(rebal_dates):
            next_rd = rebal_dates[i + 1]
        else:
            next_rd = daily_ret.index[-1]
        hold_dates = daily_ret.index[(daily_ret.index > rd) & (daily_ret.index <= next_rd)]
        for d in hold_dates:
            long_ret = daily_ret.loc[d, longs].dropna().mean()
            short_ret = daily_ret.loc[d, shorts].dropna().mean()
            if not (np.isnan(long_ret) or np.isnan(short_ret)):
                strategy_ret.loc[d] = long_ret - short_ret

    print(f"\nValid rebalances: {n_valid_rebals}, mean long={np.mean(long_size_avg):.1f}, "
          f"mean short={np.mean(short_size_avg):.1f}")

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FLOW REVERSAL SCOUT — Ben-David et al 2018")
    print("=" * 90)
    summarize("L-S Flow Reversal (long outflow / short inflow, weekly rebal)", strategy_ret)

    # Also compute reverse direction to confirm hypothesis
    reverse_strategy = -strategy_ret  # long inflow, short outflow (FLOW CHASING)
    print()
    summarize("REVERSE (long inflow / short outflow — flow chasing)", reverse_strategy)

    # Long-only outflow leg
    long_only = pd.Series(0.0, index=daily_ret.index)
    for i, rd in enumerate(rebal_dates):
        if rd not in rel_flow_5d.index:
            continue
        sig = rel_flow_5d.loc[rd].dropna()
        if len(sig) < 20:
            continue
        n_decile = max(3, len(sig) // 10)
        longs = sig.sort_values().head(n_decile).index.tolist()
        next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_ret.index[-1]
        hold_dates = daily_ret.index[(daily_ret.index > rd) & (daily_ret.index <= next_rd)]
        for d in hold_dates:
            r = daily_ret.loc[d, longs].dropna().mean()
            if not np.isnan(r):
                long_only.loc[d] = r
    print()
    summarize("Long-only outflow decile (mean reversion long)", long_only)

    # Excess vs equal-weight universe (control for market beta)
    universe_ew = daily_ret.mean(axis=1)  # equal-weight panel return
    excess = strategy_ret - 0  # L-S is dollar-neutral by construction; no excess needed
    print(f"\n  Universe equal-weight reference: Sharpe={sharpe(universe_ew):+.3f}")

    print("\n" + "=" * 90)
    print("VERDICT GUIDE")
    print("=" * 90)
    print("  Raw scout Sharpe ≥ 0.5 + NW t ≥ 2.0: STRONG → lock Path N spec")
    print("  Raw scout Sharpe 0.3-0.5: MODERATE → consider Path N spec with caveats")
    print("  Raw scout Sharpe < 0.3: WEAK → abandon / try different flow signal definition")
    print("  Reverse > L-S Sharpe: HYPOTHESIS REJECTED (flow chasing works, not reversal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
