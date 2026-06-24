"""
scripts/audit_path_m_survivorship_crsp.py — Path M survivorship-free re-test via CRSP.

Strategy:
  1. Query CRSP for all ETFs (shrcd=73) that DELISTED during 2014-2023
  2. Identify thematic-style candidates (high-vol, niche)
  3. Add to spec's 34-ticker universe
  4. Re-run Path M with point-in-time alive-ETF universe
  5. Compare to spec baseline

Goal: Resolve survivorship bias concern documented in Path M PASS_INDEPENDENT_PROVISIONAL audit.
"""
from __future__ import annotations
import sys
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from engine.path_m.thematic_momentum_strategy import (
    LOCKED_UNIVERSE_LIST, N_UNIVERSE_LOCKED,
    compute_monthly_momentum, form_long_short_cohorts, compute_strategy_returns,
)


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


def main():
    print("=" * 90)
    print("PATH M SURVIVORSHIP-FREE AUDIT VIA CRSP")
    print("=" * 90)

    from engine.universe_singlename.crsp_loader import _open_wrds_connection, bulk_fetch_crsp_daily_panel

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Query CRSP for ALL ETFs alive 2014-2023 + delisting info
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/5] Querying CRSP msenames for all ETFs (shrcd=73) 2014-2023...")
    conn = _open_wrds_connection()
    try:
        sql = """
        SELECT permno, ticker, comnam, namedt, nameendt, exchcd
        FROM crsp.msenames
        WHERE shrcd = 73
          AND namedt <= '2023-12-31'
          AND (nameendt IS NULL OR nameendt >= '2014-01-01')
          AND exchcd IN (1, 2, 3)
        ORDER BY ticker, namedt
        """
        all_etfs = conn.raw_sql(sql, date_cols=["namedt", "nameendt"])
    finally:
        conn.close()

    print(f"  Total ETF records: {len(all_etfs)}")
    print(f"  Unique permnos: {all_etfs.permno.nunique()}")
    print(f"  Unique tickers: {all_etfs.ticker.nunique()}")

    # Find PERMNOS where last nameendt < 2024-01-01 (truly delisted before window end)
    perm_last = all_etfs.groupby('permno').agg(
        first_namedt=('namedt', 'min'),
        last_nameendt=('nameendt', 'max'),
        first_ticker=('ticker', 'first'),
        last_ticker=('ticker', 'last'),
        first_comnam=('comnam', 'first'),
    ).reset_index()

    perm_last['delisted_in_window'] = perm_last['last_nameendt'].apply(
        lambda d: d < pd.Timestamp('2024-01-01') if pd.notna(d) else False
    )
    delisted = perm_last[perm_last['delisted_in_window']].copy()
    print(f"\n  ETFs delisted before 2024-01-01: {len(delisted)} unique permnos")

    # Filter delisted to "post-2010 listed" (focuses on modern ETF era,
    # excluding ancient/pre-thematic ETF closures)
    delisted['post_2010'] = delisted['first_namedt'] >= pd.Timestamp('2010-01-01')
    delisted_modern = delisted[delisted['post_2010']].copy()
    print(f"  Of these, listed post-2010 (modern ETF era): {len(delisted_modern)}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Fetch CRSP daily prices for delisted modern ETFs
    # ──────────────────────────────────────────────────────────────────────
    print("\n[2/5] Fetching CRSP daily prices for delisted ETFs to identify thematic candidates...")

    # Build query directly using permnos (not ticker) to handle ticker reuse
    delisted_permnos = delisted_modern.permno.unique().tolist()
    print(f"  Querying daily prices for {len(delisted_permnos)} permnos...")

    conn = _open_wrds_connection()
    try:
        permno_list = ",".join(str(p) for p in delisted_permnos)
        sql = f"""
        SELECT permno, date, prc, vol, ret
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '2014-01-01' AND '2023-12-31'
        ORDER BY permno, date
        """
        prices_long = conn.raw_sql(sql, date_cols=["date"])
    finally:
        conn.close()

    print(f"  Fetched {len(prices_long)} daily price observations")

    if prices_long.empty:
        print("  No price data found — exiting")
        return 1

    # Identify thematic-like candidates: high vol AND low volume (niche)
    print("\n[3/5] Identifying thematic-style delisted ETFs (high vol + niche)...")
    prices_long['ret'] = pd.to_numeric(prices_long['ret'], errors='coerce')
    prices_long['vol'] = pd.to_numeric(prices_long['vol'], errors='coerce')
    prices_long['prc'] = pd.to_numeric(prices_long['prc'], errors='coerce').abs()
    prices_long['dollar_vol'] = prices_long['prc'] * prices_long['vol']

    stats = prices_long.groupby('permno').agg(
        n_obs=('ret', 'count'),
        mean_ret=('ret', 'mean'),
        std_ret=('ret', 'std'),
        avg_dollar_vol=('dollar_vol', 'mean'),
        min_date=('date', 'min'),
        max_date=('date', 'max'),
    ).reset_index()
    stats['ann_vol'] = stats['std_ret'] * np.sqrt(252)
    stats = stats.merge(
        delisted_modern[['permno', 'first_ticker', 'last_ticker', 'first_comnam']],
        on='permno',
    )

    # Filter: characteristics consistent with thematic ETF
    # - ≥ 60 trading days (so not single-day artifacts)
    # - annualized vol ≥ 20% (sector + thematic style)
    # - avg daily dollar volume ≤ $50M (niche — broad ETFs are far higher)
    stats['is_thematic_like'] = (
        (stats['n_obs'] >= 60)
        & (stats['ann_vol'] >= 0.20)
        & (stats['avg_dollar_vol'] <= 50_000_000)
    )
    thematic_delisted = stats[stats['is_thematic_like']].copy()

    print(f"  Thematic-style delisted ETFs (vol>20% + niche volume<$50M): {len(thematic_delisted)}")
    print(f"  Top 20 (by ann_vol):")
    cols = ['first_ticker', 'last_ticker', 'first_comnam', 'min_date', 'max_date',
            'n_obs', 'ann_vol', 'avg_dollar_vol']
    print(thematic_delisted.nlargest(20, 'ann_vol')[cols].to_string())

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Build survivorship-corrected universe + prices
    # ──────────────────────────────────────────────────────────────────────
    print("\n[4/5] Building survivorship-corrected universe...")

    # Map permno → ticker (use last_ticker as final identifier)
    delisted_tickers = thematic_delisted.last_ticker.tolist()
    print(f"  Adding {len(delisted_tickers)} delisted thematic tickers to 34 alive set")
    print(f"  Combined universe: {N_UNIVERSE_LOCKED + len(delisted_tickers)} ETFs (point-in-time alive)")

    # Build CRSP prices panel by permno → re-shape to ticker
    # First, get alive ETF prices via yfinance (existing 34 universe + already cached)
    import yfinance as yf
    yfin = yf.download(LOCKED_UNIVERSE_LIST, start="2013-01-01", end="2024-01-01",
                       progress=False, auto_adjust=True)
    yfin_close = yfin["Close"] if isinstance(yfin.columns, pd.MultiIndex) else yfin
    yfin_close = yfin_close.dropna(how='all').ffill()

    # Build CRSP delisted prices into wide format
    crsp_panel = prices_long[prices_long.permno.isin(thematic_delisted.permno)].copy()
    crsp_panel = crsp_panel.merge(
        thematic_delisted[['permno', 'last_ticker']],
        on='permno',
    )
    # If duplicate tickers (rare), suffix with permno
    crsp_panel['unique_ticker'] = crsp_panel['last_ticker']
    duplicated_tickers = crsp_panel.groupby('unique_ticker')['permno'].nunique()
    duplicated_tickers = duplicated_tickers[duplicated_tickers > 1].index
    if len(duplicated_tickers) > 0:
        mask = crsp_panel['unique_ticker'].isin(duplicated_tickers)
        crsp_panel.loc[mask, 'unique_ticker'] = (
            crsp_panel.loc[mask, 'last_ticker'] + '_' + crsp_panel.loc[mask, 'permno'].astype(str)
        )

    crsp_wide = crsp_panel.pivot_table(index='date', columns='unique_ticker', values='prc',
                                        aggfunc='first')
    crsp_wide = crsp_wide.dropna(how='all').ffill()
    print(f"  CRSP delisted panel: {crsp_wide.shape[0]} dates × {crsp_wide.shape[1]} tickers")

    # Combine alive + delisted into unified panel
    # Index alignment: outer join on all dates
    yfin_close.index = pd.to_datetime(yfin_close.index)
    crsp_wide.index = pd.to_datetime(crsp_wide.index)
    combined = pd.concat([yfin_close, crsp_wide], axis=1)
    combined = combined.sort_index().loc['2014-01-01':'2023-12-31']
    print(f"  Combined panel: {combined.shape[0]} dates × {combined.shape[1]} tickers")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Re-run Path M on combined universe
    # ──────────────────────────────────────────────────────────────────────
    print("\n[5/5] Re-running Path M on survivorship-corrected universe...")

    # Strategy with point-in-time available tickers
    mom = compute_monthly_momentum(combined)
    longs, shorts = form_long_short_cohorts(mom)
    strat = compute_strategy_returns(combined, longs, shorts)

    sh_corrected = sharpe(strat.daily_returns)
    nw_corrected = nw_t(strat.daily_returns, 60)
    cum_corrected = float((1 + strat.daily_returns).cumprod().iloc[-1] - 1)
    dd_corrected = max_dd(strat.daily_returns)

    # Baseline (spec 34-ETF only) for comparison
    base_mom = compute_monthly_momentum(yfin_close.loc['2014-01-01':'2023-12-31'])
    base_longs, base_shorts = form_long_short_cohorts(base_mom)
    base_strat = compute_strategy_returns(yfin_close.loc['2014-01-01':'2023-12-31'], base_longs, base_shorts)
    sh_baseline = sharpe(base_strat.daily_returns)
    nw_baseline = nw_t(base_strat.daily_returns, 60)

    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print(f"{'Strategy variant':<55} {'Sharpe':>10} {'NW t':>10}")
    print(f"  {'Baseline (spec 34-ETF locked, alive only)':<55} {sh_baseline:>+10.4f} {nw_baseline:>+10.4f}")
    print(f"  {'Survivorship-corrected ({:d} ETFs)'.format(combined.shape[1]):<55} {sh_corrected:>+10.4f} {nw_corrected:>+10.4f}")
    print(f"  {'Δ (corrected - baseline)':<55} {sh_corrected - sh_baseline:>+10.4f} {nw_corrected - nw_baseline:>+10.4f}")

    # Sub-period
    print("\nSub-period corrected:")
    for label, sl in [('Pre-COVID', slice(None, '2019-12-31')),
                      ('COVID', slice('2020-01-01', '2021-12-31')),
                      ('Post-COVID', slice('2022-01-01', None))]:
        sub_c = strat.daily_returns.loc[sl]
        sub_b = base_strat.daily_returns.loc[sl]
        if len(sub_c) > 30:
            print(f"  {label:<15} corrected Sh={sharpe(sub_c):+.3f}  baseline Sh={sharpe(sub_b):+.3f}")

    print("\n" + "=" * 90)
    print("VERDICT INTERPRETATION")
    print("=" * 90)
    delta = sh_corrected - sh_baseline
    if abs(delta) < 0.05:
        print(f"  Sharpe Δ = {delta:+.3f} (< 0.05): survivorship effect NEGLIGIBLE")
        print(f"  Path M classification: SURVIVORSHIP-ROBUST → upgrade PASS_INDEPENDENT (strict)")
    elif delta < -0.10:
        print(f"  Sharpe Δ = {delta:+.3f} < -0.10: survivorship was INFLATING backtest")
        print(f"  Path M classification: MARGINAL after correction")
    elif delta < -0.05:
        print(f"  Sharpe Δ = {delta:+.3f} ∈ [-0.10, -0.05]: mild survivorship inflation")
        print(f"  Path M classification: PASS_INDEPENDENT_PROVISIONAL stands")
    else:
        print(f"  Sharpe Δ = {delta:+.3f} > 0: corrected Sharpe HIGHER (paradoxical)")
        print(f"  Interpretation: missed delisted ETFs would have been profitable shorts")
        print(f"  Path M classification: SURVIVORSHIP-NEGATIVE-BIAS → PASS_INDEPENDENT strict upgrade")

    return 0


if __name__ == "__main__":
    sys.exit(main())
