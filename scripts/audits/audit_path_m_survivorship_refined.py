"""
scripts/audit_path_m_survivorship_refined.py — Refined Path M survivorship audit.

Improvement over audit_path_m_survivorship_crsp.py:
  - Exclude leveraged/inverse/single-stock derivative ETF sponsors
  - Exclude products with short listing duration (< 1 year — typically fail-fast derivatives)
  - Keep only sponsors known to issue genuine thematic ETFs
  - Visual inspection of final filtered list before re-running Path M

Filter rationale:
  - PROSHARES / DIREXION / GRANITESHARES → leveraged/inverse/single-stock products
  - ACCUSHARES → VIX daily-rebalance
  - INVESTMENT MANAGERS / COLLABORATIVE → white-label single-stock
  - VALKYRIE / ALPHA ARCHITECT → crypto/active strategies, not thematic momentum
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


# Sponsors that issue thematic ETFs (KEEP)
INCLUDE_SPONSORS = [
    'GLOBAL X', 'ARK ', 'FIRST TRUST', 'KRANESHARES', 'VANECK',
    'AMPLIFY', 'INNOVATOR', 'ROUNDHILL', 'ETFMG', 'WISDOMTREE',
    'EXCHANGE TRADED CONCEPTS', 'FACTORSHARES', 'ALPS',
]

# Sponsors that issue leveraged/inverse/single-stock products (EXCLUDE)
EXCLUDE_SPONSORS = [
    'PROSHARES', 'DIREXION', 'GRANITESHARES', 'INVESTMENT MANAGERS',
    'COLLABORATIVE', 'ACCUSHARES', 'VALKYRIE', 'ALPHA ARCHITECT',
]

# Ticker patterns (single-stock products often have these endings)
SINGLE_STOCK_TICKER_PATTERNS = ['UP', 'DN', 'EQ', 'EL', 'PS', 'PT', 'BB']


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


def is_excluded_sponsor(comnam):
    if pd.isna(comnam):
        return True
    cn_upper = comnam.upper()
    return any(ex in cn_upper for ex in EXCLUDE_SPONSORS)


def is_included_sponsor(comnam):
    if pd.isna(comnam):
        return False
    cn_upper = comnam.upper()
    return any(inc in cn_upper for inc in INCLUDE_SPONSORS)


def main():
    print("=" * 90)
    print("PATH M SURVIVORSHIP REFINED AUDIT — exclude leveraged/inverse/single-stock")
    print("=" * 90)

    from engine.universe_singlename.crsp_loader import _open_wrds_connection

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Query CRSP for delisted ETFs 2014-2023 (same as before)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/5] Querying CRSP for delisted ETFs alive 2014-2023...")
    conn = _open_wrds_connection()
    try:
        sql = """
        SELECT permno, ticker, comnam, namedt, nameendt
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
    perm_last['post_2010'] = perm_last['first_namedt'] >= pd.Timestamp('2010-01-01')
    perm_last['duration_days'] = (
        perm_last['last_nameendt'] - perm_last['first_namedt']
    ).dt.days

    # Filter 1: delisted, post-2010
    f1 = perm_last[perm_last['delisted_in_window'] & perm_last['post_2010']].copy()
    print(f"  Total delisted post-2010 ETFs: {len(f1)}")

    # Filter 2: duration ≥ 365 days (excludes ultra-short-lived fail-fast products)
    f2 = f1[f1['duration_days'] >= 365].copy()
    print(f"  Of those, ≥ 1 year listed: {len(f2)}")

    # Filter 3: NOT in excluded sponsor list
    f2['is_excluded'] = f2['first_comnam'].apply(is_excluded_sponsor)
    f3 = f2[~f2['is_excluded']].copy()
    print(f"  Of those, NOT leveraged/inverse/single-stock sponsor: {len(f3)}")

    # Filter 4: IN included sponsor list (genuine thematic ETF sponsors)
    f3['is_included'] = f3['first_comnam'].apply(is_included_sponsor)
    f4 = f3[f3['is_included']].copy()
    print(f"  Of those, IN known thematic sponsor list: {len(f4)}")

    print("\n[2/5] Final filtered delisted thematic ETF candidates:")
    print(f"{'ticker':<8} {'sponsor':<35} {'listed':<12} {'closed':<12} {'days'}")
    for _, row in f4.iterrows():
        print(f"  {row.last_ticker:<8} {str(row.first_comnam)[:33]:<35} "
              f"{str(row.first_namedt)[:10]:<12} {str(row.last_nameendt)[:10]:<12} "
              f"{int(row.duration_days)}")

    if len(f4) == 0:
        print("\n  No refined thematic delisted ETFs found.")
        print("  → Survivorship bias for genuine thematic ETF universe is MINIMAL")
        return 0

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Get prices for refined candidates + include in Path M re-run
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[3/5] Fetching CRSP daily prices for {len(f4)} refined candidates...")
    refined_permnos = f4.permno.unique().tolist()

    conn = _open_wrds_connection()
    try:
        permno_list = ",".join(str(int(p)) for p in refined_permnos)
        sql = f"""
        SELECT permno, date, prc, ret, vol, cfacpr
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '2014-01-01' AND '2023-12-31'
        ORDER BY permno, date
        """
        prices_long = conn.raw_sql(sql, date_cols=["date"])
    finally:
        conn.close()

    if prices_long.empty:
        print("  No price data — exiting")
        return 1
    print(f"  Fetched {len(prices_long)} daily price observations")

    # Apply cfacpr adjustment for splits
    prices_long['prc_adj'] = (
        pd.to_numeric(prices_long['prc'], errors='coerce').abs() *
        pd.to_numeric(prices_long['cfacpr'], errors='coerce')
    )

    # Map permno -> ticker (use last_ticker)
    prices_long = prices_long.merge(f4[['permno', 'last_ticker']], on='permno')
    prices_long['unique_ticker'] = prices_long['last_ticker']
    dups = prices_long.groupby('unique_ticker')['permno'].nunique()
    dup_tickers = dups[dups > 1].index
    if len(dup_tickers) > 0:
        mask = prices_long['unique_ticker'].isin(dup_tickers)
        prices_long.loc[mask, 'unique_ticker'] = (
            prices_long.loc[mask, 'last_ticker'] + '_' + prices_long.loc[mask, 'permno'].astype(str)
        )

    crsp_wide = prices_long.pivot_table(index='date', columns='unique_ticker', values='prc_adj',
                                         aggfunc='first')
    crsp_wide.index = pd.to_datetime(crsp_wide.index)
    crsp_wide = crsp_wide.sort_index().ffill()
    print(f"  CRSP delisted refined panel: {crsp_wide.shape[0]} dates × {crsp_wide.shape[1]} tickers")

    # Get alive ETF prices via yfinance
    print(f"\n[4/5] Fetching alive 34-ticker spec universe via yfinance...")
    import yfinance as yf
    yfin = yf.download(LOCKED_UNIVERSE_LIST, start="2013-01-01", end="2024-01-01",
                       progress=False, auto_adjust=True)
    yfin_close = yfin["Close"] if isinstance(yfin.columns, pd.MultiIndex) else yfin
    yfin_close = yfin_close.dropna(how='all').ffill()
    yfin_close.index = pd.to_datetime(yfin_close.index)

    # Combine
    combined = pd.concat([yfin_close, crsp_wide], axis=1)
    combined = combined.sort_index().loc['2014-01-01':'2023-12-31']
    print(f"  Combined refined universe: {combined.shape[0]} dates × {combined.shape[1]} tickers")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Re-run Path M on refined universe
    # ──────────────────────────────────────────────────────────────────────
    print("\n[5/5] Re-running Path M on refined survivorship-corrected universe...")
    mom = compute_monthly_momentum(combined)
    longs, shorts = form_long_short_cohorts(mom)
    strat = compute_strategy_returns(combined, longs, shorts)

    sh_refined = sharpe(strat.daily_returns)
    nw_refined = nw_t(strat.daily_returns, 60)

    # Baseline
    base_mom = compute_monthly_momentum(yfin_close.loc['2014-01-01':'2023-12-31'])
    base_longs, base_shorts = form_long_short_cohorts(base_mom)
    base_strat = compute_strategy_returns(yfin_close.loc['2014-01-01':'2023-12-31'], base_longs, base_shorts)
    sh_baseline = sharpe(base_strat.daily_returns)
    nw_baseline = nw_t(base_strat.daily_returns, 60)

    print("\n" + "=" * 90)
    print("RESULTS — REFINED SURVIVORSHIP AUDIT")
    print("=" * 90)
    print(f"{'Strategy':<55} {'Sharpe':>10} {'NW t':>10}")
    print(f"  {'Baseline (34 alive, spec locked)':<55} {sh_baseline:>+10.4f} {nw_baseline:>+10.4f}")
    print(f"  {'Refined surv-corrected ({:d} thematic-genuine)'.format(combined.shape[1]):<55} {sh_refined:>+10.4f} {nw_refined:>+10.4f}")
    delta = sh_refined - sh_baseline
    print(f"  {'Δ (corrected - baseline)':<55} {delta:>+10.4f} {nw_refined - nw_baseline:>+10.4f}")

    # Sub-period
    print("\nSub-period:")
    for label, sl in [('Pre-COVID', slice(None, '2019-12-31')),
                      ('COVID', slice('2020-01-01', '2021-12-31')),
                      ('Post-COVID', slice('2022-01-01', None))]:
        sub_r = strat.daily_returns.loc[sl]
        sub_b = base_strat.daily_returns.loc[sl]
        if len(sub_r) > 30:
            print(f"  {label:<15} refined Sh={sharpe(sub_r):+.3f}  baseline Sh={sharpe(sub_b):+.3f}")

    print("\n" + "=" * 90)
    print("FINAL VERDICT INTERPRETATION")
    print("=" * 90)
    if abs(delta) < 0.05:
        print(f"  Δ = {delta:+.3f} (< 0.05): survivorship effect NEGLIGIBLE for refined thematic universe")
        print(f"  Path M classification: SURVIVORSHIP-ROBUST → upgrade PASS_INDEPENDENT (strict)")
    elif delta < -0.20:
        print(f"  Δ = {delta:+.3f} < -0.20: SUBSTANTIAL survivorship inflation")
        print(f"  Path M classification: MARGINAL or FAIL after correction")
    elif delta < -0.10:
        print(f"  Δ = {delta:+.3f} ∈ [-0.20, -0.10]: meaningful survivorship inflation")
        print(f"  Path M classification: MARGINAL_PROVISIONAL")
    elif delta < -0.05:
        print(f"  Δ = {delta:+.3f} ∈ [-0.10, -0.05]: mild survivorship inflation")
        print(f"  Path M classification: PASS_INDEPENDENT_PROVISIONAL stands (small caveat)")
    else:
        print(f"  Δ = {delta:+.3f}: refined survivorship neutral or positive")
        print(f"  Path M classification: SURVIVORSHIP-ROBUST → PASS upgrade")

    return 0


if __name__ == "__main__":
    sys.exit(main())
