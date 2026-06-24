"""
scripts/scout_coval_stafford_fire_sales.py — Coval-Stafford 2007 Mutual Fund Fire Sales scout.

Hypothesis (Coval-Stafford 2007 JFE):
  Mutual funds experiencing extreme outflows must liquidate positions →
  forced selling creates temporary downward pressure on stocks they hold →
  these stocks become oversold → rebound when selling pressure ends.

Strategy:
  - For each stock s at quarter t: compute Mutual Fund Pressure (MFP)
    pressure_s_t = sum over funds i of (normalized_flow_i_t × holding_frac_i_s_{t-1})
  - Long stocks with extreme NEGATIVE pressure (forced-sale victims, bottom decile)
  - Short stocks with extreme POSITIVE pressure (forced-buy hyped, top decile)
  - Hold 1 quarter

Data: WRDS crsp_q_mutualfunds (holdings + fund_flows + monthly_tna + fund_hdr) — FREE academic
Window: 2014-2023 (10y daily)
Universe: top-300 stocks by aggregate mutual fund ownership (mid/large cap with meaningful MF ownership)
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
        print(f"  {name}: too few obs ({len(x)})"); return
    sh = sharpe(x); nw = nw_t(x, 60)
    ret_ann = float(x.mean() * 252) * 100
    vol_ann = float(x.std(ddof=1) * np.sqrt(252)) * 100
    cum = float((1 + x).cumprod().iloc[-1] - 1) * 100
    dd = max_dd(x) * 100
    pre = sharpe(x.loc[:"2019-12-31"])
    cov = sharpe(x.loc["2020-01-01":"2021-12-31"])
    pos = sharpe(x.loc["2022-01-01":])
    all_pos = all(s > 0 for s in [pre, cov, pos] if not np.isnan(s))
    gate = "PASS" if sh >= 0.5 and nw >= 2.0 else "FAIL"
    print(f"  {name}")
    print(f"    Sharpe={sh:+.3f}  NW t={nw:+.3f}  ann_ret={ret_ann:+5.1f}%  vol={vol_ann:5.1f}%  cum={cum:+6.0f}%  DD={dd:+.0f}%")
    print(f"    Pre/COVID/Post: [{pre:+.2f}/{cov:+.2f}/{pos:+.2f}]  AllPos={all_pos}  Gate(SS 0.5/2.0)={gate}")


def main():
    from engine.universe_singlename.crsp_loader import _open_wrds_connection, bulk_fetch_crsp_daily_panel

    print("=" * 90)
    print("COVAL-STAFFORD 2007 FIRE SALES SCOUT")
    print("=" * 90)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Identify active equity mutual funds (exclude index funds)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/6] Identifying active equity mutual funds 2014-2023...")
    conn = _open_wrds_connection()
    try:
        # Active equity funds: index_fund_flag != 'D' (not index), et_flag != 'F' (not ETF)
        # Object code 'EDC' (Equity Domestic Core), 'EDS' (Stock), 'EDG' (Growth), 'EDY' (Yield)
        sql = """
        SELECT DISTINCT crsp_fundno, crsp_portno, fund_name, ticker, mgmt_name,
               index_fund_flag, et_flag, dead_flag, crsp_obj_cd, lipper_asset_cd
        FROM crsp_q_mutualfunds.fund_hdr_hist
        WHERE crsp_portno IS NOT NULL
          AND (index_fund_flag IS NULL OR index_fund_flag != 'D')
          AND (et_flag IS NULL OR et_flag != 'F')
          AND (lipper_asset_cd IN ('EQ') OR lipper_asset_cd IS NULL)
        """
        active_funds = conn.raw_sql(sql)
    except Exception as e:
        # Fallback to fund_hdr if fund_hdr_hist not available
        print(f"  Fallback (fund_hdr_hist error: {e}); trying fund_hdr...")
        conn = _open_wrds_connection()
        sql = """
        SELECT DISTINCT crsp_fundno, crsp_portno, fund_name, ticker,
               index_fund_flag, et_flag, dead_flag
        FROM crsp_q_mutualfunds.fund_hdr
        WHERE crsp_portno IS NOT NULL
          AND (index_fund_flag IS NULL OR index_fund_flag != 'D')
          AND (et_flag IS NULL OR et_flag != 'F')
        """
        active_funds = conn.raw_sql(sql)

    print(f"  Total active equity fund records: {len(active_funds):,}")
    print(f"  Unique crsp_portno: {active_funds['crsp_portno'].nunique():,}")

    active_portnos = active_funds['crsp_portno'].dropna().astype(int).unique().tolist()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Get top-300 stocks by aggregate mutual fund holdings 2014-2023
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[2/6] Querying holdings to find top-300 most-held stocks 2014-2023...")
    portno_list = ",".join(str(p) for p in active_portnos[:5000])  # cap query size
    sql = f"""
    SELECT permno, COUNT(DISTINCT crsp_portno) AS n_funds, SUM(market_val) AS total_mv
    FROM crsp_q_mutualfunds.holdings
    WHERE eff_dt BETWEEN '2014-01-01' AND '2023-12-31'
      AND permno IS NOT NULL
      AND crsp_portno IN ({portno_list})
      AND market_val > 0
    GROUP BY permno
    ORDER BY total_mv DESC
    LIMIT 300
    """
    top_stocks_df = conn.raw_sql(sql)
    top_permnos = top_stocks_df['permno'].astype(int).tolist()
    print(f"  Top-300 stocks identified, median funds-per-stock: {top_stocks_df['n_funds'].median():.0f}")
    print(f"  Top-5 most-held: {top_stocks_df.head(5)[['permno', 'n_funds', 'total_mv']].to_string(index=False)}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Query holdings for top-300 stocks across active funds 2013-2023
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[3/6] Querying full holdings for top-300 stocks × active funds...")
    permno_list = ",".join(str(p) for p in top_permnos)
    sql = f"""
    SELECT crsp_portno, permno, eff_dt, market_val, percent_tna, nbr_shares
    FROM crsp_q_mutualfunds.holdings
    WHERE eff_dt BETWEEN '2013-10-01' AND '2023-12-31'
      AND permno IN ({permno_list})
      AND crsp_portno IN ({portno_list})
    """
    holdings = conn.raw_sql(sql, date_cols=['eff_dt'])
    print(f"  Fetched {len(holdings):,} holdings rows")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Back out IMPLIED flows from monthly_tna_ret_nav (Coval-Stafford 2007 canonical)
    # implied_flow_t = TNA_t - TNA_{t-1} × (1 + mret_t)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[4/6] Querying monthly_tna_ret_nav for active funds...")
    fundno_portno_map = active_funds[['crsp_fundno', 'crsp_portno']].dropna().astype(int).drop_duplicates()
    fundno_set = sorted(set(fundno_portno_map['crsp_fundno'].tolist()))
    print(f"  Active fundno-portno pairs: {len(fundno_portno_map):,}; unique fundnos: {len(fundno_set):,}")
    fundno_to_portno_dict = dict(zip(fundno_portno_map['crsp_fundno'], fundno_portno_map['crsp_portno']))

    tnaret_chunks = []
    for chunk_start in range(0, len(fundno_set), 30000):
        chunk = fundno_set[chunk_start:chunk_start + 30000]
        flist = ",".join(str(f) for f in chunk)
        sql = f"""
        SELECT crsp_fundno, caldt, mtna, mret, mnav
        FROM crsp_q_mutualfunds.monthly_tna_ret_nav
        WHERE caldt BETWEEN '2013-12-01' AND '2023-12-31'
          AND crsp_fundno IN ({flist})
        """
        chunk_df = conn.raw_sql(sql, date_cols=['caldt'])
        tnaret_chunks.append(chunk_df)
    tnaret = pd.concat(tnaret_chunks, ignore_index=True) if tnaret_chunks else pd.DataFrame()
    tnaret['crsp_portno'] = tnaret['crsp_fundno'].astype(int).map(fundno_to_portno_dict)
    tnaret = tnaret.dropna(subset=['crsp_portno', 'mtna', 'mret'])
    tnaret['crsp_portno'] = tnaret['crsp_portno'].astype(int)
    tnaret['mtna'] = pd.to_numeric(tnaret['mtna'], errors='coerce')
    tnaret['mret'] = pd.to_numeric(tnaret['mret'], errors='coerce')
    print(f"  monthly_tna_ret_nav: {len(tnaret):,} rows (post-portno-mapped)")

    # Compute IMPLIED monthly flow per fundno: flow_t = TNA_t - TNA_{t-1} * (1+mret_t)
    tnaret = tnaret.sort_values(['crsp_fundno', 'caldt'])
    tnaret['tna_lag'] = tnaret.groupby('crsp_fundno')['mtna'].shift(1)
    tnaret['implied_flow'] = tnaret['mtna'] - tnaret['tna_lag'] * (1 + tnaret['mret'])
    tnaret['flow_rate'] = tnaret['implied_flow'] / tnaret['tna_lag']
    tnaret = tnaret.dropna(subset=['flow_rate', 'tna_lag'])
    tnaret = tnaret[(tnaret['flow_rate'] > -0.5) & (tnaret['flow_rate'] < 1.0)]  # winsorize
    print(f"  Computed implied flow rates: {len(tnaret):,} fund-month obs")

    # Aggregate monthly flows to quarterly per portno (TNA-weighted across share classes)
    tnaret['quarter'] = pd.PeriodIndex(tnaret['caldt'], freq='Q')
    # Per fundno, sum monthly flows in quarter; then weight by lagged TNA
    fund_monthly = tnaret[['crsp_fundno', 'crsp_portno', 'quarter', 'caldt', 'implied_flow', 'tna_lag']].copy()
    # Quarterly sum of implied flows + quarter-end-lag TNA per fundno
    fund_quart_per_fundno = fund_monthly.groupby(['crsp_fundno', 'crsp_portno', 'quarter'], as_index=False).agg(
        flow_q=('implied_flow', 'sum'),
        tna_q_lag=('tna_lag', 'first'),  # use start-of-quarter TNA as denominator
    )
    # Aggregate across share classes per portno-quarter: sum flows and sum lagged TNA, then divide
    fund_q = fund_quart_per_fundno.groupby(['crsp_portno', 'quarter'], as_index=False).agg(
        flow_q=('flow_q', 'sum'),
        tna_lag=('tna_q_lag', 'sum'),
    )
    fund_q['flow_rate'] = fund_q['flow_q'] / fund_q['tna_lag']
    fund_q = fund_q.dropna(subset=['flow_rate', 'tna_lag'])
    fund_q = fund_q[(fund_q['flow_rate'] > -0.5) & (fund_q['flow_rate'] < 1.0)]
    print(f"  Quarterly fund-flow panel: {len(fund_q):,} portno-quarter obs")

    # Quarterize holdings as Period
    holdings['quarter'] = pd.PeriodIndex(holdings['eff_dt'], freq='Q')
    holdings_q = holdings.sort_values('eff_dt').groupby(
        ['crsp_portno', 'permno', 'quarter'], as_index=False
    ).agg(percent_tna=('percent_tna', 'last'))

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Compute MFP (Mutual Fund Pressure) per stock-quarter
    # ──────────────────────────────────────────────────────────────────────
    print("\n[5/6] Computing per-stock MFP signal (pressure_s_t)...")
    # Holdings at quarter q-1 (lagged), flow at quarter q
    # Use Period arithmetic: holdings.quarter + 1 = the quarter when flow happens
    holdings_q['next_quarter'] = holdings_q['quarter'] + 1
    merged = holdings_q.merge(
        fund_q[['crsp_portno', 'quarter', 'flow_rate']].rename(columns={'quarter': 'next_quarter'}),
        on=['crsp_portno', 'next_quarter'],
    )
    merged['contribution'] = merged['percent_tna'] * merged['flow_rate']

    mfp = merged.groupby(['permno', 'next_quarter'], as_index=False).agg(
        mfp=('contribution', 'sum'),
        n_funds=('crsp_portno', 'nunique'),
    )
    mfp = mfp[mfp['n_funds'] >= 3]
    print(f"  MFP panel: {len(mfp):,} stock-quarter obs; mean n_funds per stock: {mfp['n_funds'].mean():.0f}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: Fetch daily returns for top-300 stocks via CRSP, run strategy
    # ──────────────────────────────────────────────────────────────────────
    print("\n[6/6] Fetching CRSP daily returns + running strategy...")
    sql = f"""
    SELECT permno, date, ret
    FROM crsp.dsf
    WHERE permno IN ({permno_list})
      AND date BETWEEN '2014-01-01' AND '2024-03-31'
    """
    daily = conn.raw_sql(sql, date_cols=['date'])
    conn.close()
    daily['ret'] = pd.to_numeric(daily['ret'], errors='coerce')
    daily['permno'] = daily['permno'].astype(int)
    daily_panel = daily.pivot_table(index='date', columns='permno', values='ret', aggfunc='first')
    print(f"  Daily returns panel: {daily_panel.shape[0]} dates × {daily_panel.shape[1]} stocks")

    # Strategy: each quarter, rank stocks by MFP, long bottom decile / short top decile
    # Hold for 1 quarter (60 trading days)
    strategy_daily = pd.Series(0.0, index=daily_panel.index)

    quarters = sorted(mfp['next_quarter'].unique())
    leg_sizes_l, leg_sizes_s = [], []
    n_valid = 0
    for q in quarters:
        q_data = mfp[mfp['next_quarter'] == q].copy()
        if len(q_data) < 30:
            continue
        q_data['permno'] = q_data['permno'].astype(int)
        q_data = q_data.sort_values('mfp')
        n_decile = max(5, len(q_data) // 10)
        longs = q_data.head(n_decile)['permno'].tolist()
        shorts = q_data.tail(n_decile)['permno'].tolist()
        leg_sizes_l.append(len(longs))
        leg_sizes_s.append(len(shorts))
        n_valid += 1

        # Convert Period to start-of-next-quarter timestamp for entry
        q_start_ts = q.to_timestamp(how='start')
        idx = daily_panel.index.searchsorted(q_start_ts)
        if idx >= len(daily_panel):
            continue
        end_idx = min(idx + 63, len(daily_panel) - 1)
        hold_dates = daily_panel.index[idx:end_idx + 1]
        for d in hold_dates:
            long_ret = daily_panel.loc[d, longs].dropna().mean()
            short_ret = daily_panel.loc[d, shorts].dropna().mean()
            if not (np.isnan(long_ret) or np.isnan(short_ret)):
                strategy_daily.loc[d] = long_ret - short_ret

    print(f"  Valid quarter cohorts: {n_valid}")
    print(f"  Mean leg sizes: long={np.mean(leg_sizes_l):.1f}, short={np.mean(leg_sizes_s):.1f}")

    # ──────────────────────────────────────────────────────────────────────
    # Results
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("COVAL-STAFFORD FIRE SALES SCOUT — RESULTS")
    print("=" * 90)
    summarize("L-S Fire Sales (long extreme outflow pressure / short inflow)", strategy_daily)

    # Long-only fire-sale victims
    long_only = pd.Series(0.0, index=daily_panel.index)
    for q in quarters:
        q_data = mfp[mfp['next_quarter'] == q].copy()
        if len(q_data) < 30: continue
        q_data['permno'] = q_data['permno'].astype(int)
        n_decile = max(5, len(q_data) // 10)
        longs = q_data.sort_values('mfp').head(n_decile)['permno'].tolist()
        q_start_ts = q.to_timestamp(how='start')
        idx = daily_panel.index.searchsorted(q_start_ts)
        if idx >= len(daily_panel): continue
        end_idx = min(idx + 63, len(daily_panel) - 1)
        for d in daily_panel.index[idx:end_idx + 1]:
            r = daily_panel.loc[d, longs].dropna().mean()
            if not np.isnan(r):
                long_only.loc[d] = r
    print()
    summarize("Long-only fire-sale victims (capacity-friendly)", long_only)

    # Universe equal-weight reference
    univ_ew = daily_panel.mean(axis=1)
    print(f"\n  Universe top-300 equal-weight reference: Sharpe={sharpe(univ_ew):+.3f}")

    print("\n" + "=" * 90)
    print("VERDICT GUIDE")
    print("=" * 90)
    print("  Single-stock sleeve gates: Sharpe ≥ 0.5 + NW t ≥ 2.0")
    print("  ETF sleeve gates: Sharpe ≥ 0.4 + NW t ≥ 1.8")
    print("  Sharpe > 0.3 + 3/3 sub-period positive: lock Path N spec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
