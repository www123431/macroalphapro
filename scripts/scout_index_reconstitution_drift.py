"""
scripts/scout_index_reconstitution_drift.py — Chen-Noronha-Singal 2004 Index Reconstitution Drift scout.

Hypothesis (Chen-Noronha-Singal 2004 *JF*):
  When S&P announces add/delete to S&P 500:
    - Index funds MUST rebalance on effective date
    - Anticipatory buying of adds → drift UP T-N to T0
    - Anticipatory selling of drops → drift DOWN T-N to T0
    - Possible post-effective reversal T+1 to T+10

Strategy variants tested:
  - Pre-event drift: long adds, short drops from T-5 to T-1 (announcement → effective)
  - Post-event reversal: short adds, long drops from T+1 to T+5
  - Combined L-S adds/drops with event windows

Data: CRSP crsp.msp500list (membership history, FREE academic) + crsp.dsf (daily returns)
Window: 2014-2023 (10y)
Assumption: announcement = T-5 trading days before effective (typical S&P notification)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

# Window assumption: announcement to effective = 5 trading days
ANNOUNCEMENT_LEAD_DAYS = 5
POST_EVENT_WINDOW = 10  # trading days post-effective to monitor reversal


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
    cum = float((1 + x).cumprod().iloc[-1] - 1) * 100
    dd = max_dd(x) * 100
    pre = sharpe(x.loc[:"2019-12-31"])
    cov = sharpe(x.loc["2020-01-01":"2021-12-31"])
    pos = sharpe(x.loc["2022-01-01":])
    all_pos = all(s > 0 for s in [pre, cov, pos] if not np.isnan(s))
    gate = "PASS" if sh >= 0.4 and nw >= 1.8 else "FAIL"
    print(f"  {name}")
    print(f"    Sharpe={sh:+.3f}  NW t={nw:+.3f}  ann_ret={ret_ann:+5.1f}%  cum={cum:+6.0f}%  DD={dd:+.0f}%")
    print(f"    Pre/COVID/Post: [{pre:+.2f}/{cov:+.2f}/{pos:+.2f}]  AllPos={all_pos}  Gate(ETF 0.4/1.8)={gate}")


def main():
    print("=" * 90)
    print("INDEX RECONSTITUTION DRIFT SCOUT — Chen-Noronha-Singal 2004")
    print("=" * 90)

    from engine.universe_singlename.crsp_loader import _open_wrds_connection

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Query CRSP msp500list for add/drop events 2014-2023
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/4] Querying CRSP S&P 500 membership changes 2014-2023...")
    conn = _open_wrds_connection()
    try:
        sql = """
        SELECT permno, start, ending
        FROM crsp.msp500list
        WHERE (start BETWEEN '2014-01-01' AND '2023-12-31')
           OR (ending BETWEEN '2014-01-01' AND '2023-12-31')
        """
        events = conn.raw_sql(sql, date_cols=['start', 'ending'])

        adds = events[(events['start'] >= '2014-01-01') & (events['start'] <= '2023-12-31')].copy()
        adds['event_type'] = 'ADD'
        adds['effective_date'] = adds['start']

        drops = events[(events['ending'] >= '2014-01-01') & (events['ending'] <= '2023-12-31')].copy()
        drops['event_type'] = 'DROP'
        drops['effective_date'] = drops['ending']

        print(f"  Add events: {len(adds)}, Drop events: {len(drops)}")
        all_events = pd.concat([adds[['permno', 'effective_date', 'event_type']],
                                drops[['permno', 'effective_date', 'event_type']]])
        all_permnos = sorted(all_events['permno'].unique().astype(int).tolist())
        print(f"  Unique permnos: {len(all_permnos)}")

        # ──────────────────────────────────────────────────────────────────
        # STEP 2: Fetch CRSP daily returns ±25 days around each event
        # ──────────────────────────────────────────────────────────────────
        print(f"\n[2/4] Fetching CRSP daily returns 2013-2024 for affected stocks...")
        permno_list = ",".join(str(p) for p in all_permnos)
        sql = f"""
        SELECT permno, date, ret
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '2013-06-01' AND '2024-03-31'
        """
        daily = conn.raw_sql(sql, date_cols=['date'])
    finally:
        conn.close()

    daily['ret'] = pd.to_numeric(daily['ret'], errors='coerce')
    daily['permno'] = daily['permno'].astype(int)
    print(f"  Fetched {len(daily):,} daily return rows")

    # Daily returns panel
    panel = daily.pivot_table(index='date', columns='permno', values='ret', aggfunc='first')
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    print(f"  Returns panel: {panel.shape[0]} dates × {panel.shape[1]} permnos")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Event study — compute drift around each event
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Running event study (T-5 to T+10 around effective date)...")

    all_events_clean = all_events.copy()
    all_events_clean['effective_date'] = pd.to_datetime(all_events_clean['effective_date'])
    all_events_clean['permno'] = all_events_clean['permno'].astype(int)

    event_returns = []  # rows: event-level event-time returns
    for _, evt in all_events_clean.iterrows():
        eff = evt['effective_date']
        permno = evt['permno']
        ev_type = evt['event_type']
        if permno not in panel.columns:
            continue
        # Find effective date in panel (or nearest prior trading day)
        idx = panel.index.searchsorted(eff)
        if idx >= len(panel) or idx < ANNOUNCEMENT_LEAD_DAYS:
            continue
        # Event window: T-5 ... T0 ... T+10
        for offset in range(-ANNOUNCEMENT_LEAD_DAYS, POST_EVENT_WINDOW + 1):
            t_idx = idx + offset
            if t_idx < 0 or t_idx >= len(panel):
                continue
            ret_val = panel.iloc[t_idx][permno]
            if pd.notna(ret_val):
                event_returns.append({
                    'permno': permno,
                    'effective_date': eff,
                    'event_type': ev_type,
                    'offset': offset,
                    'ret': float(ret_val),
                })

    er = pd.DataFrame(event_returns)
    print(f"  Event-time observations: {len(er):,}")

    # Aggregate by event-type + offset
    print(f"\n  Mean return by event-day offset (bps):")
    print(f"  {'Offset':<10} {'ADD mean (bps)':>16} {'ADD t':>10} {'DROP mean (bps)':>18} {'DROP t':>10}")
    for offset in range(-ANNOUNCEMENT_LEAD_DAYS, POST_EVENT_WINDOW + 1):
        sub_add = er[(er['offset'] == offset) & (er['event_type'] == 'ADD')]['ret']
        sub_drop = er[(er['offset'] == offset) & (er['event_type'] == 'DROP')]['ret']
        if len(sub_add) > 0:
            m_add = sub_add.mean() * 10000
            t_add = sub_add.mean() / (sub_add.std(ddof=1) / np.sqrt(len(sub_add))) if sub_add.std(ddof=1) > 0 else 0
        else:
            m_add, t_add = 0, 0
        if len(sub_drop) > 0:
            m_drop = sub_drop.mean() * 10000
            t_drop = sub_drop.mean() / (sub_drop.std(ddof=1) / np.sqrt(len(sub_drop))) if sub_drop.std(ddof=1) > 0 else 0
        else:
            m_drop, t_drop = 0, 0
        marker = ""
        if abs(t_add) > 1.96: marker += " ★ADD"
        if abs(t_drop) > 1.96: marker += " ★DROP"
        label = "EFFECTIVE" if offset == 0 else ""
        print(f"    T{offset:+3d} {label:<8}: {m_add:>+12.2f}  {t_add:>+7.2f}  {m_drop:>+14.2f}  {t_drop:>+7.2f}{marker}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Build L-S strategy at daily level
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Building L-S strategy (long adds T-5..T-1, short drops T-5..T-1)...")

    strategy = pd.Series(0.0, index=panel.index)
    n_active_events_per_day = pd.Series(0, index=panel.index)

    # Pre-event L-S (T-5 to T-1)
    for _, evt in all_events_clean.iterrows():
        eff = evt['effective_date']
        permno = evt['permno']
        ev_type = evt['event_type']
        if permno not in panel.columns: continue
        idx = panel.index.searchsorted(eff)
        if idx < ANNOUNCEMENT_LEAD_DAYS or idx >= len(panel): continue
        # T-5 to T-1 (5 trading days of pre-event drift)
        for offset in range(-ANNOUNCEMENT_LEAD_DAYS, 0):
            t_idx = idx + offset
            if t_idx < 0 or t_idx >= len(panel): continue
            ret_val = panel.iloc[t_idx][permno]
            d = panel.index[t_idx]
            if pd.notna(ret_val):
                # ADD: long; DROP: short
                signed_ret = float(ret_val) if ev_type == 'ADD' else -float(ret_val)
                strategy.loc[d] += signed_ret
                n_active_events_per_day.loc[d] += 1

    # Average by # of active events per day (portfolio normalization)
    strategy = strategy / n_active_events_per_day.replace(0, np.nan)
    strategy = strategy.fillna(0)

    # POST-event reversal (short adds, long drops T+1 to T+5)
    strategy_reversal = pd.Series(0.0, index=panel.index)
    n_active_rev = pd.Series(0, index=panel.index)
    for _, evt in all_events_clean.iterrows():
        eff = evt['effective_date']
        permno = evt['permno']
        ev_type = evt['event_type']
        if permno not in panel.columns: continue
        idx = panel.index.searchsorted(eff)
        if idx + 5 >= len(panel) or idx < 0: continue
        for offset in range(1, 6):
            t_idx = idx + offset
            if t_idx >= len(panel): continue
            ret_val = panel.iloc[t_idx][permno]
            d = panel.index[t_idx]
            if pd.notna(ret_val):
                # Reversal: ADD short, DROP long (opposite of pre-event)
                signed_ret = -float(ret_val) if ev_type == 'ADD' else float(ret_val)
                strategy_reversal.loc[d] += signed_ret
                n_active_rev.loc[d] += 1
    strategy_reversal = strategy_reversal / n_active_rev.replace(0, np.nan)
    strategy_reversal = strategy_reversal.fillna(0)

    # Add-only long pre-event (capacity-friendly)
    add_long = pd.Series(0.0, index=panel.index)
    n_add_per_day = pd.Series(0, index=panel.index)
    for _, evt in all_events_clean.iterrows():
        if evt['event_type'] != 'ADD': continue
        permno = evt['permno']
        if permno not in panel.columns: continue
        eff = evt['effective_date']
        idx = panel.index.searchsorted(eff)
        if idx < ANNOUNCEMENT_LEAD_DAYS or idx >= len(panel): continue
        for offset in range(-ANNOUNCEMENT_LEAD_DAYS, 0):
            t_idx = idx + offset
            if t_idx < 0 or t_idx >= len(panel): continue
            ret_val = panel.iloc[t_idx][permno]
            d = panel.index[t_idx]
            if pd.notna(ret_val):
                add_long.loc[d] += float(ret_val)
                n_add_per_day.loc[d] += 1
    add_long = add_long / n_add_per_day.replace(0, np.nan)
    add_long = add_long.fillna(0)

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    summarize("L-S Pre-event (long adds / short drops, T-5 to T-1)", strategy)
    print()
    summarize("Post-event Reversal (short adds / long drops, T+1 to T+5)", strategy_reversal)
    print()
    summarize("Add-only long (T-5 to T-1)", add_long)

    print("\n" + "=" * 90)
    print("VERDICT GUIDE")
    print("=" * 90)
    print("  ETF sleeve gates: Sharpe ≥ 0.4 + NW t ≥ 1.8")
    print("  PASS aggregate + 3/3 sub-period positive: lock Path N spec")
    print("  Pattern check: ADD T-5..T-1 should be POSITIVE bps; DROP T-5..T-1 should be NEGATIVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
