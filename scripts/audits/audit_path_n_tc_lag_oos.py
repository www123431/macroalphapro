"""
scripts/audit_path_n_tc_lag_oos.py — Path N TC / NW lag / OOS forward audit.

3 sensitivity tests:
  1. TC: rerun with 5/10/15/20/30bp roundtrip (S&P 500 large-cap likely 5-15bp realistic vs 30bp standing rule)
  2. NW lag: stat at lag 10/15/21/30/60 days
  3. OOS forward 2024-2026 (TRUE OOS post-spec-window)
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

from engine.path_n.reconstitution_strategy import (
    build_add_event_strategy, PRE_EVENT_DAYS_LOCKED,
)
from engine.universe_singlename.crsp_loader import _open_wrds_connection


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


def summarize(label, x):
    x = x.dropna()
    if len(x) < 30:
        print(f"  {label}: too few obs"); return
    sh = sharpe(x)
    nw60 = nw_t(x, 60)
    cum = float((1 + x).cumprod().iloc[-1] - 1) * 100
    print(f"  {label:<35} Sh={sh:+.4f}  NW60={nw60:+.4f}  cum={cum:+.0f}%  n={len(x)}")


def main():
    print("=" * 90)
    print("PATH N AUDIT — TC sensitivity / NW lag / OOS forward")
    print("=" * 90)

    # ── Fetch full data 2013-2026 ─────────────────────────────────────────────
    print("\n[1/4] Querying CRSP msp500list for ADD events 2014-2026 (incl OOS)...")
    conn = _open_wrds_connection()
    try:
        today = datetime.date.today().isoformat()
        sql = f"""
        SELECT permno, start, ending FROM crsp.msp500list
        WHERE start BETWEEN '2014-01-01' AND '{today}'
        ORDER BY start
        """
        events_raw = conn.raw_sql(sql, date_cols=['start', 'ending'])
        events = events_raw.rename(columns={'start': 'effective_date'}).copy()
        events['event_type'] = 'ADD'
        events = events[['permno', 'effective_date', 'event_type']]
        events['permno'] = events['permno'].astype(int)
        print(f"  Add events 2014-today: {len(events)}")

        # Subset for in-sample vs OOS
        in_sample_evts = events[events['effective_date'] <= pd.Timestamp('2023-12-31')]
        oos_evts = events[events['effective_date'] > pd.Timestamp('2023-12-31')]
        print(f"  In-sample 2014-2023: {len(in_sample_evts)}, OOS 2024-today: {len(oos_evts)}")

        permno_list = ",".join(str(p) for p in sorted(events['permno'].unique()))
        sql = f"""
        SELECT permno, date, ret FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '2013-06-01' AND '{today}'
        """
        daily = conn.raw_sql(sql, date_cols=['date'])
    finally:
        conn.close()

    daily['ret'] = pd.to_numeric(daily['ret'], errors='coerce')
    daily['permno'] = daily['permno'].astype(int)
    panel = daily.pivot_table(index='date', columns='permno', values='ret', aggfunc='first')
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    print(f"  Daily panel: {panel.shape}")

    # ── Audit 1: TC sensitivity ──────────────────────────────────────────────
    print("\n[2/4] Audit 1: TC sensitivity (5/10/15/20/30bp roundtrip)")
    print(f"  S&P 500 large-cap realistic TC = 5-15bp; standing rule = 30bp single-stock")
    print(f"  In-sample 2014-2023 Sharpe / NW t / annual TC drag:")
    for tc in [5, 10, 15, 20, 30]:
        result = build_add_event_strategy(in_sample_evts, panel, tc_bps_roundtrip=float(tc))
        sr = result.daily_returns.loc['2014-01-01':'2023-12-31']
        sh = sharpe(sr); nw21 = nw_t(sr, 21); nw60 = nw_t(sr, 60)
        print(f"  TC={tc:>2}bp:  Sharpe={sh:+.3f}  NW(21)={nw21:+.3f}  NW(60)={nw60:+.3f}  "
              f"TC_drag={result.tc_drag_annual_pct:.2f}%/yr")

    # ── Audit 2: NW lag sensitivity at 30bp TC (spec lock TC) ────────────────
    print("\n[3/4] Audit 2: NW lag sensitivity (TC=30bp spec-locked)")
    result_locked = build_add_event_strategy(in_sample_evts, panel, tc_bps_roundtrip=30.0)
    sr_locked = result_locked.daily_returns.loc['2014-01-01':'2023-12-31']
    sh_locked = sharpe(sr_locked)
    print(f"  Sharpe net (30bp): {sh_locked:+.3f}")
    for lag in [10, 15, 21, 30, 45, 60, 90]:
        t = nw_t(sr_locked, lag)
        gate_etf = "PASS" if t >= 1.8 else "FAIL"
        gate_ss = "PASS" if t >= 2.0 else "FAIL"
        print(f"  NW lag={lag:>3}: t={t:+.3f}   ETF_strict(1.8)={gate_etf}   SS_strict(2.0)={gate_ss}")

    # ── Audit 3: OOS forward 2024-today ──────────────────────────────────────
    print("\n[4/4] Audit 3: OOS forward 2024-today (TRUE OUT-OF-SAMPLE)")
    print("  Re-run strategy on 2014-today with TC=30bp (spec lock); split in-sample vs OOS")
    result_full = build_add_event_strategy(events, panel, tc_bps_roundtrip=30.0)
    full_net = result_full.daily_returns
    in_sample_sr = full_net.loc['2014-01-01':'2023-12-31']
    oos_sr = full_net.loc['2024-01-01':]
    print()
    summarize("In-sample 2014-2023", in_sample_sr)
    summarize("OOS forward 2024-today", oos_sr)
    sh_in = sharpe(in_sample_sr)
    sh_oos = sharpe(oos_sr)
    if sh_in and sh_in > 0:
        ratio = sh_oos / sh_in
        print(f"  OOS/in-sample Sharpe ratio: {ratio:.3f}")
        if ratio >= 1.0: tag = "OOS BETTER — robust forward"
        elif ratio >= 0.6: tag = "OOS PARTIAL retention (60-100% in-sample)"
        elif ratio >= 0.3: tag = "OOS PARTIAL DECAY"
        elif ratio >= 0: tag = "OOS WEAK"
        else: tag = "OOS REVERSED"
        print(f"  → {tag}")
    print(f"\n  OOS by year:")
    for yr in sorted(set(oos_sr.index.year)):
        sub = oos_sr[oos_sr.index.year == yr]
        if len(sub) > 30:
            print(f"    {yr}: Sharpe {sharpe(sub):+.3f}  n={len(sub)}")

    # ── Audit 4: TC sensitivity for OOS too ──────────────────────────────────
    print("\n  OOS TC sensitivity (just OOS forward 2024-today):")
    for tc in [5, 10, 15, 30]:
        res = build_add_event_strategy(events, panel, tc_bps_roundtrip=float(tc))
        oos_x = res.daily_returns.loc['2024-01-01':]
        sh = sharpe(oos_x); nw = nw_t(oos_x, 21)
        print(f"  OOS TC={tc:>2}bp: Sharpe={sh:+.3f}  NW(21)={nw:+.3f}  n={len(oos_x.dropna())}")

    print("\n" + "=" * 90)
    print("AUDIT SUMMARY GUIDE")
    print("=" * 90)
    print("  If TC=10bp realistic gives Sharpe > 0.7 + NW t > 2.0 → Path N upgrade PASS_INDEPENDENT")
    print("  If OOS forward >= 50% in-sample Sharpe → forward signal alive")
    print("  If OOS reversed → Path N MARGINAL stays / downgrade FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
