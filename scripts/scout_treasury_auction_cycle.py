"""
scripts/scout_treasury_auction_cycle.py — Lou-Yan-Zhang 2013 Treasury Auction Cycle scout.

NOT a backtest. NOT pre-registered. Just raw event-study to test if effect is alive in 2014-2023.

Mechanism (LYZ 2013): primary dealers absorb Treasury supply at auction → forced to hedge via
cash bond sales → bond prices DROP T-1 to T0 → mean-revert T+1 to T+3.

Strategy proxy: Long TLT post-auction (T+1 to T+3), Short TLT pre-auction (T-1 to T0).

Treasury auction calendar approximation (since exact API access is limited):
  - 10yr notes: 2nd Wednesday of most months (highest-impact Lou-Yan-Zhang event)
  - 30yr bonds: 2nd Thursday of most months
  - For scout we use 2nd Wednesday as the proxy auction date (10yr dominates)

Period: 2014-01-01 → 2023-12-31
"""
from __future__ import annotations
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf


def second_wednesday(year: int, month: int) -> datetime.date:
    """Return date of 2nd Wednesday of given month."""
    d = datetime.date(year, month, 1)
    # First Wednesday: day-of-week 2 (Mon=0)
    first_wed = d + datetime.timedelta(days=(2 - d.weekday()) % 7)
    return first_wed + datetime.timedelta(days=7)


def second_thursday(year: int, month: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    first_thu = d + datetime.timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + datetime.timedelta(days=7)


def build_auction_dates(start_year: int = 2014, end_year: int = 2023) -> list[datetime.date]:
    """Approximate 10yr auction dates: 2nd Wednesday of each month."""
    dates = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            dates.append(second_wednesday(y, m))
    return dates


def event_study(daily_ret: pd.Series, auction_dates: list[datetime.date],
                 event_window: tuple[int, int] = (-3, 3)) -> pd.DataFrame:
    """For each auction date, compute returns on offset days within event window.

    Returns DataFrame with rows = auction events, cols = event-day offsets.
    """
    sessions = daily_ret.index
    rows = []
    for ad in auction_dates:
        ad_ts = pd.Timestamp(ad)
        if ad_ts < sessions[0] or ad_ts > sessions[-1]:
            continue
        # Find T0 = first trading session at or after auction date
        future_sessions = sessions[sessions >= ad_ts]
        if len(future_sessions) == 0:
            continue
        t0 = future_sessions[0]
        t0_idx = sessions.searchsorted(t0)

        row = {'auction_date': ad, 't0_session': t0}
        for offset in range(event_window[0], event_window[1] + 1):
            idx = t0_idx + offset
            if 0 <= idx < len(sessions):
                row[f't{offset:+d}'] = float(daily_ret.iloc[idx])
            else:
                row[f't{offset:+d}'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def newey_west_t(daily: pd.Series, lag: int = 5) -> float:
    x = daily.dropna().values
    n = len(x)
    if n < lag + 1:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e * e).sum() / n
    s = gamma0
    for k in range(1, lag + 1):
        gk = (e[k:] * e[:-k]).sum() / n
        w = 1.0 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")


def annualized_sharpe(daily: pd.Series, ann_factor: int = 252) -> float:
    if len(daily) < 30 or daily.std(ddof=1) == 0:
        return float("nan")
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(ann_factor))


def main():
    print("Fetching TLT + IEF from yfinance (2014-2023)...")
    data = yf.download(["TLT", "IEF"], start="2014-01-01", end="2024-01-31",
                       progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how="all").ffill()
    tlt_ret = close["TLT"].pct_change().dropna()
    ief_ret = close["IEF"].pct_change().dropna()

    auction_dates = build_auction_dates(2014, 2023)
    print(f"Approx 10yr auction dates (2nd Wed of month): {len(auction_dates)} events")
    print(f"First 3: {auction_dates[:3]}; Last 3: {auction_dates[-3:]}\n")

    # ── PART 1: Event study around auction dates ─────────────────────────────
    print("=" * 70)
    print("PART 1 — Event study: TLT returns around approx auction dates")
    print("=" * 70)
    es_tlt = event_study(tlt_ret, auction_dates, event_window=(-3, 3))
    print(f"\nTLT event matrix: {len(es_tlt)} events × 7 offset days")
    print("\nMean TLT return by offset (bps/day):")
    for col in ['t-3', 't-2', 't-1', 't+0', 't+1', 't+2', 't+3']:
        if col in es_tlt.columns:
            m = es_tlt[col].mean() * 10000  # bps
            n = es_tlt[col].notna().sum()
            t = es_tlt[col].mean() / (es_tlt[col].std(ddof=1) / np.sqrt(n)) if n > 1 else 0
            print(f"  {col}: mean={m:+7.2f} bps  n={n:3d}  t-stat={t:+5.2f}")

    print("\nMean IEF return by offset (bps/day):")
    es_ief = event_study(ief_ret, auction_dates, event_window=(-3, 3))
    for col in ['t-3', 't-2', 't-1', 't+0', 't+1', 't+2', 't+3']:
        if col in es_ief.columns:
            m = es_ief[col].mean() * 10000
            n = es_ief[col].notna().sum()
            t = es_ief[col].mean() / (es_ief[col].std(ddof=1) / np.sqrt(n)) if n > 1 else 0
            print(f"  {col}: mean={m:+7.2f} bps  n={n:3d}  t-stat={t:+5.2f}")

    # ── PART 2: Strategy proxy ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2 — Strategy proxy: Long TLT [T+1, T+2, T+3] / Short TLT [T-1, T0]")
    print("=" * 70)

    # Aggregate per-event: long-leg = sum(t+1, t+2, t+3); short-leg = -sum(t-1, t0)
    es_tlt['long_leg']  = es_tlt[['t+1', 't+2', 't+3']].sum(axis=1)
    es_tlt['short_leg'] = -es_tlt[['t-1', 't+0']].sum(axis=1)
    es_tlt['strategy_event'] = es_tlt['long_leg'] + es_tlt['short_leg']
    n_eff = es_tlt['strategy_event'].notna().sum()
    mean_per_event_bps = es_tlt['strategy_event'].mean() * 10000
    std_per_event = es_tlt['strategy_event'].std(ddof=1)
    se = std_per_event / np.sqrt(n_eff)
    t_event = es_tlt['strategy_event'].mean() / se if se > 0 else 0

    print(f"\nTLT strategy aggregated per event (5 trading days L/S):")
    print(f"  Mean per event:   {mean_per_event_bps:+.2f} bps  ({es_tlt['strategy_event'].mean()*100:+.4f}%)")
    print(f"  n events:         {n_eff}")
    print(f"  Std per event:    {std_per_event * 100:.4f}%")
    print(f"  t-stat (event):   {t_event:+.4f}")

    # Annualize: ~12 events/yr × mean_per_event
    ann_return_bps = mean_per_event_bps * 12
    print(f"  Annualized:       {ann_return_bps:+.0f} bps/yr = {ann_return_bps/100:+.2f}%/yr")

    # Sharpe analog (per-event level)
    if std_per_event > 0:
        sharpe_per_event = es_tlt['strategy_event'].mean() / std_per_event
        sharpe_ann = sharpe_per_event * np.sqrt(12)  # 12 events/yr
        print(f"  Sharpe (annualized from event-level): {sharpe_ann:+.4f}")

    # Sub-period
    es_tlt['year'] = pd.to_datetime(es_tlt['auction_date']).dt.year
    print(f"\nSub-period (by year groups):")
    for label, yrs in [('Pre-COVID 2014-2019', range(2014, 2020)),
                        ('COVID 2020-2021', range(2020, 2022)),
                        ('Post-COVID 2022-2023', range(2022, 2024))]:
        mask = es_tlt['year'].isin(list(yrs))
        sub = es_tlt.loc[mask, 'strategy_event']
        if len(sub) > 3:
            m = sub.mean() * 10000
            n = len(sub)
            t = sub.mean() / (sub.std(ddof=1) / np.sqrt(n)) if n > 1 else 0
            print(f"  {label}: mean={m:+7.2f} bps  n={n:3d}  t-stat={t:+5.2f}")

    # ── PART 3: Day-of-month structural pattern (NO auction-date assumption) ─
    print("\n" + "=" * 70)
    print("PART 3 — Day-of-month pattern (any structural calendar effect on TLT)")
    print("=" * 70)
    tlt_dom = pd.DataFrame({'ret': tlt_ret})
    tlt_dom['day'] = tlt_ret.index.day
    by_day = tlt_dom.groupby('day')['ret'].agg(['mean', 'std', 'count'])
    by_day['t_stat'] = by_day['mean'] / (by_day['std'] / np.sqrt(by_day['count']))
    by_day['mean_bps'] = by_day['mean'] * 10000
    print("\nMean TLT return by day-of-month (bps; positive = bullish day pattern):")
    for d, row in by_day.iterrows():
        marker = " ★" if abs(row['t_stat']) > 1.96 else ""
        print(f"  Day {d:2d}: mean={row['mean_bps']:+7.2f} bps  n={int(row['count']):3d}  t-stat={row['t_stat']:+5.2f}{marker}")

    print("\n" + "=" * 70)
    print("HONEST DISCLOSE")
    print("=" * 70)
    print("- Used APPROXIMATE auction dates (2nd Wed of month for 10yr); real dates from Treasury.gov")
    print("- Lou-Yan-Zhang 2013 effect uses MULTIPLE security types (10yr/30yr/etc.) — scout uses one")
    print("- No TC drag applied; rebalance frequency is ~12 events/yr")
    print("- TLT is total-return ETF — captures both price + coupon; cleaner than cash bond")
    print("- 2022 bond bear was historic — may overwhelm cycle effect; check Pre-COVID sub-period")
    print("- Compare to project ETF Gate 1 thresholds: Sharpe ≥ 0.4 + NW t ≥ 1.8 (single tail)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
