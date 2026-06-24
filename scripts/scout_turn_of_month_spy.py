"""
scripts/scout_turn_of_month_spy.py — Turn-of-Month effect scout on SPY.

NOT a backtest. NOT pre-registered. Just raw event-study to test if Ariel 1987 /
Hong-Yu 2020 ToM effect persists in 2014-2023.

ToM definition (Lakonishok-Smidt 1988 / Hong-Yu 2020):
  - T-1: last business day of month
  - T+1, T+2, T+3: first 3 business days of next month
  → 4 trading days per month × 120 months ≈ 480 ToM days (out of ~2520 total)

Strategy proxy:
  Long SPY ONLY on ToM days, flat (0% return) otherwise.

Compare:
  - Mean ToM-day return vs non-ToM-day return
  - Strategy Sharpe vs buy-and-hold SPY Sharpe
  - Annualized return: mean_ToM × n_ToM/yr
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import yfinance as yf


def annualized_sharpe(daily: pd.Series, ann_factor: int = 252) -> float:
    if len(daily) < 30 or daily.std(ddof=1) == 0:
        return float("nan")
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(ann_factor))


def newey_west_t(daily: pd.Series, lag: int = 60) -> float:
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


def max_drawdown(daily: pd.Series) -> float:
    cum = (1 + daily).cumprod()
    rmax = cum.cummax()
    return float(((cum - rmax) / rmax).min())


def mark_tom_days(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Mark each trading day as ToM (T-1, T+1, T+2, T+3) or non-ToM."""
    df = pd.DataFrame({'date': daily_index})
    df['month'] = df['date'].dt.to_period('M')
    df['rank_from_end'] = df.groupby('month').cumcount(ascending=False) + 1
    df['rank_from_start'] = df.groupby('month').cumcount(ascending=True) + 1
    # T-1 = last business day of month (rank_from_end == 1)
    # T+1, T+2, T+3 = first 3 business days of month (rank_from_start ∈ {1, 2, 3})
    is_tom = (df['rank_from_end'] == 1) | (df['rank_from_start'] <= 3)
    return pd.Series(is_tom.values, index=daily_index, name='is_tom')


def event_window_breakdown(daily_ret: pd.Series, daily_index: pd.DatetimeIndex):
    """Break down returns by day-relative-to-month-boundary."""
    df = pd.DataFrame({'date': daily_index, 'ret': daily_ret.values})
    df['month'] = df['date'].dt.to_period('M')
    df['rank_from_end'] = df.groupby('month').cumcount(ascending=False) + 1
    df['rank_from_start'] = df.groupby('month').cumcount(ascending=True) + 1
    df['rel_day'] = None
    df.loc[df['rank_from_end'] == 1, 'rel_day'] = 'T-1 (last)'
    df.loc[df['rank_from_start'] == 1, 'rel_day'] = 'T+1 (first)'
    df.loc[df['rank_from_start'] == 2, 'rel_day'] = 'T+2'
    df.loc[df['rank_from_start'] == 3, 'rel_day'] = 'T+3'
    df.loc[df['rank_from_start'] == 4, 'rel_day'] = 'T+4'
    df.loc[df['rank_from_start'] == 5, 'rel_day'] = 'T+5'
    df.loc[df['rank_from_end'] == 2, 'rel_day'] = 'T-2'
    df.loc[df['rank_from_end'] == 3, 'rel_day'] = 'T-3'
    df['rel_day'] = df['rel_day'].fillna('mid-month')
    return df


def main():
    print("Fetching SPY 2014-2023 from yfinance...")
    data = yf.download("SPY", start="2014-01-01", end="2024-01-01",
                       progress=False, auto_adjust=True)
    close = data["Close"] if hasattr(data, 'columns') else data
    if hasattr(close, 'columns') and 'SPY' in close.columns:
        close = close['SPY']
    daily_ret = close.pct_change().dropna()
    print(f"SPY daily: {len(daily_ret)} obs, {daily_ret.index[0].date()} → {daily_ret.index[-1].date()}\n")

    # ── PART 1: Day-relative-to-month-boundary breakdown ─────────────────────
    print("=" * 70)
    print("PART 1 — SPY mean return by relative day to month boundary")
    print("=" * 70)
    ew = event_window_breakdown(daily_ret, daily_ret.index)
    order = ['T-3', 'T-2', 'T-1 (last)', 'T+1 (first)', 'T+2', 'T+3', 'T+4', 'T+5', 'mid-month']
    print(f"\n{'Rel day':<15} {'mean (bps)':>12} {'std %':>8} {'n':>5} {'t-stat':>8}")
    for d in order:
        sub = ew.loc[ew['rel_day'] == d, 'ret']
        if len(sub) > 0:
            m_bps = sub.mean() * 10000
            s_pct = sub.std(ddof=1) * 100
            n = len(sub)
            t = sub.mean() / (sub.std(ddof=1) / np.sqrt(n)) if sub.std(ddof=1) > 0 else 0
            print(f"  {d:<13} {m_bps:>+10.2f}    {s_pct:>6.2f}   {n:>5d}  {t:>+6.2f}")

    # ── PART 2: ToM days vs non-ToM days ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2 — ToM days (T-1 + T+1 + T+2 + T+3) vs non-ToM days")
    print("=" * 70)
    is_tom = mark_tom_days(daily_ret.index)
    tom_ret = daily_ret.loc[is_tom]
    non_tom_ret = daily_ret.loc[~is_tom]

    for label, sub in [('ToM days', tom_ret), ('non-ToM days', non_tom_ret)]:
        m_bps = sub.mean() * 10000
        s_pct = sub.std(ddof=1) * 100
        n = len(sub)
        t = sub.mean() / (sub.std(ddof=1) / np.sqrt(n))
        ann_return_if_compound = ((1 + sub.mean()) ** 252 - 1) * 100  # if held every day at this rate
        print(f"  {label:<15}: mean={m_bps:>+7.2f} bps  std={s_pct:.2f}%  n={n:>4d}  t={t:>+5.2f}  ann_if_compound={ann_return_if_compound:>+6.2f}%/yr")

    # ── PART 3: Strategy proxy — long SPY on ToM days only ───────────────────
    print("\n" + "=" * 70)
    print("PART 3 — Strategy proxy: Long SPY on ToM days only (flat else)")
    print("=" * 70)
    strategy = pd.Series(0.0, index=daily_ret.index)
    strategy.loc[is_tom] = daily_ret.loc[is_tom]

    sh = annualized_sharpe(strategy)
    nw = newey_west_t(strategy, lag=60)
    vol_ann = float(strategy.std(ddof=1) * np.sqrt(252))
    ret_ann = float(strategy.mean() * 252)
    mdd = max_drawdown(strategy)
    cum_total = float((1 + strategy).cumprod().iloc[-1] - 1)

    print(f"\n  Strategy (long SPY ToM-only):")
    print(f"    n daily obs:      {len(strategy)}")
    print(f"    n active days:    {is_tom.sum()} ({is_tom.sum()/len(strategy)*100:.1f}%)")
    print(f"    Ann return:       {ret_ann*100:+.2f}%/yr")
    print(f"    Ann vol:          {vol_ann*100:.2f}%")
    print(f"    Sharpe:           {sh:+.4f}")
    print(f"    NW t (lag 60):    {nw:+.4f}")
    print(f"    Cumulative 10y:   {cum_total*100:+.2f}%")
    print(f"    Max DD:           {mdd*100:+.2f}%")

    # Compare buy-and-hold
    bh = daily_ret
    print(f"\n  Buy-and-Hold SPY (reference):")
    print(f"    Ann return:       {(bh.mean()*252)*100:+.2f}%/yr")
    print(f"    Ann vol:          {(bh.std(ddof=1)*np.sqrt(252))*100:.2f}%")
    print(f"    Sharpe:           {annualized_sharpe(bh):+.4f}")
    print(f"    NW t (lag 60):    {newey_west_t(bh, 60):+.4f}")
    print(f"    Cumulative 10y:   {((1+bh).cumprod().iloc[-1]-1)*100:+.2f}%")
    print(f"    Max DD:           {max_drawdown(bh)*100:+.2f}%")

    # ── PART 4: ToM concentration — what % of total SPY return is in ToM days
    total_ret = (1 + daily_ret).prod() - 1
    tom_only_ret = (1 + tom_ret).prod() - 1
    print(f"\n  ToM days hold {is_tom.sum()/len(strategy)*100:.1f}% of trading days but capture "
          f"{tom_only_ret/total_ret*100:.1f}% of buy-and-hold cumulative return")

    # ── PART 5: Sub-period ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 5 — Sub-period breakdown")
    print("=" * 70)
    for label, sl in [('Pre-COVID 2014-2019', slice(None, '2019-12-31')),
                       ('COVID 2020-2021', slice('2020-01-01', '2021-12-31')),
                       ('Post-COVID 2022-2023', slice('2022-01-01', None))]:
        sub = strategy.loc[sl]
        if len(sub) > 30:
            sh_sub = annualized_sharpe(sub)
            nw_sub = newey_west_t(sub, 60)
            ret_sub = sub.mean() * 252 * 100
            print(f"  {label}: Sharpe={sh_sub:+.3f}  NW t={nw_sub:+.3f}  ann_ret={ret_sub:+.2f}%/yr")

    # ── PART 6: Strategy minus risk-free (proper Sharpe with rf estimate) ────
    # Approx rf ~ 1.5% avg over period (Fed funds blended)
    rf_daily = 0.015 / 252
    strategy_excess = strategy - rf_daily * is_tom.astype(float)
    print(f"\n  After rf adjustment (~1.5%/yr): Sharpe excess = {annualized_sharpe(strategy_excess):+.4f}")

    print("\n" + "=" * 70)
    print("HONEST DISCLOSE")
    print("=" * 70)
    print("- No TC drag applied; in real trading, ~3-5bp roundtrip per month × 12 = 0.4-0.6%/yr drag")
    print("- ToM strategy capital efficiency: only 4/21 ≈ 19% of days in market; flat 81%")
    print("- A leveraged variant (4x ToM) brings Sharpe-equivalent to higher annual return")
    print("- Hong-Yu 2020 finding: ToM effect persists in SPY through 2018")
    print("- 2014-2023 is mostly post-Hong-Yu (5y arbitrage); decay risk real")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
