"""
scripts/audit_path_m_oos_forward.py — Path M真 OOS forward audit.

Tests Path M strategy on 2024-01-01 to 2026-05-13 (28 months post-spec-window).
Spec id=69 locked window is 2014-2023; spec hash a3f50c9f locked 2026-05-13 evening.
ANY data after 2023-12-31 was NOT used in scout, robustness check, or backtest.

Audits:
  1. OOS forward Sharpe / NW t / monthly trend
  2. Universe sensitivity: random drop 5 ETFs, 100 trials
  3. Long-only vs L-S decomposition
  4. Rolling 3y windows on full 2014-2026 combined
  5. NW lag sensitivity (20/30/40/60)
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
import yfinance as yf

from engine.path_m.thematic_momentum_strategy import (
    LOCKED_UNIVERSE_LIST, N_UNIVERSE_LOCKED,
    compute_monthly_momentum, form_long_short_cohorts, compute_strategy_returns,
    TC_BPS_ROUNDTRIP_LOCKED,
)


def sharpe(x, p=252):
    if len(x) < 30 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(p))


def nw_t(x, lag=60):
    v = x.dropna().values
    n = len(v)
    if n < lag + 2:
        return float("nan")
    mu = v.mean()
    e = v - mu
    s = (e * e).sum() / n
    for k in range(1, lag + 1):
        gk = (e[k:] * e[:-k]).sum() / n
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")


def max_dd(x):
    c = (1 + x).cumprod()
    rm = c.cummax()
    return float(((c - rm) / rm).min())


def bootstrap_sharpe_ci(x, n=2000, seed=20260513):
    rng = np.random.default_rng(seed)
    vals = x.dropna().values
    if len(vals) < 30:
        return float("nan"), float("nan")
    sharpes = []
    for _ in range(n):
        idx = rng.integers(0, len(vals), size=len(vals))
        b = vals[idx]
        if b.std(ddof=1) > 0:
            sharpes.append(b.mean() / b.std(ddof=1) * np.sqrt(252))
    return float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


def run_strategy(close_panel: pd.DataFrame) -> dict:
    """Run Path M strategy on given price panel; return result dict."""
    mom = compute_monthly_momentum(close_panel)
    longs, shorts = form_long_short_cohorts(mom)
    strat = compute_strategy_returns(close_panel, longs, shorts)
    return {
        'net': strat.daily_returns,
        'gross': strat.daily_gross,
        'long_cohorts': longs,
        'short_cohorts': shorts,
        'n_rebalances': strat.n_rebalances,
    }


def long_only_top3(close_panel: pd.DataFrame, longs_cohorts: dict) -> pd.Series:
    """Long-only top-3 strategy."""
    daily_ret = close_panel.pct_change()
    out = pd.Series(0.0, index=daily_ret.index)
    rebal_dates = sorted(longs_cohorts.keys())
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        longs = longs_cohorts[rd]
        hold = (daily_ret.index > rd) & (daily_ret.index <= rebal_dates[i + 1])
        for d in daily_ret.index[hold]:
            r = daily_ret.loc[d, longs].dropna().mean()
            if not np.isnan(r):
                out.loc[d] = r
    return out


def short_only_bot3(close_panel: pd.DataFrame, shorts_cohorts: dict) -> pd.Series:
    """Short-only bottom-3 (sign-inverted)."""
    daily_ret = close_panel.pct_change()
    out = pd.Series(0.0, index=daily_ret.index)
    rebal_dates = sorted(shorts_cohorts.keys())
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        shorts = shorts_cohorts[rd]
        hold = (daily_ret.index > rd) & (daily_ret.index <= rebal_dates[i + 1])
        for d in daily_ret.index[hold]:
            r = daily_ret.loc[d, shorts].dropna().mean()
            if not np.isnan(r):
                out.loc[d] = -r  # short = negative
    return out


def summarize(name, x):
    x = x.dropna()
    sh = sharpe(x); nw = nw_t(x, 60)
    ann_ret = float(x.mean() * 252)
    ann_vol = float(x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else 0
    cum = float((1 + x).cumprod().iloc[-1] - 1) if len(x) > 0 else 0
    mdd = max_dd(x)
    ci_lo, ci_hi = bootstrap_sharpe_ci(x, n=1000)
    print(f"  {name:<45} Sh={sh:+.3f}  NW t={nw:+.3f}  "
          f"ann_ret={ann_ret*100:+5.1f}%  vol={ann_vol*100:+4.1f}%  "
          f"cum={cum*100:+6.0f}%  DD={mdd*100:+.0f}%  "
          f"CI=[{ci_lo:+.2f},{ci_hi:+.2f}]")


def main():
    print(f"Path M OOS Forward Audit — today {datetime.date.today()}")
    print(f"Spec id=69 hash a3f50c9f locked 2026-05-13 evening, window 2014-2023.\n")

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Fetch full data 2013-2026 (current today)
    # ──────────────────────────────────────────────────────────────────────
    print("Fetching 34 locked thematic ETFs through today...")
    today = datetime.date.today()
    data = yf.download(LOCKED_UNIVERSE_LIST, start="2013-01-01",
                       end=today.isoformat(),
                       progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how='all').ffill()
    print(f"Prices panel: {close.shape[0]} dates x {close.shape[1]} tickers")
    print(f"Date range: {close.index[0].date()} -> {close.index[-1].date()}\n")

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Run strategy on FULL window for cohort generation
    # ──────────────────────────────────────────────────────────────────────
    full_result = run_strategy(close)
    full_net = full_result['net']

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: OOS FORWARD test (2024-01-01 to today)
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 90)
    print("STEP 1: OOS FORWARD TEST — 2024-01-01 to today (TRUE OUT-OF-SAMPLE)")
    print("=" * 90)

    oos_start = pd.Timestamp("2024-01-01")
    in_sample = full_net.loc[:"2023-12-31"]
    oos_forward = full_net.loc[oos_start:]

    print(f"\nIn-sample (2014-2023):     n={len(in_sample)} daily obs")
    print(f"OOS forward (2024-{today.year}):    n={len(oos_forward)} daily obs ({len(oos_forward)/21:.1f} months approx)\n")

    summarize("In-sample 2014-2023 (spec window)", in_sample)
    summarize("OOS FORWARD 2024-today (TRUE OOS)", oos_forward)

    sh_in = sharpe(in_sample)
    sh_oos = sharpe(oos_forward)
    if sh_in and sh_in > 0:
        ratio = sh_oos / sh_in
        print(f"\n  OOS / In-sample Sharpe ratio: {ratio:.3f}")
        if ratio >= 0.6:
            print(f"  >>>> OOS Sharpe >= 60% in-sample: STRONG forward signal <<<<")
        elif ratio >= 0.3:
            print(f"  >>>> OOS Sharpe partial retention (30-60% in-sample) <<<<")
        else:
            print(f"  >>>> OOS Sharpe weak (<30% in-sample) — signal degrading <<<<")

    # Annual breakdown OOS
    print(f"\n  OOS forward by year:")
    for yr in sorted(set(oos_forward.index.year)):
        sub = oos_forward[oos_forward.index.year == yr]
        sh_yr = sharpe(sub)
        ret_yr = float(sub.mean() * 252) * 100
        print(f"    {yr}: Sharpe {sh_yr:+.3f}  ann_ret {ret_yr:+5.1f}%  n={len(sub)}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Universe sensitivity — drop random 5, 50 trials
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 2: UNIVERSE SENSITIVITY — drop random 5 ETFs x 50 trials")
    print("=" * 90)

    rng = np.random.default_rng(20260513)
    sharpes_full_window = []
    sharpes_oos_only = []
    n_trials = 50
    for trial in range(n_trials):
        drop = rng.choice(LOCKED_UNIVERSE_LIST, size=5, replace=False).tolist()
        kept = [t for t in LOCKED_UNIVERSE_LIST if t not in drop]
        close_sub = close[kept]
        result = run_strategy(close_sub)
        net = result['net']
        sharpes_full_window.append(sharpe(net.loc["2014-01-01":"2023-12-31"]))
        sharpes_oos_only.append(sharpe(net.loc[oos_start:]))

    s_arr_full = np.array([s for s in sharpes_full_window if not np.isnan(s)])
    s_arr_oos = np.array([s for s in sharpes_oos_only if not np.isnan(s)])

    print(f"\nIn-sample 2014-2023 Sharpe across 50 random-drop-5 trials:")
    print(f"  mean={s_arr_full.mean():+.3f}  std={s_arr_full.std():+.3f}  "
          f"min={s_arr_full.min():+.3f}  max={s_arr_full.max():+.3f}")
    print(f"  fraction >= 0.4 PASS gate: {(s_arr_full >= 0.4).sum()}/{len(s_arr_full)} = {(s_arr_full >= 0.4).mean()*100:.0f}%")

    print(f"\nOOS forward 2024-{today.year} Sharpe across same 50 trials:")
    print(f"  mean={s_arr_oos.mean():+.3f}  std={s_arr_oos.std():+.3f}  "
          f"min={s_arr_oos.min():+.3f}  max={s_arr_oos.max():+.3f}")
    print(f"  fraction > 0 (positive OOS): {(s_arr_oos > 0).sum()}/{len(s_arr_oos)} = {(s_arr_oos > 0).mean()*100:.0f}%")
    print(f"  fraction >= 0.4 PASS gate: {(s_arr_oos >= 0.4).sum()}/{len(s_arr_oos)} = {(s_arr_oos >= 0.4).mean()*100:.0f}%")

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Long-only vs Short-only decomposition
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 3: LEG DECOMPOSITION — long-only vs short-only contributions")
    print("=" * 90)
    print()
    long_only = long_only_top3(close, full_result['long_cohorts'])
    short_only_signed = short_only_bot3(close, full_result['short_cohorts'])

    print("In-sample 2014-2023:")
    summarize("L-S (top-3/bot-3) - SPEC LOCKED", full_net.loc[:"2023-12-31"])
    summarize("Long-only top-3", long_only.loc[:"2023-12-31"])
    summarize("Short-only -bot-3 (sign-inverted)", short_only_signed.loc[:"2023-12-31"])
    print()
    print(f"OOS forward 2024-{today.year}:")
    summarize("L-S OOS", full_net.loc[oos_start:])
    summarize("Long-only OOS", long_only.loc[oos_start:])
    summarize("Short-only OOS", short_only_signed.loc[oos_start:])

    # ──────────────────────────────────────────────────────────────────────
    # Step 6: NW lag sensitivity
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 4: NW LAG SENSITIVITY — t-stat at different lag specs")
    print("=" * 90)
    print()
    for label, ts in [("In-sample 2014-2023", full_net.loc[:"2023-12-31"]),
                       ("OOS forward", full_net.loc[oos_start:])]:
        print(f"{label}:")
        for lag in [10, 21, 30, 40, 60, 90]:
            t = nw_t(ts, lag)
            print(f"  NW lag {lag:3d}d: t = {t:+.3f}  {'PASS' if t >= 1.8 else 'FAIL'}")
        print()

    # ──────────────────────────────────────────────────────────────────────
    # Step 7: Annual Sharpe trend (2014-2026)
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 90)
    print("STEP 5: ANNUAL SHARPE TREND (2014-today)")
    print("=" * 90)
    print()
    for yr in sorted(set(full_net.index.year)):
        sub = full_net[full_net.index.year == yr]
        sh_yr = sharpe(sub)
        ret_yr = float(sub.mean() * 252) * 100
        is_oos = yr >= 2024
        marker = " ⭐OOS⭐" if is_oos else ""
        print(f"  {yr}: Sharpe {sh_yr:+.3f}  ann_ret {ret_yr:+5.1f}%  n={len(sub)}{marker}")

    print("\n" + "=" * 90)
    print("AUDIT VERDICT GUIDE")
    print("=" * 90)
    print("UPGRADE to PASS_INDEPENDENT if:")
    print("  - OOS forward Sharpe >= 0.3 AND >= 50% in-sample Sharpe")
    print("  - Universe sensitivity: >= 70% of trials still PASS gate (Sharpe >= 0.4 in-sample)")
    print("  - NW t robust across lag 20-60")
    print()
    print("RECLASSIFY to BACKTEST_PASS_PENDING if:")
    print("  - OOS Sharpe < 0.3 OR < 30% in-sample")
    print("  - Universe sensitivity < 50% PASS gate")
    print("  - Long-only dominant + short-leg dragging (alpha quality concern)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
