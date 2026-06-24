"""
scripts/scout_carry_a_vs_c.py — Quick-and-dirty scout for Path A vs Path C candidates.

NOT a backtest. NOT pre-registered. Just raw ETF Sharpe to decide which is worth spec lock.

A proxy: DBV (Invesco DB G10 Currency Harvest) — embeds long-3-high-rate / short-3-low-rate G10 carry
C proxy: HYG - LQD spread — credit risk premium (HY minus IG investment grade)

Period: 2014-01-01 → 2023-12-31 (same as Path E/F/G — for fair comparison)
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
    """Simple Newey-West t-stat for daily.mean() != 0."""
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


def summarize(name: str, daily: pd.Series):
    daily = daily.dropna()
    sh = annualized_sharpe(daily)
    nw = newey_west_t(daily, lag=60)
    vol_ann = float(daily.std(ddof=1) * np.sqrt(252))
    ret_ann = float(daily.mean() * 252)
    mdd = max_drawdown(daily)
    cum_total = float((1 + daily).cumprod().iloc[-1] - 1)

    # Sub-periods
    pre = daily.loc[:"2019-12-31"]
    covid = daily.loc["2020-01-01":"2021-12-31"]
    post = daily.loc["2022-01-01":]
    sh_pre = annualized_sharpe(pre)
    sh_covid = annualized_sharpe(covid)
    sh_post = annualized_sharpe(post)

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    print(f"  n daily obs:        {len(daily)}")
    print(f"  Ann return:         {ret_ann * 100:+7.2f}%/yr")
    print(f"  Ann vol:            {vol_ann * 100:7.2f}%")
    print(f"  Sharpe:             {sh:+.4f}")
    print(f"  NW t (lag 60):      {nw:+.4f}")
    print(f"  Cumulative 10y:     {cum_total * 100:+7.2f}%")
    print(f"  Max DD:             {mdd * 100:+7.2f}%")
    print(f"  Sub-period Sharpe:")
    print(f"    Pre-COVID  (2014-2019): {sh_pre:+.4f}")
    print(f"    COVID      (2020-2021): {sh_covid:+.4f}")
    print(f"    Post-COVID (2022-2023): {sh_post:+.4f}")

    # Regime stability flag
    sub_sharpes = [s for s in [sh_pre, sh_covid, sh_post] if not np.isnan(s)]
    all_positive = all(s > 0 for s in sub_sharpes)
    print(f"  All sub-period positive: {all_positive}")

    # Path F ETF gate check
    gate1_pass = sh >= 0.4 and nw >= 1.8
    print(f"  ETF Gate 1 (Sharpe ≥ 0.4 + NW t ≥ 1.8): {'PASS' if gate1_pass else 'FAIL'}")


def main():
    start, end = "2014-01-01", "2023-12-31"

    print(f"Fetching DBV, HYG, LQD from yfinance ({start} → {end})...")
    tickers = ["DBV", "HYG", "LQD"]
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how="all").ffill()

    # Path A proxy: DBV total return (auto_adjust=True includes dist)
    dbv_ret = close["DBV"].pct_change().dropna()
    summarize("Path A proxy — DBV (G10 Currency Harvest, embedded carry)", dbv_ret)

    # Path C proxy: HYG - LQD long-short spread (daily)
    hyg_ret = close["HYG"].pct_change().dropna()
    lqd_ret = close["LQD"].pct_change().dropna()
    common = hyg_ret.index.intersection(lqd_ret.index)
    spread = hyg_ret.loc[common] - lqd_ret.loc[common]
    summarize("Path C proxy — HYG-LQD spread (Credit risk premium L/S)", spread)

    # Honest disclose
    print(f"\n{'=' * 70}")
    print("HONEST DISCLOSE (this is a scout, NOT a spec-locked backtest)")
    print(f"{'=' * 70}")
    print("- No TC drag applied (would subtract ~1-2% annual on either)")
    print("- DBV embeds its own rebalancing TC internally (expense ratio ~75bp)")
    print("- HYG-LQD: dollar-neutral L/S; needs short-borrow + financing in real money")
    print("- This is RAW ETF, no signal-conditioning (Path A/C spec would add signal layer)")
    print("- Sharpe here = upper bound; signal-conditioned strategy likely WORSE not better")
    print("- Compare to Path F (Sharpe 0.41 raw / NW t 1.43) and Path G (0.35 / 1.21)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
