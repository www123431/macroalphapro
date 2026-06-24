"""
scripts/scout_thematic_etf_momentum.py — Thematic ETF cross-section momentum scout.

NOT a backtest. NOT pre-registered. Raw event-study to test if cross-section
momentum on thematic ETF universe (less-academically-covered than US sector ETFs)
has signal alive in 2014-2023.

Universe: ~22 thematic ETFs with full 2014+ history:
  - ARK family (ARKK/ARKQ/ARKW/ARKG): innovation/disruption
  - Robotics+AI (ROBO/BOTZ)
  - Semiconductor (SMH/SOXX/XSD)
  - Cybersecurity (HACK)
  - China Internet (KWEB)
  - IPO (IPO)
  - Clean energy/commodity-thematic (ICLN/TAN/LIT/REMX/COPX/SLX)
  - Aviation/transport (JETS/AIRR)
  - Consumer thematic (PBJ/PEJ)

Signal: 12-1 momentum (12 months trailing, skip last month). Long top-3 /
short bottom-3, monthly rebalance, 1 month hold.

NOTE: small universe (~22), L=3/S=3 → leg size 3 each side, high variance.
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
    s = (e * e).sum() / n
    for k in range(1, lag + 1):
        gk = (e[k:] * e[:-k]).sum() / n
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    cum = (1 + daily).cumprod()
    rmax = cum.cummax()
    return float(((cum - rmax) / rmax).min())


def cross_section_long_short(
    daily_prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    top_n: int = 3,
    bot_n: int = 3,
) -> pd.Series:
    """Cross-section momentum L-S; monthly rebal."""
    monthly_prices = daily_prices.resample('ME').last()
    sig_numer = monthly_prices.shift(skip_months)
    sig_denom = monthly_prices.shift(lookback_months)
    momentum = sig_numer / sig_denom - 1.0

    daily_ret = daily_prices.pct_change()
    out = pd.Series(0.0, index=daily_ret.index)
    rebal_dates = momentum.dropna(how='all').index

    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        next_rd = rebal_dates[i + 1]
        signal = momentum.loc[rd].dropna()
        if len(signal) < top_n + bot_n + 1:
            continue
        sorted_sig = signal.sort_values()
        shorts = sorted_sig.head(bot_n).index.tolist()
        longs = sorted_sig.tail(top_n).index.tolist()
        hold_mask = (daily_ret.index > rd) & (daily_ret.index <= next_rd)
        hold_dates = daily_ret.index[hold_mask]
        for d in hold_dates:
            long_ret = daily_ret.loc[d, longs].dropna().mean()
            short_ret = daily_ret.loc[d, shorts].dropna().mean()
            if not (np.isnan(long_ret) or np.isnan(short_ret)):
                out.loc[d] = long_ret - short_ret
    return out


def summarize(name: str, daily: pd.Series):
    daily = daily.dropna()
    sh = annualized_sharpe(daily)
    nw = newey_west_t(daily, 60)
    vol_ann = float(daily.std(ddof=1) * np.sqrt(252))
    ret_ann = float(daily.mean() * 252)
    mdd = max_drawdown(daily)
    cum = float((1 + daily).cumprod().iloc[-1] - 1)

    pre = daily.loc[:"2019-12-31"]
    covid = daily.loc["2020-01-01":"2021-12-31"]
    post = daily.loc["2022-01-01":]

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    print(f"  n daily obs:        {len(daily)}")
    print(f"  Date range:         {daily.index[0].date()} → {daily.index[-1].date()}")
    print(f"  Ann return:         {ret_ann * 100:+7.2f}%/yr")
    print(f"  Ann vol:            {vol_ann * 100:7.2f}%")
    print(f"  Sharpe:             {sh:+.4f}")
    print(f"  NW t (lag 60):      {nw:+.4f}")
    print(f"  Cumulative:         {cum * 100:+7.2f}%")
    print(f"  Max DD:             {mdd * 100:+7.2f}%")
    print(f"  Sub-period Sharpe:")
    sh_pre = annualized_sharpe(pre); sh_covid = annualized_sharpe(covid); sh_post = annualized_sharpe(post)
    print(f"    Pre-COVID  (2014-2019): {sh_pre:+.4f}")
    print(f"    COVID      (2020-2021): {sh_covid:+.4f}")
    print(f"    Post-COVID (2022-2023): {sh_post:+.4f}")
    all_pos = all(s > 0 for s in [sh_pre, sh_covid, sh_post] if not np.isnan(s))
    print(f"  All sub-period positive: {all_pos}")
    gate_pass = sh >= 0.4 and nw >= 1.8
    print(f"  ETF Gate 1 (Sharpe ≥ 0.4 + NW t ≥ 1.8): {'PASS' if gate_pass else 'FAIL'}")


def main():
    print("Fetching thematic ETF universe from yfinance (2014-2023)...")

    # 22 thematic ETFs with 2014+ history (verified)
    tickers = [
        "ARKK", "ARKQ", "ARKW", "ARKG",   # ARK innovation
        "ROBO",                            # Robotics+AI
        "SMH", "SOXX", "XSD",              # Semiconductor
        "HACK",                            # Cybersecurity
        "KWEB",                            # China internet
        "IPO",                             # IPO
        "ICLN", "TAN", "LIT", "REMX", "COPX", "SLX",  # Clean energy + commodity thematic
        "JETS", "AIRR",                    # Aviation/industrial
        "PBJ", "PEJ",                      # Consumer thematic
        "XLK", "XLE",                      # 2 sector ETFs for benchmark/comparison
    ]

    data = yf.download(tickers, start="2013-01-01", end="2023-12-31",
                       progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how='all').ffill()

    # Report data coverage
    print(f"\nData coverage per ETF (rows with data 2014+):")
    for t in tickers:
        if t in close.columns:
            n_2014 = close[t].loc["2014-01-01":].dropna().shape[0]
            first_date = close[t].dropna().index[0].date() if close[t].dropna().shape[0] > 0 else "—"
            print(f"  {t}: {n_2014} rows, first valid date {first_date}")

    close_in = close.loc["2014-01-01":"2023-12-31"]

    # Run cross-section momentum L-S (top-3 / bot-3)
    print(f"\nUniverse total: {len(tickers)} ETFs; L-S = top-3 / bottom-3 by 12-1 momentum")
    ret = cross_section_long_short(close_in, lookback_months=12, skip_months=1, top_n=3, bot_n=3)
    summarize("Path B scout — Thematic ETF Momentum (12-1, top-3/bot-3, monthly)", ret)

    # Also long-only top-3 for comparison
    print("\n" + "=" * 70)
    print("Long-only top-3 (no short leg, for reference)")
    print("=" * 70)
    monthly_prices = close_in.resample('ME').last()
    sig_numer = monthly_prices.shift(1)
    sig_denom = monthly_prices.shift(12)
    momentum = sig_numer / sig_denom - 1.0
    daily_ret = close_in.pct_change()
    long_only = pd.Series(0.0, index=daily_ret.index)
    rebal_dates = momentum.dropna(how='all').index
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates): break
        next_rd = rebal_dates[i + 1]
        signal = momentum.loc[rd].dropna()
        if len(signal) < 4: continue
        longs = signal.sort_values().tail(3).index.tolist()
        hold_mask = (daily_ret.index > rd) & (daily_ret.index <= next_rd)
        for d in daily_ret.index[hold_mask]:
            r = daily_ret.loc[d, longs].dropna().mean()
            if not np.isnan(r):
                long_only.loc[d] = r
    summarize("Long-only top-3", long_only)

    print("\n" + "=" * 70)
    print("HONEST DISCLOSE")
    print("=" * 70)
    print("- 22 ETF universe; L=3/S=3 → tiny legs, high variance")
    print("- No TC drag applied; ~3-5bp roundtrip × 12 = 0.4-0.6%/yr drag at monthly turnover")
    print("- Thematic ETFs have higher TC (3-8bp roundtrip) than broad SPY (0.5bp); real friction worse")
    print("- Cherry-picked universe (other thematic ETFs exist: SOCL/BUG/FINX/BOTZ/SKYY/ESPO etc.)")
    print("- 2020-2021 ARKK super-rally + 2022 crash dominate Post-COVID stats")
    print("- Capacity small: ARKK $7B peak / niche thematic <$200M-$1B each — institutional sleeve limited")
    print("- Compare to project gates: ETF Sharpe ≥ 0.4 + NW t ≥ 1.8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
