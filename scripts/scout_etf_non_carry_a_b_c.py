"""
scripts/scout_etf_non_carry_a_b_c.py — Quick-and-dirty scout for ETF non-carry candidates.

NOT a backtest. NOT pre-registered. Just raw Sharpe + NW t to decide which is worth spec lock.

A: Sector Rotation (11 SPDR sectors, 12-1 momentum cross-sectional, top-3/bottom-3)
B: Cross-Asset TSMOM Ensemble (7 ETFs, 1mo/3mo/12mo lookback ensemble sign-based)
C: Factor Rotation Timing (4 factor ETFs MTUM/VLUE/QUAL/USMV, 12-1 momentum top-2/bottom-2)

Period: 2014-01-01 → 2023-12-31 (same as Path E/F/G/H — for fair comparison)
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


def summarize(name: str, daily: pd.Series):
    daily = daily.dropna()
    if len(daily) < 100:
        print(f"\n{name}: too few obs ({len(daily)})")
        return
    sh = annualized_sharpe(daily)
    nw = newey_west_t(daily, lag=60)
    vol_ann = float(daily.std(ddof=1) * np.sqrt(252))
    ret_ann = float(daily.mean() * 252)
    mdd = max_drawdown(daily)
    cum_total = float((1 + daily).cumprod().iloc[-1] - 1)

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
    print(f"  Date range:         {daily.index[0].date()} → {daily.index[-1].date()}")
    print(f"  Ann return:         {ret_ann * 100:+7.2f}%/yr")
    print(f"  Ann vol:            {vol_ann * 100:7.2f}%")
    print(f"  Sharpe:             {sh:+.4f}")
    print(f"  NW t (lag 60):      {nw:+.4f}")
    print(f"  Cumulative:         {cum_total * 100:+7.2f}%")
    print(f"  Max DD:             {mdd * 100:+7.2f}%")
    print(f"  Sub-period Sharpe:")
    print(f"    Pre-COVID  (2014-2019): {sh_pre:+.4f}")
    print(f"    COVID      (2020-2021): {sh_covid:+.4f}")
    print(f"    Post-COVID (2022-2023): {sh_post:+.4f}")
    sub_sharpes = [s for s in [sh_pre, sh_covid, sh_post] if not np.isnan(s)]
    all_positive = all(s > 0 for s in sub_sharpes)
    print(f"  All sub-period positive: {all_positive}")
    gate1_pass = sh >= 0.4 and nw >= 1.8
    print(f"  ETF Gate 1 (Sharpe ≥ 0.4 + NW t ≥ 1.8): {'PASS' if gate1_pass else 'FAIL'}")


def cross_section_long_short(
    daily_prices: pd.DataFrame,
    lookback_days: int,
    skip_days: int,
    top_n: int,
    bot_n: int,
) -> pd.Series:
    """At each month-end: rank ETFs by (lookback-skip) trailing return; long top_n, short bot_n; hold 1 month."""
    # Month-end resample to find rebalance dates
    monthly_prices = daily_prices.resample('M').last()
    # 12-1 momentum: trailing 12mo return minus trailing 1mo return
    if skip_days > 0:
        # Use price ratios: p[t] / p[t-lookback] minus p[t] / p[t-skip]
        # Equivalent to: total return lookback-to-skip excluding most recent skip days
        # In monthly: p[t-1mo] / p[t-12mo]
        skip_months = max(1, skip_days // 21)
    else:
        skip_months = 0
    lookback_months = max(1, lookback_days // 21)

    # Signal at month m: ratio of price[m-skip_months] to price[m-lookback_months]
    sig_numer = monthly_prices.shift(skip_months)
    sig_denom = monthly_prices.shift(lookback_months)
    momentum = sig_numer / sig_denom - 1.0  # trailing return excl. last skip

    daily_ret = daily_prices.pct_change()
    out_returns = pd.Series(0.0, index=daily_ret.index)

    rebal_dates = momentum.dropna(how='all').index
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        next_rd = rebal_dates[i + 1]
        signal = momentum.loc[rd].dropna()
        if len(signal) < top_n + bot_n + 1:
            continue
        signal_sorted = signal.sort_values()
        shorts = signal_sorted.head(bot_n).index.tolist()
        longs = signal_sorted.tail(top_n).index.tolist()
        # Hold from rd+1 trading day to next_rd inclusive
        hold_mask = (daily_ret.index > rd) & (daily_ret.index <= next_rd)
        hold_dates = daily_ret.index[hold_mask]
        for d in hold_dates:
            long_ret = daily_ret.loc[d, longs].dropna().mean()
            short_ret = daily_ret.loc[d, shorts].dropna().mean()
            if not (np.isnan(long_ret) or np.isnan(short_ret)):
                out_returns.loc[d] = long_ret - short_ret
    return out_returns


def tsmom_ensemble(daily_prices: pd.DataFrame) -> pd.Series:
    """Cross-asset TSMOM ensemble (Hurst-Ooi-Pedersen 2017): average sign of 1mo/3mo/12mo trailing return per asset."""
    monthly_prices = daily_prices.resample('M').last()
    sig_1mo = monthly_prices.pct_change(1)
    sig_3mo = monthly_prices.pct_change(3)
    sig_12mo = monthly_prices.pct_change(12)
    # Sign ensemble: -1 / 0 / +1 per asset
    pos = (np.sign(sig_1mo).fillna(0) + np.sign(sig_3mo).fillna(0) + np.sign(sig_12mo).fillna(0)) / 3.0
    # Position per asset = pos / N_assets (equal-weight across active positions)
    n_assets = pos.shape[1]
    daily_ret = daily_prices.pct_change()
    out_returns = pd.Series(0.0, index=daily_ret.index)

    rebal_dates = pos.dropna(how='all').index
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        next_rd = rebal_dates[i + 1]
        weights = pos.loc[rd].fillna(0) / n_assets
        hold_mask = (daily_ret.index > rd) & (daily_ret.index <= next_rd)
        hold_dates = daily_ret.index[hold_mask]
        for d in hold_dates:
            day_ret = daily_ret.loc[d]
            out_returns.loc[d] = (weights * day_ret).sum()
    return out_returns


def main():
    start, end = "2014-01-01", "2023-12-31"

    # ── A. Sector Rotation (11 SPDR sectors) ─────────────────────────────────
    # Note: XLRE listed 2015-10, XLC listed 2018-06 — partial universe pre-dates
    sector_tickers = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
    print(f"\nFetching A. Sector Rotation ({len(sector_tickers)} SPDR sectors)...")
    a_data = yf.download(sector_tickers, start="2013-01-01", end=end, progress=False, auto_adjust=True)
    a_close = a_data["Close"] if isinstance(a_data.columns, pd.MultiIndex) else a_data
    a_close = a_close.ffill()
    a_close_in = a_close.loc[start:end]
    a_ret = cross_section_long_short(a_close_in, lookback_days=252, skip_days=21, top_n=3, bot_n=3)
    summarize("Path A scout — Sector Rotation (top-3/bot-3, 12-1 momentum, monthly)", a_ret)

    # ── B. Cross-Asset TSMOM Ensemble (7 ETFs) ───────────────────────────────
    asset_tickers = ["SPY", "EFA", "EEM", "AGG", "TLT", "GLD", "DBC"]
    print(f"\nFetching B. Cross-Asset TSMOM ({len(asset_tickers)} ETFs)...")
    b_data = yf.download(asset_tickers, start="2013-01-01", end=end, progress=False, auto_adjust=True)
    b_close = b_data["Close"] if isinstance(b_data.columns, pd.MultiIndex) else b_data
    b_close = b_close.ffill()
    b_close_in = b_close.loc[start:end]
    b_ret = tsmom_ensemble(b_close_in)
    summarize("Path B scout — Cross-Asset TSMOM Ensemble (HOP-2017, 1mo/3mo/12mo sign-avg)", b_ret)

    # ── C. Factor Rotation Timing (4 factor ETFs) ────────────────────────────
    factor_tickers = ["MTUM", "VLUE", "QUAL", "USMV"]
    print(f"\nFetching C. Factor Rotation ({len(factor_tickers)} factor ETFs)...")
    c_data = yf.download(factor_tickers, start="2013-01-01", end=end, progress=False, auto_adjust=True)
    c_close = c_data["Close"] if isinstance(c_data.columns, pd.MultiIndex) else c_data
    c_close = c_close.ffill()
    c_close_in = c_close.loc[start:end]
    c_ret = cross_section_long_short(c_close_in, lookback_days=252, skip_days=21, top_n=1, bot_n=1)
    summarize("Path C scout — Factor Rotation (top-1/bot-1 among MTUM/VLUE/QUAL/USMV)", c_ret)

    print(f"\n{'=' * 70}")
    print("HONEST DISCLOSE (scout, NOT spec-locked backtest)")
    print(f"{'=' * 70}")
    print("- No TC drag applied")
    print("- A: 11 sectors but XLRE listed 2015-10 + XLC 2018-06; pre-2018 uses 9-10 sectors")
    print("- B: Equal-weight across active TSMOM positions (no vol-target, no leverage cap)")
    print("- C: Only 4 factor ETFs — universe small (top-1/bot-1)")
    print("- These are RAW yfinance upper bounds; signal-conditioning + TC will reduce Sharpe")
    print("- Compare to Path F (raw 0.41), Path E (raw -0.14 / +0.11), Path H (raw 0.028)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
