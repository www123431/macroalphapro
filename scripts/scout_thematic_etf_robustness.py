"""
scripts/scout_thematic_etf_robustness.py — Path B robustness check.

Tests if initial scout finding (Sharpe 0.66 NW t 2.27) is robust to:
  1. Expanded thematic ETF universe (~35 vs original 23)
  2. ARK family exclusion (drop ARKK/ARKQ/ARKW/ARKG)
  3. Leg size variation (top-3/bot-3 vs top-5/bot-5 vs top-2/bot-2)
  4. ARKK-only standalone for reference (is "thematic mom" really "ride ARKK"?)

NOT a backtest. Pure robustness audit before considering spec lock.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import yfinance as yf


def sharpe(x, p=252):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(p)) if x.std(ddof=1) > 0 else 0.0


def nw_t(x, lag=60):
    v = x.dropna().values
    n = len(v)
    if n < lag + 1: return np.nan
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


def cross_section_ls(prices, top_n=3, bot_n=3, lookback=12, skip=1):
    monthly = prices.resample('ME').last()
    mom = monthly.shift(skip) / monthly.shift(lookback) - 1.0
    daily_ret = prices.pct_change()
    out = pd.Series(0.0, index=daily_ret.index)
    rebals = mom.dropna(how='all').index
    for i, rd in enumerate(rebals):
        if i + 1 >= len(rebals): break
        next_rd = rebals[i + 1]
        sig = mom.loc[rd].dropna()
        if len(sig) < top_n + bot_n + 1: continue
        sorted_sig = sig.sort_values()
        shorts = sorted_sig.head(bot_n).index.tolist()
        longs = sorted_sig.tail(top_n).index.tolist()
        hold = (daily_ret.index > rd) & (daily_ret.index <= next_rd)
        for d in daily_ret.index[hold]:
            lr = daily_ret.loc[d, longs].dropna().mean()
            sr = daily_ret.loc[d, shorts].dropna().mean()
            if not (np.isnan(lr) or np.isnan(sr)):
                out.loc[d] = lr - sr
    return out


def summarize(name, x, indent=""):
    x = x.dropna()
    sh = sharpe(x); nw = nw_t(x, 60)
    pre = sharpe(x.loc[:"2019-12-31"])
    cov = sharpe(x.loc["2020-01-01":"2021-12-31"])
    pos = sharpe(x.loc["2022-01-01":])
    cum = float((1 + x).cumprod().iloc[-1] - 1)
    dd = max_dd(x)
    all_pos = all(s > 0 for s in [pre, cov, pos] if not np.isnan(s))
    gate = "PASS" if sh >= 0.4 and nw >= 1.8 else "FAIL"
    print(f"{indent}{name:<55} Sh={sh:+.3f}  NWt={nw:+.3f}  "
          f"Pre/Cov/Post=[{pre:+.2f}/{cov:+.2f}/{pos:+.2f}]  "
          f"cum={cum*100:+6.0f}%  DD={dd*100:+.0f}%  AllPos={all_pos}  Gate={gate}")


def main():
    print("Fetching expanded thematic ETF universe...")

    # Expanded universe: ~35 ETFs across innovation/thematic/sector themes
    ALL_TICKERS = {
        # Innovation/Disruption family (key bubble + crash exposure)
        "ARKK": "ARK Innovation",
        "ARKQ": "ARK Robotics",
        "ARKW": "ARK Web 3.0",
        "ARKG": "ARK Genomic",
        # Tech-thematic
        "ROBO": "Robotics+AI",
        "BOTZ": "Global X Robotics (2016)",
        "SKYY": "First Trust Cloud",
        "FINX": "Global X Fintech (2016)",
        "HACK": "Cybersecurity",
        "CIBR": "First Trust Cybersec (2015)",
        "SOCL": "Global X Social Media",
        "IGV":  "iShares Software",
        "ESPO": "VanEck Gaming (2018)",
        "GAMR": "Wedbush Video Game (2016)",
        # Semiconductor
        "SMH":  "VanEck Semis",
        "SOXX": "iShares Semis",
        "XSD":  "SPDR Semis",
        # China + EM thematic
        "KWEB": "China Internet",
        # IPO/SPAC
        "IPO":  "Renaissance IPO",
        # Energy/commodity-thematic
        "ICLN": "Clean Energy",
        "TAN":  "Solar",
        "LIT":  "Lithium",
        "REMX": "Rare Earth",
        "COPX": "Copper Miners",
        "SLX":  "Steel",
        # Aviation + industrial-thematic
        "JETS": "Aviation",
        "AIRR": "Industrial",
        "ITA":  "Aerospace+Defense",
        # Consumer thematic
        "PBJ":  "Food+Beverage",
        "PEJ":  "Leisure+Entertainment",
        "IBUY": "Online Retail (2016)",
        # Biotech
        "IBB":  "Biotech (large)",
        "XBI":  "Biotech (equal-weight)",
        # Sector benchmarks (for comparison; KEEP IN UNIVERSE for breadth)
        "XLE":  "Energy sector",
        "XLK":  "Tech sector",
    }

    tickers = list(ALL_TICKERS.keys())
    print(f"Universe: {len(tickers)} ETFs")

    data = yf.download(tickers, start="2013-01-01", end="2023-12-31",
                       progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how='all').ffill()

    # Filter to ETFs with 2014+ data of at least 1500 valid days
    valid_tickers = []
    coverage = {}
    for t in tickers:
        if t not in close.columns: continue
        n = close[t].loc["2014-01-01":].dropna().shape[0]
        coverage[t] = n
        if n >= 1500:  # ~6 years
            valid_tickers.append(t)

    print(f"\nValid tickers (≥1500 days post-2014): {len(valid_tickers)}")
    short_history = [t for t in tickers if coverage.get(t, 0) < 1500]
    if short_history:
        print(f"Excluded (short history): {short_history}")

    close_in = close[valid_tickers].loc["2014-01-01":"2023-12-31"]
    print()

    # ── 1. Baseline (original scout, 22-ETF universe) ─────────────────────────
    print("=" * 80)
    print("BASELINE: full expanded universe, top-3/bot-3, 12-1 monthly")
    print("=" * 80)
    print(f"{'Universe':<55} {'Stats':<60}")
    full_ls = cross_section_ls(close_in, top_n=3, bot_n=3)
    summarize(f"Full universe (n={len(valid_tickers)})", full_ls)

    # ── 2. ARK exclusion test (drop ARK family entirely) ──────────────────────
    print("\n" + "=" * 80)
    print("ARK EXCLUSION TEST: drop ARKK/ARKQ/ARKW/ARKG, re-run")
    print("=" * 80)
    no_ark = close_in.drop(columns=[t for t in ["ARKK", "ARKQ", "ARKW", "ARKG"] if t in close_in.columns])
    print(f"Universe sans ARK: {no_ark.shape[1]} ETFs")
    no_ark_ls = cross_section_ls(no_ark, top_n=3, bot_n=3)
    summarize(f"No-ARK (n={no_ark.shape[1]})", no_ark_ls)

    # ── 3. ARK-only standalone test ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ARK-ONLY: is the signal really just 'ride ARKK'?")
    print("=" * 80)
    ark_only = close_in[[t for t in ["ARKK", "ARKQ", "ARKW", "ARKG"] if t in close_in.columns]]
    print(f"ARK universe: {ark_only.shape[1]} ETFs")
    if ark_only.shape[1] >= 4:
        ark_ls = cross_section_ls(ark_only, top_n=1, bot_n=1)
        summarize(f"ARK-only top-1/bot-1", ark_ls)

    # ── 4. Leg size variations ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LEG SIZE VARIATIONS (full universe)")
    print("=" * 80)
    for top, bot in [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]:
        if top + bot < len(valid_tickers):
            x = cross_section_ls(close_in, top_n=top, bot_n=bot)
            summarize(f"top-{top}/bot-{bot}", x)

    # ── 5. Lookback variations (HARKing risk — but informative) ───────────────
    print("\n" + "=" * 80)
    print("LOOKBACK VARIATIONS (top-3/bot-3, MOSTLY for HARKing risk audit)")
    print("=" * 80)
    print("NOTE: spec lock should use 12-1 canonical; these are noise checks.")
    for lb in [6, 9, 12, 15]:
        x = cross_section_ls(close_in, top_n=3, bot_n=3, lookback=lb, skip=1)
        summarize(f"lookback={lb}mo, skip=1mo", x)

    # ── 6. Long-only variant ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LONG-ONLY VARIANTS (more realistic capacity)")
    print("=" * 80)
    monthly = close_in.resample('ME').last()
    mom = monthly.shift(1) / monthly.shift(12) - 1.0
    daily_ret = close_in.pct_change()
    for top_n in [3, 5]:
        long_only = pd.Series(0.0, index=daily_ret.index)
        rebals = mom.dropna(how='all').index
        for i, rd in enumerate(rebals):
            if i + 1 >= len(rebals): break
            sig = mom.loc[rd].dropna()
            if len(sig) < top_n + 1: continue
            longs = sig.sort_values().tail(top_n).index.tolist()
            hold = (daily_ret.index > rd) & (daily_ret.index <= rebals[i + 1])
            for d in daily_ret.index[hold]:
                r = daily_ret.loc[d, longs].dropna().mean()
                if not np.isnan(r):
                    long_only.loc[d] = r
        summarize(f"Long-only top-{top_n}", long_only)

    # ── 7. Compare to S&P 500 buy-hold baseline ──────────────────────────────
    print("\n" + "=" * 80)
    print("BENCHMARK: S&P 500 buy-and-hold (SPY)")
    print("=" * 80)
    spy = yf.download("SPY", start="2014-01-01", end="2024-01-01", progress=False, auto_adjust=True)["Close"]
    if hasattr(spy, 'columns'):
        spy = spy.iloc[:, 0]
    spy_ret = spy.pct_change().dropna()
    summarize("SPY buy-and-hold", spy_ret)

    print("\n" + "=" * 80)
    print("VERDICT INTERPRETATION")
    print("=" * 80)
    print("- If full vs no-ARK ≈ same Sharpe → signal is broader thematic momentum")
    print("- If full >> no-ARK → signal is mostly ARK-driven (bubble-cycle bet, not robust)")
    print("- ARK-only Sharpe shows ARK 内部 momentum 强度 baseline")
    print("- Leg size variation: if Sharpe stable across top-2/3/4/5 → robust signal")
    print("- Lookback variation: only informational; spec must lock 12-1 ex-ante")
    return 0


if __name__ == "__main__":
    sys.exit(main())
