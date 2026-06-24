"""scripts/fetch_real_adv_per_sleeve.py

Fetch real per-ticker 60-day Average Daily Dollar Volume (ADV) for representative
basket of each sleeve. Used by Capacity Sim v2 to replace class-based ADV proxy.

Output: data/portfolio_replay/sleeve_adv_real_2026-05-15.json
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yfinance as yf
import numpy as np


# Representative basket per sleeve (10 tickers each, picked for coverage)
SLEEVE_REPRESENTATIVE_TICKERS = {
    "K1_BAB": [
        # SPDR sectors (representative liquid ETFs)
        "XLK", "XLF", "XLV", "XLI", "XLY", "XLE", "XLP", "XLU", "XLRE",
        # Broad-market + size/style
        "SPY", "QQQ", "IWB", "IWM", "MTUM", "USMV",
    ],
    "D_PEAD": [
        # Top S&P 500 by market cap, representative for D-PEAD's top-1500 universe
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK-B", "V", "JPM", "JNJ", "PG", "MA", "HD", "XOM",
        # Mid-cap representatives (lower decile of D-PEAD typical names)
        "MGM", "ETSY", "FCX", "HBAN", "CFG",
    ],
    "PATH_N": [
        # Recent S&P 500 NEW additions (2022-2024) — Path N's actual targets
        # Mid-cap pre-inclusion typical
        "AXON",   # added 2024
        "DECK",   # added 2024
        "PWR",    # added 2024
        "MOH",    # added 2024
        "EQT",    # added 2024
        "INSM",   # mid-cap typical Path N target
        "BLDR",   # added 2024
        "BX",     # large but recent add
        "VRTX",   # large
        "ENPH",   # mid
        "GEHC",   # 2023 split add
    ],
    "AC_TLT_GLD": [
        "TLT", "GLD",
    ],
    # CTA-PQTIX: mutual fund, no ADV (NAV-priced)
}


def fetch_adv_60d(tickers: list[str]) -> dict[str, dict]:
    """Fetch 60-day average daily $-volume per ticker.

    Returns: {ticker: {"adv_usd": float, "adv_volume_shares": float,
                        "n_days_used": int, "median_close": float,
                        "vol_uncertainty_pct": float}}
    """
    today = datetime.date.today()
    start = today - datetime.timedelta(days=120)   # 120 calendar days = ~80 trading days

    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker, start=start, end=today,
                progress=False, auto_adjust=True, multi_level_index=False,
            )
            if df.empty or "Volume" not in df.columns or "Close" not in df.columns:
                results[ticker] = {"error": "no data"}
                continue
            df["dollar_volume"] = df["Volume"] * df["Close"]
            tail = df.tail(60)
            if len(tail) < 30:
                results[ticker] = {"error": f"only {len(tail)} days"}
                continue
            adv = float(tail["dollar_volume"].mean())
            adv_std = float(tail["dollar_volume"].std())
            uncertainty = adv_std / adv if adv > 0 else 0
            results[ticker] = {
                "adv_usd":             adv,
                "adv_volume_shares":   float(tail["Volume"].mean()),
                "median_close":        float(tail["Close"].median()),
                "n_days_used":         int(len(tail)),
                "vol_uncertainty_pct": float(uncertainty),
            }
        except Exception as e:
            results[ticker] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    return results


def aggregate_sleeve_stats(sleeve_name: str, ticker_advs: dict) -> dict:
    """Aggregate per-sleeve median + percentile ADV stats."""
    valid = [d for d in ticker_advs.values() if "adv_usd" in d]
    if not valid:
        return {"sleeve": sleeve_name, "error": "no valid tickers"}

    advs = sorted(d["adv_usd"] for d in valid)
    uncertainties = [d["vol_uncertainty_pct"] for d in valid]

    return {
        "sleeve":             sleeve_name,
        "n_tickers_sampled":  len(valid),
        "adv_min_usd":        advs[0],
        "adv_p10_usd":        advs[max(0, int(0.1 * len(advs)) - 1)],
        "adv_median_usd":     advs[len(advs) // 2],
        "adv_mean_usd":       sum(advs) / len(advs),
        "adv_p90_usd":        advs[min(len(advs) - 1, int(0.9 * len(advs)))],
        "adv_max_usd":        advs[-1],
        "median_uncertainty": sorted(uncertainties)[len(uncertainties) // 2],
    }


def main() -> int:
    print("=== Fetching real per-ticker 60-day ADV ===")
    print()
    today = datetime.date.today()

    all_results: dict[str, dict] = {}
    for sleeve, tickers in SLEEVE_REPRESENTATIVE_TICKERS.items():
        print(f"Fetching {sleeve} ({len(tickers)} tickers): ", end="")
        adv = fetch_adv_60d(tickers)
        valid_count = sum(1 for d in adv.values() if "adv_usd" in d)
        print(f"{valid_count}/{len(tickers)} valid")
        stats = aggregate_sleeve_stats(sleeve, adv)
        all_results[sleeve] = {"per_ticker": adv, "stats": stats}

    print()
    print("=== Sleeve ADV Summary (60-day rolling) ===")
    print(f"{'Sleeve':<14} {'n':>4} {'Min':>12} {'P10':>12} {'Median':>12} {'Mean':>12} {'P90':>12} {'Max':>12}")
    print("-" * 96)
    for sleeve, data in all_results.items():
        s = data["stats"]
        if "error" in s:
            continue
        print(f"{sleeve:<14} {s['n_tickers_sampled']:>4} "
              f"{s['adv_min_usd']/1e6:>10.1f}M "
              f"{s['adv_p10_usd']/1e6:>10.1f}M "
              f"{s['adv_median_usd']/1e6:>10.1f}M "
              f"{s['adv_mean_usd']/1e6:>10.1f}M "
              f"{s['adv_p90_usd']/1e6:>10.1f}M "
              f"{s['adv_max_usd']/1e6:>10.1f}M")

    # Save
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"sleeve_adv_real_{today.isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "fetch_date": today.isoformat(),
        "method": "yfinance 60-day rolling Volume × Close",
        "sleeves": all_results,
    }, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
