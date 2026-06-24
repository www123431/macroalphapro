"""scripts/fetch_jp_yfinance_returns.py — bypass WRDS linking via
yfinance + IBES oftic as TSE 4-digit code.

JP firms in IBES intl use oftic = TSE code (e.g. '6030' = ADVENTURE).
Yahoo Finance uses '6030.T' for Tokyo Stock Exchange. Direct mapping
gets us JP daily returns without WRDS linking suite friction.

Trade-off (documented):
  + No WRDS linking
  + Fast (~1 min for top 500)
  + Adequate for research/exploration
  - Survivorship bias (delisted firms missing)
  - Adjustment quality varies
  - Not institutional-grade — use to TRIAGE, then go to WRDS if GREEN

For F1-F5 falsification this is acceptable since:
  F1 t-stat at n>=5000 events is robust to mild data noise
  F2 monthly Sharpe is robust to sub-percent return errors
  Failing F1/F2 on yfinance data = real failure (graveyard reinforced)
  Passing on yfinance data = warrants WRDS confirmation pass
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("yf_jp")

REPO_ROOT = Path(__file__).resolve().parent.parent
JP_EPS = REPO_ROOT / "data" / "cache" / "_jp_ibes_eps_actuals.parquet"
RET_OUT = REPO_ROOT / "data" / "cache" / "_jp_returns_yfinance.parquet"


def main(top_n: int = 500) -> int:
    print("=" * 80)
    print(f" Fetch JP returns via yfinance (top {top_n} TSE codes)")
    print("=" * 80)

    # Load IBES JP EPS panel — get top-N firms by EPS event count
    eps = pd.read_parquet(JP_EPS)
    # oftic should be the TSE code (4 digits) — filter
    eps_clean = eps.dropna(subset=["oftic"]).copy()
    eps_clean["oftic"] = eps_clean["oftic"].astype(str).str.strip()
    # Keep only 4-digit numeric oftics (real TSE codes)
    eps_clean = eps_clean[eps_clean["oftic"].str.match(r"^\d{4}$")]
    print(f"  EPS panel after TSE-code filter: {len(eps_clean):,} rows "
          f"({eps_clean['oftic'].nunique()} unique TSE codes)")

    top_firms = (eps_clean.groupby("oftic").size()
                 .sort_values(ascending=False)
                 .head(top_n).index.tolist())
    print(f"  selected top {len(top_firms)} TSE codes by event count")

    yahoo_tickers = [f"{t}.T" for t in top_firms]
    print(f"  sample: {yahoo_tickers[:5]}")

    # Batch download
    print(f"\n[yfinance batch download]")
    t0 = time.time()
    data = yf.download(
        yahoo_tickers,
        start="2014-01-01", end="2024-12-31",
        progress=True, auto_adjust=True, threads=True,
    )
    elapsed = time.time() - t0
    print(f"  download time: {elapsed:.1f}s")

    if data.empty:
        print(f"  FAIL: no data returned")
        return 1

    # data is MultiIndex columns: (field, ticker). Extract Close.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    print(f"  close-price panel shape: {close.shape}")

    # Compute daily returns
    returns = close.pct_change()
    # Strip .T suffix for cleaner column names matching IBES oftic
    returns.columns = [c.replace(".T", "") for c in returns.columns]

    n_firms_with_data = returns.notna().any().sum()
    print(f"  firms with non-empty returns: {n_firms_with_data}/{len(returns.columns)}")

    # Save
    RET_OUT.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(RET_OUT)
    print(f"\n[saved] {RET_OUT}")
    print(f"  shape: {returns.shape}")
    print(f"  date range: {returns.index.min().date()} -> {returns.index.max().date()}")

    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    sys.exit(main(top_n=n))
