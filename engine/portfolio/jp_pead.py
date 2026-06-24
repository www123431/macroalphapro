"""engine/portfolio/jp_pead.py — Japan PEAD strategy build.

Methodology (Bernard-Thomas 1989 baseline, simplified for first pass):
  - Time-series SUE = (EPS_q - EPS_q-4) / std(EPS_q - EPS_q-4 over 8q)
  - Universe-wide decile L/S (no sector neutralization yet — defer to
    PIT SN extension if first pass GREEN-leaning)
  - Hold 60 trading days post-announcement
  - Monthly aggregate via rolling event basket

Anti-overfit guard: parameters are all from Bernard-Thomas 1989
canonical — 8q rolling sigma, 4q lag, 60d hold. No tuning.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
JP_EPS = REPO_ROOT / "data" / "cache" / "_jp_ibes_eps_actuals.parquet"
JP_RET = REPO_ROOT / "data" / "cache" / "_jp_returns_yfinance.parquet"
OUT_MONTHLY = REPO_ROOT / "data" / "cache" / "_jp_pead_monthly.parquet"
OUT_DIAG = REPO_ROOT / "data" / "cache" / "_jp_pead_event_diag.parquet"

SUE_LOOKBACK_Q = 8
SUE_LAG_Q = 4
HOLD_DAYS = 60
DECILE = 0.10
MIN_FIRMS_PER_DAY = 20
MIN_PEER_GROUP = 8         # if sector-neutral added later


def _build_sue_panel(eps: pd.DataFrame) -> pd.DataFrame:
    """Build per (oftic, anndats) SUE = (EPS - EPS_lag4) / std_8q."""
    df = eps.dropna(subset=["oftic", "anndats", "eps_actual"]).copy()
    df["anndats"] = pd.to_datetime(df["anndats"])
    df["oftic"] = df["oftic"].astype(str).str.strip()
    df = df[df["oftic"].str.match(r"^\d{4}$")]
    df = df.sort_values(["oftic", "anndats"])

    # Per-firm: lag-4 EPS + rolling sigma of (EPS - EPS_lag4)
    df["eps_lag4"] = df.groupby("oftic")["eps_actual"].shift(SUE_LAG_Q)
    df["delta_eps"] = df["eps_actual"] - df["eps_lag4"]
    df["sigma_8q"] = (df.groupby("oftic")["delta_eps"]
                        .rolling(SUE_LOOKBACK_Q, min_periods=4).std()
                        .reset_index(0, drop=True))
    df["sue"] = df["delta_eps"] / df["sigma_8q"].replace(0, np.nan)
    df = df.dropna(subset=["sue"])
    # Clip extreme outliers (winsorize at ±5)
    df["sue"] = df["sue"].clip(-5, 5)
    return df[["oftic", "anndats", "eps_actual", "delta_eps", "sue"]]


def build_jp_pead_returns() -> pd.Series:
    """Build daily long-short return series + aggregate to monthly."""
    logger.info("loading inputs")
    eps = pd.read_parquet(JP_EPS)
    ret = pd.read_parquet(JP_RET)
    ret.index = pd.to_datetime(ret.index)
    # ret columns are TSE codes (e.g. '6030')

    logger.info("building SUE panel")
    sue = _build_sue_panel(eps)
    logger.info(f"  SUE panel: {len(sue):,} events / {sue['oftic'].nunique()} firms")

    # Restrict to firms with yfinance return data
    valid_tickers = set(ret.columns)
    sue = sue[sue["oftic"].isin(valid_tickers)]
    logger.info(f"  after returns-coverage filter: {len(sue):,} events / "
                f"{sue['oftic'].nunique()} firms")

    if sue.empty:
        raise RuntimeError("no overlap between IBES EPS and yfinance returns")

    # For each trading day, find active events (announce within last HOLD_DAYS)
    sue = sue.sort_values("anndats")
    anndats = sue["anndats"].values
    rows = []
    event_log = []

    for t in ret.index:
        lo_idx = np.searchsorted(anndats,
                                  np.datetime64(t - pd.Timedelta(days=HOLD_DAYS)),
                                  "right")
        hi_idx = np.searchsorted(anndats, np.datetime64(t), "right")
        if hi_idx - lo_idx < MIN_FIRMS_PER_DAY:
            continue

        win = sue.iloc[lo_idx:hi_idx]
        # Per firm latest SUE in window (multiple events overlap rarely)
        latest = win.groupby("oftic").last().reset_index()
        if len(latest) < MIN_FIRMS_PER_DAY:
            continue

        # Rank decile L/S
        thr_top = latest["sue"].quantile(1 - DECILE)
        thr_bot = latest["sue"].quantile(DECILE)
        long_tickers = latest.loc[latest["sue"] >= thr_top, "oftic"].tolist()
        short_tickers = latest.loc[latest["sue"] <= thr_bot, "oftic"].tolist()

        # Pick returns for that day
        r = ret.loc[t]
        l_rets = r.reindex(long_tickers).dropna()
        s_rets = r.reindex(short_tickers).dropna()
        if len(l_rets) < 5 or len(s_rets) < 5:
            continue
        ls = float(l_rets.mean() - s_rets.mean())
        rows.append((t, ls, len(l_rets), len(s_rets), len(latest)))

    if not rows:
        raise RuntimeError("no daily L/S returns produced")
    daily = pd.DataFrame(rows, columns=["date", "ls_return", "n_long",
                                          "n_short", "n_universe"])
    daily = daily.set_index("date").sort_index()
    daily.to_parquet(OUT_DIAG)
    logger.info(f"  daily L/S: {len(daily):,} days")

    monthly = ((1 + daily["ls_return"].clip(-0.2, 0.2)).resample("ME").prod() - 1)
    monthly.to_frame("jp_pead").to_parquet(OUT_MONTHLY)
    return monthly.rename("jp_pead")


def main():
    import math
    logging.basicConfig(level=logging.INFO)
    m = build_jp_pead_returns()
    ann_ret = float(m.mean() * 12)
    ann_vol = float(m.std() * math.sqrt(12))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    print("=" * 70)
    print(f" JP PEAD (yfinance, universe-wide decile L/S, no sector neutralization)")
    print("=" * 70)
    print(f"  n_months: {len(m)}")
    print(f"  date range: {m.index.min().date()} → {m.index.max().date()}")
    print(f"  ann return: {ann_ret:+.4f} ({ann_ret*100:+.2f}%/yr)")
    print(f"  ann vol:    {ann_vol:.4f}")
    print(f"  Sharpe:     {sharpe:+.3f}")
    print(f"  win rate:   {(m > 0).mean():.1%}")
    print(f"  saved:      {OUT_MONTHLY}")


if __name__ == "__main__":
    main()
