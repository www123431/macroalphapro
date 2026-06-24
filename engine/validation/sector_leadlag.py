"""engine/validation/sector_leadlag.py — Sector lead-lag POC (DGNSDE-class minimal viable).

Phase 2 §I.C of research_agenda_2026-05-29. After 3 consecutive equity-factor
RED candidates (Quality / VIX / Residual Momentum, all on 2013-2024 sample), this
tests a STRUCTURALLY DIFFERENT mechanism class — lead-lag information transmission
between linked instruments — that we have not tested before.

Academic anchor
---------------
- Hong, Lim, Stein 2000 "Bad News Travels Slowly" (J. of Finance): information
  diffuses through prices with delay; sectors with high analyst dispersion lag
  the market
- Cohen, Frazzini 2008 "Economic Links and Predictable Returns" (JF): customer
  shocks propagate to supplier stock returns with delay
- DGNSDE 2026 (You et al, WWW): formalizes lead-lag via DTW on time-series
  graph + SDE node encoding. Our POC tests the CORE INSIGHT (delayed info
  transmission) without the full ML machinery, in a clean tractable universe.

Construction (pre-committed)
----------------------------
- Universe: 11 SPDR sector ETFs (XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY/XLRE).
  These are the standard GICS sector ETFs, liquid (~$10B+ AUM each), free data,
  long sample (2009-2026 for XLC; 1998-2026 for the others).
- Lead-lag identification: at each month t, for each pair (i, j) compute
  cross-correlation at lags 1..5 days over past 252 trading days. For each
  sector i, identify "predictor sectors" P_i = top-3 with max(|corr_lag_k|)
  > 0.15 threshold. This is the discovered "leader" set.
- Signal at month t for sector i: weighted sum of predictor sectors' returns
  over their identified lag windows, weighted by correlation magnitude.
- Cross-sectional ranking among 11 sectors: long top 3 by signal, short
  bottom 3, equal-weighted within each leg.
- Monthly rebalance.
- Costs: 5 bp/side (ETFs are liquid).

Pre-committed parameters (NO grid search per
[[feedback-strict-gate-no-lowering-2026-05-28]]):
- xcorr_lookback_days = 252 (1 year)
- max_lag_days = 5
- predictor_threshold = 0.15 absolute correlation
- top_k_predictors = 3
- top_q_long = 3, bottom_q_short = 3 (out of 11 sectors)
- exec_cost_bps = 5.0

Strict-gate framing
-------------------
- Single pre-committed implementation; the canonical Hong-Lim-Stein direction
- Sample: 2010-2026 (~190 months after warmup) — long enough to span multiple
  regimes (2010-2014 calm, 2015 selloff, 2018 vol-mageddon, 2020 COVID,
  2022 rate-hike cycle, 2024 calm)
- Mechanism is structurally DIFFERENT from earnings (D-PEAD), roll-yield
  (carry), trend (TSMOM), profitability (Quality) — no in-house adjacency
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Pre-committed parameters ─────────────────────────────────────────────────
SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK",
                  "XLP", "XLU", "XLV", "XLY"]   # 9 canonical SPDRs (1998+)
# Note: XLC (Communications, 2018) and XLRE (Real Estate, 2015) deliberately
# excluded — the 9-sector set is the canonical pre-2018 SPDR universe and
# gives 25+ years of testable history. Adding the two newer sectors would
# crash the intersection to 8 years for trivial added breadth.

XCORR_LOOKBACK_DAYS = 252        # 1 trading year for cross-correlation estimation
MAX_LAG_DAYS = 5                 # consider lead-lag up to 5 trading days
PREDICTOR_THRESHOLD = 0.15       # min |corr| to qualify as predictor
TOP_K_PREDICTORS = 3             # use top-3 predictor sectors per target
TOP_Q_LONG = 3                   # long top 3 of 11 sectors (~28%)
BOTTOM_Q_SHORT = 3               # short bottom 3 of 11 sectors
EXEC_COST_BPS = 5.0              # ETFs are very liquid
MIN_OBS_FOR_XCORR = 200          # minimum window obs for stable correlation


@lru_cache(maxsize=1)
def _fetch_sector_etfs() -> pd.DataFrame:
    """Pull 11 sector ETF daily closes from yfinance. Cached on disk."""
    import os
    cache_path = "data/cache/_sector_leadlag_panel.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df

    import yfinance as yf
    data = yf.download(SECTOR_TICKERS, period="max",
                       progress=False, auto_adjust=True)
    close = data["Close"].dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close.to_parquet(cache_path)
    return close


def _lead_lag_xcorr(returns_window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For a (T, N) returns window, compute pairwise lead-lag correlation at
    lags 1..MAX_LAG_DAYS.

    Returns:
      best_corr[i, j]  = max over lag k of corr(ret_j(t-k), ret_i(t))
                          — j leading i by k days
      best_lag[i, j]   = the k at which the max was attained
    """
    arr = returns_window.values   # (T, N)
    T, N = arr.shape
    best_corr = np.zeros((N, N))
    best_lag = np.ones((N, N), dtype=int)
    if T < MIN_OBS_FOR_XCORR:
        return best_corr, best_lag

    # Compute per-pair max-abs-corr over the lag spectrum
    # vectorized: for each lag k, corr(arr[k:, i], arr[:-k, j])
    for k in range(1, MAX_LAG_DAYS + 1):
        if T - k < 100:
            continue
        x = arr[k:, :]           # ret_i at time t   (T-k, N)
        y = arr[:-k, :]          # ret_j at time t-k (T-k, N)
        # Standardize each column
        x_z = (x - x.mean(axis=0)) / (x.std(axis=0, ddof=1) + 1e-12)
        y_z = (y - y.mean(axis=0)) / (y.std(axis=0, ddof=1) + 1e-12)
        # Pairwise corr matrix: (N, N) where corr[i, j] = corr(x_i, y_j)
        # i.e. corr(ret_i at t, ret_j at t-k) — j leading i by k
        n = x_z.shape[0]
        corr = (y_z.T @ x_z) / n
        # corr matrix: corr[j, i] = corr(y_j_lag_k, x_i_now) = corr(ret_j(t-k), ret_i(t))
        # j leads i by k. Transpose to put leader axis as columns.
        # We want best_corr[i, j] = max over k of corr(ret_j(t-k), ret_i(t))
        # So we read corr.T[i, j] = corr[j, i]
        cT = corr.T
        better = np.abs(cT) > np.abs(best_corr)
        best_corr = np.where(better, cT, best_corr)
        best_lag = np.where(better, k, best_lag)

    return best_corr, best_lag


def build_signal_panel() -> pd.DataFrame:
    """At each month-end t, for each sector i: signal_i = weighted sum of
    predictor sectors' returns over their identified lag windows."""
    px = _fetch_sector_etfs()
    rets = px.pct_change().dropna(how="all")
    # Monthly rebalance dates (last trading day of each month)
    monthly_idx = rets.resample("ME").last().index
    monthly_idx = [d for d in monthly_idx if d in rets.index]

    sectors = list(rets.columns)
    n_sec = len(sectors)
    signals: list[dict] = []

    for t in monthly_idx:
        # Window: past XCORR_LOOKBACK_DAYS trading days ending at t (inclusive)
        end_pos = rets.index.get_loc(t)
        start_pos = max(0, end_pos - XCORR_LOOKBACK_DAYS + 1)
        window = rets.iloc[start_pos: end_pos + 1].dropna(how="any")
        if len(window) < MIN_OBS_FOR_XCORR:
            continue

        best_corr, best_lag = _lead_lag_xcorr(window)

        # For each target sector i, find top-K predictors (j ≠ i, |corr| > thresh)
        sig_row: dict[str, float] = {}
        for i, target in enumerate(sectors):
            corr_row = best_corr[i, :].copy()
            corr_row[i] = 0.0
            qualifying = np.abs(corr_row) > PREDICTOR_THRESHOLD
            if qualifying.sum() == 0:
                sig_row[target] = 0.0
                continue
            cand_idx = np.argsort(-np.abs(corr_row))   # descending abs corr
            cand_idx = [j for j in cand_idx if qualifying[j]][:TOP_K_PREDICTORS]
            if not cand_idx:
                sig_row[target] = 0.0
                continue

            # Signal = sum of (sign(corr_lag) × predictor_return_over_lag_window)
            s = 0.0
            w_sum = 0.0
            for j in cand_idx:
                k = int(best_lag[i, j])
                if end_pos - k < 0:
                    continue
                # Predictor j's return from t-k to t-1 (the lag window)
                pred_ret = float((1.0 + rets.iloc[end_pos - k + 1: end_pos + 1, j]).prod() - 1.0)
                weight = abs(corr_row[j])
                s += weight * np.sign(corr_row[j]) * pred_ret
                w_sum += weight
            sig_row[target] = s / w_sum if w_sum > 0 else 0.0

        sig_row["__as_of"] = t
        signals.append(sig_row)

    if not signals:
        return pd.DataFrame()
    df = pd.DataFrame(signals).set_index("__as_of")
    return df


def build_sector_leadlag_ls() -> pd.Series:
    """Monthly L/S return series: long top-3 / short bottom-3 by signal.
    NET of 5 bp/side execution cost."""
    px = _fetch_sector_etfs()
    rets = px.pct_change().dropna(how="all")
    signals = build_signal_panel()
    if signals.empty:
        return pd.Series(dtype=float, name="sector_leadlag")

    # For each signal date t, take position; collect return over the NEXT month
    rows: list[dict] = []
    sig_dates = list(signals.index)
    for i, t in enumerate(sig_dates[:-1]):
        next_t = sig_dates[i + 1]
        sig_row = signals.loc[t].dropna()
        sig_row = sig_row[sig_row != 0.0]
        if len(sig_row) < TOP_Q_LONG + BOTTOM_Q_SHORT:
            continue
        ranked = sig_row.sort_values()
        shorts = ranked.head(BOTTOM_Q_SHORT).index.tolist()
        longs = ranked.tail(TOP_Q_LONG).index.tolist()

        # Forward return: compound the daily returns of each sector from after t to next_t
        win = rets.loc[(rets.index > t) & (rets.index <= next_t)]
        if win.empty:
            continue
        seg_ret = (1.0 + win).prod() - 1.0
        long_ret = float(seg_ret.reindex(longs).mean()) if longs else 0.0
        short_ret = float(seg_ret.reindex(shorts).mean()) if shorts else 0.0

        # Cost: 100% turnover monthly worst case → 2 × 5bp = 10bp
        cost = 2.0 * EXEC_COST_BPS / 10000.0
        ls_net = long_ret - short_ret - cost
        rows.append({"month": next_t, "net_ret": ls_net,
                      "n_long": len(longs), "n_short": len(shorts)})

    if not rows:
        return pd.Series(dtype=float, name="sector_leadlag")
    out = pd.DataFrame(rows).set_index("month")["net_ret"].sort_index()
    # Resample to month-end (already month-aligned but normalize)
    out.index = pd.to_datetime(out.index)
    return out.rename("sector_leadlag")


def diagnostic_summary() -> dict:
    r = build_sector_leadlag_ls().dropna()
    n = len(r)
    if n < 12:
        return {"n_months": n, "error": "insufficient data"}
    sh = float(r.mean() * 12 / (r.std() * np.sqrt(12)))
    vol = float(r.std() * np.sqrt(12))
    cum = (1.0 + r).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "n_months": n,
        "range":    f"{r.index.min().date()} -> {r.index.max().date()}",
        "ann_ret":  round(float(r.mean() * 12), 4),
        "ann_vol":  round(vol, 4),
        "sharpe":   round(sh, 3),
        "t_stat":   round(sh * np.sqrt(n / 12), 2),
        "maxdd":    round(dd, 4),
        "best":     round(float(r.max()), 4),
        "worst":    round(float(r.min()), 4),
        "hit_rate": round(float((r > 0).mean()), 3),
    }
