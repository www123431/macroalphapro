"""engine/validation/vix_carry.py — VRP / VIX term-structure carry POC.

Phase 2 Task I.B of docs/decisions/research_agenda_2026-05-29.md. Tests the
single most documented robust modern asset-pricing anomaly that is COMPLETELY
absent from our current book: the variance risk premium (VRP). When VIX
futures curve is in contango (back-month > front-month), shorting front-month
futures + rolling captures the roll-yield premium.

Academic anchor
---------------
- Karagozoglu-Lin 2010 "Investing in Volatility" — original VIX-futures
  roll-yield characterization
- Eraker-Wu 2017 "Explaining the negative returns to VIX futures" — VRP as
  insurance premium for selling vol
- Carr-Wu 2009 "Variance Risk Premiums" — the canonical VRP theory paper

Mechanism (institutional story)
-------------------------------
Insurance buyers (institutional equity portfolios hedging tail risk) pay a
persistent premium for VIX futures relative to subsequent realized variance.
The seller of that insurance (us) captures the difference. Famously
asymmetric tail: huge persistent positive carry interrupted by sudden -50%
to -100% drawdowns (Feb 5, 2018 "Volmageddon" liquidated XIV ETN).

Construction (this POC, v2 after debugging signal pollution)
------------------------------------------------------------
- Trade vehicle: VXX (ProShares VIX Short-Term Futures ETN, realizable).
  Sample window 2018-01 → 2026-05 (~95 months).
- Signal: VIX (30-day implied vol) vs VIX3M (90-day implied vol), available
  2006+ — used as the CLEAN term-structure proxy:
      contango_t = VIX3M_t - VIX_t   (>0 = contango = sell-vol premium)
  This avoids the cross-ETN-decay contamination problem that breaks
  (VXZ - VXX)/VXX as a slope proxy (VXX decayed 98.6% vs VXZ 55% over
  2018-2026, so price differences ≠ term-structure).
- Trade rule (canonical sell-vol with backwardation filter):
      position_t = -1 (short VXX) when contango_{t-1} > THRESHOLD,
                    0  (flat) otherwise.
  shift(1) is the no-look-ahead guard — yesterday's spread sets today's
  position. The filter's value is sitting OUT during backwardation (VIX >
  VIX3M), historically ~20% of trading days and the cluster where naive
  short-vol gets killed (Feb 2018, March 2020).
- Sizing: target 10% annualized vol (institutional default; matches our other
  sleeves' vol target). Realized vol from rolling 21-day window, shift(1) for
  no-look-ahead. Cap leverage at 2× to prevent runaway position in low-vol
  periods.
- Costs: VXX ETN expense ratio ≈ 0.85% annual + 5 bp/side execution. Both
  applied AT THE DAILY MONTHLY RETURN LEVEL (not monthly compounded).
- Output: monthly compounded return series.

Strict gate framing (per [[feedback-strict-gate-no-lowering-2026-05-28]])
- Pre-committed parameters above (NO grid search)
- 24-month minimum (~24 months) — should comfortably pass
- Multiple regimes covered: Vol-mageddon Feb 2018, COVID Mar 2020, 2022
  rate-hike cycle, 2024 calm
- Honest gate: feed result NET of cost to engine.research.pipeline.run_gate
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Pre-committed parameters (NO grid search) ────────────────────────────────

CONTANGO_THRESHOLD = 0.0       # any positive slope qualifies
TARGET_ANNUAL_VOL = 0.10       # 10% vol target (matches other sleeves)
VOL_LOOKBACK_DAYS = 21         # 1-month realized vol
MAX_LEVERAGE = 2.0             # cap on |position|
ETN_EXPENSE_RATIO = 0.0085     # VXX expense (annual)
EXEC_COST_BPS = 5.0            # single-side execution cost (very liquid)


@lru_cache(maxsize=1)
def _fetch_vix_data() -> pd.DataFrame:
    """Pull VXX (trading vehicle) + VIX, VIX3M (signal) daily closes from
    yfinance. Cached on disk to keep gate evaluation deterministic."""
    import os
    cache_path = "data/cache/_vix_carry_panel.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        # Backward-compat: if cached panel lacks VIX3M, force re-pull
        if "VIX3M" in df.columns and "VIX" in df.columns:
            return df

    import yfinance as yf
    data = yf.download(["VXX", "^VIX", "^VIX3M"], period="max",
                       progress=False, auto_adjust=True)
    close = data["Close"].dropna(how="all").rename(
        columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
    close.index = pd.to_datetime(close.index)
    close.to_parquet(cache_path)
    return close


def _build_strategy_daily(filter_active: bool = True) -> pd.DataFrame:
    """Internal: build the daily strategy panel (gross + net returns + position).
    filter_active=True applies the contango-only filter; =False is the naive
    always-short variant (for ablation)."""
    px = _fetch_vix_data()
    df = px.dropna().copy()

    # Term-structure spread (true signal — 90d vol minus 30d vol)
    df["contango"] = df["VIX3M"] - df["VIX"]

    # VXX daily returns (the realizable trade vehicle)
    df["vxx_ret"] = df["VXX"].pct_change()
    df["vxx_vol_21d"] = df["vxx_ret"].rolling(VOL_LOOKBACK_DAYS).std() * np.sqrt(252)

    # Position: short VXX when prior-day contango>threshold; else flat.
    # shift(1) on signal AND vol → no-look-ahead.
    if filter_active:
        df["raw_position"] = np.where(
            df["contango"].shift(1) > CONTANGO_THRESHOLD, -1.0, 0.0)
    else:
        df["raw_position"] = -1.0   # always short (ablation)

    df["scale"] = (TARGET_ANNUAL_VOL / df["vxx_vol_21d"].shift(1)).clip(upper=MAX_LEVERAGE)
    df["position"] = df["raw_position"] * df["scale"].fillna(0.0)

    df["gross_ret"] = df["position"] * df["vxx_ret"]

    daily_etn_cost = ETN_EXPENSE_RATIO / 252.0
    df["delta_pos"] = df["position"].diff().abs().fillna(0.0)
    df["net_ret"] = (
        df["gross_ret"]
        - df["delta_pos"] * EXEC_COST_BPS / 10000.0
        - df["position"].abs() * daily_etn_cost
    )
    return df


def build_vix_carry_returns() -> pd.Series:
    """Monthly compounded return series, NET of cost, contango-filtered."""
    df = _build_strategy_daily(filter_active=True)
    return ((1.0 + df["net_ret"]).resample("ME").prod() - 1.0
            ).dropna().rename("vix_carry")


def build_vix_carry_gross_returns() -> pd.Series:
    """Gross-of-cost monthly series, contango-filtered."""
    df = _build_strategy_daily(filter_active=True)
    return ((1.0 + df["gross_ret"]).resample("ME").prod() - 1.0
            ).dropna().rename("vix_carry_gross")


def build_vix_carry_no_filter_returns() -> pd.Series:
    """Always-short ablation (no contango filter), NET of cost."""
    df = _build_strategy_daily(filter_active=False)
    return ((1.0 + df["net_ret"]).resample("ME").prod() - 1.0
            ).dropna().rename("vix_carry_always_short")


def diagnostic_summary() -> dict:
    """Run the full sleeve + print key diagnostics for the strict gate.
    Reports both the filtered strategy and the always-short ablation."""
    gross = build_vix_carry_gross_returns()
    net = build_vix_carry_returns()
    no_filter = build_vix_carry_no_filter_returns()

    def stats(r, label):
        n = len(r)
        sh = float(r.mean() * 12 / (r.std() * np.sqrt(12)))
        vol = float(r.std() * np.sqrt(12))
        cum = (1.0 + r).cumprod()
        dd = float((cum / cum.cummax() - 1.0).min())
        t = sh * np.sqrt(n / 12)
        return {
            "label": label,
            "n_months": n,
            "ann_ret": round(float(r.mean() * 12), 4),
            "ann_vol": round(vol, 4),
            "sharpe": round(sh, 3),
            "t_stat": round(t, 2),
            "maxdd": round(dd, 4),
            "best_month": round(float(r.max()), 4),
            "worst_month": round(float(r.min()), 4),
            "hit_rate": round(float((r > 0).mean()), 3),
        }

    return {
        "gross_filtered": stats(gross, "gross_filtered"),
        "net_filtered": stats(net, "net_filtered"),
        "net_always_short_ablation": stats(no_filter, "net_always_short_ablation"),
    }
