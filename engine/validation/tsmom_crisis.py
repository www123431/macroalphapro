"""engine/validation/tsmom_crisis.py — self-built cross-asset TSMOM crisis sleeve.

Direction 1 of the strategy-strengthening push. Prior TSMOM attempts
(Path P/Q/R) FAILED — but as standalone ALPHA (Sharpe gates). This tests
TSMOM on the INSURANCE standard instead: crisis-window contribution
(Hurst-Ooi-Pedersen "crisis alpha"). The hypothesis is that a trend-
following sleeve is crisis-positive in BOTH growth shocks (2008/2020)
AND rate shocks (2022) — because it follows the persistent move
regardless of cause — and therefore patches the documented hole in the
AC TLT/GLD insurance (which assumes flight-to-quality and broke in 2022
when bonds + equities fell together).

Construction (canonical 12-1 time-series momentum, Moskowitz-Ooi-Pedersen
2012):
  - cross-asset proxies: equities / bonds / commodities / gold / USD
  - signal = sign of trailing 12-month return (skip the most recent month)
  - position = signal, inverse-vol scaled (equal risk per instrument)
  - sleeve return = mean across instruments of signal_t * next-month return
  - vol-targeted to a configurable annual vol

This is a CANDIDATE, validated on the insurance lens (crisis contribution),
not the alpha lens. If it patches 2022 + contributes in 2008/2020 at
acceptable drag, it graduates to a real sleeve spec.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expanded liquid cross-asset proxies across 5 asset-class groups. Core
# instruments (SPY/EFA/EEM/TLT/IEF/AGG/LQD/DBC/GLD) have full 2008 GFC
# coverage; HYG/DBA/UUP start early-mid 2007 (12m signal forms ~mid-2008,
# right at the GFC — acceptable, the core carries the 2008 test).
DEFAULT_INSTRUMENTS = {
    "SPY": "equities_us",
    "EFA": "equities_developed_intl",
    "EEM": "equities_em",
    "TLT": "bonds_long_ust",
    "IEF": "bonds_intermediate_ust",
    "AGG": "bonds_aggregate",
    "LQD": "credit_ig",
    "HYG": "credit_hy",
    "DBC": "commodities_broad",
    "GLD": "gold",
    "DBA": "commodities_ag",
    "UUP": "usd_index",
}
_PRICE_CACHE = Path("data/cache/tsmom_crossasset_monthly.parquet")

# Real continuous front-month futures (yfinance) — true managed-futures
# breadth across 6 asset-class groups, full history to 2006. NOTE: yfinance
# stitches front-month contracts (not vendor-grade back-adjustment), so
# roll artifacts exist; for a SIGN-of-trailing-return TSMOM signal these
# mostly wash out (direction robust to small roll noise). Futures also
# trade cheaper than ETFs (~3-5bp round-trip) and have higher capacity.
FUTURES_INSTRUMENTS = {
    "ES=F": "equity_sp500", "NQ=F": "equity_nasdaq", "YM=F": "equity_dow",
    "ZN=F": "rates_10y", "ZB=F": "rates_30y", "ZF=F": "rates_5y",
    "GC=F": "gold", "SI=F": "silver", "HG=F": "copper",
    "CL=F": "crude", "NG=F": "natgas",
    "ZC=F": "corn", "ZW=F": "wheat", "ZS=F": "soybean",
    "6E=F": "fx_eur", "6J=F": "fx_jpy", "6B=F": "fx_gbp",
}
_FUTURES_CACHE = Path("data/cache/tsmom_futures_monthly.parquet")


def fetch_futures_monthly(
    start: str = "2006-01-01",
    end:   str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Month-end continuous front-month futures (yfinance), cached."""
    if _FUTURES_CACHE.exists() and not force_refresh:
        try:
            return pd.read_parquet(_FUTURES_CACHE)
        except Exception as exc:
            logger.warning("tsmom_crisis: futures cache read failed: %s", exc)
    import yfinance as yf
    raw = yf.download(list(FUTURES_INSTRUMENTS.keys()), start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    monthly = raw.resample("ME").last()
    _FUTURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(_FUTURES_CACHE)
    return monthly

CRISIS_WINDOWS = {
    "2008_GFC":      ("2008-09-01", "2009-03-31"),
    "2020_COVID":    ("2020-02-15", "2020-04-30"),
    "2022_RATESHOCK":("2022-01-01", "2022-10-31"),
}


def fetch_crossasset_monthly(
    start: str = "2006-01-01",
    end:   str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Month-end adjusted close for the cross-asset proxies (yfinance),
    cached. Columns = tickers, index = month-end."""
    if _PRICE_CACHE.exists() and not force_refresh:
        try:
            return pd.read_parquet(_PRICE_CACHE)
        except Exception as exc:
            logger.warning("tsmom_crisis: cache read failed: %s", exc)
    import yfinance as yf
    tickers = list(DEFAULT_INSTRUMENTS.keys())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    monthly = raw.resample("ME").last()
    _PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(_PRICE_CACHE)
    return monthly


def tsmom_signal(monthly_px: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """12-1 TSMOM signal: sign of the trailing `lookback`-month return,
    skipping the most recent `skip` month(s). +1 long / -1 short."""
    # trailing return from t-lookback to t-skip
    ret_lb = monthly_px.shift(skip) / monthly_px.shift(lookback) - 1.0
    return np.sign(ret_lb)


def multispeed_signal(
    monthly_px: pd.DataFrame,
    lookbacks:  tuple = (3, 6, 12),
    skip:       int = 1,
) -> pd.DataFrame:
    """Average of sign(trailing-return) across multiple lookback speeds.
    A fast (3m) component reacts sooner to reversals; a slow (12m)
    component captures sustained trends. The average is in [-1, 1] and
    blends the two — a standard multi-speed trend construction."""
    sigs = [tsmom_signal(monthly_px, lb, skip) for lb in lookbacks]
    return sum(sigs) / len(sigs)


def _tsmom_weights(
    monthly_px: pd.DataFrame,
    lookback=   12,
    skip:       int = 1,
    vol_window: int = 12,
) -> pd.DataFrame:
    """Monthly inverse-vol-weighted TSMOM weights, gross-normalized to 1
    each month (equal risk budget). `lookback` may be an int (single
    speed) OR a tuple of ints (multi-speed blend)."""
    rets = monthly_px.pct_change()
    if isinstance(lookback, (tuple, list)):
        sig = multispeed_signal(monthly_px, tuple(lookback), skip)
    else:
        sig = tsmom_signal(monthly_px, lookback, skip)
    inv_vol = 1.0 / rets.rolling(vol_window).std().replace(0, np.nan)
    pos = sig * inv_vol
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    return pos.div(gross, axis=0)


def build_tsmom_sleeve(
    monthly_px:   pd.DataFrame,
    lookback:     int = 12,
    skip:         int = 1,
    vol_window:   int = 12,
    target_vol:   float = 0.10,
    vol_target_mode: str = "trailing",
    trailing_window: int = 36,
    return_turnover: bool = False,
):
    """Monthly returns of an inverse-vol-weighted 12-1 TSMOM sleeve,
    vol-targeted to `target_vol` annual.

    vol_target_mode:
      'trailing' (default, DEPLOYABLE) — scale each month by
        target_vol / trailing_realized_vol(up to t-1). Ex-ante, no
        look-ahead. Uses a `trailing_window`-month rolling vol with a
        12-month minimum warmup.
      'full' — scale by full-sample realized vol (look-ahead; only for
        a quick crisis-contribution sanity check).

    return_turnover=True → returns (sleeve, turnover_series) where
    turnover_t = sum_i |w_i,t - w_i,t-1| (one-way, for cost modelling).
    """
    rets = monthly_px.pct_change()
    w = _tsmom_weights(monthly_px, lookback, skip, vol_window)
    gross_sleeve = (w.shift(1) * rets).sum(axis=1).dropna()

    if vol_target_mode == "full":
        realized = gross_sleeve.std() * np.sqrt(12)
        sleeve = gross_sleeve * (target_vol / realized) if realized > 0 else gross_sleeve
    else:  # trailing (ex-ante)
        trail_vol = gross_sleeve.rolling(trailing_window, min_periods=12).std() * np.sqrt(12)
        scalar = (target_vol / trail_vol.shift(1)).clip(upper=3.0)  # cap leverage 3x
        sleeve = (gross_sleeve * scalar).dropna()

    sleeve = sleeve.rename("tsmom_crisis")
    if return_turnover:
        turnover = (w - w.shift(1)).abs().sum(axis=1).reindex(sleeve.index).fillna(0.0)
        return sleeve, turnover
    return sleeve


def apply_tsmom_cost(
    sleeve:        pd.Series,
    turnover:      pd.Series,
    roundtrip_bps: float = 8.0,
) -> pd.Series:
    """Net sleeve = gross sleeve − turnover × round-trip cost. Liquid
    cross-asset ETFs run ~6-10bp round-trip; default 8bp. turnover is
    one-way weight change; a flip costs 2× the half-trade, so we charge
    turnover × roundtrip_bps directly (turnover already counts both
    sides of the weight delta)."""
    drag = turnover * (roundtrip_bps / 10000.0)
    return (sleeve - drag.reindex(sleeve.index).fillna(0.0)).rename("tsmom_crisis_net")


def _book_metrics_monthly(r: pd.Series) -> dict:
    r = r.dropna()
    sd = r.std(ddof=1)
    curve = (1.0 + r).cumprod()
    dd = float((curve / curve.cummax() - 1.0).min())
    return {
        "ann_return": float(r.mean() * 12),
        "ann_vol":    float(sd * np.sqrt(12)),
        "sharpe":     float(r.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan"),
        "max_dd":     dd,
    }


def combined_book_comparison(
    sleeve_weekly_path: str = "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet",
    tsmom_sleeve: pd.Series = None,
    monthly_px:   pd.DataFrame = None,
) -> dict:
    """Compare the current 5-sleeve book vs a book where the AC TLT/GLD
    insurance is REPLACED by the TSMOM crisis sleeve, at monthly frequency.

    The insurance test (G7 lens): does swapping in TSMOM reduce the book's
    drawdown — especially in 2022, where the current book's AC hedge
    failed? Returns full-period metrics + per-crisis-window book return
    for both books.
    """
    weekly = pd.read_parquet(sleeve_weekly_path)
    weekly.index = pd.to_datetime(weekly.index)
    # compound weekly -> month-end per strategy
    monthly = weekly.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)

    if tsmom_sleeve is None:
        if monthly_px is None:
            monthly_px = fetch_crossasset_monthly()
        tsmom_sleeve = build_tsmom_sleeve(monthly_px)
    ts = tsmom_sleeve.copy()
    ts.index = pd.to_datetime(ts.index)

    # Align book months with the TSMOM sleeve months
    book = monthly.join(ts.rename("TSMOM"), how="inner").dropna(
        subset=[c for c in monthly.columns])
    # Current weights (operating_model sleeve allocation)
    W_CUR = {"K1_BAB": 0.324, "D_PEAD": 0.243, "PATH_N": 0.243,
             "CTA_PQTIX": 0.090, "AC_proxy_AB_2014_23": 0.100}
    AC = "AC_proxy_AB_2014_23"
    # TSMOM-replaces-AC: same weights, AC -> TSMOM
    W_TS = {k: v for k, v in W_CUR.items() if k != AC}
    W_TS["TSMOM"] = 0.100

    def _book_ret(weights):
        cols = [c for c in weights if c in book.columns]
        w = np.array([weights[c] for c in cols], dtype=float); w /= w.sum()
        return (book[cols] * w).sum(axis=1)

    cur = _book_ret(W_CUR)
    swp = _book_ret(W_TS)

    def _crisis(series):
        out = {}
        for name, (a, b) in CRISIS_WINDOWS.items():
            seg = series[(series.index >= a) & (series.index <= b)].dropna()
            if len(seg):
                curve = (1.0 + seg).cumprod()
                out[name] = {
                    "ret": float((1.0 + seg).prod() - 1.0),
                    "dd":  float((curve / curve.cummax() - 1.0).min()),
                }
        return out

    return {
        "n_months":   len(book),
        "span":       (str(book.index.min().date()), str(book.index.max().date())),
        "current":    {"full": _book_metrics_monthly(cur), "crisis": _crisis(cur)},
        "tsmom_swap": {"full": _book_metrics_monthly(swp), "crisis": _crisis(swp)},
    }


@dataclass(frozen=True)
class CrisisContribution:
    window:        str
    tsmom_ret:     float    # cumulative TSMOM sleeve return in window
    tlt_ret:       float    # TLT cumulative (the current insurance proxy)
    gld_ret:       float    # GLD cumulative
    tlt_gld_5050:  float    # 50/50 TLT/GLD (the AC sleeve)
    spy_ret:       float    # SPY (what we're insuring against)
    patches_hole:  bool     # TSMOM positive where TLT/GLD failed


def crisis_contribution(
    monthly_px: pd.DataFrame,
    sleeve:     pd.Series,
) -> list[CrisisContribution]:
    """Cumulative return of the TSMOM sleeve vs the TLT/GLD insurance vs
    SPY in each crisis window. patches_hole = TSMOM positive AND TLT/GLD
    not (the 2022 case)."""
    rets = monthly_px.pct_change()
    out = []
    for name, (a, b) in CRISIS_WINDOWS.items():
        def _cum(series):
            seg = series[(series.index >= a) & (series.index <= b)].dropna()
            return float((1.0 + seg).prod() - 1.0) if len(seg) else float("nan")
        ts = _cum(sleeve)
        tlt = _cum(rets["TLT"]) if "TLT" in rets else float("nan")
        gld = _cum(rets["GLD"]) if "GLD" in rets else float("nan")
        spy = _cum(rets["SPY"]) if "SPY" in rets else float("nan")
        tg = (0.5 * tlt + 0.5 * gld) if (np.isfinite(tlt) and np.isfinite(gld)) else float("nan")
        patches = bool(np.isfinite(ts) and ts > 0 and np.isfinite(tg) and tg <= 0)
        out.append(CrisisContribution(
            window=name, tsmom_ret=ts, tlt_ret=tlt, gld_ret=gld,
            tlt_gld_5050=tg, spy_ret=spy, patches_hole=patches,
        ))
    return out
