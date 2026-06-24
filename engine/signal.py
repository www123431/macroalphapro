"""
Signal Generation Layer
========================
Purely structural signal computation — no LLM calls.

Implements two momentum signal families for sector ETF universe:

  1. Time-Series Momentum (TSMOM)
     Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
     Signal: sign of past 12-1 month return for each asset independently.
     CTA / trend-following standard.

  2. Cross-Sectional Momentum (CSMOM)
     Moskowitz & Grinblatt (1999) "Do Industries Explain Momentum?"
     Signal: rank sectors by past return; long top tercile, short bottom tercile.

Look-ahead prevention
---------------------
All functions accept `as_of: datetime.date`. Data fetched is strictly
available before that date. The skip_months parameter (default=1) excludes
the most recent month to avoid microstructure / bid-ask bounce bias
(Jegadeesh & Titman 1993).

Formation window convention
---------------------------
  end_price   : last available close on or before (as_of - skip_months calendar months)
  start_price : last available close on or before (as_of - lookback_months calendar months)
  raw_return  : (end_price - start_price) / start_price

This matches the standard academic 12-1 formation period.

Output conventions
------------------
  Signal values : +1.0 (long), 0.0 (neutral), -1.0 (short)
  Raw returns   : float, not annualised
  Volatility    : annualised, based on daily returns over lookback window

Integration points
------------------
  - Uses SECTOR_ETF map from engine/history.py
  - Output DataFrame consumed by engine/portfolio.py (portfolio construction)
  - Output consumed by engine/backtest.py (signal replay)
  - Raw returns / vol exposed to ui/tabs.py for display
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd
import yfinance as yf

# Streamlit was once used here for UI cache decorators. Headless cron
# environments (paper-trade daily, NAV rollup) don't have Streamlit
# installed, so we route through a small shim that no-ops the cache
# decorators when streamlit isn't importable. See engine/_streamlit_shim.py.
from engine._streamlit_shim import streamlit as st  # noqa: F401

from engine.history import SECTOR_ETF, get_active_sector_etf

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Module-level fallback (used only for type reference); runtime calls use get_active_sector_etf()
_TICKER_TO_SECTOR: dict[str, str] = {v: k for k, v in SECTOR_ETF.items()}

# Approximate trading days per month — used only for vol window, not for
# return formation (formation uses calendar month offsets for accuracy).
_TRADING_DAYS_PER_MONTH = 21
_TRADING_DAYS_PER_YEAR  = 252

# ── P6: Fixed composite weights (M2 resolution — not dynamically adjusted) ────
COMPOSITE_WEIGHTS: dict[str, float] = {
    "tsmom":      0.40,
    "csmom":      0.25,
    "carry":      0.20,
    "factor_mad": 0.10,
    "reversal":   0.05,
}


# ── Internal data fetcher ──────────────────────────────────────────────────────

def _fetch_closes(
    tickers:      list[str],
    start:        datetime.date,
    as_of:        datetime.date,
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for tickers in [start, as_of].
    Returns a DataFrame with ticker columns, date index.
    Enforces no look-ahead: as_of is the last permitted date.
    """
    try:
        raw = yf.download(
            tickers,
            start=str(start),
            end=str(as_of + datetime.timedelta(days=1)),  # yfinance end is exclusive
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()
        # Handle both single-ticker (flat) and multi-ticker (MultiIndex) responses
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        return close.dropna(how="all")
    except Exception as exc:
        logger.warning("_fetch_closes failed [%s – %s]: %s", start, as_of, exc)
        return pd.DataFrame()


def _month_offset(date: datetime.date, months: int) -> datetime.date:
    """Subtract `months` calendar months from date, clamping to valid day."""
    month = date.month - months
    year  = date.year + month // 12
    month = month % 12 or 12
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return date.replace(year=year, month=month, day=min(date.day, last_day))


# ── Core return computation ────────────────────────────────────────────────────

def compute_raw_returns(
    as_of:           datetime.date,
    lookback_months: int = 12,
    skip_months:     int = 1,
) -> pd.DataFrame:
    """
    Compute formation-period returns and realised volatility for all sector ETFs.

    Formation window: [as_of - lookback_months, as_of - skip_months]
    Volatility window: daily returns over the same period, annualised.

    Returns DataFrame with index=sector_name and columns:
        ticker       : ETF symbol
        raw_return   : cumulative return over formation window
        ann_vol      : annualised realised volatility (daily returns * sqrt(252))
        start_date   : actual start date used (last close on or before cutoff)
        end_date     : actual end date used
        obs          : number of daily return observations
    """
    end_cutoff   = _month_offset(as_of, skip_months)
    start_cutoff = _month_offset(as_of, lookback_months)

    # Buffer to ensure we get at least one observation near the cutoff dates
    fetch_start = start_cutoff - datetime.timedelta(days=15)

    active_etf = get_active_sector_etf()
    ticker_to_sector = {v: k for k, v in active_etf.items()}
    tickers = list(active_etf.values())
    closes  = _fetch_closes(tickers, fetch_start, end_cutoff)

    if closes.empty:
        logger.warning("compute_raw_returns: no price data for as_of=%s", as_of)
        return pd.DataFrame()

    records = []
    for ticker, sector in ticker_to_sector.items():
        if ticker not in closes.columns:
            continue

        series = closes[ticker].dropna()
        if len(series) < 5:
            continue

        # Restrict to [start_cutoff, end_cutoff]
        series_window = series[
            (series.index.date >= start_cutoff) &
            (series.index.date <= end_cutoff)
        ]
        if len(series_window) < 2:
            continue

        p_start = float(series_window.iloc[0])
        p_end   = float(series_window.iloc[-1])

        if p_start <= 0:
            continue

        raw_ret  = (p_end - p_start) / p_start
        daily_ret = series_window.pct_change().dropna()
        ann_vol  = float(daily_ret.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)) if len(daily_ret) > 1 else float("nan")

        records.append({
            "sector":      sector,
            "ticker":      ticker,
            "raw_return":  raw_ret,
            "ann_vol":     ann_vol,
            "start_date":  series_window.index[0].date(),
            "end_date":    series_window.index[-1].date(),
            "obs":         len(daily_ret),
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("sector")


# ── TSMOM ──────────────────────────────────────────────────────────────────────

def compute_tsmom(
    as_of:           datetime.date,
    lookback_months: int = 12,
    skip_months:     int = 1,
) -> dict[str, float]:
    """
    Time-Series Momentum signal (Moskowitz, Ooi, Pedersen 2012).

    For each sector ETF independently:
        signal = +1  if raw_return > 0  (trend up  → long)
        signal = -1  if raw_return < 0  (trend down → short)
        signal =  0  if insufficient data

    Returns {sector_name: signal}
    """
    df = compute_raw_returns(as_of, lookback_months, skip_months)
    if df.empty:
        return {}

    signals: dict[str, float] = {}
    for sector, row in df.iterrows():
        ret = row["raw_return"]
        if np.isnan(ret):
            signals[sector] = 0.0
        elif ret > 0:
            signals[sector] = 1.0
        elif ret < 0:
            signals[sector] = -1.0
        else:
            signals[sector] = 0.0

    return signals


# ── CSMOM ──────────────────────────────────────────────────────────────────────

def compute_csmom(
    as_of:           datetime.date,
    lookback_months: int = 12,
    skip_months:     int = 1,
    top_pct:         float = 1 / 3,
    bottom_pct:      float = 1 / 3,
) -> dict[str, float]:
    """
    Cross-Sectional Momentum signal (Moskowitz & Grinblatt 1999).

    Ranks all sector ETFs by formation-period return:
        Top    `top_pct`    fraction → signal = +1.0  (long)
        Bottom `bottom_pct` fraction → signal = -1.0  (short)
        Middle                        → signal =  0.0  (neutral)

    Returns {sector_name: signal}
    """
    df = compute_raw_returns(as_of, lookback_months, skip_months)
    if df.empty:
        return {}

    df_valid = df.dropna(subset=["raw_return"]).copy()
    if df_valid.empty:
        return {}

    signals: dict[str, float] = {s: 0.0 for s in df_valid.index}

    # Within-class ranking (P2-12 A-3)
    try:
        from engine.universe_manager import get_universe_by_class as _gwc
        universe_by_class = _gwc()
    except Exception:
        universe_by_class = {}

    if universe_by_class:
        for _ac, _cmap in universe_by_class.items():
            _csectors = [s for s in _cmap.keys() if s in df_valid.index]
            _sub = df_valid.loc[_csectors]
            if len(_sub) < 2:
                continue
            _n       = len(_sub)
            _n_top   = max(1, round(_n * top_pct))
            _n_bot   = max(1, round(_n * bottom_pct))
            _ranked  = _sub["raw_return"].sort_values()
            _bot_set = set(_ranked.index[:_n_bot])
            _top_set = set(_ranked.index[_n - _n_top:])
            for s in _sub.index:
                signals[s] = 1.0 if s in _top_set else (-1.0 if s in _bot_set else 0.0)
    else:
        # 全局排序回退
        n         = len(df_valid)
        n_top     = max(1, round(n * top_pct))
        n_bottom  = max(1, round(n * bottom_pct))
        ranked    = df_valid["raw_return"].sort_values()
        bottom_set = set(ranked.index[:n_bottom])
        top_set    = set(ranked.index[n - n_top:])
        for sector in df_valid.index:
            signals[sector] = 1.0 if sector in top_set else (-1.0 if sector in bottom_set else 0.0)

    return signals


# ── Combined signal DataFrame ──────────────────────────────────────────────────

def get_signal_dataframe(
    as_of:           datetime.date,
    lookback_months: int = 12,
    skip_months:     int = 1,
    use_cache:       bool = True,
) -> pd.DataFrame:
    """
    Compute TSMOM and CSMOM signals and return as a unified DataFrame.
    Intended as the primary interface for portfolio.py and backtest.py.

    Returns DataFrame with index=sector_name and columns:
        ticker       : ETF symbol
        raw_return   : formation-period return
        ann_vol      : annualised realised volatility
        tsmom        : TSMOM signal {-1, 0, +1}
        csmom        : CSMOM signal {-1, 0, +1}
        inv_vol_wt   : 1 / ann_vol (raw, before normalisation) — for vol targeting
        obs          : number of daily return observations in window
        start_date   : formation window start date actually used
        end_date     : formation window end date actually used

    use_cache=True: check SQLite snapshot cache before fetching from yfinance.
    Backtest callers should pass use_cache=False to avoid cross-date contamination.
    """
    import datetime as _dt
    today = _dt.date.today()
    # Only use cache for historical dates (not today — intraday data may be incomplete)
    _can_cache = use_cache and as_of < today
    if _can_cache:
        try:
            from engine.memory import get_signal_snapshot, save_signal_snapshot
            cached = get_signal_snapshot(as_of, lookback_months, skip_months)
            if cached is not None and not cached.empty:
                return cached
        except Exception:
            pass

    df = compute_raw_returns(as_of, lookback_months, skip_months)
    if df.empty:
        return pd.DataFrame()

    # TSMOM (with VXX polarity flip — P2-12 批次B)
    # VXX 是衰减工具，正向动量代表市场恐慌而非趋势延续，信号极性相反
    _REVERSE_MOMENTUM_TICKERS = {"VXX"}
    df["tsmom"] = df.apply(
        lambda row: (
            0.0 if np.isnan(row["raw_return"]) else
            (-1.0 if row["raw_return"] > 0 else (1.0 if row["raw_return"] < 0 else 0.0))
            if row.get("ticker") in _REVERSE_MOMENTUM_TICKERS else
            (1.0 if row["raw_return"] > 0 else (-1.0 if row["raw_return"] < 0 else 0.0))
        ),
        axis=1,
    )

    # CSMOM — within-class ranking (P2-12 批次B: 激活 A-3)
    # 不同资产类别（权益/债券/商品）截面收益不可比，需分类排序
    try:
        from engine.universe_manager import get_universe_by_class as _get_by_class
        _universe_by_class = _get_by_class()
    except Exception:
        _universe_by_class = {}

    df["csmom"] = 0.0
    if _universe_by_class:
        for _ac, _class_map in _universe_by_class.items():
            _class_sectors = [s for s in _class_map.keys() if s in df.index]
            _sub = df.loc[_class_sectors].dropna(subset=["raw_return"])
            if len(_sub) < 2:
                continue
            _n = len(_sub)
            _n_top    = max(1, round(_n / 3))
            _n_bottom = max(1, round(_n / 3))
            _ranked   = _sub["raw_return"].sort_values()
            _bot_set  = set(_ranked.index[:_n_bottom])
            _top_set  = set(_ranked.index[_n - _n_top:])
            for _s in _sub.index:
                if _s in _top_set:
                    df.loc[_s, "csmom"] = 1.0
                elif _s in _bot_set:
                    df.loc[_s, "csmom"] = -1.0
    else:
        # 回退：全局排序（universe_manager 不可用时）
        df_valid  = df.dropna(subset=["raw_return"])
        n         = len(df_valid)
        n_top     = max(1, round(n / 3))
        n_bottom  = max(1, round(n / 3))
        ranked    = df_valid["raw_return"].sort_values()
        bottom_set = set(ranked.index[:n_bottom])
        top_set    = set(ranked.index[n - n_top:])
        for sector in df.index:
            if sector in top_set:
                df.loc[sector, "csmom"] = 1.0
            elif sector in bottom_set:
                df.loc[sector, "csmom"] = -1.0

    # P1-7 + P2-4: download ~13M of daily prices once for both 21d vol and GARCH(1,1).
    # Extended window (280 BDays) gives GARCH enough observations without a second fetch.
    try:
        import yfinance as _yf
        _cutoff = pd.Timestamp(as_of)
        _start_vol = _cutoff - pd.tseries.offsets.BDay(280)
        _tickers_21 = list(df["ticker"].dropna())
        _px_vol = _yf.download(
            _tickers_21, start=_start_vol.date(), end=(as_of + pd.tseries.offsets.BDay(1)).date(),
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        _close_vol = _px_vol["Close"] if "Close" in _px_vol else _px_vol

        _vol_21:    dict[str, float] = {}
        _vol_garch: dict[str, float] = {}

        for sector, row in df.iterrows():
            tk = row.get("ticker")
            if not tk or tk not in _close_vol.columns:
                continue
            _ret_all = _close_vol[tk].dropna().pct_change().dropna()

            # 21d realised vol (P1-7)
            _ret_21 = _ret_all.tail(21)
            if len(_ret_21) >= 10:
                _vol_21[sector] = float(_ret_21.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))

            # P2-4: GARCH(1,1) one-step-ahead conditional vol
            _ret_garch = _ret_all.tail(252)
            if len(_ret_garch) >= 60:
                try:
                    from arch import arch_model as _arch_model
                    _gm = _arch_model(
                        _ret_garch * 100,   # scale to % to improve numerical stability
                        vol="Garch", p=1, q=1, dist="normal", rescale=False,
                    )
                    _gres = _gm.fit(disp="off", show_warning=False)
                    _fcast_var = float(_gres.forecast(horizon=1).variance.iloc[-1, 0])
                    # _fcast_var is in (% daily)^2; convert to annualised decimal vol
                    _vol_garch[sector] = float(np.sqrt(_fcast_var / 10_000 * _TRADING_DAYS_PER_YEAR))
                except Exception:
                    pass  # arch not installed or fit failed — fall through to fallback chain

        df["ann_vol_21d"]   = df.index.map(lambda s: _vol_21.get(s, np.nan))
        df["ann_vol_garch"] = df.index.map(lambda s: _vol_garch.get(s, np.nan))
    except Exception:
        df["ann_vol_21d"]   = np.nan
        df["ann_vol_garch"] = np.nan

    # Inverse-vol weight (raw, portfolio.py will normalise)
    df["inv_vol_wt"] = df["ann_vol"].apply(
        lambda v: 1.0 / v if (v and v > 1e-6 and not np.isnan(v)) else np.nan
    )

    if _can_cache:
        try:
            save_signal_snapshot(as_of, lookback_months, skip_months, df)
        except Exception:
            pass

    return df


# ── Convenience: signal summary for UI display ─────────────────────────────────

def signal_summary(as_of: datetime.date) -> pd.DataFrame:
    """
    Human-readable signal table for UI display.
    Returns DataFrame sorted by raw_return descending.
    """
    df = get_signal_dataframe(as_of)
    if df.empty:
        return pd.DataFrame()

    display = df[["ticker", "raw_return", "ann_vol", "tsmom", "csmom"]].copy()
    display["raw_return_%"] = (display["raw_return"] * 100).round(2)
    display["ann_vol_%"]    = (display["ann_vol"]    * 100).round(2)
    display["tsmom_label"]  = display["tsmom"].map({1.0: "↑ 做多", -1.0: "↓ 做空", 0.0: "— 中性"})
    display["csmom_label"]  = display["csmom"].map({1.0: "↑ 做多", -1.0: "↓ 做空", 0.0: "— 中性"})

    return (
        display[["ticker", "raw_return_%", "ann_vol_%", "tsmom_label", "csmom_label"]]
        .sort_values("raw_return_%", ascending=False)
        .rename(columns={
            "ticker":       "ETF",
            "raw_return_%": "12-1M 收益%",
            "ann_vol_%":    "年化波动率%",
            "tsmom_label":  "TSMOM",
            "csmom_label":  "CSMOM",
        })
    )


# ── Multi-factor scoring + Quant Gate ─────────────────────────────────────────
# All inputs are market data only — no LLM self-reports.
# Goodhart-safe: gate rules cannot be gamed by LLM output.

_GATE_DIRECTIONS = ["超配", "标配", "低配", "拦截", "通过", "中性"]
_ALL_SECTOR_DIRS = {"超配", "标配", "低配"}


def _sigmoid_norm(x: float, scale: float = 2.0) -> float:
    """Map any real number to [0, 100] via sigmoid. scale controls steepness."""
    import math
    try:
        return 100.0 / (1.0 + math.exp(-scale * x))
    except OverflowError:
        return 0.0 if x < 0 else 100.0


@st.cache_data(ttl=86400, show_spinner=False)
def compute_carry(as_of: datetime.date) -> dict[str, float]:
    """
    Net carry per sector ETF: dividend_yield − risk_free_rate.
    Koijen et al. (2018) "Carry" — negative net carry is a short signal.

    Risk-free rate: 13-week T-bill (^IRX, annualised).
    Dividend yield: trailingAnnualDividendYield from yfinance info.
    Cached 24h — yields do not change intraday.
    """
    # ── Risk-free rate from ^IRX (13-week T-bill, annualised %) ───────────────
    rf_rate = 0.0
    try:
        _irx = yf.Ticker("^IRX").fast_info
        _rf_raw = getattr(_irx, "last_price", None)
        if _rf_raw and _rf_raw > 0:
            rf_rate = float(_rf_raw) / 100.0   # ^IRX quoted as e.g. 5.32 → 0.0532
    except Exception:
        pass

    # Commodity ETFs have no meaningful carry (physical storage cost, no coupon)
    try:
        from engine.universe_manager import TICKER_METADATA as _TM
    except ImportError:
        _TM = {}

    active_etf = get_active_sector_etf()
    carries: dict[str, float] = {}
    for sector, ticker in active_etf.items():
        if _TM.get(ticker, {}).get("asset_class", "equity") == "commodity":
            carries[sector] = 0.0
            continue
        # Equity / fixed_income / real_estate: div_yield − rf
        try:
            _info = yf.Ticker(ticker).info
            div_yield = _info.get("trailingAnnualDividendYield") or 0.0
            carries[sector] = float(div_yield) - rf_rate
        except Exception:
            try:
                _fi = yf.Ticker(ticker).fast_info
                _y  = getattr(_fi, "last_dividend_value", None)
                _px = getattr(_fi, "last_price", None)
                div_yield = float(_y / _px) if (_y and _px and _px > 0) else 0.0
                carries[sector] = div_yield - rf_rate
            except Exception:
                carries[sector] = -rf_rate
    return carries


@st.cache_data(ttl=86400, show_spinner=False)
def compute_reversal(as_of: datetime.date) -> dict[str, float]:
    """
    Price-to-5Y-SMA reversal signal per sector ETF.
    z = (P_now − SMA_60M) / σ_60M  (cross-sectionally winsorised 5%/95%)
    reversal_score = −z  (below historical mean = cheap = positive signal)

    Poterba & Summers (1988); George & Hwang (2004) 52-week high effect.
    Returns {sector: reversal_score} — positive = mean-reversion buy opportunity.
    """
    # Regime-conditional: active only in 'transition' (M2 resolution).
    # Weight (5%) is kept in composite regardless — zeros here mean neutral contribution.
    try:
        from engine.regime import get_regime_on as _grv
        if getattr(_grv(as_of), "regime", "transition") != "transition":
            return {s: 0.0 for s in get_active_sector_etf()}
    except Exception:
        pass

    active_etf  = get_active_sector_etf()
    tickers     = list(active_etf.values())
    sector_map  = {v: k for k, v in active_etf.items()}

    end   = pd.Timestamp(as_of)
    start = end - pd.DateOffset(months=62)   # 60M + 2M buffer for gaps

    raw_z: dict[str, float] = {}
    try:
        px = yf.download(
            tickers, start=str(start.date()), end=str(end.date()),
            interval="1mo", progress=False, auto_adjust=True,
        )
        if isinstance(px.columns, pd.MultiIndex):
            closes = px["Close"]
        else:
            closes = px[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else pd.DataFrame()

        for ticker in tickers:
            sector = sector_map.get(ticker)
            if sector is None or ticker not in closes.columns:
                continue
            s = closes[ticker].dropna()
            if len(s) < 24:          # need ≥ 24 months for meaningful z-score
                continue
            s = s.iloc[-60:] if len(s) >= 60 else s
            sma = s.mean()
            std = s.std()
            if std < 1e-8:
                continue
            p_now = float(s.iloc[-1])
            raw_z[sector] = (p_now - sma) / std
    except Exception:
        pass

    if not raw_z:
        return {s: 0.0 for s in active_etf}

    # Cross-sectional winsorise 5%/95% then negate (below mean = buy)
    vals   = np.array(list(raw_z.values()))
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
    result: dict[str, float] = {}
    for sector, z in raw_z.items():
        z_clipped = float(np.clip(z, lo, hi))
        result[sector] = -z_clipped   # negate: below mean → positive signal
    # Fill missing sectors with neutral 0
    for sector in active_etf:
        if sector not in result:
            result[sector] = 0.0
    return result


def compute_composite_scores(
    as_of:           datetime.date,
    lookback_months: int = 12,
    skip_months:     int = 1,
) -> pd.DataFrame:
    """
    Compute a multi-factor composite score (0–100) per sector ETF.

    Factors and weights (P1-6 revised)
    ------------------------------------
    TSMOM   50%  : base score +1→100, 0→50, -1→0;
                   then CSMOM cross-sectional rank applied as truncation modifier:
                   if TSMOM=+1 but bottom-tercile rank → cap tsmom_norm at 70
                   if TSMOM=-1 but top-tercile rank    → floor tsmom_norm at 30
    Sharpe  30%  : sigmoid-normalised Sharpe of formation period (0–100)
    Regime  20%  : p_risk_on × 100 (filtered Hamilton MSM probability)

    CSMOM is no longer a standalone weight — it modulates TSMOM.
    Carry removed (data quality concerns; ETF yield is not true carry).

    Gate threshold: score < 35 blocks overweight (recalibrate after n≥10).

    Returns DataFrame index=sector, columns:
        composite_score  : 0–100 weighted composite
        tsmom_norm       : TSMOM base score after CSMOM truncation (0–100)
        csmom_rank       : 0–100 (percentile), kept for display
        sharpe_norm      : 0–100 (sigmoid)
        sharpe_raw       : raw Sharpe ratio
        regime_score     : p_risk_on × 100
    """
    base = get_signal_dataframe(as_of, lookback_months, skip_months)
    if base.empty:
        return pd.DataFrame()

    # ── P2-2 TSMOM continuization: raw_return/ann_vol (formation Sharpe)
    # replaces binary {+1→100, 0→50, -1→0}; cross-sectional min-max → [0, 100]
    _sizing_vol = base["ann_vol_21d"] if "ann_vol_21d" in base.columns else base["ann_vol"]
    _sharpe_cs  = base["raw_return"] / _sizing_vol.clip(lower=1e-6)
    _smin, _smax = _sharpe_cs.min(), _sharpe_cs.max()
    if _smax > _smin + 1e-9:
        base["tsmom_norm"] = ((_sharpe_cs - _smin) / (_smax - _smin) * 100.0).clip(0.0, 100.0)
    else:
        base["tsmom_norm"] = 50.0
    # Zero-signal assets (tsmom==0) anchored to neutral 50
    base.loc[base["tsmom"] == 0, "tsmom_norm"] = 50.0

    # ── CSMOM cross-sectional rank — within-class (P2-12 A-3)
    try:
        from engine.universe_manager import get_universe_by_class as _get_by_class2
        _ubc = _get_by_class2()
    except Exception:
        _ubc = {}

    if _ubc:
        _rank_series = pd.Series(dtype=float, name="csmom_rank")
        for _ac, _cmap in _ubc.items():
            _csectors = [s for s in _cmap.keys() if s in base.index]
            if len(_csectors) < 2:
                _rank_series = pd.concat([_rank_series, pd.Series(50.0, index=[s for s in _csectors])])
                continue
            _cls_ranks = base.loc[_csectors, "raw_return"].rank(pct=True) * 100
            _rank_series = pd.concat([_rank_series, _cls_ranks])
        base["csmom_rank"] = _rank_series.reindex(base.index).fillna(50.0)
    else:
        base["csmom_rank"] = base["raw_return"].rank(pct=True) * 100.0

    # ── CSMOM truncation: direction-rank conflict caps tsmom_norm
    def _apply_csmom_truncation(row) -> float:
        score = row["tsmom_norm"]
        rank  = row["csmom_rank"]
        if row["tsmom"] == 1.0 and rank < 33.3:
            score = min(score, 70.0)
        elif row["tsmom"] == -1.0 and rank > 66.7:
            score = max(score, 30.0)
        return score

    base["tsmom_norm"] = base.apply(_apply_csmom_truncation, axis=1)

    # ── Sharpe ratio
    def _sharpe(row) -> float:
        if row["ann_vol"] and row["ann_vol"] > 1e-6 and not np.isnan(row["ann_vol"]):
            ann_return = row["raw_return"] * (12.0 / 11.0)
            return ann_return / row["ann_vol"]
        return 0.0

    base["sharpe_raw"]  = base.apply(_sharpe, axis=1)
    base["sharpe_norm"] = base["sharpe_raw"].apply(_sigmoid_norm)

    # ── Regime score: p_risk_on from Hamilton MSM (filtered probability)
    try:
        from engine.regime import get_regime_on
        _regime_result = get_regime_on(as_of)
        _p_risk_on = getattr(_regime_result, "p_risk_on", 0.5)
        base["regime_score"] = float(_p_risk_on) * 100.0
    except Exception:
        base["regime_score"] = 50.0  # neutral fallback

    # ── FactorMAD overlay (P2-13): replaces regime_score slot when ≥3 active factors
    try:
        from engine.factor_mad import get_factor_mad_scores
        _factor_mad = get_factor_mad_scores(as_of, asset_class="equity_sector", min_factors=3)
    except Exception:
        _factor_mad = None

    if _factor_mad is not None:
        base["factor_mad_score"] = _factor_mad.reindex(base.index).fillna(50.0)
    else:
        base["factor_mad_score"] = np.nan

    # ── Carry factor (P3-1: net carry normalized 0–100 via sigmoid) ──────────
    try:
        _carry_raw = compute_carry(as_of)
        _c_vals    = np.array(list(_carry_raw.values()))
        _c_mean    = float(np.nanmean(_c_vals))
        _c_std     = float(np.nanstd(_c_vals)) or 1.0
        base["carry_norm"] = base.index.map(
            lambda s: _sigmoid_norm((_carry_raw.get(s, 0.0) - _c_mean) / _c_std)
        )
    except Exception:
        base["carry_norm"] = 50.0

    # ── Reversal factor (P3-2: price-to-5Y-SMA, normalized 0–100) ────────────
    try:
        _rev_raw = compute_reversal(as_of)
        _r_vals  = np.array(list(_rev_raw.values()))
        _r_lo, _r_hi = np.percentile(_r_vals, 5), np.percentile(_r_vals, 95)
        _r_range = max(_r_hi - _r_lo, 1e-8)
        base["reversal_norm"] = base.index.map(
            lambda s: float(np.clip(
                (_rev_raw.get(s, 0.0) - _r_lo) / _r_range * 100.0, 0.0, 100.0
            ))
        )
    except Exception:
        base["reversal_norm"] = 50.0

    # ── Fixed composite weights (P6 / M2 resolution) ─────────────────────────
    # COMPOSITE_WEIGHTS is a module-level constant; not loaded from SystemConfig.
    # Outside 'transition', compute_reversal returns all zeros → reversal_norm=0.
    # The 5% weight is kept (M2: 不重分配) so composite ceiling becomes 95 in those regimes.
    _overlay = base["factor_mad_score"].where(base["factor_mad_score"].notna(), base["regime_score"])

    base["composite_score"] = (
        COMPOSITE_WEIGHTS["tsmom"]      * base["tsmom_norm"]    +
        COMPOSITE_WEIGHTS["csmom"]      * base["csmom_rank"]    +
        COMPOSITE_WEIGHTS["carry"]      * base["carry_norm"]    +
        COMPOSITE_WEIGHTS["reversal"]   * base["reversal_norm"] +
        COMPOSITE_WEIGHTS["factor_mad"] * _overlay
    ).round(1)

    cols = ["composite_score", "tsmom_norm", "csmom_rank",
            "sharpe_norm", "sharpe_raw", "regime_score",
            "factor_mad_score", "carry_norm", "reversal_norm"]
    return base[cols]


def update_factor_ic_weights(as_of: datetime.date, lookback_months: int = 12) -> dict:
    """
    Compute rolling IC for each factor over the past lookback_months and
    update factor_ic_weights in SystemConfig.

    IC = Spearman rank correlation between factor score at t and
         forward 1-month return at t+1, averaged over lookback window.

    Called by the monthly ICIR update job (daily_batch.py).
    Returns the new weight dict.
    """
    import datetime as _dt
    from engine.memory import get_system_config, set_system_config, SimulatedMonthlyReturn, SessionFactory
    import json as _json

    factor_ics: dict[str, list[float]] = {
        "tsmom": [], "sharpe": [], "carry": [], "reversal": [], "factor_mad": [],
    }

    for lag in range(1, lookback_months + 1):
        t = (as_of.replace(day=1) - _dt.timedelta(days=1)).replace(day=1)
        t = t.replace(month=((t.month - lag - 1) % 12) + 1,
                      year=t.year + ((t.month - lag - 1) // 12))
        try:
            scores_t  = compute_composite_scores(t, lookback_months=12, skip_months=1)
            # Forward return: from SimulatedMonthlyReturn for month t+1
            with SessionFactory() as sess:
                fwd_month = (t.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
                rows = sess.query(SimulatedMonthlyReturn).filter(
                    SimulatedMonthlyReturn.return_month == fwd_month
                ).all()
            if not rows:
                continue
            fwd_ret = {r.sector: r.sector_return for r in rows if r.sector_return is not None}
            common  = scores_t.index.intersection(list(fwd_ret.keys()))
            if len(common) < 5:
                continue
            for factor, col in [
                ("tsmom", "tsmom_norm"), ("sharpe", "sharpe_norm"),
                ("carry", "carry_norm"), ("reversal", "reversal_norm"),
                ("factor_mad", "factor_mad_score"),
            ]:
                if col not in scores_t.columns:
                    continue
                x = scores_t.loc[common, col].rank()
                y = pd.Series(fwd_ret).reindex(common).rank()
                if x.std() < 1e-8 or y.std() < 1e-8:
                    continue
                ic = float(x.corr(y, method="spearman"))
                factor_ics[factor].append(ic)
        except Exception:
            continue

    # Mean IC per factor; floor at 0 (negative IC → set to 0 weight, not negative)
    mean_ics = {f: max(0.0, float(np.mean(v))) if v else 0.0
                for f, v in factor_ics.items()}

    ic_sum = sum(mean_ics.values())
    if ic_sum < 1e-8:
        return {}   # insufficient data — keep existing weights

    new_weights = {f: round(ic / ic_sum, 4) for f, ic in mean_ics.items()}
    try:
        set_system_config("factor_ic_weights", _json.dumps(new_weights))
    except Exception:
        pass
    return new_weights


def get_quant_gates(
    as_of:           datetime.date,
    regime_label:    str = "unknown",
    lookback_months: int = 12,
    skip_months:     int = 1,
) -> dict[str, dict]:
    """
    Compute per-sector quant gate constraints.
    All rules use ONLY market data — Goodhart-safe.

    Gate dict per sector:
        allowed    : list[str]  directions LLM may output
        blocked    : list[str]  directions hard-blocked
        soft_warn  : bool       composite in warning zone
        composite  : float      0–100 composite score
        tsmom      : int        -1/0/+1
        csmom      : int        -1/0/+1
        reason     : str        human-readable gate rationale
        severity   : str        'hard' | 'soft' | 'clear'

    Hard rules (applied in order):
      R1  TSMOM=-1 AND CSMOM=-1  → 超配 blocked
      R2  TSMOM=+1 AND CSMOM=+1 → 低配 blocked
      R3  composite < 20         → only 低配/拦截 allowed
      R4  composite > 80         → 低配 blocked
      R5  regime=risk-off        → 超配 blocked (portfolio-level)
    Soft rule:
      S1  composite 20–35 AND direction=超配 → warn (not hard block)
    """
    signals  = get_signal_dataframe(as_of, lookback_months, skip_months)
    scores_df = compute_composite_scores(as_of, lookback_months, skip_months)
    risk_off  = (regime_label == "risk-off")

    gates: dict[str, dict] = {}

    for sector in signals.index:
        tsmom = int(signals.loc[sector, "tsmom"])
        csmom = int(signals.loc[sector, "csmom"])
        comp  = float(scores_df.loc[sector, "composite_score"]) \
                if sector in scores_df.index else 50.0

        blocked: set[str] = set()
        reasons: list[str] = []

        # R1 — both signals bearish → block overweight
        if tsmom == -1 and csmom == -1:
            blocked.add("超配")
            reasons.append("R1: TSMOM=-1 且 CSMOM=-1")

        # R2 — both signals bullish → block underweight
        if tsmom == 1 and csmom == 1:
            blocked.add("低配")
            reasons.append("R2: TSMOM=+1 且 CSMOM=+1")

        # R3 — very low composite → only underweight allowed
        if comp < 20:
            blocked.update({"超配", "标配"})
            reasons.append(f"R3: composite={comp:.0f}<20")

        # R4 — very high composite → underweight blocked
        if comp > 80:
            blocked.add("低配")
            reasons.append(f"R4: composite={comp:.0f}>80")

        # R5 — portfolio-level risk-off → no new overweights
        if risk_off:
            blocked.add("超配")
            reasons.append("R5: regime=risk-off")

        allowed = [d for d in _ALL_SECTOR_DIRS if d not in blocked]

        # Soft warning zone
        soft_warn = (20 <= comp <= 35)

        severity = "clear"
        if "超配" in blocked and "标配" in blocked:
            severity = "hard"
        elif blocked:
            severity = "hard"
        elif soft_warn:
            severity = "soft"

        gates[sector] = {
            "allowed":   allowed,
            "blocked":   sorted(blocked),
            "soft_warn": soft_warn,
            "composite": comp,
            "tsmom":     tsmom,
            "csmom":     csmom,
            "reason":    " | ".join(reasons) if reasons else "无约束",
            "severity":  severity,
        }

    return gates


def format_gate_for_prompt(gate: dict | None, sector_name: str = "") -> str:
    """
    Format a single sector's gate dict as a concise constraint block
    to inject into Blue/Arbitration LLM prompts.
    Returns empty string if gate is None or has no blocks.
    """
    if not gate or not gate.get("blocked"):
        return ""

    blocked_str  = "、".join(gate["blocked"])
    allowed_str  = "、".join(gate["allowed"]) if gate["allowed"] else "无（全部拦截）"
    soft_str     = "⚠️ 注意：composite 处于软警告区间（20–35），给出超配需格外充分的论据。" \
                   if gate.get("soft_warn") else ""

    return (
        f"\n\n【量化门控约束 — 必须遵守】\n"
        f"板块：{sector_name or '当前板块'}\n"
        f"TSMOM={gate['tsmom']:+d}  CSMOM={gate['csmom']:+d}  "
        f"合成分={gate['composite']:.0f}/100\n"
        f"触发规则：{gate['reason']}\n"
        f"⛔ 禁止方向：{blocked_str}\n"
        f"✅ 允许方向：{allowed_str}\n"
        f"{soft_str}"
        f"（此约束来自纯量化模型，不受本次分析推理影响，必须在最终建议中遵守）\n"
    )


# ── P4-1: TSMOM-Fast signal ────────────────────────────────────────────────────

def get_fast_signal_dataframe(
    as_of:    "datetime.date",
    lookback: int = 3,
    skip:     int = 1,
) -> "pd.DataFrame":
    """TSMOM-Fast (3-1 month) direction signal.

    Direction only — no weights generated.
    Fast signal is used solely for directional confirmation and flip detection
    in _patrol_daily_tactical().  Weights always come from Slow TSMOM (12-1).
    """
    import datetime as _dt
    return get_signal_dataframe(
        as_of=as_of,
        lookback_months=lookback,
        skip_months=skip,
        use_cache=False,   # daily patrol always reads fresh data
    )
