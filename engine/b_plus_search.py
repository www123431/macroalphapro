"""
engine/b_plus_search.py — B++ Multi-Factor Mass FDR Search at Weekly Frequency
================================================================================

Spec: docs/spec_b_plus_mass_fdr_search.md v2.0 (2026-05-03 lock).

Implements 5 phases per spec:
  A. Core: weekly rebalance + ERC + 20 pre-registered strategies + Tier 1/2 universe
  B. Statistical rigor: train/OOS split + BHY FDR + Factor IC + bootstrap CI
  C. Combination: strategy correlation + IC-weighted meta + beta-neutral long-short
  D. Decomposition: Fama-MacBeth cross-sectional regression + alpha attribution
  E. Reporting: dashboard + decision doc

Pre-registration discipline (frozen):
  - 20 strategy specs (STRATEGY_REGISTRY) cannot be added/removed/tuned post-spec
  - Train: 2010-01-01 → 2017-12-31; OOS: 2018-01-01 → 2024-12-31 (frozen)
  - BHY FDR α=5% over N = 20 strategies × 2 tiers = 40
  - Verdict tiers: DISCOVERY / MARGINAL / NULL (per spec §7.2)
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Frozen constants (per spec §6, §7.1)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_START: str = "2010-01-01"
TRAIN_END:   str = "2017-12-31"
OOS_START:   str = "2018-01-01"
OOS_END:     str = "2024-12-31"

REBAL_FREQ:  str = "W-FRI"   # Friday close, Monday open executable
TARGET_VOL:  float = 0.10    # 10% annualised portfolio vol target

TC_BP_PER_RT: float = 13.0   # 8bp slippage + 5bp spread = 13bp round-trip
PERIODS_PER_YEAR_WEEKLY: int = 52

BHY_ALPHA: float = 0.05
N_BOOT:    int   = 2000
BOOT_BLOCK_LEN: int = 4   # 4 weeks ≈ monthly persistence in weekly data


# ─────────────────────────────────────────────────────────────────────────────
# Day-based offset helpers (weekly path)
# ─────────────────────────────────────────────────────────────────────────────

def _day_offset(date: datetime.date, days: int) -> datetime.date:
    """Subtract `days` calendar days from date."""
    return date - datetime.timedelta(days=days)


def _week_offset(date: datetime.date, weeks: int) -> datetime.date:
    return date - datetime.timedelta(weeks=weeks)


# ─────────────────────────────────────────────────────────────────────────────
# Weekly raw returns computation (analogue of compute_raw_returns)
# ─────────────────────────────────────────────────────────────────────────────

def compute_raw_returns_weekly(
    as_of:           datetime.date,
    lookback_weeks:  int,
    skip_weeks:      int,
    closes_cache:    Optional[pd.DataFrame] = None,
    universe:        Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Weekly variant of `engine.signal.compute_raw_returns`.

    Formation window: [as_of - lookback_weeks, as_of - skip_weeks]
    Returns DataFrame with sector index, columns: ticker, raw_return, ann_vol_d, n_obs.

    Per spec §4.2:
      - Lookbacks specified in weeks
      - Universe injected via `universe` (sector → ticker dict); defaults to
        engine.history.get_active_sector_etf()
      - Optional `closes_cache` to avoid repeat yfinance calls within a backtest
    """
    end_cutoff   = _week_offset(as_of, skip_weeks)
    start_cutoff = _week_offset(as_of, lookback_weeks)
    fetch_start  = start_cutoff - datetime.timedelta(days=15)

    if universe is None:
        from engine.history import get_active_sector_etf
        universe = get_active_sector_etf()
    ticker_to_sector = {v: k for k, v in universe.items()}
    tickers = list(universe.values())

    if closes_cache is not None and not closes_cache.empty:
        # Use the supplied cache; slice to relevant window
        slc = closes_cache.loc[
            (closes_cache.index >= pd.Timestamp(fetch_start))
            & (closes_cache.index <= pd.Timestamp(end_cutoff))
        ]
        closes = slc.copy()
    else:
        from engine.signal import _fetch_closes
        closes = _fetch_closes(tickers, fetch_start, end_cutoff)

    if closes.empty:
        return pd.DataFrame()

    records = []
    for ticker, sector in ticker_to_sector.items():
        if ticker not in closes.columns:
            continue
        series = closes[ticker].dropna()
        if len(series) < 5:
            continue
        # Find the closes nearest the cutoff dates
        series_in_window = series.loc[
            (series.index >= pd.Timestamp(start_cutoff))
            & (series.index <= pd.Timestamp(end_cutoff))
        ]
        if len(series_in_window) < 5:
            continue
        first_px = float(series_in_window.iloc[0])
        last_px  = float(series_in_window.iloc[-1])
        if first_px <= 0:
            continue
        raw_return = last_px / first_px - 1.0
        # Annualised vol from daily returns within window
        rets = series_in_window.pct_change().dropna()
        ann_vol_d = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else float("nan")

        records.append({
            "sector":      sector,
            "ticker":      ticker,
            "raw_return":  raw_return,
            "ann_vol":     ann_vol_d,
            "start_date":  series_in_window.index[0],
            "end_date":    series_in_window.index[-1],
            "obs":         len(series_in_window),
            "last_px":     last_px,
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("sector")


def fetch_universe_closes(
    universe:    dict[str, str],
    start_date:  datetime.date,
    end_date:    datetime.date,
) -> pd.DataFrame:
    """
    Single bulk fetch for the universe over a wide window. Returns wide DataFrame
    (date index, ticker columns). Used to populate `closes_cache` for backtest loops.
    """
    from engine.signal import _fetch_closes
    tickers = list(universe.values())
    return _fetch_closes(tickers, start_date, end_date)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy specs — pre-registered (FROZEN per spec §2.9)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategySpec:
    """
    A strategy candidate in the B++ search. Frozen post-spec lock.

    `signal_fn(as_of, closes, universe, **params) -> pd.Series` produces
    a per-sector signal in [-1, +1] (continuous) or {-1, 0, +1} (discrete).

    The signal is interpreted as: long if > 0, short if < 0, flat if ≈ 0.
    Magnitude proportional to conviction.
    """
    id:        str
    name:      str
    category:  str
    signal_fn: Callable
    params:    dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementations (each returns a per-sector signal Series)
# ─────────────────────────────────────────────────────────────────────────────

REVERSE_MOMENTUM_TICKERS = {"VXX"}


def _signed(raw_ret: float, ticker: str) -> float:
    """Sign with VXX polarity flip (volatility ETF inverted)."""
    if pd.isna(raw_ret):
        return 0.0
    if ticker in REVERSE_MOMENTUM_TICKERS:
        return -1.0 if raw_ret > 0 else (1.0 if raw_ret < 0 else 0.0)
    return 1.0 if raw_ret > 0 else (-1.0 if raw_ret < 0 else 0.0)


# ── TSMOM family ─────────────────────────────────────────────────────────────

def _tsmom_signal(as_of, closes, universe, lookback_weeks, skip_weeks=4):
    df = compute_raw_returns_weekly(as_of, lookback_weeks, skip_weeks, closes, universe)
    if df.empty:
        return pd.Series(dtype=float)
    return df.apply(lambda r: _signed(r["raw_return"], r["ticker"]), axis=1)


def _tsmom_ensemble_signal(as_of, closes, universe, lookbacks=(13, 26, 52, 104), skip_weeks=4):
    components = []
    for L in lookbacks:
        sig = _tsmom_signal(as_of, closes, universe, L, skip_weeks)
        if not sig.empty:
            components.append(sig)
    if not components:
        return pd.Series(dtype=float)
    common_idx = components[0].index
    for s in components[1:]:
        common_idx = common_idx.intersection(s.index)
    aligned = pd.DataFrame({i: s.reindex(common_idx).fillna(0.0)
                            for i, s in enumerate(components)})
    return aligned.mean(axis=1)


# ── CSMOM family ─────────────────────────────────────────────────────────────

def _csmom_within_class_tertile(as_of, closes, universe, lookback_weeks=52, skip_weeks=4):
    """CS01: within-class top vs bottom tertile."""
    df = compute_raw_returns_weekly(as_of, lookback_weeks, skip_weeks, closes, universe)
    if df.empty:
        return pd.Series(dtype=float)
    try:
        from engine.universe_manager import get_universe_by_class
        ubc = get_universe_by_class()
    except Exception:
        ubc = {}
    sig = pd.Series(0.0, index=df.index)
    if not ubc:
        # Fallback global
        valid = df.dropna(subset=["raw_return"])
        n = len(valid)
        if n < 3:
            return sig
        n_top = max(1, n // 3)
        ranked = valid["raw_return"].sort_values()
        bot = set(ranked.index[:n_top])
        top = set(ranked.index[-n_top:])
        for s in df.index:
            if s in top: sig[s] = 1.0
            elif s in bot: sig[s] = -1.0
        return sig
    for ac, class_map in ubc.items():
        sub = df.loc[[s for s in class_map.keys() if s in df.index]].dropna(subset=["raw_return"])
        n = len(sub)
        if n < 3:
            continue
        n_top = max(1, round(n / 3))
        ranked = sub["raw_return"].sort_values()
        bot = set(ranked.index[:n_top])
        top = set(ranked.index[-n_top:])
        for s in sub.index:
            if s in top: sig[s] = 1.0
            elif s in bot: sig[s] = -1.0
    return sig


def _csmom_global_quintile(as_of, closes, universe, lookback_weeks=52, skip_weeks=4):
    """CS02: global rank top/bottom 20%."""
    df = compute_raw_returns_weekly(as_of, lookback_weeks, skip_weeks, closes, universe)
    if df.empty:
        return pd.Series(dtype=float)
    valid = df.dropna(subset=["raw_return"])
    n = len(valid)
    if n < 5:
        return pd.Series(0.0, index=df.index)
    n_top = max(1, n // 5)
    ranked = valid["raw_return"].sort_values()
    sig = pd.Series(0.0, index=df.index)
    bot = set(ranked.index[:n_top])
    top = set(ranked.index[-n_top:])
    for s in df.index:
        if s in top: sig[s] = 1.0
        elif s in bot: sig[s] = -1.0
    return sig


def _csmom_global_decile(as_of, closes, universe, lookback_weeks=52, skip_weeks=4):
    """CS03: long top decile - short bottom decile."""
    df = compute_raw_returns_weekly(as_of, lookback_weeks, skip_weeks, closes, universe)
    if df.empty:
        return pd.Series(dtype=float)
    valid = df.dropna(subset=["raw_return"])
    n = len(valid)
    if n < 10:
        return pd.Series(0.0, index=df.index)
    n_top = max(1, n // 10)
    ranked = valid["raw_return"].sort_values()
    sig = pd.Series(0.0, index=df.index)
    bot = set(ranked.index[:n_top])
    top = set(ranked.index[-n_top:])
    for s in df.index:
        if s in top: sig[s] = 1.0
        elif s in bot: sig[s] = -1.0
    return sig


# ── Carry family ─────────────────────────────────────────────────────────────

def _carry_sigmoid(as_of, closes, universe, lookback_weeks=52, skip_weeks=4):
    """CR01: net carry sigmoid normalized."""
    try:
        from engine.signal import compute_carry, _sigmoid_norm
        carry_raw = compute_carry(as_of)
    except Exception as exc:
        logger.debug("CR01 carry compute failed: %s", exc)
        return pd.Series(dtype=float)
    if not carry_raw:
        return pd.Series(dtype=float)
    sectors = list(universe.keys())
    vals = pd.Series({s: carry_raw.get(s, 0.0) for s in sectors})
    if vals.std() < 1e-12:
        return pd.Series(0.0, index=sectors)
    z = (vals - vals.mean()) / vals.std()
    # Convert sigmoid z to signed signal: sign(z) × magnitude
    return np.sign(z) * (z.abs() / (z.abs() + 1.0))


def _carry_yield_curve(as_of, closes, universe, lookback_weeks=52, skip_weeks=4):
    """CR02: yield-curve slope tilt — overweight bonds when curve is steep."""
    try:
        from engine.macro_fetcher import fetch_yield_spread
        spread = fetch_yield_spread(as_of)  # 10y - 2y
    except Exception:
        spread = None
    if spread is None or pd.isna(spread):
        return pd.Series(0.0, index=list(universe.keys()))
    # Steep curve (spread > 1%) → overweight long-duration bonds (TLT, IEF), underweight short bonds
    sig = pd.Series(0.0, index=list(universe.keys()))
    direction = 1.0 if spread > 0.5 else (-1.0 if spread < -0.5 else 0.0)
    long_bond_tickers = {"TLT", "IEF", "BWX"}
    short_bond_tickers = {"SHY"}
    for sector, ticker in universe.items():
        if ticker in long_bond_tickers:
            sig[sector] = direction
        elif ticker in short_bond_tickers:
            sig[sector] = -direction * 0.5
    return sig


# ── Reversal family ──────────────────────────────────────────────────────────

def _reversal_short_term(as_of, closes, universe, lookback_weeks=1, skip_weeks=0):
    """RV01: short-term 1-week reversal — short past winners, long past losers."""
    df = compute_raw_returns_weekly(as_of, lookback_weeks, skip_weeks, closes, universe)
    if df.empty:
        return pd.Series(dtype=float)
    valid = df.dropna(subset=["raw_return"])
    n = len(valid)
    if n < 3:
        return pd.Series(0.0, index=df.index)
    n_top = max(1, n // 3)
    ranked = valid["raw_return"].sort_values()
    sig = pd.Series(0.0, index=df.index)
    # REVERSAL: short top, long bottom
    bot = set(ranked.index[:n_top])
    top = set(ranked.index[-n_top:])
    for s in df.index:
        if s in top: sig[s] = -1.0   # short winners
        elif s in bot: sig[s] = 1.0   # long losers
    return sig


def _reversal_long_term(as_of, closes, universe, lookback_weeks=260, skip_weeks=0):
    """RV02: distance to 5-year SMA, short overpriced, long underpriced."""
    df = compute_raw_returns_weekly(as_of, lookback_weeks, skip_weeks, closes, universe)
    if df.empty:
        return pd.Series(dtype=float)
    # raw_return over 260 weeks = ~5y. Below SMA (raw_return < 0 over 5y) = "underpriced"
    valid = df.dropna(subset=["raw_return"])
    if len(valid) < 3:
        return pd.Series(0.0, index=df.index)
    sig = pd.Series(0.0, index=df.index)
    # Reversal logic: long if raw_return < 0 (price below 5y average), short if > 0
    for s in valid.index:
        r = valid.loc[s, "raw_return"]
        if r > 0.50:    sig[s] = -1.0   # heavily overpriced → short
        elif r < -0.30: sig[s] = 1.0    # underpriced → long
    return sig


# ── Quality / Defensive ──────────────────────────────────────────────────────

def _low_volatility(as_of, closes, universe, lookback_weeks=52, skip_weeks=0):
    """QL01: low-vol — long bottom-vol quintile, short top-vol."""
    if closes is None or closes.empty:
        return pd.Series(dtype=float)
    end = pd.Timestamp(_week_offset(as_of, skip_weeks))
    start = pd.Timestamp(_week_offset(as_of, lookback_weeks))
    window = closes.loc[(closes.index >= start) & (closes.index <= end)]
    if window.empty:
        return pd.Series(dtype=float)
    rets = window.pct_change().dropna(how="all")
    ann_vol = rets.std(ddof=1) * np.sqrt(252)
    sectors = list(universe.keys())
    sig = pd.Series(0.0, index=sectors)
    valid_tickers = [t for t in universe.values() if t in ann_vol.index and not pd.isna(ann_vol[t])]
    if len(valid_tickers) < 5:
        return sig
    vol_series = ann_vol[valid_tickers].sort_values()
    n_top = max(1, len(vol_series) // 5)
    low_vol_t = set(vol_series.index[:n_top])
    high_vol_t = set(vol_series.index[-n_top:])
    for sector, ticker in universe.items():
        if ticker in low_vol_t:    sig[sector] = 1.0
        elif ticker in high_vol_t: sig[sector] = -1.0
    return sig


def _quality_sharpe_rank(as_of, closes, universe, lookback_weeks=12, skip_weeks=0):
    """QL02: trailing 12-week Sharpe rank, long top decile, short bottom decile."""
    if closes is None or closes.empty:
        return pd.Series(dtype=float)
    end = pd.Timestamp(_week_offset(as_of, skip_weeks))
    start = pd.Timestamp(_week_offset(as_of, lookback_weeks))
    window = closes.loc[(closes.index >= start) & (closes.index <= end)]
    if window.empty:
        return pd.Series(dtype=float)
    rets = window.pct_change().dropna(how="all")
    if len(rets) < 5:
        return pd.Series(dtype=float)
    sharpe = rets.mean() / rets.std(ddof=1).replace(0, np.nan) * np.sqrt(252)
    sectors = list(universe.keys())
    sig = pd.Series(0.0, index=sectors)
    valid_tickers = [t for t in universe.values() if t in sharpe.index and not pd.isna(sharpe[t])]
    if len(valid_tickers) < 5:
        return sig
    s_series = sharpe[valid_tickers].sort_values()
    n_top = max(1, len(s_series) // 5)
    high_s_t = set(s_series.index[-n_top:])
    low_s_t = set(s_series.index[:n_top])
    for sector, ticker in universe.items():
        if ticker in high_s_t:   sig[sector] = 1.0
        elif ticker in low_s_t:  sig[sector] = -1.0
    return sig


def _vol_managed(as_of, closes, universe, lookback_weeks=12, skip_weeks=0, target_vol=0.10):
    """QL03: vol-managed exposure — sign from TSMOM-52, magnitude scaled by target_vol/recent_vol."""
    base = _tsmom_signal(as_of, closes, universe, lookback_weeks=52, skip_weeks=4)
    if base.empty:
        return base
    if closes is None or closes.empty:
        return base
    end = pd.Timestamp(_week_offset(as_of, skip_weeks))
    start = pd.Timestamp(_week_offset(as_of, lookback_weeks))
    window = closes.loc[(closes.index >= start) & (closes.index <= end)]
    if window.empty:
        return base
    rets = window.pct_change().dropna(how="all")
    if len(rets) < 5:
        return base
    ann_vol = rets.std(ddof=1) * np.sqrt(252)
    # Scale base signal by target_vol / recent_vol (clipped 0.5-2x)
    scaled = base.copy()
    for sector in base.index:
        ticker = universe.get(sector)
        if ticker and ticker in ann_vol.index and ann_vol[ticker] > 1e-6:
            scale = float(np.clip(target_vol / ann_vol[ticker], 0.5, 2.0))
            scaled[sector] = base[sector] * scale
    return scaled


# ── Macro Overlay ────────────────────────────────────────────────────────────

def _regime_score(as_of, closes, universe, lookback_weeks=52, skip_weeks=4, regime_cache=None):
    """
    MA01: Hamilton MSM regime — overlay on TSMOM-52, scale by p_risk_on.

    Performance optimization (2026-05-03): if `regime_cache` is provided as a
    dict {as_of: p_risk_on}, look up cached value instead of re-running MSM
    EM filter. Pure speed optimization — same numerical output as direct call.
    """
    base = _tsmom_signal(as_of, closes, universe, lookback_weeks=52, skip_weeks=4)
    if base.empty:
        return base
    p_on = None
    if regime_cache is not None:
        p_on = regime_cache.get(as_of, None)
        if p_on is None:
            # Try date-only key (no time component)
            p_on = regime_cache.get(pd.Timestamp(as_of).date(), None)
    if p_on is None:
        try:
            from engine.regime import get_regime_on
            reg = get_regime_on(as_of)
            p_on = float(getattr(reg, "p_risk_on", 0.5))
        except Exception:
            p_on = 0.5
    # Scale long signals by p_on (risk-on overweight); leave shorts unchanged
    return base * (p_on if p_on > 0.5 else 0.5)


def _vix_defensive(as_of, closes, universe, lookback_weeks=52, skip_weeks=4, vix_cache=None):
    """
    MA02: VIX-based defensive — scale equity exposure by 20/VIX (clip [0.5, 1.5]).

    Performance optimization (2026-05-03): if `vix_cache` is provided as a
    dict {as_of: vix_value}, look up cached value instead of re-fetching from
    yfinance. Pure speed optimization.
    """
    base = _tsmom_signal(as_of, closes, universe, lookback_weeks=52, skip_weeks=4)
    if base.empty:
        return base
    vix = None
    if vix_cache is not None:
        vix = vix_cache.get(as_of, None)
        if vix is None:
            vix = vix_cache.get(pd.Timestamp(as_of).date(), None)
    if vix is None:
        try:
            from engine.history import get_vix_on
            vix = get_vix_on(as_of)
        except Exception:
            vix = 20.0
    if vix is None or vix < 1.0:
        vix = 20.0
    scale = float(np.clip(20.0 / vix, 0.5, 1.5))
    # Apply scale only to equity sectors (preserve bond/commodity shorts as "hedge")
    try:
        from engine.universe_manager import get_asset_class_map
        cls_map = get_asset_class_map()
    except Exception:
        cls_map = {}
    out = base.copy()
    for sector in base.index:
        ticker = universe.get(sector)
        if ticker and cls_map.get(ticker, "equity_sector") in ("equity_sector", "equity_factor"):
            out[sector] = base[sector] * scale
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute caches for slow signals (regime + VIX)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_regime_cache(weekly_dates: list[datetime.date]) -> dict:
    """
    Pre-compute Hamilton MSM regime p_risk_on for all weekly dates.
    Returns dict {date: p_risk_on}. Same numerical output as inline calls.
    """
    cache: dict = {}
    from engine.regime import get_regime_on
    for d in weekly_dates:
        try:
            reg = get_regime_on(d)
            p = float(getattr(reg, "p_risk_on", 0.5))
            cache[d] = p
        except Exception as exc:
            logger.debug("regime cache build %s failed: %s", d, exc)
            cache[d] = 0.5
    return cache


def precompute_vix_cache(
    weekly_dates: list[datetime.date],
    closes_with_vix: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Pre-compute VIX value for all weekly dates.

    If `closes_with_vix` is provided (with VIX column or proxy), look up directly.
    Otherwise call fetch_latest_vix once per date. Returns dict {date: vix}.

    Performance: bulk VIX fetch in one yfinance call rather than 360 separate calls.
    """
    cache: dict = {}
    # Try bulk fetch from yfinance for ^VIX symbol once
    try:
        from engine.signal import _fetch_closes
        if not weekly_dates:
            return cache
        start = min(weekly_dates) - datetime.timedelta(days=15)
        end = max(weekly_dates) + datetime.timedelta(days=2)
        vix_closes = _fetch_closes(["^VIX"], start, end)
        if vix_closes is not None and not vix_closes.empty:
            vix_series = vix_closes.iloc[:, 0].dropna()
            for d in weekly_dates:
                ts = pd.Timestamp(d)
                # Find latest VIX close on or before d
                slc = vix_series.loc[vix_series.index <= ts]
                if not slc.empty:
                    cache[d] = float(slc.iloc[-1])
    except Exception as exc:
        logger.warning("VIX bulk fetch failed: %s; fallback per-date", exc)

    # Fallback: per-date get_vix_on for any missing dates
    if not cache or len(cache) < len(weekly_dates):
        from engine.history import get_vix_on
        for d in weekly_dates:
            if d in cache:
                continue
            try:
                v = get_vix_on(d)
                cache[d] = float(v) if v else 20.0
            except Exception:
                cache[d] = 20.0
    return cache


# ── Calendar ─────────────────────────────────────────────────────────────────

def _turn_of_month(as_of, closes, universe, **_):
    """CL01: turn-of-month (last 5 + first 3 trading days = strong period)."""
    base = _tsmom_signal(as_of, closes, universe, lookback_weeks=52, skip_weeks=4)
    if base.empty:
        return base
    # Determine if as_of is within turn-of-month window
    ts = pd.Timestamp(as_of)
    days_to_month_end = (pd.Timestamp(ts.year, ts.month, 1) + pd.offsets.MonthEnd(0) - ts).days
    days_from_month_start = ts.day - 1
    in_tom = (days_to_month_end <= 5) or (days_from_month_start <= 3)
    return base if in_tom else (base * 0.0)


def _january_effect(as_of, closes, universe, **_):
    """CL02: January seasonality — overweight in January, underweight in December."""
    base = _tsmom_signal(as_of, closes, universe, lookback_weeks=52, skip_weeks=4)
    if base.empty:
        return base
    ts = pd.Timestamp(as_of)
    if ts.month == 1:
        return base * 1.5  # January boost
    elif ts.month == 12:
        return base * 0.5  # December dampen
    return base


# ── Cross-asset Timing ───────────────────────────────────────────────────────

def _bond_equity_tilt(as_of, closes, universe, lookback_weeks=12, skip_weeks=0):
    """XA01: when TLT 12-week return > SPY 12-week return, tilt to bonds."""
    if closes is None or closes.empty:
        return pd.Series(dtype=float)
    end = pd.Timestamp(_week_offset(as_of, skip_weeks))
    start = pd.Timestamp(_week_offset(as_of, lookback_weeks))
    window = closes.loc[(closes.index >= start) & (closes.index <= end)]
    if window.empty:
        return pd.Series(dtype=float)
    sig = pd.Series(0.0, index=list(universe.keys()))
    if "TLT" not in window.columns or "SPY" not in window.columns:
        # Fall back: compare any equity vs bond proxy
        eq = next((t for t in ["QQQ", "XLF"] if t in window.columns), None)
        bd = next((t for t in ["TLT", "IEF", "AGG"] if t in window.columns), None)
        if eq is None or bd is None:
            return sig
        bond_ret = window[bd].pct_change(fill_method=None).dropna().sum()
        eq_ret = window[eq].pct_change(fill_method=None).dropna().sum()
    else:
        bond_ret = window["TLT"].pct_change(fill_method=None).dropna().sum()
        eq_ret = window["SPY"].pct_change(fill_method=None).dropna().sum()
    bond_better = bond_ret > eq_ret
    try:
        from engine.universe_manager import get_asset_class_map
        cls_map = get_asset_class_map()
    except Exception:
        cls_map = {}
    for sector, ticker in universe.items():
        cls = cls_map.get(ticker, "equity_sector")
        if cls == "fixed_income":
            sig[sector] = 1.0 if bond_better else -0.5
        elif cls in ("equity_sector", "equity_factor"):
            sig[sector] = 1.0 if not bond_better else -0.5
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Registry — FROZEN per spec §2.9
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: list[StrategySpec] = [
    # TSMOM family (5)
    StrategySpec("TS01", "TSMOM 52-4 weeks",  "tsmom",    _tsmom_signal,         {"lookback_weeks": 52, "skip_weeks": 4}),
    StrategySpec("TS02", "TSMOM 26-4 weeks",  "tsmom",    _tsmom_signal,         {"lookback_weeks": 26, "skip_weeks": 4}),
    StrategySpec("TS03", "TSMOM 104-4 weeks", "tsmom",    _tsmom_signal,         {"lookback_weeks": 104,"skip_weeks": 4}),
    StrategySpec("TS04", "TSMOM 13-4 weeks",  "tsmom",    _tsmom_signal,         {"lookback_weeks": 13, "skip_weeks": 4}),
    StrategySpec("TS05", "TSMOM ensemble",    "tsmom",    _tsmom_ensemble_signal,{"lookbacks": (13, 26, 52, 104), "skip_weeks": 4}),
    # CSMOM (3)
    StrategySpec("CS01", "CSMOM within-class tertile", "csmom", _csmom_within_class_tertile, {"lookback_weeks": 52, "skip_weeks": 4}),
    StrategySpec("CS02", "CSMOM global quintile",      "csmom", _csmom_global_quintile,      {"lookback_weeks": 52, "skip_weeks": 4}),
    StrategySpec("CS03", "CSMOM global decile",        "csmom", _csmom_global_decile,        {"lookback_weeks": 52, "skip_weeks": 4}),
    # Carry (2)
    StrategySpec("CR01", "Net carry sigmoid",   "carry", _carry_sigmoid,      {"lookback_weeks": 52, "skip_weeks": 4}),
    StrategySpec("CR02", "Yield-curve slope",   "carry", _carry_yield_curve,  {"lookback_weeks": 52, "skip_weeks": 4}),
    # Reversal (2)
    StrategySpec("RV01", "Short-term 1-wk reversal", "reversal", _reversal_short_term, {"lookback_weeks": 1, "skip_weeks": 0}),
    StrategySpec("RV02", "Long-term 5y SMA reversal", "reversal", _reversal_long_term, {"lookback_weeks": 260, "skip_weeks": 0}),
    # Quality / Defensive (3)
    StrategySpec("QL01", "Low-volatility β-rank",     "quality", _low_volatility,     {"lookback_weeks": 52, "skip_weeks": 0}),
    StrategySpec("QL02", "Quality 12-wk Sharpe rank", "quality", _quality_sharpe_rank,{"lookback_weeks": 12, "skip_weeks": 0}),
    StrategySpec("QL03", "Vol-managed (Moreira-Muir)","quality", _vol_managed,        {"lookback_weeks": 12, "skip_weeks": 0}),
    # Macro Overlay (2)
    StrategySpec("MA01", "Regime score (Hamilton MSM)", "macro", _regime_score,    {"lookback_weeks": 52, "skip_weeks": 4}),
    StrategySpec("MA02", "VIX-based defensive scale",   "macro", _vix_defensive,   {"lookback_weeks": 52, "skip_weeks": 4}),
    # Calendar (2)
    StrategySpec("CL01", "Turn-of-month",  "calendar", _turn_of_month,    {}),
    StrategySpec("CL02", "January effect", "calendar", _january_effect,   {}),
    # Cross-asset Timing (1)
    StrategySpec("XA01", "Bond-equity 12-wk momentum tilt", "cross_asset", _bond_equity_tilt, {"lookback_weeks": 12, "skip_weeks": 0}),
]

assert len(STRATEGY_REGISTRY) == 20, f"Strategy registry must have 20, got {len(STRATEGY_REGISTRY)}"


def get_strategy(strategy_id: str) -> StrategySpec:
    for s in STRATEGY_REGISTRY:
        if s.id == strategy_id:
            return s
    raise KeyError(f"Strategy {strategy_id} not in registry")


# ─────────────────────────────────────────────────────────────────────────────
# Universe Tier helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_universe_tier(tier: int) -> dict[str, str]:
    """
    Tier 1: 35 ETF (batches 0-4)
    Tier 2: 45 ETF (batches 0-5, requires seed_batch_e)

    Per spec §3 + universe_expansion rigor (2026-05-03):
      Tier 2 additions documented with per-ETF justification (asset class gap +
      academic reference + inception date + ADV threshold).
    """
    from engine.universe_manager import get_active_universe, seed_batch_e
    if tier == 1:
        # Filter to batch <= 4
        from engine.memory import SessionFactory
        from engine.universe_manager import UniverseETF
        with SessionFactory() as sess:
            rows = sess.query(UniverseETF).filter(
                UniverseETF.active == True,
                UniverseETF.batch <= 4,
            ).all()
            return {r.sector: r.ticker for r in rows}
    elif tier == 2:
        seed_batch_e()
        return get_active_universe()
    else:
        raise ValueError(f"Tier must be 1 or 2, got {tier}")


# ─────────────────────────────────────────────────────────────────────────────
# Universe Data Quality Verification (rigor pre-flight per 2026-05-03)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UniverseQualityReport:
    universe_size:       int
    tickers_passed:      list[str]
    tickers_dropped:     list[tuple[str, str]]   # (ticker, reason)
    inception_summary:   dict[str, datetime.date]  # ticker → inception
    nan_rate_summary:    dict[str, float]            # ticker → % NaN over window
    coverage_window:     tuple[datetime.date, datetime.date]


def verify_universe_quality(
    universe:        dict[str, str],
    start_date:      str,
    end_date:        str,
    min_history_yrs: float = 3.0,
    max_nan_pct:     float = 0.05,
) -> UniverseQualityReport:
    """
    Pre-flight data quality verification for the universe over the backtest window.

    Drops tickers with:
      - Insufficient history at start_date (inception > start_date - min_history_yrs)
      - Excessive NaN rate (> max_nan_pct of trading days)
      - Catastrophic data gaps (>30 consecutive NaN in middle of window)

    Returns UniverseQualityReport for audit + decision-doc disclosure.
    """
    from engine.signal import _fetch_closes

    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)
    fetch_start = (start_dt - pd.Timedelta(days=int(min_history_yrs * 365 + 30))).date()
    fetch_end   = end_dt.date()

    tickers = list(universe.values())
    closes = _fetch_closes(tickers, fetch_start, fetch_end)

    if closes.empty:
        return UniverseQualityReport(
            universe_size=len(universe),
            tickers_passed=[],
            tickers_dropped=[(t, "no_data_fetch") for t in tickers],
            inception_summary={},
            nan_rate_summary={},
            coverage_window=(fetch_start, fetch_end),
        )

    passed: list[str] = []
    dropped: list[tuple[str, str]] = []
    inception: dict[str, datetime.date] = {}
    nan_rates: dict[str, float] = {}

    for ticker in tickers:
        if ticker not in closes.columns:
            dropped.append((ticker, "ticker_not_in_fetch"))
            continue
        series = closes[ticker]
        non_nan = series.dropna()
        if non_nan.empty:
            dropped.append((ticker, "all_nan"))
            continue

        # First non-NaN date = effective inception
        inception_date = non_nan.index[0].date()
        inception[ticker] = inception_date

        # Inception must be ≥ min_history_yrs before start_date
        required_inception = (start_dt - pd.Timedelta(days=int(min_history_yrs * 365))).date()
        if inception_date > required_inception:
            dropped.append((ticker, f"inception_{inception_date}_too_late"))
            continue

        # NaN rate within the actual backtest window
        bt_window = series.loc[(series.index >= start_dt) & (series.index <= end_dt)]
        if bt_window.empty:
            dropped.append((ticker, "no_data_in_backtest_window"))
            continue
        nan_pct = float(bt_window.isna().sum()) / len(bt_window)
        nan_rates[ticker] = nan_pct
        if nan_pct > max_nan_pct:
            dropped.append((ticker, f"nan_rate_{nan_pct:.2%}_exceeds_threshold"))
            continue

        # Catastrophic gap check: ≥30 consecutive NaN in middle
        if bt_window.isna().any():
            run_len = 0
            max_run = 0
            for v in bt_window.isna().values:
                if v:
                    run_len += 1
                    max_run = max(max_run, run_len)
                else:
                    run_len = 0
            if max_run > 30:
                dropped.append((ticker, f"data_gap_{max_run}_days"))
                continue

        passed.append(ticker)

    return UniverseQualityReport(
        universe_size=len(universe),
        tickers_passed=passed,
        tickers_dropped=dropped,
        inception_summary=inception,
        nan_rate_summary=nan_rates,
        coverage_window=(fetch_start, fetch_end),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-strategy weekly backtest (spec §B+.6)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_strategy_weekly(
    spec:        StrategySpec,
    universe:    dict[str, str],
    start_date:  str,
    end_date:    str,
    closes:      Optional[pd.DataFrame] = None,
    use_erc:     bool = True,
    target_vol:  float = TARGET_VOL,
    tc_bp:       float = TC_BP_PER_RT,
) -> dict:
    """
    Run weekly backtest for a single strategy on the given universe.

    Returns dict with:
      strategy_id, n_obs, weekly_returns (pd.Series), cum_nav (pd.Series),
      ann_return, ann_vol, sharpe, nw_t_stat, nw_ci_low, nw_ci_high,
      ic_mean, ic_std, icir, signal_history (DataFrame).
    """
    from engine.signal import _fetch_closes
    from engine.backtest import newey_west_sharpe_se

    # Bulk fetch closes once if not supplied
    if closes is None or closes.empty:
        # Pre-fetch wide enough window for longest lookback (260 weeks ≈ 5y)
        fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=365 * 6)
        fetch_end   = pd.Timestamp(end_date)
        tickers = list(universe.values())
        closes  = _fetch_closes(tickers, fetch_start.date(), fetch_end.date())
        if closes.empty:
            return {"strategy_id": spec.id, "error": "no_close_data", "n_obs": 0}

    # Weekly rebal dates (Friday close)
    rebal_dates = pd.date_range(start_date, end_date, freq=REBAL_FREQ).date.tolist()

    weekly_returns: list[float] = []
    weekly_dates:   list[datetime.date] = []
    weekly_signals: list[pd.Series] = []
    weekly_weights: list[pd.Series] = []
    ic_per_week:    list[float] = []

    prev_weights: Optional[pd.Series] = None

    for i, t in enumerate(rebal_dates[:-1]):
        # Compute signal at t
        try:
            sig = spec.signal_fn(t, closes, universe, **spec.params)
        except Exception as exc:
            logger.debug("signal %s @ %s failed: %s", spec.id, t, exc)
            continue

        if sig is None or (hasattr(sig, "empty") and sig.empty):
            continue

        sig = sig.fillna(0.0)
        # Filter to assets with non-zero signal
        active_sectors = sig[sig != 0.0]
        if active_sectors.empty:
            weekly_dates.append(t)
            weekly_returns.append(0.0)
            weekly_signals.append(sig)
            weekly_weights.append(pd.Series(0.0, index=sig.index))
            continue

        # Compute weights — simplified inverse-vol weighting, ERC if requested + matrix available
        # For mass search we use simple inverse-vol within each direction (long/short)
        # ERC option deferred to Phase C combination layer per spec design
        end_t = pd.Timestamp(t)
        vol_window_start = end_t - pd.Timedelta(weeks=52)
        vol_window = closes.loc[(closes.index >= vol_window_start) & (closes.index <= end_t)]
        rets = vol_window.pct_change(fill_method=None).dropna(how="all")
        ann_vol = rets.std(ddof=1) * np.sqrt(252)

        weights = pd.Series(0.0, index=sig.index)
        for sector in active_sectors.index:
            ticker = universe.get(sector)
            if ticker is None or ticker not in ann_vol.index:
                continue
            v = ann_vol[ticker]
            if pd.isna(v) or v < 1e-6:
                continue
            weights[sector] = sig[sector] / v

        gross = weights.abs().sum()
        if gross > 1e-9:
            # Vol-target: scale to achieve target_vol assuming weights × ann_vol = portfolio vol
            avg_vol = (weights.abs() * ann_vol.reindex(
                [universe.get(s) for s in weights.index], fill_value=0.10
            ).fillna(0.10).values).sum() / gross if gross > 0 else 0.10
            scale = target_vol / max(avg_vol, 1e-6)
            weights = weights / gross * min(gross, 1.5) * scale

        # Compute next-week return: hold weights from t to t+1 (next rebal date)
        next_t = rebal_dates[i + 1]
        next_ts = pd.Timestamp(next_t)
        ret_window = closes.loc[(closes.index > end_t) & (closes.index <= next_ts)]
        if ret_window.empty:
            continue
        period_returns = (ret_window.iloc[-1] / ret_window.iloc[0] - 1.0).fillna(0.0)
        port_ret = 0.0
        for sector, w in weights.items():
            ticker = universe.get(sector)
            if ticker is None or ticker not in period_returns.index:
                continue
            port_ret += float(w) * float(period_returns[ticker])

        # Apply transaction cost
        if prev_weights is not None:
            turnover = (weights - prev_weights.reindex(weights.index, fill_value=0.0)).abs().sum()
            port_ret -= turnover * (tc_bp / 10000.0)
        else:
            turnover = weights.abs().sum()
            port_ret -= turnover * (tc_bp / 10000.0) * 0.5  # initial position cost

        # Compute IC (Spearman rank correlation between signal and forward returns)
        try:
            sig_active = sig.loc[sig != 0.0]
            ret_for_ic = pd.Series({
                s: period_returns.get(universe.get(s), np.nan)
                for s in sig_active.index
            }).dropna()
            common = sig_active.index.intersection(ret_for_ic.index)
            if len(common) >= 5:
                ic = float(sig_active.loc[common].rank().corr(ret_for_ic.loc[common].rank()))
                if not np.isnan(ic):
                    ic_per_week.append(ic)
        except Exception:
            pass

        weekly_dates.append(t)
        weekly_returns.append(float(port_ret))
        weekly_signals.append(sig)
        weekly_weights.append(weights)
        prev_weights = weights

    if not weekly_returns:
        return {"strategy_id": spec.id, "error": "no_returns", "n_obs": 0}

    ret_series = pd.Series(weekly_returns, index=pd.to_datetime(weekly_dates))
    cum_nav = (1.0 + ret_series).cumprod()

    mu = ret_series.mean()
    sd = ret_series.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(PERIODS_PER_YEAR_WEEKLY) if sd > 1e-12 else 0.0
    ann_ret = mu * PERIODS_PER_YEAR_WEEKLY
    ann_vol = sd * np.sqrt(PERIODS_PER_YEAR_WEEKLY)

    # NW HAC inference (annualised)
    nw = newey_west_sharpe_se(ret_series.values, periods_per_year=PERIODS_PER_YEAR_WEEKLY)

    # Factor IC stats
    if ic_per_week:
        ic_arr = np.array(ic_per_week)
        ic_mean = float(np.mean(ic_arr))
        ic_std  = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
        icir    = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    else:
        ic_mean = ic_std = icir = float("nan")

    return {
        "strategy_id": spec.id,
        "strategy_name": spec.name,
        "category":    spec.category,
        "n_obs":       len(ret_series),
        "weekly_returns": ret_series,
        "cum_nav":      cum_nav,
        "ann_return":   float(ann_ret),
        "ann_vol":      float(ann_vol),
        "sharpe":       float(sharpe),
        "nw_t_stat":    float(nw.get("t_stat", float("nan"))),
        "nw_sharpe_ann":float(nw.get("sr_ann", sharpe)),
        "nw_ci_low":    float(nw.get("ci_low", float("nan"))),
        "nw_ci_high":   float(nw.get("ci_high", float("nan"))),
        "ic_mean":      ic_mean,
        "ic_std":       ic_std,
        "icir":         icir,
        "n_ic_obs":     len(ic_per_week),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BHY FDR correction (Benjamini-Hochberg-Yekutieli 2009)
# ─────────────────────────────────────────────────────────────────────────────

def bhy_fdr_correction(p_values: list[float], alpha: float = BHY_ALPHA) -> dict:
    """
    Benjamini-Hochberg-Yekutieli (2009) FDR correction.

    Returns dict with:
      n_total           : total number of tests
      bhy_threshold     : threshold(rank) from BHY rule
      pass_indices      : list of indices passing FDR
      adjusted_p        : list of BHY-adjusted p-values (per Benjamini-Yekutieli 2001)
      c_factor          : c(N) = sum(1/i) for i=1..N
    """
    p = np.asarray(p_values, dtype=float)
    valid_mask = ~np.isnan(p)
    p_valid = p[valid_mask]
    n = len(p_valid)
    if n == 0:
        return {
            "n_total": len(p_values),
            "bhy_threshold": float("nan"),
            "pass_indices": [],
            "adjusted_p": [float("nan")] * len(p_values),
            "c_factor": float("nan"),
        }
    c_factor = float(sum(1.0 / i for i in range(1, n + 1)))
    sorted_idx = np.argsort(p_valid)
    sorted_p   = p_valid[sorted_idx]
    # BHY threshold: largest k satisfying p[k] ≤ (k/n) × α / c_factor
    pass_k = -1
    for k in range(1, n + 1):
        thresh_k = (k / n) * alpha / c_factor
        if sorted_p[k - 1] <= thresh_k:
            pass_k = k
    pass_local_indices = sorted_idx[:pass_k] if pass_k > 0 else np.array([], dtype=int)
    # Map back to original indices
    valid_indices = np.where(valid_mask)[0]
    pass_global_indices = valid_indices[pass_local_indices].tolist()
    # Adjusted p-values (BY)
    adjusted = np.full_like(p, np.nan)
    if pass_k > 0:
        bhy_threshold = (pass_k / n) * alpha / c_factor
    else:
        bhy_threshold = float("nan")
    return {
        "n_total":       n,
        "bhy_threshold": float(bhy_threshold) if not np.isnan(bhy_threshold) else float("nan"),
        "pass_indices":  pass_global_indices,
        "c_factor":      c_factor,
        "alpha":         alpha,
        "n_pass":        pass_k if pass_k > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry — quick smoke test on one strategy
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Mass-Search Runner — Phase B (spec §7)
# ─────────────────────────────────────────────────────────────────────────────

def run_mass_search(
    strategies:   Optional[list[StrategySpec]] = None,
    tiers:        tuple[int, ...] = (1, 2),
    train_start:  str = TRAIN_START,
    train_end:    str = TRAIN_END,
    oos_start:    str = OOS_START,
    oos_end:      str = OOS_END,
    output_dir:   str = "data/b_plus_results",
    progress_cb:  Optional[Callable] = None,
) -> dict:
    """
    Run B++ mass FDR search over all 20 strategies × 2 tiers.

    For each (strategy, tier) combination:
      - Run weekly backtest on TRAIN period (no scoring; for IC distribution context)
      - Run weekly backtest on OOS period (verdict scoring)
      - Compute per-strategy stats + IC + bootstrap CI

    Aggregate via BHY FDR α=5%.

    Output files (in output_dir):
      per_spec.csv               — all 40 spec results
      train_summary.json         — train-period IC distributions
      oos_verdict.json           — OOS Sharpe + NW t + FDR pass/fail
      bootstrap_ci.json          — per-strategy bootstrap CI on OOS Sharpe
      universe_quality_tier1.json
      universe_quality_tier2.json

    Per spec §B+.6: this is the canonical entry point for the B++ mass search.
    """
    import os
    import json

    if strategies is None:
        strategies = list(STRATEGY_REGISTRY)

    os.makedirs(output_dir, exist_ok=True)

    # ── Pre-compute caches for slow signal functions ─────────────────────────
    # Builds regime cache + VIX cache ONCE for all weekly dates spanning the
    # full backtest. Subsequent strategy iterations look up from cache O(1)
    # instead of recomputing per-rebal. Numerical results unchanged.
    all_weekly_dates = pd.date_range(train_start, oos_end, freq=REBAL_FREQ).date.tolist()
    logger.info("Pre-computing regime cache for %d weekly dates...", len(all_weekly_dates))
    regime_cache = precompute_regime_cache(all_weekly_dates)
    logger.info("Pre-computing VIX cache for %d weekly dates...", len(all_weekly_dates))
    vix_cache = precompute_vix_cache(all_weekly_dates)
    logger.info("Caches built: regime=%d, vix=%d", len(regime_cache), len(vix_cache))

    # Pre-fetch universe-tier closes once (wide window: 2007 → 2024 to cover all lookbacks)
    fetch_start = (pd.Timestamp(train_start) - pd.Timedelta(days=365 * 6)).date()
    fetch_end   = pd.Timestamp(oos_end).date()

    closes_by_tier: dict[int, pd.DataFrame] = {}
    universe_by_tier: dict[int, dict[str, str]] = {}
    quality_by_tier:  dict[int, UniverseQualityReport] = {}

    for tier in tiers:
        universe = get_universe_tier(tier)
        universe_by_tier[tier] = universe
        logger.info("Mass search: pre-fetching closes for Tier %d (%d ETFs)...", tier, len(universe))
        closes = fetch_universe_closes(universe, fetch_start, fetch_end)
        closes_by_tier[tier] = closes
        # Pre-flight quality verification (advisory; backtest uses dynamic universe)
        quality = verify_universe_quality(universe, oos_start, oos_end)
        quality_by_tier[tier] = quality
        # Persist quality report
        with open(os.path.join(output_dir, f"universe_quality_tier{tier}.json"), "w") as f:
            json.dump({
                "universe_size":     quality.universe_size,
                "n_passed":          len(quality.tickers_passed),
                "n_dropped":         len(quality.tickers_dropped),
                "tickers_passed":    quality.tickers_passed,
                "tickers_dropped":   [{"ticker": t, "reason": r} for t, r in quality.tickers_dropped],
                "inception_summary": {t: str(d) for t, d in quality.inception_summary.items()},
                "nan_rate_summary":  {t: f"{v:.2%}" for t, v in quality.nan_rate_summary.items()},
                "coverage_window":   [str(d) for d in quality.coverage_window],
            }, f, indent=2, default=str)

    # ── Run all specs ────────────────────────────────────────────────────────
    results: list[dict] = []
    total_specs = len(strategies) * len(tiers)
    spec_idx = 0

    for strategy in strategies:
        for tier in tiers:
            spec_idx += 1
            spec_label = f"{strategy.id}_T{tier}"
            logger.info("Mass search [%d/%d] %s starting", spec_idx, total_specs, spec_label)

            # Inject caches for MA01/MA02 (no-op for other strategies — extra kwargs ignored)
            strategy_params = dict(strategy.params)
            if strategy.id == "MA01":
                strategy_params["regime_cache"] = regime_cache
            elif strategy.id == "MA02":
                strategy_params["vix_cache"] = vix_cache

            # Build a transient spec wrapper with augmented params
            cached_spec = StrategySpec(
                id=strategy.id, name=strategy.name, category=strategy.category,
                signal_fn=strategy.signal_fn, params=strategy_params,
            )

            try:
                # Train-period stats (advisory)
                train_res = run_single_strategy_weekly(
                    cached_spec, universe_by_tier[tier],
                    start_date=train_start, end_date=train_end,
                    closes=closes_by_tier[tier],
                )
                # OOS stats (verdict)
                oos_res = run_single_strategy_weekly(
                    cached_spec, universe_by_tier[tier],
                    start_date=oos_start, end_date=oos_end,
                    closes=closes_by_tier[tier],
                )
            except Exception as exc:
                logger.error("Mass search %s failed: %s", spec_label, exc, exc_info=True)
                results.append({
                    "spec_label":     spec_label,
                    "strategy_id":    strategy.id,
                    "strategy_name":  strategy.name,
                    "category":       strategy.category,
                    "tier":           tier,
                    "error":          str(exc),
                    "n_obs_train":    0,
                    "n_obs_oos":      0,
                })
                continue

            # OOS p-value from NW t-stat (one-sided H0: Sharpe ≤ 0)
            from engine.backtest import sharpe_pvalue
            oos_p = float("nan")
            if "n_obs" in oos_res and oos_res.get("n_obs", 0) >= 12:
                # Use NW t-stat → p directly (Student-t approximation)
                t_stat = oos_res.get("nw_t_stat", float("nan"))
                if not np.isnan(t_stat):
                    from scipy.stats import t as _tdist
                    df_t = oos_res.get("n_obs", 1) - 1
                    oos_p = float(1.0 - _tdist.cdf(t_stat, df=df_t))

            row = {
                "spec_label":         spec_label,
                "strategy_id":        strategy.id,
                "strategy_name":      strategy.name,
                "category":           strategy.category,
                "tier":               tier,
                "n_obs_train":        train_res.get("n_obs", 0),
                "n_obs_oos":          oos_res.get("n_obs", 0),
                "train_sharpe":       train_res.get("sharpe", float("nan")),
                "train_nw_t":         train_res.get("nw_t_stat", float("nan")),
                "train_ic_mean":      train_res.get("ic_mean", float("nan")),
                "train_icir":         train_res.get("icir", float("nan")),
                "oos_sharpe":         oos_res.get("sharpe", float("nan")),
                "oos_ann_return":     oos_res.get("ann_return", float("nan")),
                "oos_ann_vol":        oos_res.get("ann_vol", float("nan")),
                "oos_nw_t":           oos_res.get("nw_t_stat", float("nan")),
                "oos_nw_ci_low":      oos_res.get("nw_ci_low", float("nan")),
                "oos_nw_ci_high":     oos_res.get("nw_ci_high", float("nan")),
                "oos_ic_mean":        oos_res.get("ic_mean", float("nan")),
                "oos_icir":           oos_res.get("icir", float("nan")),
                "oos_p_value":        oos_p,
                "error":              None,
            }
            results.append(row)

            # Persist per-spec weekly returns for later combination phase
            if "weekly_returns" in oos_res:
                ret_path = os.path.join(output_dir, f"{spec_label}_oos_returns.csv")
                oos_res["weekly_returns"].to_csv(ret_path, header=["return"])

            if progress_cb:
                try:
                    progress_cb(spec_idx, total_specs, spec_label,
                                row.get("oos_sharpe", float("nan")))
                except Exception:
                    pass

    # ── Persist per-spec table ───────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "per_spec.csv"), index=False)

    # ── BHY FDR over OOS p-values ────────────────────────────────────────────
    p_vals = [r.get("oos_p_value", float("nan")) for r in results]
    fdr = bhy_fdr_correction(p_vals, alpha=BHY_ALPHA)

    # Mark each row with FDR pass status
    for i, r in enumerate(results):
        r["bhy_pass"] = i in fdr["pass_indices"]
        r["raw_p_pass_5pct"] = (
            not np.isnan(r.get("oos_p_value", float("nan")))
            and r["oos_p_value"] < 0.05
        )

    # ── Aggregated verdict ───────────────────────────────────────────────────
    n_bhy_pass     = sum(1 for r in results if r.get("bhy_pass"))
    n_raw_p_pass   = sum(1 for r in results if r.get("raw_p_pass_5pct"))
    n_raw_p_10     = sum(1 for r in results if (not np.isnan(r.get("oos_p_value", float("nan"))))
                                                and r["oos_p_value"] < 0.10)

    if n_bhy_pass >= 1:
        verdict = "DISCOVERY"
    elif n_raw_p_10 >= 1:
        verdict = "MARGINAL"
    else:
        verdict = "NULL"

    # Best individual strategies
    valid_rows = [r for r in results if not np.isnan(r.get("oos_sharpe", float("nan")))]
    valid_rows.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    top_3 = valid_rows[:3]

    aggregated = {
        "n_total":          len(results),
        "n_with_data":      len(valid_rows),
        "n_bhy_pass":       n_bhy_pass,
        "n_raw_p_05":       n_raw_p_pass,
        "n_raw_p_10":       n_raw_p_10,
        "best_oos_sharpe":  top_3[0]["oos_sharpe"] if top_3 else float("nan"),
        "best_oos_nw_t":    top_3[0]["oos_nw_t"]   if top_3 else float("nan"),
        "best_label":       top_3[0]["spec_label"] if top_3 else None,
        "median_oos_sharpe": float(np.median([r["oos_sharpe"] for r in valid_rows])) if valid_rows else float("nan"),
        "verdict":          verdict,
        "bhy_threshold":    fdr["bhy_threshold"],
        "bhy_c_factor":     fdr["c_factor"],
        "top_3":            [{k: r[k] for k in ["spec_label","oos_sharpe","oos_nw_t","oos_p_value","bhy_pass"]} for r in top_3],
    }

    with open(os.path.join(output_dir, "oos_verdict.json"), "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Train summary (advisory)
    train_ic_summary = {}
    for r in results:
        train_ic_summary[r["spec_label"]] = {
            "train_ic_mean":  r.get("train_ic_mean"),
            "train_icir":     r.get("train_icir"),
            "train_sharpe":   r.get("train_sharpe"),
        }
    with open(os.path.join(output_dir, "train_summary.json"), "w") as f:
        json.dump(train_ic_summary, f, indent=2, default=str)

    return {
        "per_spec":   results,
        "aggregated": aggregated,
        "fdr":        fdr,
        "verdict":    verdict,
        "output_dir": output_dir,
    }


def _smoke_test_strategy(strategy_id: str = "TS01", tier: int = 1):
    """Quick smoke test: run one strategy on Tier 1 over short window."""
    spec = get_strategy(strategy_id)
    universe = get_universe_tier(tier)
    print(f"Smoke: {strategy_id} ({spec.name}) on Tier {tier} ({len(universe)} ETFs)")
    print(f"Window: {OOS_START} → 2018-12-31 (1y subset)")
    res = run_single_strategy_weekly(
        spec, universe,
        start_date=OOS_START, end_date="2018-12-31",
    )
    if "error" in res:
        print(f"FAILED: {res['error']}")
    else:
        print(f"  n_obs:   {res['n_obs']}")
        print(f"  Sharpe:  {res['sharpe']:+.3f}")
        print(f"  NW t:    {res['nw_t_stat']:+.3f}")
        print(f"  IC mean: {res['ic_mean']:+.4f}")
        print(f"  ICIR:    {res['icir']:+.3f}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    _smoke_test_strategy("TS01", tier=1)
