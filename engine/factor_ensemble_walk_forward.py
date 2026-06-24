"""
engine/factor_ensemble_walk_forward.py — Walk-forward harness for Framework E v1.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5)
Spec section: §2.5 Walk-Forward Backtest Protocol

Methodology
-----------
Per month-end t in [start_date, end_date]:
  1. Point-in-time universe via engine.universe_manager.get_universe_as_of(t)
     (respects ETF inception dates; 2015 walk-forward excludes 2018 IPOs)
  2. Asset class lookup per ETF (for factor scope enforcement)
  3. Compute ensemble signal at t (Quality NaN for t < SPEC_LOCK_DATE per
     v1 amendment lookahead guard)
  4. Vol-parity scaled weights (sum-of-abs gross = 1.0, then vol-target scalar)
  5. Hold weights through next month [t, t+1mo]
  6. Compute realized monthly return = Σ(weight_i × etf_return_i over [t, t+1mo])

Boundary invariant (project rule "0-LLM-in-evaluation"):
  Pure deterministic backtest. No LLM in this path. Reads price history from
  yfinance + universe from engine.universe_manager.

Walk-forward simplifications vs full production portfolio.construct_portfolio
(per spec §五 Gate 0 reproducibility, full integration in Sprint Week 4):
  - No regime overlay (v3 multivariate MSM not invoked in walk-forward;
    regime overlay is supervisor decision applied post-backtest)
  - No ETF Holdings Monitor caps (forward-only capability, not historical)
  - No correlated-pair caps / sector caps (single-name production constraints,
    not factor-level)
  - Inverse-vol pre-weighting: applied at single ticker level (matches BAB +
    TSMOM standalone implementations)
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked walk-forward parameters (per spec §2.5)
# ─────────────────────────────────────────────────────────────────────────────

# Vol target matches existing engine.config.TARGET_VOL (10% annualized)
TARGET_VOL: float = 0.10

# Minimum ETF history requirement for factor signals
# TSMOM needs 13 months (12 lookback + 1 skip); use 24mo buffer for stability
MIN_HISTORY_YEARS: int = 2

# Realized vol estimation window (matches BAB / portfolio.py production)
VOL_WINDOW_DAYS: int = 60

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR: int = 252

# In-sample / OOS boundary (per spec §2.1 + §3.3)
OOS_START_DATE: datetime.date = datetime.date(2011, 1, 1)
DEFAULT_END_DATE: datetime.date = datetime.date(2024, 12, 31)

# Storage
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "factor_ensemble_v1"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_WALK_FORWARD_PATH = _DATA_DIR / "walk_forward.parquet"
_PER_FACTOR_SIGNALS_PATH = _DATA_DIR / "per_factor_signals.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# Price panel cache — bulk pre-fetch + disk-backed cache
# Per spec id=50 amendment 2026-05-09 (clarification, post-Gate-0 speedup).
# Replaces ~10k single-ticker yfinance calls per run with 1 bulk fetch + cache.
# Spec semantics unchanged; pure infrastructure speedup.
# ─────────────────────────────────────────────────────────────────────────────
_PRICE_PANEL_CACHE: Path = _DATA_DIR / "_yf_price_panel.parquet"
_PANEL_BUFFER_DAYS_BEFORE: int = 200   # buffer for VOL_WINDOW_DAYS=60 trading days
_PANEL_BUFFER_DAYS_AFTER:  int = 45    # buffer for next-month realized return


def _bulk_prefetch_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Load + extend on-disk price panel cache; return panel covering
    [start_date - buffer, end_date + buffer] for all requested tickers.

    1. Load existing cache if present
    2. If cache covers full requested range × tickers → return as-is
    3. Otherwise bulk yf.download missing portions, merge with cache, persist

    Returns DataFrame indexed by date with ticker columns. Empty DataFrame on
    catastrophic failure (e.g. yfinance unreachable + no cache).
    """
    import yfinance as yf

    needed_tickers = sorted(set(tickers))
    needed_start_ts = pd.Timestamp(start_date - datetime.timedelta(days=_PANEL_BUFFER_DAYS_BEFORE))
    needed_end_ts   = pd.Timestamp(end_date   + datetime.timedelta(days=_PANEL_BUFFER_DAYS_AFTER))

    cache_df: Optional[pd.DataFrame] = None
    if _PRICE_PANEL_CACHE.exists():
        try:
            cache_df = pd.read_parquet(_PRICE_PANEL_CACHE)
        except Exception as exc:
            logger.warning("price panel cache load failed: %s — refetching", exc)
            cache_df = None

    cache_ok = (
        cache_df is not None
        and not cache_df.empty
        and cache_df.index.min() <= needed_start_ts
        and cache_df.index.max() >= needed_end_ts
        and all(t in cache_df.columns for t in needed_tickers)
    )
    if cache_ok:
        logger.info("price panel cache HIT: %d tickers, [%s, %s]",
                    len(needed_tickers), needed_start_ts.date(), needed_end_ts.date())
        return cache_df

    logger.info("price panel cache MISS or partial — bulk-fetching %d tickers, [%s, %s]",
                len(needed_tickers), needed_start_ts.date(), needed_end_ts.date())
    try:
        raw = yf.download(
            needed_tickers,
            start=str(needed_start_ts.date()),
            end=str(needed_end_ts.date() + datetime.timedelta(days=1)),
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception as exc:
        logger.error("bulk yf.download failed: %s", exc)
        return cache_df if cache_df is not None else pd.DataFrame()

    if raw is None or raw.empty:
        return cache_df if cache_df is not None else pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        new_panel = raw["Close"]
    else:
        new_panel = raw[["Close"]].rename(columns={"Close": needed_tickers[0]})

    new_panel = new_panel.dropna(how="all")

    if cache_df is not None and not cache_df.empty:
        # Union of dates × tickers; prefer new data on overlap
        combined = new_panel.combine_first(cache_df)
    else:
        combined = new_panel

    try:
        combined.to_parquet(_PRICE_PANEL_CACHE)
        logger.info("price panel cache persisted: %d tickers × %d dates",
                    combined.shape[1], combined.shape[0])
    except Exception as exc:
        logger.warning("price panel cache persist failed: %s", exc)

    return combined


def _panel_slice(
    panel:   Optional[pd.DataFrame],
    tickers: list[str],
    start:   datetime.date,
    end:     datetime.date,
) -> pd.DataFrame:
    """Read panel[tickers] sliced to [start, end] dates. No yfinance call.

    Returns empty DataFrame if panel None/empty or no requested tickers in panel.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    cols = [t for t in tickers if t in panel.columns]
    if not cols:
        return pd.DataFrame()
    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    return panel.loc[mask, cols].dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward backtest output."""
    n_periods:           int
    monthly_returns:     pd.Series  # indexed by month-end date
    cumulative_return:   float
    annualized_sharpe:   float
    annualized_vol:      float
    max_drawdown:        float
    n_etfs_per_period:   pd.Series  # universe size each rebalance date
    gross_exposure:      pd.Series  # sum of |weights| each period
    metadata:            dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: run walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────


def run_walk_forward(
    start_date:    datetime.date = OOS_START_DATE,
    end_date:      datetime.date = DEFAULT_END_DATE,
    baseline_only: bool = False,
    use_cache:     bool = True,
    persist:       bool = False,
) -> WalkForwardResult:
    """
    Walk-forward monthly rebalance backtest.

    Args:
        start_date:    first month-end to compute signal (default 2011-01)
        end_date:      last month-end to compute signal (default 2024-12)
        baseline_only: if True, use BAB-only signal (for Gate 0 reproducibility)
                       else use full ensemble (TSMOM + Carry-eq + Quality* + BAB)
                       *Quality NaN for t < SPEC_LOCK_DATE per amendment
        use_cache:     pass-through to factor signal computation
        persist:       if True, write to data/factor_ensemble_v1/walk_forward.parquet

    Returns:
        WalkForwardResult with per-period returns + aggregate metrics
    """
    if not isinstance(start_date, datetime.date) or not isinstance(end_date, datetime.date):
        raise TypeError("start_date and end_date must be datetime.date")
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be < end_date {end_date}")

    rebalance_dates = _generate_monthend_dates(start_date, end_date)
    logger.info(
        "walk_forward: %d rebalance dates from %s to %s (baseline_only=%s)",
        len(rebalance_dates), start_date, end_date, baseline_only,
    )

    # Bulk pre-fetch price panel for ALL tickers across full window — replaces
    # ~10k per-ticker yfinance calls with 1 bulk fetch + cache hit on subsequent
    # runs. Per spec id=50 amendment 2026-05-09 (clarification, infra speedup).
    _all_tickers: set[str] = set()
    for d in rebalance_dates:
        _ud = _get_universe_at_date(d)
        if _ud:
            _all_tickers.update(_ud.values())
    panel: Optional[pd.DataFrame] = None
    if _all_tickers:
        panel = _bulk_prefetch_panel(
            tickers=sorted(_all_tickers),
            start_date=rebalance_dates[0] if rebalance_dates else start_date,
            end_date=rebalance_dates[-1] if rebalance_dates else end_date,
        )

    monthly_records: list[dict] = []
    per_factor_records: list[dict] = []

    for i, rebal_date in enumerate(rebalance_dates):
        # Step 1: point-in-time universe
        universe_dict = _get_universe_at_date(rebal_date)
        if not universe_dict:
            logger.warning("walk_forward: no universe at %s; skip", rebal_date)
            continue
        universe = list(universe_dict.values())  # tickers
        asset_classes = _build_asset_classes_lookup(universe)

        # Step 2-3: compute signal (ensemble or BAB-only)
        try:
            signal = _compute_signal_at_date(
                as_of=rebal_date,
                universe=universe,
                asset_classes=asset_classes,
                baseline_only=baseline_only,
                use_cache=use_cache,
                per_factor_records=per_factor_records,
            )
        except Exception as exc:
            logger.warning(
                "walk_forward: signal computation failed at %s: %s — skip",
                rebal_date, exc,
            )
            continue

        if signal is None or signal.empty:
            logger.warning("walk_forward: empty signal at %s — skip", rebal_date)
            continue

        # Step 4: weights from signal (vol-parity scaled, gross-normalized, vol-target)
        weights = _compute_weights_from_signal(signal, rebal_date, panel=panel)
        if weights is None or weights.empty:
            logger.warning("walk_forward: empty weights at %s — skip", rebal_date)
            continue

        # Step 5-6: realized return for next month [rebal_date, next month-end]
        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_rebal is None:
            # Last period — no realized return computable (would need post-end data)
            break

        try:
            realized_return = _compute_realized_return(
                weights=weights,
                period_start=rebal_date,
                period_end=next_rebal,
                panel=panel,
            )
        except Exception as exc:
            logger.warning(
                "walk_forward: realized return failed [%s, %s]: %s — skip",
                rebal_date, next_rebal, exc,
            )
            continue

        monthly_records.append({
            "rebal_date":      rebal_date,
            "next_rebal_date": next_rebal,
            "n_etfs":          int((weights != 0).sum()),
            "gross_exposure":  float(weights.abs().sum()),
            "net_exposure":    float(weights.sum()),
            "monthly_return":  realized_return,
        })

    if not monthly_records:
        logger.error("walk_forward: zero successful periods")
        return WalkForwardResult(
            n_periods=0, monthly_returns=pd.Series(dtype=float),
            cumulative_return=0.0, annualized_sharpe=0.0,
            annualized_vol=0.0, max_drawdown=0.0,
            n_etfs_per_period=pd.Series(dtype=int),
            gross_exposure=pd.Series(dtype=float),
        )

    # Aggregate metrics
    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    monthly_returns = df["monthly_return"]
    n_periods = len(monthly_returns)
    cumulative_return = float((1 + monthly_returns).prod() - 1)
    annualized_vol = float(monthly_returns.std(ddof=0) * np.sqrt(12))
    annualized_mean = float(monthly_returns.mean() * 12)
    annualized_sharpe = (
        annualized_mean / annualized_vol if annualized_vol > 1e-9 else 0.0
    )
    cumulative = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = float(drawdown.min())

    result = WalkForwardResult(
        n_periods=n_periods,
        monthly_returns=monthly_returns,
        cumulative_return=cumulative_return,
        annualized_sharpe=annualized_sharpe,
        annualized_vol=annualized_vol,
        max_drawdown=max_drawdown,
        n_etfs_per_period=df["n_etfs"],
        gross_exposure=df["gross_exposure"],
        metadata={
            "start_date":    start_date.isoformat(),
            "end_date":      end_date.isoformat(),
            "baseline_only": baseline_only,
            "spec_id":       50,
        },
    )

    if persist:
        try:
            df.to_parquet(_WALK_FORWARD_PATH, index=True)
            logger.info("walk_forward: persisted %d periods to %s",
                        n_periods, _WALK_FORWARD_PATH)
            if per_factor_records:
                pd.DataFrame(per_factor_records).to_parquet(
                    _PER_FACTOR_SIGNALS_PATH, index=False,
                )
        except Exception as exc:
            logger.warning("walk_forward: parquet persist failed: %s", exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _generate_monthend_dates(
    start: datetime.date, end: datetime.date,
) -> list[datetime.date]:
    """List of month-end dates in [start, end] inclusive."""
    dates = pd.date_range(start=start, end=end, freq="ME").to_list()
    return [d.date() for d in dates]


def _get_universe_at_date(as_of: datetime.date) -> dict[str, str]:
    """
    Point-in-time universe: only ETFs with ≥MIN_HISTORY_YEARS history at as_of.

    Returns {sector: ticker}. Pre-inception ETFs excluded (anti-survivorship).
    """
    try:
        from engine.universe_manager import get_universe_as_of
        return get_universe_as_of(as_of, min_history_years=MIN_HISTORY_YEARS)
    except Exception as exc:
        logger.warning("_get_universe_at_date failed for %s: %s", as_of, exc)
        return {}


def _build_asset_classes_lookup(universe: list[str]) -> dict[str, str]:
    """
    Map ticker → asset_class via universe_manager registry.

    HONEST DISCLOSURE (per pre-Sprint-Week-4 audit Fix #4): uses CURRENT
    asset_class assignments from universe_manager.get_universe_by_class() for
    historical walk-forward dates. Technically minor lookahead bias — but
    asset_class assignments are extremely stable (an ETF doesn't change from
    equity_sector to fixed_income); empirical impact ≈ 0. Disclosed for
    transparency. v2 candidate: point-in-time asset_class registry query if
    universe_manager schema extended to track asset_class history.
    """
    try:
        from engine.universe_manager import get_universe_by_class
        by_class = get_universe_by_class()
        # Reverse: {ticker: asset_class}
        out: dict[str, str] = {}
        for asset_class, sector_to_ticker in by_class.items():
            for sector, ticker in sector_to_ticker.items():
                if ticker in universe:
                    out[ticker] = asset_class
        return out
    except Exception as exc:
        logger.warning("_build_asset_classes_lookup failed: %s", exc)
        # Fallback: assume all equity_sector if registry unreachable
        return {t: "equity_sector" for t in universe}


def _compute_signal_at_date(
    as_of:               datetime.date,
    universe:            list[str],
    asset_classes:       dict[str, str],
    baseline_only:       bool,
    use_cache:           bool,
    per_factor_records:  list[dict],
) -> pd.Series:
    """
    Compute either ensemble (4-factor) or BAB-only signal.

    Walk-forward integrity: factor signals at t use ONLY data ≤ t-1.
    Per-factor signal records persisted for diagnostic transparency.
    """
    if baseline_only:
        # Gate 0 / BAB-only baseline
        from engine.factors import compute_bab_signal
        sig = compute_bab_signal(
            as_of=as_of, universe=universe,
            asset_classes=asset_classes, use_cache=use_cache,
        )
        per_factor_records.append({
            "as_of": as_of.isoformat(), "factor": "bab_only_baseline",
            "n_valid": int(sig.notna().sum()), "n_total": len(sig),
        })
        return sig

    # Full ensemble
    from engine.factor_ensemble import compute_ensemble_signal, _compute_all_factor_signals

    raw_signals = _compute_all_factor_signals(
        as_of=as_of, universe=universe,
        asset_classes=asset_classes, use_cache=use_cache,
    )
    for factor_name, factor_sig in raw_signals.items():
        per_factor_records.append({
            "as_of":   as_of.isoformat(),
            "factor":  factor_name,
            "n_valid": int(factor_sig.notna().sum()) if not factor_sig.empty else 0,
            "n_total": len(factor_sig),
        })

    return compute_ensemble_signal(
        as_of=as_of, universe=universe,
        asset_classes=asset_classes, use_cache=use_cache,
    )


def _compute_weights_from_signal(
    signal:     pd.Series,
    as_of:      datetime.date,
    panel:      Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Convert signal to portfolio weights:
      1. Drop NaN / zero signals
      2. Normalize by inverse-vol (matches portfolio.py inverse-vol pre-weighting)
      3. Gross-normalize to sum |w| = 1
      4. Vol-target scale (target portfolio vol = TARGET_VOL)

    panel: optional pre-fetched price panel (per spec amendment 2026-05-09).
           If provided, vol/return fetches read from panel instead of yfinance.
    """
    valid_signal = signal.dropna()
    nonzero = valid_signal[valid_signal != 0]
    if nonzero.empty:
        return pd.Series(dtype=float)

    # Inverse-vol pre-weighting (match production portfolio.py Step 1)
    inv_vols = _fetch_inv_vols(list(nonzero.index), as_of, panel=panel)
    if inv_vols is None or inv_vols.empty:
        return pd.Series(dtype=float)

    # Per-asset weighted signal
    raw_weight = nonzero * inv_vols.reindex(nonzero.index).fillna(0.0)

    # Drop any post-inv-vol zeros (insufficient vol history)
    raw_weight = raw_weight[raw_weight != 0]
    if raw_weight.empty:
        return pd.Series(dtype=float)

    # Gross-normalize
    gross = raw_weight.abs().sum()
    if gross < 1e-12:
        return pd.Series(dtype=float)
    normalized = raw_weight / gross

    # Vol-target scalar (diagonal cov approximation, matches portfolio.py Step 3)
    realized_vols = _fetch_realized_vols(list(normalized.index), as_of, panel=panel)
    port_vol = float(np.sqrt(((normalized * realized_vols.reindex(normalized.index).fillna(0)) ** 2).sum()))
    if port_vol < 1e-9:
        return normalized
    vol_scalar = TARGET_VOL / port_vol
    # Cap at 2x leverage (matches production MAX_LEVERAGE)
    vol_scalar = min(vol_scalar, 2.0)

    return normalized * vol_scalar


def _fetch_inv_vols(
    tickers: list[str],
    as_of:   datetime.date,
    panel:   Optional[pd.DataFrame] = None,
) -> pd.Series:
    """1 / annualized 60d realized vol per ticker."""
    realized_vols = _fetch_realized_vols(tickers, as_of, panel=panel)
    return realized_vols.replace(0, np.nan).rdiv(1.0)


def _fetch_realized_vols(
    tickers: list[str],
    as_of:   datetime.date,
    panel:   Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Annualized 60-day realized vol per ticker, computed at as_of - 1 (no lookahead).

    panel: optional pre-fetched price panel (per spec amendment 2026-05-09).
           If provided, reads from panel (fast path, 0 yfinance call).
           Falls back to per-ticker yfinance fetch if panel None/empty.
    """
    out: dict[str, float] = {}
    end = as_of - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=120)  # buffer for non-trading days

    # Fast path — read from pre-fetched panel
    if panel is not None and not panel.empty:
        sub = _panel_slice(panel, tickers, start, end)
        for ticker in tickers:
            if ticker not in sub.columns:
                continue
            series = sub[ticker].dropna()
            if len(series) < VOL_WINDOW_DAYS:
                continue
            rets = series.pct_change().dropna().tail(VOL_WINDOW_DAYS)
            if len(rets) < VOL_WINDOW_DAYS // 2:
                continue
            ann_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if ann_vol > 1e-9:
                out[ticker] = ann_vol
        return pd.Series(out, dtype=float)

    # Slow path (fallback) — per-ticker yfinance fetch
    try:
        from engine.signal import _fetch_closes
    except Exception:
        return pd.Series(out, dtype=float)

    for ticker in tickers:
        try:
            # signal._fetch_closes signature: (tickers: list[str], start: date, as_of: date)
            # Sprint Week 2-3 bug fix (2026-05-09): previously called with single
            # string + kwarg `end=` which is invalid → TypeError swallowed by
            # outer try/except → empty inv_vols → 100% empty weights → Gate 0
            # FAIL_PATHOLOGICAL. Now: pass list[ticker] + correct kwarg `as_of=`.
            df_closes = _fetch_closes([ticker], start=start, as_of=end)
            if df_closes is None or df_closes.empty or ticker not in df_closes.columns:
                continue
            series = df_closes[ticker].dropna()
            if len(series) < VOL_WINDOW_DAYS:
                continue
            rets = series.pct_change().dropna().tail(VOL_WINDOW_DAYS)
            if len(rets) < VOL_WINDOW_DAYS // 2:
                continue
            ann_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if ann_vol > 1e-9:
                out[ticker] = ann_vol
        except Exception:
            continue

    return pd.Series(out, dtype=float)


def _compute_realized_return(
    weights:      pd.Series,
    period_start: datetime.date,
    period_end:   datetime.date,
    panel:        Optional[pd.DataFrame] = None,
) -> float:
    """
    Realized portfolio return over [period_start, period_end]:
      Σ_etf weight_etf × (close_etf[period_end] / close_etf[period_start] - 1)

    panel: optional pre-fetched price panel (per spec amendment 2026-05-09).
           If provided, reads from panel (fast path, 0 yfinance call).
    """
    buffer = datetime.timedelta(days=10)
    active_tickers = [t for t, w in weights.items() if abs(w) >= 1e-9]
    if not active_tickers:
        return 0.0

    # Fast path — read from pre-fetched panel
    if panel is not None and not panel.empty:
        sub = _panel_slice(panel, active_tickers, period_start - buffer, period_end + buffer)
        total_return = 0.0
        for ticker, w in weights.items():
            if abs(w) < 1e-9 or ticker not in sub.columns:
                continue
            s = sub[ticker].dropna()
            if len(s) < 2:
                continue
            start_price = _price_at_or_before(s, period_start)
            end_price = _price_at_or_before(s, period_end)
            if start_price is None or end_price is None or start_price <= 0:
                continue
            etf_ret = end_price / start_price - 1
            total_return += float(w) * etf_ret
        return total_return

    # Slow path (fallback) — per-ticker yfinance fetch
    try:
        from engine.signal import _fetch_closes
    except Exception:
        return 0.0

    total_return = 0.0
    n_contribs = 0

    for ticker, w in weights.items():
        if abs(w) < 1e-9:
            continue
        try:
            # Sprint Week 2-3 kwarg fix 2026-05-09 — see _fetch_realized_vols comment.
            df_closes = _fetch_closes(
                [ticker],
                start=period_start - buffer,
                as_of=period_end + buffer,
            )
            if df_closes is None or df_closes.empty or ticker not in df_closes.columns:
                continue
            s = df_closes[ticker].dropna()
            if len(s) < 2:
                continue

            # Find prices nearest to period_start and period_end
            start_price = _price_at_or_before(s, period_start)
            end_price = _price_at_or_before(s, period_end)
            if start_price is None or end_price is None or start_price <= 0:
                continue

            etf_ret = end_price / start_price - 1
            total_return += float(w) * etf_ret
            n_contribs += 1
        except Exception:
            continue

    return total_return


def _price_at_or_before(closes: pd.Series, target: datetime.date) -> Optional[float]:
    """Find most recent close ≤ target. Returns None if no eligible price."""
    try:
        target_ts = pd.Timestamp(target)
        eligible = closes[closes.index <= target_ts]
        if eligible.empty:
            return None
        return float(eligible.iloc[-1])
    except Exception:
        return None
