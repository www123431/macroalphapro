"""
engine/forensic/anomaly_detector.py — Cohen-Polk-Vuolteenaho-style outlier flag.

Tier-1 audit #4 follow-up / Forensic redesign Phase 1 (2026-05-14).

Purpose
-------
Replace "user types ticker + date + 7 fields" entry to Forensic News with
an event-driven trigger. Scans paper-trade rows and flags daily returns
beyond expected distribution as candidates for forensic investigation.

Two complementary detectors:

  detect_outlier_strategy_days(lookback_days, z_threshold)
    Operational entry — fires daily on PaperTradeStrategyLog rows.
    Uses per-strategy daily distribution (µ, σ) derived from Sprint B
    2014-2023 replay (data/portfolio_replay/v1_combined_replay_verdict.json).
    Already-actionable today; trade-horizon-realized version below requires
    waiting for horizons to elapse.

  detect_outlier_trades_horizon_complete(...)
    Academic entry — fires per-trade once trade.date + expected_horizon_days
    has elapsed. Compares horizon-realized return to per-strategy
    E[return | horizon] distribution. Currently empty (forward day 1).

References
----------
  - Cohen-Polk-Vuolteenaho 2003 "The Value Spread" — z-score abnormal-
    return diagnostic for outlier flagging.
  - Brinson-Hood-Beebower 1986 — attribution framework (residual decomp
    lives in engine.forensic.residual_attribution, downstream).
  - Lou-Polk-Skouras 2019 "A tug of war" — trade-level signal decay
    studies (consumer of horizon-complete outliers).
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy daily distribution — Sprint B 2014-2023 replay
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth — derived from per_strategy_metrics ann_ret + ann_vol
# in v1_combined_replay_verdict.json. Hardcoded here for fast lookup; if the
# replay verdict is regenerated with materially different distributions, this
# table should be refreshed (low-frequency event).
#
# Daily mean   = ann_ret / 252       (additive return per trading day)
# Daily stdev  = ann_vol / sqrt(252) (√t-scaling under iid normal approx)
#
# These are CONSERVATIVE priors — in-sample 2014-2023 Sharpe is higher than
# expected forward (Stein-shrinkage, regime risk). Forward stdev likely
# higher than in-sample → z-threshold of 2.0 should produce roughly 5% hit
# rate ex-ante, lower ex-post (favoring fewer false alarms, more selective
# forensic queue).
_STRATEGY_DAILY_DISTRIBUTION: dict[str, dict[str, float]] = {
    "K1_BAB":     {"mu_daily": 0.000154, "sigma_daily": 0.003206},
    "D_PEAD":     {"mu_daily": 0.000379, "sigma_daily": 0.006451},
    "PATH_N":     {"mu_daily": 0.000539, "sigma_daily": 0.011748},
    "CTA_PQTIX":  {"mu_daily": 0.000180, "sigma_daily": 0.006633},
}


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OutlierStrategyDay:
    """One flagged (strategy, date) row from PaperTradeStrategyLog."""
    date:           datetime.date
    strategy_name:  str
    realized_net:   float          # daily_net_return (preferred) or daily_gross_return fallback
    realized_gross: Optional[float]
    tc_drag:        Optional[float]
    expected_mu:    float          # daily prior from Sprint B
    sigma:          float          # daily stdev from Sprint B
    z_score:        float          # (realized - mu) / sigma — signed
    abs_z:          float          # |z|
    direction:      str            # 'positive' | 'negative'
    n_positions:    int
    is_rebalance:   bool
    sleeve_id:      str
    return_field:   str            # 'daily_net_return' or 'daily_gross_return'


@dataclass(frozen=True)
class OutlierTradeHorizon:
    """One flagged trade after its horizon has elapsed."""
    trade_id:               str
    open_date:              datetime.date
    close_date:             datetime.date
    strategy_name:          str
    ticker:                 str
    side:                   str
    weight:                 float
    horizon_days:           int
    realized_return:        float
    expected_mu_at_horizon: float
    sigma_at_horizon:       float
    z_score:                float
    abs_z:                  float
    direction:              str


@dataclass(frozen=True)
class OutlierQueue:
    """Ranked output of an anomaly detection scan."""
    as_of:                  datetime.date
    lookback_days:          int
    z_threshold:            float
    strategy_day_outliers:  list[OutlierStrategyDay] = field(default_factory=list)
    trade_horizon_outliers: list[OutlierTradeHorizon] = field(default_factory=list)
    n_strategy_days_scanned: int = 0
    n_trades_scanned:        int = 0
    notes:                   list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy-day detector (operational; fires daily)
# ─────────────────────────────────────────────────────────────────────────────
def detect_outlier_strategy_days(
    lookback_days: int   = 30,
    z_threshold:   float = 2.0,
    as_of:         Optional[datetime.date] = None,
    session:       Optional[object] = None,
) -> tuple[list[OutlierStrategyDay], int]:
    """Scan PaperTradeStrategyLog for outlier daily returns.

    Returns (outliers_ranked_by_abs_z_desc, n_rows_scanned).

    Uses daily_net_return if populated (post Step 6 backfill), else falls
    back to daily_gross_return. Rows where neither is populated are skipped
    silently (early forward window before fill_daily_returns runs).
    """
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()

    as_of = as_of or datetime.date.today()
    cutoff = as_of - datetime.timedelta(days=lookback_days)

    outliers: list[OutlierStrategyDay] = []
    n_scanned = 0
    try:
        rows = (
            sess.query(PaperTradeStrategyLog)
                .filter(PaperTradeStrategyLog.date >= cutoff,
                        PaperTradeStrategyLog.date <= as_of)
                .all()
        )
        for r in rows:
            # Prefer net (post-TC) for outlier detection. Net is the true
            # P&L the supervisor would discuss in a forensic meeting.
            if r.daily_net_return is not None:
                realized = float(r.daily_net_return)
                used_field = "daily_net_return"
            elif r.daily_gross_return is not None:
                realized = float(r.daily_gross_return)
                used_field = "daily_gross_return"
            else:
                continue
            dist = _STRATEGY_DAILY_DISTRIBUTION.get(r.strategy_name)
            if dist is None or dist["sigma_daily"] <= 0:
                continue
            n_scanned += 1
            mu = dist["mu_daily"]
            sigma = dist["sigma_daily"]
            z = (realized - mu) / sigma
            if abs(z) < z_threshold:
                continue
            outliers.append(OutlierStrategyDay(
                date=r.date,
                strategy_name=r.strategy_name,
                realized_net=realized,
                realized_gross=(float(r.daily_gross_return)
                                if r.daily_gross_return is not None else None),
                tc_drag=(float(r.tc_drag_today)
                         if r.tc_drag_today is not None else None),
                expected_mu=mu,
                sigma=sigma,
                z_score=float(z),
                abs_z=float(abs(z)),
                direction="positive" if z > 0 else "negative",
                n_positions=int(r.n_positions or 0),
                is_rebalance=bool(r.is_rebalance_day),
                sleeve_id=str(r.sleeve_id or ""),
                return_field=used_field,
            ))
    finally:
        if own_session:
            sess.close()

    outliers.sort(key=lambda o: o.abs_z, reverse=True)
    return outliers, n_scanned


# ─────────────────────────────────────────────────────────────────────────────
# Trade-horizon detector (academic; fires when horizons complete)
# ─────────────────────────────────────────────────────────────────────────────
def detect_outlier_trades_horizon_complete(
    lookback_days: int   = 90,
    z_threshold:   float = 2.0,
    as_of:         Optional[datetime.date] = None,
    session:       Optional[object] = None,
) -> tuple[list[OutlierTradeHorizon], int]:
    """Per-trade outlier detection once horizon has elapsed.

    For each PaperTradeTradeLog row where trade.date + expected_horizon_days
    <= as_of:
      - Fetch close prices for ticker at trade.date and trade.date + horizon
      - Compute realized_return = (close_horizon - close_open) / close_open
      - Compare to E[return | horizon] = strategy_daily_mu × horizon,
        σ_horizon = strategy_daily_sigma × sqrt(horizon)
      - Flag if |z| > z_threshold

    Returns empty list until at least one trade has its horizon completed.
    Currently (2026-05-14 forward day 1) → always empty.

    Implementation note: batch-fetches yfinance to keep call count bounded.
    """
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeTradeLog

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()

    as_of = as_of or datetime.date.today()
    cutoff = as_of - datetime.timedelta(days=lookback_days)

    eligible_rows: list = []
    n_scanned = 0
    try:
        rows = (
            sess.query(PaperTradeTradeLog)
                .filter(PaperTradeTradeLog.date >= cutoff,
                        PaperTradeTradeLog.date <= as_of)
                .all()
        )
        for r in rows:
            horizon = int(r.expected_horizon_days or 0)
            if horizon <= 0:
                continue
            close_date = r.date + datetime.timedelta(days=horizon)
            if close_date > as_of:
                continue   # horizon not complete yet
            eligible_rows.append((r, close_date))
            n_scanned += 1
    finally:
        if own_session:
            sess.close()

    if not eligible_rows:
        return [], n_scanned

    # Batch-fetch close prices for each unique ticker over the span needed
    import yfinance as _yf
    import pandas as _pd

    tickers = sorted({r.ticker for r, _ in eligible_rows})
    earliest = min(r.date for r, _ in eligible_rows) - datetime.timedelta(days=2)
    latest   = max(cd for _, cd in eligible_rows) + datetime.timedelta(days=2)
    try:
        data = _yf.download(
            tickers, start=earliest.isoformat(), end=latest.isoformat(),
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        close = data["Close"] if "Close" in data.columns else data
        if isinstance(close, _pd.Series):
            close = close.to_frame(name=tickers[0])
        close.index = _pd.to_datetime(close.index).date
    except Exception as exc:
        logger.warning("anomaly_detector horizon yfinance fetch failed: %s", exc)
        return [], n_scanned

    outliers: list[OutlierTradeHorizon] = []
    for r, close_date in eligible_rows:
        if r.ticker not in close.columns:
            continue
        avail_dates = sorted(d for d in close.index)
        open_d  = max((d for d in avail_dates if d <= r.date),       default=None)
        close_d = max((d for d in avail_dates if d <= close_date),   default=None)
        if open_d is None or close_d is None or open_d >= close_d:
            continue
        try:
            p_open  = float(close.at[open_d,  r.ticker])
            p_close = float(close.at[close_d, r.ticker])
        except Exception:
            continue
        if not (p_open > 0 and not math.isnan(p_open) and not math.isnan(p_close)):
            continue
        realized = (p_close - p_open) / p_open
        # Apply trade side (short → negate)
        if (r.side or "").lower() == "short":
            realized = -realized
        # Scale by signed weight intensity? No — z-score is on per-trade
        # return; portfolio impact is a separate calc.
        dist = _STRATEGY_DAILY_DISTRIBUTION.get(r.strategy_name)
        if not dist or dist["sigma_daily"] <= 0:
            continue
        horizon = int(r.expected_horizon_days)
        mu_h    = dist["mu_daily"] * horizon
        sigma_h = dist["sigma_daily"] * math.sqrt(horizon)
        if sigma_h <= 0:
            continue
        z = (realized - mu_h) / sigma_h
        if abs(z) < z_threshold:
            continue
        outliers.append(OutlierTradeHorizon(
            trade_id=r.trade_id,
            open_date=r.date,
            close_date=close_date,
            strategy_name=r.strategy_name,
            ticker=r.ticker,
            side=r.side or "long",
            weight=float(r.weight or 0.0),
            horizon_days=horizon,
            realized_return=float(realized),
            expected_mu_at_horizon=float(mu_h),
            sigma_at_horizon=float(sigma_h),
            z_score=float(z),
            abs_z=float(abs(z)),
            direction="positive" if z > 0 else "negative",
        ))

    outliers.sort(key=lambda o: o.abs_z, reverse=True)
    return outliers, n_scanned


# ─────────────────────────────────────────────────────────────────────────────
# Combined queue (operational entry for UI)
# ─────────────────────────────────────────────────────────────────────────────
def build_outlier_queue(
    lookback_days: int   = 30,
    z_threshold:   float = 2.0,
    as_of:         Optional[datetime.date] = None,
) -> OutlierQueue:
    """High-level entry — both detectors + provenance metadata.

    Note: trade-horizon detector returns empty until first trade reaches
    its horizon (calendar gate ~2026-06 for shortest-horizon Path N 5d
    trades, ~2026-08 for D-PEAD 60d).
    """
    as_of = as_of or datetime.date.today()
    sd_outliers, sd_scanned = detect_outlier_strategy_days(
        lookback_days=lookback_days, z_threshold=z_threshold, as_of=as_of,
    )
    th_outliers, th_scanned = detect_outlier_trades_horizon_complete(
        lookback_days=max(lookback_days, 90), z_threshold=z_threshold, as_of=as_of,
    )
    notes = []
    if not sd_outliers and not th_outliers:
        notes.append(
            f"No outliers at |z|>{z_threshold:.1f} across "
            f"{sd_scanned} strategy-days + {th_scanned} horizon-complete trades."
        )
    if th_scanned == 0:
        notes.append(
            "Trade-horizon detector: 0 horizon-complete trades yet "
            "(forward window too short — fires once first horizons elapse)."
        )
    return OutlierQueue(
        as_of=as_of,
        lookback_days=lookback_days,
        z_threshold=z_threshold,
        strategy_day_outliers=sd_outliers,
        trade_horizon_outliers=th_outliers,
        n_strategy_days_scanned=sd_scanned,
        n_trades_scanned=th_scanned,
        notes=notes,
    )
