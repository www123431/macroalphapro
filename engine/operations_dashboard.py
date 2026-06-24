"""
Operations Dashboard helpers
============================
Pure data-query layer for the operations dashboard widgets defined in
docs/spec_operations_dashboard.md. Phase 1 covers two decision categories:

  Decision A — should we re-run the backtest now?
    - get_backtest_cadence()      : days since last Tier-A run + next due
    - get_param_drift()            : current module constants vs last run params
    - get_trigger_event_log()      : changed source files since last run

  Decision C — are current holdings within risk limits?
    - get_exposure_snapshot()      : gross / net / cap utilisation
    - get_concentration_rows()     : per-ticker weight vs ticker-specific cap
    - get_pair_concentration()     : same-direction sum across CORR_PAIRS
    - get_turnover_history()       : rolling 12M / 3M / P95 turnover

All functions return dataclasses (or None / empty list) and never raise on
missing data — the UI layer should handle Optional results gracefully.

Academic anchors:
  - Harvey/Liu/Zhu 2016, López de Prado 2018  → backtest cadence + n_trials
  - Hansen 2005                                 → pre-registration
  - Moreira & Muir 2017                         → vol-targeting / leverage
  - Asness/Moskowitz/Pedersen 2013              → factor/sector overlap
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Project root for source-file mtime checks ─────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BacktestCadence:
    last_run_id:           Optional[int]
    last_run_at:           Optional[datetime.datetime]
    last_run_window:       Optional[tuple[str, str]]      # (start_date, end_date)
    days_since:            Optional[int]
    next_quarterly_due:    datetime.date
    days_until_quarterly:  int
    status:                str           # 'green' | 'yellow' | 'red' | 'never'
    note:                  str


@dataclass
class ParamDriftRow:
    name:           str
    current_value:  str          # str so we can render '—' uniformly
    last_run_value: str
    diverged:       bool
    persisted:      bool         # False = not stored on StructuredBacktestRun


@dataclass
class TriggerEvent:
    file_path:        str
    severity:         str         # 'structural' | 'periodic'
    last_modified_at: Optional[datetime.datetime]
    triggered:        bool        # mtime > last_run_at
    days_ago:         Optional[int]


@dataclass
class ExposureSnapshot:
    snapshot_date:  datetime.date
    days_old:       int
    gross:          float
    net:            float
    n_long:         int
    n_short:        int
    gross_cap:      float         # MAX_LEVERAGE
    net_cap_high:   float         # MAX_NET
    net_cap_low:    float         # MIN_NET
    gross_status:   str           # 'green'|'yellow'|'red'
    net_status:     str
    stale:          bool


@dataclass
class ConcentrationRow:
    sector:        str
    ticker:        str
    actual_weight: float
    cap:           float          # ticker-specific or MAX_WEIGHT default
    pct_used:      float          # |w| / cap
    direction:     str            # 'long' | 'short' | 'neutral'
    status:        str            # 'green'|'yellow'|'red'


@dataclass
class PairConcentrationRow:
    ticker_a:       str
    ticker_b:       str
    weight_a:       float
    weight_b:       float
    same_direction: bool
    direction:      str            # 'long' | 'short' | 'neutral'
    combined:       float          # |w_a| + |w_b| if same_direction else 0
    pair_cap:       float
    pct_used:       float
    status:         str            # 'green'|'yellow'|'red'|'inactive'


@dataclass
class TurnoverHistory:
    n_months:           int
    monthly_series:     pd.DataFrame  # cols: month_start, turnover
    rolling_12m_avg:    Optional[float]
    rolling_3m_avg:     Optional[float]
    historical_p95:     Optional[float]
    latest_month:       Optional[datetime.date]
    latest_value:       Optional[float]
    above_p95:          bool
    trend_amplified:    bool          # 3M / 12M > 1.5
    status:             str           # 'green'|'yellow'|'red'|'insufficient'


# ── Internal helpers ──────────────────────────────────────────────────────────

def _next_quarter_start(today: datetime.date) -> datetime.date:
    """First calendar day of the *next* quarter (proxy for re-validation due)."""
    q_starts = [
        datetime.date(today.year,     1,  1),
        datetime.date(today.year,     4,  1),
        datetime.date(today.year,     7,  1),
        datetime.date(today.year,     10, 1),
        datetime.date(today.year + 1, 1,  1),
    ]
    for d in q_starts:
        if d > today:
            return d
    return q_starts[-1]


def _safe_count_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


# ── A1: Backtest cadence ──────────────────────────────────────────────────────

def get_backtest_cadence(today: Optional[datetime.date] = None) -> BacktestCadence:
    """Return cadence info for the most recent StructuredBacktestRun.

    Status thresholds (matched to docs/spec_operations_dashboard.md §3.A1):
      - 'never'  : no run in DB
      - 'red'    : days_until_quarterly < 0 OR days_since > 100
      - 'yellow' : days_since > 75 OR days_until_quarterly <= 14
      - 'green'  : otherwise
    """
    today = today or datetime.date.today()
    next_q = _next_quarter_start(today)
    days_until_q = (next_q - today).days

    last_id      = None
    last_at      = None
    last_window  = None
    days_since   = None

    try:
        from engine.memory import SessionFactory, StructuredBacktestRun
        with SessionFactory() as sess:
            row = (
                sess.query(StructuredBacktestRun)
                    .order_by(StructuredBacktestRun.id.desc())
                    .first()
            )
            if row is not None:
                last_id     = int(row.id)
                last_at     = row.created_at
                last_window = (str(row.start_date or ""), str(row.end_date or ""))
                if last_at is not None:
                    days_since = (today - last_at.date()).days
    except Exception as exc:
        logger.debug("get_backtest_cadence: db read failed: %s", exc)

    if last_id is None:
        return BacktestCadence(
            last_run_id=None, last_run_at=None, last_run_window=None,
            days_since=None, next_quarterly_due=next_q,
            days_until_quarterly=days_until_q, status="never",
            note="No Tier-A run yet — run a 5-year backtest before any decisions",
        )

    # Status thresholds
    if (days_until_q < 0) or (days_since is not None and days_since > 100):
        status = "red"
        note   = "Overdue — re-run Tier-A immediately"
    elif (days_since is not None and days_since > 75) or (days_until_q <= 14):
        status = "yellow"
        note   = "Quarterly refresh due soon"
    else:
        status = "green"
        note   = "On schedule"

    return BacktestCadence(
        last_run_id=last_id, last_run_at=last_at, last_run_window=last_window,
        days_since=days_since, next_quarterly_due=next_q,
        days_until_quarterly=days_until_q, status=status, note=note,
    )


# ── A2: Param drift ───────────────────────────────────────────────────────────

def get_param_drift() -> list[ParamDriftRow]:
    """Compare current module constants against the most recent run's params.

    Three columns are persisted in StructuredBacktestRun:
        lookback_months / skip_months / regime_scale
    Module constants TARGET_VOL / MAX_WEIGHT / REGIME_SCALE / MAX_LEVERAGE are
    *not* persisted in the current schema; we surface them with a 'not
    persisted' flag so reviewers know the comparison is asymmetric.
    """
    from engine.portfolio import (
        TARGET_VOL, MAX_WEIGHT, MAX_LEVERAGE, REGIME_SCALE,
    )

    last_lookback   = None
    last_skip       = None
    last_regime_sc  = None

    try:
        from engine.memory import SessionFactory, StructuredBacktestRun
        with SessionFactory() as sess:
            row = (
                sess.query(StructuredBacktestRun)
                    .order_by(StructuredBacktestRun.id.desc())
                    .first()
            )
            if row is not None:
                last_lookback  = row.lookback_months
                last_skip      = row.skip_months
                last_regime_sc = row.regime_scale
    except Exception as exc:
        logger.debug("get_param_drift: db read failed: %s", exc)

    rows: list[ParamDriftRow] = []

    # Persisted params (Tier-A baseline default values match below)
    def _row(name, current, last_val, default_when_no_run, persisted=True):
        last_str = f"{last_val}" if last_val is not None else "—"
        cur_str  = f"{current}"
        diverged = (
            last_val is not None and
            (str(current) != str(last_val))
        )
        return ParamDriftRow(
            name=name, current_value=cur_str, last_run_value=last_str,
            diverged=diverged, persisted=persisted,
        )

    rows.append(_row("lookback_months", 12,         last_lookback,  12))
    rows.append(_row("skip_months",     1,          last_skip,      1))
    rows.append(_row("regime_scale",    REGIME_SCALE, last_regime_sc, 0.30))
    rows.append(_row("TARGET_VOL",      TARGET_VOL,   None,         0.10, persisted=False))
    rows.append(_row("MAX_WEIGHT",      MAX_WEIGHT,   None,         0.25, persisted=False))
    rows.append(_row("MAX_LEVERAGE",    MAX_LEVERAGE, None,         2.0,  persisted=False))

    return rows


# ── A3: Trigger event log ─────────────────────────────────────────────────────

# Files watched for structural / periodic changes since the last Tier-A run.
# Spec §3.A3 — keep list small and curated; do not glob the whole engine/.
_WATCHED_FILES: list[tuple[str, str]] = [
    ("engine/signal.py",            "structural"),
    ("engine/portfolio.py",         "structural"),
    ("engine/regime.py",            "structural"),
    ("engine/universe_manager.py",  "structural"),
    ("engine/backtest.py",          "structural"),
    ("engine/factor_mad.py",        "periodic"),
]


def get_trigger_event_log(
    today: Optional[datetime.date] = None,
) -> list[TriggerEvent]:
    """Return mtime-vs-last-run-at comparison for watched source files."""
    today = today or datetime.date.today()

    last_run_at = None
    try:
        from engine.memory import SessionFactory, StructuredBacktestRun
        with SessionFactory() as sess:
            row = (
                sess.query(StructuredBacktestRun)
                    .order_by(StructuredBacktestRun.id.desc())
                    .first()
            )
            if row is not None:
                last_run_at = row.created_at
    except Exception as exc:
        logger.debug("get_trigger_event_log: db read failed: %s", exc)

    out: list[TriggerEvent] = []
    for rel_path, severity in _WATCHED_FILES:
        abs_path = os.path.join(_PROJECT_DIR, rel_path)
        if not os.path.exists(abs_path):
            continue
        try:
            mtime_ts = os.path.getmtime(abs_path)
            mtime_dt = datetime.datetime.fromtimestamp(mtime_ts)
        except OSError:
            continue

        if last_run_at is None:
            triggered = False     # no baseline → nothing is 'changed since'
        else:
            triggered = mtime_dt > last_run_at

        days_ago = (today - mtime_dt.date()).days

        out.append(TriggerEvent(
            file_path=rel_path,
            severity=severity,
            last_modified_at=mtime_dt,
            triggered=triggered,
            days_ago=days_ago,
        ))

    return out


# ── C1: Exposure snapshot ─────────────────────────────────────────────────────

def _classify_exposure(
    gross: float, net: float, gross_cap: float,
    net_high: float, net_low: float,
) -> tuple[str, str]:
    """Return (gross_status, net_status) according to spec §4.C1 thresholds."""
    gross_room = gross_cap - gross
    if gross_room < 0:
        g = "red"
    elif gross_room < 0.05:
        g = "red"
    elif gross_room < 0.10:
        g = "yellow"
    else:
        g = "green"

    if net > net_high or net < net_low:
        n = "red"
    elif (net_high - net) < 0.05 or (net - net_low) < 0.05:
        n = "red"
    elif (net_high - net) < 0.10 or (net - net_low) < 0.10:
        n = "yellow"
    else:
        n = "green"

    return g, n


def get_exposure_snapshot(
    today: Optional[datetime.date] = None,
) -> Optional[ExposureSnapshot]:
    """Aggregate latest SimulatedPosition (track='main') into gross/net/n."""
    from engine.portfolio import MAX_LEVERAGE, MAX_NET, MIN_NET
    today = today or datetime.date.today()

    # Per-sector latest snapshot (2026-05-04 fix): global MAX(snapshot_date)
    # collapsed exposure to ~1 row when rebalances touch only some sectors,
    # severely under-reporting gross/net leverage. Use the canonical helper.
    try:
        from engine.portfolio_tracker import get_current_positions
        df = get_current_positions(track="main", include_closed=False)
        if df is None or df.empty:
            return None
        weights = [float(w or 0.0) for w in df["actual_weight"].tolist()]
        try:
            snap_date = max(df["snapshot_date"].tolist())
        except Exception:
            snap_date = today
    except Exception as exc:
        logger.debug("get_exposure_snapshot: db read failed: %s", exc)
        return None

    if not weights:
        return None

    arr   = np.array(weights, dtype=float)
    gross = float(np.sum(np.abs(arr)))
    net   = float(np.sum(arr))
    n_l   = int(np.sum(arr > 1e-6))
    n_s   = int(np.sum(arr < -1e-6))

    g_status, n_status = _classify_exposure(gross, net, MAX_LEVERAGE, MAX_NET, MIN_NET)

    days_old = (today - snap_date).days
    stale    = days_old > 7

    return ExposureSnapshot(
        snapshot_date=snap_date,
        days_old=days_old,
        gross=gross, net=net,
        n_long=n_l, n_short=n_s,
        gross_cap=MAX_LEVERAGE,
        net_cap_high=MAX_NET, net_cap_low=MIN_NET,
        gross_status=g_status, net_status=n_status,
        stale=stale,
    )


# ── C2: Per-ticker concentration ──────────────────────────────────────────────

def get_concentration_rows(top_n: int = 10) -> list[ConcentrationRow]:
    """Per-ticker actual_weight vs ticker-specific cap, sorted by pct_used desc."""
    from engine.portfolio import MAX_WEIGHT
    try:
        from engine.universe_manager import get_max_weight_for_ticker
    except Exception:
        get_max_weight_for_ticker = None

    # Per-sector latest snapshot (2026-05-04 fix): global MAX collapsed to 1 row.
    try:
        from engine.memory import SessionFactory, SimulatedPosition
        from sqlalchemy import func
        import datetime as _dt
        _today = _dt.date.today()
        with SessionFactory() as sess:
            sub = (
                sess.query(
                    SimulatedPosition.sector.label("sector"),
                    func.max(SimulatedPosition.snapshot_date).label("latest_dt"),
                )
                .filter(SimulatedPosition.track == "main",
                        SimulatedPosition.snapshot_date <= _today)
                .group_by(SimulatedPosition.sector)
                .subquery()
            )
            rows = (
                sess.query(SimulatedPosition)
                    .join(sub,
                          (SimulatedPosition.sector == sub.c.sector) &
                          (SimulatedPosition.snapshot_date == sub.c.latest_dt))
                    .filter(SimulatedPosition.track == "main")
                    .all()
            )
            if not rows:
                return []
    except Exception as exc:
        logger.debug("get_concentration_rows: db read failed: %s", exc)
        return []

    out: list[ConcentrationRow] = []
    for r in rows:
        w = float(r.actual_weight or 0.0)
        if abs(w) < 1e-6:
            continue   # skip neutral positions
        ticker = (r.ticker or "").upper()
        cap = (get_max_weight_for_ticker(ticker, MAX_WEIGHT)
               if get_max_weight_for_ticker else MAX_WEIGHT)
        if cap <= 0:
            continue
        pct = abs(w) / cap

        if pct >= 0.95:
            status = "red"
        elif pct >= 0.80:
            status = "yellow"
        else:
            status = "green"

        out.append(ConcentrationRow(
            sector=r.sector or "",
            ticker=ticker,
            actual_weight=w,
            cap=float(cap),
            pct_used=pct,
            direction="long" if w > 0 else "short",
            status=status,
        ))

    out.sort(key=lambda x: x.pct_used, reverse=True)
    return out[:top_n]


# ── C3: Correlated-pair concentration ────────────────────────────────────────

def get_pair_concentration() -> list[PairConcentrationRow]:
    """For each pair in CORR_PAIRS, sum same-direction weights and compare to
    1.5 × MAX_WEIGHT. Inactive pairs (one or both legs missing) flagged."""
    from engine.portfolio import CORR_PAIRS, MAX_WEIGHT

    pair_cap = MAX_WEIGHT * 1.5

    # Per-sector latest snapshot (2026-05-04 fix). Same pattern as concentration.
    try:
        from engine.memory import SessionFactory, SimulatedPosition
        from sqlalchemy import func
        import datetime as _dt
        _today = _dt.date.today()
        with SessionFactory() as sess:
            sub = (
                sess.query(
                    SimulatedPosition.sector.label("sector"),
                    func.max(SimulatedPosition.snapshot_date).label("latest_dt"),
                )
                .filter(SimulatedPosition.track == "main",
                        SimulatedPosition.snapshot_date <= _today)
                .group_by(SimulatedPosition.sector)
                .subquery()
            )
            rows = (
                sess.query(SimulatedPosition)
                    .join(sub,
                          (SimulatedPosition.sector == sub.c.sector) &
                          (SimulatedPosition.snapshot_date == sub.c.latest_dt))
                    .filter(SimulatedPosition.track == "main")
                    .all()
            )
    except Exception as exc:
        logger.debug("get_pair_concentration: db read failed: %s", exc)
        return []

    weight_by_ticker = {
        (r.ticker or "").upper(): float(r.actual_weight or 0.0)
        for r in rows
    }

    out: list[PairConcentrationRow] = []
    for tkr_a, tkr_b in CORR_PAIRS:
        tkr_a, tkr_b = tkr_a.upper(), tkr_b.upper()
        w_a = weight_by_ticker.get(tkr_a)
        w_b = weight_by_ticker.get(tkr_b)
        if w_a is None or w_b is None:
            out.append(PairConcentrationRow(
                ticker_a=tkr_a, ticker_b=tkr_b,
                weight_a=float(w_a or 0.0), weight_b=float(w_b or 0.0),
                same_direction=False, direction="—",
                combined=0.0, pair_cap=pair_cap, pct_used=0.0,
                status="inactive",
            ))
            continue

        same_dir = (w_a * w_b > 1e-12)
        if not same_dir:
            out.append(PairConcentrationRow(
                ticker_a=tkr_a, ticker_b=tkr_b,
                weight_a=w_a, weight_b=w_b,
                same_direction=False, direction="opposite",
                combined=0.0, pair_cap=pair_cap, pct_used=0.0,
                status="green",
            ))
            continue

        combined = abs(w_a) + abs(w_b)
        pct      = combined / pair_cap if pair_cap > 0 else 0.0
        if pct >= 0.95:
            status = "red"
        elif pct >= 0.80:
            status = "yellow"
        else:
            status = "green"

        out.append(PairConcentrationRow(
            ticker_a=tkr_a, ticker_b=tkr_b,
            weight_a=w_a, weight_b=w_b,
            same_direction=True,
            direction="long" if w_a > 0 else "short",
            combined=combined, pair_cap=pair_cap, pct_used=pct,
            status=status,
        ))

    return out


# ── C4: Turnover history ──────────────────────────────────────────────────────

def get_turnover_history(window_months: int = 24) -> TurnoverHistory:
    """Aggregate SimulatedTrade.weight_delta into monthly turnover stats.

    Monthly turnover = sum of |weight_delta| in that calendar month.
    Returns rolling 12M / 3M average and the historical 95th percentile
    over the requested window.
    """
    empty = TurnoverHistory(
        n_months=0,
        monthly_series=pd.DataFrame(columns=["month_start", "turnover"]),
        rolling_12m_avg=None, rolling_3m_avg=None, historical_p95=None,
        latest_month=None, latest_value=None,
        above_p95=False, trend_amplified=False,
        status="insufficient",
    )

    try:
        from engine.memory import SessionFactory, SimulatedTrade
        with SessionFactory() as sess:
            since = datetime.date.today() - datetime.timedelta(days=window_months * 32)
            rows = (
                sess.query(SimulatedTrade.trade_date, SimulatedTrade.weight_delta)
                    .filter(SimulatedTrade.trade_date >= since)
                    .all()
            )
    except Exception as exc:
        logger.debug("get_turnover_history: db read failed: %s", exc)
        return empty

    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=["trade_date", "weight_delta"])
    df["trade_date"]  = pd.to_datetime(df["trade_date"])
    df["month_start"] = df["trade_date"].dt.to_period("M").dt.to_timestamp()
    df["abs_delta"]   = df["weight_delta"].abs().astype(float)

    monthly = (
        df.groupby("month_start")["abs_delta"].sum()
          .rename("turnover").reset_index()
          .sort_values("month_start")
    )
    n = len(monthly)
    if n < 6:
        return TurnoverHistory(
            n_months=n,
            monthly_series=monthly,
            rolling_12m_avg=None, rolling_3m_avg=None, historical_p95=None,
            latest_month=monthly.iloc[-1]["month_start"].date() if n > 0 else None,
            latest_value=float(monthly.iloc[-1]["turnover"]) if n > 0 else None,
            above_p95=False, trend_amplified=False,
            status="insufficient",
        )

    last_12 = monthly["turnover"].tail(12)
    last_3  = monthly["turnover"].tail(3)
    p95     = float(np.percentile(monthly["turnover"], 95))
    latest_v = float(monthly.iloc[-1]["turnover"])
    latest_m = monthly.iloc[-1]["month_start"].date()

    avg_12 = float(last_12.mean()) if len(last_12) else None
    avg_3  = float(last_3.mean())  if len(last_3)  else None

    above_p95 = latest_v > p95
    trend_amp = (
        avg_12 is not None and avg_3 is not None and
        avg_12 > 1e-9 and (avg_3 / avg_12) > 1.5
    )

    if above_p95 and trend_amp:
        status = "red"
    elif above_p95:
        status = "yellow"
    else:
        status = "green"

    return TurnoverHistory(
        n_months=n,
        monthly_series=monthly,
        rolling_12m_avg=avg_12, rolling_3m_avg=avg_3, historical_p95=p95,
        latest_month=latest_m, latest_value=latest_v,
        above_p95=above_p95, trend_amplified=trend_amp,
        status=status,
    )
