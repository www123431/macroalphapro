"""
engine/performance_metrics.py — TWR / MWR / HPR computation engine.

Spec: docs/spec_performance_reporting_v1.md  (sha256[:16] f1c9b693f7a6a6df).

Three methods, three audiences, per GIPS 2020 / Bacon (2019) Ch.2:

  TWR  Time-Weighted Return       Manager skill   GIPS primary
       Modified Dietz with day-weighted denominator (single-period) +
       geometric linking of daily sub-period returns (production)

  MWR  Money-Weighted Return      Investor view   GIPS secondary
       XIRR via scipy.optimize.brentq with [-0.999, 100.0] bracket;
       investor sign convention: outflow (deposit / initial NAV) < 0,
       inflow (withdraw / terminal NAV) > 0.

  HPR  Holding Period Return      Naive baseline  Educational only
       (NAV_end - NAV_start) / NAV_start ignoring cash flows.

Public API:
  compute_modified_dietz_period(nav_start, nav_end, flows, period_start, period_end) -> float
  compute_twr_geometric_link(start, end)                                            -> float
  compute_xirr(cash_flows)                                                          -> float
  compute_hpr(nav_start, nav_end)                                                   -> float
  compute_period_summary(start, end)                                                -> dict
"""
from __future__ import annotations

import datetime
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modified Dietz (single-period TWR — Bacon 2019 Ch.2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_modified_dietz_period(
    nav_start:    float,
    nav_end:      float,
    flows:        list[tuple[datetime.date, float]],
    period_start: datetime.date,
    period_end:   datetime.date,
) -> float:
    """
    Bacon (2019) Ch.2 Modified Dietz single-period return:

        r = (V_end - V_start - sum F_i) / (V_start + sum F_i × w_i)

    where w_i = (T - (t_i - period_start)) / T,  T = days in period.

    Args:
        nav_start: NAV at period start (before any flow on period_start)
        nav_end:   NAV at period end
        flows:     list of (flow_date, amount_usd); amount > 0 = INTO portfolio
        period_start, period_end: bracketing dates inclusive

    Returns:
        Period return as decimal (e.g., 0.0909 for +9.09%).
        Returns 0.0 on degenerate inputs (T <= 0, denom == 0).
    """
    T = (period_end - period_start).days
    if T <= 0:
        return 0.0
    F_total = sum(amt for _, amt in flows)
    weighted_F = 0.0
    for d, amt in flows:
        days_remaining = T - (d - period_start).days
        weight = max(0.0, min(1.0, days_remaining / T))
        weighted_F += amt * weight
    denom = nav_start + weighted_F
    if abs(denom) < 1e-9:
        return 0.0
    return (nav_end - nav_start - F_total) / denom


# ─────────────────────────────────────────────────────────────────────────────
# 2. TWR — geometric linking of daily Modified Dietz
# ─────────────────────────────────────────────────────────────────────────────

def compute_twr_geometric_link(
    start: datetime.date,
    end:   datetime.date,
    *,
    session: Any | None = None,
) -> float:
    """
    Production TWR over [start, end]: geometrically link the
    `daily_modified_dietz` field across PortfolioNavSnapshot rows whose
    snapshot_date falls in (start, end] (exclusive of start, inclusive of end).

    Returns:
        Cumulative TWR as decimal; 0.0 if no snapshots in range.
    """
    from engine.memory import PortfolioNavSnapshot, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        rows = (
            sess.query(PortfolioNavSnapshot)
                .filter(PortfolioNavSnapshot.snapshot_date > start)
                .filter(PortfolioNavSnapshot.snapshot_date <= end)
                .order_by(PortfolioNavSnapshot.snapshot_date.asc())
                .all()
        )
        if not rows:
            return 0.0
        factor = 1.0
        for r in rows:
            r_md = float(r.daily_modified_dietz or 0.0)
            factor *= (1.0 + r_md)
        return factor - 1.0
    finally:
        if own:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. MWR / XIRR
# ─────────────────────────────────────────────────────────────────────────────

def compute_xirr(
    cash_flows: list[tuple[datetime.date, float]],
    *,
    bracket:   tuple[float, float] = (-0.999, 100.0),
    tol:       float = 1e-9,
    max_iter:  int = 200,
) -> float:
    """
    Solve XIRR via scipy.optimize.brentq on bracket [-0.999, 100.0].

    cash_flows uses INVESTOR sign convention:
        amount < 0 = outflow from investor (deposit / initial NAV at t0)
        amount > 0 = inflow to investor (withdraw / terminal NAV at t_end)

    Returns annualized rate as decimal. Raises ValueError if cash flows lack
    a sign change (no IRR exists) or no root in the bracket.
    """
    from scipy.optimize import brentq

    if len(cash_flows) < 2:
        raise ValueError("compute_xirr: need >= 2 cash flows")

    pos_count = sum(1 for _, a in cash_flows if a > 0)
    neg_count = sum(1 for _, a in cash_flows if a < 0)
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            "compute_xirr: cash flows must contain both signs "
            f"(got {pos_count} positive, {neg_count} negative)"
        )

    cf = sorted(cash_flows, key=lambda x: x[0])
    t0 = cf[0][0]

    def npv(r: float) -> float:
        if r <= -1.0:
            return float("inf")
        total = 0.0
        for d, a in cf:
            years = (d - t0).days / 365.0
            total += a / ((1.0 + r) ** years)
        return total

    r_lo, r_hi = bracket
    if npv(r_lo) * npv(r_hi) > 0:
        # Try wider bracket
        r_lo, r_hi = -0.9999, 1000.0
        if npv(r_lo) * npv(r_hi) > 0:
            raise ValueError(
                "compute_xirr: no sign change in [-0.9999, 1000.0] bracket"
            )

    return float(brentq(npv, r_lo, r_hi, xtol=tol, maxiter=max_iter))


# ─────────────────────────────────────────────────────────────────────────────
# 4. HPR (naive, educational)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hpr(nav_start: float, nav_end: float) -> float:
    """
    Naive holding-period return; ignores cash flows. Pedagogical baseline
    that demonstrates cash-flow distortion when shown alongside TWR/MWR.
    """
    if abs(nav_start) < 1e-9:
        return 0.0
    return (nav_end - nav_start) / nav_start


# ─────────────────────────────────────────────────────────────────────────────
# 5. Period summary — aggregates all three methods
# ─────────────────────────────────────────────────────────────────────────────

def compute_period_summary(
    start:   datetime.date,
    end:     datetime.date,
    *,
    session: Any | None = None,
) -> dict:
    """
    Aggregate TWR (single-period Modified Dietz + geometric-linked variant) +
    MWR (XIRR) + HPR + benchmark active return over [start, end].

    Returns dict:
      {
        twr_modified_dietz   : single-period MD using nav_open[start] + nav_close[end] + flows
        twr_geometric_linked : product-link of daily MDs in (start, end]
        mwr_annualized       : XIRR of investor cash flows
        mwr_period           : period-rate IRR (not annualized)
        hpr                  : naive
        nav_start, nav_end, n_external_flows, total_external_flow,
        days, benchmark_return (vs SPY total return)
      }
    """
    from engine.memory import PortfolioNavSnapshot, CashFlow, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        # ── NAV at start: snapshot at or before start.nav_open ────────────
        snap_start = (
            sess.query(PortfolioNavSnapshot)
                .filter(PortfolioNavSnapshot.snapshot_date >= start)
                .order_by(PortfolioNavSnapshot.snapshot_date.asc())
                .first()
        )
        snap_end = (
            sess.query(PortfolioNavSnapshot)
                .filter(PortfolioNavSnapshot.snapshot_date <= end)
                .order_by(PortfolioNavSnapshot.snapshot_date.desc())
                .first()
        )
        if not snap_start or not snap_end:
            return _empty_summary(start, end)

        nav_start = float(snap_start.nav_open)
        nav_end = float(snap_end.nav_close)

        # ── External flows in (start, end] ────────────────────────────────
        flows_q = (
            sess.query(CashFlow)
                .filter(CashFlow.flow_date >= start)
                .filter(CashFlow.flow_date <= end)
                .filter(CashFlow.status == "applied")
                .filter(CashFlow.is_external.is_(True))
                .order_by(CashFlow.flow_date.asc())
                .all()
        )
        flows = [(f.flow_date, float(f.amount_usd)) for f in flows_q]

        twr_md = compute_modified_dietz_period(
            nav_start, nav_end, flows, start, end,
        )
        twr_geo = compute_twr_geometric_link(start, end, session=sess)
        hpr = compute_hpr(nav_start, nav_end)

        # ── XIRR investor cash flow series ────────────────────────────────
        # Initial: -nav_start at start
        # Each external flow: -amount_usd (deposit = INTO portfolio = OUT of
        # investor; withdrawal = OUT of portfolio = INTO investor; sign flip
        # achieves both)
        # Terminal: +nav_end at end
        investor_cf = [(start, -nav_start)]
        for d, a in flows:
            investor_cf.append((d, -a))
        investor_cf.append((end, +nav_end))
        try:
            mwr_ann = compute_xirr(investor_cf)
        except Exception as exc:
            logger.warning("compute_period_summary: XIRR failed (%s)", exc)
            mwr_ann = float("nan")

        # Period MWR (not annualized) for short-horizon comparison
        days = (end - start).days
        if math.isfinite(mwr_ann) and days > 0:
            mwr_period = (1.0 + mwr_ann) ** (days / 365.0) - 1.0
        else:
            mwr_period = float("nan")

        # ── Benchmark active return (SPY total return) ────────────────────
        bench_ret = None
        if snap_start.benchmark_close and snap_end.benchmark_close:
            try:
                bs, be = (
                    float(snap_start.benchmark_close),
                    float(snap_end.benchmark_close),
                )
                bench_ret = (be - bs) / bs if bs > 0 else None
            except Exception:
                pass

        active_ret = None
        if bench_ret is not None:
            active_ret = twr_geo - bench_ret

        return {
            "start":                  start,
            "end":                    end,
            "days":                   days,
            "nav_start":              nav_start,
            "nav_end":                nav_end,
            "n_external_flows":       len(flows),
            "total_external_flow":    sum(a for _, a in flows),
            "twr_modified_dietz":     twr_md,
            "twr_geometric_linked":   twr_geo,
            "mwr_annualized":         mwr_ann,
            "mwr_period":             mwr_period,
            "hpr":                    hpr,
            "benchmark_return":       bench_ret,
            "active_return":          active_ret,
        }
    finally:
        if own:
            sess.close()


def _empty_summary(start: datetime.date, end: datetime.date) -> dict:
    return {
        "start": start, "end": end, "days": (end - start).days,
        "nav_start": None, "nav_end": None,
        "n_external_flows": 0, "total_external_flow": 0.0,
        "twr_modified_dietz": None, "twr_geometric_linked": None,
        "mwr_annualized": None, "mwr_period": None, "hpr": None,
        "benchmark_return": None, "active_return": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. NAV-series statistics — Sharpe / Vol / Drawdown for live dashboard
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe_from_nav_series(
    start:            datetime.date | None = None,
    end:              datetime.date | None = None,
    *,
    rf_annual:        float = 0.0,
    periods_per_year: int   = 252,
    min_obs:          int   = 20,
    session:          Any | None = None,
) -> float | None:
    """
    Annualized Sharpe ratio computed from daily Modified Dietz returns in
    PortfolioNavSnapshot. Returns None when fewer than `min_obs` observations
    are available — by Lo (2002) the Sharpe estimator's sampling distribution
    is too wide to be meaningful below ~20 daily observations.
    """
    from engine.portfolio_returns import get_nav_series

    nav_df = get_nav_series(start=start, end=end, session=session)
    if nav_df.empty:
        return None
    daily = nav_df["daily_modified_dietz"].dropna()
    if len(daily) < min_obs:
        return None
    excess = daily - (rf_annual / periods_per_year)
    std = float(excess.std(ddof=1))
    if std < 1e-12:
        return None
    return float((excess.mean() / std) * (periods_per_year ** 0.5))


def compute_vol_from_nav_series(
    start:            datetime.date | None = None,
    end:              datetime.date | None = None,
    *,
    periods_per_year: int = 252,
    min_obs:          int = 20,
    session:          Any | None = None,
) -> float | None:
    """Annualized vol from daily Modified Dietz returns."""
    from engine.portfolio_returns import get_nav_series

    nav_df = get_nav_series(start=start, end=end, session=session)
    if nav_df.empty:
        return None
    daily = nav_df["daily_modified_dietz"].dropna()
    if len(daily) < min_obs:
        return None
    return float(daily.std(ddof=1) * (periods_per_year ** 0.5))


def compute_drawdown_series(
    start:    datetime.date | None = None,
    end:      datetime.date | None = None,
    *,
    session:  Any | None = None,
):
    """
    Returns pandas DataFrame indexed by date with columns:
       nav, peak, drawdown
    where drawdown = (nav − running peak) / running peak (always ≤ 0).
    Empty DataFrame if no NAV history.
    """
    import pandas as pd
    from engine.portfolio_returns import get_nav_series

    nav_df = get_nav_series(start=start, end=end, session=session)
    if nav_df.empty:
        return pd.DataFrame()
    nav = nav_df["nav_close"]
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return pd.DataFrame({"nav": nav, "peak": peak, "drawdown": dd})


def compute_dd_summary(
    start:    datetime.date | None = None,
    end:      datetime.date | None = None,
    *,
    session:  Any | None = None,
) -> dict:
    """
    Drawdown summary: current drawdown, max drawdown over period, date of max.
    All values are decimals ≤ 0 (current = today's distance below peak).
    """
    dd_df = compute_drawdown_series(start=start, end=end, session=session)
    if dd_df.empty:
        return {
            "dd_current":   None,
            "dd_max":       None,
            "dd_max_date":  None,
        }
    return {
        "dd_current":  float(dd_df["drawdown"].iloc[-1]),
        "dd_max":      float(dd_df["drawdown"].min()),
        "dd_max_date": dd_df["drawdown"].idxmin(),
    }


def compute_period_nav_change(
    start:    datetime.date | None = None,
    end:      datetime.date | None = None,
    *,
    session:  Any | None = None,
) -> tuple[float | None, float | None]:
    """
    Return (absolute_dollar_change, percent_change) for nav_close over the
    range. Both None when range has < 2 NAV snapshots.
    """
    from engine.portfolio_returns import get_nav_series

    nav_df = get_nav_series(start=start, end=end, session=session)
    if len(nav_df) < 1:
        return (None, None)
    nav_first = float(nav_df["nav_close"].iloc[0])
    nav_last = float(nav_df["nav_close"].iloc[-1])
    abs_change = nav_last - nav_first
    pct_change = (nav_last - nav_first) / nav_first if nav_first else None
    return (abs_change, pct_change)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Daily return distribution (institutional risk dashboard standard)
# ─────────────────────────────────────────────────────────────────────────────

def compute_return_distribution_stats(
    start:    datetime.date | None = None,
    end:      datetime.date | None = None,
    *,
    min_obs:  int = 5,
    session:  Any | None = None,
) -> dict | None:
    """
    Distribution statistics over daily Modified Dietz returns.

    Returns dict with n / mean / std / skew / kurt (excess, Fisher) / best /
    worst / best_date / worst_date / hit_rate. None when fewer than `min_obs`
    daily observations.

    Skewness and kurtosis follow Bacon (2019) Ch.6 institutional risk dashboard
    convention. Excess kurtosis = 0 for normal; > 0 indicates fat tails.
    """
    from engine.portfolio_returns import get_nav_series

    nav_df = get_nav_series(start=start, end=end, session=session)
    if nav_df.empty:
        return None
    daily = nav_df["daily_modified_dietz"].dropna()
    if len(daily) < min_obs:
        return None

    try:
        from scipy import stats as _sps
        skew = float(_sps.skew(daily))
        kurt = float(_sps.kurtosis(daily, fisher=True))
    except Exception:
        # Fallback to manual computation if scipy unavailable
        m = float(daily.mean())
        s = float(daily.std(ddof=1)) or 1e-12
        n = len(daily)
        skew = float(((daily - m) ** 3).sum() / n / (s ** 3))
        kurt = float(((daily - m) ** 4).sum() / n / (s ** 4) - 3.0)

    return {
        "n":           int(len(daily)),
        "mean":        float(daily.mean()),
        "std":         float(daily.std(ddof=1)),
        "skew":        skew,
        "kurt":        kurt,   # excess (normal = 0)
        "best":        float(daily.max()),
        "worst":       float(daily.min()),
        "best_date":   daily.idxmax(),
        "worst_date":  daily.idxmin(),
        "hit_rate":    float((daily > 0).sum() / len(daily)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Activity timeline — cross-table operational event stream
# ─────────────────────────────────────────────────────────────────────────────

def get_activity_timeline(
    start:    datetime.date | None = None,
    end:      datetime.date | None = None,
    *,
    limit:    int = 50,
    session:  Any | None = None,
) -> list[dict]:
    """
    Unified operational event stream merging:

      - cash_flow:   CashFlow.status='applied', is_external=True
      - decision:    DecisionLog tab_type='sector' (sector-pipeline LLM debate)
      - harking:     HARKingFlag (integrity violations)
      - rebalance:   from DecisionLog rows with weight_adjustment_pct != 0

    Returned list of dicts (sorted by date desc) with keys:
      date / type / label / severity / details

    Use this on the supervisor dashboard to show "what did the agentic AI
    actually DO this period" — operational visibility, not return-centric noise.
    """
    from engine.memory import (
        SessionFactory, CashFlow, DecisionLog, HARKingFlag,
    )

    own = session is None
    sess = session if session is not None else SessionFactory()
    events: list[dict] = []
    try:
        # ── Cash flows ─────────────────────────────────────────────────
        q_cf = sess.query(CashFlow).filter(
            CashFlow.status == "applied",
            CashFlow.is_external.is_(True),
        )
        if start is not None:
            q_cf = q_cf.filter(CashFlow.flow_date >= start)
        if end is not None:
            q_cf = q_cf.filter(CashFlow.flow_date <= end)
        for cf in q_cf.order_by(CashFlow.flow_date.desc()).limit(limit).all():
            events.append({
                "date":     cf.flow_date,
                "type":     "cash_flow",
                "label":    f"{cf.flow_type}: ${abs(cf.amount_usd):,.0f}",
                "severity": "info",
                "details":  cf.notes or
                            (f"supervisor={cf.supervisor_id}" if cf.supervisor_id else ""),
            })

        # ── Sector decisions ──────────────────────────────────────────
        q_dec = sess.query(DecisionLog).filter(
            DecisionLog.tab_type == "sector",
        )
        if start is not None:
            q_dec = q_dec.filter(DecisionLog.decision_date >= start)
        if end is not None:
            q_dec = q_dec.filter(DecisionLog.decision_date <= end)
        for d in q_dec.order_by(DecisionLog.decision_date.desc()).limit(limit).all():
            adj = d.weight_adjustment_pct or 0.0
            label = (f"{d.sector_name} {d.direction or '?'}"
                     + (f" (adj {adj:+.1%})" if abs(adj) > 1e-9 else ""))
            events.append({
                "date":     d.decision_date or (d.created_at and d.created_at.date()),
                "type":     "decision",
                "label":    label,
                "severity": "info",
                "details":  (d.ai_conclusion or "")[:120],
            })

        # ── HARKing integrity flags ────────────────────────────────────
        q_hf = sess.query(HARKingFlag)
        if start is not None:
            q_hf = q_hf.filter(HARKingFlag.detected_at >= datetime.datetime.combine(
                start, datetime.time.min))
        if end is not None:
            q_hf = q_hf.filter(HARKingFlag.detected_at <= datetime.datetime.combine(
                end, datetime.time.max))
        for f in q_hf.order_by(HARKingFlag.detected_at.desc()).limit(limit).all():
            sev_map = {"CRITICAL": "critical", "HIGH": "warn", "MEDIUM": "warn"}
            events.append({
                "date":     f.detected_at.date() if f.detected_at else None,
                "type":     "harking",
                "label":    f"{f.rule} {f.severity}: {f.spec_path}",
                "severity": sev_map.get(f.severity, "info"),
                "details":  (f.notes or "")[:120],
            })
    finally:
        if own:
            sess.close()

    # Sort + truncate
    events = [e for e in events if e["date"] is not None]
    events.sort(key=lambda e: e["date"], reverse=True)
    return events[:limit]
