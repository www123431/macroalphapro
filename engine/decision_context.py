"""
P-AUDIT v1 amendment (M3-corrected-ext-full, 2026-05-04, clarification +0 trials).

7-layer Decision Context modules + 5 EXT extensions. All deterministic
SQL / arithmetic / rule-based string formatters. **0 LLM** in this layer.

Layer map:
    L1  get_watchlist_origin     -- WatchlistEntry origin fields
    L2  get_quant_posture        -- multi-period TSMOM/CSMOM + EXT-2 league
    L3  get_regime_context       -- filtered prob + EXT-1 macro snapshot
    L4  get_portfolio_posture    -- sector exposure + EXT-4 HHI + EXT-5 dd
    L5  get_conditional_history  -- sector × direction × regime hit rate
    L6  compose_thesis           -- LLM-or-rule unified thesis dict
    L7a get_forward_preview      -- deterministic + EXT-3 calendar effects
                                    (NO Monte Carlo / probabilistic sim)

Cross-references:
    - WatchlistEntry / SimulatedPosition / SignalRecord / DecisionLog /
      RegimeSnapshot / PortfolioNavSnapshot (engine/memory.py)
    - compute_yield_curve_slope_signal / compute_macro_spread_signals
      (engine/signal.py) for EXT-1 macro snapshot

Spec: docs/spec_supervisor_approval_panel_v1.md (current_hash 0dac977bcd7c895a).
"""
from __future__ import annotations

import datetime
import json
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Static FOMC calendar (2024-2026) — EXT-3 calendar effects
# Source: federalreserve.gov / FOMC meeting schedule (regular meetings only,
# excluding emergency / unscheduled ones). Hardcoded because daily look-up is
# overkill for a static calendar; refresh annually.
# ─────────────────────────────────────────────────────────────────────────────
_FOMC_DATES_2024_2026: tuple[datetime.date, ...] = (
    datetime.date(2024, 1, 31),
    datetime.date(2024, 3, 20),
    datetime.date(2024, 5, 1),
    datetime.date(2024, 6, 12),
    datetime.date(2024, 7, 31),
    datetime.date(2024, 9, 18),
    datetime.date(2024, 11, 7),
    datetime.date(2024, 12, 18),
    datetime.date(2025, 1, 29),
    datetime.date(2025, 3, 19),
    datetime.date(2025, 5, 7),
    datetime.date(2025, 6, 18),
    datetime.date(2025, 7, 30),
    datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29),
    datetime.date(2025, 12, 10),
    datetime.date(2026, 1, 28),
    datetime.date(2026, 3, 18),
    datetime.date(2026, 4, 29),
    datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 11, 4),
    datetime.date(2026, 12, 16),
)


def _date_to_str(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, datetime.date):
        return d.isoformat()
    return str(d)


# ═════════════════════════════════════════════════════════════════════════════
# L1 — WatchlistEntry origin
# ═════════════════════════════════════════════════════════════════════════════

def get_watchlist_origin(approval_id: int, *, session: Any | None = None) -> dict:
    """
    L1: Return WatchlistEntry origin context for a PendingApproval row.

    Returns dict with `available` flag; if no linked WatchlistEntry, returns
    {"available": False}. Otherwise 14+ keys (see spec § A.L1).
    """
    from engine.memory import (
        PendingApproval, WatchlistEntry, SessionFactory,
    )

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        pa = sess.get(PendingApproval, int(approval_id))
        if pa is None or pa.watchlist_entry_id is None:
            return {"available": False}

        wl = sess.get(WatchlistEntry, int(pa.watchlist_entry_id))
        if wl is None:
            return {"available": False}

        days_in_watchlist: int | None = None
        if wl.created_date is not None and wl.triggered_date is not None:
            days_in_watchlist = (wl.triggered_date - wl.created_date).days
        elif wl.created_date is not None and pa.triggered_date is not None:
            days_in_watchlist = (pa.triggered_date - wl.created_date).days

        return {
            "available":                True,
            "watchlist_entry_id":       int(wl.id),
            "source_agent":             wl.source_agent,
            "created_date":             _date_to_str(wl.created_date),
            "triggered_date":           _date_to_str(wl.triggered_date),
            "direction":                wl.direction,
            "position_rank":            wl.position_rank,
            "regime_label_at_creation": wl.regime_label,
            "entry_tsmom_signal":       wl.entry_tsmom_signal,
            "entry_csmom_rank":         wl.entry_csmom_rank,
            "entry_composite_score":    wl.entry_composite_score,
            "entry_ann_vol":            wl.entry_ann_vol,
            "quant_baseline_weight":    wl.quant_baseline_weight,
            "llm_adjustment_pct":       wl.llm_adjustment_pct,
            "suggested_weight":         wl.suggested_weight,
            "confidence":               wl.confidence,
            "decision_log_id":          wl.decision_log_id,
            "entry_condition_summary":  _format_entry_condition(
                                              wl.entry_condition_json),
            "days_in_watchlist":        days_in_watchlist,
            "watchlist_status":         wl.status,
        }
    finally:
        if own:
            sess.close()


def _format_entry_condition(entry_condition_json: str | None) -> str | None:
    if not entry_condition_json:
        return None
    try:
        ec = json.loads(entry_condition_json)
    except Exception:
        return entry_condition_json[:200]

    if isinstance(ec, dict):
        kind = ec.get("kind") or ec.get("type") or "rule"
        threshold = ec.get("threshold")
        operator = ec.get("operator") or ">="
        target = ec.get("target") or ec.get("metric") or "composite"
        bits = []
        if target:
            bits.append(str(target))
        if operator:
            bits.append(str(operator))
        if threshold is not None:
            bits.append(str(threshold))
        if "ttl_days" in ec:
            bits.append(f"ttl={ec['ttl_days']}d")
        return f"{kind}: " + " ".join(bits) if bits else f"{kind}: (no params)"

    return json.dumps(ec, ensure_ascii=False)[:200]


# ═════════════════════════════════════════════════════════════════════════════
# L2 — Quant posture (+ EXT-2 cross-sectional league table)
# ═════════════════════════════════════════════════════════════════════════════

def get_quant_posture(
    ticker: str | None,
    sector: str | None,
    *,
    session: Any | None = None,
) -> dict:
    """
    L2: Latest signal record for the target ticker plus a sector league
    table (EXT-2) of all 18 sectors ranked by composite_score on the latest
    common date.
    """
    from engine.memory import SignalRecord, SessionFactory
    from sqlalchemy import func

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        latest_date = (
            sess.query(func.max(SignalRecord.date)).scalar()
        )
        if latest_date is None:
            return {"available": False}

        ticker_row: SignalRecord | None = None
        if ticker:
            ticker_row = (
                sess.query(SignalRecord)
                    .filter(SignalRecord.ticker == ticker)
                    .filter(SignalRecord.date == latest_date)
                    .first()
            )
        if ticker_row is None and sector:
            ticker_row = (
                sess.query(SignalRecord)
                    .filter(SignalRecord.sector == sector)
                    .filter(SignalRecord.date == latest_date)
                    .first()
            )

        composite_trend_5d: list[float] = []
        if ticker:
            trail = (
                sess.query(SignalRecord)
                    .filter(SignalRecord.ticker == ticker)
                    .order_by(SignalRecord.date.desc())
                    .limit(5)
                    .all()
            )
            composite_trend_5d = [
                float(r.composite_score) for r in reversed(trail)
                if r.composite_score is not None
            ]

        league_rows = (
            sess.query(SignalRecord)
                .filter(SignalRecord.date == latest_date)
                .order_by(SignalRecord.composite_score.desc().nullslast())
                .all()
        )

        league_table = [
            {
                "rank":             rank + 1,
                "ticker":           r.ticker,
                "sector":           r.sector,
                "composite_score":  r.composite_score,
                "csmom_rank":       r.csmom_rank,
                "tsmom_signal":     r.tsmom_signal,
                "regime_at_calc":   r.regime_at_calc,
                "gate_status":      r.gate_status,
                "highlight":        (r.ticker == ticker),
            }
            for rank, r in enumerate(league_rows)
        ]

        if ticker_row is None:
            return {
                "available":           True,
                "ticker_row_present":  False,
                "as_of_date":          _date_to_str(latest_date),
                "league_table":        league_table,
                "league_n":            len(league_table),
            }

        return {
            "available":            True,
            "ticker_row_present":   True,
            "as_of_date":           _date_to_str(latest_date),
            "tsmom_signal":         ticker_row.tsmom_signal,
            "tsmom_raw":            ticker_row.tsmom_raw,
            "csmom_rank":           ticker_row.csmom_rank,
            "carry_norm":           ticker_row.carry_norm,
            "reversal_norm":        ticker_row.reversal_norm,
            "composite_score":      ticker_row.composite_score,
            "composite_trend_5d":   composite_trend_5d,
            "gate_status":          ticker_row.gate_status,
            "regime_at_calc":       ticker_row.regime_at_calc,
            "league_table":         league_table,
            "league_n":             len(league_table),
        }
    finally:
        if own:
            sess.close()


# ═════════════════════════════════════════════════════════════════════════════
# L3 — Regime context (+ EXT-1 macro snapshot)
# ═════════════════════════════════════════════════════════════════════════════

def get_regime_context(
    ticker: str | None,
    sector: str | None,
    watchlist_created_date: str | None,
    *,
    session: Any | None = None,
    min_n: int = 10,
) -> dict:
    """
    L3 + EXT-1.

    Returns:
        regime label + filtered probabilities (P(risk_on/off/transition))
        regime_at_creation vs regime_now
        ticker × regime conditional history (n_obs gate, ex-ante caveat)
        macro_snapshot: yield curve slope / credit spread / VIX term / dollar
    """
    from engine.memory import (
        RegimeSnapshot, SimulatedPosition, SessionFactory,
    )
    from sqlalchemy import func

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        latest = (
            sess.query(RegimeSnapshot)
                .order_by(RegimeSnapshot.as_of_date.desc())
                .first()
        )
        if latest is None:
            return {"available": False, "ex_ante_caveat": True}

        p_risk_on = float(latest.p_risk_on or 0.0)
        p_risk_off = float(latest.p_risk_off or 0.0)
        p_transition = max(0.0, 1.0 - p_risk_on - p_risk_off)

        regime_at_creation: str | None = None
        if watchlist_created_date:
            try:
                wcd = datetime.date.fromisoformat(watchlist_created_date)
            except Exception:
                wcd = None
            if wcd is not None:
                creation_snap = (
                    sess.query(RegimeSnapshot)
                        .filter(RegimeSnapshot.as_of_date <= wcd)
                        .order_by(RegimeSnapshot.as_of_date.desc())
                        .first()
                )
                if creation_snap is not None:
                    regime_at_creation = creation_snap.regime

        regime_drifted = (
            (regime_at_creation is not None)
            and (regime_at_creation != latest.regime)
        )

        ticker_history = _ticker_in_regime_history(
            sess, ticker=ticker, sector=sector,
            regime_label=latest.regime, min_n=min_n,
        )

        macro = _build_macro_snapshot(latest)

        return {
            "available":           True,
            "as_of_date":          _date_to_str(latest.as_of_date),
            "regime_label":        latest.regime,
            "p_risk_on":           round(p_risk_on, 4),
            "p_risk_off":          round(p_risk_off, 4),
            "p_transition":        round(p_transition, 4),
            "regime_method":       latest.method,
            "regime_at_creation":  regime_at_creation,
            "regime_drifted":      bool(regime_drifted),
            "ticker_in_regime_history": ticker_history,
            "macro_snapshot":      macro,
            "ex_ante_caveat":      True,
            "ex_ante_note":
                "filtered probabilities; not ex-post smoothed; "
                "Diebold-Lee-Weinbach 1994 caveat applies",
        }
    finally:
        if own:
            sess.close()


def _ticker_in_regime_history(
    sess: Any,
    ticker: str | None,
    sector: str | None,
    regime_label: str,
    min_n: int = 10,
) -> dict:
    """SQL: same sector × same regime over last 720 days; n_obs gate."""
    from engine.memory import DecisionLog

    if not (ticker or sector):
        return {"n_obs": 0, "insufficient_data": True}

    cutoff = datetime.date.today() - datetime.timedelta(days=720)
    q = (
        sess.query(DecisionLog.active_return, DecisionLog.accuracy_score)
            .filter(DecisionLog.tab_type == "sector")
            .filter(DecisionLog.macro_regime == regime_label)
            .filter(DecisionLog.decision_date >= cutoff)
            .filter(DecisionLog.verified.is_(True))
    )
    if sector:
        q = q.filter(DecisionLog.sector_name == sector)
    elif ticker:
        q = q.filter(DecisionLog.ticker == ticker)

    rows = q.all()
    n = len(rows)
    if n < min_n:
        return {
            "n_obs":              n,
            "min_n_required":     min_n,
            "insufficient_data":  True,
            "hit_rate":           None,
            "mean_active_return": None,
        }

    rets = [r[0] for r in rows if r[0] is not None]
    accs = [r[1] for r in rows if r[1] is not None]
    hit_rate = (
        float(sum(1 for a in accs if a >= 0.5) / len(accs)) if accs else None
    )
    mean_ret = (float(sum(rets) / len(rets))) if rets else None
    return {
        "n_obs":              n,
        "min_n_required":     min_n,
        "insufficient_data":  False,
        "hit_rate":           hit_rate,
        "mean_active_return": mean_ret,
    }


def _build_macro_snapshot(latest_regime_row: Any) -> dict:
    """EXT-1: macro snapshot — yield curve / credit spread / VIX / dollar."""
    out: dict = {
        "yield_spread_10_2":      None,
        "yield_curve_inverted":   None,
        "credit_spread":          None,
        "vix":                    None,
        "vix_term_structure":     None,
        "dollar_index":           None,
        "data_source":            "RegimeSnapshot + signal.macro_*",
    }
    out["yield_spread_10_2"] = (
        float(latest_regime_row.yield_spread)
        if latest_regime_row.yield_spread is not None else None
    )
    if out["yield_spread_10_2"] is not None:
        out["yield_curve_inverted"] = bool(out["yield_spread_10_2"] < 0)
    out["vix"] = (
        float(latest_regime_row.vix)
        if latest_regime_row.vix is not None else None
    )

    # Best-effort: pull macro_spread_signals (HY OAS) if available
    try:
        from engine.signal import compute_macro_spread_signals
        mss = compute_macro_spread_signals(datetime.date.today())
        if isinstance(mss, dict):
            hyg = mss.get("HY_OAS") or {}
            if isinstance(hyg, dict):
                out["credit_spread"] = hyg.get("level")
    except Exception:
        pass

    return out


# ═════════════════════════════════════════════════════════════════════════════
# L4 — Portfolio posture (+ EXT-4 HHI + EXT-5 underwater duration)
# ═════════════════════════════════════════════════════════════════════════════

def get_portfolio_posture(
    approval_id: int,
    suggested_weight: float | None,
    *,
    session: Any | None = None,
) -> dict:
    """
    L4 + EXT-4 + EXT-5.

    Returns sector exposure / concentration / approve-after delta, HHI
    (current + post-approve), and drawdown metrics.

    IMPORTANT (2026-05-04 fix): uses engine.portfolio_tracker.get_current_positions
    (per-sector latest snapshot) NOT a single global MAX(snapshot_date).
    Earlier draft used the latter and reproduced the same bug fixed in
    portfolio_tracker.py: when a rebalance touches only a subset of sectors,
    the global MAX collapses to one row → narrative said "1 holding" while
    Positions page (using the correct query) showed 32. They MUST share the
    same data source.
    """
    from engine.memory import PendingApproval, SessionFactory
    from engine.portfolio_tracker import get_current_positions

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        pa = sess.get(PendingApproval, int(approval_id))
        if pa is None:
            return {"available": False}

        # Same source-of-truth as Positions / Live Dashboard pages.
        # Live dashboard further filters abs(actual_weight) < 1e-6
        # (pages/live_dashboard.py:691) — apply the same threshold here so
        # narrative head-count matches what the supervisor sees on Positions.
        # (Signature-drift fix 2026-05-24: get_current_positions() dropped its
        # track/include_closed kwargs; the latest snapshot + the abs<1e-6 filter
        # below already give "current executed positions".)
        df = get_current_positions()

        if df is None or df.empty:
            weights: list[float] = []
            sector_weights: dict[str, float] = {}
            sector_position_count: dict[str, int] = {}
            portfolio_total_value = 0.0
            latest_snap_date = None
        else:
            weights = []
            sector_weights = {}
            sector_position_count = {}
            portfolio_total_value = 0.0
            for _idx, row in df.iterrows():
                w = float(row.get("actual_weight") or 0.0)
                if abs(w) < 1e-6:
                    continue   # skip planned-but-not-executed; matches live_dashboard
                weights.append(w)
                sec = row.get("sector") or _idx
                sector_weights[sec] = sector_weights.get(sec, 0.0) + w
                sector_position_count[sec] = sector_position_count.get(sec, 0) + 1
                pv = row.get("position_value")
                if pv is not None:
                    try:
                        portfolio_total_value += float(pv)
                    except Exception:
                        pass
            try:
                latest_snap_date = max(df["snapshot_date"].tolist())
            except Exception:
                latest_snap_date = None

        sector_current_exposure_pct = (
            sector_weights.get(pa.sector, 0.0) if pa.sector else 0.0
        )
        post_approve_sector_exposure_pct = (
            sector_current_exposure_pct + (suggested_weight or 0.0)
        )
        post_approve_concentration_delta_pct = (suggested_weight or 0.0)

        gross = sum(abs(w) for w in weights)
        vol_budget_used_pct = gross

        hhi_current = sum(w * w for w in weights) if weights else 0.0
        post_weights = list(weights) + [(suggested_weight or 0.0)]
        hhi_post_approve = sum(w * w for w in post_weights) if post_weights else 0.0
        hhi_delta = hhi_post_approve - hhi_current
        hhi_interpretation = (
            "highly_concentrated" if hhi_current > 0.25 else
            "moderate"            if hhi_current > 0.15 else
            "diversified"
        )

        drawdown = _compute_drawdown_metrics(sess)
        mcr_estimate = _approx_mcr(suggested_weight)

        return {
            "available":                       True,
            "snapshot_date":                   _date_to_str(latest_snap_date),
            "portfolio_total_positions":       len(weights),
            "portfolio_total_value_usd":       round(portfolio_total_value, 2),
            "sector_position_count":           sector_position_count.get(pa.sector, 0)
                                                if pa.sector else 0,
            "sector_current_exposure_pct":     round(sector_current_exposure_pct, 4),
            "post_approve_sector_exposure_pct": round(post_approve_sector_exposure_pct, 4),
            "post_approve_concentration_delta_pct":
                                               round(post_approve_concentration_delta_pct, 4),
            "vol_budget_used_pct":             round(vol_budget_used_pct, 4),
            "mcr_estimate":                    mcr_estimate,
            "mcr_caveat":
                "simplified additive approximation; not full covariance",
            "hhi_metrics": {
                "hhi_current":           round(hhi_current, 4),
                "hhi_post_approve":      round(hhi_post_approve, 4),
                "hhi_delta":             round(hhi_delta, 4),
                "hhi_interpretation":    hhi_interpretation,
            },
            "drawdown_metrics":          drawdown,
        }
    finally:
        if own:
            sess.close()


def _approx_mcr(suggested_weight: float | None) -> dict:
    """
    Simplified additive MCR approximation (NOT full covariance):
    - Treat sector vol as a flat 18% annualized proxy
    - MCR_i ≈ |delta_w_i| × σ_proxy
    """
    if suggested_weight is None:
        return {"available": False}
    sigma_proxy = 0.18
    mcr_bps = abs(float(suggested_weight)) * sigma_proxy * 10_000
    return {
        "available":         True,
        "sigma_proxy_pct":   sigma_proxy * 100.0,
        "mcr_bps":           round(mcr_bps, 2),
        "delta_weight":      float(suggested_weight),
    }


def _compute_drawdown_metrics(sess: Any) -> dict:
    """EXT-5: underwater duration + 90d / 1y max DD on PortfolioNavSnapshot."""
    from engine.memory import PortfolioNavSnapshot

    rows = (
        sess.query(PortfolioNavSnapshot.snapshot_date,
                   PortfolioNavSnapshot.nav_close)
            .order_by(PortfolioNavSnapshot.snapshot_date.asc())
            .all()
    )
    if not rows:
        return {"available": False}

    dates = [r[0] for r in rows]
    navs = [float(r[1]) for r in rows]
    n = len(navs)

    running_peak = navs[0]
    underwater_days = 0
    cur_underwater = 0
    for v in navs:
        if v >= running_peak:
            running_peak = v
            cur_underwater = 0
        else:
            cur_underwater += 1
        underwater_days = cur_underwater  # latest streak

    current_drawdown_pct = (navs[-1] - running_peak) / running_peak if running_peak else 0.0

    today = dates[-1]
    cutoff_90d = today - datetime.timedelta(days=90)
    cutoff_365d = today - datetime.timedelta(days=365)

    def _max_dd(series_navs: list[float]) -> float:
        if not series_navs:
            return 0.0
        peak = series_navs[0]; max_dd = 0.0
        for v in series_navs:
            peak = max(peak, v)
            dd = (v - peak) / peak if peak else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    navs_90 = [v for d, v in zip(dates, navs) if d >= cutoff_90d]
    navs_1y = [v for d, v in zip(dates, navs) if d >= cutoff_365d]

    return {
        "available":                True,
        "current_drawdown_pct":     round(current_drawdown_pct, 4),
        "underwater_days":          int(underwater_days),
        "max_drawdown_90d_pct":     round(_max_dd(navs_90), 4),
        "max_drawdown_1y_pct":      round(_max_dd(navs_1y), 4),
        "n_snapshots":              n,
    }


# ═════════════════════════════════════════════════════════════════════════════
# L5 — Conditional history
# ═════════════════════════════════════════════════════════════════════════════

def get_conditional_history(
    sector: str | None,
    direction: str | None,
    regime_label: str | None,
    *,
    session: Any | None = None,
    min_n: int = 5,
    lookback_days: int = 720,
) -> dict:
    """
    L5: same sector × same direction × same regime hit rate over last 720d.
    n_obs < min_n → numeric values None + insufficient_data flag.
    """
    from engine.memory import DecisionLog, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        if not (sector and direction):
            return {
                "n_obs": 0, "insufficient_data": True,
                "min_n_required": min_n, "lookback_days": lookback_days,
                "hit_rate": None, "mean_active_return": None,
                "median_holding_days": None,
            }

        cutoff = datetime.date.today() - datetime.timedelta(days=lookback_days)
        q = (
            sess.query(DecisionLog)
                .filter(DecisionLog.tab_type == "sector")
                .filter(DecisionLog.sector_name == sector)
                .filter(DecisionLog.decision_date >= cutoff)
                .filter(DecisionLog.verified.is_(True))
        )
        if direction in ("long", "超配"):
            q = q.filter(DecisionLog.direction.in_(("long", "超配")))
        elif direction in ("short", "低配"):
            q = q.filter(DecisionLog.direction.in_(("short", "低配")))
        if regime_label:
            q = q.filter(DecisionLog.macro_regime == regime_label)

        rows = q.all()
        n = len(rows)
        if n < min_n:
            return {
                "n_obs":              n,
                "min_n_required":     min_n,
                "lookback_days":      lookback_days,
                "insufficient_data":  True,
                "hit_rate":           None,
                "mean_active_return": None,
                "median_holding_days": None,
            }

        rets = [r.active_return for r in rows if r.active_return is not None]
        accs = [r.accuracy_score for r in rows if r.accuracy_score is not None]
        holdings = [r.barrier_days for r in rows if r.barrier_days is not None]

        def _median(xs: list) -> float | None:
            if not xs: return None
            sx = sorted(xs); m = len(sx) // 2
            return float(sx[m]) if len(sx) % 2 else float((sx[m-1] + sx[m]) / 2)

        return {
            "n_obs":              n,
            "min_n_required":     min_n,
            "lookback_days":      lookback_days,
            "insufficient_data":  False,
            "hit_rate":           float(sum(1 for a in accs if a >= 0.5) / len(accs))
                                  if accs else None,
            "mean_active_return": float(sum(rets) / len(rets)) if rets else None,
            "median_holding_days": _median(holdings),
        }
    finally:
        if own:
            sess.close()


# ═════════════════════════════════════════════════════════════════════════════
# L6 — Thesis composer (LLM-or-rule, unified shape)
# ═════════════════════════════════════════════════════════════════════════════

def compose_thesis(
    decision_log_payload: dict | None,
    watchlist_origin: dict | None,
    quant_posture: dict | None,
    regime_context: dict | None,
) -> dict:
    """
    L6: Unified thesis dict regardless of whether DecisionLog (LLM debate)
    exists. When DecisionLog is None, builds a deterministic rule-based
    string from WatchlistEntry + signal state.

    Returns dict with key 'thesis_source' = "decision_log" or "rule_based".
    """
    if decision_log_payload and decision_log_payload.get("available"):
        return {
            "thesis_source":     "decision_log",
            "key_thesis":        decision_log_payload.get("key_thesis"),
            "primary_risk":      decision_log_payload.get("primary_risk"),
            "debate_excerpt":    decision_log_payload.get("debate_summary_excerpt"),
            "decision_id":       decision_log_payload.get("decision_id"),
            "ui_label":          "thesis 来自 LLM 辩论 (DecisionLog)",
        }

    return _compose_rule_based_thesis(watchlist_origin, quant_posture, regime_context)


def _compose_rule_based_thesis(
    wl: dict | None,
    qp: dict | None,
    rc: dict | None,
) -> dict:
    """No-LLM string composer. Deterministic given the same inputs."""
    direction = (wl or {}).get("direction") or "neutral"
    ticker = None
    sector = None
    composite = (wl or {}).get("entry_composite_score")
    ann_vol = (wl or {}).get("entry_ann_vol")
    tsmom = (wl or {}).get("entry_tsmom_signal")
    csmom_rank = (wl or {}).get("entry_csmom_rank")
    regime_at_create = (wl or {}).get("regime_label_at_creation")
    regime_now = (rc or {}).get("regime_label")

    # Prefer current quant_posture values when available (more recent than
    # entry-time snapshot)
    if qp and qp.get("ticker_row_present"):
        composite = qp.get("composite_score") if qp.get("composite_score") is not None else composite
        tsmom = qp.get("tsmom_signal") if qp.get("tsmom_signal") is not None else tsmom

    bits: list[str] = []
    if direction:
        bits.append(direction.upper())
    if tsmom is not None:
        bits.append(f"TSMOM={'+1' if tsmom > 0 else '-1' if tsmom < 0 else '0'}")
    if composite is not None:
        bits.append(f"composite={composite:.0f}/100" if isinstance(composite, (int, float))
                     else f"composite={composite}")
    if csmom_rank is not None:
        try:
            csmom_pct = float(csmom_rank) * 100 if float(csmom_rank) <= 1.0 else float(csmom_rank)
            bits.append(f"CSMOM_rank={csmom_pct:.0f}%")
        except Exception:
            pass
    if ann_vol is not None:
        try:
            bits.append(f"ann_vol={float(ann_vol)*100:.1f}%" if float(ann_vol) <= 1.0 else f"ann_vol={ann_vol}")
        except Exception:
            pass
    if regime_now:
        bits.append(f"regime={regime_now}")

    rule_thesis = " · ".join(bits) if bits else "no signal data available"

    risk_bits: list[str] = []
    if regime_at_create and regime_now and regime_at_create != regime_now:
        risk_bits.append(f"regime drifted {regime_at_create}→{regime_now}")
    if composite is not None and isinstance(composite, (int, float)) and composite < 60:
        risk_bits.append(f"composite below typical entry threshold (60)")
    if ann_vol is not None:
        try:
            if float(ann_vol) > 0.30:
                risk_bits.append("ann_vol > 30% (vol expansion risk)")
        except Exception:
            pass
    if not risk_bits:
        risk_bits = ["regime drift", "composite drop below entry gate", "vol expansion >2σ"]
    rule_risk = " | ".join(risk_bits)

    return {
        "thesis_source":     "rule_based",
        "key_thesis":        rule_thesis,
        "primary_risk":      rule_risk,
        "debate_excerpt":    None,
        "decision_id":       None,
        "ui_label":          "thesis: 量化规则合成（无 LLM）",
    }


# ═════════════════════════════════════════════════════════════════════════════
# L7a — Forward preview (deterministic) + EXT-3 calendar effects
# ═════════════════════════════════════════════════════════════════════════════

def get_forward_preview(
    approval_id: int,
    suggested_weight: float | None,
    *,
    session: Any | None = None,
) -> dict:
    """
    L7a + EXT-3.

    Approve path: position $ delta / sector exposure delta / cost bps.
    Reject path: watchlist revert state / days-to-next-trigger heuristic.
    Calendar effects: FOMC blackout / pre-FOMC drift / turn-of-month / earnings.

    NO probabilistic / Monte Carlo / forward P&L distribution.
    """
    from engine.memory import (
        PendingApproval, PortfolioNavSnapshot, SystemConfig, SessionFactory,
    )
    from sqlalchemy import func

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        pa = sess.get(PendingApproval, int(approval_id))
        if pa is None:
            return {"available": False}

        latest_nav_row = (
            sess.query(PortfolioNavSnapshot)
                .order_by(PortfolioNavSnapshot.snapshot_date.desc())
                .first()
        )
        nav = float(latest_nav_row.nav_close) if latest_nav_row else None
        if nav is None:
            try:
                cfg = (
                    sess.query(SystemConfig)
                        .filter_by(key="paper_trading_nav")
                        .first()
                )
                nav = float(cfg.value) if cfg else 1_000_000.0
            except Exception:
                nav = 1_000_000.0

        sw = float(suggested_weight or 0.0)
        position_value_delta_usd = sw * nav
        estimated_trade_cost_bps = 7.0  # default roundtrip; conservative
        estimated_trade_cost_usd = (
            abs(position_value_delta_usd) * estimated_trade_cost_bps / 10_000
        )

        approve_path = {
            "position_value_delta_usd":    round(position_value_delta_usd, 2),
            "sector_exposure_after_pct":   None,  # filled by caller from L4
            "concentration_change_pct":    round(sw, 4),
            "estimated_trade_cost_bps":    estimated_trade_cost_bps,
            "estimated_trade_cost_usd":    round(estimated_trade_cost_usd, 2),
            "nav_basis":                   round(nav, 2),
        }

        reject_path = {
            "watchlist_revert_status":
                "watching" if pa.approval_type == "entry" else
                "rejected_logged" if pa.approval_type == "risk_control" else
                "rebalance_skipped" if pa.approval_type == "rebalance" else
                "no_op",
            "days_to_next_potential_trigger_estimate":
                _next_trigger_estimate(pa.approval_type),
        }

        calendar_effects = _calendar_effects_today(pa.triggered_date or datetime.date.today())

        return {
            "available":          True,
            "approve_path":       approve_path,
            "reject_path":        reject_path,
            "calendar_effects":   calendar_effects,
            "no_montecarlo_note": "deterministic preview only; "
                                  "no probabilistic forward simulation by design",
        }
    finally:
        if own:
            sess.close()


def _next_trigger_estimate(approval_type: str | None) -> int | None:
    """Coarse rule-based estimate; not statistical."""
    if approval_type == "entry":
        return 5  # composite-trend retrigger heuristic
    if approval_type == "risk_control":
        return 3  # if alert persists, repolling expected within ~3 trading days
    return None


def compose_supervisor_narrative(base: dict, dc: dict) -> str:
    """
    Deterministic markdown-formatted narrative paragraphs that read like a
    sell-side analyst's note. Aggregates all 7 layers + 5 EXT into 6 prose
    sections so the supervisor can read instead of reading dashboards.

    **0 LLM**. Pure Python f-string + if-else templating. Same input ⇒ same
    output, byte-identical.
    """
    paras: list[str] = []
    paras.append(_para_why_this(base, dc))
    paras.append(_para_market_env(dc))
    paras.append(_para_quant_overview(dc))
    paras.append(_para_approve_impact(dc))
    paras.append(_para_history(dc))
    paras.append(_para_risks(dc))
    return "\n\n".join(p for p in paras if p)


# ── Narrative paragraph builders (deterministic) ─────────────────────────────

def _direction_zh(d: str | None) -> str:
    return {"long": "看多", "short": "看空", "neutral": "中性",
            "超配": "看多 (超配)", "低配": "看空 (低配)"}.get(d or "", d or "—")


def _regime_zh(r: str | None) -> str:
    return {
        "risk-on":    "risk-on（风险偏好）",
        "risk-off":   "risk-off（避险）",
        "transition": "transition（过渡）",
    }.get(r or "", r or "—")


def _para_why_this(base: dict, dc: dict) -> str:
    l1 = dc.get("watchlist_origin") or {}
    sector = base.get("sector") or "—"
    ticker = base.get("ticker") or "—"
    appr_type = base.get("approval_type") or ""

    title = "**为什么是这个标的**"

    if not l1.get("available"):
        cond = (base.get("triggered_condition") or "（未记录触发原因）").strip()
        type_zh = {
            "entry": "入场触发", "risk_control": "风控告警",
            "rebalance": "再平衡", "cash_flow": "资金流",
        }.get(appr_type, appr_type)
        return f"{title}\n这是一条 **{type_zh}** 类审批：{cond}（无 WatchlistEntry 关联）。"

    src = l1.get("source_agent") or "未记录的 agent"
    src_label = {
        "quant_agent":     "Quant agent",
        "research_agent":  "Research agent",
    }.get(src, src)

    days = l1.get("days_in_watchlist")
    days_str = f"{days} 天前" if days is not None else "之前"

    direction = (l1.get("direction") or "").lower()
    direction_label = _direction_zh(direction)

    sw = l1.get("suggested_weight")
    qb = l1.get("quant_baseline_weight")
    llm_adj = l1.get("llm_adjustment_pct")

    sw_str = f"{sw*100:.2f}%" if isinstance(sw, (int, float)) else "—"
    qb_str = f"{qb*100:.2f}%" if isinstance(qb, (int, float)) else "—"
    llm_adj_str = (
        f"{llm_adj*100:+.2f}pp" if isinstance(llm_adj, (int, float)) else "0pp"
    )

    ec = (l1.get("entry_condition_summary") or "").strip()
    ec_str = ec or "原始入场规则未详记录"

    confidence = l1.get("confidence")
    conf_clause = (
        f"，整体 confidence {confidence}/100" if confidence is not None else ""
    )

    return (
        f"{title}\n"
        f"{src_label} 在 **{days_str}**（{l1.get('created_date','—')}）将 "
        f"**{ticker}（{sector}）** 放进 watchlist，方向 **{direction_label}**，"
        f"建议权重 **{sw_str}**（quant 基线 {qb_str} + LLM 调整 {llm_adj_str}）"
        f"{conf_clause}。今天因 `{ec_str}` 触发入场。"
    )


def _para_market_env(dc: dict) -> str:
    l3 = dc.get("regime_context") or {}
    l7 = dc.get("forward_preview") or {}
    title = "**当下市场环境**"

    if not l3.get("available"):
        return f"{title}\n暂无 regime 数据。"

    regime = l3.get("regime_label") or "—"
    p_on = l3.get("p_risk_on") or 0.0
    p_off = l3.get("p_risk_off") or 0.0
    p_tr = l3.get("p_transition") or 0.0

    bits: list[str] = [f"当前处于 {_regime_zh(regime)} regime"]
    p_max = max(p_on, p_off, p_tr)
    if p_max > 0.85:
        bits.append(
            f"filtered probability P({regime})={p_max:.3f} 较为肯定"
        )
    elif p_max < 0.55:
        bits.append("filtered prob 三态接近，制度判断不确定")

    if l3.get("regime_drifted"):
        bits.append(
            f"注意：自 watchlist 创建时的 "
            f"{_regime_zh(l3.get('regime_at_creation'))} regime 已漂移到当前的 {regime}"
        )

    macro = l3.get("macro_snapshot") or {}
    macro_bits: list[str] = []

    yc = macro.get("yield_spread_10_2")
    if yc is not None:
        if yc < 0:
            macro_bits.append(
                f"10y-2y 利差 {yc:+.2f}pp（**已倒挂——历史上 12-18 月内的衰退信号**）"
            )
        else:
            macro_bits.append(f"10y-2y 利差 {yc:+.2f}pp（未倒挂）")

    vix = macro.get("vix")
    if vix is not None:
        if vix > 30:
            macro_bits.append(f"VIX {vix:.1f}（**高位**）")
        elif vix < 12:
            macro_bits.append(f"VIX {vix:.1f}（极低，complacency）")
        else:
            macro_bits.append(f"VIX {vix:.1f}（区间内）")

    cs = macro.get("credit_spread")
    if cs is not None:
        macro_bits.append(f"credit spread {cs:.2f}")

    para = "，".join(bits) + "。"
    if macro_bits:
        para += " " + "，".join(macro_bits) + "。"

    ce = l7.get("calendar_effects") or {}
    cal_alerts: list[str] = []
    if ce.get("in_pre_fomc_drift_window"):
        days_to = ce.get("days_to_next_fomc")
        cal_alerts.append(
            f"**今天距下次 FOMC 会议只剩 {days_to} 天，正好落在 pre-FOMC drift window**"
            f"（Lucca-Moench 2015 文献观察到此窗口股市有正漂移；商品 ETF 相关性弱）"
        )
    elif ce.get("in_fomc_blackout_window"):
        cal_alerts.append("**今天处于 FOMC blackout window**（前后 2 个交易日内）")
    elif ce.get("days_to_next_fomc") is not None:
        cal_alerts.append(f"距下次 FOMC 还有 {ce['days_to_next_fomc']} 天")

    if ce.get("in_turn_of_month"):
        cal_alerts.append("月末 turn-of-month 区间（Lakonishok-Smidt 1988）")
    if ce.get("month_end_window_dressing"):
        cal_alerts.append("月末 window-dressing 区间")

    if cal_alerts:
        para += " 时间窗口：" + "；".join(cal_alerts) + "。"

    return f"{title}\n{para}"


def _para_quant_overview(dc: dict) -> str:
    l2 = dc.get("quant_posture") or {}
    l1 = dc.get("watchlist_origin") or {}
    title = "**量化全景**"

    if not l2.get("available"):
        return f"{title}\n暂无 SignalRecord 数据。"

    if not l2.get("ticker_row_present"):
        league_n = l2.get("league_n", 0)
        return (
            f"{title}\n"
            f"该 ticker 不在 universe 内 — 当前 universe 共 {league_n} 个标的可参考。"
        )

    composite = l2.get("composite_score")
    composite_str = f"{composite:.0f}/100" if isinstance(composite, (int, float)) else "—"

    bits: list[str] = [f"当前综合评分 **{composite_str}**"]

    if isinstance(composite, (int, float)):
        if composite < 60:
            bits.append("**低于典型入场阈值 60，信号边际偏弱**")
        elif composite >= 75:
            bits.append("高于典型阈值 60，信号偏强")
        else:
            bits.append("略高于典型阈值 60，正常入场区间")

    ts = l2.get("tsmom_signal")
    if ts is not None:
        ts_zh = {1: "+1（上行趋势）", -1: "-1（下行趋势）", 0: "0（无趋势）"}.get(
            int(ts) if isinstance(ts, (int, float)) else None, str(ts)
        )
        bits.append(f"TSMOM 信号 {ts_zh}")

    direction = (l1.get("direction") or "").lower() if l1.get("available") else ""
    if ts is not None and direction:
        try:
            if int(ts) == 1 and direction == "long":
                bits.append("方向与 TSMOM 一致")
            elif int(ts) == -1 and direction == "short":
                bits.append("方向与 TSMOM 一致")
            elif int(ts) != 0 and direction in ("long", "short"):
                bits.append(f"**注意：建议方向 {direction} 与 TSMOM ({int(ts)}) 不一致**")
        except Exception:
            pass

    csmom_rank = l2.get("csmom_rank")
    if csmom_rank is not None:
        try:
            v = float(csmom_rank)
            pct = v * 100 if v <= 1.0 else v
            bits.append(f"CSMOM rank {pct:.0f}%")
        except Exception:
            pass

    league = l2.get("league_table") or []
    if league:
        target = next((r for r in league if r.get("highlight")), None)
        if target:
            bits.append(f"在 {len(league)} 个标的中排第 {target.get('rank')} 名")

    gate = l2.get("gate_status")
    if gate:
        bits.append(f"gate 状态 {gate}")

    return f"{title}\n{'，'.join(bits)}。"


def _para_approve_impact(dc: dict) -> str:
    l4 = dc.get("portfolio_posture") or {}
    l7 = dc.get("forward_preview") or {}
    title = "**批准影响**"

    if not (l4.get("available") and l7.get("available")):
        return f"{title}\n组合 / 后果数据不全。"

    n_pos = l4.get("portfolio_total_positions") or 0
    sec_count = l4.get("sector_position_count") or 0
    sec_now = l4.get("sector_current_exposure_pct") or 0.0
    sec_after = l4.get("post_approve_sector_exposure_pct") or 0.0

    ap = l7.get("approve_path") or {}
    pos_delta = ap.get("position_value_delta_usd") or 0.0
    cost = ap.get("estimated_trade_cost_usd") or 0.0
    nav = ap.get("nav_basis") or 0.0

    bits: list[str] = [
        f"你目前组合 **{n_pos} 个持仓**，该板块持仓 {sec_count} 个，敞口 {sec_now*100:.2f}%"
    ]
    bits.append(
        f"批准后板块敞口 → **{sec_after*100:.2f}%**（增量 {(sec_after-sec_now)*100:+.2f}pp）"
    )
    bits.append(f"约 **${pos_delta:,.0f}**（按当前 NAV ${nav:,.0f}）")
    bits.append(f"交易成本约 ${cost:,.2f}")

    hhi = l4.get("hhi_metrics") or {}
    if hhi:
        h_now = hhi.get("hhi_current") or 0.0
        h_after = hhi.get("hhi_post_approve") or 0.0
        interp = hhi.get("hhi_interpretation") or ""
        delta = h_after - h_now
        if abs(delta) < 0.005:
            bits.append(f"HHI 维持 {h_now:.4f}（{interp}，几乎无变化）")
        else:
            bits.append(f"HHI 从 {h_now:.4f} → {h_after:.4f}（Δ {delta:+.4f}）")

    return f"{title}\n{'，'.join(bits)}。"


def _para_history(dc: dict) -> str:
    l5 = dc.get("conditional_history") or {}
    title = "**历史能撑得住吗**"

    if l5.get("insufficient_data"):
        n = l5.get("n_obs", 0)
        min_n = l5.get("min_n_required", 5)
        return (
            f"{title}\n"
            f"该 sector × direction × regime 的历史样本数 **n={n} < {min_n}**，"
            "**统计上没有有意义的过去表现可以参考**——这是 paper trading 阶段早期的常态，"
            "随时间累积才能给出条件 hit rate。"
        )

    n = l5.get("n_obs")
    hr = (l5.get("hit_rate") or 0) * 100
    mar = (l5.get("mean_active_return") or 0) * 100
    mh = l5.get("median_holding_days")
    lookback = l5.get("lookback_days", 720)

    bits = [
        f"过去 {lookback} 天该 sector × direction × regime 共 **{n}** 条历史已验证决策"
    ]
    bits.append(f"hit rate **{hr:.0f}%**")
    if hr >= 60:
        bits.append("**历史上该组合表现良好**")
    elif hr <= 40:
        bits.append("**历史 hit rate 偏低，需谨慎**")
    bits.append(f"平均 active return {mar:+.2f}%")
    if mh is not None:
        bits.append(f"中位持有期 {mh:.0f} 天")

    return f"{title}\n{'，'.join(bits)}。"


def _para_risks(dc: dict) -> str:
    l1 = dc.get("watchlist_origin") or {}
    l2 = dc.get("quant_posture") or {}
    l3 = dc.get("regime_context") or {}
    l4 = dc.get("portfolio_posture") or {}
    l6 = dc.get("thesis_module") or {}
    l7 = dc.get("forward_preview") or {}
    title = "**我会重点看的风险**"

    risks: list[str] = []

    # Risk: composite below threshold
    composite = l2.get("composite_score") if l2.get("ticker_row_present") else None
    if isinstance(composite, (int, float)) and composite < 60:
        risks.append(
            f"composite **{composite:.0f}** 低于典型入场阈值 60 — 信号边际弱，false-positive 风险偏高"
        )

    # Risk: pre-FOMC / blackout
    ce = l7.get("calendar_effects") or {}
    if ce.get("in_pre_fomc_drift_window"):
        days_to = ce.get("days_to_next_fomc", 1)
        risks.append(
            f"今天是 pre-FOMC 窗口，**{days_to} 天内会议结果可能让 regime 切换**，"
            "TSMOM signal 可能在 24h 内反转"
        )
    elif ce.get("in_fomc_blackout_window"):
        risks.append("FOMC blackout window，会议前后市场不确定性高")

    # Risk: regime drifted
    if l3.get("regime_drifted"):
        risks.append(
            f"自 watchlist 创建以来 regime 已从 "
            f"{l3.get('regime_at_creation','—')} 漂移到 {l3.get('regime_label','—')}，"
            "入场假设可能已失效"
        )

    # Risk: extreme filtered probability
    p_on = l3.get("p_risk_on") or 0.0
    p_off = l3.get("p_risk_off") or 0.0
    p_tr = l3.get("p_transition") or 0.0
    if max(p_on, p_off) > 0.95:
        regime = "risk-on" if p_on > p_off else "risk-off"
        risks.append(
            f"{regime} regime filtered probability >0.95 是极端读数，"
            "短期内反转概率非零"
        )
    elif p_tr > 0.5:
        risks.append(
            f"P(transition)={p_tr:.2f} > 0.5，制度本身处于不确定状态"
        )

    # Risk: macro yield curve inverted
    macro = l3.get("macro_snapshot") or {}
    if macro.get("yield_curve_inverted"):
        risks.append("yield curve 倒挂——历史 12-18 月衰退信号")

    # Risk: HHI concentration
    hhi = l4.get("hhi_metrics") or {}
    if hhi.get("hhi_interpretation") == "highly_concentrated":
        risks.append(
            f"组合 HHI {hhi.get('hhi_current',0):.4f} 显示 highly concentrated — "
            "加仓会进一步放大集中度"
        )

    # Risk: portfolio in deep drawdown
    dd = l4.get("drawdown_metrics") or {}
    if dd.get("available"):
        cur_dd = dd.get("current_drawdown_pct") or 0.0
        if cur_dd < -0.10:
            risks.append(
                f"组合当前回撤 {cur_dd*100:+.2f}%，underwater {dd.get('underwater_days',0)} 天 — "
                "加仓时机谨慎"
            )

    # Risk: direction vs TSMOM mismatch
    direction = (l1.get("direction") or "").lower() if l1.get("available") else ""
    ts = l2.get("tsmom_signal")
    if ts is not None and direction:
        try:
            if (int(ts) == 1 and direction == "short") or \
               (int(ts) == -1 and direction == "long"):
                risks.append(
                    f"建议方向 ({direction}) 与 TSMOM 信号 ({int(ts)}) 不一致"
                )
        except Exception:
            pass

    # Fallback to L6 primary_risk if no rule-based risks fired
    if not risks:
        primary = l6.get("primary_risk")
        if primary:
            risks.append(primary)
        else:
            risks.append("无显著风险信号识别（rule-based 检测）")

    bullets = "\n".join(f"- {r}" for r in risks)
    return f"{title}\n{bullets}"


def _calendar_effects_today(d: datetime.date) -> dict:
    """EXT-3: FOMC / turn-of-month / earnings flags."""
    next_fomc = next((x for x in _FOMC_DATES_2024_2026 if x >= d), None)
    days_to_next_fomc = (next_fomc - d).days if next_fomc else None

    in_fomc_blackout_window = (
        next_fomc is not None and 0 <= (next_fomc - d).days <= 2
    )
    in_pre_fomc_drift_window = (
        next_fomc is not None and (next_fomc - d).days == 1
    )

    # Turn-of-month: last 4 trading days (≈ last 6 calendar days) or
    # first 3 trading days of next month (≈ first 5 calendar)
    import calendar
    last_day_of_month = calendar.monthrange(d.year, d.month)[1]
    in_turn_of_month = (
        (last_day_of_month - d.day) <= 5
        or d.day <= 4
    )

    # Earnings blackout — best-effort. We don't pull yfinance here (latency);
    # set False as the safe default and let UI explain "not implemented in v1".
    in_earnings_blackout = False

    month_end_window_dressing = (last_day_of_month - d.day) <= 2

    return {
        "as_of_date":                _date_to_str(d),
        "next_fomc_date":            _date_to_str(next_fomc),
        "days_to_next_fomc":         days_to_next_fomc,
        "in_fomc_blackout_window":   bool(in_fomc_blackout_window),
        "in_pre_fomc_drift_window":  bool(in_pre_fomc_drift_window),
        "in_turn_of_month":          bool(in_turn_of_month),
        "in_earnings_blackout":      in_earnings_blackout,
        "earnings_data_note":
            "earnings blackout flag not populated in v1 "
            "(requires yfinance per-ticker fetch; deferred to v2)",
        "month_end_window_dressing": bool(month_end_window_dressing),
        "academic_refs":
            "Lucca-Moench 2015 pre-FOMC drift; Lakonishok-Smidt 1988 turn-of-month",
    }
