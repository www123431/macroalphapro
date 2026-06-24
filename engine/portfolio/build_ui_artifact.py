"""
engine/portfolio/build_ui_artifact.py — Sub-phase A1 (2026-05-15 evening
UI architecture: pre-compute artifact pattern).

Why: per user feedback page-switch performance via live DB on every Streamlit
re-run is the固有 bottleneck. Architectural fix is the institutional "lambda
architecture" / "materialized view" pattern:

  Daily batch (06:00 SGT) → write artifact JSON → UI pages read artifact

This module is the WRITE side. Sub-phase A2 will adapt pages to read artifact
first, falling back to live DB if missing.

Doctrine compliance:
  - 0-LLM-in-DECISION: this module is pure read-side data extraction; LLM-free
  - LLM-risk-side rule: not affected (no LLM in artifact build)
  - 7-agent ceiling: not affected (infrastructure, not agent)
  - Hash-locked spec: not affected (we only READ specs, never write)

Artifact schema v1 (data/ui_artifact/<YYYY-MM-DD>.json):

  _meta:                build metadata (date, version, ts)
  book_snapshot:        Brief §1 KPIs (gross/net/n_tickers/active_strategies)
  strategy_states:      per-strategy latest row (6 strategies post Spec 80 2026-05-28)
  positions:            per-strategy per-ticker drill (latest snapshot)
  nav_timeseries:       forward NAV series (paper-trade window)
  trade_log_recent:     last 30 days Sprint H trade rows
  pending_approvals:    Tier 3 queue (all OPEN)
  cb_state:             circuit-breaker evaluation
  spec_registry_summary: spec counts + production-locked subset
  capability_evidence:  docs/capability_evidence/ file index
  rho_sentinel:         correlation_sentinel latest output

Size: typical ~150-300KB JSON. Compresses well (~50-80KB gzip).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_REPO_ROOT          = Path(__file__).resolve().parent.parent.parent
_ARTIFACT_DIR       = _REPO_ROOT / "data" / "ui_artifact"
_CAPABILITY_EV_DIR  = _REPO_ROOT / "docs" / "capability_evidence"
_REPLAY_VERDICT_PATH = _REPO_ROOT / "data" / "portfolio_replay" / "v1_combined_replay_verdict.json"

ARTIFACT_SCHEMA_VERSION = 2

# Brief-extended sections added in v2 (2026-05-16, Phase 2A.2a) for Dash
# 1:1 port of original Streamlit pages/executive_brief.py. v2 stays
# backward-compatible with v1 readers — new sections are additive only.


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_ui_artifact(
    as_of_date:    Optional[datetime.date] = None,
    output_path:   Optional[Path]          = None,
    forward_lookback_days: int             = 365,
    trade_log_lookback_days: int           = 30,
) -> Path:
    """Build a single JSON artifact containing everything UI pages need.

    Args:
        as_of_date:   build for this date (default = today UTC)
        output_path:  override default data/ui_artifact/<date>.json
        forward_lookback_days: how many forward-paper-trade days to include
        trade_log_lookback_days: how many days of Sprint H trade rows to include

    Returns:
        Path to written artifact file.

    Idempotent: re-running for same date overwrites. Atomic via tmp-then-rename.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today()

    if output_path is None:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _ARTIFACT_DIR / f"{as_of_date.isoformat()}.json"

    artifact: dict[str, Any] = {
        "_meta": {
            "as_of_date":              as_of_date.isoformat(),
            "build_ts_utc":            datetime.datetime.utcnow().isoformat() + "Z",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "build_version":           "v18",
            "build_source":            "engine.portfolio.build_ui_artifact.build_ui_artifact",
        },
    }

    # Build each section best-effort; one section failure must not break others
    for section_name, builder in [
        # ── v1 sections (kept) ──────────────────────────────────────────────
        ("book_snapshot",          _build_book_snapshot),
        ("sleeve_attribution",     _build_sleeve_attribution),
        ("strategy_states",        _build_strategy_states),
        ("positions",              _build_positions),
        ("nav_timeseries",         lambda d: _build_nav_timeseries(d, lookback_days=forward_lookback_days)),
        ("trade_log_recent",       lambda d: _build_trade_log_recent(d, lookback_days=trade_log_lookback_days)),
        ("pending_approvals",      _build_pending_approvals),
        ("cb_state",               _build_cb_state),
        ("spec_registry_summary",  _build_spec_registry_summary),
        ("capability_evidence",    _build_capability_evidence_index),
        ("rho_sentinel",           _build_rho_sentinel),
        ("backtest_summary",       _build_backtest_summary),
        # ── v2 Brief-extended sections (2026-05-16 Phase 2A.2a) ─────────────
        ("regime",                 _build_regime),
        ("today_actions",          _build_today_actions),
        ("attention_alert",        _build_attention_alert),
        ("current_dd",             _build_current_dd),
        ("var_overlay",            _build_var_overlay),
        ("tier1_health",           _build_tier1_health),
        ("tier2_health",           _build_tier2_health),
        ("baseline_rho",           _build_baseline_rho),
        ("top_conviction",         _build_top_conviction),
        # ── L1 deployment surface for 4-leg carry sleeve (spec 77 §10, 2026-05-28) ──
        ("carry_book_status",      _build_carry_book_status),
    ]:
        try:
            artifact[section_name] = builder(as_of_date)
        except Exception as exc:
            logger.warning("build_ui_artifact: section %s failed: %s", section_name, exc)
            artifact[section_name] = {"_error": str(exc)[:200]}

    # Atomic write: temp → rename
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(output_path)

    size_kb = output_path.stat().st_size // 1024
    logger.info("build_ui_artifact: wrote %s (%d KB)", output_path.name, size_kb)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_book_snapshot(as_of_date: datetime.date) -> dict:
    """Brief §1 — book gross/net/n_tickers/n_strategies."""
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.portfolio.paper_trade_combined import (
        PAPER_TRADE_SLEEVE_ALLOCATION, LEVERAGE_FACTOR, STRATEGY_DISPLAY_META,
    )

    PAPER_LIVE_START = datetime.date(2026, 5, 13)
    SLEEVE_TARGET = dict(PAPER_TRADE_SLEEVE_ALLOCATION)
    INTRA_FACTOR = {
        s: (m["sleeve_id"], m["intra_sleeve_w"])
        for s, m in STRATEGY_DISPLAY_META.items()
    }

    out = {
        "gross":              0.0,
        "net":                0.0,
        "n_tickers":          0,
        "snapshot_date":      None,
        "active_strategies":  [],
        "n_strategies":       0,
        "forward_day":        None,
    }
    with SessionFactory() as s:
        latest_date = (s.query(PaperTradeStrategyLog.date)
                        .order_by(PaperTradeStrategyLog.date.desc())
                        .first())
        if not latest_date:
            return out
        latest_date = latest_date[0]
        out["snapshot_date"] = latest_date.isoformat()
        out["forward_day"]   = (latest_date - PAPER_LIVE_START).days
        rows = s.query(PaperTradeStrategyLog).filter_by(date=latest_date).all()
        by_ticker: dict[str, float] = {}
        for r in rows:
            if r.status != "OK":
                continue
            out["active_strategies"].append(r.strategy_name)
            sleeve, intra_default = INTRA_FACTOR.get(r.strategy_name, ("", 1.0))
            sleeve_target = SLEEVE_TARGET.get(sleeve, 0)
            book_factor = sleeve_target * float(r.intra_sleeve_weight or intra_default) * LEVERAGE_FACTOR
            try:
                positions = json.loads(r.positions_json) if r.positions_json else {}
            except Exception:
                positions = {}
            for ticker, intra_w in positions.items():
                try:
                    bw = book_factor * float(intra_w)
                except Exception:
                    continue
                by_ticker[ticker] = by_ticker.get(ticker, 0.0) + bw
        out["n_strategies"] = len(rows)
        out["n_tickers"]    = len(by_ticker)
        out["gross"]        = round(sum(abs(v) for v in by_ticker.values()), 6)
        out["net"]          = round(sum(by_ticker.values()), 6)
    return out


def _build_sleeve_attribution(as_of_date: datetime.date) -> dict:
    """Per-sleeve daily P&L attribution for the latest paper-trade date.

    Reads PaperTradeStrategyLog rows + book_weight per strategy and aggregates
    per-strategy daily net return × book_weight up to the SLEEVE level the UI
    Book page expects (one row per displayed strategy on /book Sleeve
    Attribution panel).

    The /book frontend expects `sleeve_attribution` as a flat dict:
        { strategy_name: contribution_bps_of_book, ... }
    where contribution = book_weight × daily_net_return × 10000.

    Returns {} when the latest log date has no usable rows so the UI
    falls back to "No sleeve attribution in the latest artifact." per the
    existing render path.
    """
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.portfolio.paper_trade_combined import (
        STRATEGY_DISPLAY_META, get_strategy_book_weight,
    )

    out: dict[str, float] = {}
    with SessionFactory() as s:
        latest_date = (s.query(PaperTradeStrategyLog.date)
                        .order_by(PaperTradeStrategyLog.date.desc())
                        .first())
        if not latest_date:
            return out
        latest_date = latest_date[0]
        rows = s.query(PaperTradeStrategyLog).filter_by(date=latest_date).all()
        for r in rows:
            if r.daily_net_return is None:
                continue
            try:
                bw = float(get_strategy_book_weight(r.strategy_name))
            except Exception:
                # Fallback to STRATEGY_DISPLAY_META.book_w if available
                meta = STRATEGY_DISPLAY_META.get(r.strategy_name, {})
                bw = float(meta.get("book_w", 0.0))
            try:
                ret = float(r.daily_net_return)
            except Exception:
                continue
            # Contribution to book NAV in basis points (1bp = 0.0001)
            contribution_bps = round(bw * ret * 10000.0, 4)
            out[r.strategy_name] = contribution_bps
    return out


def _build_strategy_states(as_of_date: datetime.date) -> list[dict]:
    """Per-strategy latest row state (6 strategies post Spec 80 2026-05-28)."""
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.portfolio.paper_trade_combined import (
        STRATEGY_DISPLAY_META, STRATEGY_ORDER, get_strategy_book_weight,
    )

    out: list[dict] = []
    with SessionFactory() as s:
        latest_date = (s.query(PaperTradeStrategyLog.date)
                        .order_by(PaperTradeStrategyLog.date.desc())
                        .first())
        if latest_date is None:
            return out
        rows = s.query(PaperTradeStrategyLog).filter_by(date=latest_date[0]).all()
        rmap = {r.strategy_name: r for r in rows}
        for sname in STRATEGY_ORDER:
            r = rmap.get(sname)
            meta = STRATEGY_DISPLAY_META[sname]
            entry = {
                "strategy_name":        sname,
                "display_short":        meta["display_short"],
                "spec_id":              meta["spec_id"],
                "spec_hash_short":      meta["spec_hash_short"],
                "sleeve_id":            meta["sleeve_id"],
                "intra_sleeve_w":       meta["intra_sleeve_w"],
                "book_weight":          round(get_strategy_book_weight(sname), 6),
                "color":                meta["color"],
                "doctrine":             meta["doctrine"],
                "universe":             meta["universe"],
                "rebalance_days":       meta["rebalance_days"],
            }
            if r is not None:
                entry.update({
                    "date":                 r.date.isoformat() if r.date else None,
                    "status":               r.status,
                    "n_positions":          int(r.n_positions or 0),
                    "intra_sleeve_weight":  float(r.intra_sleeve_weight or 0),
                    "is_rebalance_day":     bool(r.is_rebalance_day) if r.is_rebalance_day is not None else None,
                    "daily_gross_return":   float(r.daily_gross_return) if r.daily_gross_return is not None else None,
                    "tc_drag_today":        float(r.tc_drag_today or 0) if hasattr(r, "tc_drag_today") else None,
                })
            else:
                entry["status"] = "no_row"
            out.append(entry)
    return out


def _build_positions(as_of_date: datetime.date) -> list[dict]:
    """Per-strategy per-ticker positions drill (latest snapshot)."""
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    out: list[dict] = []
    with SessionFactory() as s:
        latest_date = (s.query(PaperTradeStrategyLog.date)
                        .order_by(PaperTradeStrategyLog.date.desc())
                        .first())
        if latest_date is None:
            return out
        rows = s.query(PaperTradeStrategyLog).filter_by(date=latest_date[0]).all()
        for r in rows:
            if not r.positions_json:
                continue
            try:
                positions = json.loads(r.positions_json)
            except Exception:
                continue
            for ticker, intra_w in positions.items():
                out.append({
                    "strategy_name":  r.strategy_name,
                    "ticker":         str(ticker).upper(),
                    "intra_weight":   float(intra_w),
                    "snapshot_date":  r.date.isoformat() if r.date else None,
                })
    return out


def _build_nav_timeseries(as_of_date: datetime.date, lookback_days: int = 365) -> list[dict]:
    """Forward NAV series — combined book return per day."""
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.portfolio.paper_trade_combined import (
        STRATEGY_ORDER, get_strategy_book_weight,
    )

    cutoff = as_of_date - datetime.timedelta(days=lookback_days)
    by_date: dict[datetime.date, dict[str, float]] = {}
    with SessionFactory() as s:
        rows = (s.query(PaperTradeStrategyLog)
                 .filter(PaperTradeStrategyLog.date >= cutoff)
                 .order_by(PaperTradeStrategyLog.date.asc())
                 .all())
    for r in rows:
        if r.daily_gross_return is None:
            continue
        by_date.setdefault(r.date, {})[r.strategy_name] = float(r.daily_gross_return)

    book_weights = {s: get_strategy_book_weight(s) for s in STRATEGY_ORDER}
    out: list[dict] = []
    nav = 1.0
    for d in sorted(by_date):
        per_strat = by_date[d]
        combined = sum(book_weights.get(s, 0) * per_strat.get(s, 0.0) for s in STRATEGY_ORDER)
        nav = nav * (1.0 + combined)
        out.append({
            "date":                d.isoformat(),
            "combined_return":     round(combined, 8),
            "cumulative_nav":      round(nav, 8),
            "per_strategy_return": {s: round(v, 8) for s, v in per_strat.items()},
        })
    return out


def _build_trade_log_recent(as_of_date: datetime.date, lookback_days: int = 30) -> list[dict]:
    """Recent Sprint H trade attribution rows."""
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeTradeLog

    cutoff = as_of_date - datetime.timedelta(days=lookback_days)
    out: list[dict] = []
    with SessionFactory() as s:
        rows = (s.query(PaperTradeTradeLog)
                 .filter(PaperTradeTradeLog.date >= cutoff)
                 .order_by(PaperTradeTradeLog.date.desc())
                 .all())
    for r in rows:
        out.append({
            "date":                r.date.isoformat() if r.date else None,
            "trade_id":            r.trade_id,
            "strategy_name":       r.strategy_name,
            "spec_id":             r.spec_id,
            "spec_hash_short":     r.spec_hash_short,
            "sleeve_id":           r.sleeve_id,
            "ticker":              r.ticker,
            "side":                r.side,
            "weight":              float(r.weight) if r.weight is not None else None,
            "signal_value":        float(r.signal_value) if r.signal_value is not None else None,
            "event_trigger":       r.event_trigger,
            "expected_horizon_days": r.expected_horizon_days,
            "is_rebalance_day":    bool(r.is_rebalance_day) if r.is_rebalance_day is not None else None,
        })
    return out


def _build_pending_approvals(as_of_date: datetime.date) -> list[dict]:
    """Tier 3 PendingApproval queue (all OPEN)."""
    from engine.memory import SessionFactory
    from engine.db_models import PendingApproval

    out: list[dict] = []
    with SessionFactory() as s:
        rows = (s.query(PendingApproval)
                 .filter(PendingApproval.status == "pending")
                 .order_by(PendingApproval.created_at.desc())
                 .all())
    for r in rows:
        # PendingApproval doesn't have 'summary' column — use approval_type as
        # a label hint (post_hoc_note when populated is a fuller note)
        label = (r.post_hoc_note or "")[:200] if r.post_hoc_note else (r.approval_type or "")[:200]
        out.append({
            "id":             r.id,
            "approval_type":  r.approval_type,
            "approval_class": r.approval_class,
            "priority":       r.priority,
            "status":         r.status,
            "label":          label,
            "created_at":     r.created_at.isoformat() if r.created_at else None,
            "deadline":       r.approval_deadline.isoformat() if r.approval_deadline else None,
        })
    return out


def _build_cb_state(as_of_date: datetime.date) -> dict:
    """Circuit-breaker evaluation."""
    try:
        from engine.circuit_breaker import evaluate as _cb_eval
        _cb_state = _cb_eval()
        return {
            "level":  _cb_state.level,
            "reason": _cb_state.reason or "",
        }
    except Exception as exc:
        return {"level": "unknown", "reason": f"evaluate failed: {exc}"}


def _build_spec_registry_summary(as_of_date: datetime.date) -> dict:
    """Spec registry summary: total count, by status, production-locked subset."""
    from engine.memory import SessionFactory
    from engine.db_models import SpecRegistry

    out = {"total": 0, "by_status": {}, "production_locked": []}
    with SessionFactory() as s:
        rows = s.query(SpecRegistry).all()
    out["total"] = len(rows)
    for r in rows:
        st = (r.status or "unknown").lower()
        out["by_status"][st] = out["by_status"].get(st, 0) + 1
        if (r.status or "").lower() in ("production_locked", "production"):
            out["production_locked"].append({
                "id":         r.id,
                "spec_hash_short": (r.spec_hash or "")[:8],
                "title":      r.title or "",
            })
    return out


def _build_carry_book_status(as_of_date: datetime.date) -> dict:
    """L1 deployment surface for 4-leg cross-asset carry sleeve (spec 77 §10).

    Model-tracked at 30% of book risk. Daily-marked from cached futures curves
    (cmdty + FX + US-rates + G10-rates-XC). NOT in PAPER_TRADE_SLEEVE_ALLOCATION
    yet — model NAV computed via engine.portfolio.combined_book.build_combined_book
    but no real paper orders generated. Deployment gates G1-G4 enumerate the path
    to full paper / real-fill / real-capital deployment.
    """
    try:
        from engine.portfolio.combined_book import (
            build_combined_book, build_carry_book, voltarget, book_stats,
            DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_TARGET_VOL,
        )
        carry_only = voltarget(build_carry_book(), DEFAULT_TARGET_VOL)
        combined = build_combined_book(carry_risk_weight=DEFAULT_CARRY_RISK_WEIGHT)
        equity_only = build_combined_book(carry_risk_weight=0.0)
        cs = book_stats(combined)
        es = book_stats(equity_only)
        co = book_stats(carry_only)

        return {
            "spec_id":             77,
            "spec_hash_short":     "1726cf18",
            "construction":        "4-leg risk-parity carry (cmdty + FX + US-rates + G10-rates-XC)",
            "amendments":          ["§9 dedupe bug fix + US rates leg (2026-05-28)",
                                    "§10 G10 cross-country bond futures (2026-05-28)"],
            "deployment_status":   "MODEL_MARKED_NOT_PAPER_ORDERS",
            "deployment_level":    "L1_visibility",
            "carry_risk_weight":   DEFAULT_CARRY_RISK_WEIGHT,
            "honest_deploy_sharpe_oos": 0.83,
            "metrics_validation_grade": {
                "carry_only_sharpe":  round(co.get("sharpe", float("nan")), 3),
                "carry_only_maxdd":   round(co.get("maxdd", float("nan")), 4),
                "equity_only_sharpe": round(es.get("sharpe", float("nan")), 3),
                "equity_only_maxdd":  round(es.get("maxdd", float("nan")), 4),
                "combined_sharpe":    round(cs.get("sharpe", float("nan")), 3),
                "combined_maxdd":     round(cs.get("maxdd", float("nan")), 4),
                "combined_n_months":  cs.get("n"),
            },
            "improvement_over_2leg_predecessor": {
                "carry_standalone_sharpe": [0.66, round(co.get("sharpe", 0.0), 2)],
                "combined_book_sharpe":    [0.96, round(cs.get("sharpe", 0.0), 2)],
                "combined_book_maxdd":     [-0.0822, round(cs.get("maxdd", 0.0), 4)],
            },
            "deployment_gates_pending": [
                {"id": "G1", "desc": "IB futures broker integration for real fills",
                 "owner": "user (account signup)", "blocking": "real fills only"},
                {"id": "G2", "desc": "RM-wire carry as new sleeve_class with futures-style caps",
                 "owner": "engineering", "blocking": "L2 paper rebalance"},
                {"id": "G3", "desc": "Add to PAPER_TRADE_SLEEVE_ALLOCATION (5-sleeve × 0.7 + carry @ 0.3)",
                 "owner": "engineering + Devil's Advocate review", "blocking": "L2 paper rebalance"},
                {"id": "G4", "desc": "24mo forward OOS (2028-05)",
                 "owner": "time", "blocking": "real capital only"},
            ],
            "doctrine_compliance": {
                "strict_gate_passed": True,
                "honest_n_trials":   20,
                "deflated_sr":       1.0000,
                "ff5umd_alpha_t":    0.057,
                "oos_sharpe":        0.828,
                "third_of_third_sharpe": 0.840,
            },
        }
    except Exception as exc:
        return {
            "deployment_status": "BUILD_FAILED",
            "_error":            str(exc)[:200],
        }


def _build_capability_evidence_index(as_of_date: datetime.date) -> list[dict]:
    """List all capability_evidence/*.md files with metadata."""
    out: list[dict] = []
    if not _CAPABILITY_EV_DIR.exists():
        return out
    for path in sorted(_CAPABILITY_EV_DIR.glob("*.md")):
        try:
            stat = path.stat()
            out.append({
                "filename":   path.name,
                "rel_path":   str(path.relative_to(_REPO_ROOT)).replace("\\", "/"),
                "size_kb":    stat.st_size // 1024,
                "mtime_iso":  datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except Exception:
            continue
    return out


def _build_rho_sentinel(as_of_date: datetime.date) -> dict:
    """Correlation sentinel latest output."""
    try:
        from engine.portfolio.correlation_sentinel import run_correlation_sentinel
        result = run_correlation_sentinel(as_of=as_of_date)
        return {
            "as_of":         result.as_of.isoformat() if result.as_of else None,
            "trailing_days": result.trailing_days if hasattr(result, "trailing_days") else None,
            "correlations":  [
                {
                    "pair_a":         c.pair_a,
                    "pair_b":         c.pair_b,
                    "rho_trailing":   float(c.rho_trailing) if c.rho_trailing is not None else None,
                    "rho_baseline":   float(c.rho_baseline) if c.rho_baseline is not None else None,
                    "delta":          float(c.delta) if c.delta is not None else None,
                    "severity":       c.severity,
                }
                for c in result.correlations
            ],
        }
    except Exception as exc:
        return {"_error": str(exc)[:200]}


def _build_backtest_summary(as_of_date: datetime.date) -> dict:
    """Read backtest verdict JSON if exists."""
    if not _REPLAY_VERDICT_PATH.exists():
        return {"_missing": True}
    try:
        v = json.loads(_REPLAY_VERDICT_PATH.read_text(encoding="utf-8"))
        return v.get("combined_metrics", {}) or v
    except Exception as exc:
        return {"_error": str(exc)[:200]}


# ─────────────────────────────────────────────────────────────────────────────
# v2 section builders (2026-05-16, Phase 2A.2a — Brief 1:1 port prerequisites)
# ─────────────────────────────────────────────────────────────────────────────

def _combined_daily_returns(as_of_date: datetime.date) -> list[tuple[datetime.date, float]]:
    """Helper — daily combined returns ordered ascending. Used by _build_current_dd
    and _build_var_overlay. Single source of truth for forward-window NAV math.
    """
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.portfolio.paper_trade_combined import (
        STRATEGY_ORDER, get_strategy_book_weight,
    )

    book_w = {s: get_strategy_book_weight(s) for s in STRATEGY_ORDER}
    by_date: dict[datetime.date, dict[str, float]] = {}
    with SessionFactory() as s:
        rows = (s.query(PaperTradeStrategyLog)
                 .order_by(PaperTradeStrategyLog.date.asc())
                 .all())
    for r in rows:
        if r.daily_gross_return is None:
            continue
        by_date.setdefault(r.date, {})[r.strategy_name] = float(r.daily_gross_return)
    out: list[tuple[datetime.date, float]] = []
    for d in sorted(by_date):
        per = by_date[d]
        combined = sum(book_w.get(s, 0.0) * per.get(s, 0.0) for s in STRATEGY_ORDER)
        out.append((d, combined))
    return out


def _build_regime(as_of_date: datetime.date) -> dict:
    """Latest RegimeSnapshot — used by Brief identity strip."""
    try:
        from engine.memory import SessionFactory, RegimeSnapshot
    except Exception:
        return {"regime": None, "_unavailable": "RegimeSnapshot not exported"}
    with SessionFactory() as s:
        row = (s.query(RegimeSnapshot)
                .order_by(RegimeSnapshot.as_of_date.desc())
                .first())
    if row is None:
        return {"regime": None}
    return {
        "regime":       row.regime,
        "p_risk_on":    float(row.p_risk_on)  if row.p_risk_on  is not None else None,
        "p_risk_off":   float(row.p_risk_off) if row.p_risk_off is not None else None,
        "vix":          float(row.vix)        if row.vix        is not None else None,
        "as_of_date":   row.as_of_date.isoformat() if row.as_of_date else None,
        "days_stale":   (as_of_date - row.as_of_date).days if row.as_of_date else None,
    }


def _build_today_actions(as_of_date: datetime.date) -> dict:
    """Counters for Brief §3 Today's Actions:
       Sprint H new rows today / ETF caps active / 365d cost / Watchdog severity /
       pending approvals total + critical."""
    from sqlalchemy import func
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeTradeLog, PendingApproval

    out = {
        "sprint_h_today":      0,
        "etf_caps_active":     0,
        "etf_cost_365d":       0.0,
        "watchdog_severity":   None,
        "watchdog_n_findings": 0,
        "pa_pending":          0,
        "pa_critical":         0,
    }
    with SessionFactory() as s:
        out["sprint_h_today"] = (s.query(func.count(PaperTradeTradeLog.trade_id))
                                  .filter(func.date(PaperTradeTradeLog.date) == as_of_date)
                                  .scalar()) or 0
        out["pa_pending"]     = (s.query(func.count(PendingApproval.id))
                                  .filter(PendingApproval.status == "pending")
                                  .scalar()) or 0
        out["pa_critical"]    = (s.query(func.count(PendingApproval.id))
                                  .filter(PendingApproval.status == "pending",
                                          PendingApproval.priority == "critical")
                                  .scalar()) or 0
    try:
        from engine.etf_holdings_risk_monitor import _load_cap_state, get_cost_status
        out["etf_caps_active"] = len(_load_cap_state())
        out["etf_cost_365d"]   = float(get_cost_status().get("trailing_365d_total_usd", 0.0))
    except Exception:
        pass
    # Watchdog severity from most recent _run.json
    try:
        wd_dir = _REPO_ROOT / "data" / "ops_watchdog"
        if wd_dir.exists():
            runs = sorted(wd_dir.glob("*_run.json"), reverse=True)
            if runs:
                payload = json.loads(runs[0].read_text(encoding="utf-8"))
                findings = payload.get("findings", []) or []
                out["watchdog_n_findings"] = len(findings)
                if findings:
                    sevs = {f.get("severity", "") for f in findings}
                    if {"HIGH", "CRITICAL", "SEVERE"} & sevs:
                        out["watchdog_severity"] = "HIGH"
                    elif "MID" in sevs:
                        out["watchdog_severity"] = "MID"
                    else:
                        out["watchdog_severity"] = "LOW"
                else:
                    out["watchdog_severity"] = "CLEAN"
    except Exception:
        pass
    return out


def _build_attention_alert(as_of_date: datetime.date) -> dict:
    """Single highest-priority alert across the system (Brief attention strip).
    Priority chain mirrors pages/executive_brief.py:462-569. Deterministic
    derivation from today_actions + rho_sentinel."""
    actions = _build_today_actions(as_of_date)
    sev = actions.get("watchdog_severity")
    nf  = actions.get("watchdog_n_findings", 0)

    if sev in ("HIGH", "CRITICAL", "SEVERE"):
        return {"level": "critical", "label": "WATCHDOG", "value": sev,
                "detail": f"{nf} findings · last 06:10 SGT", "dest": "/system"}
    if actions.get("pa_critical", 0) > 0:
        return {"level": "critical", "label": "PA CRITICAL",
                "value":  f"{actions['pa_critical']} awaiting",
                "detail": "Tier 3 supervisor sign-off required", "dest": "/approvals"}
    if actions.get("etf_caps_active", 0) > 0:
        return {"level": "warn", "label": "ETF CAP",
                "value":  f"{actions['etf_caps_active']} active",
                "detail": f"MAX_WEIGHT 25→15% per name · trailing 365d ${actions['etf_cost_365d']:.2f}",
                "dest":   "/risk"}
    # ρ sentinel CRITICAL on active pairs
    try:
        from engine.portfolio.correlation_sentinel import run_correlation_sentinel
        sent = run_correlation_sentinel(as_of=as_of_date)
        crit = [c for c in (sent.correlations or [])
                if getattr(c, "severity", "") == "CRITICAL"]
        if crit:
            c0 = crit[0]
            return {"level": "warn", "label": "ρ DRIFT", "value": "CRITICAL",
                    "detail": f"{c0.pair_a} × {c0.pair_b} ρ={float(c0.rho_trailing):+.2f} · diversification broken",
                    "dest":   "/risk"}
    except Exception:
        pass
    if sev == "MID":
        return {"level": "caution", "label": "WATCHDOG", "value": "MID",
                "detail": f"{nf} findings · review when convenient", "dest": "/system"}
    if actions.get("pa_pending", 0) > 0:
        return {"level": "caution", "label": "PA QUEUE",
                "value":  f"{actions['pa_pending']} pending",
                "detail": "all normal priority · no immediate action", "dest": "/approvals"}
    # ρ sentinel WARN
    try:
        from engine.portfolio.correlation_sentinel import run_correlation_sentinel
        sent = run_correlation_sentinel(as_of=as_of_date)
        warn = [c for c in (sent.correlations or [])
                if getattr(c, "severity", "") == "WARN"]
        if warn:
            c0 = warn[0]
            return {"level": "caution", "label": "ρ DRIFT", "value": "WARN",
                    "detail": f"{c0.pair_a} × {c0.pair_b} ρ={float(c0.rho_trailing):+.2f} · watch",
                    "dest":   "/risk"}
    except Exception:
        pass
    return {"level": "clean", "label": "CLEAN", "value": "all systems nominal",
            "detail": "next paper-trade cycle 06:00 SGT · next Watchdog 06:10 SGT",
            "dest":   None}


def _build_current_dd(as_of_date: datetime.date) -> dict:
    """Current drawdown = (NAV_today / running_peak) − 1. Forward-window only."""
    rets = _combined_daily_returns(as_of_date)
    if not rets:
        return {"dd_pct": None, "n_days": 0, "peak_date": None,
                "running_peak_nav": 1.0,
                "reason": "no daily_gross_return populated yet"}
    nav  = 1.0
    peak = 1.0
    peak_date = rets[0][0]
    for d, r in rets:
        nav *= (1.0 + r)
        if nav > peak:
            peak      = nav
            peak_date = d
    dd_pct = (nav / peak) - 1.0 if peak > 0 else 0.0
    return {
        "dd_pct":           round(dd_pct, 8),
        "n_days":           len(rets),
        "peak_date":        peak_date.isoformat(),
        "running_peak_nav": round(peak, 8),
        "current_nav":      round(nav, 8),
    }


def _build_var_overlay(as_of_date: datetime.date) -> dict:
    """1-week VaR / CVaR overlay. Forward window if ≥30w available, else backtest."""
    rets = _combined_daily_returns(as_of_date)
    n_fwd_weeks = len(rets) // 5
    try:
        import numpy as np
        import pandas as pd
        if n_fwd_weeks >= 30:
            ser = pd.Series([r for _, r in rets])
            ser = (1 + ser).rolling(5).apply(np.prod, raw=True) - 1
            ser = ser.dropna()
            source = f"forward · {len(ser)}w"
        else:
            p = _REPO_ROOT / "data" / "portfolio_replay" / "v1_combined_returns_weekly.parquet"
            if not p.exists():
                return {"var": None, "cvar": None, "source": "n/a",
                        "n_obs": 0, "divergence": False}
            ser = pd.read_parquet(p)["combined_return"]
            source = f"backtest · {len(ser)}w"
        if len(ser) < 30:
            return {"var": None, "cvar": None, "source": source,
                    "n_obs": int(len(ser)), "divergence": False}
        from engine.risk_metrics import compute_var_block
        vb = compute_var_block(ser, alpha=0.05)
        return {
            "var":        float(vb.historical),
            "cvar":       float(vb.es_historical),
            "source":     source,
            "n_obs":      int(len(ser)),
            "divergence": bool(vb.divergence_warning),
        }
    except Exception as exc:
        return {"var": None, "cvar": None, "source": "error",
                "n_obs": 0, "divergence": False, "_error": str(exc)[:200]}


def _build_tier1_health(as_of_date: datetime.date) -> list[dict]:
    """T1 production-critical: 5 strats + paper_trade + Sprint H + Watchdog + ρ Sentinel."""
    from sqlalchemy import func
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeStrategyLog, PaperTradeTradeLog
    from engine.portfolio.paper_trade_combined import (
        STRATEGY_ORDER, STRATEGY_DISPLAY_META,
    )

    items: list[dict] = []
    rmap: dict[str, Any] = {}
    latest_date: Optional[datetime.date] = None
    with SessionFactory() as s:
        row = (s.query(PaperTradeStrategyLog.date)
                .order_by(PaperTradeStrategyLog.date.desc()).first())
        if row:
            latest_date = row[0]
            for r in s.query(PaperTradeStrategyLog).filter_by(date=latest_date).all():
                rmap[r.strategy_name] = r
        n_sprint_h_total = (s.query(func.count(PaperTradeTradeLog.trade_id)).scalar()) or 0

    for sname in STRATEGY_ORDER:
        label = STRATEGY_DISPLAY_META[sname]["display_short"]
        r = rmap.get(sname)
        if r is None:
            items.append({"label": label, "status": "idle", "sub": "no row"})
        elif r.status == "OK":
            items.append({"label": label, "status": "ok", "sub": f"{r.n_positions}p"})
        elif r.status == "NO_SIGNAL":
            items.append({"label": label, "status": "idle", "sub": "no signal"})
        else:
            items.append({"label": label, "status": "alert", "sub": r.status})

    # paper_trade orchestrator
    if latest_date is None:
        items.append({"label": "paper_trade", "status": "idle", "sub": "never"})
    else:
        age_h = (datetime.datetime.now()
                 - datetime.datetime.combine(latest_date, datetime.time.min)
                ).total_seconds() / 3600.0
        if age_h <= 36:
            items.append({"label": "paper_trade", "status": "ok",   "sub": f"{int(age_h)}h"})
        else:
            items.append({"label": "paper_trade", "status": "warn", "sub": f"{int(age_h)}h stale"})

    items.append({
        "label":  "Sprint H",
        "status": "ok" if n_sprint_h_total > 0 else "idle",
        "sub":    f"{n_sprint_h_total} rows",
    })

    # Watchdog freshness
    try:
        wd_dir = _REPO_ROOT / "data" / "ops_watchdog"
        if wd_dir.exists():
            runs = sorted(wd_dir.glob("*_run.json"), reverse=True)
            if runs:
                age_h = (datetime.datetime.now()
                         - datetime.datetime.fromtimestamp(runs[0].stat().st_mtime)
                        ).total_seconds() / 3600.0
                if age_h <= 36:
                    items.append({"label": "Watchdog", "status": "ok",   "sub": f"{int(age_h)}h"})
                else:
                    items.append({"label": "Watchdog", "status": "warn", "sub": f"{int(age_h)}h stale"})
            else:
                items.append({"label": "Watchdog", "status": "idle", "sub": "never"})
        else:
            items.append({"label": "Watchdog", "status": "idle", "sub": "no dir"})
    except Exception:
        items.append({"label": "Watchdog", "status": "idle", "sub": "—"})

    # ρ Sentinel
    try:
        from engine.portfolio.correlation_sentinel import run_correlation_sentinel
        r = run_correlation_sentinel(as_of=as_of_date)
        sev = getattr(r, "severity", None)
        if sev == "CRITICAL":
            items.append({"label": "ρ Sentinel", "status": "alert", "sub": "CRITICAL"})
        elif sev == "WARN":
            items.append({"label": "ρ Sentinel", "status": "warn",  "sub": "WARN"})
        elif sev == "CLEAN":
            items.append({"label": "ρ Sentinel", "status": "ok",    "sub": "clean"})
        else:
            items.append({"label": "ρ Sentinel", "status": "idle",  "sub": "insuf data"})
    except Exception:
        items.append({"label": "ρ Sentinel", "status": "idle", "sub": "—"})

    return items


def _build_tier2_health(as_of_date: datetime.date) -> list[dict]:
    """T2 production-optional: ETF Holdings + others."""
    items: list[dict] = []
    try:
        from engine.etf_holdings_risk_monitor import (
            _load_cap_state, ETF_HOLDINGS_DEPLOYMENT_MODE, get_cost_status,
            _CAP_STATE_PATH,
        )
        caps  = _load_cap_state()
        cost  = get_cost_status()
        stale = False
        try:
            mtime_d = (datetime.datetime.now()
                       - datetime.datetime.fromtimestamp(_CAP_STATE_PATH.stat().st_mtime)
                      ).days
            stale = mtime_d > 35
        except Exception:
            mtime_d = None
        status = "warn" if stale else ("ok" if caps or cost.get("trailing_365d_total_usd") else "idle")
        sub = f"{len(caps)} caps · ${cost.get('trailing_365d_total_usd', 0):.2f}/365d · {ETF_HOLDINGS_DEPLOYMENT_MODE}"
        if stale:
            sub = f"{mtime_d}d stale · " + sub
        items.append({"label": "ETF Holdings", "status": status, "sub": sub})
    except Exception:
        items.append({"label": "ETF Holdings", "status": "idle", "sub": "—"})
    return items


def _build_baseline_rho(as_of_date: datetime.date) -> dict:
    """Read pairwise_correlation map from Sprint B replay verdict.
    Brief diversification heatmap uses this until forward ρ Sentinel unlocks."""
    if not _REPLAY_VERDICT_PATH.exists():
        return {"_missing": True, "pairwise_correlation": {}}
    try:
        v = json.loads(_REPLAY_VERDICT_PATH.read_text(encoding="utf-8"))
        pairwise = v.get("pairwise_correlation", {}) or {}
        return {
            "pairwise_correlation": {k: float(v) for k, v in pairwise.items()},
            "source":               "data/portfolio_replay/v1_combined_replay_verdict.json",
        }
    except Exception as exc:
        return {"_error": str(exc)[:200], "pairwise_correlation": {}}


def _build_top_conviction(as_of_date: datetime.date) -> dict:
    """Single highest |signal_value| Sprint H trade for latest persisted date."""
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradeTradeLog

    with SessionFactory() as s:
        latest = (s.query(PaperTradeTradeLog.date)
                   .order_by(PaperTradeTradeLog.date.desc()).first())
        if not latest:
            return {"status": "INSUF"}
        d_latest = latest[0]
        rows = (s.query(PaperTradeTradeLog)
                 .filter_by(date=d_latest)
                 .filter(PaperTradeTradeLog.signal_value.isnot(None))
                 .all())
    if not rows:
        return {"status": "INSUF"}
    top = max(rows, key=lambda r: abs(float(r.signal_value or 0.0)))
    return {
        "status":   "OK",
        "date":     d_latest.isoformat() if d_latest else None,
        "ticker":   top.ticker,
        "strategy": top.strategy_name,
        "signal":   float(top.signal_value),
        "side":     top.side,
        "weight":   float(top.weight) if top.weight is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _json_default(obj):
    """Custom JSON serializer for datetime + edge types."""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Public reader (Sub-phase A2 will use this in pages)
# ─────────────────────────────────────────────────────────────────────────────
def load_latest_artifact(
    artifact_dir:        Optional[Path] = None,
    max_age_hours:       float          = 26.0,
) -> Optional[dict]:
    """Load the most recent artifact within max_age_hours. Returns None if
    none found or all are stale.

    Pages use this in artifact-first read pattern; fall back to live DB if
    return is None.
    """
    if artifact_dir is None:
        artifact_dir = _ARTIFACT_DIR
    if not artifact_dir.exists():
        return None
    candidates = sorted(artifact_dir.glob("*.json"), reverse=True)
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
    for path in candidates:
        if path.name.endswith(".tmp"):
            continue
        try:
            stat = path.stat()
            if datetime.datetime.fromtimestamp(stat.st_mtime) < cutoff:
                continue
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


if __name__ == "__main__":
    # CLI entry — run as `py -3.11 -m engine.portfolio.build_ui_artifact`
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = build_ui_artifact()
    print(f"Artifact written: {path}", file=sys.stderr)
    size_kb = path.stat().st_size // 1024
    print(f"Size: {size_kb} KB", file=sys.stderr)
