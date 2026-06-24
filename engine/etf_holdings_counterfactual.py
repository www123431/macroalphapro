"""
engine/etf_holdings_counterfactual.py — Dual-track P&L counterfactual (Sprint Week 5).

Pre-registration: docs/spec_etf_holdings_llm_risk_monitor.md (id=49)
Spec section: §2.9 Counterfactual Tracking (deterministic, audit-grade) + §3.1 L1
Mechanism integrity (verdict framework requires Δ_pnl = Σ(weight_diff × return)).

Purpose
-------
At each monthly rebalance, persist BOTH portfolio weight vectors:
  • Track A: actual portfolio (caps applied if any active)
  • Track B: counterfactual (caps disabled, in-memory only, never traded)

Daily: compute per-ETF return × weight diff → P&L delta. Append to
counterfactual_pnl.parquet. Cumulative analysis at 24mo verdict point.

Boundary invariant (project rule "0-LLM-in-evaluation"):
  Pure deterministic computation. No LLM in this path. Reads from disk
  (cap_state + dual_track_snapshots) + fetches yfinance prices, computes
  weighted-sum, persists parquet.

Data plane:
  data/etf_holdings_risk_monitor/dual_track_snapshots.parquet
    Schema: snapshot_date, etf, track_a_weight, track_b_weight, weight_diff
  data/etf_holdings_risk_monitor/counterfactual_pnl.parquet
    Schema: date, snapshot_date, track_a_pnl, track_b_pnl, delta,
            n_diff_etfs, capped_etfs (csv)
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "etf_holdings_risk_monitor"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_DUAL_TRACK_SNAPSHOTS_PATH = _DATA_DIR / "dual_track_snapshots.parquet"
_COUNTERFACTUAL_PNL_PATH = _DATA_DIR / "counterfactual_pnl.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Public API: dual-track snapshot at rebalance
# ─────────────────────────────────────────────────────────────────────────────


def compute_dual_track_snapshot(
    as_of:        datetime.date,
    signal_df:    pd.DataFrame,
    regime:       Any | None = None,
    **portfolio_kwargs,
) -> dict:
    """
    Compute Track A (caps active) and Track B (caps disabled) portfolio weights
    at rebalance day. Both runs use identical signal + regime; only difference
    is whether ETF Holdings caps are applied.

    Returns:
        dict with keys:
            snapshot_date:    as_of (ISO string)
            track_a_weights:  {ticker: weight}  (caps applied)
            track_b_weights:  {ticker: weight}  (caps disabled)
            capped_etfs:      list of ETF tickers where A and B differ
            track_a_warnings: list (from construct_portfolio)
            track_b_warnings: list

    Errors gracefully (returns empty dict with `status: error`).
    """
    from engine.portfolio import construct_portfolio

    try:
        # Track A — actual production (caps applied if active)
        track_a = construct_portfolio(
            signal_df=signal_df,
            regime=regime,
            as_of=as_of,
            _disable_etf_holdings_caps=False,  # caps active
            **portfolio_kwargs,
        )

        # Track B — counterfactual (caps disabled; in-memory only)
        track_b = construct_portfolio(
            signal_df=signal_df,
            regime=regime,
            as_of=as_of,
            _disable_etf_holdings_caps=True,  # caps SKIPPED for counterfactual
            **portfolio_kwargs,
        )
    except Exception as exc:
        logger.error("compute_dual_track_snapshot failed: %s", exc)
        return {
            "status":          "error",
            "snapshot_date":   as_of.isoformat(),
            "error":           str(exc),
        }

    # Convert to {ticker: weight} dicts
    weights_a = _portfolio_weights_to_dict(track_a)
    weights_b = _portfolio_weights_to_dict(track_b)

    # Identify ETFs where A != B (those affected by cap)
    all_tickers = set(weights_a.keys()) | set(weights_b.keys())
    capped_etfs = sorted([
        t for t in all_tickers
        if abs(weights_a.get(t, 0.0) - weights_b.get(t, 0.0)) > 1e-9
    ])

    return {
        "status":           "ok",
        "snapshot_date":    as_of.isoformat(),
        "track_a_weights":  weights_a,
        "track_b_weights":  weights_b,
        "capped_etfs":      capped_etfs,
        "n_capped":         len(capped_etfs),
        "track_a_warnings": list(getattr(track_a, "warnings", [])),
        "track_b_warnings": list(getattr(track_b, "warnings", [])),
    }


def _portfolio_weights_to_dict(portfolio_weights: Any) -> dict[str, float]:
    """Extract {ticker: weight} dict from PortfolioWeights dataclass."""
    if portfolio_weights is None:
        return {}
    weights_attr = getattr(portfolio_weights, "weights", None)
    if weights_attr is None:
        return {}
    # weights might be pd.Series or dict
    if isinstance(weights_attr, pd.Series):
        return {str(t): float(w) for t, w in weights_attr.items() if abs(w) > 1e-12}
    if isinstance(weights_attr, dict):
        return {str(t): float(w) for t, w in weights_attr.items() if abs(w) > 1e-12}
    return {}


def persist_dual_track_snapshot(snapshot: dict) -> bool:
    """
    Append snapshot rows to dual_track_snapshots.parquet (long format,
    one row per ETF per snapshot).

    Returns True if persisted, False on error.
    """
    if snapshot.get("status") != "ok":
        logger.warning("persist_dual_track_snapshot: skipping non-ok snapshot")
        return False

    snapshot_date = snapshot["snapshot_date"]
    a = snapshot["track_a_weights"]
    b = snapshot["track_b_weights"]
    all_tickers = sorted(set(a.keys()) | set(b.keys()))

    rows = []
    for ticker in all_tickers:
        wa = a.get(ticker, 0.0)
        wb = b.get(ticker, 0.0)
        rows.append({
            "snapshot_date":  snapshot_date,
            "etf":            ticker,
            "track_a_weight": wa,
            "track_b_weight": wb,
            "weight_diff":    wa - wb,
        })

    new_df = pd.DataFrame(rows)
    if _DUAL_TRACK_SNAPSHOTS_PATH.exists():
        try:
            existing = pd.read_parquet(_DUAL_TRACK_SNAPSHOTS_PATH)
            # Drop rows with same snapshot_date (idempotent re-run)
            existing = existing[existing["snapshot_date"] != snapshot_date]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as exc:
            logger.warning(
                "persist_dual_track_snapshot: existing parquet read failed (%s); overwriting",
                exc,
            )
            combined = new_df
    else:
        combined = new_df

    try:
        combined.to_parquet(_DUAL_TRACK_SNAPSHOTS_PATH, index=False)
        logger.info(
            "persist_dual_track_snapshot: %d rows for %s",
            len(new_df), snapshot_date,
        )
        return True
    except Exception as exc:
        logger.error("persist_dual_track_snapshot: write failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API: latest snapshot lookup
# ─────────────────────────────────────────────────────────────────────────────


def get_latest_dual_track_snapshot(as_of: datetime.date) -> Optional[dict]:
    """
    Return latest dual-track snapshot at or before as_of, as
    {snapshot_date, track_a_weights, track_b_weights, capped_etfs}.

    None if no snapshot available.
    """
    if not _DUAL_TRACK_SNAPSHOTS_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_DUAL_TRACK_SNAPSHOTS_PATH)
    except Exception as exc:
        logger.warning("get_latest_dual_track_snapshot: parquet read failed: %s", exc)
        return None

    if df.empty:
        return None

    df["_date"] = pd.to_datetime(df["snapshot_date"])
    cutoff = pd.Timestamp(as_of)
    df = df[df["_date"] <= cutoff]
    if df.empty:
        return None

    latest_date = df["_date"].max()
    latest = df[df["_date"] == latest_date]

    a_weights = dict(zip(latest["etf"], latest["track_a_weight"]))
    b_weights = dict(zip(latest["etf"], latest["track_b_weight"]))
    capped_etfs = sorted([
        t for t, wd in zip(latest["etf"], latest["weight_diff"])
        if abs(wd) > 1e-9
    ])

    return {
        "snapshot_date":    latest_date.date().isoformat(),
        "track_a_weights":  a_weights,
        "track_b_weights":  b_weights,
        "capped_etfs":      capped_etfs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-ETF return fetch
# ─────────────────────────────────────────────────────────────────────────────


def fetch_etf_returns_for_date(
    as_of:    datetime.date,
    tickers:  list[str],
) -> dict[str, float]:
    """
    Fetch close-to-close 1-day return for each ETF on as_of.
    return = close[as_of] / close[prior_trading_day] - 1

    Returns {ticker: return}; ETFs with missing data are absent from dict
    (caller handles via .get(t, 0.0) defaulting to 0).
    """
    out: dict[str, float] = {}
    if not tickers:
        return out

    try:
        from engine.signal import _fetch_closes
    except Exception as exc:
        logger.error("fetch_etf_returns_for_date: signal._fetch_closes unavailable: %s", exc)
        return out

    # Fetch a 10-day window ending at as_of to get prior close
    end = as_of
    start = end - datetime.timedelta(days=10)

    for ticker in tickers:
        try:
            closes = _fetch_closes(
                ticker, start=start.isoformat(), end=end.isoformat(),
            )
            if closes is None or len(closes) < 2:
                continue
            s = closes.dropna()
            if len(s) < 2:
                continue
            latest = float(s.iloc[-1])
            prior = float(s.iloc[-2])
            if prior <= 0:
                continue
            out[ticker] = latest / prior - 1.0
        except Exception as exc:
            logger.debug(
                "fetch_etf_returns_for_date: fetch failed for %s: %s",
                ticker, exc,
            )
            continue

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API: daily P&L delta
# ─────────────────────────────────────────────────────────────────────────────


def compute_daily_pnl_delta(
    as_of:    datetime.date,
    snapshot: Optional[dict] = None,
) -> dict:
    """
    Compute Track A vs Track B P&L delta for as_of using latest snapshot
    (or provided one).

    P&L_track = Σ_etf (weight_etf × return_etf)
    delta = P&L_a - P&L_b
          = Σ_etf (weight_diff_etf × return_etf)

    Resilient: missing ETF returns default to 0 (no return assumed).

    Returns:
        {status, date, snapshot_date, track_a_pnl, track_b_pnl, delta,
         n_diff_etfs, capped_etfs}
    """
    if snapshot is None:
        snapshot = get_latest_dual_track_snapshot(as_of)
    if snapshot is None:
        return {
            "status": "no_snapshot",
            "date":   as_of.isoformat(),
            "delta":  0.0,
        }

    weights_a = snapshot["track_a_weights"]
    weights_b = snapshot["track_b_weights"]
    snapshot_date = snapshot["snapshot_date"]

    # Don't compute delta on snapshot day itself (no realized return yet)
    if as_of.isoformat() <= snapshot_date:
        return {
            "status":         "skipped_pre_snapshot",
            "date":           as_of.isoformat(),
            "snapshot_date":  snapshot_date,
            "delta":          0.0,
        }

    # Fetch returns for all relevant ETFs
    all_tickers = sorted(set(weights_a.keys()) | set(weights_b.keys()))
    returns = fetch_etf_returns_for_date(as_of, all_tickers)

    # Compute portfolio P&L for each track
    track_a_pnl = sum(
        weights_a.get(t, 0.0) * returns.get(t, 0.0)
        for t in all_tickers
    )
    track_b_pnl = sum(
        weights_b.get(t, 0.0) * returns.get(t, 0.0)
        for t in all_tickers
    )
    delta = track_a_pnl - track_b_pnl

    n_diff_etfs = sum(
        1 for t in all_tickers
        if abs(weights_a.get(t, 0.0) - weights_b.get(t, 0.0)) > 1e-9
    )
    capped_etfs = snapshot.get("capped_etfs", [])

    return {
        "status":         "ok",
        "date":           as_of.isoformat(),
        "snapshot_date":  snapshot_date,
        "track_a_pnl":    round(track_a_pnl, 8),
        "track_b_pnl":    round(track_b_pnl, 8),
        "delta":          round(delta, 8),
        "n_diff_etfs":    n_diff_etfs,
        "capped_etfs":    capped_etfs,
        "n_etfs_with_returns": len(returns),
    }


def persist_daily_pnl_delta(record: dict) -> bool:
    """Append daily P&L record to counterfactual_pnl.parquet."""
    if record.get("status") not in {"ok", "skipped_pre_snapshot", "no_snapshot"}:
        return False

    new_row = {
        "date":                 record.get("date"),
        "snapshot_date":        record.get("snapshot_date"),
        "track_a_pnl":          record.get("track_a_pnl", 0.0),
        "track_b_pnl":          record.get("track_b_pnl", 0.0),
        "delta":                record.get("delta", 0.0),
        "n_diff_etfs":          record.get("n_diff_etfs", 0),
        "capped_etfs":          ",".join(record.get("capped_etfs", [])),
        "n_etfs_with_returns":  record.get("n_etfs_with_returns", 0),
        "status":               record.get("status", "unknown"),
    }
    new_df = pd.DataFrame([new_row])

    if _COUNTERFACTUAL_PNL_PATH.exists():
        try:
            existing = pd.read_parquet(_COUNTERFACTUAL_PNL_PATH)
            existing = existing[existing["date"] != new_row["date"]]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    try:
        combined.to_parquet(_COUNTERFACTUAL_PNL_PATH, index=False)
        return True
    except Exception as exc:
        logger.error("persist_daily_pnl_delta: write failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API: cumulative metrics for verdict
# ─────────────────────────────────────────────────────────────────────────────


def compute_cumulative_metrics(
    window_days: Optional[int] = None,
) -> dict:
    """
    Aggregate counterfactual P&L parquet → cumulative metrics for verdict
    framework (spec §3.1 L3 cumulative description).

    Args:
        window_days: lookback (None = full history)

    Returns:
        {
          n_obs:                 # of daily P&L observations
          n_active_days:         # of days with non-zero delta
          cumulative_delta:      sum of deltas (P&L diff, fractional)
          cumulative_track_a:    sum track A returns
          cumulative_track_b:    sum track B returns
          delta_sharpe_annualized: sqrt(252) × mean(delta) / std(delta) (descriptive)
          delta_max_drawdown:    cumulative drawdown of (A - B) series
          n_capped_events:       # of distinct capped_etfs entries (any cap fired)
          status: ok | empty | error
        }
    """
    if not _COUNTERFACTUAL_PNL_PATH.exists():
        return {"status": "empty", "reason": "no_parquet_file"}

    try:
        df = pd.read_parquet(_COUNTERFACTUAL_PNL_PATH)
    except Exception as exc:
        return {"status": "error", "reason": f"parquet_read_failed: {exc}"}

    if df.empty:
        return {"status": "empty", "reason": "empty_parquet"}

    if window_days is not None:
        df["_date"] = pd.to_datetime(df["date"])
        cutoff = df["_date"].max() - pd.Timedelta(days=window_days)
        df = df[df["_date"] >= cutoff]

    df = df[df["status"] == "ok"]
    if df.empty:
        return {
            "status":      "empty",
            "reason":      "no_ok_records",
            "n_obs":       0,
            "n_active_days": 0,
        }

    delta = df["delta"].astype(float).values
    track_a = df["track_a_pnl"].astype(float).values
    track_b = df["track_b_pnl"].astype(float).values

    cumulative_delta = float(np.sum(delta))
    cumulative_a    = float(np.sum(track_a))
    cumulative_b    = float(np.sum(track_b))

    # Annualized ΔSharpe (descriptive only — not statistical inference)
    delta_sharpe = 0.0
    if len(delta) >= 5 and np.std(delta) > 1e-12:
        delta_sharpe = float(np.sqrt(252.0) * np.mean(delta) / np.std(delta))

    # Cumulative max drawdown of (A - B) series
    cumulative_series = np.cumsum(delta)
    running_max = np.maximum.accumulate(cumulative_series)
    drawdown_series = cumulative_series - running_max
    max_dd = float(np.min(drawdown_series)) if len(drawdown_series) else 0.0

    # Count active days (capped_etfs non-empty)
    n_active = int(
        df["capped_etfs"]
          .fillna("")
          .apply(lambda s: len(s) > 0)
          .sum()
    )

    return {
        "status":                 "ok",
        "n_obs":                  int(len(df)),
        "n_active_days":          n_active,
        "cumulative_delta":       round(cumulative_delta, 8),
        "cumulative_track_a":     round(cumulative_a, 8),
        "cumulative_track_b":     round(cumulative_b, 8),
        "delta_sharpe_annualized": round(delta_sharpe, 4),
        "delta_max_drawdown":     round(max_dd, 8),
        "delta_mean":             round(float(np.mean(delta)), 8),
        "delta_std":              round(float(np.std(delta)), 8),
    }
