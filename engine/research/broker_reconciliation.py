"""engine/research/broker_reconciliation.py — P1a of liveness layer
(2026-06-02 senior ops protocol).

Confirms that what the engine *intended* to submit, the broker actually
*received and filled*. This catches the silent-failure mode where the
daily script logs "n_orders=114" but in reality Alpaca rejected all 114
(e.g. PDT freeze / account margin call / IP block) without us noticing.

Trust hierarchy:
  1. Local _paper_submit_<date>.json — authoritative record of what the
     execution layer reported submitting + filling.
  2. Live Alpaca account state (get_positions + get_account) — ground
     truth of what's actually on the books right now.

Both are best-effort: the function never raises. If the submit artifact
is missing or Alpaca is unreachable, it returns a status field describing
the gap so the caller (heartbeat row) can downgrade liveness verdict
honestly instead of mis-reporting OK.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


# Reconciliation status enum — appears verbatim in heartbeat rows and
# the frontend KPI cell, so changing these strings is a UI break.
STATUS_OK                 = "ok"
STATUS_NO_SUBMIT_ARTIFACT = "no_submit_artifact"
STATUS_NO_BROKER_KEY      = "no_broker_key"
STATUS_BROKER_UNREACHABLE = "broker_unreachable"
STATUS_FILL_SHORTFALL     = "fill_shortfall"     # n_fills < n_orders


def _load_submit_artifact(as_of: _dt.date) -> Optional[dict]:
    """data/_paper_submit_<YYYY-MM-DD>.json shape:
       { as_of, n_tickers, gross_weight, report: {
           broker, paper, equity_before, n_orders, n_fills,
           orders[], fills[], skipped_below_min[], warnings[] } }
    Returns None if the artifact doesn't exist for as_of."""
    p = REPO_ROOT / "data" / f"_paper_submit_{as_of.isoformat()}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("submit artifact for %s is not parseable", as_of)
        return None


def _try_live_alpaca() -> Optional[dict]:
    """Best-effort Alpaca status pull. Returns:
      { equity, cash, n_positions, gross_exposure } or None if any step
    (import, env, network) fails. Never raises."""
    try:
        from engine.execution.alpaca_adapter import AlpacaAdapter
    except Exception as exc:
        logger.info("alpaca adapter import failed (likely env): %s", exc)
        return None
    try:
        adapter = AlpacaAdapter()
        acct = adapter.get_account()
        positions = adapter.get_positions()
    except Exception as exc:
        logger.info("alpaca live fetch failed: %s", exc)
        return None
    gross = sum(abs(p.market_value) for p in positions.values())
    return {
        "equity":           float(acct.equity),
        "cash":             float(acct.cash),
        "buying_power":     float(acct.buying_power),
        "n_positions":      len(positions),
        "gross_exposure":   round(gross, 2),
        "position_tickers": sorted(positions.keys()),
    }


def reconcile(as_of: _dt.date) -> dict:
    """Compare intended (engine) vs submitted (broker) vs actual (live).

    Returns a structured dict suitable for embedding in a heartbeat row.
    Fields are always present (None if unavailable) so UI render logic
    can rely on a stable shape."""
    out: dict = {
        "status":              None,
        "as_of":               as_of.isoformat(),
        "n_orders_intended":   None,
        "n_orders_submitted":  None,
        "n_fills":             None,
        "fill_rate":           None,
        "equity_before":       None,
        "broker_ack":          None,
        "n_warnings":          None,
        "live":                None,
        "explanation":         None,
    }

    artifact = _load_submit_artifact(as_of)
    if artifact is None:
        out["status"]      = STATUS_NO_SUBMIT_ARTIFACT
        out["explanation"] = (
            f"No data/_paper_submit_{as_of.isoformat()}.json found. "
            f"Either the execution layer didn't run today, or it ran "
            f"as dry-run without persistence."
        )
        out["live"] = _try_live_alpaca()    # still try live, informational
        return out

    report = artifact.get("report") or {}
    n_intended = int(artifact.get("n_tickers") or 0)
    n_submitted = int(report.get("n_orders") or 0)
    n_fills     = int(report.get("n_fills")  or 0)
    out["n_orders_intended"]  = n_intended
    out["n_orders_submitted"] = n_submitted
    out["n_fills"]            = n_fills
    out["fill_rate"]          = round(n_fills / n_submitted, 4) if n_submitted else None
    out["equity_before"]      = report.get("equity_before")
    out["broker_ack"]         = report.get("broker")
    out["n_warnings"]         = len(report.get("warnings") or [])

    out["live"] = _try_live_alpaca()

    # Verdict logic — order of precedence
    if out["live"] is None and report.get("broker") != "alpaca_paper":
        out["status"] = STATUS_NO_BROKER_KEY
        out["explanation"] = (
            f"Submit artifact present but broker={report.get('broker')!r} "
            f"and live Alpaca pull unavailable. Cannot verify fills hit the wire."
        )
        return out

    if out["live"] is None:
        out["status"] = STATUS_BROKER_UNREACHABLE
        out["explanation"] = (
            "Alpaca account API unreachable from this host. Submit artifact "
            "reports orders + fills but we can't independently confirm. "
            "Check ALPACA_KEY / ALPACA_SECRET env and network."
        )
        return out

    if n_submitted > 0 and n_fills < n_submitted:
        out["status"] = STATUS_FILL_SHORTFALL
        out["explanation"] = (
            f"Broker accepted {n_submitted} orders but only {n_fills} filled. "
            f"Shortfall = {n_submitted - n_fills}. Likely market closed / "
            f"insufficient liquidity / partial-fill regime; investigate."
        )
        return out

    out["status"] = STATUS_OK
    out["explanation"] = (
        f"Reconciled: {n_submitted} orders → {n_fills} fills; live account "
        f"holds {out['live']['n_positions']} positions, equity "
        f"${out['live']['equity']:,.0f}."
    )
    return out
