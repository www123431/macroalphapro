"""engine.governance.llm_budget — operational LLM budget guardrail.

Doctrine (5b.2, 2026-06-02):
    chat_ask + research_ops_paper_scorer + research_ops_weekly_digest +
    every other LLM-using agent writes to engine.llm_cost_ledger.
    Without a budget cap, costs can run away during a chat loop or
    paper-flood week. This module owns the budget configuration; the
    /ops UI lets the user set it; the cost ledger reading + percent-
    of-budget calc happens here.

Single SoT for the budget: data/governance/llm_budget.json. Schema:

    {
      "monthly_cap_usd":    float,        # total cap across all agents
      "agent_caps_usd":     {agent_id: float},   # per-agent monthly cap
      "alert_threshold_pct": float        # alert when usage > this % of cap
    }

Defaults baked here so a missing file just yields safe behaviour.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUDGET_PATH = _REPO_ROOT / "data" / "governance" / "llm_budget.json"


DEFAULT_MONTHLY_CAP_USD     = 50.0
DEFAULT_ALERT_THRESHOLD_PCT = 80.0
# Per-agent default ratios (sum need not equal 1; these are SOFT caps).
DEFAULT_AGENT_CAPS_USD = {
    "chat_ask":                       20.0,
    "research_ops_paper_scorer":       5.0,
    "research_ops_weekly_digest":      2.0,
    "decay_sentinel":                  3.0,
    "anomaly_sentinel":                3.0,
    "attribution_analyst":             3.0,
    "audit_recorder":                  2.0,
    "chief_of_staff":                  5.0,
    "devils_advocate":                 5.0,
    "risk_manager":                    5.0,
    "dq_inspector":                    2.0,
}


def load_budget() -> dict[str, Any]:
    """Return current budget config, applying defaults for missing fields."""
    if not _BUDGET_PATH.is_file():
        return {
            "monthly_cap_usd":     DEFAULT_MONTHLY_CAP_USD,
            "agent_caps_usd":      dict(DEFAULT_AGENT_CAPS_USD),
            "alert_threshold_pct": DEFAULT_ALERT_THRESHOLD_PCT,
            "_source":             "default (no llm_budget.json)",
        }
    try:
        data = json.loads(_BUDGET_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("llm_budget: failed to read %s", _BUDGET_PATH)
        return {
            "monthly_cap_usd":     DEFAULT_MONTHLY_CAP_USD,
            "agent_caps_usd":      dict(DEFAULT_AGENT_CAPS_USD),
            "alert_threshold_pct": DEFAULT_ALERT_THRESHOLD_PCT,
            "_source":             "default (read error)",
        }
    out = {
        "monthly_cap_usd":     float(data.get("monthly_cap_usd",     DEFAULT_MONTHLY_CAP_USD)),
        "agent_caps_usd":      dict(data.get("agent_caps_usd",       DEFAULT_AGENT_CAPS_USD)),
        "alert_threshold_pct": float(data.get("alert_threshold_pct", DEFAULT_ALERT_THRESHOLD_PCT)),
        "_source":             str(_BUDGET_PATH),
        "_last_updated":       data.get("_last_updated"),
    }
    return out


def save_budget(
    *,
    monthly_cap_usd:     Optional[float] = None,
    agent_caps_usd:      Optional[dict[str, float]] = None,
    alert_threshold_pct: Optional[float] = None,
) -> dict[str, Any]:
    """Merge-update budget config and write back to disk. Returns the
    updated state. NULL/None args mean 'keep current'."""
    current = load_budget()
    new_monthly = monthly_cap_usd if monthly_cap_usd is not None else current["monthly_cap_usd"]
    new_caps    = agent_caps_usd  if agent_caps_usd  is not None else current["agent_caps_usd"]
    new_alert   = alert_threshold_pct if alert_threshold_pct is not None else current["alert_threshold_pct"]
    if new_monthly < 0:
        raise ValueError("monthly_cap_usd must be ≥ 0")
    if not 0 <= new_alert <= 100:
        raise ValueError("alert_threshold_pct must be in [0, 100]")
    for k, v in (new_caps or {}).items():
        if v < 0:
            raise ValueError(f"agent_caps_usd[{k}] must be ≥ 0")

    payload = {
        "monthly_cap_usd":     float(new_monthly),
        "agent_caps_usd":      {k: float(v) for k, v in (new_caps or {}).items()},
        "alert_threshold_pct": float(new_alert),
        "_last_updated":       _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BUDGET_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return load_budget()


def compute_usage() -> dict[str, Any]:
    """Aggregate current-month LLM spend from the cost ledger + compare
    to budget. Returns full snapshot for UI rendering."""
    from engine import llm_cost_ledger as L
    today = _dt.datetime.utcnow().date()
    month_start = today.replace(day=1)
    calls_mtd = L.get_calls(since=month_start, until=today)

    spend_total = round(sum(c.cost_usd for c in calls_mtd), 6)
    by_agent: dict[str, dict[str, Any]] = {}
    for c in calls_mtd:
        d = by_agent.setdefault(c.agent_id, {
            "spend_usd": 0.0, "calls": 0, "last_ts": None,
        })
        d["spend_usd"] = round(d["spend_usd"] + c.cost_usd, 6)
        d["calls"]    += 1
        if not d["last_ts"] or c.ts > d["last_ts"]:
            d["last_ts"] = c.ts

    budget = load_budget()
    monthly_cap = float(budget["monthly_cap_usd"])
    alert_thr   = float(budget["alert_threshold_pct"])
    agent_caps  = dict(budget["agent_caps_usd"])

    # Per-agent breakdown w/ cap + status
    agents_out: list[dict[str, Any]] = []
    for aid, d in by_agent.items():
        cap = float(agent_caps.get(aid, 0.0))
        pct = (d["spend_usd"] / cap * 100.0) if cap > 0 else None
        status = "ok"
        if pct is not None:
            if pct >= 100.0:    status = "over"
            elif pct >= alert_thr: status = "alert"
        agents_out.append({
            "agent_id":   aid,
            "spend_usd":  d["spend_usd"],
            "calls":      d["calls"],
            "cap_usd":    cap if cap > 0 else None,
            "pct_of_cap": round(pct, 1) if pct is not None else None,
            "status":     status,
            "last_ts":    d["last_ts"],
        })
    agents_out.sort(key=lambda r: -r["spend_usd"])

    total_pct = (spend_total / monthly_cap * 100.0) if monthly_cap > 0 else None
    total_status = "ok"
    if total_pct is not None:
        if total_pct >= 100.0:    total_status = "over"
        elif total_pct >= alert_thr: total_status = "alert"

    return {
        "month_start":      month_start.isoformat(),
        "as_of":            today.isoformat(),
        "monthly_cap_usd":  monthly_cap,
        "alert_threshold_pct": alert_thr,
        "total_spend_usd":  spend_total,
        "total_pct_of_cap": round(total_pct, 1) if total_pct is not None else None,
        "total_status":     total_status,
        "agents":           agents_out,
        "n_agents_alert":   sum(1 for a in agents_out if a["status"] == "alert"),
        "n_agents_over":    sum(1 for a in agents_out if a["status"] == "over"),
    }
