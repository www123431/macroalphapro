"""
engine/agents/ops_watchdog/tools.py — 10 read-only tools for Watchdog ReAct.

Per spec §2.4 (LOCKED tool inventory): 5 NEW + 5 REUSED from Tool 1's
`engine.quant_co_pilot.tools`. All tools return ToolResult(success, data,
error_msg) — same convention as Tool 1 so the dispatcher pattern is shared.

INVARIANT: Watchdog never writes to production tables (portfolio /
simulated_positions / simulated_trades / portfolio_nav_snapshots /
universe_etfs). These tools READ ONLY. Spec §6 forbidden-modifications
asserts zero production-write capability from Watchdog's perspective.

The 5 NEW tools (this module):
  - read_audit_findings:      query AuditFinding rows on/around a date
  - read_cycle_state:         query CycleState rows (most recent N)
  - read_trade_log:           query SimulatedTrade rows on a date
  - read_nav_change:          query PortfolioNavSnapshot rows (last N days)
  - read_historical_baseline: mean / sigma / p99 of a metric over lookback

The 5 REUSED tools (re-exported from Tool 1):
  - read_spec_registry / search_amendments / read_capability_evidence /
    read_memory_file / read_verdict_json
"""
from __future__ import annotations

import datetime
import json
import math
import time
from typing import Any, Optional

# Phase 1 Agent Observability v1 — Gap #1 trace recording (2026-05-15).
# record_tool_call() auto-joins to current AGENT_INVOCATION_ID via os.environ
# (set by @track_agent_invocation in observability.py). Best-effort: if no
# active invocation context, the call is dropped silently and the agent
# functions normally.
try:
    from engine.agents.observability import record_tool_call as _record_tool_call
except Exception:    # pragma: no cover — observability is optional at import time
    def _record_tool_call(**_):  # type: ignore[misc]
        pass

# Reuse Tool 1's ToolResult shape — same dispatcher convention
from engine.quant_co_pilot.tools import (
    ToolResult,
    read_capability_evidence,
    read_memory_file,
    read_spec_registry,
    read_verdict_json,
    search_amendments,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5 NEW Watchdog tools (read-only DB queries)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: Optional[str]) -> datetime.date:
    """Parse YYYY-MM-DD or default to today."""
    if not date_str:
        return datetime.date.today()
    return datetime.date.fromisoformat(date_str)


def read_audit_findings(
    date_str:        Optional[str] = None,
    severity_filter: Optional[str] = None,
    status_filter:   Optional[str] = "OPEN",
    limit:           int           = 50,
) -> ToolResult:
    """
    Return AuditFinding rows detected on `date_str` (default: today).

    Args:
        date_str: YYYY-MM-DD; default today
        severity_filter: "LOW"|"MID"|"HIGH"|None (no filter)
        status_filter:   "OPEN"|"RESOLVED"|"IGNORED"|None (default "OPEN")
        limit: max rows returned (default 50)

    Returns ToolResult.data = list[{
        id, run_id, rule_name, severity, detected_at,
        snapshot, status,
    }]
    """
    try:
        from engine.auto_audit_models import AuditFinding
        from engine.memory import SessionFactory
        import json as _json
    except Exception as exc:
        return ToolResult(False, None, f"import_failed: {exc!s}")

    try:
        target_date = _parse_date(date_str)
    except ValueError as exc:
        return ToolResult(False, None, f"bad date_str: {exc!s}")

    day_start = datetime.datetime.combine(target_date, datetime.time(0, 0))
    day_end   = day_start + datetime.timedelta(days=1)

    try:
        with SessionFactory() as s:
            q = (s.query(AuditFinding)
                  .filter(AuditFinding.detected_at >= day_start,
                          AuditFinding.detected_at < day_end))
            if severity_filter:
                q = q.filter(AuditFinding.severity == severity_filter)
            if status_filter:
                q = q.filter(AuditFinding.status == status_filter)
            rows = q.order_by(AuditFinding.detected_at.desc()).limit(limit).all()
    except Exception as exc:
        return ToolResult(False, None, f"query_failed: {exc!s}")

    out = []
    for r in rows:
        try:
            snap = _json.loads(r.snapshot_json) if r.snapshot_json else {}
        except Exception:
            snap = {"_parse_error": True, "raw_excerpt": (r.snapshot_json or "")[:200]}
        out.append({
            "id":          r.id,
            "run_id":      r.run_id,
            "rule_name":   r.rule_name,
            "severity":    r.severity,
            "detected_at": r.detected_at.isoformat() if r.detected_at else None,
            "snapshot":    snap,
            "status":      r.status,
        })
    return ToolResult(True, {"date": target_date.isoformat(),
                             "n_rows": len(out), "rows": out}, None)


def read_cycle_state(
    date_str: Optional[str] = None,
    n:        int           = 5,
) -> ToolResult:
    """
    Return the most recent `n` CycleState rows (across all cycle_types).
    Optionally filtered to cycles whose started_at falls on `date_str`.

    Returns ToolResult.data = list[{
        id, cycle_type, as_of_date, status, gate, started_at, finished_at,
        elapsed_s, error_log (truncated 500c), result_summary (truncated 500c),
    }]
    """
    try:
        from engine.db_models import CycleState
        from engine.memory import SessionFactory
    except Exception as exc:
        return ToolResult(False, None, f"import_failed: {exc!s}")

    try:
        with SessionFactory() as s:
            q = s.query(CycleState)
            if date_str:
                try:
                    target = _parse_date(date_str)
                except ValueError as exc:
                    return ToolResult(False, None, f"bad date_str: {exc!s}")
                day_start = datetime.datetime.combine(target, datetime.time(0, 0))
                day_end   = day_start + datetime.timedelta(days=1)
                q = q.filter(CycleState.started_at >= day_start,
                             CycleState.started_at < day_end)
            rows = q.order_by(CycleState.started_at.desc()).limit(n).all()
    except Exception as exc:
        return ToolResult(False, None, f"query_failed: {exc!s}")

    out = []
    for r in rows:
        out.append({
            "id":             r.id,
            "cycle_type":     r.cycle_type,
            "as_of_date":     r.as_of_date.isoformat() if r.as_of_date else None,
            "status":         r.status,
            "gate":           r.gate,
            "started_at":     r.started_at.isoformat() if r.started_at else None,
            "finished_at":    r.finished_at.isoformat() if r.finished_at else None,
            "elapsed_s":      r.elapsed_s,
            "error_log":      (r.error_log or "")[:500] if r.error_log else None,
            "result_summary": (r.result_summary or "")[:500] if r.result_summary else None,
        })
    return ToolResult(True, {"n_rows": len(out), "rows": out}, None)


def read_trade_log(
    date_str: Optional[str] = None,
    limit:    int           = 50,
) -> ToolResult:
    """
    Return SimulatedTrade rows on `date_str` (default: today).

    Returns ToolResult.data = list[{
        id, trade_date, sector, ticker, action, weight_before, weight_after,
        weight_delta, cost_bps, trigger_reason, sleeve_id,
    }]
    """
    try:
        from engine.db_models import SimulatedTrade
        from engine.memory import SessionFactory
    except Exception as exc:
        return ToolResult(False, None, f"import_failed: {exc!s}")

    try:
        target_date = _parse_date(date_str)
    except ValueError as exc:
        return ToolResult(False, None, f"bad date_str: {exc!s}")

    try:
        with SessionFactory() as s:
            rows = (s.query(SimulatedTrade)
                     .filter(SimulatedTrade.trade_date == target_date)
                     .order_by(SimulatedTrade.id.desc())
                     .limit(limit)
                     .all())
    except Exception as exc:
        return ToolResult(False, None, f"query_failed: {exc!s}")

    out = []
    for r in rows:
        out.append({
            "id":             r.id,
            "trade_date":     r.trade_date.isoformat() if r.trade_date else None,
            "sector":         r.sector,
            "ticker":         r.ticker,
            "action":         r.action,
            "weight_before":  r.weight_before,
            "weight_after":   r.weight_after,
            "weight_delta":   r.weight_delta,
            "cost_bps":       r.cost_bps,
            "trigger_reason": r.trigger_reason,
            "sleeve_id":      getattr(r, "sleeve_id", None),
        })
    return ToolResult(True, {"date": target_date.isoformat(),
                             "n_rows": len(out), "rows": out}, None)


def read_nav_change(
    date_str:    Optional[str] = None,
    n_days_back: int           = 5,
) -> ToolResult:
    """
    Return PortfolioNavSnapshot rows up to and including `date_str` for the
    last `n_days_back` records (default: 5).

    Returns ToolResult.data = list[{
        snapshot_date, nav_open, external_flow, nav_after_flow, nav_close,
        gross_pnl, benchmark_close, daily_modified_dietz, notes,
    }]
    """
    try:
        from engine.db_models import PortfolioNavSnapshot
        from engine.memory import SessionFactory
    except Exception as exc:
        return ToolResult(False, None, f"import_failed: {exc!s}")

    try:
        target_date = _parse_date(date_str)
    except ValueError as exc:
        return ToolResult(False, None, f"bad date_str: {exc!s}")

    try:
        with SessionFactory() as s:
            rows = (s.query(PortfolioNavSnapshot)
                     .filter(PortfolioNavSnapshot.snapshot_date <= target_date)
                     .order_by(PortfolioNavSnapshot.snapshot_date.desc())
                     .limit(max(1, n_days_back))
                     .all())
    except Exception as exc:
        return ToolResult(False, None, f"query_failed: {exc!s}")

    out = []
    for r in rows:
        out.append({
            "snapshot_date":         r.snapshot_date.isoformat(),
            "nav_open":              r.nav_open,
            "external_flow":         r.external_flow,
            "nav_after_flow":        r.nav_after_flow,
            "nav_close":             r.nav_close,
            "gross_pnl":             r.gross_pnl,
            "benchmark_close":       r.benchmark_close,
            "daily_modified_dietz":  r.daily_modified_dietz,
            "notes":                 r.notes,
        })
    return ToolResult(True, {"reference_date": target_date.isoformat(),
                             "n_rows": len(out), "rows": out}, None)


def read_historical_baseline(
    metric:        str,
    lookback_days: int = 60,
) -> ToolResult:
    """
    Compute mean / sigma / p99 / min / max for one of three metrics:
      - "nav_return"       : PortfolioNavSnapshot.daily_modified_dietz, last N days
      - "weight_delta"     : |SimulatedTrade.weight_delta|, last N days
      - "tc_bps_per_unit"  : SimulatedTrade.cost_bps / |weight_delta|, last N days

    Args:
        metric:        one of the 3 above
        lookback_days: window size (default 60)

    Returns ToolResult.data = {
        metric, lookback_days, n_obs, mean, sigma, p99, min, max
    }
    """
    if metric not in {"nav_return", "weight_delta", "tc_bps_per_unit"}:
        return ToolResult(False, None,
                          f"unknown metric '{metric}'; valid: "
                          "nav_return|weight_delta|tc_bps_per_unit")
    if lookback_days <= 0:
        return ToolResult(False, None, "lookback_days must be > 0")

    try:
        from engine.db_models import PortfolioNavSnapshot, SimulatedTrade
        from engine.memory import SessionFactory
    except Exception as exc:
        return ToolResult(False, None, f"import_failed: {exc!s}")

    today  = datetime.date.today()
    cutoff = today - datetime.timedelta(days=lookback_days)

    values: list[float] = []
    try:
        with SessionFactory() as s:
            if metric == "nav_return":
                rows = (s.query(PortfolioNavSnapshot.daily_modified_dietz)
                         .filter(PortfolioNavSnapshot.snapshot_date >= cutoff)
                         .all())
                values = [float(r[0]) for r in rows if r[0] is not None]
            elif metric == "weight_delta":
                rows = (s.query(SimulatedTrade.weight_delta)
                         .filter(SimulatedTrade.trade_date >= cutoff)
                         .all())
                values = [abs(float(r[0])) for r in rows if r[0] is not None]
            elif metric == "tc_bps_per_unit":
                rows = (s.query(SimulatedTrade.cost_bps,
                                SimulatedTrade.weight_delta)
                         .filter(SimulatedTrade.trade_date >= cutoff)
                         .filter(SimulatedTrade.cost_bps.isnot(None))
                         .all())
                for cb, wd in rows:
                    if wd is None or abs(float(wd)) < 1e-6:
                        continue
                    values.append(float(cb) / abs(float(wd)))
    except Exception as exc:
        return ToolResult(False, None, f"query_failed: {exc!s}")

    n = len(values)
    if n == 0:
        return ToolResult(True, {
            "metric":        metric,
            "lookback_days": lookback_days,
            "n_obs":         0,
            "mean":          None, "sigma": None, "p99": None,
            "min":           None, "max": None,
        }, None)

    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    sigma = math.sqrt(var)
    sorted_vals = sorted(values)
    p99_idx = max(0, int(round(0.99 * (n - 1))))
    p99 = sorted_vals[p99_idx]

    return ToolResult(True, {
        "metric":        metric,
        "lookback_days": lookback_days,
        "n_obs":         n,
        "mean":          round(mean, 8),
        "sigma":         round(sigma, 8),
        "p99":           round(p99, 8),
        "min":           round(sorted_vals[0], 8),
        "max":           round(sorted_vals[-1], 8),
    }, None)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry + dispatcher (10 tools: 5 NEW + 5 REUSED from Tool 1)
# ─────────────────────────────────────────────────────────────────────────────

WATCHDOG_TOOL_REGISTRY: dict[str, Any] = {
    # NEW (5)
    "read_audit_findings":       read_audit_findings,
    "read_cycle_state":          read_cycle_state,
    "read_trade_log":            read_trade_log,
    "read_nav_change":           read_nav_change,
    "read_historical_baseline":  read_historical_baseline,
    # REUSED from Tool 1 (5) — same function references, no wrapping
    "read_spec_registry":        read_spec_registry,
    "search_amendments":         search_amendments,
    "read_capability_evidence":  read_capability_evidence,
    "read_memory_file":          read_memory_file,
    "read_verdict_json":         read_verdict_json,
}

WATCHDOG_TOOL_NAMES: tuple[str, ...] = tuple(WATCHDOG_TOOL_REGISTRY.keys())


WATCHDOG_TOOL_DESCRIPTIONS = """
1. read_audit_findings(date_str: str|null, severity_filter: str|null, status_filter: str|null, limit: int = 50)
   → {date, n_rows, rows[{id, rule_name, severity, detected_at, snapshot, status}]}
   Watchdog mode evidence: queries today's AuditFinding rows (Phase 1 rules wrote these).

2. read_cycle_state(date_str: str|null, n: int = 5)
   → {n_rows, rows[{cycle_type, status, gate, started_at, finished_at, elapsed_s, error_log}]}
   Daily-batch cycle health: status/error trail of recent orchestrator runs.

3. read_trade_log(date_str: str|null, limit: int = 50)
   → {date, n_rows, rows[{ticker, action, weight_delta, cost_bps, trigger_reason, sleeve_id}]}
   Trade evidence: SimulatedTrade rows for a date; pair with signal & NAV to spot anomalies.

4. read_nav_change(date_str: str|null, n_days_back: int = 5)
   → {reference_date, n_rows, rows[{snapshot_date, nav_close, daily_modified_dietz, external_flow}]}
   NAV trajectory: last N PortfolioNavSnapshot rows to contextualize today's move.

5. read_historical_baseline(metric: str, lookback_days: int = 60)
   → {metric, n_obs, mean, sigma, p99, min, max}
   Baseline stats for {"nav_return", "weight_delta", "tc_bps_per_unit"} over a window.

6. read_spec_registry(spec_id: int)
   → {spec_path, status, current_hash, n_trials_contributed, factor_kind, amendment_log[]}
   Spec lookup (REUSED from Tool 1).

7. search_amendments(reason_substring: str, limit: int = 10)
   → list[{spec_id, kind, reason, n_trials_added, at}]
   Amendment audit trail (REUSED from Tool 1).

8. read_capability_evidence(filename: str)
   → str (full markdown)
   Past verdict/capability evidence file (REUSED from Tool 1).

9. read_memory_file(memory_filename: str)
   → str (full markdown; project_*.md / feedback_*.md / etc.)
   Memory recall (REUSED from Tool 1).

10. read_verdict_json(verdict_path: str)
    → dict (verdict JSON content under data/*)
    Verdict JSON (REUSED from Tool 1).
"""


def dispatch_watchdog_tool(action: str, action_input: Optional[dict]) -> Any:
    """
    Dispatch a Watchdog tool call. Returns dict-shape for ReAct observations:
      success → {"data": ...}
      failure → {"error": "..."}

    Unknown tools fail loud (caller's run_react_agent enforces strict allow list).

    Each call is recorded to data/agent_trace_log.jsonl via record_tool_call()
    (Phase 1 Agent Observability v1 trace recording, Gap #1 fix 2026-05-15).
    The record is best-effort: any exception in tracing is swallowed and the
    actual tool result is returned unchanged so trace failures cannot break
    Watchdog itself.
    """
    args_preview = ""
    try:
        args_preview = json.dumps(action_input or {}, ensure_ascii=False)[:200]
    except Exception:
        args_preview = str(action_input)[:200]

    t0 = time.monotonic()

    if action not in WATCHDOG_TOOL_REGISTRY:
        latency_ms = int((time.monotonic() - t0) * 1000)
        try:
            _record_tool_call(
                tool_name=action, args_preview=args_preview, result_preview="",
                latency_ms=latency_ms, success=False,
                error_message=f"unknown tool '{action}'",
            )
        except Exception:
            pass
        return {"error": f"unknown tool '{action}'; valid: {sorted(WATCHDOG_TOOL_REGISTRY)}"}

    fn = WATCHDOG_TOOL_REGISTRY[action]
    error_msg: Optional[str] = None
    success = True
    try:
        result: ToolResult = fn(**(action_input or {}))
    except TypeError as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        try:
            _record_tool_call(
                tool_name=action, args_preview=args_preview, result_preview="",
                latency_ms=latency_ms, success=False,
                error_message=f"arg mismatch: {exc!s}",
            )
        except Exception:
            pass
        return {"error": f"tool '{action}' arg mismatch: {exc!s}"}
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        try:
            _record_tool_call(
                tool_name=action, args_preview=args_preview, result_preview="",
                latency_ms=latency_ms, success=False,
                error_message=f"raised: {exc!s}",
            )
        except Exception:
            pass
        return {"error": f"tool '{action}' raised: {exc!s}"}

    latency_ms = int((time.monotonic() - t0) * 1000)
    if not result.success:
        success = False
        error_msg = result.error_msg or "tool returned failure"
        try:
            _record_tool_call(
                tool_name=action, args_preview=args_preview, result_preview="",
                latency_ms=latency_ms, success=False, error_message=error_msg,
            )
        except Exception:
            pass
        return {"error": error_msg}

    # Success path — record before returning
    try:
        result_preview = ""
        try:
            result_preview = json.dumps(result.data, ensure_ascii=False, default=str)[:200]
        except Exception:
            result_preview = str(result.data)[:200]
        _record_tool_call(
            tool_name=action, args_preview=args_preview, result_preview=result_preview,
            latency_ms=latency_ms, success=True, error_message=None,
        )
    except Exception:
        pass
    return {"data": result.data}
