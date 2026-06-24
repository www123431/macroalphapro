"""
engine/agents/observability.py — Agent Observability v1 (Phase 1).

Production-grade per-invocation tracking for agentic AI components.
Parallels existing data/llm_cost_ledger.jsonl (per-LLM-call) at HIGHER abstraction:
each agent INVOCATION (which may emit multiple LLM calls) gets one record.

Design pattern: lightweight project-internal observability (LangSmith / Langfuse
without their bloat / vendor-lock). 1-dev scale, file-based JSONL persistence.

Doctrine:
  - 0-LLM-in-DECISION preserved (this module ONLY measures, never decides)
  - Respects 7-agent ceiling per feedback_agent_addition_rule.md (instruments
    EXISTING agents, does not add new agents)
  - LLM stays risk-side per feedback_llm_risk_side_not_alpha_side.md
    (this is observability, not LLM-as-alpha)

Usage pattern:
    @track_agent_invocation(agent_id="ops_watchdog")
    def run_watchdog(...) -> WatchdogRunResult:
        ...
        return result

Decorator auto-emits one record to data/agent_slo_metrics.jsonl per invocation.
"""
from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import os
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


# Resolve repo-relative path to agent metrics ledger
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_METRICS_PATH_DEFAULT = _REPO_ROOT / "data" / "agent_slo_metrics.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode taxonomy (enumerated)
# ─────────────────────────────────────────────────────────────────────────────
class FailureMode(str, Enum):
    """Enumerated failure modes for agent invocations.

    Used for classification + dashboard breakdown. Add new modes ONLY when
    truly distinct (not when 1 instance of a similar failure pops up).
    """
    NONE              = "none"                  # invocation succeeded
    TIMEOUT           = "timeout"               # invocation exceeded time budget
    API_ERROR         = "api_error"             # upstream LLM provider error
    SCHEMA_BREAK      = "schema_break"          # output failed schema validation
    HALLUCINATION     = "hallucination"         # output schema-valid but content wrong
                                                 # (v1: NOT auto-detected; manual flag only)
    COST_OVERRUN      = "cost_overrun"          # cost exceeded budget
    WRONG_TOOL        = "wrong_tool"            # tool call validation failed
    EMPTY_OUTPUT      = "empty_output"          # invocation completed but no useful output
    DOWNSTREAM_FAILED = "downstream_failed"     # depends on other agent that failed
    PYTHON_EXCEPTION  = "python_exception"      # uncaught Python error
    DATA_UNAVAILABLE  = "data_unavailable"      # required input data missing
    CIRCUIT_BREAKER   = "circuit_breaker"       # halted by upstream circuit breaker


# ─────────────────────────────────────────────────────────────────────────────
# Agent invocation record schema
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class AgentInvocationRecord:
    """One record per agent invocation (whole agent run, not per LLM call).

    quality_signals: agent-specific quality metrics dict, extensible without
    schema changes. v1 examples:
      - cross_llm_consistency_score: float (devils_advocate Gemini vs DeepSeek)
      - tool_execution_success_rate: float (Watchdog deterministic tools)
      - output_length_anomaly: bool (>3σ from rolling mean)
      - self_consistency_score: float (optional N-shot stability check)
      - schema_partial_match: float (0..1 partial schema correctness)
    Each agent's @track_agent_invocation can pass quality_extractor callable
    to populate this dict. v1 deploys with schema_valid as primary signal;
    quality_signals reserved for v1.5+ enhancements.
    """
    ts:                  str
    agent_id:            str
    invocation_id:       str
    parent_run_id:       Optional[str]
    start_ts:            str
    end_ts:              str
    latency_ms:          int
    success:             bool
    failure_mode:        str
    error_type:          Optional[str]
    error_message:       Optional[str]
    output_schema_valid: Optional[bool]
    n_llm_calls:         int
    total_cost_usd:      float
    n_tool_calls:        Optional[int]
    extra:               dict
    quality_signals:     dict           # NEW v1: extensible quality metrics


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def append_invocation_record(
    record:       AgentInvocationRecord,
    metrics_path: Optional[Path] = None,
) -> None:
    """Append one JSON line to agent metrics ledger.

    Idempotent at file-level (each call adds exactly one line).
    Thread-safe via OS-level append (single open-write-close per call).
    """
    path = metrics_path or _METRICS_PATH_DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(record)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _truncate(s: Optional[str], max_len: int = 500) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ─────────────────────────────────────────────────────────────────────────────
# Decorator: auto-instrument an agent function
# ─────────────────────────────────────────────────────────────────────────────
def track_agent_invocation(
    agent_id:           str,
    schema_validator:   Optional[Callable[[Any], bool]] = None,
    extract_extra:      Optional[Callable[[Any], dict]] = None,
    quality_extractor:  Optional[Callable[[Any], dict]] = None,
):
    """Decorator that wraps an agent function with invocation tracking.

    Args:
        agent_id: matches the agent identifier used in data/llm_cost_ledger.jsonl
        schema_validator: optional fn(result) -> bool checking output structural validity
        extract_extra: optional fn(result) -> dict for agent-specific context fields

    Behavior:
        - Generates invocation_id (UUID4) at start
        - Times the invocation
        - Captures Python exceptions → PYTHON_EXCEPTION failure mode
        - On success: applies schema_validator if provided
        - Joins n_llm_calls + total_cost_usd from cost_ledger by invocation_id
        - Appends one AgentInvocationRecord to agent_slo_metrics.jsonl

    The agent function CAN access the invocation_id via os.environ
    "AGENT_INVOCATION_ID" if it needs to pass to LLM cost ledger entries.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            invocation_id = str(uuid.uuid4())
            parent_run_id = os.environ.get("AGENT_PARENT_RUN_ID")
            # Set invocation_id in env so LLM calls can tag (if cost_ledger writer cooperates)
            old_env = os.environ.get("AGENT_INVOCATION_ID")
            os.environ["AGENT_INVOCATION_ID"] = invocation_id

            start_ts_iso = datetime.datetime.utcnow().isoformat() + "Z"
            t0 = time.monotonic()
            success = False
            failure_mode = FailureMode.NONE.value
            error_type: Optional[str] = None
            error_message: Optional[str] = None
            output_schema_valid: Optional[bool] = None
            extra: dict = {}
            quality_signals: dict = {}
            result = None

            try:
                result = fn(*args, **kwargs)
                success = True

                # Apply schema validator if provided
                if schema_validator is not None:
                    try:
                        valid = bool(schema_validator(result))
                        output_schema_valid = valid
                        if not valid:
                            failure_mode = FailureMode.SCHEMA_BREAK.value
                            success = False
                    except Exception as schema_exc:
                        output_schema_valid = False
                        failure_mode = FailureMode.SCHEMA_BREAK.value
                        success = False
                        error_message = f"schema_validator raised: {schema_exc}"

                # Extract agent-specific context
                if extract_extra is not None:
                    try:
                        extra = extract_extra(result) or {}
                    except Exception as extra_exc:
                        extra = {"extract_extra_error": str(extra_exc)[:100]}

                # Extract agent-specific quality signals (LCS-class proxies, etc.)
                if quality_extractor is not None:
                    try:
                        quality_signals = quality_extractor(result) or {}
                    except Exception as q_exc:
                        quality_signals = {"quality_extractor_error": str(q_exc)[:100]}

            except Exception as e:
                success = False
                failure_mode = FailureMode.PYTHON_EXCEPTION.value
                error_type = type(e).__name__
                error_message = _truncate(traceback.format_exc())
                # Re-raise to preserve calling code's exception handling expectations
                # but record metrics first
                t1 = time.monotonic()
                end_ts_iso = datetime.datetime.utcnow().isoformat() + "Z"
                _emit(
                    AgentInvocationRecord(
                        ts=end_ts_iso, agent_id=agent_id, invocation_id=invocation_id,
                        parent_run_id=parent_run_id,
                        start_ts=start_ts_iso, end_ts=end_ts_iso,
                        latency_ms=int((t1 - t0) * 1000),
                        success=False, failure_mode=failure_mode,
                        error_type=error_type,
                        error_message=_truncate(error_message),
                        output_schema_valid=None, n_llm_calls=0, total_cost_usd=0.0,
                        n_tool_calls=None, extra=extra, quality_signals=quality_signals,
                    )
                )
                _restore_env(old_env)
                raise

            # Successful path (or schema-failed but no Python exception)
            t1 = time.monotonic()
            end_ts_iso = datetime.datetime.utcnow().isoformat() + "Z"
            llm_stats = _aggregate_llm_calls_for_invocation(invocation_id)

            _emit(
                AgentInvocationRecord(
                    ts=end_ts_iso, agent_id=agent_id, invocation_id=invocation_id,
                    parent_run_id=parent_run_id,
                    start_ts=start_ts_iso, end_ts=end_ts_iso,
                    latency_ms=int((t1 - t0) * 1000),
                    success=success, failure_mode=failure_mode,
                    error_type=error_type,
                    error_message=_truncate(error_message),
                    output_schema_valid=output_schema_valid,
                    n_llm_calls=llm_stats["n_calls"],
                    total_cost_usd=llm_stats["cost_usd"],
                    n_tool_calls=extra.get("n_tool_calls"),
                    extra=extra, quality_signals=quality_signals,
                )
            )
            _restore_env(old_env)
            return result

        return wrapper
    return decorator


def _emit(record: AgentInvocationRecord) -> None:
    try:
        append_invocation_record(record)
    except Exception as e:
        logger.warning("agent observability: failed to write metric: %s", e)


def _restore_env(old_env: Optional[str]) -> None:
    if old_env is None:
        os.environ.pop("AGENT_INVOCATION_ID", None)
    else:
        os.environ["AGENT_INVOCATION_ID"] = old_env


# ─────────────────────────────────────────────────────────────────────────────
# Trace recording (Gap #1 fix, 2026-05-15) — per-tool-call detail capture
#
# LLM calls are already trace-able via llm_cost_ledger.jsonl (each entry tagged
# with extra.invocation_id). Tool calls (Watchdog ReAct dispatch) were not.
# This layer adds an agent_trace_log.jsonl tool-event ledger so that any
# invocation_id can be reconstructed into a full chronological tree of events
# via load_trace_for_invocation().
#
# Doctrine compliance:
#   - 0-LLM-in-DECISION preserved (this layer is pure measurement)
#   - 7-agent ceiling unchanged (no new agents)
#   - LLM-risk-side rule unchanged (does not change forensic/devils_advocate
#     interfaces — those stay LLM-risk-side and stay 1-shot LLM)
# ─────────────────────────────────────────────────────────────────────────────
_TRACE_LOG_PATH_DEFAULT = _REPO_ROOT / "data" / "agent_trace_log.jsonl"


def record_tool_call(
    *,
    tool_name:       str,
    args_preview:    str = "",
    result_preview:  str = "",
    latency_ms:      int = 0,
    success:         bool = True,
    error_message:   Optional[str] = None,
    invocation_id:   Optional[str] = None,
    trace_path:      Optional[Path] = None,
) -> None:
    """Append a tool_call event to data/agent_trace_log.jsonl.

    Auto-joins to current AGENT_INVOCATION_ID (set by @track_agent_invocation)
    via os.environ — mirrors llm_cost_ledger.record_call() join pattern, so
    no contextvar refactor needed.

    args_preview / result_preview truncated to 200 chars (privacy + storage).
    If invocation_id is None and AGENT_INVOCATION_ID unset, the event is
    dropped silently (we don't want tool calls outside an agent context
    polluting the trace ledger).
    """
    inv_id = invocation_id or os.environ.get("AGENT_INVOCATION_ID")
    if not inv_id:
        return

    entry = {
        "ts":              datetime.datetime.utcnow().isoformat() + "Z",
        "invocation_id":   inv_id,
        "event_type":      "tool_call",
        "tool_name":       str(tool_name),
        "args_preview":    _truncate(args_preview, max_len=200),
        "result_preview":  _truncate(result_preview, max_len=200),
        "latency_ms":      int(latency_ms),
        "success":         bool(success),
        "error_message":   _truncate(error_message, max_len=200),
    }

    path = trace_path or _TRACE_LOG_PATH_DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        # Tracing failures must not break the agent itself.
        logger.warning("record_tool_call: write failed: %s", e)


@dataclasses.dataclass
class TraceEvent:
    """One event inside an invocation trace tree (LLM call or tool call)."""
    ts:              str
    event_type:      str       # "llm_call" or "tool_call"
    name:            str       # provider+model for LLM, tool_name for tool
    latency_ms:      int
    success:         bool
    detail:          dict      # event-type-specific payload (redacted)


@dataclasses.dataclass
class TraceTree:
    """Reconstructed trace for one agent invocation.

    invocation: the AgentInvocationRecord dict from agent_slo_metrics.jsonl
    events:     chronologically sorted child events (LLM + tool)
    """
    invocation:    dict
    events:        list[TraceEvent]


def load_trace_for_invocation(
    invocation_id:    str,
    metrics_path:     Optional[Path] = None,
    cost_path:        Optional[Path] = None,
    trace_path:       Optional[Path] = None,
) -> Optional[TraceTree]:
    """Reconstruct full trace tree for one invocation.

    Joins:
      - agent_slo_metrics.jsonl  → root invocation
      - llm_cost_ledger.jsonl    → LLM events (tagged extra.invocation_id)
      - agent_trace_log.jsonl    → tool events (tagged invocation_id)

    Returns None if invocation_id not found in metrics ledger.

    Performance: scans last ~5MB of each file (recent-window optimization).
    For typical 100 invocations/day this covers ~3 days, ample for drill use.
    """
    m_path = metrics_path or _METRICS_PATH_DEFAULT
    c_path = cost_path or (_REPO_ROOT / "data" / "llm_cost_ledger.jsonl")
    t_path = trace_path or _TRACE_LOG_PATH_DEFAULT

    def _iter_recent_jsonl(path: Path, window_bytes: int = 5_000_000):
        """Yield parsed JSON dicts from the tail of a JSONL file.

        Reads up to the last `window_bytes` bytes. Only discards a partial
        leading line when we actually had to seek past byte 0 (small-file
        bug fix 2026-05-15: previously we always discarded the first line,
        losing 1 event in files smaller than the window).
        """
        with open(path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            start = max(0, size - window_bytes)
            f.seek(start)
            if start > 0:
                f.readline()    # discard partial line only if we mid-seeked
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # 1. Find root invocation
    root: Optional[dict] = None
    if m_path.exists():
        try:
            for d in _iter_recent_jsonl(m_path):
                if d.get("invocation_id") == invocation_id:
                    root = d
        except Exception as e:
            logger.debug("load_trace: metrics scan failed: %s", e)
    if root is None:
        return None

    events: list[TraceEvent] = []

    # 2. LLM events from cost ledger
    if c_path.exists():
        try:
            for d in _iter_recent_jsonl(c_path):
                extra = d.get("extra") or {}
                if extra.get("invocation_id") != invocation_id:
                    continue
                events.append(TraceEvent(
                    ts=str(d.get("ts", "")),
                    event_type="llm_call",
                    name=f"{d.get('provider','?')}/{d.get('model','?')}",
                    latency_ms=int(d.get("latency_ms") or 0),
                    success=True,
                    detail={
                        "prompt_tokens":     d.get("prompt_tokens"),
                        "completion_tokens": d.get("completion_tokens"),
                        "cost_usd":          d.get("cost_usd"),
                        "scope":             d.get("scope"),
                        "prompt_hash":       extra.get("prompt_hash"),
                    },
                ))
        except Exception as e:
            logger.debug("load_trace: cost scan failed: %s", e)

    # 3. Tool events from trace log
    if t_path.exists():
        try:
            for d in _iter_recent_jsonl(t_path):
                if d.get("invocation_id") != invocation_id:
                    continue
                events.append(TraceEvent(
                    ts=str(d.get("ts", "")),
                    event_type="tool_call",
                    name=str(d.get("tool_name", "?")),
                    latency_ms=int(d.get("latency_ms") or 0),
                    success=bool(d.get("success", True)),
                    detail={
                        "args_preview":   d.get("args_preview"),
                        "result_preview": d.get("result_preview"),
                        "error_message":  d.get("error_message"),
                    },
                ))
        except Exception as e:
            logger.debug("load_trace: tool scan failed: %s", e)

    # 4. Sort chronologically
    events.sort(key=lambda e: e.ts)
    return TraceTree(invocation=root, events=events)


def _aggregate_llm_calls_for_invocation(invocation_id: str) -> dict:
    """Scan cost_ledger.jsonl for entries tagged with this invocation_id.

    Returns: {"n_calls": int, "cost_usd": float}

    Best-effort: cost_ledger may not yet have invocation_id tagging
    (legacy entries). Returns 0/0 if no matches.
    """
    cost_ledger_path = _REPO_ROOT / "data" / "llm_cost_ledger.jsonl"
    if not cost_ledger_path.exists():
        return {"n_calls": 0, "cost_usd": 0.0}

    n_calls = 0
    cost_usd = 0.0
    try:
        # Only scan recent entries to avoid full-file IO.
        # Small-file bug fix 2026-05-15: only discard a partial leading line
        # when we actually seeked past byte 0; otherwise we lose 1 entry
        # in files smaller than the window.
        with open(cost_ledger_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            start = max(0, size - 5_000_000)
            f.seek(start)
            if start > 0:
                f.readline()
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = d.get("extra") or {}
                if extra.get("invocation_id") == invocation_id:
                    n_calls += 1
                    cost_usd += float(d.get("cost_usd") or 0.0)
    except Exception as e:
        logger.debug("aggregate_llm_calls failed: %s", e)
    return {"n_calls": n_calls, "cost_usd": cost_usd}


# ─────────────────────────────────────────────────────────────────────────────
# Compliance computation helpers (used by dashboard + Watchdog rule)
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class AgentSLO:
    """Per-agent compliance target locked in spec_agent_observability_v1."""
    agent_id:           str
    latency_p95_ms:     int       # max acceptable p95 latency
    success_rate_min:   float     # minimum success rate (0..1)
    cost_per_run_max:   float     # max acceptable cost per invocation
    schema_valid_min:   float     # min schema validity rate
    description:        str = ""


# Default compliance targets (anchored to current observed values + 50% buffer)
DEFAULT_AGENT_SLOS: dict[str, AgentSLO] = {
    "ops_watchdog": AgentSLO(
        agent_id="ops_watchdog",
        latency_p95_ms=180_000,           # 3 min p95
        success_rate_min=0.95,
        cost_per_run_max=0.10,
        schema_valid_min=0.98,
        description="Daily 06:10 SGT monitoring agent; ReAct with Gemini 2.5 Flash",
    ),
    "forensic_devils_advocate": AgentSLO(
        agent_id="forensic_devils_advocate",
        latency_p95_ms=120_000,           # 2 min p95
        success_rate_min=0.95,
        cost_per_run_max=0.05,
        schema_valid_min=0.98,
        description="Dual-LLM Gemini PRIMARY + DeepSeek DEVIL consistency check",
    ),
    "forensic_news_context": AgentSLO(
        agent_id="forensic_news_context",
        latency_p95_ms=90_000,            # 90s p95
        success_rate_min=0.90,
        cost_per_run_max=0.02,
        schema_valid_min=0.95,
        description="DD investigation news context summarizer (AV + Yahoo RSS + Vertex)",
    ),
}


def load_metrics(
    metrics_path: Optional[Path] = None,
    days_lookback: int = 30,
) -> list[dict]:
    """Load recent agent invocation metrics from JSONL ledger."""
    path = metrics_path or _METRICS_PATH_DEFAULT
    if not path.exists():
        return []
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days_lookback)
    cutoff_iso = cutoff.isoformat() + "Z"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("ts", "") >= cutoff_iso:
                records.append(d)
    return records


def compute_agent_slo_compliance(
    agent_id:    str,
    records:     list[dict],
    slo:         Optional[AgentSLO] = None,
) -> dict:
    """Compute compliance for one agent over given record set."""
    if slo is None:
        slo = DEFAULT_AGENT_SLOS.get(agent_id)
    if slo is None:
        return {"error": f"no compliance target defined for {agent_id}"}

    agent_records = [r for r in records if r.get("agent_id") == agent_id]
    if not agent_records:
        return {
            "agent_id": agent_id, "n_invocations": 0,
            "compliance_overall": "INSUFFICIENT_DATA",
            "latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None,
            "success_rate": None, "cost_p95_usd": None, "schema_valid_rate": None,
        }

    latencies = sorted(int(r.get("latency_ms") or 0) for r in agent_records)
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[min(n - 1, int(0.95 * n))]
    p99 = latencies[min(n - 1, int(0.99 * n))]

    success_rate = sum(1 for r in agent_records if r.get("success")) / n
    costs = [float(r.get("total_cost_usd") or 0.0) for r in agent_records]
    cost_p95 = sorted(costs)[min(n - 1, int(0.95 * n))]

    schema_records = [r for r in agent_records if r.get("output_schema_valid") is not None]
    if schema_records:
        schema_valid_rate = sum(1 for r in schema_records if r.get("output_schema_valid")) / len(schema_records)
    else:
        schema_valid_rate = None

    # Compliance per target
    latency_pass = p95 <= slo.latency_p95_ms
    success_pass = success_rate >= slo.success_rate_min
    cost_pass = cost_p95 <= slo.cost_per_run_max
    schema_pass = (schema_valid_rate is None) or (schema_valid_rate >= slo.schema_valid_min)

    all_pass = latency_pass and success_pass and cost_pass and schema_pass
    overall = "PASS" if all_pass else "FAIL"

    return {
        "agent_id": agent_id, "n_invocations": n,
        "latency_p50_ms": p50, "latency_p95_ms": p95, "latency_p99_ms": p99,
        "latency_p95_pass": latency_pass, "latency_p95_target_ms": slo.latency_p95_ms,
        "success_rate": success_rate, "success_pass": success_pass,
        "success_target_min": slo.success_rate_min,
        "cost_p95_usd": cost_p95, "cost_pass": cost_pass,
        "cost_target_max": slo.cost_per_run_max,
        "schema_valid_rate": schema_valid_rate, "schema_pass": schema_pass,
        "schema_target_min": slo.schema_valid_min,
        "compliance_overall": overall,
    }
