"""engine/research/trace_log.py — Phase 4f: lightweight tracing.

OpenTelemetry-style nested-span observability, but DIY:
  - Stores spans to data/research/spans.jsonl (append-only)
  - Uses contextvars for parent / trace propagation (async-safe)
  - No Jaeger / Tempo / SDK dependency — single-user lab doesn't need
    a separate trace UI; the Cockpit reads from spans.jsonl directly

Span shape:
  {
    "trace_id":   "...",      # rooted at the workflow / request
    "span_id":    "...",
    "parent_id":  "..." | None,
    "name":       "tool.query_graveyard",
    "start_ms":   1700000000000,
    "end_ms":     1700000000150,
    "duration_ms": 150,
    "ok":         true / false,
    "attrs":      {...},      # arbitrary tags (workflow_id, agent_name, ...)
    "error":      "..." | None
  }

Usage:
  with span("tool.query_graveyard", workflow_id="l4-abc", args={...}):
      result = do_thing()
      add_attr(result_size=len(result))

The span() context manager:
  - Generates a new span_id
  - Sets parent_id from contextvar
  - Pushes self onto contextvar so child spans nest correctly
  - Records start_ms on enter, end_ms + duration_ms + ok on exit
  - Captures exceptions → records error attr, ok=False, re-raises
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPANS_LEDGER = REPO_ROOT / "data" / "research" / "spans.jsonl"

# Context vars carrying the current trace + active span across async hops.
_current_trace_id: contextvars.ContextVar[Optional[str]] = \
    contextvars.ContextVar("trace_id", default=None)
_current_span_id: contextvars.ContextVar[Optional[str]] = \
    contextvars.ContextVar("span_id", default=None)
_current_attrs: contextvars.ContextVar[Optional[dict]] = \
    contextvars.ContextVar("attrs", default=None)


def _gen_id(prefix: str = "") -> str:
    """Short hex id. 12 hex = 48 bits of entropy = adequate for lab."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _append_span_row(row: dict) -> None:
    """Best-effort persistence. Never raises into the traced code path."""
    try:
        SPANS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with SPANS_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.exception("trace_log persist failed (non-fatal)")


def get_trace_id() -> Optional[str]:
    """Read the current trace id (None if no span open).  External callers
    use this to thread the trace through HTTP headers / Temporal attrs."""
    return _current_trace_id.get()


def start_trace(trace_id: Optional[str] = None, **root_attrs: Any) -> str:
    """Explicitly set the trace root (used by REST entry / Temporal
    workflow start so all children share one trace_id). Returns the id
    so callers can stash it for cross-process propagation."""
    tid = trace_id or _gen_id("trace-")
    _current_trace_id.set(tid)
    _current_span_id.set(None)
    if root_attrs:
        # carry root attributes downward to be merged with every span
        _current_attrs.set(dict(root_attrs))
    return tid


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[dict]:
    """Open a span. Re-entrant + async-safe via contextvars.

    Yields the in-progress span dict so caller can add_attr() mid-span.
    """
    # Auto-bootstrap a trace if caller didn't explicitly start one
    trace_id = _current_trace_id.get()
    if trace_id is None:
        trace_id = _gen_id("trace-")
        _current_trace_id.set(trace_id)

    span_id = _gen_id("span-")
    parent_id = _current_span_id.get()
    inherited = _current_attrs.get() or {}
    record = {
        "trace_id":  trace_id,
        "span_id":   span_id,
        "parent_id": parent_id,
        "name":      name,
        "start_ms":  int(time.time() * 1000),
        "attrs":     {**inherited, **attrs},
        "ok":        True,
        "error":     None,
        "ended":     False,
    }

    tok_span = _current_span_id.set(span_id)
    try:
        yield record
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        _current_span_id.reset(tok_span)
        record["end_ms"] = int(time.time() * 1000)
        record["duration_ms"] = record["end_ms"] - record["start_ms"]
        record["ended"] = True
        # Persist a clean copy (don't write the live mutable dict
        # so add_attr mid-span doesn't race on flush)
        persisted = {k: v for k, v in record.items() if k != "ended"}
        _append_span_row(persisted)


def add_attr(**kv: Any) -> None:
    """Add attributes to the currently-open span. No-op outside a span.

    Inherits any root attrs so workflow_id/etc filters still match
    the attr_update marker alongside its parent span."""
    sid = _current_span_id.get()
    if sid is None:
        return
    inherited = _current_attrs.get() or {}
    _append_span_row({
        "trace_id":  _current_trace_id.get(),
        "span_id":   sid,
        "kind":      "attr_update",
        "ts_ms":     int(time.time() * 1000),
        "attrs":     {**inherited, **kv},
    })


# ── Readers ────────────────────────────────────────────────────────────


def read_spans(
    trace_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    limit: int = 500,
) -> list[dict]:
    """Read recent spans newest-first. Filter by trace_id or by
    workflow_id (matched against attrs.workflow_id)."""
    if not SPANS_LEDGER.is_file():
        return []
    out: list[dict] = []
    with SPANS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "attr_update":
                # Attr-update markers are kept in stream for now; UI
                # can fold them into the parent span if it wants.
                pass
            if trace_id and r.get("trace_id") != trace_id:
                continue
            if workflow_id:
                attrs = r.get("attrs") or {}
                if attrs.get("workflow_id") != workflow_id:
                    continue
            out.append(r)
    out.reverse()
    return out[: max(1, limit)]


def reset_for_test() -> None:
    """Tests use this to wipe the spans ledger + reset contextvars."""
    if SPANS_LEDGER.is_file():
        SPANS_LEDGER.unlink()
    _current_trace_id.set(None)
    _current_span_id.set(None)
    _current_attrs.set(None)
