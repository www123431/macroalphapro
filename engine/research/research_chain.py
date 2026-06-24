"""engine/research/research_chain.py — Frontier 4 (2026-06-01):
multi-step research chains.

Declarative DAG runner over the existing tool registry. A Chain is a
sequence of Steps; each Step invokes a tool, can guard execution on
prior step outputs, and contributes its result to the chain context
for downstream steps.

NOT autonomous agents looping over tools forever — chains are
FINITE, DECLARATIVE, AUDITABLE. The choice of what comes after
"find_paper" is decided at chain-definition time (by the senior),
NOT by an LLM at runtime. This makes chain behavior reproducible
and chain failures debuggable.

Why declarative not LLM-orchestrated:
  - LLM-orchestrated chains drift — the same prompt produces a
    5-step chain one day and a 12-step chain the next.
  - Declarative chains compose with the pre-commit audit ledger:
    every chain run produces the same step shape, so calibration /
    cost tracking / failure mode analysis stay consistent.
  - One source of truth: chains call the SAME tools from llm_tools.TOOLS
    that the council uses. Adding a tool there auto-exposes it to
    chains. Single registry doctrine.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_RUNS_LEDGER = REPO_ROOT / "data" / "research" / "chain_runs.jsonl"


# ── Chain definition ──────────────────────────────────────────────────


@dataclass
class Step:
    """One step in a chain.

    name:    unique within the chain; downstream steps reference outputs
             via "{{steps.<name>.<field>}}" in args / guard.
    tool:    a name in engine.research.llm_tools.TOOLS — keeps the
             chain inside the audited tool registry. To call an LLM
             or run the pipeline, route through a tool wrapper (see
             chain_library) rather than special-casing here.
    args:    dict of tool args. String values may contain template
             references like "{{steps.find_paper.result_dict.title}}"
             which are resolved against the chain context at runtime.
    guard:   optional condition string evaluated against chain context.
             If falsy, step is SKIPPED with status="skipped". Use the
             same template syntax for context refs.
    on_failure: "halt" (default) stops the chain; "continue" records
             the error + moves to the next step. Use "continue" for
             non-critical lookups (e.g. arxiv search optional).
    """
    name:        str
    tool:        str
    args:        dict = field(default_factory=dict)
    guard:       Optional[str] = None
    on_failure:  str = "halt"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chain:
    """Static chain definition.

    Chains live in code (engine.research.chain_library) so they get
    code review + version control. Defining chains in YAML would let
    users hot-swap them; that's deferred until we know which chains
    are stable enough to externalize.
    """
    chain_id:    str
    description: str
    steps:       list[Step]

    def to_dict(self) -> dict:
        return {
            "chain_id":    self.chain_id,
            "description": self.description,
            "steps":       [s.to_dict() for s in self.steps],
        }


# ── Step result ──────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Outcome of one Step execution."""
    name:          str
    status:        str   # "ok" / "skipped" / "failed"
    tool:          str
    args_resolved: dict = field(default_factory=dict)
    result:        Any = None
    error:         Optional[str] = None
    elapsed_ms:    float = 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        # Result may be a large dict — truncate the serialized form
        # in the ledger but keep the live in-memory object for
        # downstream steps to reference.
        return out


@dataclass
class ChainRun:
    """Full chain execution record."""
    chain_id:     str
    run_id:       str
    started_at:   str
    finished_at:  Optional[str]
    status:       str   # "completed" / "halted" / "running"
    steps:        list[StepResult]
    initial_context: dict
    elapsed_s:    float = 0.0

    def to_dict(self) -> dict:
        return {
            "chain_id":        self.chain_id,
            "run_id":          self.run_id,
            "started_at":      self.started_at,
            "finished_at":     self.finished_at,
            "status":          self.status,
            "initial_context": self.initial_context,
            "elapsed_s":       round(self.elapsed_s, 2),
            "steps":           [s.to_dict() for s in self.steps],
        }


# ── Template + guard resolution ───────────────────────────────────────

_TEMPLATE_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def _resolve_path(path: str, ctx: dict) -> Any:
    """Resolve a dotted path like 'steps.find_paper.result.title' against ctx.

    Returns None on any missing segment (safer than KeyError mid-chain;
    callers can guard on truthiness)."""
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            try:
                cur = getattr(cur, part)
            except AttributeError:
                return None
    return cur


def _resolve_template(value: Any, ctx: dict) -> Any:
    """Replace `{{...}}` references in strings; recurse into dicts/lists.

    Non-string scalars pass through. If a string is ENTIRELY one ref
    (i.e. "{{foo}}"), we return the resolved object (preserving type:
    dict / int / etc.). For mixed strings ("{{foo}} happened") we
    coerce to string.
    """
    if isinstance(value, str):
        m_full = _TEMPLATE_RE.fullmatch(value.strip())
        if m_full:
            return _resolve_path(m_full.group(1).strip(), ctx)
        def repl(m: re.Match) -> str:
            r = _resolve_path(m.group(1).strip(), ctx)
            return "" if r is None else str(r)
        return _TEMPLATE_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_template(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_template(v, ctx) for v in value]
    return value


def _eval_guard(guard: Optional[str], ctx: dict) -> bool:
    """Evaluate a guard. Truthy → run; falsy → skip.

    Implementation is intentionally restricted: we resolve any template
    references, then return Python truthiness of the result. NO eval()
    of arbitrary expressions — chain definitions live in code reviewed
    by the senior, but defense in depth is cheap. Multi-clause guards
    (e.g. AND/OR) require the chain to split into multiple steps with
    individual single-template guards.
    """
    if not guard:
        return True
    resolved = _resolve_template(guard, ctx)
    return bool(resolved)


# ── Chain runner ─────────────────────────────────────────────────────


def run_chain(
    chain: Chain,
    *,
    initial_context: Optional[dict] = None,
    tool_dispatcher: Optional[Callable[[str, dict], Any]] = None,
) -> ChainRun:
    """Execute a chain. Synchronous; LLM-tool calls block.

    tool_dispatcher defaults to engine.research.llm_tools.dispatch —
    the same dispatcher the MCP server + REST shim use. Tests can
    inject a fake.
    """
    if tool_dispatcher is None:
        from engine.research.llm_tools import dispatch as _dispatch
        def tool_dispatcher(name, args):  # type: ignore[no-redef]
            return _dispatch(name, **args)

    run_id = f"chain-{uuid.uuid4().hex[:12]}"
    started = _dt.datetime.utcnow()
    ctx: dict = {"steps": {}, "initial": initial_context or {}}

    step_results: list[StepResult] = []
    status = "completed"
    t0 = time.perf_counter()

    for step in chain.steps:
        # Guard
        if not _eval_guard(step.guard, ctx):
            step_results.append(StepResult(
                name=step.name, status="skipped",
                tool=step.tool, args_resolved={},
                result=None, elapsed_ms=0.0,
            ))
            ctx["steps"][step.name] = {"status": "skipped", "result": None}
            continue

        # Resolve args
        try:
            args_resolved = _resolve_template(step.args, ctx) or {}
        except Exception as exc:
            logger.exception("arg resolution failed for step %s", step.name)
            step_results.append(StepResult(
                name=step.name, status="failed", tool=step.tool,
                error=f"arg resolution: {exc}", elapsed_ms=0.0,
            ))
            if step.on_failure == "halt":
                status = "halted"
                break
            continue

        # Dispatch
        s_t0 = time.perf_counter()
        try:
            result = tool_dispatcher(step.tool, args_resolved)
            elapsed_ms = (time.perf_counter() - s_t0) * 1000.0
            sr = StepResult(
                name=step.name, status="ok", tool=step.tool,
                args_resolved=args_resolved, result=result,
                elapsed_ms=round(elapsed_ms, 1),
            )
            ctx["steps"][step.name] = {"status": "ok", "result": result}
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - s_t0) * 1000.0
            logger.exception("step %s failed", step.name)
            sr = StepResult(
                name=step.name, status="failed", tool=step.tool,
                args_resolved=args_resolved, error=str(exc)[:300],
                elapsed_ms=round(elapsed_ms, 1),
            )
            ctx["steps"][step.name] = {"status": "failed", "error": str(exc)[:300]}
            step_results.append(sr)
            if step.on_failure == "halt":
                status = "halted"
                break
            continue

        step_results.append(sr)

    elapsed_s = time.perf_counter() - t0
    finished = _dt.datetime.utcnow()
    chain_run = ChainRun(
        chain_id=chain.chain_id,
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds") + "Z",
        finished_at=finished.isoformat(timespec="seconds") + "Z",
        status=status,
        steps=step_results,
        initial_context=initial_context or {},
        elapsed_s=elapsed_s,
    )
    _append_chain_run_log(chain_run)
    return chain_run


# ── Ledger ────────────────────────────────────────────────────────────


def _truncate_for_ledger(obj: Any, max_chars: int = 1500) -> Any:
    """Recursively truncate large strings so the ledger row stays small."""
    if isinstance(obj, str):
        return obj if len(obj) <= max_chars else obj[: max_chars] + "...(truncated)"
    if isinstance(obj, dict):
        return {k: _truncate_for_ledger(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_for_ledger(v, max_chars) for v in obj]
    return obj


def _append_chain_run_log(run: ChainRun) -> None:
    """Persist run to JSONL. Best-effort + truncating to keep rows
    bounded (tool results can be megabytes)."""
    try:
        CHAIN_RUNS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = _truncate_for_ledger(run.to_dict())
        with CHAIN_RUNS_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.exception("chain run log append failed (non-fatal)")


def read_recent_chain_runs(
    *,
    chain_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Read recent chain runs newest-first."""
    if not CHAIN_RUNS_LEDGER.is_file():
        return []
    out: list[dict] = []
    with CHAIN_RUNS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if chain_id and r.get("chain_id") != chain_id:
                continue
            out.append(r)
    out.reverse()
    return out[: max(1, int(limit))]
