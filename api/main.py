"""api/main.py — FastAPI app exposing the quant-fund engine to the new frontend.

Phase 0 (UI migration): read-only endpoints over existing engine functions + the agent
constellation, plus an SSE chat endpoint that streams the existing chat_turn loop. Adding a
route never touches the engine or Streamlit; this layer is the strangler-fig seam.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the repo root importable when run as `uvicorn api.main:app` from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(
    title="MacroAlphaPro API",
    version="0.1.0",
    description="Institutional quant-fund backend: book state, decay monitor, agent "
                "constellation. 0-LLM-in-DECISION — endpoints serve deterministic engine "
                "output; the agent layer narrates only.",
)

# CORS for the Next.js dev server (and a future deployed origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase A.2: SSE streaming endpoint for live candidate_pipeline progress
try:
    from api.routes_pipeline_stream import router as _pipeline_stream_router
    app.include_router(_pipeline_stream_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("pipeline stream router failed to load: %s", _exc)

# Phase 4a.6: shared REST shim over the 9 Session-3 research tools.
# Serves BOTH the Cockpit (monitoring + future write controls) and
# Assistant (visualization) UI sections. One source of truth, one
# audit ledger (data/research/ui_tool_calls.jsonl) — adding a tool
# to engine.research.llm_tools.TOOLS auto-exposes it on MCP + REST.
try:
    from api.routes_research_tools import router as _research_tools_router
    app.include_router(_research_tools_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("research tools router failed to load: %s", _exc)

# Paper-driven research chain endpoints (T7 backend) —
# /api/paper_chain/papers, /hypotheses, /forward-vectors, /lessons, /overview.
# Read-only; the chain doctrine (2026-06-04 locked) is enforced upstream
# at the schema layer (see engine.research_store.red_lessons.schema
# .GroundingMethod + PRETRAIN_GROUNDED_FREEZE_TS).
try:
    from api.routes_paper_chain import router as _paper_chain_router
    app.include_router(_paper_chain_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("paper_chain router failed to load: %s", _exc)

# F14a (2026-06-05) — autopilot dry-run preview surface.
# /api/autopilot/dry-run/latest      → JSON plan (deterministic recompute)
# /api/autopilot/dry-run/latest.md   → last cron's rendered markdown
# READ-ONLY pre-F14b. No side effects / LLM / compose() calls.
try:
    from api.routes_autopilot import router as _autopilot_router
    app.include_router(_autopilot_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("autopilot router failed to load: %s", _exc)

# Phase 1.6 (2026-06-05) — papers_curator daily digest surface.
# /api/papers_curator/incoming → joined cache + judgments + summaries
# /api/papers_curator/skip     → mark a candidate as user-skipped
try:
    from api.routes_papers_curator import router as _papers_curator_router
    app.include_router(_papers_curator_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("papers_curator router failed to load: %s", _exc)

# Phase 2.0 step 12 (2026-06-06) — strengthener (Employee B) verdicts.
# /api/strengthener/approvals               → pending B verdicts (APPROVE / AMENDMENT)
# /api/strengthener/approvals/resolve POST  → record principal's decision
try:
    from api.routes_strengthener import router as _strengthener_router
    app.include_router(_strengthener_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("strengthener router failed to load: %s", _exc)

# Phase 2.0 step 15 (2026-06-07) — chief_of_staff weekly orchestrator.
# POST /api/chief_of_staff/run  → one weekly session (D → A → B → memo)
try:
    from api.routes_chief_of_staff import router as _cos_router
    app.include_router(_cos_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("chief_of_staff router failed to load: %s", _exc)

# R2.5 — session event SSE tail + forward-approval polling for Claude
# hooks. Closes the bidirectional collab loop: UI sees Claude's emits
# live; Claude sees the user's approves without manual hand-off.
try:
    from api.routes_sessions_stream import router as _sessions_stream_router
    app.include_router(_sessions_stream_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("sessions stream router failed to load: %s", _exc)

# R3.1 — typed-intent persistence so page-level CTAs ("Audit session",
# "Pipeline test", "Open session →") file a structured intent Claude
# can poll on a hook, instead of being silent URL-jumps that Claude
# can't see.
try:
    from api.routes_intents import router as _intents_router
    app.include_router(_intents_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("intents router failed to load: %s", _exc)

# R4.2 — local usage telemetry. Append-only jsonl tracking page
# views + key event clicks so future trim decisions have data.
try:
    from api.routes_telemetry import router as _telemetry_router
    app.include_router(_telemetry_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("telemetry router failed to load: %s", _exc)

# Operator Console (2026-06-23) — UI-triggered pipeline stations.
# Foundation: schema + station ABC + typed event store + cost ledger +
# routes. Phase 0a ships with empty station registry; stations attach
# in subsequent phases per docs/architecture/operator_console.md.
try:
    from api.routes_operator_console import router as _operator_console_router
    app.include_router(_operator_console_router)
    # Mark abandoned `running` jobs from prior server invocation as
    # `recovered_unknown` (R6 lossy-restart mitigation).
    from engine.operator_console import store as _opcon_store
    from engine.operator_console.schema import JobState as _JobState
    _orphans = _opcon_store.scan_orphaned_running_jobs()
    for _jid in _orphans:
        _opcon_store.update_job_state(_jid, state=_JobState.RECOVERED_UNKNOWN)
    if _orphans:
        import logging as _l
        _l.getLogger(__name__).warning(
            "operator_console: recovered %d orphaned running jobs on startup",
            len(_orphans),
        )
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("operator_console router failed to load: %s", _exc)

# P1-E — global data search (Cmd-K free-text across papers /
# hypotheses / lessons / sleeves). Backs the CommandPalette's
# "Data" group so users can type "momentum" and see every
# matching artifact, not just routes.
try:
    from api.routes_global_search import router as _global_search_router
    app.include_router(_global_search_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("global search router failed to load: %s", _exc)

# Composer (C.4 2026-06-05) — spec → atomic components → series.
# The end of the project's epistemic backbone: every verdict is now
# reproducible end-to-end from a spec_hash.
try:
    from api.routes_composer import router as _composer_router
    app.include_router(_composer_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("composer router failed to load: %s", _exc)

# Hypothesis spec (B.3-B.4 2026-06-05) — typed structured spec layer.
# The project's epistemic backbone: every claim has a deterministic
# spec_hash so re-running the same hypothesis is reproducible.
try:
    from api.routes_hypothesis_spec import router as _hypothesis_spec_router
    app.include_router(_hypothesis_spec_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("hypothesis_spec router failed to load: %s", _exc)

# Series factory (2026-06-05) — build returns series on demand per
# family. Closes the 'pick a parquet' UX gap: every hypothesis with a
# registered family becomes auto-buildable instead of forcing the user
# to manually map onto unrelated cached parquets.
try:
    from api.routes_series_factory import router as _series_factory_router
    app.include_router(_series_factory_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("series_factory router failed to load: %s", _exc)

# UI U4 (2026-06-05) — unified agent activity feed for the Lab sidebar.
try:
    from api.routes_agent_activity import router as _agent_activity_router
    app.include_router(_agent_activity_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("agent activity router failed to load: %s", _exc)

# Agentic Phase 2.2 (2026-06-05) — workflow_executor control plane.
# Status/pause/resume/run endpoints for the autonomous workflow runner.
try:
    from api.routes_workflow_executor import router as _workflow_executor_router
    app.include_router(_workflow_executor_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("workflow_executor router failed to load: %s", _exc)

# Agentic Phase 1.2 (2026-06-04) — AgentHealth dashboard.
# Scans agent activity logs and surfaces "last ran X ago, OK/error" per
# agent so the user can SEE that autonomous agents are working. Without
# this, autonomy is invisible.
try:
    from api.routes_agent_health import router as _agent_health_router
    app.include_router(_agent_health_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("agent health router failed to load: %s", _exc)

# AI-native N1 (2026-06-04) — "Book 当日简报" 早间叙事 (中文).
# Daily LLM-generated 3-paragraph memo synthesizing book health +
# research pipeline + watch items. Cached per-day so cost is ~$1/month.
try:
    from api.routes_daily_memo import router as _daily_memo_router
    app.include_router(_daily_memo_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("daily memo router failed to load: %s", _exc)

# AI-native Step 3 (2026-06-04) — "员工感" Chief of Staff Daily Directive.
# Aggregates state across all surfaces (DQ / decay / queue / sessions /
# intents / audit_verifier results) and tells the user what to do today.
# Pure deterministic aggregation; LLM narration can layer on later.
try:
    from api.routes_daily_directive import router as _daily_directive_router
    app.include_router(_daily_directive_router)
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("daily directive router failed to load: %s", _exc)

# AI-native Step 2 (2026-06-04) — import the audit_verifier subscriber.
# Module import is the side-effect that wires it to the EventBus
# (factor_verdict_filed -> verify_factor_verdict_lineage). First real
# autonomous agentic loop in the codebase: every factor_verdict_filed
# emit now triggers C1-C4 lineage checks synchronously, results go to
# data/audit_verifier/lineage_results.jsonl. See file docstring.
try:
    import engine.agents.audit_verifier  # noqa: F401
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("audit_verifier failed to load: %s", _exc)

# AI-native Step 4 (2026-06-04) — second reactive subscriber.
# graveyard_collision listens for intent_filed events and writes a
# warnings.jsonl row when a candidate's family + name + claim collides
# with a past RED verdict. Addresses the "我会不会重复测一个 RED" anxiety.
try:
    import engine.agents.graveyard_collision  # noqa: F401
except Exception as _exc:
    import logging as _l
    _l.getLogger(__name__).warning("graveyard_collision failed to load: %s", _exc)

# ── Observability: request-id + structured access log + per-route latency metrics ─────────────
import collections as _collections  # noqa: E402
import logging as _logging  # noqa: E402
import uuid as _uuid  # noqa: E402

_access_log = _logging.getLogger("api.access")
# path -> {count, errors, lat: deque[ms]} ; only /api* + /health tracked (static assets skipped).
_METRICS: dict[str, dict] = {}


def _record(path: str, status: int, ms: float, rid: str) -> None:
    if not (path.startswith("/api") or path == "/health"):
        return
    m = _METRICS.setdefault(path, {"count": 0, "errors": 0, "lat": _collections.deque(maxlen=500)})
    m["count"] += 1
    if status >= 500:
        m["errors"] += 1
    m["lat"].append(ms)
    _access_log.info("req_id=%s method=%s path=%s status=%s latency_ms=%.1f", rid,
                     "", path, status, ms)


@app.middleware("http")
async def _observability(request, call_next):
    rid = request.headers.get("X-Request-ID") or _uuid.uuid4().hex[:12]
    t0 = _time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        _record(request.url.path, 500, (_time.perf_counter() - t0) * 1000.0, rid)
        raise
    resp.headers["X-Request-ID"] = rid
    _record(request.url.path, resp.status_code, (_time.perf_counter() - t0) * 1000.0, rid)
    return resp


@app.get("/api/metrics", tags=["meta"])
def metrics() -> dict:
    """In-process API observability: per-route request count, error count, p50/p95 latency
    (rolling window of the last 500 requests). No external APM — single-process scale."""
    def _pctl(vals: list, q: float):
        if not vals:
            return None
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 1)
    routes = {}
    for path, m in sorted(_METRICS.items()):
        lat = list(m["lat"])
        routes[path] = {"count": m["count"], "errors": m["errors"],
                        "p50_ms": _pctl(lat, 0.5), "p95_ms": _pctl(lat, 0.95)}
    return {"routes": routes, "total_requests": sum(m["count"] for m in _METRICS.values())}


def _parse(json_str: str) -> dict:
    """Tool functions return JSON strings; surface an error result as a 502 not silent data."""
    data = json.loads(json_str)
    if isinstance(data, dict) and data.get("error"):
        raise HTTPException(status_code=502, detail=data["error"])
    return data


# ── tiny in-process TTL cache for compute-heavy read endpoints ───────────────
# Repeated hits (multiple tabs, fast manual refresh) return a value computed at most every `ttl`
# seconds instead of re-running the work each request. Errors are NOT cached (they raise through).
# Single-process / single-user scale — no Redis. The client polls on its own 60s cadence anyway.
import time as _time  # noqa: E402

_TTL_CACHE: dict[str, tuple[float, dict]] = {}


def _cached(key: str, ttl: float, produce):
    now = _time.time()
    hit = _TTL_CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    val = produce()                      # if produce() raises, nothing is cached
    _TTL_CACHE[key] = (now, val)
    return val


# Soft deadline: run a recompute on a worker thread; if it exceeds `timeout` seconds, return 503
# instead of blocking the request indefinitely. (The orphaned thread finishes on its own — a soft
# guard, not hard cancellation; adequate at single-process scale for endpoints that should be fast.)
import concurrent.futures as _futures  # noqa: E402

_EXEC = _futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="api-compute")


def _cached_compute(key: str, ttl: float, timeout: float, produce):
    def _guarded():
        fut = _EXEC.submit(produce)
        try:
            return fut.result(timeout=timeout)
        except _futures.TimeoutError:
            raise HTTPException(status_code=503, detail=f"{key} compute timed out (>{timeout:.0f}s)")
    return _cached(key, ttl, _guarded)


# ── System info / cache control ──────────────────────────────────────
# Operational endpoints for the /ops "System" footer: tell the user
# which code is actually running + give them a button to force a cache
# clear when the answer "it should have updated but didn't" comes up.

import os as _os  # noqa: E402

_PROCESS_STARTED_AT = _time.time()


def _git_short_sha() -> str:
    """Return short git SHA of HEAD, or 'unknown'. Best-effort, never raises."""
    try:
        import subprocess as _sp
        r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=2.0,
                    cwd=str(_PATH(__file__).resolve().parent.parent))
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty() -> bool:
    """True if working tree has uncommitted changes. Best-effort."""
    try:
        import subprocess as _sp
        r = _sp.run(["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=2.0,
                    cwd=str(_PATH(__file__).resolve().parent.parent))
        if r.returncode == 0:
            # Filter out the noisy data/ + __pycache__ churn so it reflects
            # SOURCE changes that haven't been committed.
            lines = [ln for ln in r.stdout.splitlines()
                     if ln.strip() and not any(ln[3:].startswith(p) for p in
                          ("data/", "__pycache__/", "engine/__pycache__/",
                           ".streamlit/", ".claude/", "macro_alpha_memory.db"))]
            return len(lines) > 0
    except Exception:
        pass
    return False


from pathlib import Path as _PATH  # noqa: E402


@app.get("/api/system/version", tags=["governance"])
def system_version() -> dict:
    """Backend runtime info — git SHA + uptime + cache stats. Drives the
    /ops "System" footer so when "I changed code but the page doesn't
    update" comes up, the user can verify backend vs frontend SHA in one glance."""
    import datetime as _dt
    uptime_s = int(_time.time() - _PROCESS_STARTED_AT)
    return {
        "git_sha":         _git_short_sha(),
        "git_dirty":       _git_dirty(),
        "uptime_s":        uptime_s,
        "uptime_human":    f"{uptime_s // 86400}d {(uptime_s % 86400) // 3600}h {(uptime_s % 3600) // 60}m",
        "process_started_iso": _dt.datetime.utcfromtimestamp(_PROCESS_STARTED_AT).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_cached_keys":   len(_TTL_CACHE),
        "cached_keys":     sorted(_TTL_CACHE.keys()),
    }


@app.post("/api/system/cache/invalidate", tags=["governance"])
def system_cache_invalidate(key: Optional[str] = None) -> dict:
    """Force-invalidate cached compute results. With no `key`, drops ALL
    entries; with `key`, drops just that one (or no-op if not present).
    Useful when "I know the underlying data changed but the cached version
    is still being served" — the only path that bypasses TTL otherwise is
    a process restart."""
    if key is None:
        n = len(_TTL_CACHE)
        _TTL_CACHE.clear()
        return {"invalidated": "ALL", "n_dropped": n}
    existed = key in _TTL_CACHE
    _TTL_CACHE.pop(key, None)
    return {"invalidated": key, "n_dropped": 1 if existed else 0}


# ── Data refresh job (the staleness banner's remediation ACTION) ──────────────────────────────
# A warning that gives no remediation just creates alert fatigue. The "data stale" banner triggers
# THIS: the exact production daily job (scripts/run_paper_trade_daily.py --skip-feed-refresh) run as
# a background subprocess — re-runs the 5-strategy orchestrator, RM/DQ pre-trade gates, persists
# positions/NAV/strategy-log, and rebuilds the UI artifact. Isolated (subprocess can't crash the
# server), guarded against concurrent runs, exit-code surfaced honestly (incl. HARD-HALT reasons).
import subprocess as _subprocess  # noqa: E402
import threading as _threading    # noqa: E402

_REFRESH_LOCK = _threading.Lock()
_REFRESH_STATE: dict = {"running": False, "trigger": None, "started_at": None, "finished_at": None,
                        "exit_code": None, "ok": None, "message": None, "log_tail": None}
_REFRESH_EXIT_MSG = {
    0: "Refresh complete — orchestrator ran, positions/NAV persisted, artifact rebuilt.",
    1: "Partial: ran but some DB writes failed. See log tail.",
    2: "Orchestrator failed — nothing written. See log tail.",
    3: "Ran with a non-blocking feed-refresh error.",
    4: "Halted: circuit breaker SEVERE — manual reset required before a refresh can run.",
    5: "Halted: Risk Manager pre-trade HARD HALT — book NOT persisted (a risk gate tripped).",
    6: "Halted: DQ Inspector HARD HALT — data quality critical.",
    -1: "Refresh timed out (>600s).",
    -2: "Refresh failed to launch.",
}


def _run_refresh_job() -> None:
    import datetime as _dt
    script = ROOT / "scripts" / "run_paper_trade_daily.py"
    try:
        proc = _subprocess.run(
            [sys.executable, str(script), "--skip-feed-refresh"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
        code = proc.returncode
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
    except _subprocess.TimeoutExpired:
        code, tail = -1, "timed out after 600s"
    except Exception as exc:  # noqa: BLE001
        code, tail = -2, f"launch failed: {exc}"
    with _REFRESH_LOCK:
        _REFRESH_STATE.update(running=False, finished_at=_dt.datetime.utcnow().isoformat(),
                              exit_code=code, ok=(code == 0),
                              message=_REFRESH_EXIT_MSG.get(code, f"exit {code}"), log_tail=tail)


def _start_refresh(trigger: str) -> dict:
    """Begin a refresh if one isn't already running. trigger ∈ {'manual','auto'}."""
    import datetime as _dt
    with _REFRESH_LOCK:
        if _REFRESH_STATE["running"]:
            return {**_REFRESH_STATE, "already_running": True}
        _REFRESH_STATE.update(running=True, trigger=trigger, started_at=_dt.datetime.utcnow().isoformat(),
                              finished_at=None, exit_code=None, ok=None, message=None, log_tail=None)
    _threading.Thread(target=_run_refresh_job, daemon=True).start()
    return {**_REFRESH_STATE, "already_running": False}


# Self-heal: a data REFRESH is operational (deterministic, NOT a decision) — it should run
# automatically, never wait on a human click. (Book ACTIONS still require approval — that is the
# real HITL line.) This in-process loop auto-refreshes when the LIVE book is genuinely stale; it's
# a fallback for when the external daily scheduler (Task Scheduler 06:00 SGT) didn't run. Guarded:
# fires only when stale + not already running + past a cooldown, so it never thrashes nor double-
# runs the cron (if the cron ran, the book is fresh → the loop stays idle).
_AUTO_REFRESH_COOLDOWN_S = 2 * 3600.0
_AUTO_REFRESH = {"last_attempt": 0.0}


def _auto_refresh_loop() -> None:
    import time as _t
    _t.sleep(10)                       # let startup settle (and keep test imports from triggering)
    while True:
        try:
            stale = _freshness_payload().get("overall") == "stale"
            cooled = (_t.time() - _AUTO_REFRESH["last_attempt"]) > _AUTO_REFRESH_COOLDOWN_S
            if stale and not _REFRESH_STATE["running"] and cooled:
                logger.info("auto-refresh: book stale → starting refresh (self-heal)")
                _AUTO_REFRESH["last_attempt"] = _t.time()
                _start_refresh("auto")
        except Exception:
            logger.exception("auto-refresh loop tick failed")
        _t.sleep(900)                  # re-check every 15 min


# Don't auto-run under pytest (importing api.main must never launch the daily job).
if "pytest" not in sys.modules:
    _threading.Thread(target=_auto_refresh_loop, daemon=True).start()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "macroalphapro-api", "version": app.version}


def _pick_artifact(as_of: "str | None"):
    """The UI artifact whose date is the latest ON OR BEFORE `as_of` (time-travel), else the most
    recent if as_of is None. Returns a Path or None. The shared as-of selector for book/risk views."""
    from pathlib import Path as _P
    ad = _P("data/ui_artifact")
    files = sorted(ad.glob("*.json"), reverse=True) if ad.is_dir() else []
    if not files:
        return None
    if not as_of:
        return files[0]
    return next((f for f in files if f.stem <= as_of), files[-1])


@app.get("/api/book/dates", tags=["book"])
def book_dates() -> dict:
    """Available as-of dates for time-travel — one per UI artifact, ascending; `latest` is live."""
    from pathlib import Path as _P
    ad = _P("data/ui_artifact")
    dates = sorted(f.stem for f in ad.glob("*.json")) if ad.is_dir() else []
    return {"dates": dates, "latest": dates[-1] if dates else None}


@app.get("/api/book/state", tags=["book"])
def book_state(as_of: str | None = None) -> dict:
    """Paper-trade book state (per-strategy status, combined gross/net, sleeve attribution) from the
    cached UI artifact — fast, no orchestrator re-run. `as_of` (YYYY-MM-DD) time-travels to the
    artifact on/before that date."""
    from engine.agents.persona.tools import read_today_book_state
    return _parse(read_today_book_state(as_of))


@app.get("/api/book/nav", tags=["book"])
def book_nav(days_back: int = 120) -> dict:
    """Recent NAV path + daily Modified-Dietz returns (PortfolioNavSnapshot) for the /book NAV
    sparkline + return strip. `days_back` clamped to [1, 365] by the tool. n_rows=0 (with a
    message) when the snapshot table is empty — not an error."""
    from engine.agents.persona.tools import read_nav_history
    return _parse(read_nav_history(days_back))


@app.get("/api/book/trades", tags=["book"])
def book_trades(limit: int = 100) -> dict:
    """Recent trade blotter from the artifact's trade_log (date / strategy / sleeve / ticker /
    side / weight / signal / event trigger), newest first, capped at `limit` (max 500)."""
    return _cached(f"book_trades:{limit}", 60.0, lambda: _book_trades_payload(limit))


def _book_trades_payload(limit: int) -> dict:
    from pathlib import Path as _P
    ad = _P("data/ui_artifact")
    files = sorted(ad.glob("*.json"), reverse=True) if ad.is_dir() else []
    if not files:
        return {"as_of": None, "n_total": 0, "trades": []}
    d = json.loads(files[0].read_text(encoding="utf-8"))
    raw = d.get("trade_log_recent", []) or []
    raw = sorted(raw, key=lambda r: (r.get("date") or "", r.get("strategy_name") or ""), reverse=True)
    cap = max(1, min(int(limit), 500))
    trades = [{"date": r.get("date"), "strategy": r.get("strategy_name"), "sleeve": r.get("sleeve_id"),
               "ticker": r.get("ticker"), "side": r.get("side"), "weight": r.get("weight"),
               "signal": r.get("signal_value"), "event": r.get("event_trigger")} for r in raw[:cap]]
    return {"as_of": (d.get("_meta") or {}).get("as_of_date"), "n_total": len(raw), "trades": trades}


@app.get("/api/book/positions", tags=["book"])
def book_positions(limit: int = 400, as_of: str | None = None) -> dict:
    """Combined per-ticker HOLDINGS — what the book holds: Σ(intra_weight × book_weight) per ticker
    from the UI artifact (the SAME reconstruction /api/risk uses). Each ticker's net weight + side
    (long/short) + contributing strategies/sleeves, sorted by |weight|; plus a summary
    (n_long/n_short/gross/net/biggest). `as_of` time-travels. Cached 60s."""
    return _cached(f"book_positions:{limit}:{as_of}", 60.0, lambda: _book_positions_payload(limit, as_of))


def _book_positions_payload(limit: int, as_of: str | None = None) -> dict:
    f = _pick_artifact(as_of)
    if f is None:
        return {"as_of": None, "n": 0, "n_long": 0, "n_short": 0, "gross": 0.0, "net": 0.0, "biggest": None, "positions": []}
    d = json.loads(f.read_text(encoding="utf-8"))
    meta = d.get("_meta") or {}
    states = d.get("strategy_states") or []
    raw = d.get("positions") or []
    bw = {s.get("strategy_name"): (s.get("book_weight") or 0.0) for s in states}
    sleeve_of = {s.get("strategy_name"): s.get("sleeve_id") for s in states}
    agg: dict[str, dict] = {}
    for p in raw:
        tkr, strat = p.get("ticker"), p.get("strategy_name")
        w = (p.get("intra_weight") or 0.0) * bw.get(strat, 0.0)
        if abs(w) < 1e-9 or not tkr:
            continue
        a = agg.setdefault(tkr, {"ticker": tkr, "weight": 0.0, "strategies": set(), "sleeves": set()})
        a["weight"] += w
        if strat:
            a["strategies"].add(strat)
        if sleeve_of.get(strat):
            a["sleeves"].add(sleeve_of[strat])
    # Lineage (provenance): the rule/signal that put each ticker in the book, per contributing
    # strategy — from the trade log. Makes "every position is rule-driven" VERIFIABLE (the position-
    # level version of verdict→evidence). Latest entry per (ticker, strategy).
    lin: dict[tuple, dict] = {}
    for tr in (d.get("trade_log_recent") or []):
        key = (tr.get("ticker"), tr.get("strategy_name"))
        if key[0] and (key not in lin or (tr.get("date") or "") >= (lin[key].get("date") or "")):
            lin[key] = tr

    def _legs(ticker: str, strats: list[str]) -> list[dict]:
        out = []
        for st in strats:
            tr = lin.get((ticker, st))
            if tr is None:
                continue
            out.append({
                "strategy": st, "signal": tr.get("signal_value"), "event": tr.get("event_trigger"),
                "is_rebalance": bool(tr.get("is_rebalance_day")), "horizon_days": tr.get("expected_horizon_days"),
                "spec_id": tr.get("spec_id"), "spec_hash": tr.get("spec_hash_short"), "date": tr.get("date"),
            })
        return out

    rows = [{"ticker": a["ticker"], "weight": round(a["weight"], 6),
             "side": "long" if a["weight"] > 0 else "short",
             "strategies": sorted(a["strategies"]), "sleeves": sorted(a["sleeves"]),
             "legs": _legs(a["ticker"], sorted(a["strategies"]))}
            for a in agg.values() if abs(a["weight"]) >= 1e-9]
    rows.sort(key=lambda r: -abs(r["weight"]))
    return {
        "as_of": meta.get("as_of_date") or f.stem,
        "n": len(rows),
        "n_long": sum(1 for r in rows if r["side"] == "long"),
        "n_short": sum(1 for r in rows if r["side"] == "short"),
        "gross": round(sum(abs(r["weight"]) for r in rows), 4),
        "net": round(sum(r["weight"] for r in rows), 4),
        "biggest": rows[0] if rows else None,
        "positions": rows[: max(1, min(int(limit), 1000))],
    }


@app.get("/api/book/overlay", tags=["book"])
def book_overlay() -> dict:
    """Operator discretionary OVERLAY sleeve — human-originated positions filed via the CoS
    propose→approve→execute loop, held SEPARATE from the systematic book and measured on their
    own (engine.overlay_executor). Read-only; the executor only runs behind a human approval."""
    try:
        from engine.overlay_executor import read_overlay, read_overlay_trades
        book = read_overlay()
        book["recent_trades"] = read_overlay_trades(limit=30)
        return book
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"overlay read failed: {exc}")


@app.get("/api/book/combined", tags=["book"])
def book_combined() -> dict:
    """Deployed book stats.

    2026-06-02 amendment: now returns BOTH the actually-deployed 5-sleeve
    regime-conditional book (config C, the live allocation since 2026-05-30)
    AND the 2-mechanism narrative comparison (equity vs equity+carry) the
    earlier UI was built around. The 5-sleeve `deployed` block is the
    truthful "what we're actually running" datum; the 2-mechanism block is
    preserved as historical narrative (the carry-uplift story).

    Heavy (rebuilds from cached futures + equity + TSMOM + hedges); cached 1h."""
    def _produce() -> dict:
        from engine.portfolio.combined_book import (DEFAULT_BOOK_VOL_TARGET,
                                                     DEFAULT_CARRY_RISK_WEIGHT,
                                                     DEFAULT_TSMOM_RISK_WEIGHT,
                                                     DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
                                                     DEFAULT_MOM_HEDGE_RISK_WEIGHT,
                                                     book_stats, build_combined_book)
        w, bvt = DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_BOOK_VOL_TARGET
        # 2-mechanism narrative (kept for the carry-uplift story).
        combined_2m = build_combined_book(carry_risk_weight=w, book_vol_target=bvt)
        equity_only = build_combined_book(carry_risk_weight=0.0, book_vol_target=bvt)
        # ACTUALLY DEPLOYED — 5-sleeve regime-conditional config C (2026-05-30).
        # Insurance weights vary by VIX regime: CALM 0/0, NORMAL 5/2, STRESS 10/5.
        deployed_5 = build_combined_book(book_vol_target=bvt, regime_conditional=True)
        # PRE-INSURANCE reference — same 3 alpha sleeves (equity + carry + tsmom)
        # WITHOUT crisis_hedge or mom_hedge. Shows the cost of buying insurance:
        # roughly -0.07 Sharpe in exchange for +1pp maxDD improvement per the
        # 2026-05-30 deploy decision (config A baseline vs config C deployed).
        pre_insurance_3 = build_combined_book(
            book_vol_target=bvt,
            tsmom_risk_weight=DEFAULT_TSMOM_RISK_WEIGHT,
            crisis_risk_weight=0.0,
            mom_hedge_risk_weight=0.0,
        )
        cum = (1 + deployed_5.dropna()).cumprod()
        return {
            "available": True,
            "deployed": {
                "config_name": "config C — regime-conditional 5-sleeve",
                "deploy_date": "2026-05-30",
                "book_vol_target": bvt,
                "stats": book_stats(deployed_5),
                "sleeves": [
                    {"name": "equity_book",      "role": "alpha",         "base_weight": round(1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT - DEFAULT_CRISIS_HEDGE_RISK_WEIGHT - DEFAULT_MOM_HEDGE_RISK_WEIGHT, 3)},
                    {"name": "cross_asset_carry","role": "alpha",         "base_weight": DEFAULT_CARRY_RISK_WEIGHT},
                    {"name": "cross_asset_tsmom","role": "alpha",         "base_weight": DEFAULT_TSMOM_RISK_WEIGHT},
                    {"name": "crisis_hedge_tlt_gld", "role": "diversifier", "base_weight": DEFAULT_CRISIS_HEDGE_RISK_WEIGHT, "regime_modulated": True},
                    {"name": "mom_hedge_overlay","role": "insurance",     "base_weight": DEFAULT_MOM_HEDGE_RISK_WEIGHT,    "regime_modulated": True},
                ],
                "regime_grids": {
                    "CALM":   {"crisis": 0.00, "mom_hedge": 0.00},
                    "NORMAL": {"crisis": 0.05, "mom_hedge": 0.02},
                    "STRESS": {"crisis": 0.10, "mom_hedge": 0.05},
                },
                "note": ("Live since 2026-05-30 per config-C deploy decision. Each leg "
                         "vol-targeted ~10%, then blended at the regime-conditional "
                         "weights, then scaled to 10% book vol. Insurance sleeves "
                         "(crisis hedge + MTUM short) pay premium only in NORMAL/STRESS "
                         "regimes (VIX 1y z-score classifier, ±1σ thresholds)."),
            },
            # 2026-06-02 — pre-insurance comparison row for the Tearsheet.
            # Same 3 alpha sleeves at the same vol target, NO crisis/mom hedges.
            "pre_insurance_3_mech": {
                "config_name": "alpha-only 3-mech (no hedges)",
                "stats": book_stats(pre_insurance_3),
                "note": ("Reference comparison: deployed 5-sleeve config C minus "
                         "the two regime-modulated insurance sleeves. The Sharpe "
                         "difference is the INSURANCE PREMIUM — by design we pay "
                         "a small Sharpe haircut to buy maxDD improvement and "
                         "crisis-regime protection."),
            },
            "narrative_2_mechanism": {
                "title": "Mechanism narrative · 2-sleeve carry uplift",
                "carry_risk_weight": w,
                "combined": book_stats(combined_2m),
                "equity_only": book_stats(equity_only),
                "note": ("Historical narrative kept for the carry-uplift story. This is "
                         "NOT the live deployed config — see `deployed` above for that. "
                         "Both rows shown at the same 10% vol — carry's benefit is the "
                         "Sharpe uplift."),
            },
            # back-compat: legacy fields for the older 2-mechanism card.
            "carry_risk_weight": w,
            "book_vol_target": bvt,
            "combined": book_stats(combined_2m),
            "equity_only": book_stats(equity_only),
            "dates": [str(d)[:10] for d in cum.index],
            "equity_curve": [round(float(v), 6) for v in cum.values],
            "note": ("Deployed config: equity book + cross-asset carry at 30% carry "
                     "risk-weight, book sized to 10% vol-target. Both rows shown at the "
                     "same 10% vol — carry's benefit is the Sharpe uplift. Carry marked "
                     "daily forward (engine.portfolio.carry_sleeve.build_carry_daily_returns)."),
        }
    try:
        return _cached_compute("book_combined", ttl=3600, timeout=300, produce=_produce)
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


@app.get("/api/deploy/manifest", tags=["governance"])
def deploy_manifest() -> dict:
    """The deployment manifest — single source of truth for "what's live".

    Reads data/portfolio/active_deployment.yaml via the registry loader.
    The /book Tearsheet binds here for the truthful config snapshot
    (sleeves, weights, regime grids, signing specs, deploy date,
    expected stats), even if the legacy /api/book/combined response
    shape hasn't been refreshed.

    Also returns `code_drift_issues`: any sleeve where the Python
    constant in engine.portfolio.combined_book disagrees with the YAML
    manifest. Empty list = healthy. Non-empty = someone edited a
    constant without updating the manifest — investigate via
    scripts/deploy_config.py.
    """
    try:
        from engine.portfolio.deployed_registry import load_active, assert_constants_match
        from engine.portfolio.combined_book import (
            DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_TSMOM_RISK_WEIGHT,
            DEFAULT_CRISIS_HEDGE_RISK_WEIGHT, DEFAULT_MOM_HEDGE_RISK_WEIGHT,
            DEFAULT_BOOK_VOL_TARGET,
        )
        cfg = load_active()
        drift = assert_constants_match(
            carry_risk_weight    = DEFAULT_CARRY_RISK_WEIGHT,
            tsmom_risk_weight    = DEFAULT_TSMOM_RISK_WEIGHT,
            crisis_risk_weight   = DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
            mom_hedge_risk_weight= DEFAULT_MOM_HEDGE_RISK_WEIGHT,
            book_vol_target      = DEFAULT_BOOK_VOL_TARGET,
        )
        return {
            "available":      True,
            "config_id":      cfg.id,
            "label":          cfg.label,
            "summary":        cfg.summary,
            "deploy_date":    cfg.deploy_date,
            "days_since_deploy": cfg.days_since_deploy,
            "book_vol_target": cfg.book_vol_target,
            "signing_spec_ids": list(cfg.signing_spec_ids),
            "expected_stats": cfg.expected_stats,
            "sleeves": [
                {
                    "name":             s.name,
                    "role":             s.role,
                    "base_weight":      s.base_weight,
                    "regime_modulated": s.regime_modulated,
                    "builder":          s.builder,
                    "target_vol":       s.target_vol,
                    "signing_spec_ids": list(s.signing_spec_ids),
                }
                for s in cfg.sleeves
            ],
            "regime_grids":     cfg.regime_grids,
            "regime_classifier": cfg.regime_classifier,
            "code_drift_issues": drift,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:300]}


@app.get("/api/book/tracking", tags=["book"])
def book_tracking() -> dict:
    """Accumulating "does live deliver backtest" tracker (audit item F): live realized returns vs
    the backtest expectation. With only days of live NAV this is NOT yet significant — it accrues
    into a verdict over ~6mo. Honest: backtest ref = the live 5-sleeve replay, not the 1.04
    market-neutral construct. Cached briefly (cheap)."""
    def _produce() -> dict:
        from engine.validation.live_vs_backtest import build_tracking
        return build_tracking()
    try:
        return _cached("book_tracking", 120.0, _produce)
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


@app.get("/api/execution", tags=["book"])
def execution_reconcile() -> dict:
    """Consolidated paper-book execution view. EQUITY legs reconcile against the PAPER broker
    (Alpaca if ALPACA_KEY configured, else offline sim) — actual positions vs model target weights,
    exposing drift / tracking error / breaks (queued-not-filled, unborrowable shorts). The CARRY
    futures legs run on the durable internal FuturesSimAdapter (data/execution/futures_sim_state.json)
    — its NAV/contracts are surfaced too. (Trend sleeve deploys on Alpaca ETFs per spec 75, not here.)
    0-LLM, read-only. Cached 60s."""
    def _produce() -> dict:
        from collections import Counter
        from engine.execution.reconcile import reconcile
        from engine.execution.run_paper_execution import build_target_weights_from_artifact
        weights, asof = build_target_weights_from_artifact()
        order_status: dict = {}
        try:
            from engine.execution.alpaca_adapter import AlpacaAdapter
            adapter = AlpacaAdapter()                       # raises if not paper-configured
            try:
                order_status = dict(Counter(o.get("status") for o in adapter.get_orders()))
            except Exception:
                order_status = {}
        except Exception:
            from engine.execution.sim_adapter import SimAdapter
            adapter = SimAdapter(state_path="data/execution/sim_state.json")
        rec = reconcile(adapter, weights)                   # 2-3 broker calls (account + positions)
        rec["as_of"] = asof
        rec["order_status"] = order_status
        rec["undeployed_weight"] = round(max(0.0, 1.0 - rec["gross_actual"]), 4)  # cash gap (incl. untradeable)
        # carry futures sleeve — durable internal sim (read state file; no recompute)
        try:
            import json as _json
            from pathlib import Path as _P
            fp = _P("data/execution/futures_sim_state.json")
            if fp.exists():
                fs = _json.loads(fp.read_text(encoding="utf-8"))
                navh = fs.get("nav_history", [])
                rec["futures_sleeve"] = {
                    "venue": "futures_sim (carry, $10M institutional scale)",
                    "equity": round(float(fs.get("equity", 0.0)), 2),
                    "n_contracts": len([c for c in fs.get("contracts", {}).values() if abs(c) > 1e-9]),
                    "nav_points": len(navh),
                    "last_nav": navh[-1] if navh else None,
                }
        except Exception:
            pass
        rec["available"] = True
        return rec
    try:
        return _cached("execution_reconcile", 60.0, _produce)
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


@app.get("/api/book/risk-contrib", tags=["risk"])
def book_risk_contrib() -> dict:
    """Position-level RISK decomposition (MCTR / component-risk): each holding's share of BOOK
    volatility — not its weight (a 13% position can be 5% or 40% of risk). Σ from the cached
    holdings return panel; signed weights → negative %-risk = the name diversifies the book. LIVE
    book only (time-travel would need a historical panel). available:False until the panel is built
    (engine.validation.risk_contribution.build_returns_panel). Cached 120s."""
    return _cached_compute("risk_contrib", 120.0, 15.0, _risk_contrib_payload)


def _risk_contrib_payload() -> dict:
    import datetime as _dt
    import pandas as pd
    from engine.validation.risk_contribution import PANEL_PATH, compute_risk_contributions
    if not PANEL_PATH.is_file():
        return {"available": False, "reason": "returns panel not built yet — run build_returns_panel"}
    pos = _book_positions_payload(1000)
    weights = {p["ticker"]: p["weight"] for p in pos.get("positions", [])}
    if not weights:
        return {"available": False, "reason": "no holdings in the book"}
    out = compute_risk_contributions(weights, pd.read_parquet(PANEL_PATH))
    out["as_of"] = pos.get("as_of")
    out["panel_built"] = _dt.datetime.fromtimestamp(PANEL_PATH.stat().st_mtime).isoformat(timespec="minutes")
    return out


@app.get("/api/book/factor-exposure", tags=["risk"])
def book_factor_exposure() -> dict:
    """Cross-asset factor exposure: the book regressed on 5 macro ETF-proxy factors (equity SPY /
    rates TLT / credit HYG−LQD / commodity DBC / dollar UUP) → β per factor + each factor's % of
    book variance + R²/idiosyncratic. The cross-asset footprint FF5 can't show. LIVE book.
    available:False until the panel is built. Cached 120s."""
    return _cached_compute("factor_exposure", 120.0, 15.0, _factor_exposure_payload)


def _factor_exposure_payload() -> dict:
    import pandas as pd
    from engine.validation.factor_exposure import compute_factor_exposure
    from engine.validation.risk_contribution import PANEL_PATH
    if not PANEL_PATH.is_file():
        return {"available": False, "reason": "returns panel not built yet"}
    pos = _book_positions_payload(1000)
    weights = {p["ticker"]: p["weight"] for p in pos.get("positions", [])}
    if not weights:
        return {"available": False, "reason": "no holdings in the book"}
    out = compute_factor_exposure(weights, pd.read_parquet(PANEL_PATH))
    out["as_of"] = pos.get("as_of")
    return out


@app.get("/api/book/scenarios", tags=["risk"])
def book_scenarios() -> dict:
    """Book stress on the cached holdings panel: HISTORICAL replay (worst/best cumulative 1d/5d/20d
    of today's book + which positions drove the worst day) and a 1-factor equity-beta SHOCK
    (−20%/−10%/+10% market → book P&L). LIVE book. available:False until the panel is built. Cached 120s."""
    return _cached_compute("scenarios", 120.0, 15.0, _scenarios_payload)


def _scenarios_payload() -> dict:
    import pandas as pd
    from engine.validation.risk_contribution import PANEL_PATH
    from engine.validation.scenario_stress import compute_scenarios
    if not PANEL_PATH.is_file():
        return {"available": False, "reason": "returns panel not built yet"}
    pos = _book_positions_payload(1000)
    weights = {p["ticker"]: p["weight"] for p in pos.get("positions", [])}
    if not weights:
        return {"available": False, "reason": "no holdings in the book"}
    out = compute_scenarios(weights, pd.read_parquet(PANEL_PATH))
    out["as_of"] = pos.get("as_of")
    return out


@app.get("/api/brief", tags=["book"])
def daily_brief() -> dict:
    """Latest daily brief snapshot: regime (+ change), risk-on probability, long/short counts,
    today's activity (entries / invalidations / rebalances), signal flips + risk alerts. The
    morning read. Read-only DB."""
    return _cached("daily_brief", 60.0, _daily_brief_payload)


def _book_long_short() -> dict | None:
    """Live long/short counts from the SAME combined-book artifact /api/risk reconstructs.

    The daily-batch writer never sets daily_brief_snapshots.n_long/n_short (dead columns →
    always 0), and those counts came from the legacy sector-rotation pipeline, not the live
    5-strategy book. Compute them here from the latest UI artifact so the morning brief reflects
    the REAL book (consistent with /api/risk + /api/book/state). Returns None if no artifact."""
    from pathlib import Path as _P
    try:
        import pandas as pd
        files = sorted(_P("data/ui_artifact").glob("*.json"), reverse=True)
        if not files:
            return None
        data = json.loads(files[0].read_text(encoding="utf-8"))
        meta = data.get("_meta") or {}
        states = data.get("strategy_states") or []
        positions = data.get("positions") or []
        bw = {s.get("strategy_name"): (s.get("book_weight") or 0.0) for s in states}
        combined: dict[str, float] = {}
        for p in positions:
            t = p.get("ticker")
            combined[t] = combined.get(t, 0.0) + (p.get("intra_weight") or 0.0) * bw.get(p.get("strategy_name"), 0.0)
        ser = pd.Series(combined, dtype="float64")
        if not len(ser):
            return None
        thr = 1e-6
        return {
            "n_long":     int((ser > thr).sum()),
            "n_short":    int((ser < -thr).sum()),
            "book_as_of": meta.get("as_of_date") or meta.get("as_of") or files[0].stem,
        }
    except Exception:
        return None


def _daily_brief_payload() -> dict:
    """Daily brief — primary source is the LIVE ui_artifact (refresh updates
    it daily). The legacy daily_brief_snapshots table writer died 2026-05-21
    (sector-rotation pipeline retired); reading from it left the UI showing
    a 24-day stale "as of" date on 2026-06-14. Fix: derive as_of + regime
    from the live artifact's _meta + regime dicts, and expose
    `regime_as_of` separately so the UI can flag when REGIME (a separate
    cron) is stale even when the BOOK is fresh. Snapshot table is only
    used now for the optional icir_month / signal_flips / risk_alerts
    payload, with empty defaults when the table is empty / stale."""
    import datetime as _dt
    from pathlib import Path as _P

    # Primary: live ui_artifact (the refresh action keeps this fresh).
    live = _book_long_short()
    artifact_as_of: str | None = None
    regime_payload: dict | None = None
    try:
        files = sorted(_P("data/ui_artifact").glob("*.json"), reverse=True)
        if files:
            data = json.loads(files[0].read_text(encoding="utf-8"))
            artifact_as_of = (data.get("_meta") or {}).get("as_of_date") or files[0].stem
            regime_payload = data.get("regime") or None
    except Exception:
        pass

    # Optional: try the snapshot table for the extra "morning narrative"
    # fields (signal flips / risk alerts / IC). Empty if writer is dead.
    extras: dict = {
        "regime_prev": None, "regime_changed": False,
        "n_entries": 0, "n_invalidations": 0, "n_rebalance": 0,
        "icir_month": None, "signal_flips": [], "risk_alerts": [],
        "table_as_of": None,
    }
    try:
        import sqlalchemy as sa
        from engine.memory import SessionFactory
        with SessionFactory() as s:
            cols = [c[1] for c in s.execute(sa.text(
                "PRAGMA table_info(daily_brief_snapshots)")).fetchall()]
            row = s.execute(sa.text(
                "SELECT * FROM daily_brief_snapshots ORDER BY as_of_date DESC, id DESC LIMIT 1")).fetchone()
        if row:
            d = dict(zip(cols, row))
            def _jl(k):
                try: return json.loads(d.get(k) or "[]")
                except Exception: return []
            extras.update({
                "regime_prev":     d.get("regime_prev"),
                "regime_changed":  bool(d.get("regime_changed")),
                "n_entries":       d.get("n_entries") or 0,
                "n_invalidations": d.get("n_invalidations") or 0,
                "n_rebalance":     d.get("n_rebalance") or 0,
                "icir_month":      d.get("icir_month"),
                "signal_flips":    _jl("signal_flips_json"),
                "risk_alerts":     _jl("risk_alerts_json"),
                "table_as_of":     d.get("as_of_date"),
            })
    except Exception:
        # Table read failures are not fatal — the live artifact carries
        # the load-bearing as_of and regime.
        pass

    out: dict = {
        "as_of":            artifact_as_of,
        "regime":           (regime_payload or {}).get("regime"),
        "regime_as_of":     (regime_payload or {}).get("as_of_date"),
        "regime_days_stale":(regime_payload or {}).get("days_stale"),
        "p_risk_on":        (regime_payload or {}).get("p_risk_on"),
        "regime_prev":      extras["regime_prev"],
        "regime_changed":   extras["regime_changed"],
        "n_long":           live["n_long"]  if live else 0,
        "n_short":          live["n_short"] if live else 0,
        "book_as_of":       live["book_as_of"] if live else artifact_as_of,
        "long_short_source": "live_book" if live else "none",
        "n_entries":        extras["n_entries"],
        "n_invalidations":  extras["n_invalidations"],
        "n_rebalance":      extras["n_rebalance"],
        "icir_month":       extras["icir_month"],
        "signal_flips":     extras["signal_flips"],
        "risk_alerts":      extras["risk_alerts"],
        "extras_table_as_of": extras["table_as_of"],
    }
    return out


# Per-source staleness thresholds (days). CRITICAL: only check sources that (a) reflect the LIVE
# 5-strategy paper-trade book AND (b) the "Refresh data" action actually updates — otherwise the
# banner is a dead-end (warns about something the user can't fix). Verified 2026-05-24 by running a
# real refresh: it updates the UI artifact + paper_trade_strategy_log. simulated_positions /
# daily_brief_snapshots are LEGACY sector-rotation tables (abandoned by the paper-trade pipeline);
# portfolio_nav_snapshots is a separate writer — none are touched by the refresh, so they are NOT
# the book's freshness and were a false alarm. We track only what the refresh fixes.
_FRESH_THRESHOLDS = {"book": 2, "strategy_log": 2}


@app.get("/api/freshness", tags=["meta"])
def freshness() -> dict:
    """Single as-of authority across the independently-cadenced pipelines (book artifact / daily
    brief / NAV / positions). Each source -> {as_of, age_days, threshold_days, stale}; overall is
    stale if ANY source exceeds its tolerance. The nav consumes this to distinguish 'server live'
    (/health) from 'data fresh' — a green pill must never imply fresh data when the book is weeks
    old. Read-only, cached 30s."""
    return _cached("freshness", 30.0, _freshness_payload)


def _freshness_payload() -> dict:
    import datetime as _dt
    from pathlib import Path as _P
    today = _dt.date.today()

    def _age(d) -> int | None:
        if d is None:
            return None
        if isinstance(d, str):
            try:
                d = _dt.date.fromisoformat(d[:10])
            except Exception:
                return None
        if isinstance(d, _dt.datetime):
            d = d.date()
        return (today - d).days if isinstance(d, _dt.date) else None

    # book artifact (latest data/ui_artifact/<date>.json) — what the UI actually renders.
    book_as_of = None
    try:
        files = sorted(_P("data/ui_artifact").glob("*.json"), reverse=True)
        if files:
            meta = (json.loads(files[0].read_text(encoding="utf-8")).get("_meta") or {})
            book_as_of = meta.get("as_of_date") or meta.get("as_of") or files[0].stem
    except Exception:
        pass

    # paper_trade_strategy_log — the source-of-truth DB behind the book (the refresh persists it).
    strat_as_of = None
    try:
        from sqlalchemy import func
        from engine.db_models import PaperTradeStrategyLog
        from engine.memory import SessionFactory
        with SessionFactory() as s:
            strat_as_of = s.query(func.max(PaperTradeStrategyLog.date)).scalar()
    except Exception:
        pass

    raw = {"book": book_as_of, "strategy_log": strat_as_of}
    sources: list[dict] = []
    any_stale = False
    worst: int | None = None
    for name, asof in raw.items():
        age = _age(asof)
        thr = _FRESH_THRESHOLDS.get(name, 5)
        stale = age is not None and age > thr
        any_stale = any_stale or stale
        if age is not None:
            worst = age if worst is None else max(worst, age)
        sources.append({
            "source": name, "as_of": str(asof) if asof is not None else None,
            "age_days": age, "threshold_days": thr, "stale": bool(stale),
        })
    return {
        "as_of": today.isoformat(), "sources": sources,
        "overall": "stale" if any_stale else "fresh", "worst_age_days": worst,
    }


@app.post("/api/ops/refresh", tags=["ops"])
def ops_refresh_start() -> dict:
    """MANUAL override of the data refresh (the book self-heals automatically when stale — see
    _auto_refresh_loop — so this is a 'force now', mainly used to RETRY after an auto-refresh
    failed). Kicks off the production daily paper-trade job in the background. Deterministic +
    user-triggered + local-only; the only heavy mutating endpoint besides /api/approvals/resolve.
    Concurrent runs are refused (returns the in-flight status)."""
    return _start_refresh("manual")


@app.get("/api/ops/refresh", tags=["ops"])
def ops_refresh_status() -> dict:
    """Status of the data-refresh job: running / exit_code / ok / message / log_tail. Cheap
    in-memory read (poll it while a refresh is in flight)."""
    with _REFRESH_LOCK:
        return dict(_REFRESH_STATE)


@app.get("/api/book/perf", tags=["book"])
def book_perf() -> dict:
    """Backtest performance series for /book — equity curve, underwater drawdown,
    52-week rolling Sharpe, summary stats. Clearly a BACKTEST (the live paper-trade
    NAV is shown separately).

    Window: 2014-09 to 2023-12 (weekly). The START date is the PRINCIPLED
    "all-components-honest" boundary — every sleeve in the deployed 5-sleeve
    config C could honestly have functioned from here. Binding constraint =
    OptionMetrics SPX skew surface starts 2014-01-02 (required by the Path C
    put_spread tail hedge sleeve, which replaced the broken mom_hedge_overlay
    on 2026-05-31). See data/portfolio_replay/replay_window_meta.json for the
    full audit trail of why this — and not an earlier date — is the right
    floor for the deployed config.

    Downsampled to ~150 points for chart rendering."""
    return _cached_compute("book_perf", 300.0, 15.0, _book_perf_payload)


def _book_perf_payload() -> dict:
    import pandas as pd
    from pathlib import Path as _P
    rp = _P("data/portfolio_replay")
    ser = None
    f1 = rp / "v1_combined_returns_weekly.parquet"
    if f1.is_file():
        df = pd.read_parquet(f1)
        ser = df.iloc[:, 0] if getattr(df, "shape", (0, 0))[1] >= 1 else df.squeeze()
    if ser is None or len(ser) == 0:
        f2 = rp / "v2_per_strategy_returns_5sleeve_weekly.parquet"
        if f2.is_file():
            # Backtest replay reconstructs HISTORICAL book performance using the
            # weights deployed during the backtest WINDOW (pre-2026-05-28). DO NOT
            # update these to current Spec 80 weights — that would change the
            # historical curve to reflect a strategy that wasn't actually
            # backtested. Current LIVE allocation lives in registry/adapters.py;
            # for current/forward NAV see /api/book/combined endpoint.
            w = {"K1_BAB": 0.486, "D_PEAD": 0.3645, "PATH_N": 0.3645,
                 "CTA_PQTIX": 0.135, "AC_proxy_AB_2014_23": 0.15}
            df = pd.read_parquet(f2)
            ser = (df * pd.Series(w)).sum(axis=1)
    if ser is None or len(ser.dropna()) == 0:
        raise HTTPException(status_code=404, detail="no backtest replay returns")

    ser = ser.dropna()
    n = len(ser)
    eq = (1.0 + ser).cumprod()
    dd = eq / eq.cummax() - 1.0
    roll = ser.rolling(52)
    rsharpe = (roll.mean() / roll.std()) * (52 ** 0.5)

    step = max(1, n // 150)
    idx = ser.index[::step]
    dates = [d.date().isoformat() for d in idx]
    equity = [round(float(eq.loc[d]), 4) for d in idx]
    drawdown = [round(float(dd.loc[d] * 100), 2) for d in idx]
    import math
    rs = [None if math.isnan(float(rsharpe.loc[d])) else round(float(rsharpe.loc[d]), 2) for d in idx]

    ann_ret = float(eq.iloc[-1] ** (52.0 / n) - 1.0)
    ann_vol = float(ser.std() * (52 ** 0.5))
    sharpe = round(ann_ret / ann_vol, 3) if ann_vol else None
    # 2026-06-02 — surface the window-provenance meta so the UI can
    # show why the backtest stops where it does (and offer an "extend"
    # path). See data/portfolio_replay/replay_window_meta.json.
    window_meta = None
    try:
        import json as _json
        meta_p = _P("data/portfolio_replay/replay_window_meta.json")
        if meta_p.is_file():
            window_meta = _json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "n_weeks": n,
        "start": ser.index[0].date().isoformat(), "end": ser.index[-1].date().isoformat(),
        "stats": {"ann_ret": round(ann_ret, 4), "ann_vol": round(ann_vol, 4),
                  "sharpe": sharpe, "max_dd": round(float(dd.min()), 4)},
        "dates": dates, "equity": equity, "drawdown": drawdown, "rolling_sharpe": rs,
        "window_meta": window_meta,
    }


@app.get("/api/dq", tags=["risk"])
def dq_report() -> dict:
    """DQ Inspector live data-quality verdict — runs the pre-batch FRESHNESS gates (modes 1-4:
    FRED macro / BAB cache / PEAD panel / S&P 500 feed) read-only, with a junior-analyst run-level
    rationale. Cached 120s (hits FRED + file mtime). Data quality is a first-class control
    (SR 11-7 / BCBS 239) — surfaced so 'is my data fit to trade on?' has an answer."""
    def _produce() -> dict:
        import datetime as _dt
        from engine.agents.dq_inspector.gates import evaluate_pre_batch
        from engine.agents.dq_inspector.narrator import narrate_dq_summary
        today = _dt.date.today()
        breaches = evaluate_pre_batch(today)
        hard = [b for b in breaches if getattr(b, "severity", "") == "HARD_HALT"]
        warn = [b for b in breaches if getattr(b, "severity", "") == "SOFT_WARN"]
        verdict = "HALT" if hard else ("WARN" if warn else "CLEAN")
        checks = [{"mode_id": b.mode_id, "severity": b.severity, "rule": b.rule_description,
                   "observed": b.observed_value, "threshold": b.threshold,
                   "affected": list(b.affected) if b.affected else []} for b in breaches]
        return {
            "as_of": today.isoformat(), "verdict": verdict, "n_breaches": len(breaches),
            "checks": checks, "rationale": narrate_dq_summary(breaches),
            "scope": "pre-batch freshness (modes 1-4: FRED / BAB cache / PEAD panel / S&P500 feed)",
            "decided_by": "engine.agents.dq_inspector.gates.evaluate_pre_batch",
            "narrated_by": "DQ Inspector",
        }
    try:
        return _cached("dq_report", 120.0, _produce)
    except Exception as exc:
        return {"available": False, "verdict": "UNKNOWN", "reason": str(exc)[:200]}


@app.get("/api/decay/report", tags=["risk"])
def decay_report() -> dict:
    """Decay Sentinel deterministic book-health report (latest daily artifact): per-mechanism
    role-aware health, pairwise downside/stress correlation, verdict, recommended allocation.
    Enriched with verdict_basis + decision provenance so the UI can show WHY (the math decides,
    the agent narrates — 0-LLM-in-DECISION made visible)."""
    from engine.agents.persona.tools import read_decay_sentinel_report
    return _enrich_decay_verdict(_parse(read_decay_sentinel_report()))


def _enrich_decay_verdict(rep: dict) -> dict:
    """SURFACE (never recompute) the deterministic basis of `overall` + decision provenance.
    The engine sets overall = ACTION if any ALERT alarm, else WATCH if any WARN, else HEALTHY
    (INFO never escalates) — so the basis is exactly the alarms at the deciding level. We FILTER
    the alarms already in the payload; we do not re-derive the verdict.

    2026-06-02 — also expose as_of_age_days so the UI can render a StalenessBadge
    (> 3d amber, > 14d red) without each consumer re-parsing dates."""
    try:
        overall = rep.get("overall")
        deciding = {"ACTION": "ALERT", "WATCH": "WARN"}.get(overall or "")
        alarms = rep.get("alarms") or []
        driving = [a.get("message") for a in alarms if deciding and a.get("level") == deciding]
        rep["verdict_basis"] = {
            "rule": "overall = ACTION if any ALERT alarm, else WATCH if any WARN, else HEALTHY "
                    "(INFO never escalates the book verdict)",
            "deciding_level": deciding,
            "driving_alarms": driving,
            "n_driving": len(driving),
        }
        rep["decided_by"] = "engine.validation.decay_sentinel.sentinel_report"
        rep["narrated_by"] = "Decay Sentinel"
    except Exception:
        pass
    # Compute staleness — non-fatal if as_of is missing or malformed.
    try:
        import datetime as _dt
        as_of_raw = rep.get("as_of")
        if isinstance(as_of_raw, str) and as_of_raw:
            as_of_dt = _dt.date.fromisoformat(as_of_raw[:10])
            age = (_dt.date.today() - as_of_dt).days
            rep["as_of_age_days"] = int(age)
    except Exception:
        pass
    return rep


@app.get("/api/approvals", tags=["governance"])
def approvals(include_resolved: bool = False, limit: int = 50) -> dict:
    """Governance & exception inbox (human-ON-the-loop). Read-only; the human decides and
    deterministic engine code executes — the LLM never auto-approves. The live systematic book
    does NOT route here (it auto-executes; risk control = automatic RM HARD-HALT). Each item
    carries an `effect` (what Approve actually does) and the response carries the routing
    `charter` — see engine.approval_charter. Not cached (must be fresh)."""
    import sqlalchemy as sa
    from engine.approval_charter import CHARTER, approval_effect
    from engine.memory import SessionFactory
    want = ["id", "created_at", "approval_type", "approval_class", "priority", "ticker", "sector",
            "triggered_condition", "triggered_date", "suggested_weight", "position_rank",
            "llm_confidence", "contradicts_quant", "approval_deadline", "review_rationale",
            "review_category", "status", "resolved_at", "resolved_by", "rejection_reason"]
    try:
        with SessionFactory() as s:
            avail = {c[1] for c in s.execute(sa.text("PRAGMA table_info(pending_approvals)")).fetchall()}
            sel = [c for c in want if c in avail]
            where = "" if include_resolved else "WHERE status = 'pending'"
            rows = s.execute(sa.text(
                f"SELECT {', '.join(sel)} FROM pending_approvals {where} "
                f"ORDER BY created_at DESC LIMIT :lim"), {"lim": max(1, min(int(limit), 200))}).fetchall()
            n_pending = s.execute(sa.text("SELECT COUNT(*) FROM pending_approvals WHERE status='pending'")).scalar()
        items = []
        for r in rows:
            it = dict(zip(sel, r))
            for k, v in list(it.items()):
                if v is not None and not isinstance(v, (str, int, float, bool)):
                    it[k] = str(v)
            eff = approval_effect(it.get("approval_type"))
            it["effect_en"], it["effect_zh"], it["executes"] = eff["en"], eff["zh"], eff["executes"]
            items.append(it)
        return {"n_pending": int(n_pending or 0), "charter": CHARTER, "approvals": items}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"approvals read failed: {exc}")


class ResolveRequest(BaseModel):
    ids: list[int]
    approved: bool
    rationale: str = ""
    category: str = "operator_review"


# ── v2 Governance Approval Gateway (deploy decisions queue) ─────────
# Separate namespace under /api/governance/approvals so it coexists
# cleanly with the legacy /api/approvals (ticker-level watchlist queue).
# v2 is for promote-to-paper-trade / promote-to-live / weight method
# change / manifest edit — the institutional two-eye + cooling-off
# gate for any change to active_deployment.yaml.


# ── Research Ops Inbox (composite research-process intel) ──────────


class ResearchOpsVisitBody(BaseModel):
    visited_ts: Optional[str] = None    # ISO; defaults to now


@app.get("/api/research_ops/literature", tags=["governance"])
def research_ops_literature(since: Optional[str] = None) -> dict:
    """Academic literature reading queue: LLM-scored papers + weekly digest.

    Split from /api/research_ops/inbox 2026-06-02 per the
    notifications-vs-reading-material doctrine. Papers + digest are
    reading material (deep work in Lab), not triage notifications."""
    def _produce() -> dict:
        from engine.inbox.composer import compose_literature
        return compose_literature(since_iso=since)
    try:
        return _cached(f"research_ops_literature:{since or ''}", 60.0, _produce)
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


@app.get("/api/research_ops/inbox", tags=["governance"])
def research_ops_inbox(since: Optional[str] = None) -> dict:
    """Research Ops inbox — composite research-process intel.

    Returns 4-lane aggregate (engine / direction / methodology / graveyard).
    Pass `since=<iso>` to mark items older than the timestamp as
    already-read (used for the topbar unread badge).

    Doctrine inline in response. Cheap (jsonl tails + file mtimes); no
    new computation. Cached 60s to avoid hammering the underlying probes."""
    def _produce() -> dict:
        from engine.inbox.composer import compose_inbox
        return compose_inbox(since_iso=since)
    try:
        return _cached(f"research_ops_inbox:{since or ''}", 60.0, _produce)
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


_RESEARCH_OPS_VISIT_PATH = _PATH(__file__).resolve().parent.parent / "data" / "research_ops_last_visit.json"


@app.get("/api/research_ops/last_visit", tags=["governance"])
def research_ops_last_visit() -> dict:
    """Return the last-visited timestamp recorded for the inbox.

    Used by the topbar inbox icon to compute the unread badge — items
    newer than this ts are 'unread'."""
    if _RESEARCH_OPS_VISIT_PATH.is_file():
        try:
            import json as _json
            return _json.loads(_RESEARCH_OPS_VISIT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"visited_ts": None}


@app.post("/api/research_ops/last_visit", tags=["governance"])
def research_ops_record_visit(body: ResearchOpsVisitBody) -> dict:
    """Record a visit timestamp. Called from /research-ops on mount."""
    import datetime as _dt
    import json as _json
    ts = body.visited_ts or _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _RESEARCH_OPS_VISIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"visited_ts": ts}
    _RESEARCH_OPS_VISIT_PATH.write_text(_json.dumps(payload), encoding="utf-8")
    # Bust the inbox cache so the next read reflects the new "since".
    for k in list(_TTL_CACHE.keys()):
        if k.startswith("research_ops_inbox:"):
            _TTL_CACHE.pop(k, None)
    return payload


class V2CreateRequest(BaseModel):
    request_type:        str
    title:               str
    summary:             str
    proposed_payload:    dict
    current_state:       dict
    evidence_pack:       Optional[dict] = None
    cooling_off_seconds: int = 86400
    expires_in_seconds:  int = 7 * 86400
    created_by:          Optional[str] = None


class V2DecisionRequest(BaseModel):
    decided_by: str
    reason:     Optional[str] = None
    # For approve only — bypass cooling-off if set true.
    force_pre_cooling: bool = False


@app.get("/api/governance/approvals", tags=["governance"])
def v2_list_approvals(
    status: Optional[str] = None,
    request_type: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """List governance approval requests from data/governance/approval_ledger.jsonl.

    Each item is the FOLDED current state (create + decisions merged).
    Optionally filter by status (pending/approved/rejected/expired) or
    request_type. Default limit 100, max 500."""
    from engine.governance.approval_ledger import list_requests, count_pending
    try:
        rows = list_requests(
            status=status,
            request_type=request_type,
            limit=max(1, min(int(limit), 500)),
        )
        return {
            "available":   True,
            "n_total":     len(rows),
            "n_pending":   count_pending(),
            "items":       rows,
        }
    except Exception as exc:
        logger.exception("v2_list_approvals failed")
        return {"available": False, "reason": str(exc)[:200]}


@app.post("/api/governance/approvals", tags=["governance"])
def v2_create_approval(req: V2CreateRequest) -> dict:
    """Create a new governance approval request.

    Required: request_type, title, summary, proposed_payload, current_state.
    Default cooling_off = 24h, expires = 7 days. Returns the request id +
    the folded state of the newly-created (pending) row."""
    from engine.governance.approval_ledger import create_request, get_request
    try:
        rid = create_request(
            request_type        = req.request_type,
            title               = req.title,
            summary             = req.summary,
            proposed_payload    = req.proposed_payload,
            current_state       = req.current_state,
            evidence_pack       = req.evidence_pack,
            cooling_off_seconds = req.cooling_off_seconds,
            expires_in_seconds  = req.expires_in_seconds,
            created_by          = req.created_by,
        )
        return {"id": rid, "row": get_request(rid)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("v2_create_approval failed")
        raise HTTPException(status_code=500, detail=str(exc)[:200])


@app.get("/api/governance/approvals/{request_id}", tags=["governance"])
def v2_get_approval(request_id: str) -> dict:
    """Get the folded state of one approval request."""
    from engine.governance.approval_ledger import get_request
    row = get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"approval {request_id} not found")
    return row


@app.post("/api/governance/approvals/{request_id}/approve", tags=["governance"])
def v2_approve(request_id: str, req: V2DecisionRequest) -> dict:
    """Approve a pending request. By default refuses if still in cooling-off
    (institutional two-eye + cooling-off doctrine). Pass force_pre_cooling
    true to override (will be flagged fast_approve=True in the ledger)."""
    from engine.governance.approval_ledger import approve_request
    try:
        return approve_request(
            request_id,
            approved_by       = req.decided_by,
            note              = req.reason,
            force_pre_cooling = req.force_pre_cooling,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/governance/approvals/{request_id}/reject", tags=["governance"])
def v2_reject(request_id: str, req: V2DecisionRequest) -> dict:
    """Reject a pending request. Reason mandatory (≥10 chars)."""
    from engine.governance.approval_ledger import reject_request
    if not req.reason or len(req.reason.strip()) < 10:
        raise HTTPException(status_code=422,
                            detail="rejection reason must be ≥ 10 chars")
    try:
        return reject_request(
            request_id,
            rejected_by = req.decided_by,
            reason      = req.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/approvals/resolve", tags=["governance"])
def resolve_approvals(req: ResolveRequest) -> dict:
    """Human approve/reject of pending proposals → engine.approval_workflow.bulk_resolve_pending_
    approvals (per-row validated side-effects + GIPS audit trail + red-line batch guard). A rationale
    is REQUIRED (audit). This is the ONLY mutating endpoint; local-only, human-triggered."""
    if not req.ids:
        raise HTTPException(status_code=422, detail="no approval ids")
    if not req.rationale.strip():
        raise HTTPException(status_code=422, detail="a rationale is required (audit trail)")
    from engine.approval_workflow import bulk_resolve_pending_approvals
    try:
        return bulk_resolve_pending_approvals(
            approval_ids     = [int(i) for i in req.ids],
            approved         = bool(req.approved),
            resolved_by      = "ui_operator",
            review_rationale = req.rationale.strip(),
            review_category  = (req.category or "supervisor_discretion").strip(),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"resolve failed: {exc}")


@app.get("/api/approvals/{approval_id}", tags=["governance"])
def approval_detail(approval_id: int) -> dict:
    """Drill-down decision context for ONE pending approval (P-AUDIT v1, deterministic, 0-LLM):
    the full get_approval_context() module set (action + governing spec + circuit-breaker status +
    HARKing flags + decision-time quant numbers + reject-cascade preview + 7-layer decision
    support) PLUS similar past precedents and the decision replay timeline.

    This is EVIDENCE for the human's decision — it never renders a verdict or a recommendation
    (the LLM is not in this layer; all retrieval is deterministic SQL + cosine). Compute-guarded
    (get_approval_context can fall back to a yfinance historical replay) + cached 30s."""
    def _produce() -> dict:
        from engine.approval_context import (
            REVIEW_CATEGORIES, get_approval_context,
            get_decision_replay, get_similar_past_approvals,
        )
        ctx = get_approval_context(int(approval_id))
        if not ctx.get("found"):
            return {"found": False, "approval_id": int(approval_id)}
        # Replay is pure SQL — best-effort, never fail the detail on it.
        try:
            ctx["decision_replay"] = get_decision_replay(int(approval_id))
        except Exception:
            ctx["decision_replay"] = []
        # Precedent runs a sentence-transformer RAG; the FIRST call after a cold start loads a
        # ~90MB model (seconds). Give it its OWN short budget so the cold load never holds the
        # core decision context hostage — degrade to "unavailable" (the model warms for next time).
        try:
            _fut = _EXEC.submit(get_similar_past_approvals, int(approval_id), 3)
            ctx["similar_past"] = _fut.result(timeout=5.0)
            ctx["similar_past_status"] = "ok"
        except Exception:
            ctx["similar_past"] = []
            ctx["similar_past_status"] = "unavailable"
        ctx["review_categories"] = list(REVIEW_CATEGORIES)
        return ctx

    ctx = _cached_compute(f"approval_detail::{approval_id}", ttl=30.0, timeout=15.0, produce=_produce)
    if not ctx.get("found"):
        raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")
    return ctx


@app.get("/api/risk", tags=["risk"])
def risk_console(as_of: str | None = None) -> dict:
    """Live Risk-Manager verdict console (spec id=69). Reconstructs the combined book from the UI
    artifact, runs the FAITHFUL gate functions that need only the combined series + VaR/ES (modes
    1a/3/4/5/6/6b/7/7b/8), and reports the full 13-mode grid (threshold · observed · verdict) + book
    risk metrics + overall severity. Signal-dependent modes (1b/2/9/10) reference the last
    orchestrator run. `as_of` time-travels to the artifact on/before that date. Read-only,
    deterministic — the math decides, this only displays."""
    return _cached_compute(f"risk:{as_of}", 20.0, 15.0, lambda: _risk_payload(as_of))


def _risk_payload(as_of: str | None = None) -> dict:
    import datetime as _dt
    try:
        import pandas as pd
        picked = _pick_artifact(as_of)
        if picked is None:
            raise HTTPException(status_code=404, detail="no UI artifact to evaluate")
        files = [picked]   # downstream references files[0]
        data = json.loads(picked.read_text(encoding="utf-8"))
        meta = data.get("_meta") or {}
        book = data.get("book_snapshot") or {}
        states = data.get("strategy_states") or []
        positions = data.get("positions") or []
        var_overlay = data.get("var_overlay") or {}

        # combined book weights = Σ_strategy (intra_weight × book_weight)
        bw = {s.get("strategy_name"): (s.get("book_weight") or 0.0) for s in states}
        combined: dict[str, float] = {}
        for p in positions:
            w = (p.get("intra_weight") or 0.0) * bw.get(p.get("strategy_name"), 0.0)
            combined[p.get("ticker")] = combined.get(p.get("ticker"), 0.0) + w
        ser = pd.Series(combined, dtype="float64") if combined else pd.Series(dtype="float64")

        gross = float(ser.abs().sum()) if len(ser) else float(book.get("gross") or 0.0)
        net = float(ser.sum()) if len(ser) else float(book.get("net") or 0.0)
        max_w = float(ser.abs().max()) if len(ser) else 0.0
        short_ratio = float(ser[ser < 0].abs().sum() / gross) if gross > 0 else 0.0
        hhi = float(((ser.abs() / gross) ** 2).sum()) if gross > 0 else 0.0
        n_ok = sum(1 for s in states if s.get("status") == "OK")
        var95 = var_overlay.get("var")
        es95 = var_overlay.get("cvar")

        # run the faithful gates that need only `combined` + var/es
        from engine.agents.risk_manager import gates as G
        from engine.agents.risk_manager.thresholds import RISK_THRESHOLDS as TH, BOOK_SINGLE_TICKER_ABS_CAP as CAP1A
        breaches = []
        for fn, args in [
            (G.gate_mode_1a_book_abs_cap, (ser,)), (G.gate_mode_3_gross_leverage, (ser,)),
            (G.gate_mode_4_net_exposure, (ser,)), (G.gate_mode_5_hhi, (ser,)),
            (G.gate_mode_8_short_side_ratio, (ser,)),
            (G.gate_mode_6_var_95, (var95,)), (G.gate_mode_6b_var_95_model_integrity, (var95,)),
            (G.gate_mode_7_es_95, (es95,)), (G.gate_mode_7b_es_95_model_integrity, (es95,)),
        ]:
            try:
                breaches += fn(*args)
            except Exception:
                pass
        by_mode = {b.mode_id: b for b in breaches}
        severity = G.classify_severity(breaches)

        def row(mode_id, name, observed, threshold, live=True):
            b = by_mode.get(mode_id)
            verdict = (b.severity if b else "PASS")  # HARD_HALT / SOFT_WARN / PASS
            return {"mode_id": mode_id, "name": name, "observed": observed,
                    "threshold": threshold, "verdict": verdict, "live": live,
                    "detail": (b.rule_description if b else "")}

        grid = [
            row("1a", "Single-name book cap", round(max_w, 4), f"≤ {CAP1A:.0%}"),
            row("1b", "Intra-strategy concentration", None, "per sleeve-class", live=False),
            row("2",  "Sleeve drift", None, f"≤ {TH.sleeve_drift_relative_max:.0%} rel", live=False),
            row("3",  "Gross leverage", round(gross, 4), f"≤ {TH.gross_leverage_max:.2f}×"),
            row("4",  "Net exposure", round(net, 4), f"[{TH.net_exposure_min:.0%}, {TH.net_exposure_max:.0%}]"),
            row("5",  "HHI concentration", round(hhi, 4), f"≤ {TH.hhi_max:.2f}"),
            row("6",  "VaR-95 (1-day)", (round(var95, 4) if var95 is not None else None), f"warn {TH.var_95_soft_warn:.0%} / halt {TH.var_95_hard_halt:.0%}"),
            row("7",  "ES-95 (1-day)", (round(es95, 4) if es95 is not None else None), f"warn {TH.es_95_soft_warn:.0%} / halt {TH.es_95_hard_halt:.0%}"),
            row("8",  "Short-side ratio", round(short_ratio, 4), f"≤ {TH.short_side_max_of_gross:.0%} of gross"),
            row("9",  "Min OK strategies", n_ok, f"≥ {TH.min_ok_strategies}"),
            row("10", "Cross-cancel tickers", None, f"≤ {TH.cross_cancel_ticker_max}", live=False),
        ]
        # Mode 9 verdict computed here (not a combined-only gate): WARN if below min.
        for r in grid:
            if r["mode_id"] == "9" and isinstance(n_ok, int):
                r["verdict"] = "PASS" if n_ok >= TH.min_ok_strategies else "HARD_HALT"
                r["live"] = True

        # Junior-analyst run-level rationale: Why CLEAR/HALT + binding constraint + headroom.
        # Computed from per-mode utilization (observed / numeric limit); the narrator owns the prose.
        def _u(mode, observed, limit, otxt, ltxt):
            return {"mode": mode, "observed_txt": otxt, "limit_txt": ltxt,
                    "util": (observed / limit if (observed is not None and limit) else float("nan"))}
        utils = [
            _u("single-name cap", max_w, CAP1A, f"{max_w:.1%}", f"{CAP1A:.0%}"),
            _u("gross leverage", gross, TH.gross_leverage_max, f"{gross:.2f}×", f"{TH.gross_leverage_max:.2f}×"),
            _u("HHI", hhi, TH.hhi_max, f"{hhi:.3f}", f"{TH.hhi_max:.2f}"),
            _u("short-side", short_ratio, TH.short_side_max_of_gross, f"{short_ratio:.0%}", f"{TH.short_side_max_of_gross:.0%}"),
            {"mode": "net exposure", "observed_txt": f"{net:+.0%}",
             "limit_txt": f"[{TH.net_exposure_min:.0%}, {TH.net_exposure_max:.0%}]",
             "util": (net / TH.net_exposure_max if net >= 0 and TH.net_exposure_max
                      else (net / TH.net_exposure_min if TH.net_exposure_min else float("nan")))},
        ]
        if var95 is not None:
            utils.append(_u("VaR-95", var95, TH.var_95_hard_halt, f"{var95:.1%}", f"halt {TH.var_95_hard_halt:.0%}"))
        if es95 is not None:
            utils.append(_u("ES-95", es95, TH.es_95_hard_halt, f"{es95:.1%}", f"halt {TH.es_95_hard_halt:.0%}"))
        if isinstance(n_ok, int) and n_ok > 0:
            utils.append({"mode": "OK strategies", "observed_txt": f"{n_ok}",
                          "limit_txt": f"≥ {TH.min_ok_strategies}", "util": TH.min_ok_strategies / n_ok})
        try:
            from engine.agents.risk_manager.narrator import narrate_risk_summary
            rationale = narrate_risk_summary(utils, severity, G.any_hard_halt(breaches))
        except Exception as _exc:
            rationale = ""

        return {
            "as_of": meta.get("as_of_date") or files[0].stem,
            "overall_severity": severity,
            "halt": G.any_hard_halt(breaches),
            "n_breaches": len(breaches),
            "rationale": rationale,
            "metrics": {"gross": round(gross, 4), "net": round(net, 4), "hhi": round(hhi, 4),
                        "max_weight": round(max_w, 4), "short_ratio": round(short_ratio, 4),
                        "n_positions": book.get("n_tickers") or len(positions),
                        "n_strategies": book.get("n_strategies") or len(states), "n_ok": n_ok,
                        "var95": var95, "es95": es95},
            "modes": grid,
            "decided_by": "engine.agents.risk_manager.gates",
            "narrated_by": "Risk Manager",
            "note": "Modes 1b/2/10 are signal-dependent — evaluated during the orchestrator run, "
                    "shown here for completeness. The rest are recomputed live from the book.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"risk console failed: {exc}")


@app.get("/api/provenance", tags=["meta"])
def provenance() -> dict:
    """Data-vintage / point-in-time panel — per-source as-of + age, from FAST sources only
    (file mtimes, the artifact build stamp, NAV last date, a DB max). FRED is network-verified by
    the daily DQ pre-batch gate, not re-fetched here (keeps the page instant). Makes the
    'point-in-time clean' claim verifiable."""
    return _cached_compute("provenance", 30.0, 10.0, _provenance_payload)


def _provenance_payload() -> dict:
    import datetime as _dt
    from pathlib import Path as _P

    today = _dt.date.today()

    def _bdays(d: _dt.date | None):
        if not d:
            return None
        if d >= today:
            return 0
        n, cur = 0, d + _dt.timedelta(days=1)
        while cur <= today:
            if cur.weekday() < 5:
                n += 1
            cur += _dt.timedelta(days=1)
        return n

    sources: list[dict] = []

    # 1. paper-trade UI artifact (book snapshot vintage + build stamp)
    try:
        ad = _P("data/ui_artifact")
        files = sorted(ad.glob("*.json"), reverse=True) if ad.is_dir() else []
        if files:
            meta = json.loads(files[0].read_text(encoding="utf-8")).get("_meta", {})
            sources.append({"source": "Paper-trade book artifact", "kind": "book snapshot",
                            "as_of": meta.get("as_of_date"), "built": meta.get("build_ts_utc")})
    except Exception:
        pass

    # 2. file-mtime sources (prices + earnings panel)
    for label, kind, paths in [
        ("yfinance ETF prices (BAB)", "prices", ["data/cache/bab_compat.parquet"]),
        ("D-PEAD signal panel", "earnings panel",
         ["data/cache/_pead_ts_panel_2014_2023.parquet", "data/cache/pead_ts_signal_panel.parquet"]),
    ]:
        p = next((_P(x) for x in paths if _P(x).is_file()), None)
        if p:
            mt = _dt.datetime.fromtimestamp(p.stat().st_mtime)
            sources.append({"source": label, "kind": kind,
                            "as_of": mt.date().isoformat(), "built": mt.isoformat(timespec="seconds")})

    # 3. NAV snapshot last date
    try:
        from engine.agents.persona.tools import read_nav_history
        nav = json.loads(read_nav_history(7))
        if nav.get("last_date"):
            sources.append({"source": "NAV snapshot", "kind": "nav", "as_of": nav["last_date"]})
    except Exception:
        pass

    # 4. FRED — network-verified by the daily gate, not re-fetched here
    sources.append({"source": "FRED macro series", "kind": "macro", "as_of": None,
                    "note": "network-verified by the DQ pre-batch gate (06:30 SGT); not re-fetched here"})

    for s in sources:
        try:
            s["bdays_stale"] = _bdays(_dt.date.fromisoformat(s["as_of"])) if s.get("as_of") else None
        except Exception:
            s["bdays_stale"] = None

    return {
        "as_of": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sources": sources,
        "point_in_time": "Signals use only data knowable at trade time; the book is look-ahead "
                         "audited. Vintages above are the latest persisted as-of per source.",
    }


# 2026-06-02 — Acknowledge ledger so users can mark alerts as
# "seen / handled" and have them disappear from the default view.
# Stored as JSONL so it's plain audit-friendly text. Natural key per
# alert/anomaly is hashed because the underlying rows lack ids.
from pathlib import Path as _Path
import hashlib as _hashlib

ALERT_ACK_LEDGER = _Path(__file__).resolve().parent.parent / "data" / "research" / "alert_ack_ledger.jsonl"


def _alert_natural_key(row: dict, kind: str) -> str:
    """Stable hash of the fields that identify an alert/anomaly row.
    The backend rows lack a primary id, so we hash the human-meaningful
    natural key: alert = (source, date, mode_id, rule_description);
    anomaly = (ticker, scan_date, detector, event_class)."""
    if kind == "anomaly":
        material = "|".join([
            str(row.get("ticker") or ""), str(row.get("scan_date") or ""),
            str(row.get("detector") or ""), str(row.get("event_class") or ""),
        ])
    else:
        material = "|".join([
            str(row.get("source") or ""), str(row.get("date") or ""),
            str(row.get("mode_id") or ""), str(row.get("rule_description") or ""),
        ])
    return _hashlib.blake2b(material.encode("utf-8"), digest_size=8).hexdigest()


def _load_ack_keys() -> dict[str, dict]:
    """Return {key: latest_ack_row} for every currently-acknowledged
    item. Latest-row-wins per key; a tombstone row (kind=='tombstone'
    with justification=='__unack__') drops the key out of the result
    so a previously-ack'd alert can be reactivated for review."""
    if not ALERT_ACK_LEDGER.is_file():
        return {}
    latest: dict[str, dict] = {}
    with ALERT_ACK_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            k = row.get("alert_key")
            if k:
                latest[k] = row
    # Tombstones suppress
    out: dict[str, dict] = {}
    for k, row in latest.items():
        if row.get("kind") == "tombstone":
            continue
        out[k] = row
    return out


@app.get("/api/alerts", tags=["risk"])
def alerts(days_back: int = 30) -> dict:
    """Operational monitor: persisted Risk-Manager + DQ-Inspector alerts and forensic anomaly
    flags (fast DB reads). Note: the LIVE DQ pre-batch gate is NOT run here (it does network
    freshness checks, ~30s+) — this surfaces what the daily crons already persisted.

    Each row gets enriched with `alert_key` (stable hash of the natural
    identifier) and `is_acknowledged` (boolean) so the UI can hide
    already-handled alerts by default."""
    # The cache bucket includes days_back; ack state changes outside
    # the cache so consumers see ack updates within 30s.
    return _cached_compute(f"alerts:{days_back}", 30.0, 15.0, lambda: _alerts_payload(days_back))


def _alerts_payload(days_back: int) -> dict:
    import datetime as _dt
    from engine.agents.persona.tools import query_recent_alerts, query_recent_anomalies
    al = json.loads(query_recent_alerts(days_back=days_back, severity_min="LIGHT", source="all"))
    an = json.loads(query_recent_anomalies(days_back=days_back, min_confidence=1, detector="all"))
    ack_keys = _load_ack_keys()

    def _enrich(rows: list, kind: str) -> list:
        out = []
        for r in rows:
            k = _alert_natural_key(r, kind)
            r2 = dict(r)
            r2["alert_key"] = k
            r2["is_acknowledged"] = k in ack_keys
            if k in ack_keys:
                r2["ack_ts"] = ack_keys[k].get("ts")
                r2["ack_justification"] = ack_keys[k].get("justification")
            out.append(r2)
        return out

    return {
        "as_of":       _dt.date.today().isoformat(),
        "days_back":   days_back,
        "n_alerts":    al.get("n_alerts", 0),
        "alerts":      _enrich(al.get("alerts", []), "alert"),
        "n_anomalies": an.get("n_flags", 0),
        "anomalies":   _enrich(an.get("flags", []), "anomaly"),
    }


class _AlertAckRequest(BaseModel):
    alert_key: str
    kind: str = "alert"            # "alert" | "anomaly"
    justification: str = ""        # optional reason — written to ledger
    actor: str = "ui"


@app.post("/api/alerts/acknowledge", tags=["risk"])
def alerts_acknowledge(body: _AlertAckRequest) -> dict:
    """Append an acknowledgement to alert_ack_ledger.jsonl. The /alerts
    endpoint reads this ledger every call so the UI sees the ack within
    one refresh cycle (≤30s cache). Re-ack of the same key is allowed
    and appended (preserves audit trail of repeated reviews)."""
    import datetime as _dt
    if not body.alert_key or not body.alert_key.strip():
        raise HTTPException(status_code=422, detail="alert_key is required")
    try:
        ALERT_ACK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts":            _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "alert_key":     body.alert_key,
            "kind":          body.kind,
            "justification": (body.justification or "")[:500],
            "actor":         (body.actor or "ui")[:60],
        }
        with ALERT_ACK_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return {"ok": True, "alert_key": body.alert_key}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class _AlertUnackRequest(BaseModel):
    alert_key: str


@app.post("/api/alerts/unacknowledge", tags=["risk"])
def alerts_unacknowledge(body: _AlertUnackRequest) -> dict:
    """Soft un-ack: appends a tombstone row that suppresses prior ack
    rows for the same key. Audit trail preserved.

    Implementation note: the ack-key state is computed by walking the
    ledger latest-row-wins per key; tombstone is a row with
    justification='__unack__' which is treated as 'not acknowledged'
    by _load_ack_keys's caller."""
    import datetime as _dt
    if not body.alert_key:
        raise HTTPException(status_code=422, detail="alert_key required")
    try:
        ALERT_ACK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts":            _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "alert_key":     body.alert_key,
            "kind":          "tombstone",
            "justification": "__unack__",
            "actor":         "ui",
        }
        with ALERT_ACK_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class LlmBudgetUpdate(BaseModel):
    monthly_cap_usd:     Optional[float] = None
    agent_caps_usd:      Optional[dict] = None
    alert_threshold_pct: Optional[float] = None


@app.get("/api/ops/llm_budget", tags=["ops"])
def ops_llm_budget() -> dict:
    """Operational LLM budget guardrail: month-to-date spend vs cap +
    per-agent breakdown w/ alert/over status. Sourced from
    engine.governance.llm_budget which reads
    data/governance/llm_budget.json (defaults applied if missing)."""
    from engine.governance import llm_budget as B
    try:
        return {
            "available":   True,
            "budget":      B.load_budget(),
            "usage":       B.compute_usage(),
        }
    except Exception as exc:
        logger.exception("ops_llm_budget failed")
        return {"available": False, "reason": str(exc)[:200]}


@app.post("/api/ops/llm_budget", tags=["ops"])
def ops_llm_budget_update(req: LlmBudgetUpdate) -> dict:
    """Update the LLM budget config. Any None field keeps its current value."""
    from engine.governance import llm_budget as B
    try:
        updated = B.save_budget(
            monthly_cap_usd     = req.monthly_cap_usd,
            agent_caps_usd      = req.agent_caps_usd,
            alert_threshold_pct = req.alert_threshold_pct,
        )
        return {"available": True, "budget": updated, "usage": B.compute_usage()}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("ops_llm_budget_update failed")
        raise HTTPException(status_code=500, detail=str(exc)[:200])


@app.get("/api/ops/cost", tags=["ops"])
def ops_cost() -> dict:
    """LLM cost ledger rollup for the Agentic AI-Ops panel: spend today / 7d / 30d / lifetime,
    total calls, and breakdowns by agent and by provider. Read-only; deterministic; the cost
    governance that backs the chat budget guard."""
    return _cached_compute("ops_cost", 30.0, 15.0, _ops_cost_payload)


def _ops_cost_payload() -> dict:
    import datetime as _dt
    try:
        from engine import llm_cost_ledger as L
        today = _dt.datetime.utcnow().date()
        since7 = today - _dt.timedelta(days=6)
        since30 = today - _dt.timedelta(days=29)
        win = lambda since: round(sum(e.cost_usd for e in L.get_calls(since=since, until=today)), 6)

        by_agent = L.get_total_by_agent()
        by_provider = L.get_total_by_provider()
        return {
            "as_of":        today.isoformat(),
            "today_usd":    L.get_total_today(),
            "last7_usd":    win(since7),
            "last30_usd":   win(since30),
            "lifetime_usd": L.get_lifetime_total(),
            "calls_total":  L.get_call_count(),
            "by_agent": [
                {"agent_id": k, "total_usd": v["total_usd"], "calls": v["calls"],
                 "last_ts": v["last_ts"], "providers": v["providers"]}
                for k, v in sorted(by_agent.items(), key=lambda kv: -kv[1]["total_usd"])
            ],
            "by_provider": [
                {"provider": k, "total_usd": v}
                for k, v in sorted(by_provider.items(), key=lambda kv: -kv[1])
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"cost ledger read failed: {exc}")


@app.get("/api/ops/health", tags=["ops"])
def ops_health() -> dict:
    """Agentic AI-Ops health: latency/success SLO (from agent_slo_metrics.jsonl), provider
    key-pool + locked workload routing, and governance posture (frozen-manifest fingerprint +
    SR-11-7 drift check + eval-case count + 0-LLM/authority/pinned flags). Each section is
    fail-soft (its own error sub-object) so one missing source never 500s the page."""
    return _cached_compute("ops_health", 30.0, 15.0, _ops_health_payload)


def _ops_health_payload() -> dict:
    import json as _json

    def _pctl(vals: list, q: float):
        if not vals:
            return None
        s = sorted(vals)
        return int(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))])

    def _agg(rs: list) -> dict:
        lat = [r.get("latency_ms") for r in rs if isinstance(r.get("latency_ms"), (int, float))]
        n = len(rs)
        succ = sum(1 for r in rs if r.get("success"))
        return {"n": n, "success_rate": round(succ / n, 4) if n else None,
                "p50_ms": _pctl(lat, 0.5), "p95_ms": _pctl(lat, 0.95)}

    out: dict = {}

    # ── SLO ──
    try:
        rows: list[dict] = []
        p = ROOT / "data" / "agent_slo_metrics.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(_json.loads(line))
                    except Exception:
                        pass
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r.get("agent_id", "?"), []).append(r)
        slo = _agg(rows)
        slo["by_agent"] = [
            {"agent_id": a, **_agg(rs), "last_ts": max((x.get("ts") for x in rs), default=None)}
            for a, rs in sorted(by.items(), key=lambda kv: -len(kv[1]))
        ]
        out["slo"] = slo
    except Exception as exc:
        out["slo"] = {"error": str(exc)}

    # ── providers (key pool + locked routing) ──
    try:
        from engine.llm.call import _WORKLOAD_ROUTING
        routing = [{"workload": w, "provider": pr, "model": mo}
                   for w, (pr, mo) in sorted(_WORKLOAD_ROUTING.items())]
        keys: list[dict] = []
        kp = ROOT / ".streamlit" / "key_pool_stats.json"
        if kp.exists():
            d = _json.loads(kp.read_text(encoding="utf-8"))
            for k, v in d.items():
                keys.append({"label": v.get("label", k), "status": v.get("status"),
                             "today_calls": v.get("today_calls"), "today_errors": v.get("today_errors"),
                             "total_calls": v.get("total_calls"), "total_errors": v.get("total_errors"),
                             "last_used": v.get("last_used")})
        out["providers"] = {"routing": routing, "keys": keys}
    except Exception as exc:
        out["providers"] = {"error": str(exc)}

    # ── governance (manifest fingerprint + drift) ──
    try:
        from engine.agents.eval.manifest import load_frozen, check_manifest
        from engine.agents.eval.cases import CASES
        frozen = load_frozen()
        chk = check_manifest()
        agents = [{"agent_id": a, "model": v.get("model"), "n_tools": v.get("n_tools"),
                   "prompt_sha": v.get("prompt_sha"), "tools_sha": v.get("tools_sha")}
                  for a, v in sorted(frozen.items())]
        out["governance"] = {
            "clean": chk["clean"], "changed": chk["changed"],
            "added": chk["added"], "removed": chk["removed"],
            "agents": agents, "eval_cases": len(CASES),
            "posture": {"llm_in_decision": False, "authority_enforced": True, "manifest_pinned": True},
        }
    except Exception as exc:
        out["governance"] = {"error": str(exc)}

    return out


# ── Agent behavioral-eval scores (the agentic dual-line, made provable) ───────────────────────
# /ops shows the governance POSTURE (frozen manifest, drift, case count); this surfaces the actual
# SCORES (per-case pass-rate + Wilson CI). Reading is free; RUNNING costs LLM, so it is opt-in
# (a cost-confirmed button) and runs the production runner as a background subprocess.
_EVAL_LOCK = _threading.Lock()
_EVAL_STATE: dict = {"running": False, "started_at": None, "finished_at": None,
                     "exit_code": None, "ok": None, "message": None}


def _run_eval_job() -> None:
    import datetime as _dt
    try:
        proc = _subprocess.run(
            [sys.executable, "-m", "engine.agents.eval.runner", "--live", "--save"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        )
        code = proc.returncode
        msg = "Eval complete — scores refreshed." if code == 0 else f"Eval exited {code} (see server log)."
    except _subprocess.TimeoutExpired:
        code, msg = -1, "Eval timed out (>900s)."
    except Exception as exc:  # noqa: BLE001
        code, msg = -2, f"Eval failed to launch: {exc}"
    with _EVAL_LOCK:
        _EVAL_STATE.update(running=False, finished_at=_dt.datetime.utcnow().isoformat(),
                           exit_code=code, ok=(code == 0), message=msg)


@app.get("/api/ops/eval-latest", tags=["ops"])
def ops_eval_latest() -> dict:
    """Last persisted agent behavioral-eval report (data/validation/agent_eval_report.json, written
    by `python -m engine.agents.eval.runner --save`): static all_pass + per-case live pass-rate +
    Wilson CI + cost + the report timestamp. {found: false} if eval has never been run. Read-only,
    free."""
    f = ROOT / "data" / "validation" / "agent_eval_report.json"
    if not f.is_file():
        return {"found": False}
    try:
        import datetime as _dt
        rep = json.loads(f.read_text(encoding="utf-8"))
        st = rep.get("static") or {}
        lv = rep.get("live") or {}
        cases = [{"case_id": c.get("case_id"), "agent_id": c.get("agent_id"), "pass": c.get("pass"),
                  "n": c.get("n"), "pass_rate": c.get("pass_rate"), "wilson_ci": c.get("wilson_ci")}
                 for c in (lv.get("cases") or []) if "error" not in c]
        return {
            "found": True,
            "generated_at": _dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
            "static_all_pass": st.get("all_pass"),
            "live": ({"pass_rate": lv.get("pass_rate"), "wilson_ci": lv.get("wilson_ci"),
                      "runs": lv.get("runs"), "runs_passed": lv.get("runs_passed"),
                      "n_cases": lv.get("n_cases"), "n_samples": lv.get("n_samples"),
                      "total_cost_usd": lv.get("total_cost_usd"), "cases": cases} if lv else None),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"eval-latest read failed: {exc}")


@app.post("/api/ops/eval-run", tags=["ops"])
def ops_eval_run() -> dict:
    """Run the live behavioral eval (background subprocess: runner --live --save) and refresh the
    scores. Costs LLM → gated on the provider key (503 if absent) and meant to be behind a UI cost-
    confirm. Concurrent runs refused. Poll GET /api/ops/eval-run for status; then re-read eval-latest."""
    if not _anthropic_key_present():
        raise HTTPException(status_code=503, detail="LLM provider not configured — eval needs ANTHROPIC_API_KEY.")
    import datetime as _dt
    with _EVAL_LOCK:
        if _EVAL_STATE["running"]:
            return {**_EVAL_STATE, "already_running": True}
        _EVAL_STATE.update(running=True, started_at=_dt.datetime.utcnow().isoformat(),
                           finished_at=None, exit_code=None, ok=None, message=None)
    _threading.Thread(target=_run_eval_job, daemon=True).start()
    return {**_EVAL_STATE, "already_running": False}


@app.get("/api/ops/eval-run", tags=["ops"])
def ops_eval_run_status() -> dict:
    with _EVAL_LOCK:
        return dict(_EVAL_STATE)


@app.get("/api/research/graveyard", tags=["research"])
def research_graveyard() -> dict:
    """The factor graveyard: honest negatives from the alpha campaign. Curated, version-
    controlled data (data/research/graveyard.json) — each entry maps to a verdict record
    (docs/capability_evidence/*) or a campaign commit. Live (surviving) mechanisms come from
    /api/decay/report, so the two views never disagree."""
    path = ROOT / "data" / "research" / "graveyard.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="graveyard.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/research/gate-runs", tags=["research"])
def research_gate_runs(limit: int = 100) -> dict:
    """Research-automation gate ledger: every hypothesis run through the rigorous gate
    (engine.research.pipeline) with its verdict (GREEN/YELLOW/RED), residual-α, deflated-SR,
    and the n_trials it was judged against. The campaign's multiple-testing record."""
    try:
        from engine.research.pipeline import read_ledger
        runs = read_ledger(max(1, min(int(limit), 500)))
        return {"n": len(runs), "runs": runs}
    except Exception as exc:
        return {"n": 0, "runs": [], "error": str(exc)[:200]}


class NominateRequest(BaseModel):
    url: str | None = None
    id: str | None = None
    title: str | None = None     # ignored, bookmarklet sends for logging


@app.post("/api/research/discovery/nominate", tags=["research"])
def research_discovery_nominate(req: NominateRequest) -> dict:
    """Manual paper nomination: paste a DOI / arxiv URL / OpenAlex Work ID /
    SSRN URL and we fetch metadata, score, and write to discovery_queue.jsonl.

    Senior 2026-05-30: replaces the standalone localhost:8770 review_ui
    server. Same nominate() core function — this just exposes it via the
    existing FastAPI server. Bookmarklet should POST to this endpoint.
    """
    from engine.research.discovery.review_ui import nominate
    raw = req.url or req.id or ""
    if not raw:
        raise HTTPException(status_code=400, detail="missing url or id field")
    try:
        return nominate(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@app.get("/api/research/discovery/queues", tags=["research"])
def research_discovery_queues(limit: int = 20) -> dict:
    """Current paper discovery queues — review (primary) + borderline
    (spot-check). Most recent first. Limit caps each list independently."""
    from engine.research.discovery.review_ui import (
        DISCOVERY_BORDERLINE, DISCOVERY_QUEUE, read_queue,
    )
    n = max(1, min(int(limit), 200))
    return {
        "review":     read_queue(DISCOVERY_QUEUE, limit=n),
        "borderline": read_queue(DISCOVERY_BORDERLINE, limit=n),
    }


class QueueActionRequest(BaseModel):
    source_id: str
    reason: str | None = None     # only used by skip
    target_status: str | None = None    # only used by promote (defaults PENDING)
    # Huatai 借鉴 ③: structured iteration discipline
    hypothesis: str | None = None        # promote-time
    failure_attribution: str | None = None  # skip-time enum


@app.post("/api/research/discovery/promote", tags=["research"])
def research_discovery_promote(req: QueueActionRequest) -> dict:
    """Promote a queued paper into the mechanism library as a PENDING
    stub. Removes the queue entry. The strict-gate pipeline will pick
    up the stub on its next library scan.

    Optional `hypothesis` field captures WHAT this candidate is testing
    (Huatai 借鉴 ③). Stored in promotion_metadata for audit trail."""
    from engine.research.discovery.queue_actions import promote
    try:
        return promote(
            req.source_id,
            target_status=(req.target_status or "PENDING"),
            hypothesis=req.hypothesis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


class ApproveBindingRequest(BaseModel):
    mechanism_id: str
    template_id: str
    binding: dict
    required_data: list[str] | None = None


@app.post("/api/research/discovery/approve_binding", tags=["research"])
def research_discovery_approve_binding(req: ApproveBindingRequest) -> dict:
    """Approve an LLM-proposed (or user-edited) binding for a promoted
    mechanism. Writes execution_template + required_data to the YAML;
    after this the forward_oos_runner can simulate the mechanism."""
    from engine.research.discovery.queue_actions import approve_binding
    try:
        return approve_binding(
            req.mechanism_id,
            template_id=req.template_id,
            binding=req.binding,
            required_data=req.required_data,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@app.post("/api/research/discovery/skip", tags=["research"])
def research_discovery_skip(req: QueueActionRequest) -> dict:
    """Skip a queued paper. Removes from queue, appends to
    discovery_rejected.jsonl (which feeds the graveyard for future
    similar-candidate auto-flagging).

    Optional `failure_attribution` field categorizes WHY this candidate
    is failing (Huatai 借鉴 ③). One of: data_quality / regime /
    cost_binding / spec_overfit / decay_postpub / crowding /
    implementation / off_topic / unclear. Propagates into graveyard
    reader so similar candidates inherit the failure_mode."""
    from engine.research.discovery.queue_actions import skip
    try:
        return skip(
            req.source_id,
            reason=(req.reason or "user_skip"),
            failure_attribution=req.failure_attribution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@app.get("/api/research/discovery/watchlist", tags=["research"])
def research_discovery_watchlist() -> dict:
    """Forward OOS watchlist: promoted mechanisms being tracked for
    real-data calibration vs auto-gate synthetic verdict."""
    from engine.research.discovery.forward_oos_observer import (
        get_watchlist, watchlist_summary,
    )
    try:
        return {
            "summary": watchlist_summary(),
            "entries": get_watchlist(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@app.get("/api/research/discovery/watchlist/{mechanism_id}", tags=["research"])
def research_discovery_watchlist_detail(mechanism_id: str) -> dict:
    """Detail view + calibration delta (auto-gate vs forward-OOS) for
    a specific watchlist entry."""
    from engine.research.discovery.forward_oos_observer import (
        check_implementation_status, compute_calibration_delta, get_watchlist,
    )
    try:
        all_entries = get_watchlist()
        entry = next(
            (e for e in all_entries if e.get("mechanism_id") == mechanism_id),
            None,
        )
        if not entry:
            raise HTTPException(status_code=404,
                                  detail=f"{mechanism_id} not in watchlist")
        return {
            "entry":          entry,
            "implementation": check_implementation_status(mechanism_id),
            "calibration":    compute_calibration_delta(mechanism_id),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@app.get("/api/research/discovery/health", tags=["research"])
def research_discovery_health() -> dict:
    """Pipeline health check — discovery + book + ops side.

    Status: OK / WARN / ALERT / UNKNOWN — frontend can render the
    top-of-Research-page badge from this. Each check has detail +
    remedy (borrowed from PIT audit display style)."""
    from engine.research.discovery.pipeline_health import report
    try:
        return report()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@app.get("/api/research/discovery/bookmarklet", tags=["research"])
def research_discovery_bookmarklet() -> dict:
    """Returns the JS bookmarklet snippet for one-click paper add. The
    frontend renders this as a draggable bookmark on the Research page.
    Single source of truth for the JS so the snippet auto-tracks the
    real API endpoint (POST /api/research/discovery/nominate)."""
    js = (
        "javascript:(function(){"
        "fetch('http://localhost:8000/api/research/discovery/nominate',{method:'POST',"
        "headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({url:location.href,title:document.title})"
        "}).then(r=>r.json()).then(d=>{"
        "if(d.detail||d.error)alert('Error: '+(d.detail||d.error));"
        "else alert('Added: '+(d.title||'').slice(0,80)+"
        "' (conf='+d.confidence.toFixed(2)+', '+d.routing+')');"
        "}).catch(e=>alert('Server not running? '+e));"
        "})();"
    )
    return {
        "bookmarklet": js,
        "endpoint":   "/api/research/discovery/nominate",
        "instructions": (
            "Drag the [Add to Research Queue] link below to your bookmarks "
            "bar. On any paper page (arxiv / DOI / OpenAlex / SSRN) click "
            "the bookmark to instantly add the paper. Requires the API "
            "server running on localhost:8000."
        ),
    }


@app.get("/api/research/pit-audit", tags=["research"])
def pit_audit() -> dict:
    """Point-in-time / look-ahead integrity audit — the #1 quant credibility lever: evidence that no
    future data leaks into a signal. Surfaces the latest pit_audit_book_*.json (per-strategy PIT
    controls + code anchors) + pit_audit_dpead_*.json (the deployed SUE panel's look-ahead checks,
    incl. honestly-FLAGged limitations). Read-only; makes the landing's 'point-in-time clean' claim
    verifiable. available:False if no audit has been run."""
    vd = ROOT / "data" / "validation"

    def _latest(prefix: str):
        fs = sorted(vd.glob(f"{prefix}_*.json")) if vd.is_dir() else []
        try:
            return json.loads(fs[-1].read_text(encoding="utf-8")) if fs else None
        except Exception:
            return None

    book, dpead = _latest("pit_audit_book"), _latest("pit_audit_dpead")
    if not book and not dpead:
        return {"available": False}
    return {"available": True, "book": book, "dpead": dpead}


# ── Research event store (typed Claude↔project event ledger) ───────


# ── Phase 1.2 / 4.1 / B surfaces (rigor / audit / belief) ──────────
#
# These three feeds were emitting to jsonl but had no API surface →
# /dashboard could not show whether the safety nets (post-GREEN rigor,
# adversarial audit, belief-layer feedback) were firing. Added 2026-06-14
# per UI-architecture audit ("safety rails invisible = no safety rails").

@app.get("/api/research/post_green_rigor/recent", tags=["research"])
def post_green_rigor_recent(days: int = 7, limit: int = 50) -> dict:
    """Recent post-GREEN rigor results (OOS / spanning / borrow-cost).
    Reads data/research/post_green_rigor.jsonl tail. Newest first.
    `flags` non-empty => surfaced as critical_flag for UI badge."""
    days = max(1, min(int(days), 90))
    limit = max(1, min(int(limit), 500))
    p = ROOT / "data" / "research" / "post_green_rigor.jsonl"
    if not p.is_file():
        return {"n": 0, "n_flagged": 0, "rows": []}
    import datetime as _dt
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    rows: list[dict] = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            ts_raw = (r.get("ts") or "").rstrip("Z")
            try:
                ts = _dt.datetime.fromisoformat(ts_raw)
            except Exception:
                continue
            if ts < cutoff:
                continue
            rows.append({
                "rigor_id":         r.get("rigor_id"),
                "ts":               r.get("ts"),
                "verdict_event_id": r.get("verdict_event_id"),
                "hypothesis_id":    r.get("hypothesis_id"),
                "family":           r.get("family"),
                "template_name":    r.get("template_name"),
                "original_verdict": r.get("original_verdict"),
                "oos_status":       (r.get("post_pub_oos") or {}).get("status"),
                "oos_verdict":      (r.get("post_pub_oos") or {}).get("oos_verdict"),
                "spanning_status":  (r.get("spanning") or {}).get("status"),
                "spanning_alpha_t": (r.get("spanning") or {}).get("alpha_t"),
                "borrow_status":    (r.get("borrow_cost") or {}).get("status"),
                "flags":            r.get("flags") or [],
            })
    except Exception as exc:
        logger.exception("post_green_rigor_recent failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])
    rows.sort(key=lambda x: x.get("ts") or "", reverse=True)
    rows = rows[:limit]
    n_flagged = sum(1 for r in rows if r["flags"])
    return {"n": len(rows), "n_flagged": n_flagged, "rows": rows}


@app.get("/api/research/external_audits/recent", tags=["research"])
def external_audits_recent(days: int = 7, limit: int = 50) -> dict:
    """Recent external-LLM adversarial audit calls (Phase 1.2).
    Reads data/research/external_audits.jsonl tail. severity in
    {"skipped","ok","concern","critical"}. concern/critical count for
    KpiHero badge."""
    days = max(1, min(int(days), 90))
    limit = max(1, min(int(limit), 500))
    p = ROOT / "data" / "research" / "external_audits.jsonl"
    if not p.is_file():
        return {"n": 0, "n_concern": 0, "n_critical": 0, "rows": []}
    import datetime as _dt
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    rows: list[dict] = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            ts_raw = (r.get("ts") or "").rstrip("Z")
            try:
                ts = _dt.datetime.fromisoformat(ts_raw)
            except Exception:
                continue
            if ts < cutoff:
                continue
            rows.append({
                "audit_id":           r.get("audit_id"),
                "ts":                 r.get("ts"),
                "audit_subject":      r.get("audit_subject"),
                "subject_ref":        r.get("subject_ref"),
                "provider":           r.get("provider"),
                "severity":           r.get("severity"),
                "flagged_categories": r.get("flagged_categories") or [],
                "cost_estimate_usd":  r.get("cost_estimate_usd"),
            })
    except Exception as exc:
        logger.exception("external_audits_recent failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])
    rows.sort(key=lambda x: x.get("ts") or "", reverse=True)
    rows = rows[:limit]
    n_concern  = sum(1 for r in rows if r.get("severity") == "concern")
    n_critical = sum(1 for r in rows if r.get("severity") == "critical")
    return {"n": len(rows), "n_concern": n_concern, "n_critical": n_critical, "rows": rows}


@app.get("/api/research/belief/families", tags=["research"])
def belief_families(min_obs: int = 3) -> dict:
    """Per-family belief summary (Phase B) backing the synthesis prompt.
    Drives /dashboard belief-state panel + tells user which families
    Sonnet is being told to EXPLORE / AVOID / treat as THIN."""
    try:
        from engine.research.belief_synthesis_context import build_belief_summary
        beliefs = build_belief_summary(min_obs_per_family=max(1, int(min_obs)))
        families = [
            {
                "family":         b.family,
                "n_obs":          b.n_obs,
                "n_green":        b.n_green,
                "n_marginal":     b.n_marginal,
                "n_red":          b.n_red,
                "direction_hint": b.direction_hint,
            }
            for b in beliefs
        ]
        return {
            "n_families":   len(families),
            "n_total_obs":  sum(b["n_obs"] for b in families),
            "n_green_total":   sum(b["n_green"] for b in families),
            "n_marginal_total":sum(b["n_marginal"] for b in families),
            "n_red_total":     sum(b["n_red"] for b in families),
            "families":     families,
        }
    except Exception as exc:
        logger.exception("belief_families failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/belief/calibration", tags=["research"])
def belief_calibration() -> dict:
    """Surface the belief layer's published calibration headline numbers
    on the UI so the system's most important honest finding is visible
    everywhere — not buried in a markdown file (2026-06-23).

    Reads pre-computed rigor JSON (refreshed daily by
    daily-belief-refresh cron 06:35). Cheap; no bootstrap on request path.

    Returns:
        predictor_brier:        T1 mean Brier (observed predictor)
        predictor_ci_lo/hi:     T1 95% bootstrap CI bounds
        random_baseline:        4/9 ≈ 0.444 (definitional 3-class random)
        family_prior_brier:     T2 time-aware family-prior baseline
                                (the FAIR baseline, no future-info leak)
        delta_predictor_minus_fp:  predictor LOSES by this much (positive
                                   = honest negative finding)
        delta_ci_lo/hi:         95% CI on the delta
        n_autopsies:            sample size
        hl_p_value:             Hosmer-Lemeshow calibration GoF p-value
        hl_calibrated:          T6 verdict (False = REJECTED)
    """
    try:
        import json as _json
        from pathlib import Path as _Path
        path = _Path(__file__).resolve().parent.parent / "data" / "research" / "belief_track_record_rigor.json"
        if not path.is_file():
            return {"available": False, "reason": "rigor JSON not generated yet"}
        d = _json.loads(path.read_text(encoding="utf-8"))
        t1 = d.get("T1_overall_brier_bootstrap", {})
        t2 = d.get("T2_time_aware_family_prior", {})
        t6 = d.get("T6_hosmer_lemeshow", {})
        return {
            "available":               True,
            "n_autopsies":             d.get("n_autopsies", 0),
            "predictor_brier":         round(t1.get("observed_mean", 0.0), 4),
            "predictor_ci_lo":         round(t1.get("ci_95_lo",   0.0), 4),
            "predictor_ci_hi":         round(t1.get("ci_95_hi",   0.0), 4),
            "random_baseline":         round(t1.get("baseline_random_3class", 0.4444), 4),
            "predictor_beats_random":  bool(t1.get("significantly_better", False)),
            "family_prior_brier":      round(t2.get("time_aware_family_prior_brier", 0.0), 4),
            "delta_predictor_minus_fp":round(t2.get("mean_delta_predictor_minus_fp", 0.0), 4),
            "delta_ci_lo":             round(t2.get("delta_ci_95_lo", 0.0), 4),
            "delta_ci_hi":             round(t2.get("delta_ci_95_hi", 0.0), 4),
            "hl_chi2":                 round(t6.get("chi2", 0.0), 3),
            "hl_p_value":              round(t6.get("p_value", 1.0), 4),
            # JSON stores this as a string literal 'True'/'False', not a bool.
            "hl_calibrated":           str(t6.get("calibrated_fail_to_reject_h0_at_0_05", "True")).lower() == "true",
        }
    except Exception as exc:
        logger.exception("belief_calibration failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/workflow/counts", tags=["research"])
def workflow_counts() -> dict:
    """Aggregator for the workflow-trace SVG page (2026-06-23).

    Returns one row per stage in the paper → verdict pipeline so the
    UI can draw a single picture of the end-to-end system state.

    Cheap: reads jsonl line counts (mmap-friendly) + 1 cached JSON
    for the Brier headline. No bootstrap, no compute. Safe at 30s
    poll cadence.
    """
    import json as _json
    from collections import Counter as _Counter
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent

    def _wc(rel: str) -> int:
        p = root / rel
        if not p.is_file():
            return 0
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    n_papers_cached    = _wc("data/papers_curator/cache.jsonl")
    n_papers_registry  = _wc("data/research_store/papers_registry.jsonl")
    n_hypotheses       = _wc("data/research_store/hypotheses.jsonl")
    n_specs            = _wc("data/research_store/hypothesis_specs.jsonl")
    n_predictions      = _wc("data/research/predictions.jsonl")
    n_autopsies        = _wc("data/research/autopsies.jsonl")

    events_path = root / "data" / "research_store" / "events.jsonl"
    ec: _Counter = _Counter()
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                ec[_json.loads(s).get("event_type", "unknown")] += 1
            except _json.JSONDecodeError:
                continue

    # Belief headline (reuse the calibration endpoint's logic)
    brier = None
    delta_vs_fp = None
    try:
        rigor_path = root / "data" / "research" / "belief_track_record_rigor.json"
        if rigor_path.is_file():
            d = _json.loads(rigor_path.read_text(encoding="utf-8"))
            t1 = d.get("T1_overall_brier_bootstrap", {})
            t2 = d.get("T2_time_aware_family_prior", {})
            brier = round(t1.get("observed_mean", 0.0), 4)
            delta_vs_fp = round(t2.get("mean_delta_predictor_minus_fp", 0.0), 4)
    except Exception:
        pass

    return {
        "stages": [
            {
                "key":   "papers",
                "label": "Papers ingested",
                "count": n_papers_cached,
                "sub":   f"{n_papers_registry} registered",
                "href":  "/research/papers",
                "description": "arxiv q-fin + SSRN crawl, ClaimType-tagged",
            },
            {
                "key":   "synthesis",
                "label": "Synthesis runs",
                "count": ec.get("papers_curator_synthesis_run", 0),
                "sub":   "cross-paper hypothesis generation",
                "href":  "/research/papers/incoming",
                "description": "papers_curator cross-source synthesis (Sonnet)",
            },
            {
                "key":   "hypotheses",
                "label": "Hypotheses",
                "count": n_hypotheses,
                "sub":   f"{ec.get('forward_vector_created', 0)} forward vectors",
                "href":  "/research/hypothesis",
                "description": "extracted + reviewed candidate strategies",
            },
            {
                "key":   "specs",
                "label": "FactorSpecs",
                "count": n_specs,
                "sub":   "hash-locked, audit-trail",
                "href":  "/research/forward",
                "description": "executable FactorSpec — deterministic dispatch input",
            },
            {
                "key":   "predictions",
                "label": "Predictions",
                "count": n_predictions,
                "sub":   "AIR-GAPPED from verdict logic",
                "href":  "/research/calibration",
                "description": "Belief Layer Phase 1: predict-commit before dispatch",
            },
            {
                "key":   "verdicts",
                "label": "Verdicts",
                "count": ec.get("factor_verdict_filed", 0),
                "sub":   f"{ec.get('capability_evidence_filed', 0)} evidence files",
                "href":  "/research/lessons",
                "description": "GREEN/MARGINAL/RED from rigor dispatch",
            },
            {
                "key":   "autopsies",
                "label": "Autopsies",
                "count": n_autopsies,
                "sub":   "prediction ↔ verdict join",
                "href":  "/research/calibration",
                "description": "Belief Layer Phase 2: closed-loop measurement",
            },
            {
                "key":   "belief",
                "label": "Brier",
                "count": brier if brier is not None else 0,
                "is_float": True,
                "sub":   (
                    f"+{delta_vs_fp:.3f} vs family-prior"
                    if delta_vs_fp is not None else
                    "calibration headline"
                ),
                "href":  "/research/calibration",
                "description": "Belief Layer Phase 3: published calibration track record",
            },
        ],
        "secondary_counts": {
            "memory_doctrine_locked":  ec.get("memory_doctrine_locked", 0),
            "decay_alert":             ec.get("decay_alert", 0),
            "post_green_rigor_run":    ec.get("post_green_rigor_run", 0),
            "dq_breach":               ec.get("dq_breach", 0),
            "council_critique":        ec.get("council_critique", 0),
        },
    }


@app.get("/api/ops/cron_health", tags=["ops"])
def ops_cron_health() -> dict:
    """OPS hygiene (2026-06-15). For each registered cron's health
    file, report when it last ran successfully + how many hours stale.
    KpiHero chip aggregates these into "Crons N/M OK"."""
    import datetime as _dt
    health_dir = ROOT / "data" / "agents" / "_health"
    if not health_dir.is_dir():
        return {"n": 0, "rows": [], "n_healthy": 0, "n_stale": 0}
    # Convention: each cron writes one .jsonl with rows {agent_id, ts,
    # status, ...}. We pick newest row per file + measure staleness.
    rows = []
    now = _dt.datetime.utcnow()
    for fp in sorted(health_dir.glob("*.jsonl")):
        last = None
        for ln in fp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if last is None or (r.get("ts") or "") > (last.get("ts") or ""):
                last = r
        if last is None:
            continue
        try:
            ts = _dt.datetime.fromisoformat((last.get("ts") or "").rstrip("Z"))
            hours = round((now - ts).total_seconds() / 3600.0, 1)
        except Exception:
            hours = None
        rows.append({
            "agent_id":      last.get("agent_id") or fp.stem,
            "last_ts":       last.get("ts"),
            "hours_stale":   hours,
            "last_status":   last.get("status"),
            "is_healthy":    bool(hours is not None and hours < 36
                                  and (last.get("status") == "ok")),
        })
    n_healthy = sum(1 for r in rows if r["is_healthy"])
    n_stale   = sum(1 for r in rows if not r["is_healthy"])
    return {"n": len(rows), "rows": rows,
            "n_healthy": n_healthy, "n_stale": n_stale}


@app.get("/api/ops/llm_budget_chip", tags=["ops"])
def ops_llm_budget_chip(monthly_cap_usd: float = 100.0) -> dict:
    """OPS hygiene (2026-06-15). Month-to-date LLM spend vs cap.
    Designed for the KpiHero chip — returns a single-shot status
    (ok / warn / alert) based on % consumed."""
    import datetime as _dt
    ledger = ROOT / "data" / "llm_cost_ledger.jsonl"
    now = _dt.datetime.utcnow()
    month_iso = now.strftime("%Y-%m")
    spent = 0.0
    if ledger.is_file():
        for ln in ledger.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            ts = r.get("ts") or r.get("timestamp") or ""
            if ts[:7] != month_iso:
                continue
            try:
                spent += float(r.get("cost_usd") or r.get("cost") or 0)
            except Exception:
                pass
    pct = (spent / monthly_cap_usd * 100.0) if monthly_cap_usd > 0 else 0
    if pct >= 100:
        tone = "alert"
    elif pct >= 80:
        tone = "warn"
    else:
        tone = "ok"
    return {
        "spent_mtd_usd":   round(spent, 2),
        "monthly_cap_usd": monthly_cap_usd,
        "pct_consumed":    round(pct, 1),
        "tone":            tone,
        "month":           month_iso,
    }


@app.get("/api/research/brainstorm/metrics", tags=["research"])
def brainstorm_metrics() -> dict:
    """End-to-end measurement substrate (2026-06-15) — per-pack funnel
    (drafts → promoted → GREEN), LLM calibration (novelty bucket vs
    actual outcome), failure-mode histogram (last 90d). Read-time
    JOIN over brainstorm_drafts / decisions / hypotheses /
    factor_verdict_filed events / red_attributions. No LLM, no
    persistence — pure analysis."""
    try:
        from engine.research.brainstorm_metrics import compute_metrics
        return compute_metrics()
    except Exception as exc:
        logger.exception("brainstorm_metrics failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/red_attribution/{verdict_event_id}", tags=["research"])
def red_attribution_get(verdict_event_id: str) -> dict:
    """List failure attributions for one RED verdict."""
    try:
        from engine.research.red_attribution import for_verdict
        rows = for_verdict(verdict_event_id)
        return {"verdict_event_id": verdict_event_id, "n": len(rows), "rows": rows}
    except Exception as exc:
        logger.exception("red_attribution_get failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/red_attribution/{verdict_event_id}", tags=["research"])
def red_attribution_post(verdict_event_id: str,
                          red_category: str,
                          rationale: str,
                          hypothesis_id: str | None = None,
                          attributed_by: str = "principal") -> dict:
    """Record RED failure attribution. Mandatory enum category +
    non-trivial rationale (Tetlock 2015 forecast journal discipline)."""
    try:
        from engine.research.red_attribution import attribute, AttributionError
        import dataclasses as _dc
        row = attribute(verdict_event_id, red_category, rationale,
                         hypothesis_id=hypothesis_id,
                         attributed_by=attributed_by)
        return {"status": "ok", "attribution": _dc.asdict(row)}
    except AttributionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("red_attribution_post failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/red_attribution/_enums", tags=["research"])
def red_attribution_enums() -> dict:
    """Available red_category enum values + a short description per."""
    return {
        "categories": [
            {"key": "data_regime_mismatch",
             "desc": "data window doesn't cover the regime where idea matters"},
            {"key": "mechanism_wrong",
             "desc": "economic intuition is just wrong"},
            {"key": "already_spanned",
             "desc": "FF5+MOM explains it (R12 lesson should've caught)"},
            {"key": "cost_killed",
             "desc": "alpha disappears under realistic costs"},
            {"key": "sub_period_unstable",
             "desc": "GREEN in some sub-period, RED in another"},
            {"key": "novelty_overclaim",
             "desc": "γ catalog already had it (γ missed the match)"},
            {"key": "power_too_low",
             "desc": "n insufficient to detect realistic effect"},
            {"key": "implementation_bug",
             "desc": "our code, not the idea"},
            {"key": "graveyard_dup",
             "desc": "close to existing RED autopsy (collision missed)"},
            {"key": "data_unavailable",
             "desc": "required data doesn't exist / is paywalled"},
            {"key": "other",
             "desc": "rationale should be specific"},
        ],
    }


@app.get("/api/research/brainstorm/drafts", tags=["research"])
def brainstorm_drafts(limit: int = 50, pack: str | None = None) -> dict:
    """List brainstorm drafts (newest first). Annotates each draft
    with its latest decision (promote / reject / pending) so the UI
    can show decision status inline."""
    try:
        from engine.research.brainstorm.divergent_generator import list_drafts
        from engine.research.brainstorm.promoter import decision_for_idea
        rows = list_drafts(limit=limit, pack=pack)
        for r in rows:
            dec = decision_for_idea(r.get("idea_id", ""))
            r["decision"] = dec   # None = pending
        return {"n": len(rows), "rows": rows}
    except Exception as exc:
        logger.exception("brainstorm_drafts failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/brainstorm/run", tags=["research"])
def brainstorm_run(pack: str, trigger: str = "manual",
                    context: str = "") -> dict:
    """Trigger a brainstorm session (single Sonnet call, ~$0.05).
    Returns the new ideas in-line for immediate display."""
    try:
        from engine.research.brainstorm.divergent_generator import brainstorm_session
        import dataclasses as _dc
        ideas = brainstorm_session(pack, trigger=trigger, trigger_context=context)
        if ideas is None:
            raise HTTPException(status_code=502,
                detail="brainstorm LLM call failed or did not emit tool")
        return {
            "pack": pack,
            "trigger": trigger,
            "n_ideas": len(ideas),
            "ideas": [_dc.asdict(i) for i in ideas],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("brainstorm_run failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/brainstorm/drafts/{idea_id}/prevet", tags=["research"])
def brainstorm_prevet(idea_id: str) -> dict:
    """Run γ replication check on a brainstorm draft idea (no hypothesis
    promotion needed). Asks Sonnet 'is this claim already in HXZ 2020 /
    MP 2016 / LR 2018 / FF 2018 catalogs?' before PM spends time. ~$0.05.
    Uses idea_id as the persist key so the result is fetchable via the
    standard replication endpoint."""
    try:
        from engine.research.brainstorm.divergent_generator import list_drafts
        from engine.research.replication_checker import check_replication_text
        import dataclasses as _dc
        # Look up draft
        drafts = list_drafts(limit=500)
        idea = next((d for d in drafts if d.get("idea_id") == idea_id), None)
        if idea is None:
            raise HTTPException(status_code=404,
                detail=f"brainstorm idea {idea_id} not found")
        check = check_replication_text(
            claim_text         = idea.get("claim_one_line", ""),
            mechanism_family   = "(brainstorm)",
            target_asset_class = idea.get("target_asset_class", ""),
            persist_key        = idea_id,
        )
        if check is None:
            raise HTTPException(status_code=502,
                detail="γ replication LLM call failed")
        row = _dc.asdict(check)
        row["flags"] = [_dc.asdict(f) for f in check.flags]
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("brainstorm_prevet failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/brainstorm/drafts/{idea_id}/decide", tags=["research"])
def brainstorm_decide(idea_id: str, decision: str,
                       rationale: str = "",
                       decided_by: str = "principal") -> dict:
    """PM decision: promote (→ writes hypothesis.jsonl row) or reject.
    Mandatory non-empty rationale per audit P1 accountability."""
    try:
        from engine.research.brainstorm.promoter import decide, PromoterError
        row = decide(idea_id, decision, rationale, decided_by=decided_by)
        return {"status": "ok", "decision": row}
    except PromoterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("brainstorm_decide failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/brainstorm/seed_packs", tags=["research"])
def brainstorm_seed_packs() -> dict:
    """List 7 available seed packs with metadata for the UI selector."""
    try:
        from engine.research.brainstorm.divergent_generator import (
            VALID_PACKS, _load_pack,
        )
        out = []
        for p in VALID_PACKS:
            try:
                pack = _load_pack(p)
                out.append({
                    "name": pack.get("name"),
                    "domain": pack.get("domain"),
                    "short_description": pack.get("short_description", "").strip(),
                    "n_principles": len(pack.get("principles") or []),
                    "n_examples": len(pack.get("example_applications") or []),
                })
            except Exception:
                continue
        return {"n": len(out), "packs": out}
    except Exception as exc:
        logger.exception("brainstorm_seed_packs failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/brainstorm/lessons", tags=["research"])
def brainstorm_lessons(refresh: bool = False) -> dict:
    """Layer 1 of the brainstorm architecture — DETERMINISTIC lessons
    distilled from our own autopsy / decay / pre-mortem / replication /
    transfer / verdict history. Becomes prior context for Layer 3
    multi-provider divergent brainstorm. NEVER uses an LLM (rules in
    Python only — using LLM here = same-model self-reinforcement).

    refresh=true re-runs the distiller before returning."""
    try:
        from engine.research.brainstorm.lesson_distiller import (
            distill_lessons, persist_lessons, load_lessons,
        )
        import dataclasses as _dc
        if refresh:
            lessons = distill_lessons()
            persist_lessons(lessons)
            rows = [_dc.asdict(L) for L in lessons]
        else:
            rows = load_lessons()
        # Quick breakdown for the UI badge
        kind_counts: dict[str, int] = {}
        for L in rows:
            k = (L.get("kind") or "").split("_")[0]
            kind_counts[k] = kind_counts.get(k, 0) + 1
        return {"n": len(rows), "kind_counts": kind_counts, "rows": rows}
    except Exception as exc:
        logger.exception("brainstorm_lessons failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/replication/{hypothesis_id}", tags=["research"])
def replication_get(hypothesis_id: str) -> dict:
    """γ Replication Check history for a hypothesis (newest first)."""
    try:
        from engine.research.replication_checker import list_for_hypothesis
        rows = list_for_hypothesis(hypothesis_id)
        return {
            "hypothesis_id": hypothesis_id,
            "n": len(rows),
            "rows": rows,
            "latest": rows[0] if rows else None,
        }
    except Exception as exc:
        logger.exception("replication_get failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/replication/{hypothesis_id}/generate", tags=["research"])
def replication_generate(hypothesis_id: str) -> dict:
    """Run a fresh γ Replication Check (~$0.05 Sonnet call)."""
    try:
        from engine.research.replication_checker import check_replication
        import dataclasses as _dc
        check = check_replication(hypothesis_id)
        if check is None:
            raise HTTPException(status_code=502,
                detail="replication-checker LLM call failed or did not emit tool")
        row = _dc.asdict(check)
        row["flags"] = [_dc.asdict(f) for f in check.flags]
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("replication_generate failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/transfers/by_sleeve/{sleeve_id}", tags=["research"])
def transfers_by_sleeve(sleeve_id: str) -> dict:
    """β Cross-Domain Transfer proposals for one deployed sleeve.
    Returns history (newest first)."""
    try:
        from engine.research.cross_domain_transfer import list_for_sleeve
        rows = list_for_sleeve(sleeve_id)
        return {"sleeve_id": sleeve_id, "n": len(rows), "rows": rows}
    except Exception as exc:
        logger.exception("transfers_by_sleeve failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/transfers/recent", tags=["research"])
def transfers_recent(limit: int = 50) -> dict:
    """All recent cross-asset transfer proposals across deployed sleeves."""
    try:
        from engine.research.cross_domain_transfer import list_all_recent
        rows = list_all_recent(limit=limit)
        return {"n": len(rows), "rows": rows}
    except Exception as exc:
        logger.exception("transfers_recent failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/transfers/by_sleeve/{sleeve_id}/generate", tags=["research"])
def transfers_generate(sleeve_id: str) -> dict:
    """Generate fresh cross-asset transfer proposals for a deployed
    sleeve (~$0.30, ~15s). UI button or monthly cron may call."""
    try:
        from engine.research.cross_domain_transfer import propose_transfers
        import dataclasses as _dc
        proposals = propose_transfers(sleeve_id)
        if proposals is None:
            raise HTTPException(status_code=502,
                detail="cross-domain LLM call failed or did not emit tool")
        return {
            "sleeve_id": sleeve_id,
            "n_proposed": len(proposals),
            "proposals": [_dc.asdict(p) for p in proposals],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("transfers_generate failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/pre_mortem/{hypothesis_id}", tags=["research"])
def pre_mortem_get(hypothesis_id: str) -> dict:
    """Return all pre-mortem reports for a hypothesis (newest first)
    + the latest one as `latest` convenience. Empty list if no
    pre-mortem has been generated yet for this hyp."""
    try:
        from engine.research.pre_mortem import list_for_hypothesis
        rows = list_for_hypothesis(hypothesis_id)
        return {
            "hypothesis_id": hypothesis_id,
            "n": len(rows),
            "rows": rows,
            "latest": rows[0] if rows else None,
        }
    except Exception as exc:
        logger.exception("pre_mortem_get failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/pre_mortem/{hypothesis_id}/generate", tags=["research"])
def pre_mortem_generate(hypothesis_id: str) -> dict:
    """Run a fresh pre-mortem for this hypothesis (Sonnet ~$0.05).
    Idempotent in the sense that history is preserved — a new row is
    appended to pre_mortems.jsonl on each call. UI button or cron may
    invoke. Returns the new report."""
    try:
        from engine.research.pre_mortem import generate_pre_mortem
        import dataclasses as _dc
        report = generate_pre_mortem(hypothesis_id)
        if report is None:
            raise HTTPException(status_code=502,
                detail="pre-mortem LLM call failed or did not emit tool")
        row = _dc.asdict(report)
        row["failure_modes"] = [_dc.asdict(fm) for fm in report.failure_modes]
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("pre_mortem_generate failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/decay_retest/queue", tags=["research"])
def decay_retest_queue(include_completed: bool = False) -> dict:
    """Decay re-test queue (Phase 9 reactive subscriber, 2026-06-14).
    Pending = enqueued but not yet processed. Set include_completed
    to also see status transitions for audit."""
    try:
        from engine.research.decay_retest import list_queue
        rows = list_queue(include_completed=include_completed)
        return {"n": len(rows), "rows": rows}
    except Exception as exc:
        logger.exception("decay_retest_queue failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/decay_retest/results", tags=["research"])
def decay_retest_results(limit: int = 50) -> dict:
    """Latest decay-retest verdicts. Each row carries the CONFIRMED_DECAY
    / NOISE_INDISTINGUISHABLE / INSUFFICIENT_DATA verdict, the Chow p-
    value, the bootstrap CI on recent rolling Sharpe, and the originating
    decay_alert event_id when triggered by the cron subscriber."""
    try:
        from engine.research.decay_retest import list_results
        rows = list_results(limit=limit)
        return {"n": len(rows), "rows": rows}
    except Exception as exc:
        logger.exception("decay_retest_results failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/decay_retest/enqueue", tags=["research"])
def decay_retest_enqueue(sleeve_id: str, parent_event_id: str | None = None) -> dict:
    """Manually enqueue a sleeve for re-test. Dedup window 24h."""
    try:
        from engine.research.decay_retest import enqueue_retest
        rid = enqueue_retest(sleeve_id, triggered_by="manual",
                              parent_event_id=parent_event_id)
        return {"status": "ok", "retest_id": rid, "sleeve_id": sleeve_id}
    except Exception as exc:
        logger.exception("decay_retest_enqueue failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/research/decay_retest/run", tags=["research"])
def decay_retest_run(limit: int = 5) -> dict:
    """Drain up to `limit` pending queue rows and run the retest. The
    daily cron wraps this; the manual endpoint is for UI 'process now'
    buttons. Returns the produced result rows."""
    try:
        from engine.research.decay_retest import process_queue
        import dataclasses as _dc
        results = process_queue(limit=limit)
        return {
            "n_processed": len(results),
            "results": [_dc.asdict(r) for r in results],
        }
    except Exception as exc:
        logger.exception("decay_retest_run failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research/graveyard_collisions/{hypothesis_id}", tags=["research"])
def graveyard_collisions(hypothesis_id: str, top_k: int = 5,
                          min_score: float = 0.20) -> dict:
    """Detect "this hyp looks like one we've already killed" collisions.

    Added 2026-06-14 from the [[project-deferred-multi-agent]] reactive-
    subscribers queue. Without this, the user has to manually search
    /research/lessons to verify a fresh hypothesis isn't a re-proposal
    of something already RED'd. McLean-Pontiff 2016 confirms 32-58%
    Sharpe drop on re-mining anomalies; cheaper to catch at intake.

    Scoring (deliberately simple, no embedding dep — sentence-transformers
    is heavy at import time and the precision/recall floor here is fine
    for a "did we already kill this?" inbox check):
       0.50 weight: family exact match
       0.50 weight: claim-text token Jaccard

    Compared against:
       - RED autopsies (autopsies.jsonl filter actual_verdict=RED)
       - RED factor_verdict_filed events (research store)

    Returns top-K matches with score breakdown. Caller surfaces inline
    on /research/hypothesis + /research/forward + /approvals decision UX."""
    import re as _re
    _TOKEN = _re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

    def _tokens(s: str) -> set:
        if not s:
            return set()
        return {t.lower() for t in _TOKEN.findall(s)} - {
            "the","and","for","with","this","that","from","into","over",
            "than","more","less","when","where","what","which","each","very",
            "factor","portfolio","returns","return","monthly","daily","weekly",
            "stocks","stock","sample","analysis","using","tested","strategy",
            "strategies","based","using","results","result","period","period",
        }

    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # 1) Load the source hypothesis
    hyp_path = ROOT / "data" / "research_store" / "hypotheses.jsonl"
    src_hyp = None
    if hyp_path.is_file():
        for ln in hyp_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("hypothesis_id") == hypothesis_id:
                src_hyp = r
                break
    if src_hyp is None:
        raise HTTPException(status_code=404,
            detail=f"hypothesis {hypothesis_id} not found")

    src_family = (src_hyp.get("mechanism_family") or "").lower()
    src_claim = src_hyp.get("claim") or {}
    src_one_line = ""
    if isinstance(src_claim, dict):
        src_one_line = src_claim.get("one_line") or ""
    elif src_claim is not None:
        src_one_line = str(src_claim)
    src_text = (
        f"{src_one_line} {src_hyp.get('mechanism_subtype') or ''} "
        + " ".join((src_hyp.get("required_data") or [])[:5])
    )
    src_tokens = _tokens(src_text)

    candidates: list[dict] = []

    # 2) RED autopsies
    autopsy_path = ROOT / "data" / "research" / "autopsies.jsonl"
    if autopsy_path.is_file():
        # Build hyp-id → claim lookup from hypotheses.jsonl for richer text
        hyp_lookup: dict[str, dict] = {}
        if hyp_path.is_file():
            for ln in hyp_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln: continue
                try:
                    h = json.loads(ln)
                    hyp_lookup[h.get("hypothesis_id", "")] = h
                except Exception:
                    continue
        for ln in autopsy_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("superseded_by") or r.get("actual_verdict") != "RED":
                continue
            cand_hyp_id = r.get("hypothesis_id") or ""
            if cand_hyp_id == hypothesis_id:
                continue
            cand_family = (r.get("strategy_family") or "").lower()
            cand_hyp = hyp_lookup.get(cand_hyp_id, {})
            cand_claim_raw = cand_hyp.get("claim") or {}
            cand_one_line = ""
            if isinstance(cand_claim_raw, dict):
                cand_one_line = cand_claim_raw.get("one_line") or ""
            elif cand_claim_raw is not None:
                cand_one_line = str(cand_claim_raw)
            cand_text = f"{cand_one_line} {cand_hyp.get('mechanism_subtype') or ''}"
            # "OTHER" is the catch-all bucket; matching on it would flag
            # every cross-asset / unclassified RED as a collision. Skip.
            fam_score = 0.0
            if (src_family and cand_family
                and src_family != "other" and cand_family != "other"
                and (src_family == cand_family
                     or src_family in cand_family
                     or cand_family in src_family)):
                fam_score = 0.5
            jac = _jaccard(src_tokens, _tokens(cand_text))
            total = fam_score + jac * 0.5
            if total < min_score:
                continue
            candidates.append({
                "kind":            "autopsy_red",
                "ts":              r.get("ts"),
                "hypothesis_id":   cand_hyp_id,
                "family":          r.get("strategy_family"),
                "claim_excerpt":   cand_text[:200],
                "score":           round(total, 3),
                "family_match":    bool(fam_score),
                "jaccard":         round(jac, 3),
            })

    # 3) RED factor_verdict_filed events
    try:
        from engine.research_store import store
        for e in store.filter_events(event_type="factor_verdict_filed", limit=500):
            if str(e.verdict).split(".")[-1] != "RED":
                continue
            ev_md = e.metrics or {}
            cand_hyp_id = ev_md.get("hypothesis_id") or ""
            if cand_hyp_id == hypothesis_id:
                continue
            cand_family = (e.family or "").lower()
            cand_text = (e.summary or "") + " " + (e.subject_id or "")
            # "OTHER" is the catch-all bucket; matching on it would flag
            # every cross-asset / unclassified RED as a collision. Skip.
            fam_score = 0.0
            if (src_family and cand_family
                and src_family != "other" and cand_family != "other"
                and (src_family == cand_family
                     or src_family in cand_family
                     or cand_family in src_family)):
                fam_score = 0.5
            jac = _jaccard(src_tokens, _tokens(cand_text))
            total = fam_score + jac * 0.5
            if total < min_score:
                continue
            candidates.append({
                "kind":            "verdict_red",
                "ts":              e.ts,
                "hypothesis_id":   cand_hyp_id or None,
                "event_id":        e.event_id,
                "family":          e.family,
                "claim_excerpt":   cand_text[:200],
                "score":           round(total, 3),
                "family_match":    bool(fam_score),
                "jaccard":         round(jac, 3),
            })
    except Exception:
        pass

    candidates.sort(key=lambda c: -c["score"])
    return {
        "hypothesis_id":   hypothesis_id,
        "src_family":      src_hyp.get("mechanism_family"),
        "n_total_red":     len(candidates),
        "top_collisions":  candidates[:top_k],
        "score_doctrine":  "0.5*family_match + 0.5*claim_token_jaccard "
                           "(min_score filter applied)",
    }


@app.get("/api/research/family/{family_id}", tags=["research"])
def family_detail(family_id: str, limit: int = 50) -> dict:
    """Family-level object view (Phase 6, 2026-06-14). Aggregates the
    belief layer + recent verdicts + autopsies + pending hypotheses
    for one family into the data drive a single first-class page on
    /research/family?id=<family_id>. Family identifier here is the
    autopsy `strategy_family` value (e.g. VRP, EVENT_DRIFT, CARRY_FX).
    See [[feedback-strategy-family-vs-claim-family-2026-06-12]] for the
    naming axis."""
    out: dict = {
        "family":               family_id,
        "belief":               None,
        "recent_verdicts":      [],
        "autopsies":            [],
        "pending_hypotheses":   [],
        "n_verdicts_total":     0,
        "n_pending_total":      0,
        "bailey_ldp_n_trials":  None,    # Phase 7 (2026-06-14)
    }

    # Bailey-Lopez de Prado DSR denominator. Per the Aug-2014 paper §3,
    # N = number of INDEPENDENT model configurations tried in the SAME
    # family. Surfacing here lets the principal see "VRP family has N=8,
    # one more trial → DSR threshold creeps up by k" before approving a
    # new VRP variant. See engine/research/family_trial_counter.py for
    # the doctrine + per-family overrides.
    try:
        from engine.research.family_trial_counter import (
            count_trials_in_family, count_library_entries_in_family,
            FAMILY_BUFFER_OVERRIDES, DEFAULT_EXPLORATION_BUFFER,
        )
        n_trials = count_trials_in_family(family_id)
        lib_count = count_library_entries_in_family(family_id)
        buffer = FAMILY_BUFFER_OVERRIDES.get(family_id.lower(), DEFAULT_EXPLORATION_BUFFER)
        out["bailey_ldp_n_trials"] = {
            "n_trials":            n_trials,
            "library_entries":     lib_count,
            "exploration_buffer":  buffer,
            "doctrine_ref":        "Bailey-Lopez de Prado 2014 §3 — DSR denominator",
        }
    except Exception:
        pass

    # 1) Belief summary (uses the same autopsy ledger pipeline)
    try:
        from engine.research.belief_synthesis_context import build_belief_summary
        for b in build_belief_summary(min_obs_per_family=1):
            if b.family == family_id:
                out["belief"] = {
                    "family":          b.family,
                    "n_obs":           b.n_obs,
                    "n_green":         b.n_green,
                    "n_marginal":      b.n_marginal,
                    "n_red":           b.n_red,
                    "direction_hint":  b.direction_hint,
                }
                break
    except Exception:
        pass

    # 2) Recent verdicts via the typed research store. Match `family`
    #    on the event AND on metrics.strategy_family (the spec-derived
    #    label used by the autopsy denominator) — events tagged with
    #    EITHER form should appear here.
    try:
        from engine.research_store import store
        all_v = store.filter_events(event_type="factor_verdict_filed", limit=500)
        matched = []
        for e in all_v:
            ev_fam = e.family or ""
            md_fam = (e.metrics or {}).get("strategy_family") or ""
            if family_id in (ev_fam, md_fam):
                matched.append({
                    "event_id":      e.event_id,
                    "ts":            e.ts,
                    "verdict":       str(e.verdict).split(".")[-1],
                    "subject_id":    e.subject_id,
                    "summary":       (e.summary or "")[:200],
                    "hypothesis_id": (e.metrics or {}).get("hypothesis_id"),
                })
        out["n_verdicts_total"] = len(matched)
        out["recent_verdicts"] = matched[:limit]
    except Exception as exc:
        logger.exception("family_detail verdicts read failed")
        out["error_verdicts"] = str(exc)[:200]

    # 3) Autopsies — direct read filtered by strategy_family.
    autopsy_path = ROOT / "data" / "research" / "autopsies.jsonl"
    if autopsy_path.is_file():
        try:
            for ln in autopsy_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("superseded_by"):
                    continue
                if r.get("strategy_family") != family_id:
                    continue
                out["autopsies"].append({
                    "autopsy_id":         r.get("autopsy_id"),
                    "ts":                 r.get("ts"),
                    "hypothesis_id":      r.get("hypothesis_id"),
                    "actual_verdict":     r.get("actual_verdict"),
                    "surprise_direction": r.get("surprise_direction"),
                    "surprise_magnitude": r.get("surprise_magnitude"),
                    "brier_component":    r.get("brier_component"),
                })
        except Exception:
            pass

    # 4) Pending hypotheses (unresolved + family match — both forms).
    hyp_path = ROOT / "data" / "research_store" / "hypotheses.jsonl"
    if hyp_path.is_file():
        try:
            for ln in hyp_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                mech_fam = r.get("mechanism_family") or ""
                if family_id != mech_fam:
                    # Best-effort substring match too (CARRY ↔ CARRY_FX)
                    if family_id.lower() not in mech_fam.lower() and \
                       mech_fam.lower() not in family_id.lower():
                        continue
                rs = r.get("review_state") or ""
                if rs in ("approved", "rejected"):
                    continue
                claim = r.get("claim") or {}
                out["pending_hypotheses"].append({
                    "hypothesis_id":   r.get("hypothesis_id"),
                    "source_paper_id": r.get("source_paper_id"),
                    "mechanism_family": mech_fam,
                    "review_state":    rs,
                    "claim_one_line":  (claim.get("one_line") if isinstance(claim, dict) else str(claim))[:200],
                    "created_ts":      r.get("created_ts"),
                })
        except Exception:
            pass
        out["n_pending_total"] = len(out["pending_hypotheses"])
        out["pending_hypotheses"] = out["pending_hypotheses"][:limit]

    return out


@app.get("/api/research/hypothesis/{hypothesis_id}", tags=["research"])
def hypothesis_detail(hypothesis_id: str) -> dict:
    """Single hypothesis object view (Phase 6, 2026-06-14). Returns the
    hypothesis row + resolution status + linked verdicts + safety-rail
    context, designed to drive /research/hypothesis?id=<hyp_id>. This
    is the natural target when a user clicks a hyp from a verdict
    drill, a forward-queue row, or a family detail page."""
    out: dict = {
        "hypothesis_id":   hypothesis_id,
        "hypothesis":      None,
        "source_paper":    None,
        "resolution":      None,
        "verdicts":        [],
        "safety_rails":    None,
    }

    # 1) Hypothesis row
    hyp_path = ROOT / "data" / "research_store" / "hypotheses.jsonl"
    if hyp_path.is_file():
        for ln in hyp_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("hypothesis_id") == hypothesis_id:
                out["hypothesis"] = r
                break

    if out["hypothesis"] is None:
        raise HTTPException(status_code=404,
            detail=f"hypothesis {hypothesis_id} not found")

    # 2) Resolution (from strengthener resolutions ledger if present)
    try:
        from engine.agents.strengthener.approval_view import (
            find_verdict, _load_resolutions, _DEFAULT_RESOLUTIONS_PATH,
        )
        res_map = _load_resolutions(_DEFAULT_RESOLUTIONS_PATH)
        res = res_map.get(hypothesis_id)
        if res is not None:
            out["resolution"] = {
                "decision":    res.decision,
                "rationale":   res.rationale,
                "resolved_ts": res.resolved_ts,
                "resolved_by": res.resolved_by,
            }
        v = find_verdict(hypothesis_id)
        if v and "verdict_type" in v:
            out["b_verdict"] = {
                "verdict_type":     v.get("verdict_type"),
                "confidence":       v.get("confidence"),
                "one_line_summary": v.get("one_line_summary"),
                "reasoning":        v.get("reasoning"),
            }
    except Exception:
        pass

    # 3) Verdicts (factor_verdict_filed events linked to this hyp)
    try:
        from engine.research_store import store
        for e in store.filter_events(event_type="factor_verdict_filed", limit=500):
            if (e.metrics or {}).get("hypothesis_id") == hypothesis_id:
                out["verdicts"].append({
                    "event_id":   e.event_id,
                    "ts":         e.ts,
                    "verdict":    str(e.verdict).split(".")[-1],
                    "subject_id": e.subject_id,
                    "summary":    (e.summary or "")[:200],
                    "family":     e.family,
                })
    except Exception:
        pass

    # 4) Safety rails (re-use Phase 5 aggregator)
    try:
        out["safety_rails"] = safety_rails_for_hypothesis(hypothesis_id)
    except Exception:
        pass

    return out


@app.get("/api/research/safety_rails_for_hypothesis/{hypothesis_id}", tags=["research"])
def safety_rails_for_hypothesis(hypothesis_id: str) -> dict:
    """Aggregated safety-rail summary for one hypothesis, designed for
    inline use on /approvals row decision UX.

    Returns the post-GREEN rigor rows linked to this hypothesis (by
    `hypothesis_id` direct match), any external audit rows linked via
    the hypothesis's verdict_event_ids, and the belief-layer family
    summary (so the principal sees "VRP family is EXPLORE 8G/0R, but
    THIS verdict has a SHORT_FEE_KILLS flag" inline before approving).

    Added 2026-06-14 for Phase 5: capital decision points must show
    backend gate state inline, not require navigation to a separate
    page. Reading from disk on every call (small ledgers; cache the
    result client-side if the hot path proves expensive)."""
    import datetime as _dt
    out = {
        "hypothesis_id":   hypothesis_id,
        "rigor":           [],
        "audits":          [],
        "belief_family":   None,
        "verdict_event_ids": [],
        "n_critical":      0,
        "n_concern":       0,
        "n_flagged":       0,
    }

    # 1) Rigor rows directly keyed by hypothesis_id
    rigor_path = ROOT / "data" / "research" / "post_green_rigor.jsonl"
    if rigor_path.is_file():
        for ln in rigor_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("hypothesis_id") != hypothesis_id:
                continue
            flags = r.get("flags") or []
            out["rigor"].append({
                "rigor_id":         r.get("rigor_id"),
                "ts":               r.get("ts"),
                "verdict_event_id": r.get("verdict_event_id"),
                "family":           r.get("family"),
                "template_name":    r.get("template_name"),
                "oos_status":       (r.get("post_pub_oos") or {}).get("status"),
                "spanning_status":  (r.get("spanning") or {}).get("status"),
                "borrow_status":    (r.get("borrow_cost") or {}).get("status"),
                "flags":            flags,
            })
            out["n_flagged"] += len(flags)
            vid = r.get("verdict_event_id")
            if vid and vid not in out["verdict_event_ids"]:
                out["verdict_event_ids"].append(vid)

    # 2) Walk research store for any verdict_events tagged with this
    #    hypothesis_id (some rigor entries may not carry hypothesis_id
    #    on legacy rows; pull verdicts via the typed API to catch those).
    try:
        from engine.research_store import store
        for e in store.filter_events(event_type="factor_verdict_filed", limit=200):
            if (e.metrics or {}).get("hypothesis_id") == hypothesis_id:
                if e.event_id not in out["verdict_event_ids"]:
                    out["verdict_event_ids"].append(e.event_id)
    except Exception:
        pass

    # 3) Audit rows linked via subject_ref ∈ verdict_event_ids
    audit_path = ROOT / "data" / "research" / "external_audits.jsonl"
    if audit_path.is_file() and out["verdict_event_ids"]:
        evset = set(out["verdict_event_ids"])
        for ln in audit_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("subject_ref") not in evset:
                continue
            sev = r.get("severity") or "skipped"
            out["audits"].append({
                "audit_id":           r.get("audit_id"),
                "ts":                 r.get("ts"),
                "provider":           r.get("provider"),
                "severity":           sev,
                "flagged_categories": r.get("flagged_categories") or [],
                "subject_ref":        r.get("subject_ref"),
            })
            if sev == "critical":
                out["n_critical"] += 1
            elif sev == "concern":
                out["n_concern"] += 1

    # 4) Belief family. Hypothesis carries `mechanism_family` (paper-tagged,
    #    e.g. CARRY) while the autopsy ledger uses `strategy_family`
    #    (spec-derived, e.g. CARRY_FX). Per [[feedback-strategy-family
    #    -vs-claim-family-2026-06-12]] these are intentionally distinct.
    #    Match by direct equality first, then case-insensitive substring,
    #    then a small acronym synonym map for the common cases where the
    #    autopsy uses a 3-letter abbrev (VRP) but the hyp tag uses the
    #    full name (VOL_RISK_PREMIUM).
    _FAMILY_SYNONYMS = {
        "VOL_RISK_PREMIUM":  "VRP",
        "EARNINGS_DRIFT":    "EVENT_DRIFT",
        "POST_EARNINGS_DRIFT": "EVENT_DRIFT",
        "CROSS_SECTION":     "CROSS_SEC_UNKNOWN",
    }
    try:
        from engine.agents.strengthener.approval_view import find_hypothesis_family
        from engine.research.belief_synthesis_context import build_belief_summary
        family = find_hypothesis_family(hypothesis_id) or None
        if family:
            beliefs = build_belief_summary(min_obs_per_family=1)
            fam_lower = family.lower()
            matched = next((b for b in beliefs if b.family == family), None)
            if matched is None:
                syn = _FAMILY_SYNONYMS.get(family.upper())
                if syn:
                    matched = next((b for b in beliefs if b.family == syn), None)
            if matched is None:
                matched = next(
                    (b for b in beliefs
                     if fam_lower in b.family.lower() or b.family.lower() in fam_lower),
                    None,
                )
            if matched is not None:
                out["belief_family"] = {
                    "family":         matched.family,
                    "hyp_family":     family,
                    "match_kind":     "exact" if matched.family == family else "substring",
                    "n_obs":          matched.n_obs,
                    "n_green":        matched.n_green,
                    "n_marginal":     matched.n_marginal,
                    "n_red":          matched.n_red,
                    "direction_hint": matched.direction_hint,
                }
            else:
                out["belief_family"] = {
                    "family":         None,
                    "hyp_family":     family,
                    "match_kind":     "no_belief_data_yet",
                    "n_obs":          0,
                    "n_green":        0,
                    "n_marginal":     0,
                    "n_red":          0,
                    "direction_hint": "thin (no autopsy data for this family yet)",
                }
    except Exception:
        pass

    return out


@app.get("/api/research_store/events", tags=["research"])
def research_store_events(
    event_type: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    verdict: str | None = None,
    family: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> dict:
    """Typed query over the research event store (M1 2026-06-02).

    Replaces ad-hoc scraping of capability_evidence / memory / factory_ledger /
    gate_runs. All consumers (UI surfaces, audit tools) should read here.
    Producers (Claude / cron) emit via engine.research_store.emit.*.

    Filters all conjunctive (AND). `since` compares ts string lexically — safe
    because ts is ISO-8601 UTC. Newest first.
    """
    try:
        from engine.research_store import store
        limit = max(1, min(int(limit), 1000))
        events = store.filter_events(
            event_type=event_type, subject_type=subject_type,
            subject_id=subject_id, verdict=verdict,
            family=family, since=since, limit=limit,
        )
        return {
            "n":      len(events),
            "events": [e.to_dict() for e in events],
        }
    except Exception as exc:
        logger.exception("research_store_events failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research_store/subjects", tags=["research"])
def research_store_subjects(family: str | None = None) -> dict:
    """List registered subjects (the controlled vocabulary backing events)."""
    try:
        from engine.research_store import registry
        subs = registry.list_subjects(family=family)
        return {
            "n":        len(subs),
            "subjects": [
                {
                    "subject_id":   s.subject_id,
                    "subject_type": s.subject_type,
                    "family":       s.family,
                    "description":  s.description,
                    "canonical_paper_id": s.canonical_paper_id,
                    "created_ts":   s.created_ts,
                    "created_by":   s.created_by,
                }
                for s in subs
            ],
        }
    except Exception as exc:
        logger.exception("research_store_subjects failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research_store/lineage/{event_id}", tags=["research"])
def research_store_lineage(event_id: str) -> dict:
    """Walk parent chain of a given event (DAG). Returns the event +
    ordered ancestors. Used by /lab/library to show evidence-chain
    when rendering a verdict."""
    try:
        from engine.research_store import store
        seen: set[str] = set()
        chain: list[dict] = []
        cursor = store.by_event_id(event_id)
        if cursor is None:
            raise HTTPException(status_code=404, detail=f"event {event_id} not found")
        while cursor is not None and cursor.event_id not in seen:
            seen.add(cursor.event_id)
            chain.append(cursor.to_dict())
            if not cursor.parent_event_ids:
                break
            # Only follow first parent (most lineages are linear; DAG support
            # later if we genuinely need many-to-one)
            cursor = store.by_event_id(cursor.parent_event_ids[0])
        return {"n": len(chain), "chain": chain}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("research_store_lineage failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


def _decay_ack_chain(events) -> dict[str, dict]:
    """G.5 + H (2026-06-09): walk ack/unack chains for a list of
    decay_alert events (newest first). Returns dict mapping ORIGINAL
    event_id → its latest ack/unack state.

    An "original" alert is an event with `decay_watch` tag but NOT
    `acknowledged` or `unacknowledged` tags. Acks/unacks reference
    their predecessor via parent_event_ids; walking the chain back
    finds the original.

    The LATEST event in each chain (newest-first ordering wins)
    determines current state. If latest is `acknowledged`, the
    original is currently acked; if latest is `unacknowledged`,
    it's been re-opened.

    Returns:
      { original_event_id: {
          is_acknowledged: bool,
          latest_event_id: str,
          latest_ts:       str,
          latest_action:   str | None,
          latest_reason:   str | None,
          latest_actor:    str | None,
          history: [ {event_id, ts, kind, action, reason, actor}, ... ],
        } }
    """
    by_id = {e.event_id: e for e in events}

    def _is_admin_event(e) -> bool:
        tags = e.tags or ()
        return "acknowledged" in tags or "unacknowledged" in tags

    def _find_root(e):
        """Walk parent chain until we find an event with no
        acknowledged/unacknowledged tag."""
        seen: set[str] = set()
        cur = e
        while cur is not None and cur.event_id not in seen:
            seen.add(cur.event_id)
            if not _is_admin_event(cur):
                return cur.event_id
            parents = cur.parent_event_ids or ()
            if not parents:
                return None
            cur = by_id.get(parents[0])
        return None

    # H fix: build chain by FORWARD WALK from each original via
    # parent_event_ids edges, not by ts sort. Two events emitted in
    # the same second (ack + immediate unack in tests) can't be
    # tie-broken by ts alone; chain topology is the canonical order.
    successor_of: dict[str, list] = {}
    for e in events:
        if not _is_admin_event(e):
            continue
        for parent_id in (e.parent_event_ids or ()):
            successor_of.setdefault(parent_id, []).append(e)

    chain_per_root: dict[str, list] = {}
    for e in events:
        if _is_admin_event(e):
            continue
        # walk forward from this original
        ordered: list = []
        seen_in_chain: set[str] = {e.event_id}
        # BFS: each event has at most one successor in practice
        # (you ack, then either unack-once or re-ack); take whichever
        # exists.
        cur = e
        while True:
            children = successor_of.get(cur.event_id, [])
            if not children:
                break
            # If multiple acks pointed at same parent (shouldn't happen
            # normally), take the newest. Tie-break stable.
            children = sorted(children, key=lambda x: x.ts)
            nxt = children[-1]
            if nxt.event_id in seen_in_chain:
                break
            ordered.append(nxt)
            seen_in_chain.add(nxt.event_id)
            cur = nxt
        if ordered:
            # newest-last from the walk; expose newest-first to match
            # the existing API shape
            chain_per_root[e.event_id] = list(reversed(ordered))

    state: dict[str, dict] = {}
    for root, chain in chain_per_root.items():
        # chain is newest-first after the reverse
        latest = chain[0]
        is_acked = "acknowledged" in (latest.tags or ())
        m = latest.metrics or {}
        state[root] = {
            "is_acknowledged": is_acked,
            "latest_event_id": latest.event_id,
            "latest_ts":       latest.ts,
            "latest_action":   m.get("action"),
            "latest_reason":   m.get("reason"),
            "latest_actor":    m.get("actor"),
            "history": [
                {
                    "event_id": ev.event_id,
                    "ts":       ev.ts,
                    "kind":     ("acknowledged"
                                  if "acknowledged" in (ev.tags or ())
                                  else "unacknowledged"),
                    "action":   (ev.metrics or {}).get("action"),
                    "reason":   (ev.metrics or {}).get("reason"),
                    "actor":    (ev.metrics or {}).get("actor"),
                }
                for ev in chain
            ],
        }
    return state


class _DecayAlertAckRequest(BaseModel):
    """G.5 (2026-06-09): acknowledge a canonical decay_alert event.

    Per CLAUDE.md doctrine "events are immutable; to correct, emit a
    new event with parent_event_ids pointing to the prior". The ack
    appends a NEW decay_alert event with:
      - same subject_id
      - verdict = NEUTRAL  (downgrade — alert reviewed)
      - tags + ("acknowledged",)
      - parent_event_ids = (original_event_id,)
      - metrics.action, metrics.reason, metrics.actor, metrics.ack_ts

    This keeps the audit trail intact and lets the UI detect
    acknowledged-vs-open via parent-event walk.
    """
    action:    str    # one of ACK_ACTIONS below
    reason:    str    # required, min 10 chars — institutional standard
    actor:     str | None = None    # defaults to "ui" when omitted


_ACK_ACTIONS = frozenset({
    "reviewed_no_action",     # looked at, decided allocation OK as-is
    "reduced_allocation",     # separately reduced this sleeve's allocation
    "scheduled_review",       # will revisit at next portfolio review
    "false_positive",         # disagree with the trigger conditions
})
_ACK_REASON_MIN_CHARS = 10    # forces real engagement, not click-through


@app.post(
    "/api/research_store/decay_alert/{event_id}/acknowledge",
    tags=["research"],
)
def research_store_decay_alert_acknowledge(
    event_id: str, body: _DecayAlertAckRequest,
) -> dict:
    """Acknowledge a decay_alert by emitting a follow-up canonical
    event (event_id chain via parent_event_ids).

    Per [[feedback-research-auto-capital-human-2026-06-05]] the ack
    payload records what the PRINCIPAL decided to do (or not do) —
    capital allocation decisions remain HUMAN. The ack is an
    observation of that decision, NOT the decision itself.
    """
    if body.action not in _ACK_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=(f"action must be one of {sorted(_ACK_ACTIONS)}; "
                       f"got {body.action!r}"),
        )
    reason = (body.reason or "").strip()
    if len(reason) < _ACK_REASON_MIN_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(f"reason must be >= {_ACK_REASON_MIN_CHARS} chars; "
                       "institutional ack standard"),
        )
    try:
        from engine.research_store import store, emit
        original = store.by_event_id(event_id)
        if original is None:
            raise HTTPException(status_code=404,
                                   detail=f"event {event_id} not found")
        if original.event_type != "decay_alert":
            raise HTTPException(
                status_code=422,
                detail=(f"event {event_id} is {original.event_type}, "
                           "not decay_alert"),
            )

        # H fix (2026-06-09): chain off the LATEST event in the
        # existing ack/unack chain so re-acks form a proper linear
        # history. parent_event_ids contracts: original = original
        # alert, ack = latest event in chain (original OR prior
        # unack OR prior ack).
        events_subject = store.filter_events(
            event_type="decay_alert",
            subject_id=original.subject_id, limit=500,
        )
        canonical = [e for e in events_subject
                       if "decay_watch" in (e.tags or ())]
        chain = _decay_ack_chain(canonical)
        cur_state = chain.get(event_id)
        if cur_state is None:
            parent_for_new = event_id   # no prior chain
        else:
            parent_for_new = cur_state["latest_event_id"]

        import datetime as _dt
        ack_ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        actor  = (body.actor or "ui")[:60]
        ack_metrics = {
            "action":             body.action,
            "reason":             reason[:1000],
            "actor":              actor,
            "ack_ts":             ack_ts,
            "original_event_id":  event_id,
            # propagate the original's severity for UI display
            "original_severity":  (original.metrics or {}).get("severity"),
            "original_triggers":  (original.metrics or {}).get("triggers_hit"),
        }
        new_event_id = emit.decay_alert(
            subject_id       = original.subject_id,
            verdict          = "NEUTRAL",   # downgrade — alert reviewed
            metrics          = ack_metrics,
            artifacts        = {},
            summary          = (
                f"Acknowledged ({body.action}) by {actor}: "
                + reason[:200]
            ),
            parent_event_ids = (parent_for_new,),
            tags             = ("decay_watch", "acknowledged"),
            actor            = f"ui:{actor}",
        )
        return {
            "ok":             True,
            "ack_event_id":   new_event_id,
            "original_event_id": event_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("decay_alert acknowledge failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


class _DecayAlertUnackRequest(BaseModel):
    """H (2026-06-09): re-open an acknowledged decay_alert. Emits a
    follow-up event with `unacknowledged` tag chained back to the
    latest ack in the chain. The original alert returns to
    "actionable" state and resurfaces in the inbox.
    """
    reason:    str    # required, ≥10 chars
    actor:     str | None = None


@app.post(
    "/api/research_store/decay_alert/{event_id}/unacknowledge",
    tags=["research"],
)
def research_store_decay_alert_unacknowledge(
    event_id: str, body: _DecayAlertUnackRequest,
) -> dict:
    """Reverse an acknowledgement. event_id is the ORIGINAL alert
    event_id (not the ack event); the endpoint locates the latest
    ack in its chain and emits an `unacknowledged` event pointing
    back to it.

    Errors:
      404 — event_id not found
      422 — event is not a decay_alert / not currently acked /
            reason too short
    """
    reason = (body.reason or "").strip()
    if len(reason) < _ACK_REASON_MIN_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(f"reason must be >= {_ACK_REASON_MIN_CHARS} chars; "
                       "institutional unack standard"),
        )
    try:
        from engine.research_store import store, emit
        original = store.by_event_id(event_id)
        if original is None:
            raise HTTPException(status_code=404,
                                   detail=f"event {event_id} not found")
        if original.event_type != "decay_alert":
            raise HTTPException(
                status_code=422,
                detail=(f"event {event_id} is {original.event_type}, "
                           "not decay_alert"),
            )
        # Compute current ack state. event_id must be currently acked
        # to be unack-able.
        events = store.filter_events(
            event_type="decay_alert",
            subject_id=original.subject_id, limit=500,
        )
        canonical = [e for e in events if "decay_watch" in (e.tags or ())]
        state = _decay_ack_chain(canonical)
        cur = state.get(event_id)
        if cur is None or not cur.get("is_acknowledged"):
            raise HTTPException(
                status_code=422,
                detail=("event is not currently acknowledged; "
                           "nothing to unack"),
            )
        latest_ack_event_id = cur["latest_event_id"]

        import datetime as _dt
        unack_ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        actor    = (body.actor or "ui")[:60]
        unack_metrics = {
            "action":             "unack",
            "reason":             reason[:1000],
            "actor":              actor,
            "ack_ts":             unack_ts,
            "original_event_id":  event_id,
            "ack_event_id":       latest_ack_event_id,
        }
        new_event_id = emit.decay_alert(
            subject_id       = original.subject_id,
            verdict          = "NEUTRAL",   # admin event, not a re-trigger
            metrics          = unack_metrics,
            artifacts        = {},
            summary          = (
                f"Re-opened (unacknowledged) by {actor}: " + reason[:200]
            ),
            parent_event_ids = (latest_ack_event_id,),
            tags             = ("decay_watch", "unacknowledged"),
            actor            = f"ui:{actor}",
        )
        return {
            "ok":               True,
            "unack_event_id":   new_event_id,
            "original_event_id": event_id,
            "reverted_ack_event_id": latest_ack_event_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("decay_alert unacknowledge failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research_store/decay_audit/{subject_id}", tags=["research"])
def research_store_decay_audit(subject_id: str, limit: int = 50) -> dict:
    """G.4 (2026-06-09): drill-down endpoint for the inbox decay_watch
    + spec_robustness + anchor_orthogonality rows. Returns canonical
    events for one subject so /lab/decay/detail can render the full
    Tier C audit trail without the UI scraping events.jsonl.

    Returns:
      decay_alerts:   list of decay_alert events (newest first), filtered
                      to canonical Tier C trigger (`decay_watch` tag)
      factor_verdicts: list of factor_verdict_filed events whose metrics
                      carry subsample_stability / specification_robustness
                      / anchor_orthogonality (lens outputs)

    NOT included: legacy SLM decay_alert rows (no `decay_watch` tag) —
    those are surfaced by the existing /api/research/decay/sleeve
    timeline endpoint.
    """
    try:
        from engine.research_store import store
        limit = max(1, min(int(limit), 500))

        # All decay_alert events for this subject (newest first), then
        # filter to canonical Tier C (decay_watch tag)
        decay_raw = store.filter_events(
            event_type="decay_alert", subject_id=subject_id, limit=limit,
        )
        decay_canonical = [
            e for e in decay_raw if "decay_watch" in (e.tags or ())
        ]

        # G.5 + H (2026-06-09): compute current ack state per ORIGINAL
        # event via chain walk. Handles ack → unack → re-ack chains.
        ack_state = _decay_ack_chain(decay_canonical)

        # factor_verdict events carrying Tier C lens outputs. We surface
        # ALL such events for this subject so the detail page can show
        # the full audit history of subsample / spec_robust / anchor
        # outputs.
        verdicts = store.filter_events(
            event_type="factor_verdict_filed", subject_id=subject_id,
            limit=limit,
        )
        return {
            "subject_id":       subject_id,
            "n_decay_alerts":   len(decay_canonical),
            "decay_alerts":     [
                # G.5: each event gets ack/unack info attached. Admin
                # events (ack/unack) carry `is_admin_event=True` so
                # the UI filters them from the main list (they show
                # as history under their referenced original).
                {
                    **e.to_dict(),
                    "is_ack_event":   "acknowledged"   in (e.tags or ()),
                    "is_unack_event": "unacknowledged" in (e.tags or ()),
                    "is_admin_event": (
                        "acknowledged"   in (e.tags or ())
                        or "unacknowledged" in (e.tags or ())
                    ),
                    # ack_info is the LATEST state for this event (if
                    # this is an original). Acks/unacks themselves get
                    # null here.
                    "ack_info":      ack_state.get(e.event_id),
                }
                for e in decay_canonical
            ],
            "n_factor_verdicts": len(verdicts),
            "factor_verdicts":  [e.to_dict() for e in verdicts],
        }
    except Exception as exc:
        logger.exception("research_store_decay_audit failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get(
    "/api/research_store/verdict/{event_id}", tags=["research"],
)
def research_store_verdict_detail(event_id: str) -> dict:
    """L+M (2026-06-09): drill-down for a single factor_verdict_filed
    event. Used by /research/verdict/[event_id] to render the full
    lens output stack (anchor_orthogonality + subsample_stability +
    specification_robustness + industry + cross_asset). Surfaces the
    target for G.2 (overfit) and G.3 (anchor-spanned) inbox rows.

    Returns 404 if the event_id isn't found or isn't a
    factor_verdict_filed event.
    """
    try:
        from engine.research_store import store
        evt = store.by_event_id(event_id)
        if evt is None:
            raise HTTPException(status_code=404,
                                   detail=f"event {event_id} not found")
        if evt.event_type != "factor_verdict_filed":
            raise HTTPException(
                status_code=422,
                detail=(f"event {event_id} is {evt.event_type}, "
                           "not factor_verdict_filed"),
            )
        return {
            "event":  evt.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("research_store_verdict_detail failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/research_store/summary", tags=["research"])
def research_store_summary() -> dict:
    """Aggregate stats for the store — counts by event_type, verdict, family.
    Used by Cockpit + as a health check."""
    try:
        from engine.research_store import store
        from collections import Counter
        events = store.all_events()
        return {
            "n_total":    len(events),
            "by_event_type": dict(Counter(e.event_type.value for e in events).most_common()),
            "by_verdict":    dict(Counter(e.verdict.value    for e in events).most_common()),
            "by_family":     dict(Counter(e.family for e in events if e.family).most_common(20)),
            "first_ts":   min((e.ts for e in events), default=None),
            "latest_ts":  max((e.ts for e in events), default=None),
        }
    except Exception as exc:
        logger.exception("research_store_summary failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# ── Capacity sub-MVP (Gap C 2026-06-03) ────────────────────────────


@app.get("/api/capacity/families", tags=["research"])
def capacity_families() -> dict:
    """List registered families with capacity-class + threshold AUMs."""
    try:
        from engine.capacity import list_supported_families
        return {"families": list_supported_families()}
    except Exception as exc:
        logger.exception("capacity_families failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/capacity/estimate", tags=["research"])
def capacity_estimate(family: str) -> dict:
    """Family-keyed capacity estimate. Used by SessionLauncher pre-flight
    + /lab/roadmap axis cards (mirror of decay forecast pattern).

    For detailed AUM-level simulation, use engine.portfolio.
    capacity_simulator (Pastor-Stambaugh / Berk-Green framework).
    """
    try:
        from engine.capacity import estimate_for_family
        return estimate_for_family(family).to_dict()
    except Exception as exc:
        logger.exception("capacity_estimate failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# ── Research roadmap (Gap A 2026-06-03) ────────────────────────────


class AxisUpsertRequest(BaseModel):
    axis_id:              str
    name:                 str
    state:                str   # active / queued / paused / closed
    tier:                 str   # committed / candidate / scratchpad
    rationale:            str
    outcome:              str = "NONE"
    parent_axis_id:       str | None = None
    family:               str | None = None
    related_subject_ids:  list[str] = []
    related_memory_files: list[str] = []
    next_actions:         list[str] = []
    blocking_notes:       str = ""


@app.get("/api/roadmap/axes", tags=["research"])
def roadmap_list_axes(
    state:  str | None = None,
    tier:   str | None = None,
    family: str | None = None,
) -> dict:
    """List research axes (the typed roadmap). Optional filters by
    state / tier / family. Sorted active → queued → paused → closed."""
    try:
        from engine.roadmap import store
        axes = store.list_axes(state=state, tier=tier, family=family)
        return {
            "n":    len(axes),
            "axes": [a.to_dict() for a in axes],
        }
    except Exception as exc:
        logger.exception("roadmap_list_axes failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/roadmap/axes/{axis_id}", tags=["research"])
def roadmap_get_axis(axis_id: str) -> dict:
    """Return a single axis by id."""
    try:
        from engine.roadmap import store
        axis = store.get_axis(axis_id)
        if axis is None:
            raise HTTPException(status_code=404, detail=f"axis {axis_id} not found")
        return axis.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("roadmap_get_axis failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/roadmap/axes", tags=["research"])
def roadmap_upsert_axis(req: AxisUpsertRequest) -> dict:
    """Insert or update an axis. Auto-attaches decay estimate when family
    is provided (Gap B integration)."""
    try:
        from engine.roadmap import store as rstore
        decay = None
        capacity = None
        if req.family:
            try:
                from engine.decay_forecast import estimate_for_family as _decay_est
                decay = _decay_est(req.family).to_dict()
            except Exception:
                pass    # decay attachment is best-effort
            try:
                from engine.capacity import estimate_for_family as _cap_est
                capacity = _cap_est(req.family).to_dict()
            except Exception:
                pass    # capacity attachment is best-effort
        axis = rstore.upsert_axis(
            axis_id=req.axis_id,
            name=req.name,
            state=req.state,
            tier=req.tier,
            rationale=req.rationale,
            outcome=req.outcome,
            parent_axis_id=req.parent_axis_id,
            family=req.family,
            related_subject_ids=tuple(req.related_subject_ids),
            related_memory_files=tuple(req.related_memory_files),
            next_actions=tuple(req.next_actions),
            blocking_notes=req.blocking_notes,
            decay_estimate=decay,
            capacity_estimate=capacity,
            actor="user-via-ui",
        )
        return axis.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as exc:
        logger.exception("roadmap_upsert_axis failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# ── Forward decay forecast (Gap B 2026-06-03) ──────────────────────


@app.get("/api/decay_forecast/families", tags=["research"])
def decay_forecast_families() -> dict:
    """List supported factor families with their MP 2016 / LR 2018 decay
    parameters. Used by SessionLauncher pre-flight wizard to populate a
    family dropdown so users pick the right one for their candidate."""
    try:
        from engine.decay_forecast import list_supported_families
        return {"families": list_supported_families()}
    except Exception as exc:
        logger.exception("decay_forecast_families failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/decay_forecast/estimate", tags=["research"])
def decay_forecast_estimate(
    family:           str,
    baseline_alpha:   float | None = None,
    publication_year: int | None = None,
) -> dict:
    """Family-keyed forward decay estimate for a CANDIDATE (no library
    entry yet). Used by SessionLauncher pre-flight wizard to render the
    "alpha mortality" 3-number badge before user commits to a research_new
    session.

    Per Two Sigma factor-proposal pattern: every new candidate should
    surface (empirical decay / theoretical upper-bound / forward 5y α)
    BEFORE you spend hours validating it. Family-typical mortality
    catches known-dead mechanisms before the strict gate does.
    """
    try:
        from engine.decay_forecast import estimate_for_family
        e = estimate_for_family(
            family=family,
            baseline_alpha=baseline_alpha,
            publication_year=publication_year,
        )
        return e.to_dict()
    except Exception as exc:
        logger.exception("decay_forecast_estimate failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# ── Sessions (typed user-initiated workflow protocol) ──────────────


class SessionOpenRequest(BaseModel):
    session_type: str       # "research_new" | "audit" | "ops" | "doctrine" | "exploration"
    title:        str


class SessionPreflightRequest(BaseModel):
    cockpit_reviewed:        bool = False
    decay_alerts_count:      int = 0
    dq_breaches_count:       int = 0
    graveyard_search_query:  str = ""
    graveyard_hits_count:    int = 0
    library_overlap_checked: bool = False
    goal:                    str = ""
    notes:                   str = ""


class SessionAbandonRequest(BaseModel):
    reason: str = ""


@app.post("/api/sessions/open", tags=["sessions"])
def sessions_open(req: SessionOpenRequest) -> dict:
    """Open a new session in pending_preflight state.

    Sets the active singleton pointer so subsequent emits auto-tag with
    this session_id + session_type. Per CLAUDE.md "Session Protocol
    Doctrine" — every user-initiated work block should run inside a
    typed session.
    """
    try:
        from engine.sessions import lifecycle
        from engine.sessions.schema import SessionType
        try:
            stype = SessionType(req.session_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(f"invalid session_type {req.session_type!r}; "
                        f"must be one of: {[t.value for t in SessionType]}"),
            )
        session = lifecycle.open_session(stype, title=req.title.strip() or "(untitled)")
        return session.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sessions_open failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/sessions/{session_id}/preflight", tags=["sessions"])
def sessions_preflight(session_id: str, req: SessionPreflightRequest) -> dict:
    """Record preflight digest + transition pending_preflight → in_flight.

    Returns 409 with explicit missing-field list if the type's checker
    rejects the digest. UI should re-render the wizard with the listed
    fields highlighted.
    """
    try:
        from engine.sessions import lifecycle
        from engine.sessions.schema import PreflightDigest
        from engine.sessions.exceptions import (
            SessionNotFoundError, PreflightIncompleteError,
            InvalidStateTransitionError,
        )
        digest = PreflightDigest(**req.model_dump())
        try:
            session = lifecycle.record_preflight(session_id, digest)
            return session.to_dict()
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        except PreflightIncompleteError as e:
            raise HTTPException(status_code=409, detail={
                "error":   "preflight_incomplete",
                "missing": e.missing,
            })
        except InvalidStateTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sessions_preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/sessions/{session_id}/close", tags=["sessions"])
def sessions_close(session_id: str) -> dict:
    """Close a session — runs the type's exit_check.

    Returns 409 with the unmet-requirements list if exit conditions are
    not satisfied. Caller can then either emit the required artifacts
    or call /abandon (which bypasses the checker but records the
    bypass reason).
    """
    try:
        from engine.sessions import lifecycle
        from engine.sessions.exceptions import (
            SessionNotFoundError, ExitConditionsUnmetError,
            InvalidStateTransitionError,
        )
        try:
            session = lifecycle.close_session(session_id)
            return session.to_dict()
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        except ExitConditionsUnmetError as e:
            raise HTTPException(status_code=409, detail={
                "error":        "exit_conditions_unmet",
                "session_type": e.session_type,
                "missing":      e.missing,
            })
        except InvalidStateTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sessions_close failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.post("/api/sessions/{session_id}/abandon", tags=["sessions"])
def sessions_abandon(session_id: str, req: SessionAbandonRequest) -> dict:
    """Abandon a session — bypasses exit_check.

    For sessions that legitimately produce no artifacts (cancelled
    exploration, false-alarm audit). Reason is recorded in the exit
    report for audit history.
    """
    try:
        from engine.sessions import lifecycle
        from engine.sessions.exceptions import (
            SessionNotFoundError, InvalidStateTransitionError,
        )
        try:
            session = lifecycle.abandon_session(session_id, reason=req.reason)
            return session.to_dict()
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        except InvalidStateTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sessions_abandon failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/sessions/active", tags=["sessions"])
def sessions_active() -> dict:
    """Return the active session pointer + full session object + derived
    sub-phase (awaiting_claude / claude_working / awaiting_close), or
    {active: null} if no session is open.

    Polled by ActiveSessionBanner + /lab/today; emit reads the singleton
    pointer for auto-tagging.
    """
    try:
        from engine.sessions import store
        from engine.sessions.phase import derive_phase
        from dataclasses import asdict
        active = store.get_active()
        if not active:
            return {"active": None}
        session = store.get_session(active["session_id"])
        if session is None:
            return {"active": active, "session": None, "phase": None}
        phase_info = derive_phase(session)
        return {
            "active":  active,
            "session": session.to_dict(),
            "phase":   {
                "phase":             phase_info.phase.value,
                "next_action_label": phase_info.next_action_label,
                "next_action_kind":  phase_info.next_action_kind,
                "last_activity_ts":  phase_info.last_activity_ts,
                "n_events":          phase_info.n_events,
                "n_commits":         phase_info.n_commits,
            },
        }
    except Exception as exc:
        logger.exception("sessions_active failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/sessions", tags=["sessions"])
def sessions_list(
    limit:        int = 50,
    state:        str | None = None,
    session_type: str | None = None,
) -> dict:
    """List sessions newest first. Filter by state / session_type."""
    try:
        from engine.sessions import store
        limit = max(1, min(int(limit), 500))
        sessions = store.list_sessions(
            limit=limit, state=state, session_type=session_type,
        )
        return {
            "n":        len(sessions),
            "sessions": [s.to_dict() for s in sessions],
        }
    except Exception as exc:
        logger.exception("sessions_list failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# IMPORTANT: static /types must be declared BEFORE dynamic /{session_id}
# otherwise FastAPI matches 'types' as a session_id and 404s.
@app.get("/api/sessions/types", tags=["sessions"])
def sessions_types() -> dict:
    """Static metadata for the 5 session types — descriptions, expected
    durations, exit-condition summaries. Drives the SessionLauncher UI
    so it doesn't hard-code copy."""
    try:
        from engine.sessions import protocols
        from engine.sessions.schema import SessionType
        out = []
        for stype in SessionType:
            mod = protocols.for_type(stype)
            out.append({
                "session_type":       stype.value,
                "description":        getattr(mod, "DESCRIPTION", ""),
                "expected_duration":  getattr(mod, "EXPECTED_DURATION", ""),
            })
        return {"types": out}
    except Exception as exc:
        logger.exception("sessions_types failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/sessions/{session_id}", tags=["sessions"])
def sessions_get(session_id: str) -> dict:
    """Return full state of a single session. Includes linked events +
    git commits gathered by the lifecycle at close time."""
    try:
        from engine.sessions import store
        from engine.research_store import store as event_store
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        # Pull events tagged with this session_id
        events = []
        for ev in event_store.all_events():
            if any(t == f"session:{session_id}" for t in ev.tags):
                events.append(ev.to_dict())
        return {
            "session": session.to_dict(),
            "events":  events,
            "n_events": len(events),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sessions_get failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@app.get("/api/agents", tags=["agents"])
def agents() -> dict:
    """The agent constellation directory: the Chief of Staff (supervisor) + 7 specialists, each
    with its real persona fields (tool palette, workload/model routing, spec reference, iteration
    cap) so the /agents page renders ground truth, not a hand-maintained list. Backward-compatible:
    `specialists[].agent_id/scope` and `specialists.length` are preserved."""
    import json as _json
    from engine.agents.persona import (
        CHIEF_OF_STAFF, RISK_MANAGER, DQ_INSPECTOR, DECAY_SENTINEL,
        ANOMALY_SENTINEL, ATTRIBUTION_ANALYST, AUDIT_RECORDER, DEVILS_ADVOCATE,
    )
    from engine.agents.persona.tools import list_personas

    directory = _json.loads(list_personas())
    scopes = {s["agent_id"]: s["scope"] for s in directory.get("specialists", [])}

    def card(p, kind: str, scope: str) -> dict:
        return {
            "agent_id":       p.agent_id,
            "name":           p.name,
            "kind":           kind,                       # "supervisor" | "specialist"
            "role_id":        p.role_id,
            "workload":       p.workload,                 # provider+model routing string
            "spec_ref":       p.spec_ref,
            "max_iterations": p.max_iterations,
            "tools":          [t["name"] for t in p.tools],
            "scope":          scope,
        }

    # Specialist order = display order on the page (most operationally central first).
    specialists = [RISK_MANAGER, DQ_INSPECTOR, DECAY_SENTINEL, ANOMALY_SENTINEL,
                   ATTRIBUTION_ANALYST, AUDIT_RECORDER, DEVILS_ADVOCATE]
    return {
        "chief_of_staff": card(
            CHIEF_OF_STAFF, "supervisor",
            "single user-facing entry point; routes to specialists (Supervisor pattern, "
            "≤3 delegations/turn, Pattern-5 autonomous debate banned)",
        ),
        "specialists":    [card(p, "specialist", scopes.get(p.agent_id, "")) for p in specialists],
        "delegation_rule": directory.get("delegation_rule", ""),
    }


# ── /api/chat — the ONLY LLM-touching endpoint (SSE) ─────────────────────────
# Streams the Chief of Staff turn as Server-Sent Events: the CoS narrates + routes to
# specialists; book decisions stay deterministic in the engine (0-LLM-in-DECISION). Guards:
# (1) provider key present, else clean 503; (2) per-day USD cap on chat spend; (3) local-only
# (the app binds 127.0.0.1). The read-only pages work fully without any of this.
_CHAT_DAILY_USD_CAP = 2.00  # blueprint open-decision #4 placeholder; raise once a number is set


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # Anthropic-format messages the client holds (stateless server)


def _anthropic_key_present() -> bool:
    """CoS routes to the anthropic workload — check that key is configured (secrets.toml/env)."""
    try:
        from engine.llm.providers.anthropic_provider import _get_api_key
        return bool(_get_api_key())
    except Exception:
        return False


def _chat_budget_left() -> float:
    """USD remaining under today's chat cap (CoS-side spend; delegated specialist cost is extra
    but bounded to ≤3 delegations/turn). Fails open if the ledger can't be read."""
    try:
        from engine.llm_cost_ledger import get_total_today
        return _CHAT_DAILY_USD_CAP - get_total_today("chief_of_staff")
    except Exception:
        return _CHAT_DAILY_USD_CAP


@app.post("/api/chat", tags=["agents"])
def chat(req: ChatRequest):
    """Stream a Chief-of-Staff turn as SSE (text/event-stream). Events: start / iteration /
    assistant_text / tool_call / tool_result / done / error — see chat_turn_events()."""
    from fastapi.responses import StreamingResponse

    if not req.message.strip():
        raise HTTPException(status_code=422, detail="empty message")
    if not _anthropic_key_present():
        raise HTTPException(
            status_code=503,
            detail="LLM provider not configured — set ANTHROPIC_API_KEY in "
                   ".streamlit/secrets.toml or env to enable the chat terminal. "
                   "The read-only pages work without it.",
        )
    if _chat_budget_left() <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"daily chat budget reached (${_CHAT_DAILY_USD_CAP:.2f}/day). "
                   f"Resets tomorrow; the read-only pages are unaffected.",
        )

    def event_stream():
        from engine.agents.persona import CHIEF_OF_STAFF
        from engine.agents.persona.base import chat_turn_events
        try:
            for ev in chat_turn_events(CHIEF_OF_STAFF, req.message, history=req.history):
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:  # never leak a traceback into the stream
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── Serve the built frontend at one origin ───────────────────────────────────
# `cd frontend && npm run build` emits frontend/out/ (Next static export). FastAPI serves
# it so http://localhost:8000 = the whole app on ONE port (no second server, no CORS).
# A SPA-aware resolver (registered AFTER /api + /health + /docs) maps a route to the right
# static file — {path}.html, {path}/index.html, the raw asset, else the client-routed
# index — so direct loads, refreshes, and deep-links all work (Next export emits
# dashboard.html, which StaticFiles(html=True) alone would 404 at /dashboard).
_FRONTEND_OUT = ROOT / "frontend" / "out"
if _FRONTEND_OUT.is_dir():
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/_next", StaticFiles(directory=str(_FRONTEND_OUT / "_next")), name="next-assets")

    _RESERVED = {"health", "docs", "redoc", "openapi.json"}

    # Dead-link safety map (Phase 1 of UI restructure, 2026-06-14).
    # `output: "export"` disables Next.js redirects(), so we centralize
    # 308 (Permanent Redirect) handling here BEFORE the SPA fallback.
    # 308 vs 301 — 308 preserves request method + body (better semantics)
    # and is cached by browsers, so bookmarks update on first visit.
    # Each entry: old_path_prefix → new_path. Prefix-match so query
    # strings + sub-paths flow through unchanged.
    _DEAD_PAGE_REDIRECTS: dict[str, str] = {
        # Phase 1 (2026-06-14): pages superseded / demoted (R5 cleanup
        # made them rail-less but URLs were still live, accreting
        # bookmarks + Cmd-K hits).
        "lab/cockpit":              "/dashboard",
        "lab/factor-lab":           "/research/forward",
        "lab/cosine-heatmap":       "/research/forward/anchors",
        "lab/outcomes":             "/research/lessons",
        "lab/chains":               "/research/forward",
        "lab/series":               "/research/lessons",
        "lab/axes":                 "/research/roadmap",
        # Phase 2 (2026-06-14): /lab/today god-page merged into /dashboard
        "lab/today":                "/dashboard",
        "lab/l4":                   "/dashboard",
        # Phase 3 (2026-06-14): /lab/* → /research/* namespace migration
        # (and /lab/liveness → /ops/liveness). Prefix-match means
        # /lab/library/detail also catches; sub-path tail is dropped
        # per the _redirect_for doctrine (target root with browser-
        # cached 308 → user re-deep-links via the new page's controls).
        "lab/library":              "/research/library",
        "lab/decay":                "/research/decay",
        "lab/literature":           "/research/reading",
        "lab/sessions":             "/research/sessions",
        "lab/roadmap":              "/research/roadmap",
        "lab/liveness":             "/ops/liveness",
        # Older retired routes
        "lab/council":              "/research/library",
        # Duplicate paper viewer
        "research/papers/view":     "/research/papers",
    }

    # Subset of dead-page redirects where the target is a true namespace
    # successor (same page, new URL). For these we PRESERVE the sub-path
    # tail so deep-links like /lab/library/detail?sleeve=X cleanly become
    # /research/library/detail?sleeve=X. For the rest (page killed and
    # remapped to something thematically related) we drop the tail per
    # the standard rule.
    _NAMESPACE_MIGRATIONS = {
        "lab/library", "lab/decay", "lab/literature",
        "lab/sessions", "lab/roadmap", "lab/liveness",
    }

    def _redirect_for(clean: str) -> str | None:
        """Return target URL if `clean` matches a dead-page prefix, else None.
        For namespace-migration prefixes (Phase 3), the sub-path tail is
        preserved so deep-links survive (/lab/library/detail?sleeve=X →
        /research/library/detail?sleeve=X). For pure dead-page entries
        the tail is dropped — landing the user at the target root is the
        honest behavior since the sub-path has no stable equivalent."""
        for prefix, target in _DEAD_PAGE_REDIRECTS.items():
            if clean == prefix or clean.startswith(prefix + "/"):
                if prefix in _NAMESPACE_MIGRATIONS:
                    tail = clean[len(prefix):]    # "" or "/sub..."
                    return f"{target}{tail}"
                return target
        return None

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        if full_path.startswith("api/") or full_path in _RESERVED:
            raise HTTPException(status_code=404, detail="not found")
        clean = full_path.strip("/")
        # Dead-page redirect FIRST (so deleting source files in a future
        # commit can't accidentally bypass the redirect through the SPA
        # fallback below).
        if (target := _redirect_for(clean)):
            return RedirectResponse(target, status_code=308)
        candidates = (
            [_FRONTEND_OUT / "index.html"] if clean == ""
            else [_FRONTEND_OUT / f"{clean}.html", _FRONTEND_OUT / clean / "index.html", _FRONTEND_OUT / clean]
        )
        for c in candidates:
            if c.is_file():
                return FileResponse(str(c))
        return FileResponse(str(_FRONTEND_OUT / "index.html"))  # client-routed fallback
else:
    @app.get("/", tags=["meta"])
    def _no_build() -> dict:
        return {"message": "Frontend build not found. Run `cd frontend && npm run build` "
                           "to generate frontend/out, then reload. API is live at /api/* and /docs."}

