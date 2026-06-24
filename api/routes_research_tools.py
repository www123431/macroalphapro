"""api/routes_research_tools.py — Phase 4a.6 shared REST shim over
the 9 Session-3 research tools.

Serves both UI sections (Cockpit + Assistant). The same TOOLS registry
already exposed via MCP (engine.research.mcp_server, for Claude Code
desktop) is exposed here as REST for the Next.js frontend. Single
source of truth: adding a tool to engine.research.llm_tools.TOOLS
auto-exposes it on both surfaces.

Audit ledger (data/research/ui_tool_calls.jsonl) records EVERY call —
read or write — with caller identity, args, result hash, latency. The
hook is here even though current 9 tools are all read-only, so the
write tools coming later (override_graveyard, promote_to_paper,
deploy_capital) can ship without bolting audit on retroactively.

Public surface:
  GET  /api/research/tools                  — list 9 tools + schemas
  POST /api/research/call/{tool_name}       — invoke a tool with args
  GET  /api/research/audit?limit=N          — recent audit entries
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from engine.research.llm_tools import (
    TOOLS, dispatch, tool_specs_for_anthropic,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LEDGER = REPO_ROOT / "data" / "research" / "ui_tool_calls.jsonl"


# ── Audit ledger ──────────────────────────────────────────────────────


def _sha16(payload: Any) -> str:
    """16-hex-char digest of any JSON-serializable payload."""
    blob = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _append_audit(
    tool_name: str,
    args: dict,
    result_hash: Optional[str],
    ok: bool,
    latency_ms: float,
    caller: str,
    error: Optional[str] = None,
) -> None:
    """Append one audit entry. Best-effort; never raises into the request
    handler (audit failure must not break the user-facing call)."""
    try:
        AUDIT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts":          _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tool":        tool_name,
            "args":        args,
            "args_hash":   _sha16(args),
            "result_hash": result_hash,
            "ok":          ok,
            "latency_ms":  round(latency_ms, 1),
            "caller":      caller,
            "error":       error,
        }
        with AUDIT_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        logger.exception("audit ledger append failed (non-fatal)")


def _classify_caller(referer: Optional[str], user_agent: Optional[str]) -> str:
    """Best-effort caller identity from request headers.

    The frontend sets X-Research-Caller in fetch(); browsers reliably
    send Referer too. Falls back to 'unknown' rather than guessing —
    audit value depends on not lying about provenance.
    """
    if referer:
        # 4d.6 IA refactor + 2026-06-01 layout Phase 1: legacy
        # /research/cockpit + /research/assistant redirect stubs
        # have been retired; only /lab/* paths remain.
        if "/lab/cockpit" in referer:
            return "ui_lab_cockpit"
        if "/lab/assistant" in referer:
            return "ui_lab_assistant"
        if "/lab/series" in referer:
            return "ui_lab_series"
        if "/research/candidate" in referer:
            return "ui_candidate"
    if user_agent and "python-requests" in (user_agent or "").lower():
        return "script"
    return "unknown"


# ── Public endpoints ──────────────────────────────────────────────────


@router.get("/tools")
def list_tools() -> dict:
    """List the 9 research tools with full Anthropic-format input
    schemas. The frontend uses this to render dynamic forms / tool
    pickers; agents use this to know what's available."""
    return {
        "n_tools": len(TOOLS),
        "tools":   tool_specs_for_anthropic(),
    }


class _CallRequest(BaseModel):
    args: dict


@router.post("/call/{tool_name}")
def call_tool(
    tool_name: str,
    body: _CallRequest,
    x_research_caller: Optional[str] = Header(None),
    referer:           Optional[str] = Header(None),
    user_agent:        Optional[str] = Header(None),
) -> dict:
    """Dispatch a tool by name. Args validated against the tool's
    registered Pydantic schema by dispatch(). Audit-logged on every
    call (success or failure).

    Caller identity is taken from X-Research-Caller header if set,
    otherwise inferred from Referer; falls back to 'unknown'."""
    if tool_name not in TOOLS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown tool: {tool_name!r}. "
                   f"GET /api/research/tools for the registry.",
        )

    caller = x_research_caller or _classify_caller(referer, user_agent)
    args = body.args or {}
    t0 = time.perf_counter()

    try:
        result = dispatch(tool_name, **args)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result_hash = _sha16(result)
        _append_audit(tool_name, args, result_hash, ok=True,
                      latency_ms=latency_ms, caller=caller)
        return {
            "tool":        tool_name,
            "ok":          True,
            "result":      result,
            "result_hash": result_hash,
            "latency_ms":  round(latency_ms, 1),
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        _append_audit(tool_name, args, None, ok=False,
                      latency_ms=latency_ms, caller=caller,
                      error=str(exc)[:200])
        logger.exception("research tool %s failed", tool_name)
        raise HTTPException(
            status_code=502,
            detail=f"tool {tool_name} failed: {exc}",
        )


@router.get("/council/runs")
def council_runs(
    limit: int = Query(50, ge=1, le=500),
    consensus: Optional[str] = Query(None,
        description="filter by APPROVE / NEEDS_REVISION / REJECT"),
) -> dict:
    """Phase 4b.5: list recent 3-agent council runs (newest first).

    Backs the Cockpit "Council activity" panel — each row gives
    consensus / proposal title / family / elapsed / per-critic verdicts.
    Drill-down via /api/research/council/run/{run_id}."""
    from engine.research.agent_council import read_council_runs
    runs = read_council_runs(limit=limit, consensus=consensus)
    return {"n": len(runs), "runs": runs}


@router.get("/council/run/{run_id}")
def council_run_detail(run_id: str) -> dict:
    """Drill-down: full council run by id. Includes the proposal
    body, every critic's verdict + tool calls + rationale."""
    from engine.research.agent_council import read_council_run_by_id
    row = read_council_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"no council run with id={run_id!r}")
    return row


class _CouncilTriggerRequest(BaseModel):
    seed_idea: str
    confirm_cost: bool = False  # explicit ack of LLM cost (~50s + tokens)
    candidate_returns_path: Optional[str] = None  # Phase 4d: if set, pipeline runs empirically
    # Frontier 1 (2026-06-01): structured reflection round. Opt-in,
    # ~2x token cost. Each critic gets ONE shot to read the peer
    # verdict and revise. Pattern 6, NOT Pattern 5 autonomous debate.
    enable_reflection: bool = False


@router.post("/council/run")
def council_trigger(
    body: _CouncilTriggerRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Phase 4b.5 + 4c: trigger a council run.

    Path A (Temporal available, 4c+):
      Enqueues an L4DiscoveryWorkflow + returns workflow_id IMMEDIATELY.
      UI polls GET /api/research/council/workflow/{workflow_id} for status.

    Path B (Temporal not running):
      Falls back to synchronous run (~30-60s block). UI sees the
      finished result in the response. Audit-logged identically.

    Requires confirm_cost=true — each invocation costs LLM tokens.
    """
    if not body.confirm_cost:
        raise HTTPException(
            status_code=400,
            detail="confirm_cost=true required (each run costs LLM tokens)",
        )
    if not body.seed_idea or len(body.seed_idea.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="seed_idea too short — give the architect a concrete idea",
        )

    import asyncio
    from engine.research.agent_council import _load_anthropic_key
    if not _load_anthropic_key():
        raise HTTPException(
            status_code=503,
            detail="no ANTHROPIC_API_KEY configured; council unavailable",
        )

    caller = x_research_caller or "ui_cockpit"
    t0 = time.perf_counter()

    # Path A — Temporal
    try:
        from engine.research.l4_temporal_client import (
            enqueue_council_workflow, is_temporal_available,
        )
        if asyncio.run(is_temporal_available(timeout=0.5)):
            handle = asyncio.run(enqueue_council_workflow(
                body.seed_idea,
                candidate_returns_path=body.candidate_returns_path,
            ))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            _append_audit(
                "council_run_async",
                {"seed_idea_len": len(body.seed_idea),
                 "path": "temporal", "workflow_id": handle["workflow_id"]},
                result_hash=handle["workflow_id"], ok=True,
                latency_ms=latency_ms, caller=caller,
            )
            return {
                "workflow_id": handle["workflow_id"],
                "temporal_run_id": handle["run_id"],
                "path":         "temporal",
                "status":       "RUNNING",
                "message":      ("workflow enqueued; poll GET "
                                  "/api/research/council/workflow/"
                                  f"{handle['workflow_id']} for status"),
            }
    except Exception as exc:
        logger.warning("Temporal path failed, falling back to sync: %s", exc)

    # Path B — sync fallback
    from engine.research.agent_council import run_full_council
    try:
        proposal, council = asyncio.run(run_full_council(
            body.seed_idea, enable_reflection=body.enable_reflection,
        ))
    except Exception as exc:
        logger.exception("council trigger failed (sync fallback)")
        raise HTTPException(status_code=502,
                            detail=f"council run failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    _append_audit(
        "council_run",
        {"seed_idea_len":     len(body.seed_idea),
         "path":              "sync",
         "reflection":        body.enable_reflection,
         "round_1_consensus": council.round_1_consensus},
        result_hash=council.run_id, ok=True,
        latency_ms=latency_ms, caller=caller,
    )
    return {
        "run_id":              council.run_id,
        "consensus":           council.consensus,
        "rationale":           council.rationale,
        "elapsed_s":           council.elapsed_s,
        "proposal":            proposal.to_dict(),
        "n_critics":           len(council.verdicts),
        "path":                "sync",
        "reflection_enabled":  council.reflection_enabled,
        "round_1_consensus":   council.round_1_consensus,
        "round_1_rationale":   council.round_1_rationale,
        "reflection_actions":  [
            v.reflection_action for v in council.verdicts
            if v.reflection_action is not None
        ],
    }


@router.get("/council/suggestions")
def council_suggestions(limit: int = Query(10, ge=1, le=50)) -> dict:
    """Phase 4d.5: L1 candidate seed recommender.

    Ranked list of seed ideas worth exploring next — combines library
    UNTESTED entries with the senior-curated seed pool, scored by
    underexplored × no-cousin × role-gap heuristics. NO LLM call —
    pure data scan (fast, deterministic, repeatable)."""
    from engine.research.suggestion_engine import get_candidate_suggestions
    return get_candidate_suggestions(limit=limit)


# ── Frontier 2 (2026-06-01): L4 cron continuous background ──────────


@router.get("/l4/cron/status")
def l4_cron_status() -> dict:
    """Snapshot of the L4 cron Schedule + recent cron fires.

    Returns:
      schedule:    paused/running/next_run/cron_spec (from Temporal)
      recent_runs: last 20 cron fires from l4_cron_runs.jsonl

    Tolerates Temporal being down — schedule.exists=False is honest,
    not an error. UI renders "cron offline" rather than crashing."""
    import asyncio
    from engine.research.l4_cron import cron_status
    try:
        return asyncio.run(cron_status())
    except Exception as exc:
        logger.exception("cron_status failed")
        return {
            "schedule":    {"exists": False, "error": str(exc)[:200]},
            "recent_runs": [],
        }


class _CronEnableRequest(BaseModel):
    cron_spec: str = "0 9 * * *"   # daily 09:00 server-local
    paused:    bool = False
    confirm_cost: bool = False     # ack ongoing LLM token spend


@router.post("/l4/cron/enable")
def l4_cron_enable(
    body: _CronEnableRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Enable / update the L4 cron Schedule. Idempotent.

    Each fire spends ~$0.05-0.20 in LLM tokens (council + reflection).
    confirm_cost=true required to prevent accidental enablement."""
    if not body.confirm_cost:
        raise HTTPException(
            status_code=400,
            detail="confirm_cost=true required — daily cron spends LLM tokens",
        )
    import asyncio
    from engine.research.l4_cron import enable_l4_cron
    caller = x_research_caller or "unknown"
    try:
        out = asyncio.run(enable_l4_cron(
            cron_spec=body.cron_spec, paused=body.paused,
        ))
    except Exception as exc:
        logger.exception("enable_l4_cron failed")
        raise HTTPException(status_code=502,
                            detail=f"failed to enable cron: {exc}")
    _append_audit("l4_cron_enable",
                  {"cron": body.cron_spec, "paused": body.paused},
                  result_hash=out.get("schedule_id"), ok=True,
                  latency_ms=0.0, caller=caller)
    return out


@router.post("/l4/cron/disable")
def l4_cron_disable(
    delete: bool = Query(False, description="delete vs. pause (default: pause)"),
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Pause (default, reversible) or delete the L4 cron Schedule."""
    import asyncio
    from engine.research.l4_cron import disable_l4_cron
    caller = x_research_caller or "unknown"
    try:
        out = asyncio.run(disable_l4_cron(delete=delete))
    except Exception as exc:
        logger.exception("disable_l4_cron failed")
        raise HTTPException(status_code=502,
                            detail=f"failed to disable cron: {exc}")
    _append_audit("l4_cron_disable",
                  {"delete": delete},
                  result_hash=out.get("schedule_id"), ok=bool(out.get("ok")),
                  latency_ms=0.0, caller=caller)
    return out


# ── Frontier 4 (2026-06-01): multi-step research chains ──────────────


@router.get("/chains")
def chain_catalogue() -> dict:
    """List all registered research chains + their step structure.

    Backs an upcoming Lab/Chains page; for now consumed by Claude Code
    and direct API callers."""
    from engine.research.chain_library import list_chains
    return {"chains": list_chains()}


class _ChainRunRequest(BaseModel):
    chain_id:        str
    initial_context: dict = {}
    confirm_cost:    bool = False   # ack tool-call cost


@router.post("/chains/run")
def chain_run(
    body: _ChainRunRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Execute one chain synchronously. Each step is a tool call —
    no LLM unless the chain explicitly includes an LLM-calling tool.

    confirm_cost only blocks chains that touch external services
    (arxiv / sec_edgar / fred); pure-local chains run without it."""
    from engine.research.chain_library import get_chain
    from engine.research.research_chain import run_chain
    try:
        chain = get_chain(body.chain_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # External-tool detection — chains that touch external services
    # must explicitly confirm cost. arxiv has rate limits; sec_edgar
    # has Fair Access policy; fred has API quota.
    EXTERNAL_TOOLS = {"arxiv_search", "sec_edgar_search", "fred_query"}
    needs_cost_confirm = any(
        s.tool in EXTERNAL_TOOLS for s in chain.steps
    )
    if needs_cost_confirm and not body.confirm_cost:
        raise HTTPException(
            status_code=400,
            detail=(f"chain {body.chain_id} touches external services; "
                    "confirm_cost=true required"),
        )

    caller = x_research_caller or "unknown"
    t0 = time.perf_counter()
    try:
        run = run_chain(chain, initial_context=body.initial_context)
    except Exception as exc:
        logger.exception("chain_run failed")
        raise HTTPException(status_code=502,
                            detail=f"chain run failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    _append_audit("chain_run",
                  {"chain_id": body.chain_id,
                   "status":   run.status,
                   "n_steps":  len(run.steps)},
                  result_hash=run.run_id, ok=True,
                  latency_ms=latency_ms, caller=caller)
    return run.to_dict()


@router.get("/chains/runs")
def chain_runs(
    chain_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """List recent chain runs newest-first. Filterable by chain_id."""
    from engine.research.research_chain import read_recent_chain_runs
    rows = read_recent_chain_runs(chain_id=chain_id, limit=limit)
    return {"n": len(rows), "runs": rows}


# ── UI v2 (2026-06-01): Factor Lab — PFH + axes ──────────────────


@router.get("/factor_lab/catalog")
def factor_lab_catalog() -> dict:
    """Axis catalog snapshot + the 'tested vs untested' tuple list.

    Backs the factor matrix heatmap and the KPI strip on /lab/factor-lab.
    All numbers come from filesystem; no LLM / Anthropic call."""
    from engine.research.pfh.axis_catalog import (
        enumerate_untested_tuples, load_axis_catalog,
    )
    from engine.research.pfh.catalog import (
        load_labeled_mechanisms, overall_base_rate,
    )
    cat = load_axis_catalog()
    untested = enumerate_untested_tuples(cat)
    labels = load_labeled_mechanisms()
    br = overall_base_rate(labels)
    return {
        "universes":   sorted(cat.universes),
        "signals":     sorted(cat.signals),
        "weightings":  sorted(cat.weightings),
        "n_possible":  cat.n_possible,
        "n_untested":  cat.n_untested,
        "tested_tuples":   [list(t) for t in sorted(cat.tested_tuples)],
        "untested_tuples": [list(t) for t in untested],
        "labels_summary":  br,
    }


@router.get("/factor_lab/axes/details")
def factor_lab_axes_details() -> dict:
    """Full YAML payload for every universe / signal / weighting,
    so the /lab/axes browser can render details without N round-trips."""
    from pathlib import Path
    import yaml as _yaml
    repo_root = Path(__file__).resolve().parent.parent
    base = repo_root / "data" / "feature_store"

    def _load_yaml_dir(subdir: str) -> list[dict]:
        d = base / subdir
        out: list[dict] = []
        if d.is_dir():
            for p in sorted(d.glob("*.yaml")):
                if p.name.startswith("_"):
                    continue
                try:
                    raw = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                except Exception as exc:
                    out.append({"name": p.stem, "error": str(exc)[:120]})
                    continue
                out.append({**raw, "filename": p.name})
        return out

    return {
        "universes":   _load_yaml_dir("_universes"),
        "signals":     _load_yaml_dir("_signal_recipes"),
        "weightings":  _load_yaml_dir("_weightings"),
    }


class _PFHSuggestRequest(BaseModel):
    k:               int = 6
    mode:            str = "constrained"   # "open" | "constrained"
    max_per_family:  int = 2
    max_per_universe: int = 2
    write_specs:     bool = True    # required for inline materialize
    prior_strength:  float = 4.0


@router.post("/factor_lab/pfh/suggest")
def factor_lab_pfh_suggest(body: _PFHSuggestRequest,
                              x_research_caller: Optional[str] = Header(None)) -> dict:
    """Synchronous PFH suggestion run. Writes the top-K compose-spec
    YAMLs so the materialize endpoint can immediately consume them."""
    from engine.research.pfh.proposer import suggest_top_k
    caller = x_research_caller or "ui_factor_lab"
    try:
        out = suggest_top_k(
            k=body.k, mode=body.mode,
            max_per_family=body.max_per_family,
            max_per_universe=body.max_per_universe,
            prior_strength=body.prior_strength,
            write_specs=body.write_specs,
            write_ledger=True,
        )
    except Exception as exc:
        logger.exception("PFH suggest failed")
        raise HTTPException(status_code=502, detail=f"PFH failed: {exc}")
    _append_audit("factor_lab_pfh_suggest",
                  {"k": body.k, "mode": body.mode,
                   "n_candidates": out.get("n_candidates_total")},
                  result_hash=out.get("run_id"), ok=True,
                  latency_ms=0.0, caller=caller)
    return out


class _FactorLabMaterializeRequest(BaseModel):
    spec_id: str
    force:   bool = False


@router.post("/factor_lab/materialize")
def factor_lab_materialize(body: _FactorLabMaterializeRequest,
                              x_research_caller: Optional[str] = Header(None)) -> dict:
    """Materialize one compose-spec inline so the UI can show Sharpe
    next to the PFH suggestion that produced it.

    Errors are caught and returned as part of the response (not raised)
    so the UI can render a per-row failure state without breaking the
    surrounding suggestion list."""
    from engine.feature_store import materialize_spec
    caller = x_research_caller or "ui_factor_lab"
    try:
        r = materialize_spec(body.spec_id, force=body.force, strict_sanity=False)
        _append_audit("factor_lab_materialize",
                      {"spec_id": body.spec_id, "force": body.force},
                      result_hash=r.get("input_hash"), ok=True,
                      latency_ms=0.0, caller=caller)
        return {"ok": True, "result": r}
    except Exception as exc:
        logger.exception("factor_lab materialize failed for %s", body.spec_id)
        _append_audit("factor_lab_materialize",
                      {"spec_id": body.spec_id, "force": body.force},
                      result_hash=None, ok=False,
                      latency_ms=0.0, caller=caller,
                      error=str(exc)[:200])
        return {"ok": False, "error": str(exc)[:300]}


@router.get("/factor_lab/spec/{spec_id}")
def factor_lab_spec_detail(spec_id: str) -> dict:
    """Full compose-spec / function-wrapper-spec YAML + materialize
    history + Bayesian posterior context for the spec's family.

    Backs /lab/factor-lab/detail. Spec_id may be a function-wrapper
    spec (e.g. cross_asset_carry_4leg) or a compose-spec (e.g.
    eq_mom_12_1_us_real) or a PFH-emitted spec (pfh_constrained_*)."""
    from pathlib import Path
    import yaml as _yaml
    repo_root = Path(__file__).resolve().parent.parent
    spec_path = repo_root / "data" / "feature_store" / "_specs" / f"{spec_id}.yaml"
    if not spec_path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"spec {spec_id} not found")
    try:
        raw = _yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"YAML parse failed: {exc}")

    spec_kind = "compose" if "compose" in raw else "function"

    # Find materialized output(s) — addressed by spec_id.v{version}.{hash}
    computed_dir = repo_root / "data" / "feature_store" / "_computed"
    materializations: list[dict] = []
    if computed_dir.is_dir():
        prefix = f"{spec_id}.v"
        for p in sorted(computed_dir.glob(f"{prefix}*.meta.json")):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                materializations.append({
                    "meta_filename":     p.name,
                    "parquet_filename":  p.name.replace(".meta.json", ".parquet"),
                    "input_hash":        meta.get("input_hash"),
                    "materialized_at":   meta.get("materialized_at"),
                    "elapsed_s":         meta.get("elapsed_s"),
                    "spec_kind":         meta.get("spec_kind"),
                    "validation":        meta.get("validation"),
                    "compose_axes":      meta.get("compose_axes"),
                })
            except Exception:
                pass

    # Resolve axis-component detail if compose-spec
    axes_detail: dict = {}
    if spec_kind == "compose":
        compose_block = raw.get("compose") or {}
        def _resolve_ref(axis_dir: str, ref: str) -> dict:
            p = repo_root / "data" / "feature_store" / axis_dir / f"{ref}.yaml"
            if p.is_file():
                try:
                    return _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                except Exception:
                    return {}
            return {"_missing": True, "ref": ref}
        def _ref(entry):
            return entry.get("ref") if isinstance(entry, dict) else str(entry)
        if "universe" in compose_block:
            axes_detail["universe"]  = _resolve_ref("_universes",     _ref(compose_block["universe"]))
        if "signal"   in compose_block:
            axes_detail["signal"]    = _resolve_ref("_signal_recipes",_ref(compose_block["signal"]))
        if "weighting" in compose_block:
            axes_detail["weighting"] = _resolve_ref("_weightings",    _ref(compose_block["weighting"]))

    # PFH posterior context — look up the family this spec implies
    posterior_context: Optional[dict] = None
    try:
        from engine.research.pfh.catalog import (
            load_labeled_mechanisms, overall_base_rate, per_family_counts,
        )
        from engine.research.pfh.bayesian import score_candidate
        from engine.research.pfh.constrained_generator import _infer_family_from_signal
        labels = load_labeled_mechanisms()
        br = overall_base_rate(labels)
        fam_counts = per_family_counts(labels)
        # Infer family
        family: Optional[str] = raw.get("mechanism_library_id")  # if function-wrapper, may carry library_id
        if not family and spec_kind == "compose":
            compose_block = raw.get("compose") or {}
            sig = compose_block.get("signal")
            sig_ref = sig.get("ref") if isinstance(sig, dict) else sig
            if sig_ref:
                family = _infer_family_from_signal(sig_ref)
        if family:
            cell = fam_counts.get(family, {"n_green": 0, "n_yellow": 0, "n_red": 0})
            post = score_candidate(
                n_green=cell["n_green"], n_yellow=cell["n_yellow"], n_red=cell["n_red"],
                base_rate=br["p_green"] or 0.5, prior_strength=4.0,
            )
            posterior_context = {
                "family":           family,
                "n_green":          cell["n_green"],
                "n_yellow":         cell["n_yellow"],
                "n_red":            cell["n_red"],
                "posterior_mean":   post.posterior_mean,
                "credible_05":      post.credible_05,
                "credible_95":      post.credible_95,
                "base_rate_used":   br["p_green"],
            }
    except Exception as exc:
        logger.exception("posterior_context failed for %s", spec_id)

    return {
        "spec_id":           spec_id,
        "spec_kind":         spec_kind,
        "yaml":              raw,
        "axes_detail":       axes_detail,
        "materializations":  materializations,
        "posterior_context": posterior_context,
        "filename":          spec_path.name,
    }


@router.get("/factor_lab/pfh/history")
def factor_lab_pfh_history(limit: int = Query(20, ge=1, le=100)) -> dict:
    """Recent PFH suggestion runs newest-first (for the "history" tab
    on /lab/factor-lab)."""
    from engine.research.pfh.proposer import read_pfh_history
    runs = read_pfh_history(limit=limit)
    return {"n": len(runs), "runs": runs}


# ── Path C UI (2026-06-01): lab page data endpoints ────────────────


@router.get("/strict_gate/funnel")
def strict_gate_funnel() -> dict:
    """Aggregate pipeline_self_audit.jsonl into a per-step funnel.

    Each step gets {n_total, n_pass, n_fail, n_skip, n_warn,
    n_error, pass_rate}. The ordered list matches the canonical
    strict-gate sequence so the UI can render left-to-right as
    "candidates surviving at each gate".

    The ORDER below mirrors the doctrine pipeline order in
    engine.research.candidate_pipeline (run_candidate_pipeline);
    skips don't count against the step but reduce the population
    that flows downstream — visible in the funnel as gap-fill.
    """
    from pathlib import Path
    import json as _json
    from collections import Counter

    audit_path = Path(__file__).resolve().parent.parent / "data" / "research" / "pipeline_self_audit.jsonl"

    # Canonical order (ground truth for the funnel viz). Steps
    # missing from any audit row default to all-zero counts.
    CANONICAL = [
        ("data_quality",            "Data quality",         "fresh / consistent"),
        ("H10_evaluate_candidate",  "Evaluate (H10)",       "role + H8/H9 strict gate"),
        ("H2_cousin_check",         "Cousin check (H2)",    "vs. library cousins"),
        ("H6_post_pub_evidence",    "Post-pub evidence",    "evidence chain present"),
        ("graveyard_check",         "Graveyard check",      "no RED match in family"),
        ("cost_model_check",        "Cost model",           "ADV / impact / fee budget"),
        ("factor_budget_delta",     "Factor budget Δ",      "BARRA budget vs. live"),
        ("multi_aum_cost",          "Multi-AUM cost",       "scale-aware drag"),
        ("regime_stratified_BARRA", "Regime stratified",    "per-regime BARRA / DGU"),
        ("correlation_matrix",      "Correlation matrix",   "vs. deployed sleeves"),
        ("sub_period_robustness",   "Sub-period robust",    "split-sample stability"),
        ("block_bootstrap_significance", "Block bootstrap", "Politis-Romano p<0.05"),
        ("quarter_concentration",   "Quarter concentration","top-3 ARC drop test"),
        ("honest_deploy_sharpe",    "Honest deploy Sharpe", "OOS conservative pick"),
        ("ablation_vs_parent",      "Ablation vs parent",   "isolation test"),
        ("H7_kill_this_proposal",   "H7 kill check",        "explicit refute attempt"),
        ("devils_advocate",         "Devil's Advocate",     "LLM persona critique"),
    ]

    step_counts: dict[str, Counter] = {k: Counter() for k, _, _ in CANONICAL}
    rows_seen = 0
    total_candidates = 0
    if audit_path.is_file():
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                try:
                    r = _json.loads(s)
                except _json.JSONDecodeError:
                    continue
                rows_seen += 1
                for cand in r.get("results", []) or []:
                    total_candidates += 1
                    statuses = cand.get("step_statuses") or {}
                    for step, status in statuses.items():
                        if step in step_counts:
                            step_counts[step][status] += 1

    steps_out: list[dict] = []
    for key, label, hint in CANONICAL:
        c = step_counts[key]
        n_total = sum(c.values())
        n_pass  = c.get("PASS", 0)
        n_fail  = c.get("FAIL", 0)
        n_skip  = c.get("SKIP", 0)
        n_warn  = c.get("WARN", 0)
        n_error = c.get("ERROR", 0)
        n_evaluated = n_total - n_skip
        pass_rate = (n_pass / n_evaluated) if n_evaluated > 0 else None
        steps_out.append({
            "key":         key,
            "label":       label,
            "hint":        hint,
            "n_total":     n_total,
            "n_pass":      n_pass,
            "n_fail":      n_fail,
            "n_skip":      n_skip,
            "n_warn":      n_warn,
            "n_error":     n_error,
            "n_evaluated": n_evaluated,
            "pass_rate":   pass_rate,
        })

    return {
        "n_audit_rows":     rows_seen,
        "n_candidates":     total_candidates,
        "steps":            steps_out,
    }


@router.get("/library/{mechanism_id}/doctrine")
def library_doctrine_memory(mechanism_id: str) -> dict:
    """Return memory snippets that mention this mechanism's family.

    Closes Collab-P2 (R2.6 audit): the doctrine memory at
    ~/.claude/projects/.../memory/*.md is load-bearing for Claude's
    behavior. Quants looking at a sleeve detail had no way to see
    "what doctrines / past decisions apply to this family?". This
    endpoint scans the memory directory and surfaces relevant
    paragraphs.

    Strategy: regex-search each .md file for the mechanism's family
    name (or the mechanism_id itself, as a fallback). Return each
    hit with file name, a paragraph snippet anchored on the match,
    and the parent memory link.

    Performance note: ~50 small .md files; full scan ~5ms. No cache
    needed; freshness > speed.
    """
    from pathlib import Path
    import os
    import re

    # Resolve the memory directory. Same path the auto-memory system
    # uses (see CLAUDE.md auto-memory section).
    memory_dir = Path(os.path.expanduser(
        r"~\.claude\projects\c--Users-${USER}-Desktop-intern\memory"
    ))
    if not memory_dir.is_dir():
        return {
            "mechanism_id": mechanism_id,
            "family":       None,
            "n_hits":       0,
            "hits":         [],
            "warning":      f"memory directory not found: {memory_dir}",
        }

    # Need the YAML's family to filter on. Reuse library_mechanism_detail.
    family = None
    try:
        detail = library_mechanism_detail(mechanism_id)
        family = (detail.get("yaml") or {}).get("family")
    except Exception:
        pass

    # Search terms: family + mechanism_id (latter as fallback). Both
    # case-insensitive whole-word matches.
    needles: list[str] = []
    if family:                needles.append(family.lower())
    if mechanism_id:          needles.append(mechanism_id.lower())
    if not needles:
        return {
            "mechanism_id": mechanism_id,
            "family":       family,
            "n_hits":       0,
            "hits":         [],
            "warning":      "no family or mechanism_id to search for",
        }

    hits: list[dict] = []
    for md_path in sorted(memory_dir.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        text_lc = text.lower()
        # Cheap pre-filter
        if not any(n in text_lc for n in needles):
            continue

        # For each match, anchor a 300-char snippet
        for n in needles:
            for m in re.finditer(re.escape(n), text_lc):
                start = max(0, m.start() - 180)
                end   = min(len(text), m.end() + 220)
                snippet = text[start:end].strip()
                # Add ellipses if truncated
                if start > 0:        snippet = "… " + snippet
                if end < len(text):  snippet = snippet + " …"
                hits.append({
                    "file":       md_path.name,
                    "match_term": n,
                    "snippet":    snippet,
                    "line_no":    text[:m.start()].count("\n") + 1,
                })
                # One snippet per (file, term) — the first hit suffices;
                # second occurrence in the same file rarely adds insight
                break

    # Dedupe by (file, match_term)
    seen = set()
    deduped: list[dict] = []
    for h in hits:
        key = (h["file"], h["match_term"])
        if key in seen: continue
        seen.add(key)
        deduped.append(h)

    return {
        "mechanism_id": mechanism_id,
        "family":       family,
        "search_terms": needles,
        "n_hits":       len(deduped),
        "hits":         deduped,
    }


@router.get("/library/lifecycle")
def library_lifecycle() -> dict:
    """Strategy Lifecycle Manager state + transition history per
    library strategy_id. Powers the V3 Gantt timeline on /lab/library.

    For each strategy_id in data/strategy_lifecycle.db.strategy_state:
      - current_state, proposed_at, audited_at, paper_trade_started,
        shadow_started, live_started, decommissioned_at, allocation_pct
      - transitions: full ordered list of (from_state, to_state,
        transition_at, actor, reason) tuples for the timeline.

    Strategies present in library_inventory but NOT tracked by SLM
    get a synthetic single-state row {current_state: "UNTRACKED",
    transitions: []} so the Gantt can render a "no lifecycle" row.
    """
    from pathlib import Path
    import sqlite3
    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / "data" / "strategy_lifecycle.db"
    inv = library_inventory()["entries"]

    states: dict[str, dict] = {}
    transitions_by_id: dict[str, list[dict]] = {}

    if db_path.is_file():
        con = sqlite3.connect(str(db_path))
        try:
            for r in con.execute(
                "SELECT strategy_id, current_state, proposed_at, audited_at, "
                "approved_at, paper_trade_started, shadow_started, live_started, "
                "decommissioned_at, current_allocation_pct, target_allocation_pct, "
                "library_yaml_path FROM strategy_state"
            ):
                states[r[0]] = {
                    "strategy_id":         r[0],
                    "current_state":       r[1],
                    "proposed_at":         r[2],
                    "audited_at":          r[3],
                    "approved_at":         r[4],
                    "paper_trade_started": r[5],
                    "shadow_started":      r[6],
                    "live_started":        r[7],
                    "decommissioned_at":   r[8],
                    "current_allocation_pct": r[9],
                    "target_allocation_pct":  r[10],
                    "library_yaml_path":   r[11],
                }
            for r in con.execute(
                "SELECT strategy_id, from_state, to_state, transition_at, actor, reason "
                "FROM state_transitions ORDER BY transition_at"
            ):
                transitions_by_id.setdefault(r[0], []).append({
                    "from_state":     r[1],
                    "to_state":       r[2],
                    "transition_at":  r[3],
                    "actor":          r[4],
                    "reason":         r[5],
                })
        finally:
            con.close()

    rows: list[dict] = []
    seen_ids: set[str] = set()
    # Tracked strategies first (newest proposed first)
    tracked = sorted(states.values(),
                     key=lambda x: x.get("proposed_at") or "",
                     reverse=True)
    for s in tracked:
        sid = s["strategy_id"]
        seen_ids.add(sid)
        rows.append({**s, "transitions": transitions_by_id.get(sid, [])})

    # Library entries not tracked by SLM: synthetic UNTRACKED row
    for e in inv:
        sid = e["id"]
        if sid in seen_ids: continue
        rows.append({
            "strategy_id":       sid,
            "current_state":     "UNTRACKED",
            "proposed_at":       e.get("audit_date"),
            "audited_at":        e.get("audit_date"),
            "approved_at":       None,
            "paper_trade_started": None,
            "shadow_started":    None,
            "live_started":      None,
            "decommissioned_at": None,
            "current_allocation_pct": 0.0,
            "target_allocation_pct":  None,
            "library_yaml_path": e.get("filename"),
            "transitions":       [],
            "purpose":           e.get("purpose"),
            "family":            e.get("family"),
        })

    return {
        "n":          len(rows),
        "n_tracked":  len(tracked),
        "strategies": rows,
    }


@router.get("/library/inventory")
def library_inventory() -> dict:
    """Per-mechanism library YAML inventory for /lab/library page.

    Returns one row per mechanism YAML with identity + purpose +
    audit status fields parsed out so the UI doesn't need to parse
    YAML client-side."""
    from pathlib import Path
    import yaml as _yaml
    repo_root = Path(__file__).resolve().parent.parent
    lib_dir = repo_root / "data" / "research" / "mechanism_library"
    out: list[dict] = []
    if lib_dir.is_dir():
        for p in sorted(lib_dir.glob("*.yaml")):
            if p.name.startswith("_"):
                continue
            try:
                raw = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                out.append({"id": p.stem, "error": str(exc)[:120]})
                continue
            out.append({
                "id":            raw.get("id") or p.stem,
                "family":        raw.get("family"),
                "parent_family": raw.get("parent_family"),
                "purpose":       raw.get("purpose"),
                "canonical_paper_id": raw.get("canonical_paper_id"),
                "ca_filter_k_method": raw.get("ca_filter_k_method"),
                "audit_date":    (raw.get("audit") or {}).get("audited_date"),
                "schema_version": raw.get("_schema_version"),
                "filename":      p.name,
            })
    return {"n": len(out), "entries": out}


@router.get("/library/{mechanism_id}")
def library_mechanism_detail(mechanism_id: str) -> dict:
    """Full mechanism YAML + computed associations for
    /lab/library/[mechanism_id] detail page.

    Associations:
      - decay_history: rows from decay_sentinel_history.jsonl filtered
        to this mechanism's library_id (helps senior see "is this
        deployed sleeve decaying?")
      - graveyard_cousins: graveyard entries with the same family
      - canonical_paper: parsed from canonical_paper_id if present
    """
    from pathlib import Path
    import yaml as _yaml
    repo_root = Path(__file__).resolve().parent.parent
    lib_dir = repo_root / "data" / "research" / "mechanism_library"
    p = lib_dir / f"{mechanism_id}.yaml"
    if not p.is_file():
        raise HTTPException(status_code=404,
                            detail=f"mechanism {mechanism_id} not found")
    try:
        raw = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"YAML parse failed: {exc}")

    # Decay history for this mechanism
    decay_rows: list[dict] = []
    decay_path = repo_root / "data" / "research" / "decay_sentinel_history.jsonl"
    if decay_path.is_file():
        with decay_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("library_id") == mechanism_id:
                    decay_rows.append(r)
    decay_rows.reverse()

    # Graveyard cousins (same family)
    cousins: list[dict] = []
    family = raw.get("family")
    if family:
        graveyard_path = repo_root / "data" / "research" / "graveyard.json"
        if graveyard_path.is_file():
            try:
                gv = json.loads(graveyard_path.read_text(encoding="utf-8"))
                entries = gv.get("entries", []) if isinstance(gv, dict) else gv
                for e in entries:
                    if e.get("family") == family:
                        cousins.append(e)
            except Exception:
                pass

    return {
        "mechanism_id": mechanism_id,
        "yaml":         raw,
        "filename":     p.name,
        "decay_history": decay_rows,
        "graveyard_cousins": cousins,
    }


@router.get("/decay/sleeve/{sleeve}")
def decay_sleeve_timeline(sleeve: str) -> dict:
    """Per-sleeve decay timeline for /lab/decay/[sleeve] page.

    Returns ALL audit rows for one sleeve (chronological, oldest-first
    for charting) + summary stats."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "data" / "research" / "decay_sentinel_history.jsonl"
    rows: list[dict] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("sleeve") == sleeve:
                    rows.append(r)
    # Chronological (oldest first) for trend charting
    rows.sort(key=lambda r: r.get("audit_date") or "")
    # Summary stats
    sharpes = [r.get("trailing_sharpe") for r in rows
                if r.get("trailing_sharpe") is not None]
    alerts = [r for r in rows
               if (r.get("alert_level") or "").upper() not in ("OK", "")]
    return {
        "sleeve":    sleeve,
        "rows":      rows,
        "n_audits":  len(rows),
        "n_alerts":  len(alerts),
        "library_id": rows[0]["library_id"] if rows else None,
        "first_audit": rows[0]["audit_date"] if rows else None,
        "last_audit":  rows[-1]["audit_date"] if rows else None,
        "sharpe_min":  min(sharpes) if sharpes else None,
        "sharpe_max":  max(sharpes) if sharpes else None,
        "sharpe_last": rows[-1].get("trailing_sharpe") if rows else None,
    }


@router.get("/decay/history")
def decay_history(
    limit: int = Query(200, ge=1, le=2000),
    sleeve: Optional[str] = Query(None),
) -> dict:
    """Recent decay-sentinel audit rows from
    data/research/decay_sentinel_history.jsonl, optionally filtered to
    one sleeve. Used by /lab/decay page."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "data" / "research" / "decay_sentinel_history.jsonl"
    rows: list[dict] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if sleeve and r.get("sleeve") != sleeve:
                    continue
                rows.append(r)
    rows.reverse()
    return {"n": len(rows), "rows": rows[: max(1, int(limit))]}


# ── Frontier A (2026-06-01): per-critic calibration ────────────────


@router.get("/critic/calibration")
def critic_calibration(
    since_days: int = Query(90, ge=1, le=730),
) -> dict:
    """Per-critic accuracy + pairwise agreement + marginal-information
    gain. Backs the calibration panel; the marginal_info field is the
    "is this critic earning its keep" KPI."""
    from engine.research.critic_calibration import critic_calibration_report
    return critic_calibration_report(since_days=since_days)


@router.get("/critic/{critic_name}/accuracy")
def critic_accuracy(
    critic_name: str,
    since_days: int = Query(90, ge=1, le=730),
    family: Optional[str] = Query(None),
) -> dict:
    """One critic's accuracy, optionally filtered to one family.

    Useful for the "is theorist better on mechanism questions vs
    statistical questions" view."""
    from engine.research.critic_calibration import compute_critic_accuracy
    return compute_critic_accuracy(critic_name,
                                     since_days=since_days,
                                     family=family)


# ── Frontier 3 (2026-06-01): calibration feedback loop ──────────────


@router.get("/calibration/proposed-rules")
def calibration_proposed_rules(
    status: Optional[str] = Query(None,
        description="filter by pending / accepted / rejected"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """Queue of LLM-synthesized rules awaiting human review.

    Each entry was generated from a cluster of council_wrong iterations
    (council disagreed with the empirical pipeline ≥ N times on the
    same family). Humans accept (promote to intuition_rules.yaml
    manually) or reject (keeps the rule out + records the reason)."""
    from engine.research.calibration_feedback import read_proposed_rules
    rows = read_proposed_rules(status=status, limit=limit)
    return {"n": len(rows), "proposed_rules": rows}


class _CalibrationScanRequest(BaseModel):
    since_days:       int = 30
    min_cluster_size: int = 2
    max_synthesize:   int = 5
    confirm_cost:     bool = False


@router.post("/calibration/scan")
def calibration_scan(
    body: _CalibrationScanRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Run a calibration scan: find council_wrong → cluster → synthesize.

    Each cluster synthesized costs ~$0.02-0.10 in LLM tokens. Requires
    confirm_cost=true."""
    if not body.confirm_cost:
        raise HTTPException(
            status_code=400,
            detail="confirm_cost=true required — scan spends LLM tokens per cluster",
        )
    from engine.research.agent_council import _load_anthropic_key
    if not _load_anthropic_key():
        raise HTTPException(
            status_code=503,
            detail="no ANTHROPIC_API_KEY configured; calibration scan unavailable",
        )
    from engine.research.calibration_feedback import run_calibration_scan
    caller = x_research_caller or "unknown"
    t0 = time.perf_counter()
    try:
        out = run_calibration_scan(
            since_days=body.since_days,
            min_cluster_size=body.min_cluster_size,
            max_synthesize=body.max_synthesize,
        )
    except Exception as exc:
        logger.exception("calibration_scan failed")
        raise HTTPException(status_code=502,
                            detail=f"calibration scan failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    _append_audit("calibration_scan",
                  {"since_days": body.since_days,
                   "min_cluster_size": body.min_cluster_size,
                   "n_synthesized": out.get("n_synthesized")},
                  result_hash=None, ok=True,
                  latency_ms=latency_ms, caller=caller)
    return out


class _ReviewProposedRequest(BaseModel):
    status: str   # "accepted" | "rejected"
    note:   Optional[str] = None


@router.post("/calibration/proposed-rules/{proposal_id}/review")
def calibration_review(
    proposal_id: str,
    body: _ReviewProposedRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Mark a proposed rule as accepted or rejected.

    Accepting DOES NOT auto-write to intuition_rules.yaml — that step
    is deliberately manual so a human reviews the rule's wording
    before it enters the council's knowledge base. The status flip
    here just records the decision + closes it in the queue."""
    if body.status not in ("accepted", "rejected"):
        raise HTTPException(status_code=400,
                            detail="status must be 'accepted' or 'rejected'")
    from engine.research.calibration_feedback import review_proposed_rule
    caller = x_research_caller or "unknown"
    try:
        out = review_proposed_rule(
            proposal_id, status=body.status,
            reviewer=caller, note=body.note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("review_proposed_rule failed")
        raise HTTPException(status_code=500, detail=str(exc))
    _append_audit("calibration_review",
                  {"proposal_id": proposal_id, "status": body.status},
                  result_hash=proposal_id, ok=True,
                  latency_ms=0.0, caller=caller)
    return out


@router.get("/l4/iterations")
def l4_iterations(
    limit: int = Query(50, ge=1, le=500),
    consensus: Optional[str] = Query(None),
    alignment: Optional[str] = Query(None),
) -> dict:
    """Phase 4d: list recent L4 discovery loop iterations + calibration KPI.
    Backs the Cockpit Outcomes tab and the calibration cell in the KPI strip."""
    from engine.research.outcome_ledger import (
        calibration_summary, read_l4_iterations,
    )
    rows = read_l4_iterations(
        limit=limit, consensus=consensus, alignment=alignment,
    )
    cal = calibration_summary(limit=200)
    return {"n": len(rows), "iterations": rows, "calibration": cal}


class _PromoteRequest(BaseModel):
    iteration_id: str
    justification: str = ""


@router.post("/l4/promote")
def l4_promote_iteration(
    body: _PromoteRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Phase 4e: promote an L4 iteration to a library DRAFT yaml.

    Senior-grade promotion rules:
      - effective_consensus must be APPROVE
      - pipeline.ran must be True (no promotion of un-empirically-
        tested ideas)
      - Writes to data/research/mechanism_library/_drafts/<id>.yaml
        which library validators IGNORE (paths starting with _)
      - Senior must complete required fields + move to canonical
        directory to actually deploy
    """
    if not body.justification.strip():
        raise HTTPException(
            status_code=422,
            detail="justification text required for audit trail",
        )

    from engine.research.library_promoter import promote_iteration_to_draft
    from engine.research.outcome_ledger import read_iteration_by_id

    iteration = read_iteration_by_id(body.iteration_id)
    if iteration is None:
        raise HTTPException(
            status_code=404,
            detail=f"no L4 iteration with id={body.iteration_id!r}",
        )

    effective = iteration.get("effective_consensus")
    if effective != "APPROVE":
        raise HTTPException(
            status_code=422,
            detail=(f"iteration effective_consensus={effective!r} — "
                    "only APPROVE iterations can promote"),
        )
    pipeline = iteration.get("pipeline") or {}
    if not pipeline.get("ran"):
        raise HTTPException(
            status_code=422,
            detail=("pipeline did not run for this iteration — promotion "
                    "requires empirical pipeline_v2 result"),
        )

    result = promote_iteration_to_draft(iteration)
    _append_audit(
        "l4_promote",
        {"iteration_id":  body.iteration_id,
         "draft_path":    result["draft_path"],
         "justification": body.justification[:300]},
        result_hash=None, ok=True, latency_ms=0.0,
        caller=x_research_caller or "unknown",
    )
    return result


@router.get("/l4/iterations/{iteration_id}")
def l4_iteration_detail(iteration_id: str) -> dict:
    """Drill-down: one L4 iteration by id."""
    from engine.research.outcome_ledger import read_iteration_by_id
    row = read_iteration_by_id(iteration_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"no iteration with id={iteration_id!r}")
    return row


class _OverrideRequest(BaseModel):
    verdict: str  # APPROVE | REJECT | NEEDS_REVISION
    justification: str = ""  # audit text


@router.post("/council/workflow/{workflow_id}/pause")
def council_workflow_pause(
    workflow_id: str,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Phase 4e: send pause signal to a running L4 workflow.

    Workflow blocks at next wait_condition checkpoint. Idempotent.
    Audit-logged."""
    import asyncio
    from engine.research.l4_temporal_client import signal_pause
    try:
        result = asyncio.run(signal_pause(workflow_id))
    except Exception as exc:
        logger.exception("pause signal failed")
        raise HTTPException(status_code=502,
                            detail=f"pause failed: {exc}")
    _append_audit(
        "workflow_pause", {"workflow_id": workflow_id},
        result_hash=None, ok=True, latency_ms=0.0,
        caller=x_research_caller or "unknown",
    )
    return result


@router.post("/council/workflow/{workflow_id}/resume")
def council_workflow_resume(
    workflow_id: str,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Phase 4e: resume a paused workflow."""
    import asyncio
    from engine.research.l4_temporal_client import signal_resume
    try:
        result = asyncio.run(signal_resume(workflow_id))
    except Exception as exc:
        logger.exception("resume signal failed")
        raise HTTPException(status_code=502,
                            detail=f"resume failed: {exc}")
    _append_audit(
        "workflow_resume", {"workflow_id": workflow_id},
        result_hash=None, ok=True, latency_ms=0.0,
        caller=x_research_caller or "unknown",
    )
    return result


@router.post("/council/workflow/{workflow_id}/override")
def council_workflow_override(
    workflow_id: str,
    body: _OverrideRequest,
    x_research_caller: Optional[str] = Header(None),
) -> dict:
    """Phase 4e: human override of council verdict mid-workflow.

    verdict ∈ {APPROVE, REJECT, NEEDS_REVISION}. The workflow honours
    this instead of LLM consensus for routing (pipeline skip vs run)
    and the ledger persists both the LLM consensus AND the override
    so the audit shows the human intervention."""
    valid = {"APPROVE", "REJECT", "NEEDS_REVISION"}
    if body.verdict not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"verdict must be one of {valid}; got {body.verdict!r}",
        )
    if not body.justification.strip():
        raise HTTPException(
            status_code=422,
            detail="justification text required for audit trail",
        )

    import asyncio
    from engine.research.l4_temporal_client import signal_override_verdict
    try:
        result = asyncio.run(
            signal_override_verdict(workflow_id, body.verdict),
        )
    except Exception as exc:
        logger.exception("override signal failed")
        raise HTTPException(status_code=502,
                            detail=f"override failed: {exc}")
    _append_audit(
        "workflow_override",
        {"workflow_id": workflow_id, "verdict": body.verdict,
         "justification": body.justification[:300]},
        result_hash=None, ok=True, latency_ms=0.0,
        caller=x_research_caller or "unknown",
    )
    return result


@router.get("/council/workflow/{workflow_id}")
def council_workflow_status(workflow_id: str) -> dict:
    """Phase 4c: poll a running Temporal L4 workflow's live state.

    Returns wf_status + current stage + (once available) proposal +
    consensus. The UI polls this every 2-5s while a workflow is
    RUNNING; stops when wf_status becomes COMPLETED / FAILED.
    """
    import asyncio
    from engine.research.l4_temporal_client import query_workflow_status
    try:
        result = asyncio.run(query_workflow_status(workflow_id))
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"workflow query failed: {exc}")
    if result.get("wf_status") == "NOT_FOUND":
        raise HTTPException(status_code=404,
                            detail=f"workflow {workflow_id} not found")
    return result


@router.get("/parquets")
def list_parquets(
    include_internal: bool = Query(True),
    limit: int = Query(200, ge=1, le=1000),
) -> dict:
    """Phase Lab-Step-A: cached return-series inventory for /lab/series.
    Lightweight metadata only; the parquet body is loaded only when
    senior clicks 'Run pipeline' downstream."""
    from engine.research.parquet_browser import scan_parquets
    return scan_parquets(include_internal=include_internal, limit=limit)


@router.get("/sleeves/ca_calibration")
def sleeves_ca_calibration() -> dict:
    """Phase 5.7 follow-up: surface per-deployed-sleeve CA filter
    calibration status. Reads each library YAML, returns the
    cost_model.ca_filter_* fields.

    Used by Cockpit "Sleeve calibration" panel — senior glances at
    method=pbb_sweep_calibrated vs paper_default to know which sleeves
    are evidence-backed and which still need signal-series exposure +
    real k-sweep."""
    import yaml as _yaml
    from pathlib import Path as _P
    lib = _P("data/research/mechanism_library")
    rows: list[dict] = []
    if not lib.is_dir():
        return {"n": 0, "sleeves": []}
    for yp in sorted(lib.glob("*.yaml")):
        if yp.name.startswith("_"):
            continue
        try:
            doc = _yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        status = doc.get("status_in_our_book") or doc.get("status")
        if status not in ("DEPLOYED", "PENDING_DEPLOY"):
            continue
        cm = doc.get("cost_model") or {}
        rows.append({
            "id":                 doc.get("id", yp.stem),
            "status":             status,
            "family":             doc.get("family"),
            "ca_filter_k":        cm.get("ca_filter_k"),
            "ca_filter_k_method": cm.get("ca_filter_k_method"),
            "ca_filter_k_audit_date": cm.get("ca_filter_k_audit_date"),
            "ca_signal_type":     cm.get("ca_signal_type"),
            "tcost_round_trip_bps": cm.get("tcost_round_trip_bps"),
        })
    return {"n": len(rows), "sleeves": rows}


@router.get("/traces")
def get_traces(
    workflow_id: Optional[str] = Query(None),
    trace_id: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
) -> dict:
    """Phase 4f: read recent trace spans. Filterable by workflow_id
    (matches attrs.workflow_id) or trace_id (top-level grouping).

    Each entry is one span (or attr_update marker). Cockpit reads this
    to render the timeline view inside an iteration drill-down."""
    from engine.research.trace_log import read_spans
    spans = read_spans(
        trace_id=trace_id, workflow_id=workflow_id, limit=limit,
    )
    return {"n": len(spans), "spans": spans}


@router.get("/audit")
def recent_audit(
    limit: int = Query(50, ge=1, le=500),
    tool:  Optional[str] = Query(None),
    caller: Optional[str] = Query(None),
) -> dict:
    """Tail of the audit ledger — newest first. Filterable by tool name
    or caller identity. Cheap (last N lines from JSONL)."""
    if not AUDIT_LEDGER.is_file():
        return {"n": 0, "entries": []}

    # Read last N lines without loading the whole file — for a 100k-line
    # ledger this matters. Simple tail with seek-from-end heuristic.
    with AUDIT_LEDGER.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        # Heuristic: read up to ~256 KB to cover several hundred entries
        f.seek(max(0, size - 256 * 1024), 0)
        tail_bytes = f.read()
    text = tail_bytes.decode("utf-8", errors="replace")
    raw_lines = text.splitlines()[-(limit * 4):]  # widen for filter loss

    entries: list[dict] = []
    for line in reversed(raw_lines):  # newest first
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if tool and e.get("tool") != tool:
            continue
        if caller and e.get("caller") != caller:
            continue
        entries.append(e)
        if len(entries) >= limit:
            break

    return {"n": len(entries), "entries": entries}


# ── Chat workbench Phase 2 + 3 (2026-06-01): ledger + /ask RAG ─────


CHAT_LEDGER = REPO_ROOT / "data" / "research" / "chat_ledger.jsonl"


class _ChatLogTurn(BaseModel):
    command_id: str
    command: str
    kind: str            # response.kind (pfh_suggestions, error, ...)
    ok: bool
    summary: Optional[str] = None  # short text summary for ledger row


@router.post("/chat/log_turn")
def chat_log_turn(body: _ChatLogTurn,
                   x_research_caller: Optional[str] = Header(None)) -> dict:
    """Phase 2: append one chat turn to data/research/chat_ledger.jsonl.

    Frontend calls this after each command executes. The ledger row is
    minimal (no payload body) to keep size bounded — payloads stay in
    upstream ledgers (pfh_suggestions, council_runs, etc.)."""
    caller = x_research_caller or "ui_chat_workbench"
    try:
        CHAT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts":         _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "command_id": body.command_id,
            "command":    body.command,
            "kind":       body.kind,
            "ok":         body.ok,
            "summary":    (body.summary or "")[:300],
            "caller":     caller,
        }
        with CHAT_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return {"ok": True, "ts": row["ts"]}
    except Exception as exc:
        logger.exception("chat_log_turn failed")
        return {"ok": False, "error": str(exc)[:200]}


@router.get("/chat/recent")
def chat_recent(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Phase 2: read recent chat turns newest-first (audit surface)."""
    if not CHAT_LEDGER.is_file():
        return {"n": 0, "turns": []}
    out: list[dict] = []
    with CHAT_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    out.reverse()
    return {"n": len(out), "turns": out[: max(1, int(limit))]}


# ── Phase 3: /ask scoped LLM with RAG over ledgers ─────────────────


def _retrieve_context_for_ask(question: str, max_rows_per_ledger: int = 8) -> dict:
    """Hybrid retrieval over 11 research ledgers (was 5; AI-native
    2026-06-04 added 6 more for accuracy + dynamism: research_events,
    papers, hypotheses, doctrines, audit_lineage, graveyard_warnings).

    DYNAMISM CONTRACT (user explicitly asked: "知识库要是动态的"):
      - The keyword path re-reads every jsonl file from disk on EVERY
        call. No cache, no memoization, no warm state. New emits to
        any source become visible to chat on the NEXT question — no
        re-deploy, no re-index.
      - The semantic path uses an embedding index that CAN go stale
        when new rows are appended without re-index. Mitigation: the
        keyword path is ALWAYS unioned in, so exact-id matches and
        recent rows are still surfaced even if the index is stale.
        Re-indexing is a separate cron (engine.research.embeddings).
      - Doctrines (CLAUDE.md / AGENTS.md) are re-read from disk on
        every call too, so CLAUDE.md edits land in chat instantly.

    Returns dict of ledger_name → list of small payload dicts. Each
    payload is minimized (key fields only) so the LLM prompt stays
    under token budget.

    Order of preference (per ledger):
      1. Semantic search (MiniLM-L6-v2) via engine.research.embeddings
         if the per-ledger index exists.
      2. Keyword + recency fallback (the previous MVP path) when the
         index is missing or fails to import (e.g. sentence-transformers
         not installed in a deployment).

    Why hybrid: vector recall solves synonym misses (e.g. "trailing
    performance" → rows with "trailing_sharpe"), but the keyword path
    is still valuable when the user names an exact id ("the abc123
    run") — those are surface-form matches not semantic ones, so we
    union the two when both are available.
    """
    qlow = question.lower()
    repo = Path(__file__).resolve().parent.parent

    # ── Semantic path ──
    semantic: dict[str, list[dict]] = {}
    try:
        from engine.research import embeddings as _E
        status = _E.index_status()
        if any(s.get("indexed") for s in status.values()):
            semantic = _E.search_all(question, top_k=max_rows_per_ledger)
    except Exception:
        # Any failure (missing model, file IO, torch not loadable) →
        # silently fall through to keyword. Logged once per process by
        # the caller surface.
        logger.warning("semantic retrieval unavailable, falling back to keyword",
                       exc_info=True)
        semantic = {}

    def _load_jsonl(path: Path, n: int = max_rows_per_ledger) -> list[dict]:
        if not path.is_file():
            return []
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        out.reverse()
        return out[:n]

    def _filter_keyword(rows: list[dict], keys: list[str],
                        *, ledger_name: str = "") -> list[dict]:
        """Filter rows whose stringified content overlaps with question
        keywords. Falls back to recency if no overlap.

        2026-06-04: added RECENCY BOOST per user feedback "知识库要是动态的".
        Final score = keyword_overlap + recency_weight, where the weight
        decays with age. A row from today out-scores a row from 6 months
        ago with the same keyword overlap. This matters because the user
        is emitting new events continuously and the answer to "today's
        question" should lean on today's data, not the all-time max-
        relevance row.

        T2.3 (2026-06-05 audit R2 fix): added LEDGER-NAME BOOST.
        When the question mentions a ledger name ("decay_audits",
        "audit_lineage", "graveyard") or any of the ledger's identifying
        terms (the `keys` arg, e.g. ["proposal"] for l4_iterations),
        every row in that ledger gets +LEDGER_NAME_BOOST. Pre-T2.3 the
        retriever treated ledger names as generic tokens; if the user
        asked "what's in decay_audits", rows from decay_audits would
        rarely contain the literal string "decay_audits" (rows describe
        individual events) and the retriever would miss the obvious
        intent.

        Recency uses any of {ts, verified_ts, created_ts, checked_ts,
        filed_ts, materialized_at} present on the row.
        """
        tokens = {t for t in re.findall(r"\w{3,}", qlow)
                   if t not in ("the", "what", "how", "are", "did", "this",
                                 "that", "with", "from", "for", "and", "our")}

        # T2.3: ledger-name boost. Build the set of tokens that, when
        # mentioned in the question, indicate the user is asking ABOUT
        # this ledger (rather than asking about content that happens to
        # live in it). Variants cover snake_case, space-separated, and
        # singular/plural forms.
        ledger_match_tokens: set[str] = set()
        for t in [ledger_name] + list(keys or []):
            if not t:
                continue
            t_low = t.lower().strip()
            ledger_match_tokens.add(t_low)
            ledger_match_tokens.add(t_low.replace("_", " "))
            ledger_match_tokens.add(t_low.replace("_", ""))
            if t_low.endswith("s"):
                ledger_match_tokens.add(t_low[:-1])
            else:
                ledger_match_tokens.add(t_low + "s")
        ledger_name_match = any(t in qlow for t in ledger_match_tokens if t)
        LEDGER_NAME_BOOST = 2.0
        # Precompute "now" once; expensive ops outside the loop.
        import datetime as _dt
        now = _dt.datetime.utcnow()
        TS_FIELDS = ("ts", "verified_ts", "created_ts", "checked_ts",
                     "filed_ts", "materialized_at", "audit_date")

        def _recency_weight(row: dict) -> float:
            """0..1.5 score. Today = 1.5, 7d = 1.0, 30d = 0.6, 180d = 0.2,
            unknown timestamp = 0.3 (neutral)."""
            ts_str = None
            for f in TS_FIELDS:
                v = row.get(f)
                if v:
                    ts_str = str(v); break
            if not ts_str:
                return 0.3
            try:
                t = _dt.datetime.fromisoformat(ts_str.rstrip("Z")[:19])
            except Exception:
                return 0.3
            days = max(0.0, (now - t).total_seconds() / 86400.0)
            if days <= 1:    return 1.5
            if days <= 7:    return 1.0
            if days <= 30:   return 0.6
            if days <= 180:  return 0.3
            return 0.15

        if not tokens:
            # No question keywords — pure recency sort
            scored = [(_recency_weight(r), r) for r in rows]
            scored.sort(key=lambda x: -x[0])
            return [r for _, r in scored[:max_rows_per_ledger]]

        scored: list[tuple[float, dict]] = []
        for r in rows:
            blob = json.dumps(r, default=str).lower()
            overlap = sum(1 for t in tokens if t in blob)
            # Score = keyword overlap + recency + ledger-name boost (T2.3).
            score = float(overlap) + _recency_weight(r)
            if ledger_name_match:
                score += LEDGER_NAME_BOOST
            # Admit a row if: keyword hit, recent enough, OR the user
            # explicitly named this ledger (T2.3 — surface ledger
            # contents when the question is ABOUT the ledger).
            if overlap > 0 or _recency_weight(r) >= 1.0 or ledger_name_match:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return rows[:max_rows_per_ledger]
        return [r for _, r in scored[:max_rows_per_ledger]]

    # 1. L4 iterations (proposal + council + pipeline)
    l4_rows = _load_jsonl(repo / "data" / "research" / "l4_iterations.jsonl",
                          n=50)
    l4_filtered = _filter_keyword(l4_rows, ["proposal", "l4", "iteration"],
                                  ledger_name="l4_iterations")

    # 2. PFH suggestions
    pfh_rows = _load_jsonl(repo / "data" / "research" / "pfh_suggestions.jsonl",
                            n=20)
    pfh_filtered = _filter_keyword(pfh_rows, ["top", "pfh", "suggestion"],
                                   ledger_name="pfh_suggestions")

    # 3. Council runs (truncated — full text bodies are big)
    council_rows = _load_jsonl(repo / "data" / "research" / "council_runs.jsonl",
                                 n=30)
    council_filtered = []
    for r in _filter_keyword(council_rows, ["consensus", "council", "critique"],
                              ledger_name="council_critiques")[:6]:
        # Keep small subset of fields
        council_filtered.append({
            "run_id":    r.get("run_id"),
            "ts":        r.get("ts"),
            "stage":     r.get("stage"),
            "consensus": r.get("consensus"),
            "proposal_title":  (r.get("proposal") or {}).get("title"),
            "proposal_family": (r.get("proposal") or {}).get("family"),
            "rationale": (r.get("rationale") or "")[:300],
        })

    # 4. Decay sentinel history
    decay_rows = _load_jsonl(repo / "data" / "research" / "decay_sentinel_history.jsonl",
                              n=30)
    decay_filtered = _filter_keyword(decay_rows, ["sleeve", "decay", "sentinel"],
                                      ledger_name="decay_audits")

    # 5. Feature store materializations — scan meta files
    materializations: list[dict] = []
    computed_dir = repo / "data" / "feature_store" / "_computed"
    if computed_dir.is_dir():
        for p in sorted(computed_dir.glob("*.meta.json"))[:30]:
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                materializations.append({
                    "spec_id":          m.get("spec_id"),
                    "materialized_at":  m.get("materialized_at"),
                    "validation":       m.get("validation"),
                    "compose_axes":     m.get("compose_axes"),
                })
            except Exception:
                pass
    mat_filtered = _filter_keyword(materializations, ["spec_id", "materialize"],
                                    ledger_name="materializations")

    # 6. research_store events (factor_verdict_filed + lessons + everything
    #    typed via engine.research_store.emit.*). This is the canonical
    #    record of WHAT HAPPENED in research; previously absent from chat
    #    context so chat could not answer "what's the latest CARRY lesson"
    #    or "how many RED verdicts on family X". Each row is reduced to
    #    its essential fields to fit prompt budget.
    rs_events_raw = _load_jsonl(repo / "data" / "research_store" / "events.jsonl",
                                 n=80)
    rs_events: list[dict] = []
    for r in rs_events_raw:
        rs_events.append({
            "event_id":     r.get("event_id"),
            "event_type":   r.get("event_type"),
            "ts":           r.get("ts"),
            "subject_id":   r.get("subject_id"),
            "subject_type": r.get("subject_type"),
            "verdict":      r.get("verdict"),
            "family":       r.get("family"),
            "summary":      (r.get("summary") or "")[:300],
        })
    rs_events_filtered = _filter_keyword(rs_events, ["verdict", "factor", "event"],
                                         ledger_name="research_events")

    # 7. papers_registry — every ingested paper, its title + DOI + shelves.
    #    Allows chat to answer "what papers do we have on X" without
    #    guessing.
    papers_raw = _load_jsonl(repo / "data" / "research_store" / "papers_registry.jsonl",
                              n=80)
    papers: list[dict] = []
    for r in papers_raw:
        papers.append({
            "paper_id":   r.get("paper_id"),
            "doi":        r.get("doi"),
            "title":      r.get("title"),
            "year":       r.get("year"),
            "shelves":    r.get("shelves"),
            "abstract":   (r.get("abstract") or "")[:250],
        })
    papers_filtered = _filter_keyword(papers, ["paper", "library", "literature"],
                                       ledger_name="papers")

    # 8. hypotheses — claims extracted from papers. Drives /research/forward
    #    and the direction_proposer. Chat needs these to answer "which
    #    untested CARRY hypothesis is highest priority".
    hyp_raw = _load_jsonl(repo / "data" / "research_store" / "hypotheses.jsonl",
                           n=120)
    hyps: list[dict] = []
    for r in hyp_raw:
        hyps.append({
            "hypothesis_id":      r.get("hypothesis_id"),
            "source_paper_id":    r.get("source_paper_id"),
            "claim":              (r.get("claim") or "")[:280],
            "mechanism_family":   r.get("mechanism_family"),
            "mechanism_subtype":  r.get("mechanism_subtype"),
            "predicted_direction": (r.get("predicted_direction") or {}) if isinstance(r.get("predicted_direction"), dict) else r.get("predicted_direction"),
        })
    hyps_filtered = _filter_keyword(hyps, ["claim", "hypothesis", "hypotheses"],
                                     ledger_name="hypotheses")

    # 9. doctrine — CLAUDE.md (project root) + frontend/AGENTS.md +
    #    MEMORY.md index. These are the WORLD RULES (PAPER→HYPOTHESIS→
    #    TEST→VERDICT chain, family-aware n_trials, Session Protocol).
    #    Without them in context chat doesn't know your strict-gate rules.
    doctrines: list[dict] = []
    for rel_path in ("CLAUDE.md",
                     "frontend/CLAUDE.md",
                     "frontend/AGENTS.md"):
        p = repo / rel_path
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        # Chunk into ~600-char paragraphs; keep paragraph order via index
        paras = [chunk.strip() for chunk in txt.split("\n\n") if chunk.strip()]
        for i, para in enumerate(paras):
            doctrines.append({
                "doctrine_id": f"{rel_path}#p{i}",
                "source":      rel_path,
                "text":        para[:600],
            })
    doctrines_filtered = _filter_keyword(doctrines, ["doctrine", "memory", "lesson"],
                                         ledger_name="doctrines")[:max_rows_per_ledger * 2]

    # 10. audit_verifier lineage results — the reactive subscriber that
    #     checks every factor_verdict_filed event. Chat should be able to
    #     answer "any recent WARN/FAIL on lineage?". DYNAMIC: re-read on
    #     every query so new agent emits are visible without re-deploy.
    av_raw = _load_jsonl(repo / "data" / "audit_verifier" / "lineage_results.jsonl",
                          n=40)
    av_rows: list[dict] = []
    for r in av_raw:
        av_rows.append({
            "audit_id":           r.get("audit_id"),
            "verified_ts":        r.get("verified_ts"),
            "research_event_id":  r.get("research_event_id"),
            "subject_id":         r.get("subject_id"),
            "family":             r.get("family"),
            "verdict":            r.get("verdict"),
            "checks":             [{"check": c.get("check"), "status": c.get("status")}
                                   for c in (r.get("checks") or [])],
        })
    av_filtered = _filter_keyword(av_rows, ["audit", "lineage", "verifier"],
                                   ledger_name="audit_lineage")

    # 11. graveyard_collision warnings — fired when an intent is filed
    #     with a candidate that looks like a past RED. Chat should be
    #     able to answer "did anything just collide with the graveyard?".
    gc_raw = _load_jsonl(repo / "data" / "graveyard_collision" / "warnings.jsonl",
                          n=40)
    gc_rows: list[dict] = []
    for r in gc_raw:
        gc_rows.append({
            "warning_id":     r.get("warning_id"),
            "checked_ts":     r.get("checked_ts"),
            "intent_id":      r.get("intent_id"),
            "candidate_name": r.get("candidate_name"),
            "family":         r.get("family"),
            "subtype":        r.get("subtype"),
            "verdict":        r.get("verdict"),
            "reason":         r.get("reason"),
            "n_scanned":      r.get("n_scanned"),
        })
    gc_filtered = _filter_keyword(gc_rows, ["graveyard", "collision", "red"],
                                   ledger_name="graveyard_warnings")

    keyword_ctx = {
        "l4_iterations":   l4_filtered,
        "pfh_suggestions": pfh_filtered,
        "council_runs":    council_filtered,
        "decay_audits":    decay_filtered,
        "materializations": mat_filtered,
        # AI-native 2026-06-04: 6 new sources for accuracy + dynamism
        "research_events":   rs_events_filtered,
        "papers":            papers_filtered,
        "hypotheses":        hyps_filtered,
        "doctrines":         doctrines_filtered,
        # Reactive-agent outputs — fresh on every query (no cache)
        "audit_lineage":     av_filtered,
        "graveyard_warnings": gc_filtered,
    }

    if not semantic:
        return keyword_ctx

    # Union: semantic first (best ranked), then keyword rows not yet
    # represented. Dedup keys per-ledger differ — use a tuple of the
    # natural identifier where one exists, else stringified payload.
    def _natural_key(ledger: str, row: dict) -> str:
        if ledger == "l4_iterations":
            return f"l4:{row.get('iteration_id') or row.get('ts')}"
        if ledger == "council_runs":
            return f"run:{row.get('run_id') or row.get('ts')}"
        if ledger == "decay_audits":
            return f"d:{row.get('sleeve')}:{row.get('audit_date')}"
        if ledger == "pfh_suggestions":
            return f"pfh:{row.get('ts')}"
        if ledger == "materializations":
            return f"mat:{row.get('spec_id')}:{row.get('materialized_at')}"
        # AI-native 2026-06-04 new ledgers
        if ledger == "research_events":
            return f"ev:{row.get('event_id')}"
        if ledger == "papers":
            return f"p:{row.get('paper_id')}"
        if ledger == "hypotheses":
            return f"h:{row.get('hypothesis_id')}"
        if ledger == "doctrines":
            return f"doc:{row.get('doctrine_id')}"
        if ledger == "audit_lineage":
            return f"av:{row.get('audit_id')}"
        if ledger == "graveyard_warnings":
            return f"gc:{row.get('warning_id')}"
        return json.dumps(row, default=str, sort_keys=True)[:200]

    merged: dict[str, list[dict]] = {}
    for ledger in ("l4_iterations", "pfh_suggestions", "council_runs",
                   "decay_audits", "materializations",
                   "research_events", "papers", "hypotheses", "doctrines",
                   "audit_lineage", "graveyard_warnings"):
        out: list[dict] = []
        seen: set[str] = set()
        for r in semantic.get(ledger, []):
            k = _natural_key(ledger, r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        # Top up from keyword results to fill quota — surfaces exact-id
        # matches the semantic ranker can miss.
        for r in keyword_ctx.get(ledger, []):
            if len(out) >= max_rows_per_ledger:
                break
            k = _natural_key(ledger, r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        merged[ledger] = out
    return merged


_ASK_SYSTEM_PROMPT = """\
You are a SENIOR QUANT RESEARCH ASSISTANT with READ access to the
user's research ledgers. Your job: answer the user's question USING
ONLY the provided context.

STRICT RULES:
  1. NEVER invent facts not in the context. If the answer requires
     information NOT in the context, say "I don't have that in the
     ledgers" and suggest which ledger might contain it.
  2. CITE specific entity IDs when referring to specific items.
     Format citations as `[type:id]` so the UI can render them as
     links AND verify against the store. Valid types:
       run_id          — council_runs row
       iteration_id    — l4_iterations row
       spec_id         — materializations / spec registry
       sleeve          — deployed mechanism (library YAML)
       event_id        — research_store events.jsonl (verdicts, etc)
       paper_id        — papers_registry.jsonl
       hypothesis_id   — hypotheses.jsonl
       doctrine        — CLAUDE.md / AGENTS.md chunk (e.g. doctrine:CLAUDE.md#p4)
     Example: "Per the council's APPROVE on [run_id:abc123def456], ..."
  3. Be CONCISE. Senior quants don't want padding. 2-3 short
     paragraphs maximum unless the user explicitly asked for detail.
  4. NEVER claim a number you can compute yourself. If asked for a
     Sharpe or other stat, quote it verbatim from the ledger row.
  5. If multiple ledger rows are relevant, name them all (do not
     cherry-pick).
  6. Doctrines (CLAUDE.md / AGENTS.md chunks) are the project RULE
     BOOK. When a question concerns a threshold, gate, or procedure,
     QUOTE the doctrine verbatim before paraphrasing. Wrong rule >
     no rule, so don't guess.
  7. If you cannot find a citation for a factual claim, mark the
     claim with `[unverified]` instead of leaving it bare. The user
     prefers an honest "I don't know" over a confident invention.

FIELD REFERENCE for each ledger (so you know what's there without
having to scan every row):

  doctrines         { doctrine_id, source, text }
                      source: CLAUDE.md | frontend/CLAUDE.md | frontend/AGENTS.md
                      doctrine_id is "source#pN" where N is paragraph index
  research_events   { event_id, event_type, ts, subject_id, subject_type,
                      verdict, family, summary }
                      event_type in: factor_verdict_filed, capability_evidence_filed,
                      memory_doctrine_locked, spec_amended, deploy_changed,
                      decay_alert, dq_breach, council_critique
                      verdict in: GREEN, MARGINAL, RED, NEUTRAL
  papers            { paper_id, doi, title, year, shelves, abstract }
                      shelves in: doctrine_method, green_motivation,
                      yellow_motivation, green_critique, red_motivation, red_critique
  hypotheses        { hypothesis_id, source_paper_id, claim,
                      mechanism_family, mechanism_subtype, predicted_direction }
  audit_lineage     { audit_id, verified_ts, research_event_id, subject_id,
                      family, verdict, checks }
                      verdict in: CLEAN, WARN, FAIL, SKIP
                      checks list each have {check: C1..C4, status: PASS/WARN/FAIL}
  graveyard_warnings{ warning_id, checked_ts, intent_id, candidate_name,
                      family, subtype, verdict, reason, n_scanned }
                      verdict in: CLEAN, WARN, RISK, SKIP
  decay_audits      { sleeve, audit_date, trailing_sharpe, ... }
  materializations  { spec_id, materialized_at, validation, compose_axes }
  council_runs      { run_id, ts, stage, consensus, proposal_title,
                      proposal_family, rationale }
  pfh_suggestions   pfh-loop top suggestions per audit window
  l4_iterations     L4 loop full iterations (proposal + council + pipeline)
"""


class _AskRequest(BaseModel):
    question: str
    confirm_cost: bool = False
    # 2026-06-02 — session-aware /ask. When session_id is provided,
    # prior turns are loaded as conversational context and the new
    # exchange is appended. When omitted, a new session is created and
    # the id is returned so the caller (Cmd-K Ask, side-panel chat,
    # /chat page) can stitch follow-up turns into the same thread.
    session_id: Optional[str] = None
    # Commit Y 2026-06-04 — caller passes the user's CURRENT PAGE
    # context as an authoritative source. Goes into the system prompt
    # tail, NOT into the user-message turn list (so it can't be
    # overridden by a chatty follow-up). Set by HelpOnThisPage and
    # CmdK Ask; null when chat is opened generically.
    page_context: Optional[str] = None


# ── Citation verifier (Commit Z 2026-06-04) ─────────────────────


def _verify_citations(citations: list[dict]) -> list[dict]:
    """Mark each citation with exists=True/False based on whether the
    referenced id resolves in the corresponding source. Caps at 50 ids
    total (longer answers are rare). Returns the input list with an
    extra 'exists' key per item; no removal.
    """
    if not citations:
        return citations
    citations = citations[:50]
    repo = Path(__file__).resolve().parent.parent

    # Lazy index loaders — one per type. Each returns a set of valid ids.
    _cache: dict[str, set] = {}

    def _ids_from_jsonl(rel_path: str, key: str) -> set:
        p = repo / rel_path
        if not p.is_file():
            return set()
        out: set[str] = set()
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        v = row.get(key)
                        if v:
                            out.add(str(v))
                    except Exception:
                        continue
        except Exception:
            logger.warning("verifier: failed to scan %s for key=%s", rel_path, key)
        return out

    def _load(typ: str) -> set:
        if typ in _cache:
            return _cache[typ]
        s: set = set()
        if   typ == "event_id":
            s = _ids_from_jsonl("data/research_store/events.jsonl",       "event_id")
        elif typ == "paper_id":
            s = _ids_from_jsonl("data/research_store/papers_registry.jsonl", "paper_id")
        elif typ == "hypothesis_id":
            s = _ids_from_jsonl("data/research_store/hypotheses.jsonl",   "hypothesis_id")
        elif typ == "audit_id":
            s = _ids_from_jsonl("data/audit_verifier/lineage_results.jsonl", "audit_id")
        elif typ == "warning_id":
            s = _ids_from_jsonl("data/graveyard_collision/warnings.jsonl",   "warning_id")
        elif typ == "run_id":
            s = _ids_from_jsonl("data/research/council_runs.jsonl",       "run_id")
        elif typ == "iteration_id":
            s = _ids_from_jsonl("data/research/l4_iterations.jsonl",      "iteration_id")
        elif typ == "spec_id":
            # spec_id lives in materializations meta files + the spec registry
            computed = repo / "data" / "feature_store" / "_computed"
            if computed.is_dir():
                for p in computed.glob("*.meta.json"):
                    try:
                        m = json.loads(p.read_text(encoding="utf-8"))
                        v = m.get("spec_id")
                        if v:
                            s.add(str(v))
                    except Exception:
                        continue
        elif typ == "sleeve":
            lib = repo / "data" / "research" / "mechanism_library"
            if lib.is_dir():
                for p in lib.glob("*.yaml"):
                    s.add(p.stem)
        elif typ == "doctrine":
            # doctrine ids look like "CLAUDE.md#p4" — verify by checking
            # the file exists. The chunk index is paragraph-based and we
            # don't validate the #pN part beyond non-negativity.
            for rel in ("CLAUDE.md", "frontend/CLAUDE.md", "frontend/AGENTS.md"):
                if (repo / rel).is_file():
                    s.add(rel)
        _cache[typ] = s
        return s

    for c in citations:
        typ = c.get("type", "")
        cid = c.get("id", "")
        try:
            if typ == "doctrine":
                # doctrine ids = "<source>#p<N>" — split + check source file
                if "#" in cid:
                    src = cid.split("#", 1)[0]
                else:
                    src = cid
                c["exists"] = src in _load("doctrine")
            else:
                c["exists"] = cid in _load(typ)
        except Exception:
            c["exists"] = False
    return citations


# Session storage — one JSONL per session under data/research/chat_sessions/.
# Each line is one exchange: {ts, question, answer, citations, retrieval_mode}.
CHAT_SESSION_DIR = REPO_ROOT / "data" / "research" / "chat_sessions"


def _derive_session_title(first_question: Optional[str]) -> Optional[str]:
    """Build a short human-readable title from the FIRST question of a
    session, the same way Claude.ai / ChatGPT label their threads in
    the side panel. Pure deterministic; cheap; safe on Chinese + English
    + mixed.

    Pipeline:
      1. Strip leading slash command ("/explain foo" -> "foo")
      2. Trim whitespace + collapse runs
      3. Strip a handful of leading interrogative stems for English so
         "What is the DSR threshold?" -> "the DSR threshold"
      4. Cut at the first sentence-end punctuation if there is one
         within 60 chars (so we don't include "?" / "." / "。")
      5. Hard-cap at 48 chars for Latin, 22 chars for CJK heavy
      6. Strip trailing punctuation
    Returns None if input is empty.
    """
    if not first_question:
        return None
    s = first_question.strip()
    if not s:
        return None
    # 1. Leading slash command
    if s.startswith("/"):
        sp = s.find(" ")
        s = s[sp + 1:].strip() if sp > 0 else ""
    if not s:
        return None
    # 2. Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # 3. Leading interrogative trim (English only — short list, safe)
    LEAD = ("what is the ", "what's the ", "what is ", "what's ",
            "how do i ", "how do we ", "how can i ", "how do you ",
            "why does ", "why is ", "tell me about ", "explain ",
            "can you ", "could you ", "please ")
    low = s.lower()
    for prefix in LEAD:
        if low.startswith(prefix):
            s = s[len(prefix):]
            break
    # 4. Cut at sentence-end punctuation within first 60 chars
    cut = -1
    for i, ch in enumerate(s[:60]):
        if ch in ".?!。？！":
            cut = i; break
    if cut > 0:
        s = s[:cut]
    # 5. Hard cap (mixed CJK / Latin) — count CJK chars 2x to roughly
    #    approximate display width
    width = 0
    out_chars: list[str] = []
    for ch in s:
        w = 2 if ord(ch) > 0x2E80 else 1
        if width + w > 48:
            break
        out_chars.append(ch)
        width += w
    s = "".join(out_chars).strip()
    # 6. Strip trailing punctuation / whitespace
    s = s.rstrip(" .?!。？！,，:：")
    # 7. First-character capitalize for English-looking titles
    if s and s[0].isascii() and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return s or None


def _new_session_id() -> str:
    import uuid as _uuid
    return _uuid.uuid4().hex[:14]


def _session_path(session_id: str) -> Path:
    # Keep ids hex-only so we can map them to filenames without escaping.
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe:
        safe = _new_session_id()
    return CHAT_SESSION_DIR / f"{safe}.jsonl"


def _read_session_turns(session_id: str, *, limit: int = 12) -> list[dict]:
    """Return last `limit` turns of a session, newest LAST (chronological)
    so they can be concatenated into the LLM prompt directly. Best-effort
    — corrupt rows are skipped."""
    p = _session_path(session_id)
    if not p.is_file():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-max(1, int(limit)):]


def _append_session_turn(session_id: str, turn: dict) -> None:
    """Append one exchange to the session file. Best-effort."""
    try:
        CHAT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        with _session_path(session_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, default=str) + "\n")
    except Exception:
        logger.exception("session turn append failed (non-fatal)")


@router.post("/chat/ask")
def chat_ask(body: _AskRequest,
              x_research_caller: Optional[str] = Header(None)) -> dict:
    """Phase 3: /ask scoped LLM with RAG over local ledgers.

    Cost gate: confirm_cost=true required (each query is ~$0.01-0.05
    in Anthropic tokens). Frontend sets this automatically when user
    types /ask, so the gate is informational not blocking.

    Retrieval: keyword + recency over 5 ledgers (l4_iterations,
    pfh_suggestions, council_runs, decay_history, materializations).
    No vector store. See _retrieve_context_for_ask senior note.

    Returns: {answer, citations (parsed from [type:id] markers),
    n_context_rows per ledger, model, elapsed_s}."""
    if not body.question.strip():
        raise HTTPException(status_code=422,
                            detail="question must be non-empty")
    if not body.confirm_cost:
        raise HTTPException(
            status_code=400,
            detail="confirm_cost=true required (LLM token spend)")

    # Anthropic key check
    try:
        from engine.research.agent_council import _load_anthropic_key
    except ImportError:
        raise HTTPException(status_code=503,
                            detail="agent_council module not available")
    key = _load_anthropic_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="no ANTHROPIC_API_KEY in env or .streamlit/secrets.toml")

    caller = x_research_caller or "ui_chat_workbench"
    t0 = time.perf_counter()

    # Retrieve context
    context = _retrieve_context_for_ask(body.question)
    n_rows = {k: len(v) for k, v in context.items()}

    # T2.4 (2026-06-05 audit R3 fix): zero-context circuit breaker.
    # If EVERY ledger returned 0 rows, the LLM has nothing to ground
    # its answer in and will confabulate. Short-circuit with a typed
    # "no_context_found" response instructing the user how to narrow,
    # WITHOUT calling the LLM (saves tokens, prevents hallucination).
    total_ctx_rows = sum(n_rows.values())
    if total_ctx_rows == 0:
        elapsed_s = time.perf_counter() - t0
        session_id_zc = body.session_id or _new_session_id()
        no_ctx_answer = (
            "I couldn't find any local ledger rows matching your question. "
            "Without grounded context, I won't speculate. To get a useful "
            "answer, try one of:\n"
            "  - Name a ledger explicitly: \"in decay_audits, ...\" / "
            "\"in research_events, ...\" / \"in graveyard_warnings, ...\"\n"
            "  - Include a specific ID: spec_hash, paper_id, "
            "hypothesis_id, event_id, or sleeve_id\n"
            "  - Be more concrete about the time window or factor family\n"
            "Available ledgers: doctrines, research_events, papers, "
            "hypotheses, audit_lineage, graveyard_warnings, decay_audits, "
            "materializations, pfh_suggestions, council_runs, l4_iterations."
        )
        zc_row = {
            "session_id":      session_id_zc,
            "ts":              _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "question":        body.question,
            "answer":          no_ctx_answer,
            "n_context_rows":  n_rows,
            "status":          "no_context_found",
            "short_circuit":   "T2.4_zero_context",
            "elapsed_s":       round(elapsed_s, 3),
            "caller":          caller,
        }
        _append_session_turn(session_id_zc, zc_row)
        logger.info("chat_ask: zero-context short-circuit q=%r", body.question[:80])
        return {
            "answer":         no_ctx_answer,
            "citations":      [],
            "n_context_rows": n_rows,
            "model":          "(short-circuit, no LLM call)",
            "elapsed_s":      round(elapsed_s, 3),
            "session_id":     session_id_zc,
            "status":         "no_context_found",
            "short_circuit": "T2.4_zero_context",
        }

    # Cheap probe: if any context row carries a _semantic_score, the
    # semantic path participated. (False positives possible if the
    # ledger is genuinely empty; that's fine — informational only.)
    retrieval_mode = "semantic+keyword" if any(
        any("_semantic_score" in r for r in v)
        for v in context.values()
    ) else "keyword"

    # Build user prompt. Order matters: put the most authoritative +
    # token-cheap sources FIRST so they survive any truncation. Doctrines
    # are the RULE BOOK — they always lead. Then concrete state (events,
    # papers, hypotheses, decay). l4 + council are verbose so they trail.
    ordered_context = {
        "doctrines":          context.get("doctrines", []),
        "research_events":    context.get("research_events", []),
        "papers":             context.get("papers", []),
        "hypotheses":         context.get("hypotheses", []),
        "audit_lineage":      context.get("audit_lineage", []),
        "graveyard_warnings": context.get("graveyard_warnings", []),
        "decay_audits":       context.get("decay_audits", []),
        "materializations":   context.get("materializations", []),
        "pfh_suggestions":    context.get("pfh_suggestions", []),
        "council_runs":       context.get("council_runs", []),
        "l4_iterations":      context.get("l4_iterations", []),
    }

    # X.1a 2026-06-04 — cost discipline. Per-question budget caps the
    # context payload and a compact JSON encoder strips whitespace.
    # Together they shave 60-70% input tokens vs the previous
    # indent=2 + 40k flat cap.
    #
    # Budget heuristic:
    #   - short question (<80 chars) and no explicit ID  -> 8k chars
    #   - long question (>=80 chars) OR contains a token
    #     that looks like an id (uuid hex or _ prefix)   -> 20k chars
    #   - default                                        -> 12k chars
    q_text = body.question.strip()
    q_len  = len(q_text)
    looks_like_id = bool(re.search(r"\b[a-f0-9]{8,}\b|[a-z_]+_id", q_text.lower()))
    if q_len < 80 and not looks_like_id:
        ctx_budget = 8_000
    elif q_len >= 80 or looks_like_id:
        ctx_budget = 20_000
    else:
        ctx_budget = 12_000
    compact_json = json.dumps(ordered_context, default=str,
                              separators=(",", ":"))[:ctx_budget]

    user_msg = (
        f"QUESTION: {q_text}\n\n"
        f"CONTEXT (11 ledger slices, recency-weighted, "
        f"{len(compact_json)} chars / budget {ctx_budget}):\n"
        f"```json\n{compact_json}\n```\n\n"
        f"Answer using ONLY the context above. Cite specific IDs as "
        f"`[type:id]`. Doctrines are the RULE BOOK (cite as "
        f"`[doctrine:CLAUDE.md#p4]`); research_events carry verdicts "
        f"(cite as `[event_id:uuid]`); papers + hypotheses ground every "
        f"factor claim (cite as `[paper_id:...]` / `[hypothesis_id:...]`). "
        f"If the question is about a numeric threshold or strict-gate "
        f"rule, QUOTE the doctrine verbatim — don't paraphrase."
    )

    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=503,
                            detail="anthropic SDK not installed")
    # Multi-turn: when session_id is provided, include the last few
    # exchanges as conversational history (alternating user/assistant).
    # Cap at the most recent 6 exchanges so older context doesn't push
    # ledger retrieval out of the prompt budget.
    session_id = body.session_id or _new_session_id()
    is_new_session = not (body.session_id and _session_path(body.session_id).is_file())
    prior_turns = _read_session_turns(session_id, limit=6) if not is_new_session else []
    msgs: list[dict] = []
    for t in prior_turns:
        q = (t.get("question") or "").strip()
        a = (t.get("answer") or "").strip()
        if q:
            msgs.append({"role": "user", "content": q})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})

    # Commit Y 2026-06-04 — page context lifted into the system prompt
    # tail so it functions as an authoritative source (immutable across
    # follow-up turns) instead of a one-shot user message. Bounded to
    # ~800 chars; longer is wasted tokens for the same orientation.
    sys_prompt = _ASK_SYSTEM_PROMPT
    if body.page_context:
        pc = body.page_context.strip()[:800]
        sys_prompt = (
            sys_prompt
            + "\n\n"
            + "USER'S CURRENT PAGE (authoritative — treat as fact about "
              "the user's session, not a user claim):\n"
            + pc
        )

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            temperature=0.2,
            system=sys_prompt,
            messages=msgs,
        )
    except Exception as exc:
        logger.exception("chat_ask LLM call failed")
        raise HTTPException(status_code=502,
                            detail=f"LLM call failed: {exc}")

    answer = "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text").strip()
    elapsed_s = time.perf_counter() - t0

    # 2026-06-02 — close the chat-cost monitoring gap. Every /chat/ask
    # turn now appears in /ops cost panel under agent_id="chat_ask".
    # Failures here MUST NOT break the user-facing answer, so wrap defensively.
    try:
        from engine.llm.pricing import compute_cost
        from engine.llm_cost_ledger import record_call
        u = getattr(resp, "usage", None)
        in_tok  = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        cache_r = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_w = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        cost_usd = compute_cost(
            model              = "claude-sonnet-4-6",
            input_tokens       = in_tok,
            output_tokens      = out_tok,
            cache_read_tokens  = cache_r,
            cache_write_tokens = cache_w,
        )
        record_call(
            agent_id          = "chat_ask",
            provider          = "anthropic",
            model             = "claude-sonnet-4-6",
            prompt_tokens     = in_tok + cache_r + cache_w,
            completion_tokens = out_tok,
            cost_usd          = cost_usd,
            latency_ms        = int(elapsed_s * 1000),
            scope             = retrieval_mode or "",
            extra             = {
                "session_id":      session_id,
                "n_prior_turns":   len(prior_turns),
                "n_context_rows":  n_rows,
                "cache_read":      cache_r,
                "cache_write":     cache_w,
                "caller":          caller,
            },
        )
    except Exception:
        logger.exception("chat_ask: cost ledger record_call failed (non-fatal)")

    # Parse [type:id] citations from the answer. Extended 2026-06-04
    # (Commit Z) to cover the new ledger types added in Commit X.
    citation_re = re.compile(
        r"\[(run_id|iteration_id|spec_id|sleeve|"
        r"event_id|paper_id|hypothesis_id|doctrine|audit_id|warning_id"
        r"):([A-Za-z0-9_\-\.#/]+)\]"
    )
    citations = [{"type": m.group(1), "id": m.group(2)}
                  for m in citation_re.finditer(answer)]
    # Dedup preserving order
    seen = set()
    citations_dedup: list[dict] = []
    for c in citations:
        k = (c["type"], c["id"])
        if k not in seen:
            seen.add(k)
            citations_dedup.append(c)

    # Commit Z 2026-06-04 — citation verification.
    # For every cited [type:id], check that the id actually exists in the
    # corresponding source. Unresolved citations get exists=False; the
    # frontend marks them red and tells the user "this citation could
    # not be verified", catching hallucinations immediately. Without
    # this, model can fabricate an event_id or paper_id and the UI
    # renders it like any other valid link.
    #
    # All checks are bounded (cap at 50 ids, < 10ms total) so cost is
    # negligible. Failures are logged + treated as exists=False; verify
    # MUST NOT crash the answer return path.
    citations_dedup = _verify_citations(citations_dedup)
    n_cited     = len(citations_dedup)
    n_resolved  = sum(1 for c in citations_dedup if c.get("exists") is True)
    n_unverified = sum(1 for c in citations_dedup if c.get("exists") is False)
    # Also count [unverified] markers — model self-confessed it didn't
    # have a citation. Tracking this lets us trust-calibrate the chat.
    n_self_unverified = len(re.findall(r"\[unverified\]", answer))

    _append_audit("chat_ask",
                   {"question_len":      len(body.question),
                    "n_citations":       n_cited,
                    "n_resolved":        n_resolved,
                    "n_unverified":      n_unverified,
                    "n_self_unverified": n_self_unverified,
                    "session_id":        session_id,
                    "n_prior_turns":     len(prior_turns)},
                   result_hash=None, ok=True,
                   latency_ms=elapsed_s * 1000.0, caller=caller)

    # Append this exchange to the session file so the next Ask call
    # picks up where this one left off.
    _append_session_turn(session_id, {
        "ts":             _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "question":       body.question,
        "answer":         answer,
        "citations":      citations_dedup,
        "retrieval_mode": retrieval_mode,
        "elapsed_s":      round(elapsed_s, 2),
    })

    return {
        "answer":          answer,
        "citations":       citations_dedup,
        # Commit Z 2026-06-04 — verification counts. Frontend uses these
        # to render the trust badge (resolved / unverified / self-flagged).
        "verification": {
            "n_cited":           n_cited,
            "n_resolved":        n_resolved,
            "n_unverified":      n_unverified,
            "n_self_unverified": n_self_unverified,
        },
        "n_context_rows": n_rows,
        "retrieval_mode": retrieval_mode,
        "model":          "claude-sonnet-4-6",
        "elapsed_s":      round(elapsed_s, 2),
        "question":       body.question,
        "session_id":     session_id,
        "n_prior_turns":  len(prior_turns),
    }


# ── Session admin endpoints (PR-A, 2026-06-02) ─────────────────────


@router.post("/chat/session/new")
def chat_session_new() -> dict:
    """Allocate a fresh chat session id. Cheap — no actual file is
    written until the first /chat/ask call lands."""
    return {"session_id": _new_session_id()}


@router.get("/chat/session/{session_id}")
def chat_session_get(session_id: str) -> dict:
    """Read full session history (chronological, newest LAST). Used
    by the side-panel chat + /chat full page to show prior turns."""
    p = _session_path(session_id)
    if not p.is_file():
        return {"session_id": session_id, "n_turns": 0, "turns": []}
    turns: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                turns.append(json.loads(line))
            except Exception:
                continue
    first_q = next((str(t.get("question") or "")[:120] for t in turns if t.get("question")), None)
    return {
        "session_id": session_id,
        "n_turns":    len(turns),
        "title":      _derive_session_title(first_q),
        "turns":      turns,
    }


@router.get("/chat/sessions")
def chat_sessions_list(limit: int = Query(40, ge=1, le=200)) -> dict:
    """List all chat sessions with summary metadata so the side panel
    can render a switcher (most recently active first).

    Returns array of {session_id, n_turns, first_question (truncated),
    last_ts}. Sessions with zero turns (allocated but never used) are
    surfaced too — they're cheap to ignore client-side but useful for
    "just-created, waiting for first ask" state."""
    if not CHAT_SESSION_DIR.is_dir():
        return {"n": 0, "sessions": []}
    rows: list[dict] = []
    for p in CHAT_SESSION_DIR.glob("*.jsonl"):
        first_q: Optional[str] = None
        last_ts: Optional[str] = None
        n_turns = 0
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    n_turns += 1
                    if first_q is None and row.get("question"):
                        first_q = str(row.get("question"))[:120]
                    if row.get("ts"):
                        last_ts = str(row.get("ts"))
        except Exception:
            continue
        rows.append({
            "session_id":     p.stem,
            "n_turns":        n_turns,
            "first_question": first_q,
            "title":          _derive_session_title(first_q),
            "last_ts":        last_ts,
        })
    # Most recently active first
    rows.sort(key=lambda r: r.get("last_ts") or "", reverse=True)
    return {"n": len(rows), "sessions": rows[:limit]}


@router.delete("/chat/session/{session_id}")
def chat_session_delete(session_id: str) -> dict:
    """Hard-delete one chat session. Renames the file to .deleted as a
    soft fence rather than unlinking outright — preserves audit trail
    while removing the session from the switcher list."""
    p = _session_path(session_id)
    if not p.is_file():
        return {"ok": True, "existed": False}
    try:
        # Move to .deleted suffix so it stops being listed but file
        # content is recoverable for audit.
        target = p.with_suffix(".jsonl.deleted")
        if target.exists():
            target.unlink()
        p.rename(target)
        return {"ok": True, "existed": True, "archived_to": str(target.name)}
    except Exception as exc:
        logger.exception("chat_session_delete failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Embedding index admin ──────────────────────────────────────────


@router.get("/chat/embedding_status")
def chat_embedding_status() -> dict:
    """Diagnostic — show which ledger indices exist + row counts.
    Used by the chat workbench to surface "semantic ON/OFF" badge."""
    try:
        from engine.research import embeddings as _E
        return {"available": True, "ledgers": _E.index_status()}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


class _RebuildRequest(BaseModel):
    ledger: Optional[str] = None    # None ⇒ rebuild all
    confirm: bool = False


@router.post("/chat/embedding_rebuild")
def chat_embedding_rebuild(body: _RebuildRequest) -> dict:
    """Rebuild one or all ledger embedding indices. Cheap (seconds)
    for the current scale; runs synchronously."""
    if not body.confirm:
        raise HTTPException(status_code=400,
                            detail="confirm=true required (rebuild is synchronous)")
    try:
        from engine.research import embeddings as _E
    except ImportError as exc:
        raise HTTPException(status_code=503,
                            detail=f"embeddings module unavailable: {exc}")
    t0 = time.perf_counter()
    try:
        if body.ledger:
            results = [_E.build_index(body.ledger)]
        else:
            results = _E.build_all()
    except Exception as exc:
        logger.exception("embedding rebuild failed")
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_s = time.perf_counter() - t0
    return {"results": results, "elapsed_s": round(elapsed_s, 2)}


# ── Liveness layer (P0b 2026-06-02) ────────────────────────────────


@router.get("/liveness/status")
def liveness_status(
    limit: int = Query(14, ge=1, le=60),
    fresh: bool = Query(True, description="If true, overlay LIVE data-source freshness probes onto the latest heartbeat so the banner self-heals when a source recovers between cron runs"),
) -> dict:
    """Single-call surface for the topbar banner + Cockpit hero block.

    Returns:
      verdict   — assess_liveness() output for the most recent expected
                  weekday run (OK / WARN_STATUS / ALERT_NO_SHOW / INFO_*)
      recent    — most recent N heartbeat rows (newest first), suitable
                  for the 14-day calendar grid
      summary   — small object the topbar reads (tone + headline)

    2026-06-02: added `fresh=true` (default) so the banner overlays a
    LIVE probe of data_sources on top of the latest heartbeat. Without
    this the banner shows "1 source DEAD" forever after the cron snapshot
    captured a transient outage, even if the source has since recovered.
    Pass fresh=false to see the raw frozen heartbeat snapshot.
    """
    try:
        from engine.research import liveness_heartbeat as L
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    import datetime as _dt
    now = _dt.datetime.utcnow()
    verdict = L.assess_liveness(now_utc=now)
    recent  = L.read_recent(limit=limit)

    # Overlay live freshness probes onto the most recent heartbeat row.
    # This is the "self-heal" mechanism — the banner sees current state
    # rather than the morning's snapshot.
    if fresh and recent:
        try:
            from engine.research.data_freshness import check_sources, summarize
            live_sources = check_sources()
            live_summary = summarize(live_sources)
            latest = dict(recent[0])
            latest["data_sources"]   = live_sources
            latest["data_freshness"] = live_summary
            latest["_freshness_live_overlay_ts"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            recent = [latest, *recent[1:]]

            # 2026-06-02 — also patch the VERDICT. assess_liveness() at
            # liveness_heartbeat.py:285-303 downgrades cron-OK runs to
            # WARN_STATUS when the heartbeat row carries worst_status='dead'.
            # That downgrade was made BEFORE our overlay ran, so when the
            # live sources are actually healthy we need to lift the verdict
            # back to OK — otherwise the banner shows the stale warning
            # forever despite recent[0] being healthy.
            overlaid_worst = (live_summary or {}).get("worst_status")
            v_code = verdict.get("verdict")
            v_expl = (verdict.get("explanation") or "").lower()
            was_data_freshness_downgrade = (
                v_code == "WARN_STATUS"
                and ("data source" in v_expl or "dead pipe" in v_expl)
                and overlaid_worst not in ("dead",)
            )
            if was_data_freshness_downgrade:
                verdict = dict(verdict)
                verdict["verdict"] = "OK"
                age_min = verdict.get("age_min", "?")
                verdict["explanation"] = (
                    f"Run for {verdict.get('as_of','?')} succeeded; "
                    f"earlier data-source DEAD warning cleared "
                    f"(worst={overlaid_worst}) per live overlay at "
                    f"{now.strftime('%H:%M:%SZ')}."
                )
                verdict["_self_healed_from_data_freshness_warn"] = True
        except Exception as exc:
            # Live overlay is best-effort; on failure fall back to the
            # frozen heartbeat snapshot rather than break the banner.
            logger.exception("liveness_status live overlay failed: %s", exc)

    # Compact summary for the topbar — must not require client logic
    v = verdict.get("verdict", "")
    if v == "OK":
        tone, headline = "ok", f"live · {verdict.get('age_min','?')}m ago"
    elif v == "WARN_STATUS":
        latest = verdict.get("latest") or {}
        tone = "warn"
        headline = (f"{latest.get('as_of','?')} halted at "
                     f"{latest.get('halted_at_step') or latest.get('status','?')}")
    elif v == "ALERT_NO_SHOW":
        tone = "danger"
        headline = (f"missing heartbeat — {verdict.get('as_of','?')} "
                     f"cron may have failed")
    elif v == "INFO_WEEKEND":
        tone, headline = "muted", "weekend — no run expected"
    else:
        tone, headline = "muted", "off-hours"

    return {
        "verdict": verdict,
        "recent":  recent,
        "summary": {
            "verdict_code": v,
            "tone":         tone,
            "headline":     headline,
        },
    }


# ────────────────────────────────────────────────────────────────────
# Stage A piece 6: RED-outcome surface for /research/forward UI tab
#
# Lists recent factor_verdict_filed events with verdict=RED so the
# principal (and A's anti-rut substrate awareness) can see WHICH
# directions have already been ruled out — preventing A from
# proposing them again as fresh candidates.
#
# JOIN strategy: factor_verdict_filed.subject_id = auto_<hash> is the
# strict-gate factor subject; we walk back via
# candidate_pipeline_started events (which carry source_hypothesis_id
# in metrics) to recover the originating hypothesis and its source
# paper title (joined against hypotheses store).
# ────────────────────────────────────────────────────────────────────
@router.get("/red_outcomes")
def red_outcomes(
    days:  int = Query(30, ge=1, le=365,
                        description="Window in days (default 30)"),
    limit: int = Query(50, ge=1, le=500,
                        description="Max items returned (default 50)"),
) -> dict:
    """RED-verdict outcomes within the rolling window, joined with
    source hypothesis + paper title when discoverable.

    Response shape (machine-stable):
      {
        since:        ISO-8601 cutoff,
        n_total:      total matching events in window,
        n_returned:   items in the response (capped at limit),
        items: [
          {
            event_id:           str,
            subject_id:         str,
            family:             str,
            verdict_ts:         ISO-8601,
            score:              int (0-7),
            summary:            short verdict summary,
            source_hypothesis_id: str | null,
            source_paper_title: str | null,
            source_paper_id:    str | null,
          },
          ...
        ]
      }
    Items sorted newest first. NULL source_* fields mean we couldn't
    walk the JOIN — usually because the candidate_pipeline_started
    event was emitted without source_hypothesis_id (pre-piece-7b
    candidates).
    """
    from engine.research_store.store import filter_events

    cutoff = (_dt.datetime.utcnow()
               - _dt.timedelta(days=days)
               ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Pull RED verdicts in window
    red_events = filter_events(
        event_type = "factor_verdict_filed",
        verdict    = "RED",
        since      = cutoff,
        limit      = limit,
    )

    # 2. For each subject_id, find the candidate_pipeline_started event
    # (or candidate_skipped_pre_compute) to recover source_hypothesis_id.
    # Bulk-load pipeline events once to keep this O(N+M) not O(N*M).
    pipeline_starts = filter_events(
        event_type = "candidate_pipeline_started",
        since      = cutoff,
    )
    by_subject: dict[str, str] = {}
    for ev in pipeline_starts:
        sid = ev.subject_id
        if sid and sid not in by_subject:
            metrics = ev.metrics or {}
            hid = metrics.get("source_hypothesis_id")
            if hid:
                by_subject[sid] = str(hid)

    # 3. Bulk-load Hypothesis registry so we can resolve paper title
    try:
        from engine.research_store.hypothesis.store import load_hypotheses
        all_hyps = load_hypotheses()
        # Latest version per id
        latest_by_id: dict[str, object] = {}
        for h in all_hyps:
            prior = latest_by_id.get(h.hypothesis_id)
            if prior is None or h.version > prior.version:
                latest_by_id[h.hypothesis_id] = h
    except Exception as exc:
        logger.warning("red_outcomes: load_hypotheses failed: %s", exc)
        latest_by_id = {}

    # 4. Bulk-load papers_registry to get paper titles
    paper_title_by_id: dict[str, str] = {}
    papers_path = (REPO_ROOT / "data" / "research_store"
                    / "papers_registry.jsonl")
    if papers_path.is_file():
        try:
            with papers_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                    except Exception:
                        continue
                    pid = row.get("paper_id")
                    title = row.get("title")
                    if pid and title:
                        paper_title_by_id[str(pid)] = str(title)
        except Exception as exc:
            logger.warning("red_outcomes: papers_registry read failed: %s",
                            exc)

    # 5. Build response items with JOIN
    items: list[dict] = []
    for ev in red_events:
        metrics = ev.metrics or {}
        source_hyp_id = by_subject.get(ev.subject_id or "")
        source_paper_id: Optional[str] = None
        source_paper_title: Optional[str] = None
        if source_hyp_id and source_hyp_id in latest_by_id:
            h = latest_by_id[source_hyp_id]
            pid = getattr(h, "source_paper_id", "") or ""
            if not pid:
                syn = tuple(getattr(h, "synthesizes_paper_ids", ()) or ())
                if syn:
                    pid = syn[0]
            if pid:
                source_paper_id = str(pid)
                source_paper_title = paper_title_by_id.get(str(pid))
        items.append({
            "event_id":             ev.event_id,
            "subject_id":           ev.subject_id,
            "family":               ev.family,
            "verdict_ts":           ev.ts,
            "score":                int(metrics.get("score") or 0),
            "summary":              ev.summary or "",
            "source_hypothesis_id": source_hyp_id,
            "source_paper_id":      source_paper_id,
            "source_paper_title":   source_paper_title,
        })

    return {
        "since":      cutoff,
        "n_total":    len(red_events),
        "n_returned": len(items),
        "items":      items,
    }


# ────────────────────────────────────────────────────────────────────
# Tier C L3-2 (2026-06-08): unified Tier-C verdict feed including
# self_doubt assessment.
#
# Sister to /red_outcomes but covers ALL three Sharpe-based verdicts
# (GREEN/MARGINAL/RED) and surfaces the new L3-2 self_doubt payload
# when present. The frontend can then display confidence + caveats
# inline so the principal doesn't rubber-stamp a clean-looking GREEN.
#
# Filter is tier_c_auto-tagged events only — older non-Tier-C verdict
# subjects don't leak in.
# ────────────────────────────────────────────────────────────────────
@router.get("/tier_c_verdicts")
def tier_c_verdicts(
    days:  int = Query(30, ge=1, le=365,
                        description="Window in days (default 30)"),
    verdicts: str = Query("GREEN,MARGINAL,RED",
                            description=("Comma list of verdict codes "
                                          "to include. Default all three.")),
    limit: int = Query(50, ge=1, le=500,
                        description="Max items returned (default 50)"),
    investment_role: Optional[str] = Query(
        None,
        description=("Filter by 7-axis investment_role "
                      "(alpha / insurance / diversifier / hedge / overlay). "
                      "Pre-v2 events without this field are matched via "
                      "legacy fallback inference (defaults to 'alpha').")),
) -> dict:
    """Tier-C auto-dispatched verdicts within the rolling window with
    L3-2 self_doubt inline.

    Response shape (machine-stable):
      {
        since:        ISO-8601 cutoff,
        verdicts:     [str, ...]      # filter applied
        n_total:      total matching events in window,
        n_returned:   items in response (capped at limit),
        items: [
          {
            event_id:           str,
            subject_id:         str,
            family:             str,
            verdict:            "GREEN" | "MARGINAL" | "RED",
            verdict_ts:         ISO-8601,
            summary:            str,
            signal_kind:        str,
            spec_hash:          str,
            template_version:   str,
            # Inline key metrics — UI doesn't need to spelunk full payload
            sharpe:             float | null,
            nw_t_stat:          float | null,
            n_months:           int   | null,
            avg_turnover:       float | null,
            cost_robust_verdict: str  | null,
            replication: {
              status:           str   | null,
              our_t:            float | null,
              paper_reported_t: float | null,
              t_gap:            float | null,
            } | null,
            # L3-2 SELF-DOUBT — None when the dispatch predates L3-2
            # (commit ad2db4bf, 2026-06-08) or the LLM call failed
            self_doubt: {
              confidence:              float (0-0.99),
              confidence_reason:       str,
              caveats:                 [str, ...],
              methodological_concerns: [str, ...],
              suspicious_metrics:      [str, ...],
              assessment_ts:           ISO-8601,
              model:                   str,
            } | null,
          },
          ...
        ]
      }
    Items sorted newest first.
    """
    from engine.research_store.store import filter_events

    cutoff = (_dt.datetime.utcnow()
               - _dt.timedelta(days=days)
               ).strftime("%Y-%m-%dT%H:%M:%SZ")

    wanted_verdicts = {
        v.strip().upper() for v in verdicts.split(",") if v.strip()
    } or {"GREEN", "MARGINAL", "RED"}

    # filter_events doesn't accept verdict=list or tags=, so we pull
    # broader and filter in-process. Over-pull to absorb the verdict +
    # tag narrowing without missing items past the requested limit.
    raw = filter_events(
        event_type = "factor_verdict_filed",
        since      = cutoff,
        limit      = max(limit * 5, 200),
    )

    items: list[dict] = []
    for ev in raw:
        # Tier C auto-tagged only — keeps non-Tier-C verdicts out of
        # this feed (other subjects may share factor_verdict_filed).
        if "tier_c_auto" not in (ev.tags or ()):
            continue
        v = (ev.verdict or "").upper()
        if v not in wanted_verdicts:
            continue
        m = ev.metrics or {}
        # Replication block (None when template didn't emit it)
        rep_raw = m.get("replication") or {}
        replication_out = None
        if rep_raw and rep_raw.get("status"):
            replication_out = {
                "status":           rep_raw.get("status"),
                "our_t":            rep_raw.get("our_t"),
                "paper_reported_t": rep_raw.get("paper_reported_t"),
                "t_gap":            rep_raw.get("t_gap"),
            }
        # self_doubt block (None when pre-L3-2 or LLM call failed)
        sd_raw = m.get("self_doubt") or None
        self_doubt_out = None
        if isinstance(sd_raw, dict):
            self_doubt_out = {
                "confidence":              sd_raw.get("confidence"),
                "confidence_reason":       sd_raw.get("confidence_reason"),
                "caveats":                 list(sd_raw.get("caveats") or ()),
                "methodological_concerns": list(
                    sd_raw.get("methodological_concerns") or ()),
                "suspicious_metrics":      list(
                    sd_raw.get("suspicious_metrics") or ()),
                "assessment_ts":           sd_raw.get("assessment_ts"),
                "model":                   sd_raw.get("model"),
            }
        items.append({
            "event_id":            ev.event_id,
            "subject_id":          ev.subject_id,
            "family":              ev.family,
            "verdict":             v,
            "verdict_ts":          ev.ts,
            "summary":             ev.summary or "",
            "signal_kind":         m.get("signal_kind") or "",
            "spec_hash":           m.get("auto_test_spec_hash") or "",
            "template_version":    m.get("template_version") or "",
            "sharpe":              m.get("sharpe"),
            "nw_t_stat":           m.get("nw_t_stat"),
            "n_months":            m.get("n_months"),
            "avg_turnover":        m.get("avg_turnover"),
            "cost_robust_verdict": m.get("cost_robust_verdict"),
            "replication":         replication_out,
            "self_doubt":          self_doubt_out,
            # L2-4 prep: parquet path with monthly PnL (gross /
            # net_13bp / net_80bp / turnover). null on dispatches
            # before 2026-06-08 or when persistence failed.
            "pnl_series_parquet":  m.get("pnl_series_parquet"),
            # L2-4 Commit 3: anchor-orthogonality residual-α
            # regression vs Ken French FF5+MOM. null pre-L2-4 or
            # when anchor library missing / regression failed.
            "anchor_orthogonality": m.get("anchor_orthogonality"),
            # L2-5 Commit 2: subsample stability decomposition
            # (N-split per-window Sharpe + aggregate flags). null
            # pre-L2-5 or when insufficient history.
            "subsample_stability": m.get("subsample_stability"),
            # L2-6 Commit 3: JOINT-model industry extension
            # (FF5+MOM ∪ 12-Industry alpha + Δα + subset F-test).
            # Per post-FWL-fix 2026-06-09.
            "industry_extension": m.get("industry_extension"),
            # Cross-asset lite Commit 4: JOINT model extended with
            # 5 FRED macro regime regressors (VIX, DXY, BAA, term,
            # breakeven). Up to 23-regressor kitchen-sink test.
            "cross_asset_extension": m.get("cross_asset_extension"),
            # Phase 1 Commit 3 (role-routing): audit trail of WHICH
            # lenses ran and WHY. Each entry: {lens, action, reason,
            # applicable_required?}. null pre-2026-06-09.
            "routing_decisions": m.get("routing_decisions"),
            # Phase 1 Commit 7: explicit investment_role surfaced for
            # filtering. Pre-v2 events have None → legacy fallback in
            # response shape uses default "alpha".
            "investment_role":   m.get("investment_role"),
        })

    # Phase 1 Commit 7: investment_role filter with legacy fallback
    # Pre-v2 events don't have investment_role in metrics; we infer
    # "alpha" as the conservative default (matches legacy dispatch).
    if investment_role is not None:
        role_query = investment_role.strip().lower()
        def _matches_role(item: dict) -> bool:
            explicit = item.get("investment_role")
            if explicit:
                return str(explicit).lower() == role_query
            # Legacy fallback: pre-v2 events default to alpha
            return role_query == "alpha"
        items = [it for it in items if _matches_role(it)]

    items.sort(key=lambda x: x["verdict_ts"] or "", reverse=True)
    n_total = len(items)
    items = items[:limit]

    return {
        "since":      cutoff,
        "verdicts":   sorted(wanted_verdicts),
        "n_total":    n_total,
        "n_returned": len(items),
        "items":      items,
    }


# ────────────────────────────────────────────────────────────────────
# Stage C Tier B-1: anchor library browse surface
#
# Returns the 27 canonical T1+T2 papers A sees in its synthesis
# prompt — the "three libraries" doctrine made real. Powers
# /research/forward/anchors UI page so the principal can browse the
# anchor library directly (which papers are T1 vs T2, what each
# one anchors, when they were enriched).
# ────────────────────────────────────────────────────────────────────
@router.get("/anchor_library")
def anchor_library() -> dict:
    """List the T1_DOCTRINE + T2_ANCHOR papers in papers_registry
    with their tier_anchor_summary, grouped by tier.

    Response shape:
      {
        n_total:      int,
        n_t1:         int,
        n_t2:         int,
        items: [
          {
            paper_id:           str (canonical full),
            title:              str,
            authors:            [str, ...],
            year:               int,
            venue:              str,
            doi:                str,
            tier:               "T1_DOCTRINE" | "T2_ANCHOR",
            tier_rationale:     str,
            tier_anchor_summary:str,
            tier_classified_ts: str,
            mechanism_family:   str (best-effort from any cross-ref),
          },
          ...
        ],
      }
    """
    from engine.research_store.papers.store import load_registry
    from engine.research_store.papers.schema import PaperTier

    raw = load_registry()
    # Latest-per-paper_id dedup (chain history artifact)
    by_pid: dict = {}
    for p in raw:
        prior = by_pid.get(p.paper_id)
        if prior is None or p.version > prior.version:
            by_pid[p.paper_id] = p
    latest = list(by_pid.values())

    # DOI dedup — prefer entry with tier_anchor_summary set
    by_doi: dict = {}
    no_doi: list = []
    for p in latest:
        d = (p.doi or "").strip().lower()
        if not d:
            no_doi.append(p)
            continue
        cur = by_doi.get(d)
        if cur is None:
            by_doi[d] = p
        elif (p.tier_anchor_summary and not cur.tier_anchor_summary):
            by_doi[d] = p
    functional = list(by_doi.values()) + no_doi

    items = []
    for p in functional:
        if p.tier not in (PaperTier.T1_DOCTRINE, PaperTier.T2_ANCHOR):
            continue
        items.append({
            "paper_id":           p.paper_id,
            "title":              p.title,
            "authors":            list(p.authors or ()),
            "year":               p.year,
            "venue":              p.venue or "",
            "doi":                p.doi or "",
            "tier":               p.tier.value,
            "tier_rationale":     p.tier_rationale or "",
            "tier_anchor_summary":p.tier_anchor_summary or "",
            "tier_classified_ts": p.tier_classified_ts or "",
        })

    # Sort: T1 first (most precious), then T2; within tier by year desc
    items.sort(key=lambda x: (0 if x["tier"] == "T1_DOCTRINE" else 1,
                                -(x["year"] or 0)))

    n_t1 = sum(1 for it in items if it["tier"] == "T1_DOCTRINE")
    n_t2 = sum(1 for it in items if it["tier"] == "T2_ANCHOR")
    return {
        "n_total": len(items),
        "n_t1":    n_t1,
        "n_t2":    n_t2,
        "items":   items,
    }
