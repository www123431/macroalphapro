"""api/routes_pipeline_stream.py — Phase A.2: SSE streaming endpoint
for live candidate_pipeline progress.

Frontend wires to GET /api/pipeline/stream/{candidate_id} and receives
text/event-stream messages as each LangGraph node completes:

  event: step_start
  data: {"step": "h10", "ts": "..."}

  event: step_complete
  data: {"step": "h10", "status": "PASS", "verdict": "..."}

  event: pipeline_complete
  data: {"final_decision": "...", "rationale": "..."}

  event: pipeline_error
  data: {"error": "..."}

Uses LangGraph's graph.astream() for native async streaming of node
state transitions (Phase A.1 v2 pipeline required).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import AsyncIterator, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def _sse_format(event: str, data: dict) -> str:
    """Format an SSE message frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_pipeline(
    series_path: str,
    proposal_name: str,
    proposed_role: str,
    mechanism_id: Optional[str],
    proposal_dict: Optional[dict],
    thread_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """Run v2 pipeline via graph.astream and yield SSE frames.

    Phase 4a: if thread_id given, run with SqliteSaver durable
    checkpointer so the same id can resume via /api/pipeline/resume.
    Default behavior (thread_id=None) is non-durable (cheap, no DB
    write) — matches prior behavior for backward compat.
    """
    ckpt_cm = None  # initialize before try so finally can always check
    try:
        # Import lazily — LangGraph + v2 pipeline modules are heavy
        from engine.research.candidate_pipeline_v2 import (
            CandidateState, build_pipeline_graph, _sqlite_checkpointer,
            make_thread_id,
        )
        import engine.research.sleeves  # noqa: F401 — registers sleeves

        # Load candidate returns
        try:
            s = pd.read_parquet(series_path).iloc[:, 0]
            s.index = pd.to_datetime(s.index)
        except Exception as exc:
            yield _sse_format("pipeline_error",
                              {"error": f"failed to load {series_path}: {exc}"})
            return

        state = CandidateState(
            candidate_returns=s, proposal_name=proposal_name,
            proposed_role=proposed_role, mechanism_id=mechanism_id,
            proposal_dict=proposal_dict, phase=3,
        )

        # Compile + optionally bind durable checkpointer
        if thread_id is not None:
            tid = thread_id or make_thread_id(proposal_name)
            ckpt_cm = _sqlite_checkpointer()
            ckpt = ckpt_cm.__enter__()
            graph = build_pipeline_graph().compile(checkpointer=ckpt)
            invoke_config = {"configurable": {"thread_id": tid}}
        else:
            tid = None
            ckpt_cm = None
            graph = build_pipeline_graph().compile()
            invoke_config = None

        # Emit start frame
        yield _sse_format("pipeline_start", {
            "proposal_name": proposal_name,
            "n_months": len(s),
            "gross_sharpe": float(
                (s.mean() * 12) / (s.std() * (12 ** 0.5))
            ),
            "thread_id": tid,
            "durable": tid is not None,
        })

        # F7 (2026-06-05): two bugs fixed in this block.
        #
        # Bug 1 (duplicate H10): pre-F7 used stream_mode="updates" which
        # yields {node_name: changed_state}. The handler emitted
        # step_results[-1] every chunk. When a routing node (e.g.
        # short_circuit_end) fired without adding a new step, the
        # PREVIOUS step_result re-emitted — user saw H10 twice.
        #
        # Bug 2 (double execution): after astream finished, the handler
        # called graph.invoke(state) AGAIN to get the final state. That
        # re-runs the entire pipeline from START, doubling wall-clock
        # and producing more duplicate frontend events. User saw
        # "awaiting next step" spinning long after the actual verdict.
        #
        # Fix:
        #   - stream_mode="values" => each chunk is the FULL state
        #   - track n_emitted_steps; emit only NEW step_results
        #   - keep last chunk as final_state_dict; no re-invoke
        stream_kwargs = {"stream_mode": "values"}
        if invoke_config is not None:
            stream_kwargs["config"] = invoke_config

        final_state_dict: dict = {}
        n_emitted = 0
        async for chunk in graph.astream(state, **stream_kwargs):
            # In values mode, chunk IS the cumulative state dict.
            chunk_dict = chunk if isinstance(chunk, dict) else (
                chunk.__dict__ if hasattr(chunk, "__dict__") else {}
            )
            final_state_dict = chunk_dict
            step_results = chunk_dict.get("step_results", []) or []
            # Emit only the steps we haven't yet
            while n_emitted < len(step_results):
                sr = step_results[n_emitted]
                yield _sse_format("step_complete", {
                    "node":      getattr(sr, "step_name", "") or "",
                    "step_name": getattr(sr, "step_name", ""),
                    "status":    getattr(sr, "status", ""),
                    "verdict":   getattr(sr, "verdict", ""),
                })
                n_emitted += 1
            await asyncio.sleep(0.01)

        yield _sse_format("pipeline_complete", {
            "final_decision":         final_state_dict.get("final_decision"),
            "rationale":              (final_state_dict.get("rationale") or "")[:500],
            "candidate_relation":     final_state_dict.get("candidate_relation"),
            "most_correlated_sleeve": final_state_dict.get("most_correlated_sleeve"),
            "most_correlated_value":  final_state_dict.get("most_correlated_value"),
            "thread_id":              tid,
        })

    except Exception as exc:
        logger.exception("pipeline streaming error")
        yield _sse_format("pipeline_error", {"error": str(exc)})
    finally:
        if ckpt_cm is not None:
            try:
                ckpt_cm.__exit__(None, None, None)
            except Exception:
                logger.exception("checkpointer close failed")


@router.get("/stream")
async def stream_pipeline(
    series_path: str = Query(..., description="parquet path of returns series"),
    proposal_name: str = Query("candidate"),
    proposed_role: str = Query("alpha_seeker"),
    mechanism_id: Optional[str] = Query(None),
    family: Optional[str] = Query(None),
    parent_family: Optional[str] = Query("equity_factor"),
    economics_text: Optional[str] = Query(""),
    thread_id: Optional[str] = Query(
        None, description="if given, run durable with SqliteSaver",
    ),
) -> StreamingResponse:
    """SSE endpoint streaming each pipeline node's verdict in real time."""
    proposal_dict = None
    if family:
        proposal_dict = {
            "family":        family,
            "parent_family": parent_family,
            "required_data": [],
            "economics_text": economics_text or "",
        }
    return StreamingResponse(
        _stream_pipeline(series_path, proposal_name, proposed_role,
                         mechanism_id, proposal_dict, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/state/{thread_id}")
def get_pipeline_state(thread_id: str) -> dict:
    """Phase 4a: inspect a durable pipeline run by thread_id.

    Returns 404 if the thread is unknown. Otherwise: the latest
    checkpoint state (n_steps completed + final_decision if any).
    """
    from engine.research.candidate_pipeline_v2 import get_checkpoint_state
    snap = get_checkpoint_state(thread_id)
    if snap is None:
        raise HTTPException(status_code=404,
                            detail=f"no checkpoint for thread_id={thread_id!r}")
    return {
        "thread_id": thread_id,
        "n_steps_completed": len(snap.get("step_results") or []),
        "final_decision": snap.get("final_decision"),
        "candidate_relation": snap.get("candidate_relation"),
        "rationale": (snap.get("rationale") or "")[:500],
        "short_circuited": snap.get("short_circuited"),
    }


@router.get("/candidates")
def list_candidates() -> dict:
    """Quick list of candidate parquet files known to system. Used by
    frontend dropdown.

    F1 (2026-06-05): this endpoint kept the pre-baked 5-parquet list
    as the "legacy" track. The new paper-grounded 88 candidates surface
    via /api/paper_chain/forward-vectors/ranked (the F1 join endpoint).
    Frontend groups the two tracks in the dropdown.
    """
    from pathlib import Path
    cache_dir = Path("data/cache")
    candidates = []
    for known in [
        ("_dpead_sn_pit_monthly.parquet", "post_earnings_drift_pit_sn",
         "alpha_seeker", "post_earnings_drift", "earnings_underreaction"),
        ("_ltr_monthly_long.parquet", "long_term_reversal",
         "alpha_seeker", "long_term_reversal", "long_term_reversal"),
        ("_issuance_monthly.parquet", "issuance_anomaly",
         "alpha_seeker", "issuance", "issuance"),
        ("_tail_hedge_put_spread_monthly.parquet", "tail_hedge_put_spread",
         "insurance", "tail_hedge_put_spread", "tail_hedge"),
        ("_jp_pead_monthly.parquet", "jp_pead",
         "alpha_seeker", "jp_pead", "forward-earnings information"),
    ]:
        path = cache_dir / known[0]
        if path.exists():
            candidates.append({
                "series_path":    str(path),
                "proposal_name":  known[1],
                "proposed_role":  known[2],
                "mechanism_id":   known[3],
                "family":         known[4],
                "track":          "pre_baked_legacy",
            })
    return {"candidates": candidates, "track_label": "Pre-baked legacy parquets"}


# ── F2: prepare-from-fv ───────────────────────────────────────────


@router.post("/prepare-from-fv")
def prepare_from_forward_vector(
    source_hypothesis_id: str = Query(..., description="hypothesis id to compose"),
    force: bool = Query(False, description="re-compose even if cached"),
) -> dict:
    """F2 (2026-06-05): given a forward-vector / hypothesis id, look up
    the latest HypothesisSpec, validate coverage, and call composer.
    Returns the parquet path the frontend then passes to /stream.

    Failure modes are explicit (no silent substitution per LdP §2):
      404  no spec for this hypothesis_id
      422  spec.claim_type != FACTOR_HYPOTHESIS (or non-factor)
      412  composer coverage gaps — body lists role + expected_key
      500  composer raised during build (data fetch / signal compute)

    Response on success:
      {
        spec_hash, parquet_path, n_obs, from_cache, elapsed_s,
        proposal_name, proposed_role, family, signal_type,
        stream_url:    URL the frontend opens as EventSource
      }
    """
    from engine.hypothesis_spec.store import latest_for
    from engine.hypothesis_spec.hash import spec_hash
    from engine.hypothesis_spec.enums import ClaimType
    from engine.composer.contract import is_spec_covered
    from engine.composer import composer as _composer
    from urllib.parse import urlencode

    spec = latest_for(source_hypothesis_id)
    if spec is None:
        raise HTTPException(status_code=404,
                             detail=f"no hypothesis_spec for hypothesis_id={source_hypothesis_id!r}")
    if spec.claim_type != ClaimType.FACTOR_HYPOTHESIS:
        raise HTTPException(
            status_code=422,
            detail={
                "error":      "not_factor_hypothesis",
                "claim_type": spec.claim_type.value,
                "message":    (f"hypothesis_id={source_hypothesis_id} has "
                                f"claim_type={spec.claim_type.value}; "
                                f"only FACTOR_HYPOTHESIS specs flow into "
                                f"the pipeline (other types are research "
                                f"evidence, not testable strategies)."),
            },
        )

    covered, gaps = is_spec_covered(spec)
    if not covered:
        raise HTTPException(
            status_code=412,
            detail={
                "error":           "missing_components",
                "spec_hash":       spec_hash(spec),
                "n_gaps":          len(gaps),
                "gaps":            [
                    {"role": g.role.value, "expected_key": g.expected_key,
                     "reason": ("unknown_extracted"
                                if g.expected_key == "UNKNOWN"
                                or g.expected_key.endswith("__UNKNOWN")
                                else "missing")}
                    for g in gaps
                ],
                "message":         (f"composer cannot build this spec yet: "
                                     f"{len(gaps)} component(s) missing or "
                                     f"extractor returned UNKNOWN. UI shows "
                                     f"per-role gaps so engineering knows "
                                     f"what to implement next."),
            },
        )

    # Call composer
    try:
        result = _composer.compose(spec, force=force)
    except Exception as exc:
        logger.exception("composer raised for hypothesis_id=%s", source_hypothesis_id)
        raise HTTPException(status_code=500,
                             detail=f"composer build failed: {type(exc).__name__}: {exc}")

    if not result.get("ok"):
        raise HTTPException(status_code=500,
                             detail=f"composer returned not-ok: {result.get('error')}")

    parquet_path = result["path"]
    proposal_name = f"fv_{source_hypothesis_id[:8]}"

    # F4 (2026-06-05): emit candidate_pipeline_started so the
    # forward_vector decision is queryable via research_store.
    # subject_id uses proposal_name (the canonical-ish factor identity
    # the eventual factor_verdict_filed event uses too) so a future
    # filter_events(subject_id=proposal_name) returns both events.
    try:
        from engine.research_store import emit as _rs_emit
        from engine.research_store import registry as _registry
        from engine.research_store.schema import SubjectType as _SubjectType
        try:
            _registry.register_subject(
                proposal_name,
                subject_type=_SubjectType.factor,
                family=spec.family.value,
                description=(f"Pipeline-test candidate auto-registered from "
                             f"forward_vector / hypothesis_id "
                             f"{source_hypothesis_id}"),
                created_by="prepare_from_fv",
            )
        except Exception:
            pass
        _rs_emit.candidate_pipeline_started(
            subject_id=proposal_name,
            spec_hash=result["spec_hash"],
            source_hypothesis_id=source_hypothesis_id,
            family=spec.family.value,
            metrics={
                "n_obs":      int(result.get("n_obs", 0)),
                "from_cache": bool(result.get("from_cache", False)),
                "elapsed_s":  float(result.get("elapsed_s", 0.0)),
            },
            tags=("F4_traceability", "from_prepare_from_fv"),
            actor="prepare_from_fv",
        )
    except Exception as _exc:
        # Traceability emit must never break the primary path
        logger.warning("F4 emit failed (non-fatal): %s", _exc)

    # Build the /stream URL the frontend uses as EventSource
    primary_leg = spec.legs[0] if spec.legs else None
    stream_qs = urlencode({
        "series_path":   parquet_path,
        "proposal_name": proposal_name,
        "proposed_role": "alpha_seeker",
        "mechanism_id":  primary_leg.signal_type.value if primary_leg else "UNKNOWN",
        "family":        spec.family.value,
        "parent_family": spec.universe.asset_class.value.lower() + "_factor",
        "economics_text": (spec.claim_text or "")[:300],
    })

    return {
        "ok":             True,
        "spec_hash":      result["spec_hash"],
        "parquet_path":   parquet_path,
        "n_obs":          result.get("n_obs", 0),
        "from_cache":     result.get("from_cache", False),
        "elapsed_s":      result.get("elapsed_s", 0.0),
        "proposal_name":  proposal_name,
        "proposed_role": "alpha_seeker",
        "family":         spec.family.value,
        "signal_type":    primary_leg.signal_type.value if primary_leg else "UNKNOWN",
        "stream_url":     f"/api/pipeline/stream?{stream_qs}",
    }
