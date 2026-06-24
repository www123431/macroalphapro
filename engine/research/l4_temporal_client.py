"""engine/research/l4_temporal_client.py — Phase 4c: graceful
client helper for the L4 Temporal workflow.

Centralizes:
  - is_temporal_available()       — non-blocking ping
  - enqueue_council_workflow()    — start workflow + return id
                                     immediately (no ~50s block)
  - query_workflow_status()        — Temporal Query API (live state)
  - wait_for_workflow_result()     — block until completion (testing)

Design principle: the REST shim chooses sync-or-async via
is_temporal_available(). When the dev server isn't running the
council still works synchronously (current behavior). When it IS
running the user sees a non-blocking trigger + live status.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Optional

from temporalio.client import Client

logger = logging.getLogger(__name__)

DEFAULT_TEMPORAL_ADDRESS = os.environ.get(
    "TEMPORAL_ADDRESS", "localhost:7233",
)
DEFAULT_NAMESPACE = "default"

# Cache a single client per process — connection is reusable across
# enqueue / query calls; recreating per call costs ~50ms each time.
_CLIENT: Optional[Client] = None
_CLIENT_LOCK = asyncio.Lock()


async def _get_client(address: str = DEFAULT_TEMPORAL_ADDRESS) -> Client:
    global _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = await Client.connect(address, namespace=DEFAULT_NAMESPACE)
        return _CLIENT


async def is_temporal_available(
    address: str = DEFAULT_TEMPORAL_ADDRESS,
    timeout: float = 0.5,
) -> bool:
    """Non-blocking probe: can we reach a Temporal server?

    Returns False on connection failure / timeout, never raises.
    Used by the REST shim to decide sync vs async path.
    """
    try:
        # Connect attempt with a short timeout
        await asyncio.wait_for(_get_client(address), timeout=timeout)
        return True
    except Exception as exc:
        logger.debug("Temporal not available at %s: %s", address, exc)
        return False


async def enqueue_council_workflow(
    seed_idea: str,
    *,
    candidate_returns_path: Optional[str] = None,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
) -> dict:
    """Start an L4DiscoveryWorkflow + return its workflow_id IMMEDIATELY.

    The workflow runs in the worker process; this function does NOT
    block on completion. Caller polls query_workflow_status(workflow_id)
    for progress (or wait_for_workflow_result for blocking semantics).

    candidate_returns_path (4d): if provided, the workflow runs the
    inner candidate_pipeline_v2 against this parquet after the council
    finishes. If None, pipeline is skipped with a clear reason in the
    ledger.
    """
    from engine.research.l4_workflow import (
        L4DiscoveryWorkflow, TASK_QUEUE_L4,
    )
    client = await _get_client(address)
    workflow_id = f"l4-{uuid.uuid4().hex[:12]}"
    handle = await client.start_workflow(
        L4DiscoveryWorkflow.run,
        args=[seed_idea, candidate_returns_path],
        id=workflow_id,
        task_queue=TASK_QUEUE_L4,
    )
    return {
        "workflow_id": workflow_id,
        "run_id":      handle.first_execution_run_id,
    }


async def query_workflow_status(
    workflow_id: str,
    *,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
) -> dict:
    """Read live workflow state via Temporal Queries (non-blocking).

    Returns the workflow's current stage + proposal (if reached) +
    consensus (if reached). Workflow status from describe() tells us
    if it's RUNNING / COMPLETED / FAILED / TERMINATED.
    """
    from engine.research.l4_workflow import L4DiscoveryWorkflow
    client = await _get_client(address)
    handle = client.get_workflow_handle(workflow_id)

    try:
        desc = await handle.describe()
        wf_status = desc.status.name if desc.status else "UNKNOWN"
    except Exception as exc:
        return {"workflow_id": workflow_id, "error": str(exc),
                "wf_status": "NOT_FOUND"}

    # Queries — these only work while workflow is RUNNING or completed
    stage = proposal = consensus = paused = None
    for q_name, target in [
        ("get_stage",     "stage"),
        ("get_proposal",  "proposal"),
        ("get_consensus", "consensus"),
        ("is_paused",     "paused"),
    ]:
        try:
            val = await handle.query(q_name)
            if target == "stage":     stage = val
            if target == "proposal":  proposal = val
            if target == "consensus": consensus = val
            if target == "paused":    paused = val
        except Exception:
            pass

    return {
        "workflow_id": workflow_id,
        "wf_status":   wf_status,
        "stage":       stage,
        "proposal":    proposal,
        "consensus":   consensus,
        "paused":      paused,
    }


# ── Signals (Phase 4e human-in-loop) ──────────────────────────────────


async def signal_pause(
    workflow_id: str,
    *,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
) -> dict:
    """Send pause signal to a running workflow. Workflow will block at
    the next workflow.wait_condition checkpoint until resume() is sent.

    Returns {ok, signaled_at} or {error}. Idempotent — sending pause
    twice has no additional effect."""
    import datetime as _dt
    client = await _get_client(address)
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("pause")
    return {
        "ok": True,
        "signaled_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


async def signal_resume(
    workflow_id: str,
    *,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
) -> dict:
    """Send resume signal to a paused workflow. Idempotent."""
    import datetime as _dt
    client = await _get_client(address)
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("resume")
    return {
        "ok": True,
        "signaled_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


async def signal_override_verdict(
    workflow_id: str,
    verdict: str,
    *,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
) -> dict:
    """Send human override of council verdict. The workflow honours
    this instead of LLM consensus when computing downstream routing
    (e.g. whether to run pipeline).

    verdict ∈ {APPROVE, REJECT, NEEDS_REVISION}. Send BEFORE the
    workflow reaches the consensus-routing line for full effect
    (typically while stage is 'critiquing' or earlier)."""
    import datetime as _dt
    valid = {"APPROVE", "REJECT", "NEEDS_REVISION"}
    if verdict not in valid:
        raise ValueError(
            f"verdict must be one of {valid}; got {verdict!r}"
        )
    client = await _get_client(address)
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("override_verdict", verdict)
    return {
        "ok": True,
        "verdict": verdict,
        "signaled_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


async def wait_for_workflow_result(
    workflow_id: str,
    *,
    address: str = DEFAULT_TEMPORAL_ADDRESS,
    timeout: float = 600.0,
) -> dict:
    """Block until the workflow completes; return its final result
    (CouncilWorkflowResult as a dict). Used by tests + the sync REST
    fallback path; the UI doesn't call this."""
    from engine.research.l4_workflow import L4DiscoveryWorkflow
    client = await _get_client(address)
    handle = client.get_workflow_handle(workflow_id)
    result = await asyncio.wait_for(handle.result(), timeout=timeout)
    # Temporal's default JSON serde returns the dataclass as a plain
    # dict to the caller, so check before converting (asdict raises on
    # non-dataclass inputs).
    if isinstance(result, dict):
        return result
    from dataclasses import asdict, is_dataclass
    return asdict(result) if is_dataclass(result) else dict(result)
