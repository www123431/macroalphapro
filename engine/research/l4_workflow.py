"""engine/research/l4_workflow.py — Phase 4c: Temporal workflow +
activities for the L4 outer-ring discovery loop.

Per orchestration RFC §3.2 — the OUTER ring (long-running, durable,
human-in-loop signals) is Temporal. The INNER ring (deterministic
18-step pipeline) stays as LangGraph (engine.research.candidate_
pipeline_v2). This module wires the outer to call the inner.

Workflow:
  L4DiscoveryWorkflow(seed_idea) → CouncilWorkflowResult
    ├─ Activity propose_activity(seed) → ProposalDict (architect)
    ├─ Activity critique_activity(proposal) → CouncilVerdict (theorist + DA)
    └─ (Future 4d: pipeline_activity + ledger_activity)

Activities call into the existing sync agent_council functions via
to_thread; workflow code itself contains no LLM calls (deterministic
replay safety).

Key Temporal semantics this skeleton enforces:
  - Activities have own retry policy (transient LLM errors retry)
  - Workflow signals: human can pause / override (placeholder for 4e)
  - Workflow queries: UI can read live state without blocking
  - Task queue: l4-discovery — separate from any future TQs
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Optional

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# Avoid heavy imports inside @workflow.defn body — sandbox-safe by
# guarding with workflow.unsafe.imports_passed_through()
with workflow.unsafe.imports_passed_through():
    from engine.research.agent_council import (
        CouncilVerdict, ProposalDict,
    )

logger = logging.getLogger(__name__)

TASK_QUEUE_L4 = "l4-discovery"


# ── Activity input / output dataclasses ────────────────────────────────


@dataclass
class ProposeInput:
    seed_idea: str


@dataclass
class CritiqueInput:
    proposal_dict: dict  # serialized ProposalDict
    # Frontier 1 (2026-06-01): opt-in structured reflection round.
    enable_reflection: bool = False


@dataclass
class PipelineInput:
    proposal_dict: dict
    candidate_returns_path: Optional[str] = None


@dataclass
class LedgerInput:
    workflow_id: str
    proposal_dict: dict
    council_dict: dict
    pipeline_report_dict: Optional[dict] = None
    elapsed_s: float = 0.0
    human_override_verdict: Optional[str] = None  # 4e: surfaced for audit


@dataclass
class CouncilWorkflowResult:
    seed_idea: str
    proposal_dict: dict
    consensus: str
    rationale: str
    run_id: str                       # council_runs.jsonl row id
    n_critics: int
    elapsed_s: float
    # 4d additions
    iteration_id: Optional[str] = None       # l4_iterations.jsonl row
    pipeline_ran: bool = False
    pipeline_final_decision: Optional[str] = None
    verdict_alignment: Optional[str] = None


# ── Activities (heavy work — LLM calls live here) ──────────────────────


@activity.defn(name="l4_propose_activity")
async def propose_activity(inp: ProposeInput) -> dict:
    """Run architect_propose synchronously off-event-loop. Returns the
    ProposalDict serialized to a plain dict (Temporal-friendly)."""
    from engine.research.agent_council import architect_propose
    from engine.research.trace_log import span, start_trace
    workflow_id = activity.info().workflow_id
    start_trace(workflow_id=workflow_id)
    with span("activity.propose", kind_class="activity",
              workflow_id=workflow_id):
        prop = await asyncio.to_thread(architect_propose, inp.seed_idea)
    return prop.to_dict()


@activity.defn(name="l4_critique_activity")
async def critique_activity(inp: CritiqueInput) -> dict:
    """Run critique_council (theorist + DA fan-out). The Stage-B
    parallelism inside critique_council is its own asyncio.gather —
    Temporal sees this activity as one unit."""
    from engine.research.agent_council import (
        ProposalDict, critique_council,
    )
    from engine.research.trace_log import span, start_trace
    workflow_id = activity.info().workflow_id
    start_trace(workflow_id=workflow_id)
    with span("activity.critique", kind_class="activity",
              workflow_id=workflow_id):
        prop = ProposalDict(**inp.proposal_dict)
        council = await critique_council(
            prop, enable_reflection=inp.enable_reflection,
        )
    return council.to_dict()


async def _pipeline_activity_body(inp: PipelineInput) -> dict:
    """Run the deterministic 18-step candidate_pipeline_v2 against
    the architect's proposal. If no candidate_returns parquet is
    provided, skip with a clear reason — this is the honest signal
    that L4 needs a data-loader before it can fully autonomous-run.

    NOT durable on the inner ring (durable=False): Temporal IS the
    durability boundary for the outer ring, double-checkpointing
    would just bloat the inner ring SQLite for no recovery benefit.
    """
    if not inp.candidate_returns_path:
        return {
            "ran":            False,
            "skipped_reason": "no candidate_returns_path provided",
            "final_decision": None,
            "rationale":      "L4 autonomous-mode does not yet build "
                              "candidate returns from proposal — pass "
                              "candidate_returns_path to run the inner "
                              "pipeline empirically.",
            "step_results":   [],
            "candidate_returns_path": None,
        }

    from dataclasses import asdict
    from pathlib import Path
    import pandas as pd

    from engine.research.candidate_pipeline_v2 import (
        run_candidate_pipeline_v2,
    )

    p = Path(inp.candidate_returns_path)
    if not p.is_file():
        return {
            "ran":            False,
            "skipped_reason": f"parquet not found: {inp.candidate_returns_path}",
            "final_decision": None,
            "rationale":      "",
            "step_results":   [],
            "candidate_returns_path": str(p),
        }
    series = pd.read_parquet(p).iloc[:, 0]
    series.index = pd.to_datetime(series.index)

    report = await asyncio.to_thread(
        run_candidate_pipeline_v2,
        candidate_returns=series,
        proposal_name=inp.proposal_dict.get("title", "candidate"),
        proposed_role=inp.proposal_dict.get("proposed_role"),
        mechanism_id=inp.proposal_dict.get("mechanism_id"),
        proposal_dict=inp.proposal_dict,
        durable=False,
    )
    rep = asdict(report)
    rep["ran"] = True
    rep["candidate_returns_path"] = str(p)
    return rep


@activity.defn(name="l4_pipeline_activity")
async def pipeline_activity(inp: PipelineInput) -> dict:
    """Trace-wrapped pipeline activity. Body lives in
    _pipeline_activity_body (kept separate so the activity decorator
    sees a simple wrapper)."""
    from engine.research.trace_log import span, start_trace
    workflow_id = activity.info().workflow_id
    start_trace(workflow_id=workflow_id)
    with span("activity.pipeline", kind_class="activity",
              workflow_id=workflow_id,
              has_returns_path=bool(inp.candidate_returns_path)):
        return await _pipeline_activity_body(inp)


@activity.defn(name="l4_ledger_activity")
async def ledger_activity(inp: LedgerInput) -> dict:
    """Append one row to the L4 iteration ledger. Returns the
    iteration_id so the workflow can surface it in its result."""
    from engine.research.outcome_ledger import append_l4_iteration
    from engine.research.trace_log import span, start_trace
    workflow_id = activity.info().workflow_id
    start_trace(workflow_id=workflow_id)
    with span("activity.ledger", kind_class="activity",
              workflow_id=workflow_id):
        iteration_id = await asyncio.to_thread(
            append_l4_iteration,
            workflow_id=inp.workflow_id,
            proposal=inp.proposal_dict,
            council=inp.council_dict,
            pipeline_report=inp.pipeline_report_dict,
            elapsed_s=inp.elapsed_s,
            human_override_verdict=inp.human_override_verdict,
        )
    return {"iteration_id": iteration_id}


# ── Workflow ──────────────────────────────────────────────────────────


@workflow.defn(name="L4DiscoveryWorkflow")
class L4DiscoveryWorkflow:
    """One discovery iteration: propose → critique → return.

    Stage 4c skeleton. Stage 4d will append: pipeline_activity (calls
    candidate_pipeline_v2) + ledger_activity (writes outcome ledger).
    Stage 4e will add Signals (pause/override) + Queries (live state).
    """

    def __init__(self) -> None:
        self._stage: str = "init"
        self._proposal_dict: Optional[dict] = None
        self._consensus: Optional[str] = None
        self._paused: bool = False
        self._override_verdict: Optional[str] = None
        # 4d additions surfaced via Query
        self._pipeline_decision: Optional[str] = None
        self._iteration_id: Optional[str] = None

    # ── Queries (UI reads, never block) ──────────────────────────────

    @workflow.query
    def get_stage(self) -> str:
        return self._stage

    @workflow.query
    def get_proposal(self) -> Optional[dict]:
        return self._proposal_dict

    @workflow.query
    def get_consensus(self) -> Optional[str]:
        return self._consensus

    @workflow.query
    def is_paused(self) -> bool:
        return self._paused

    @workflow.query
    def get_pipeline_decision(self) -> Optional[str]:
        return self._pipeline_decision

    @workflow.query
    def get_iteration_id(self) -> Optional[str]:
        return self._iteration_id

    # ── Signals (human-in-loop interventions) ────────────────────────

    @workflow.signal
    def pause(self) -> None:
        self._paused = True

    @workflow.signal
    def resume(self) -> None:
        self._paused = False

    @workflow.signal
    def override_verdict(self, verdict: str) -> None:
        """Human stamps a verdict in. After 4e the UI exposes a button
        for this; the workflow honours it instead of LLM consensus."""
        self._override_verdict = verdict

    # ── Main flow ────────────────────────────────────────────────────

    @workflow.run
    async def run(
        self,
        seed_idea: str,
        candidate_returns_path: Optional[str] = None,
        enable_reflection: bool = False,
    ) -> CouncilWorkflowResult:
        # Block here if pre-paused (4e human-in-loop, no-op for now)
        await workflow.wait_condition(lambda: not self._paused)

        wf_start_s = workflow.time()

        # Stage A: architect proposes
        self._stage = "proposing"
        retry = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_attempts=3,
            backoff_coefficient=2.0,
        )
        proposal_dict = await workflow.execute_activity(
            propose_activity,
            ProposeInput(seed_idea=seed_idea),
            start_to_close_timeout=timedelta(minutes=3),
            retry_policy=retry,
        )
        self._proposal_dict = proposal_dict

        # Optional human break (4e)
        await workflow.wait_condition(lambda: not self._paused)

        # Stage B: critique council
        self._stage = "critiquing"
        council_dict = await workflow.execute_activity(
            critique_activity,
            CritiqueInput(
                proposal_dict=proposal_dict,
                enable_reflection=enable_reflection,
            ),
            start_to_close_timeout=timedelta(minutes=10) if enable_reflection
                                   else timedelta(minutes=5),
            retry_policy=retry,
        )
        consensus = self._override_verdict or council_dict.get("consensus", "UNKNOWN")
        self._consensus = consensus

        # Optional human break before expensive pipeline (4e)
        await workflow.wait_condition(lambda: not self._paused)

        # Stage C: pipeline — run for APPROVE + NEEDS_REVISION,
        # skip for REJECT (don't burn pipeline compute on dead ideas)
        pipeline_report_dict: Optional[dict] = None
        if consensus != "REJECT":
            self._stage = "running_pipeline"
            pipeline_report_dict = await workflow.execute_activity(
                pipeline_activity,
                PipelineInput(
                    proposal_dict=proposal_dict,
                    candidate_returns_path=candidate_returns_path,
                ),
                start_to_close_timeout=timedelta(minutes=20),
                retry_policy=retry,
            )
            self._pipeline_decision = pipeline_report_dict.get("final_decision")

        # Stage D: ledger — write the iteration row (even if pipeline
        # skipped — we still want the council outcome recorded)
        self._stage = "writing_ledger"
        elapsed_s = workflow.time() - wf_start_s
        ledger_result = await workflow.execute_activity(
            ledger_activity,
            LedgerInput(
                workflow_id=workflow.info().workflow_id,
                proposal_dict=proposal_dict,
                council_dict=council_dict,
                pipeline_report_dict=pipeline_report_dict,
                elapsed_s=elapsed_s,
                human_override_verdict=self._override_verdict,
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry,
        )
        self._iteration_id = ledger_result.get("iteration_id")
        self._stage = "done"

        # Verdict alignment computed during ledger write — re-derive
        # here so the workflow result carries it without an extra read
        from engine.research.outcome_ledger import _classify_alignment
        alignment = _classify_alignment(
            consensus,
            (pipeline_report_dict or {}).get("final_decision"),
        )

        return CouncilWorkflowResult(
            seed_idea=seed_idea,
            proposal_dict=proposal_dict,
            consensus=consensus,
            rationale=council_dict.get("rationale", ""),
            run_id=council_dict.get("run_id", ""),
            n_critics=len(council_dict.get("verdicts") or []),
            elapsed_s=elapsed_s,
            iteration_id=self._iteration_id,
            pipeline_ran=bool(pipeline_report_dict
                              and pipeline_report_dict.get("ran")),
            pipeline_final_decision=self._pipeline_decision,
            verdict_alignment=alignment,
        )


# ── Worker entrypoint ─────────────────────────────────────────────────


async def run_worker(address: str = "localhost:7233") -> None:
    """Long-running worker process. Started via
    `python -m engine.research.l4_worker`. Picks tasks off the
    l4-discovery + l4-cron TQs and runs activities + workflows in
    this process.

    2026-06-01: Frontier 2 added L4CronWorkflow + 2 cron activities
    on a sibling task queue so the cron tier can be scaled/paused
    independently of ad-hoc trigger traffic. Both TQs run inside ONE
    worker process by default — splitting workers is only needed when
    traffic grows enough that cron starvation becomes a risk.
    """
    from temporalio.client import Client
    from temporalio.worker import Worker

    from engine.research.l4_cron import (
        L4CronWorkflow, TASK_QUEUE_L4_CRON,
        log_cron_run_activity, pick_seed_activity,
    )

    client = await Client.connect(address)

    discovery_worker = Worker(
        client,
        task_queue=TASK_QUEUE_L4,
        workflows=[L4DiscoveryWorkflow],
        activities=[
            propose_activity, critique_activity,
            pipeline_activity, ledger_activity,
        ],
    )
    cron_worker = Worker(
        client,
        task_queue=TASK_QUEUE_L4_CRON,
        workflows=[L4CronWorkflow],
        activities=[pick_seed_activity, log_cron_run_activity],
    )
    logger.info(
        "L4 worker started — discovery_tq=%s, cron_tq=%s, address=%s",
        TASK_QUEUE_L4, TASK_QUEUE_L4_CRON, address,
    )
    await asyncio.gather(discovery_worker.run(), cron_worker.run())
