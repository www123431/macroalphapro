"""engine/research/candidate_pipeline_v2.py — Phase A: LangGraph
state-machine implementation of candidate_pipeline.

REUSES existing step functions (_run_h10, _run_data_quality, etc.) from
candidate_pipeline v1 to ensure parity. The graph structure replaces
the sequential Python loop, giving:

  - Conditional edges (FAIL at H10 short-circuits past data_quality)
  - State persistence (resumable, time-travel debug-able)
  - SSE streaming support (Phase A.2 — frontend live progress)
  - Per-node retry semantics (Phase A.3 — transient WRDS failures)
  - Multi-agent council substrate (Phase B — each agent is a node)

PARITY TARGET (Phase A.1 must-pass):
  v2.run_candidate_pipeline(...).final_decision == v1.run(...).final_decision
  for at least PIT SN + LTR test cases.

Streaming + retry + frontend wire = Phase A.2-A.5 (next sessions).
"""
from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import pandas as pd
from langgraph.graph import END, START, StateGraph

# Phase 4a — durable checkpointer. SqliteSaver matches single-user lab
# scale (NOT Postgres per [[project-orchestration-rfc-2026-06-01]] §3.1).
from langgraph.checkpoint.sqlite import SqliteSaver


class _PickleSerde:
    """Pickle-based SerializerProtocol for SqliteSaver.

    Why pickle, not msgpack: CandidateState carries pd.Series +
    StepResult dataclasses + nested pd.DataFrame across nodes — none of
    which the default ormsgpack serializer handles. Pickle handles all
    of them natively without per-type registration.

    Tradeoff: pickle is Python-version + class-layout coupled. If we
    rename CandidateState fields, prior checkpoints become unreadable.
    For single-user lab scale that is acceptable; the outer Temporal
    ring (4c+) will use msgpack with explicit Activity input/output
    types — that boundary is what survives version bumps.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, blob = data
        if type_ != "pickle":
            raise ValueError(f"_PickleSerde got non-pickle type={type_!r}")
        return pickle.loads(blob)

# Reuse v1 step functions wholesale — don't reinvent
from engine.research.candidate_pipeline import (
    PipelineReport,
    StepResult,
    _classify_replacement_or_addition,
    _compute_meta_decision,
    _run_ablation_vs_parent,
    _run_block_bootstrap_significance,
    _run_quarter_concentration,
    _run_correlation_matrix,
    _run_cost_model_check,
    _run_data_quality_check,
    _run_devils_advocate,
    _run_factor_budget_delta,
    _run_graveyard_check,
    _run_h10,
    _run_h2,
    _run_h6,
    _run_h7,
    _run_honest_deploy_sharpe,
    _run_multi_aum_check,
    _run_regime_stratified,
    _run_sub_period_robustness,
)

logger = logging.getLogger(__name__)


# ── State definition ──────────────────────────────────────────────────


@dataclass
class CandidateState:
    """State that flows through the LangGraph pipeline."""

    # Inputs
    candidate_returns: pd.Series
    proposal_name: str = "candidate"
    proposed_role: Optional[str] = None
    mechanism_id: Optional[str] = None
    proposal_dict: Optional[dict] = None
    parent_returns_path: Optional[str] = None
    phase: int = 3

    # Accumulated results
    step_results: list[StepResult] = field(default_factory=list)
    h10_pl: Optional[dict] = None
    role_used: Optional[str] = None
    role_inferred: bool = False
    h10_accept: bool = False
    manifest: Optional[dict] = None

    # Control flow
    short_circuited: bool = False
    short_circuit_reason: str = ""

    # Outputs
    final_decision: Optional[str] = None
    rationale: Optional[str] = None
    candidate_relation: str = "UNKNOWN"
    most_correlated_sleeve: Optional[str] = None
    most_correlated_value: Optional[float] = None


# ── Helper to build candidate_info for hygiene checks ─────────────────


def _build_candidate_info(proposal_dict: dict | None) -> dict | None:
    if not proposal_dict:
        return None
    return {
        "family":         proposal_dict.get("family"),
        "parent_family":  proposal_dict.get("parent_family"),
        "required_data":  proposal_dict.get("required_data") or [],
        "economics_text": proposal_dict.get("economics_text", ""),
        "post_pub_decay": proposal_dict.get("post_pub_decay") or {},
    }


# ── Node functions (thin wrappers around v1 step functions) ───────────


def node_build_manifest(state: CandidateState) -> CandidateState:
    try:
        from engine.research.repro_manifest import build_manifest
        state.manifest = build_manifest(pipeline_config={
            "phase":         state.phase,
            "proposed_role": state.proposed_role,
            "mechanism_id":  state.mechanism_id,
            "proposal_name": state.proposal_name,
            "version":       "v2_langgraph",
        })
    except Exception as exc:
        logger.warning("manifest build failed: %s", exc)
    return state


def node_h10(state: CandidateState) -> CandidateState:
    step, role_used, role_inferred, h10_accept, h10_pl_cache = _run_h10(
        state.candidate_returns, state.proposal_name,
        state.proposed_role, state.phase,
    )
    state.step_results.append(step)
    state.role_used = role_used
    state.role_inferred = role_inferred
    state.h10_accept = h10_accept
    state.h10_pl = h10_pl_cache
    if step.status == "FAIL":
        state.short_circuited = True
        state.short_circuit_reason = "H10 rejected"
    return state


def node_data_quality(state: CandidateState) -> CandidateState:
    step = _run_data_quality_check(state.candidate_returns)
    state.step_results.append(step)
    if step.status == "FAIL":
        state.short_circuited = True
        state.short_circuit_reason = "data quality hard-fail"
    return state


def node_h2(state: CandidateState) -> CandidateState:
    if not state.mechanism_id:
        state.step_results.append(StepResult(
            step_name="H2_cousin_check", status="SKIP",
            key_findings={"reason": "no mechanism_id provided"},
            verdict="skipped — provide mechanism_id to enable",
        ))
        return state
    ci = _build_candidate_info(state.proposal_dict)
    step = _run_h2(state.mechanism_id, candidate_info=ci)
    state.step_results.append(step)
    if step.status == "FAIL":
        state.short_circuited = True
        state.short_circuit_reason = "H2 hard-reject"
    return state


def node_h6(state: CandidateState) -> CandidateState:
    if not state.mechanism_id:
        state.step_results.append(StepResult(
            step_name="H6_post_pub_evidence", status="SKIP",
            key_findings={"reason": "no mechanism_id provided"},
            verdict="skipped — provide mechanism_id to enable",
        ))
        return state
    ci = _build_candidate_info(state.proposal_dict)
    step = _run_h6(state.mechanism_id, candidate_info=ci)
    state.step_results.append(step)
    if step.status == "FAIL":
        state.short_circuited = True
        state.short_circuit_reason = "H6 post-pub rejected"
    return state


def node_h7(state: CandidateState) -> CandidateState:
    if not state.proposal_dict:
        state.step_results.append(StepResult(
            step_name="H7_kill_this_proposal", status="SKIP",
            key_findings={"reason": "no proposal dict"},
            verdict="skipped — supply proposal dict to enable",
        ))
        return state
    step = _run_h7(state.proposal_dict)
    state.step_results.append(step)
    if step.status == "FAIL":
        state.short_circuited = True
        state.short_circuit_reason = "H7 fatal"
    return state


def node_graveyard(state: CandidateState) -> CandidateState:
    step = _run_graveyard_check(
        state.proposal_name, state.mechanism_id, state.candidate_returns,
    )
    state.step_results.append(step)
    return state


def node_cost_model(state: CandidateState) -> CandidateState:
    step = _run_cost_model_check(state.candidate_returns, state.proposal_name)
    state.step_results.append(step)
    return state


def node_regime_stratified(state: CandidateState) -> CandidateState:
    step = _run_regime_stratified(
        state.candidate_returns, state.role_used or state.proposed_role,
        state.phase, state.proposal_name,
    )
    state.step_results.append(step)
    # For insurance role, regime FAIL is blocking per v1
    if step.status == "FAIL" and (state.role_used or "") == "insurance":
        state.short_circuited = True
        state.short_circuit_reason = "insurance hypothesis fails regime test"
    return state


def node_factor_budget(state: CandidateState) -> CandidateState:
    step = _run_factor_budget_delta(
        state.candidate_returns, state.role_used or state.proposed_role,
        state.proposal_name, state.phase,
    )
    state.step_results.append(step)
    return state


def node_multi_aum(state: CandidateState) -> CandidateState:
    step = _run_multi_aum_check(state.candidate_returns, state.proposal_name)
    state.step_results.append(step)
    return state


def node_sub_period(state: CandidateState) -> CandidateState:
    step = _run_sub_period_robustness(
        state.candidate_returns, state.phase, state.proposal_name,
    )
    state.step_results.append(step)
    return state


def node_correlation(state: CandidateState) -> CandidateState:
    step = _run_correlation_matrix(state.candidate_returns, state.proposal_name)
    state.step_results.append(step)
    return state


def node_ablation(state: CandidateState) -> CandidateState:
    step = _run_ablation_vs_parent(
        state.candidate_returns, state.proposal_name,
        state.parent_returns_path,
    )
    state.step_results.append(step)
    return state


def node_block_bootstrap_significance(state: CandidateState) -> CandidateState:
    """Phase 5.2 — P-D8: statistical significance of Sharpe-diff vs
    parent via paired stationary block bootstrap. Complements the
    P-D7 ablation heuristic."""
    step = _run_block_bootstrap_significance(
        state.candidate_returns, state.proposal_name,
        state.parent_returns_path,
    )
    state.step_results.append(step)
    return state


def node_quarter_concentration(state: CandidateState) -> CandidateState:
    """Phase 5.4 — P-D9: per-quarter return distribution +
    concentration-risk verdict. Catches lucky-quarter dependence."""
    step = _run_quarter_concentration(
        state.candidate_returns, state.proposal_name,
    )
    state.step_results.append(step)
    return state


def node_honest_deploy_sharpe(state: CandidateState) -> CandidateState:
    step = _run_honest_deploy_sharpe(
        state.candidate_returns, state.proposal_name, state.mechanism_id,
    )
    state.step_results.append(step)
    return state


def node_devils_advocate(state: CandidateState) -> CandidateState:
    # Pre-compute relation so DA has context (11th catch fix)
    pre_relation, pre_top_sleeve, pre_top_val = \
        _classify_replacement_or_addition(state.step_results)
    step = _run_devils_advocate(
        state.proposal_name, state.role_used or state.proposed_role,
        state.h10_pl, state.step_results,
        candidate_relation=pre_relation,
        most_correlated_sleeve=pre_top_sleeve,
        most_correlated_value=pre_top_val,
    )
    state.step_results.append(step)
    return state


def node_compute_meta_decision(state: CandidateState) -> CandidateState:
    final_decision, rationale = _compute_meta_decision(
        state.step_results, state.h10_accept,
    )
    state.final_decision = final_decision
    state.rationale = rationale

    relation, mc_sleeve, mc_value = _classify_replacement_or_addition(
        state.step_results,
    )
    state.candidate_relation = relation
    state.most_correlated_sleeve = mc_sleeve
    state.most_correlated_value = mc_value

    # REPLACEMENT pathway re-classification (P-D6 doctrine)
    if relation == "REPLACEMENT" and final_decision in (
            "BORDERLINE_REVIEW", "SOFT_REJECT"):
        state.final_decision = "PROMOTE_AS_REPLACEMENT"
        state.rationale = (
            f"candidate is REPLACEMENT for existing sleeve "
            f"{mc_sleeve!r} (corr {mc_value:+.2f}). The correlation "
            f"WARN is EXPECTED and re-classified as informational. "
            f"Re-audit cost_model + factor_exposure + capacity for the "
            f"variant before deploy."
        )
    return state


def node_short_circuit_end(state: CandidateState) -> CandidateState:
    """Emit HARD_REJECT when a critical step failed."""
    last = state.step_results[-1] if state.step_results else None
    state.final_decision = "HARD_REJECT"
    state.rationale = f"{state.short_circuit_reason}: {last.verdict if last else 'unknown'}"
    return state


# ── Conditional edge routers ──────────────────────────────────────────


def route_after_h10(state: CandidateState) -> Literal["short_circuit", "continue"]:
    return "short_circuit" if state.short_circuited else "continue"


def route_after_dq(state: CandidateState) -> Literal["short_circuit", "continue"]:
    return "short_circuit" if state.short_circuited else "continue"


def route_after_h2(state: CandidateState) -> Literal["short_circuit", "continue"]:
    return "short_circuit" if state.short_circuited else "continue"


def route_after_h6(state: CandidateState) -> Literal["short_circuit", "continue"]:
    return "short_circuit" if state.short_circuited else "continue"


def route_after_h7(state: CandidateState) -> Literal["short_circuit", "continue"]:
    return "short_circuit" if state.short_circuited else "continue"


def route_after_regime(state: CandidateState) -> Literal["short_circuit", "continue"]:
    return "short_circuit" if state.short_circuited else "continue"


# ── Graph builder ──────────────────────────────────────────────────────


def build_pipeline_graph() -> StateGraph:
    """Build the 15-step pipeline as a LangGraph state machine.

    All edges + conditional routing match v1's sequential + short-
    circuit logic precisely to ensure parity.
    """
    g = StateGraph(CandidateState)

    # Phase A.3: per-step retry policy (transient WRDS / network errors)
    # Each step retries up to 2 times with 1s backoff before propagating.
    from langgraph.types import RetryPolicy
    retry_policy = RetryPolicy(max_attempts=2, initial_interval=1.0,
                                backoff_factor=2.0)

    # Add all nodes — most with retry; pure-compute nodes (manifest +
    # short-circuit) don't need retry.
    g.add_node("build_manifest",        node_build_manifest)
    g.add_node("h10",                   node_h10, retry_policy=retry_policy)
    g.add_node("data_quality",          node_data_quality)
    g.add_node("h2",                    node_h2, retry_policy=retry_policy)
    g.add_node("h6",                    node_h6, retry_policy=retry_policy)
    g.add_node("h7",                    node_h7)
    g.add_node("graveyard",             node_graveyard, retry_policy=retry_policy)
    g.add_node("cost_model",            node_cost_model)
    g.add_node("regime_stratified",     node_regime_stratified,
               retry_policy=retry_policy)
    g.add_node("factor_budget",         node_factor_budget,
               retry_policy=retry_policy)
    g.add_node("multi_aum",             node_multi_aum)
    g.add_node("sub_period",            node_sub_period,
               retry_policy=retry_policy)
    g.add_node("correlation",           node_correlation,
               retry_policy=retry_policy)
    g.add_node("ablation",              node_ablation)
    g.add_node("block_bootstrap_significance", node_block_bootstrap_significance)
    g.add_node("quarter_concentration", node_quarter_concentration)
    g.add_node("honest_deploy_sharpe",  node_honest_deploy_sharpe)
    g.add_node("devils_advocate",       node_devils_advocate,
               retry_policy=retry_policy)
    g.add_node("compute_meta_decision", node_compute_meta_decision)
    g.add_node("short_circuit_end",     node_short_circuit_end)

    # Edges
    g.add_edge(START, "build_manifest")
    g.add_edge("build_manifest", "h10")
    g.add_conditional_edges("h10", route_after_h10, {
        "short_circuit": "short_circuit_end",
        "continue":      "data_quality",
    })
    g.add_conditional_edges("data_quality", route_after_dq, {
        "short_circuit": "short_circuit_end",
        "continue":      "h2",
    })
    g.add_conditional_edges("h2", route_after_h2, {
        "short_circuit": "short_circuit_end",
        "continue":      "h6",
    })
    g.add_conditional_edges("h6", route_after_h6, {
        "short_circuit": "short_circuit_end",
        "continue":      "h7",
    })
    g.add_conditional_edges("h7", route_after_h7, {
        "short_circuit": "short_circuit_end",
        "continue":      "graveyard",
    })
    g.add_edge("graveyard",        "cost_model")
    g.add_edge("cost_model",       "regime_stratified")
    g.add_conditional_edges("regime_stratified", route_after_regime, {
        "short_circuit": "short_circuit_end",
        "continue":      "factor_budget",
    })
    g.add_edge("factor_budget",         "multi_aum")
    g.add_edge("multi_aum",             "sub_period")
    g.add_edge("sub_period",            "correlation")
    g.add_edge("correlation",           "ablation")
    g.add_edge("ablation",              "block_bootstrap_significance")
    g.add_edge("block_bootstrap_significance", "quarter_concentration")
    g.add_edge("quarter_concentration", "honest_deploy_sharpe")
    g.add_edge("honest_deploy_sharpe",  "devils_advocate")
    g.add_edge("devils_advocate",       "compute_meta_decision")
    g.add_edge("compute_meta_decision", END)
    g.add_edge("short_circuit_end",     END)

    return g


# ── Durable checkpointing (Phase 4a) ──────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DB = REPO_ROOT / "data" / "research" / \
    "langgraph_checkpoints.db"


@contextmanager
def _sqlite_checkpointer(
    db_path: str | Path | None = None,
) -> Iterator[SqliteSaver]:
    """Yield a SqliteSaver backed by a file under data/research/.

    SqliteSaver requires a sqlite3.Connection; we manage one inside the
    context so callers don't need to know the wire format. Single-user
    lab assumption — no connection pooling needed.
    """
    path = Path(db_path or os.environ.get(
        "LANGGRAPH_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=_PickleSerde())
    finally:
        conn.close()


def make_thread_id(proposal_name: str) -> str:
    """Generate a deterministic-prefix thread_id for one pipeline run.

    Format: '{proposal_slug}-{uuid4_short}' — slug is human-greppable in
    the checkpoint DB; uuid4 prevents accidental collision when the same
    proposal is re-run."""
    slug = "".join(c if c.isalnum() else "_"
                   for c in proposal_name.strip().lower())[:32] or "candidate"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


# ── Public entrypoint ─────────────────────────────────────────────────


def run_candidate_pipeline_v2(
    candidate_returns: pd.Series,
    proposal_name: str = "candidate",
    proposed_role: str | None = None,
    mechanism_id: str | None = None,
    proposal_dict: dict | None = None,
    parent_returns_path: str | None = None,
    phase: int = 3,
    *,
    thread_id: str | None = None,
    durable: bool = True,
) -> PipelineReport:
    """v2 LangGraph pipeline. Same signature + return type as v1 for
    drop-in compatibility.

    PARITY: v2 must produce identical final_decision + step_results
    structure as v1 for the same input. Verified via
    tests/test_candidate_pipeline_v2_parity.py.
    """
    initial_state = CandidateState(
        candidate_returns=candidate_returns,
        proposal_name=proposal_name,
        proposed_role=proposed_role,
        mechanism_id=mechanism_id,
        proposal_dict=proposal_dict,
        parent_returns_path=parent_returns_path,
        phase=phase,
    )

    if durable:
        tid = thread_id or make_thread_id(proposal_name)
        config = {"configurable": {"thread_id": tid}}
        with _sqlite_checkpointer() as ckpt:
            graph = build_pipeline_graph().compile(checkpointer=ckpt)
            final_state_dict = graph.invoke(initial_state, config=config)
    else:
        # in-memory only — used by parity tests and quick smoke runs
        graph = build_pipeline_graph().compile()
        final_state_dict = graph.invoke(initial_state)

    # LangGraph returns a dict for AddableValuesDict; rebuild dataclass
    fs = CandidateState(**{k: final_state_dict[k]
                           for k in CandidateState.__dataclass_fields__
                           if k in final_state_dict})

    return PipelineReport(
        proposal_name=fs.proposal_name,
        role_used=fs.role_used,
        role_was_inferred=fs.role_inferred,
        step_results=fs.step_results,
        final_decision=fs.final_decision or "UNKNOWN",
        rationale=fs.rationale or "",
        candidate_relation=fs.candidate_relation,
        most_correlated_sleeve=fs.most_correlated_sleeve,
        most_correlated_value=fs.most_correlated_value,
        reproducibility_manifest=fs.manifest,
    )


# ── Resume / inspect (Phase 4a) ───────────────────────────────────────


def get_checkpoint_state(thread_id: str) -> dict | None:
    """Inspect the latest checkpoint for a thread_id.

    Returns the dict of CandidateState values at the last completed
    node, or None if the thread is unknown. Used by:
      - resume_candidate_pipeline_v2 — to continue from interruption
      - L4 observability — to query "what did pipeline X find at H8"
      - frontend — to render the in-progress state
    """
    config = {"configurable": {"thread_id": thread_id}}
    with _sqlite_checkpointer() as ckpt:
        graph = build_pipeline_graph().compile(checkpointer=ckpt)
        snap = graph.get_state(config)
        if snap is None or not snap.values:
            return None
        return dict(snap.values)


def resume_candidate_pipeline_v2(thread_id: str) -> PipelineReport:
    """Resume an interrupted pipeline run from its last checkpoint.

    Use case: process died mid-pipeline (e.g., during the slow factor-
    exposure delta step). Re-running with the same thread_id picks up
    from the last committed checkpoint instead of replaying from start.

    Raises ValueError if no prior state exists for thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    with _sqlite_checkpointer() as ckpt:
        graph = build_pipeline_graph().compile(checkpointer=ckpt)
        snap = graph.get_state(config)
        if snap is None or not snap.values:
            raise ValueError(
                f"no checkpoint found for thread_id={thread_id!r}; "
                f"use run_candidate_pipeline_v2(...) for a new run"
            )
        # invoke with None input continues from last checkpoint
        final_state_dict = graph.invoke(None, config=config)

    fs = CandidateState(**{k: final_state_dict[k]
                           for k in CandidateState.__dataclass_fields__
                           if k in final_state_dict})

    return PipelineReport(
        proposal_name=fs.proposal_name,
        role_used=fs.role_used,
        role_was_inferred=fs.role_inferred,
        step_results=fs.step_results,
        final_decision=fs.final_decision or "UNKNOWN",
        rationale=fs.rationale or "",
        candidate_relation=fs.candidate_relation,
        most_correlated_sleeve=fs.most_correlated_sleeve,
        most_correlated_value=fs.most_correlated_value,
        reproducibility_manifest=fs.manifest,
    )
