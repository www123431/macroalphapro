"""
engine/quant_co_pilot — Quant Research Operations Co-Pilot.

Pre-registration: docs/spec_quant_co_pilot_decision_lineage_v1.md (id=53, hash f1fac6e7e8c1)
Project axis: per project_reframe_quant_alpha_agentic_ops_2026-05-09.md (agentic AI 辅线)

Stage 3a (this module): Tool 1 — Decision Lineage Assistant (ReAct + tool use)
Stage 3b (future): Tool 2 — Verdict Adversarial Reviewer (multi-agent)
Stage 3c (future): Tool 3 — Cross-Spec Pattern Recall (RAG + function calling)
Stage 4  (future): Tool 4 — Autonomous Research Workflow Co-Pilot (Sakana pattern)

This package gives concrete agentic AI capability that passes removal test:
deletes from project = supervisor must manually grep + integrate spec_registry +
git history + memory + verdict.json — measurably worse workflow.

Per spec §六 hash-locked + Tier R audit hooks active.
"""
from engine.quant_co_pilot.base import (
    N_STEPS_MAX_LOCKED,
    COST_BUDGET_USD_LOCKED,
    LATENCY_BUDGET_MS_LOCKED,
    TEMPERATURE_LOCKED,
    TraceStep,
    TraceResult,
    Citation,
    ValidationResult,
    validate_citations,
    run_react_agent,
)
from engine.quant_co_pilot.tools import (
    TOOL_REGISTRY,
    TOOL_NAMES,
    dispatch_tool,
    ToolResult,
)
from engine.quant_co_pilot.decision_lineage import (
    DecisionLineageAgent,
    answer_lineage_query,
)
from engine.quant_co_pilot.eval_harness import (
    EVAL_QUERIES_LOCKED,
    PASS_CRITERIA_LOCKED,
    N_RUNS_PER_QUERY,
    QueryEvalResult,
    EvalReport,
    run_eval,
)

__all__ = [
    "N_STEPS_MAX_LOCKED",
    "COST_BUDGET_USD_LOCKED",
    "LATENCY_BUDGET_MS_LOCKED",
    "TEMPERATURE_LOCKED",
    "TraceStep",
    "TraceResult",
    "Citation",
    "ValidationResult",
    "validate_citations",
    "run_react_agent",
    "TOOL_REGISTRY",
    "TOOL_NAMES",
    "dispatch_tool",
    "ToolResult",
    "DecisionLineageAgent",
    "answer_lineage_query",
    "EVAL_QUERIES_LOCKED",
    "PASS_CRITERIA_LOCKED",
    "N_RUNS_PER_QUERY",
    "QueryEvalResult",
    "EvalReport",
    "run_eval",
]
