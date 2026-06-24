"""
engine/quant_co_pilot/decision_lineage.py — Decision Lineage Agent wrapper.

Pre-registration: docs/spec_quant_co_pilot_decision_lineage_v1.md (id=53)

Thin wrapper around base.run_react_agent + tools.dispatch_tool that exposes
a clean public API for UI / eval harness / programmatic callers.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from engine.quant_co_pilot.base import (
    TraceResult,
    N_STEPS_MAX_LOCKED,
    COST_BUDGET_USD_LOCKED,
    LATENCY_BUDGET_MS_LOCKED,
    run_react_agent,
)
from engine.quant_co_pilot.tools import (
    TOOL_DESCRIPTIONS,
    TOOL_NAMES,
    dispatch_tool,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DecisionLineageAgent:
    """Agent that answers project-history queries via ReAct + 9 tools.

    Use:
        agent = DecisionLineageAgent()
        result = agent.answer("为什么 PRODUCTION_SIGNAL=ql01_bab?")
        print(result.annotated_answer)
        print(f"cost ${result.cost_usd:.4f}, latency {result.latency_ms}ms")
    """
    max_steps:            int   = N_STEPS_MAX_LOCKED
    cost_budget_usd:      float = COST_BUDGET_USD_LOCKED
    latency_budget_ms:    int   = LATENCY_BUDGET_MS_LOCKED

    def answer(self, query: str) -> TraceResult:
        """Run a single query through the ReAct agent.

        Returns:
            TraceResult including final_answer, citations, annotated_answer,
            full step trace, cost, latency, abort_reason if any.
        """
        return run_react_agent(
            query=query,
            tool_dispatcher=dispatch_tool,
            tool_descriptions=TOOL_DESCRIPTIONS,
            max_steps=self.max_steps,
            cost_budget_usd=self.cost_budget_usd,
            latency_budget_ms=self.latency_budget_ms,
            valid_tool_names=set(TOOL_NAMES),
        )


def answer_lineage_query(query: str) -> TraceResult:
    """Convenience function for one-shot query."""
    return DecisionLineageAgent().answer(query)
