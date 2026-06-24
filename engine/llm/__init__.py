"""
engine/llm/ — multi-provider LLM wrapper.

Per [[feedback-llm-provider-role-specialization-2026-05-19]]: workloads
are routed to providers based on substantive fit (tool-use reliability,
long context, persona, cost), not vendor allegiance. This module is
the thin function-based abstraction (not a full Router class — YAGNI
for current single-user MSBA scale).

Public API:
  call(workload, system, user, tools, ...) -> LLMCallResult

Provider routing (LOCKED 2026-05-19):
  - "narrator"          → anthropic claude-haiku-4-5
  - "rm_agent"          → anthropic claude-sonnet-4-6
  - "devils_advocate"   → deepseek v4_pro          (stub until DeepSeek key)
  - "massive_context"   → deepseek v4_pro          (stub until DeepSeek key)

Cost tracking: every successful call records to engine.llm_cost_ledger
(unified across providers). caller supplies agent_id; provider+model
auto-resolved by workload routing.
"""
from engine.llm.call import LLMCallResult, ToolCall, call

__all__ = [
    "LLMCallResult",
    "ToolCall",
    "call",
]
