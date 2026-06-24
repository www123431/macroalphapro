"""
engine/llm/providers/ — per-provider call adapters.

Each adapter exposes one function: call_provider(model, system, user,
tools, ...) -> _RawCallResult. The orchestrator in engine.llm.call
selects the adapter based on workload routing.
"""
