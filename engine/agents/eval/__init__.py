"""engine/agents/eval — agent behavioral eval harness (institutional, 0-LLM scoring).

The agentic-line analog of the quant-line point-in-time audit: prove the personas obey
their behavioral contract (no hallucinated numbers, refuse out-of-scope, route to the
right peer, never soften a deterministic verdict), with a DETERMINISTIC scorer — math
grades the model, NOT an LLM judge (we do not trust an LLM to grade itself; same
0-LLM-in-DECISION doctrine that governs the book).

Two tiers (share the same case set + scorer):
  - Tier A  static contract  : no LLM, runs in CI — read-only tool palette, prompt
                               encodes the refusal/routing/grounding policy, allowlisting.
  - Tier B  live behavioral  : calls chat_turn, scores each held-out case deterministically
                               (tool called? refused? routed? banned phrase? fabricated
                               number?). Gated behind RUN_AGENT_EVAL / --live (costs tokens).

contract.py  — Expectation types + deterministic scorer + static-contract checks
cases.py     — the held-out behavioral case set per persona
runner.py    — static + live runners + CLI
"""
