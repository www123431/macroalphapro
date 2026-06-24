"""
engine/agents/persona/base.py — generic persona agent loop.

Defines AgentPersona (single source of truth for one agent's identity)
and a workload-agnostic chat_turn() that drives the tool-using
conversation loop. Per-agent files (risk_manager.py / dq_inspector.py
/ devils_advocate.py / ...) define AgentPersona instances that this
module consumes.

Pattern 5 BAN enforcement: chat_turn takes the caller's `history`
verbatim. Each Streamlit page maintains a per-agent session_state
key (chat_history_<agent_id>), so two agents never share history
state — even if a future group-chat UI is built, the routing layer
must filter each agent's view of history before calling chat_turn.
Agent-to-agent autonomous interaction is structurally impossible
through this API.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# AgentPersona — one agent's identity
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class AgentPersona:
    """A single agent's identity: persona, capability, routing, defaults.

    Fields:
      name:              display name (e.g. "Risk Manager")
      role_id:           persona role id (e.g. "head_of_risk_blackrock_slack")
      agent_id:          one of engine.llm_cost_ledger.ALLOWED_AGENT_IDS
                         (cost ledger key — typo-fails fast)
      workload:          engine.llm.call workload string → provider+model
      system_prompt:     full persona system prompt; should include tone
                         constraints (banned phrases, NO EMOJIS, READ-ONLY
                         scope), tool descriptions, cross-agent reference
                         policy, examples
      tools:             list of Anthropic-format tool schemas
      tool_executor:     name → JSON-string callable; default execute_tool
      spec_ref:          human-readable spec reference for the agent's
                         doctrinal source (e.g. "spec id=69 — hash in
                         SpecRegistry, looked up at runtime")
      max_iterations:    hard cap on tool round-trips per user turn
      default_effort:    low | medium | high (Sonnet 4.6 / Opus only)
      default_max_tokens: per-iteration completion cap
    """
    name:              str
    role_id:           str
    agent_id:          str
    workload:          str
    system_prompt:     str
    tools:             list[dict]
    # tool_executor returns (content_str, is_error) tuple. is_error
    # propagates into the Anthropic tool_result block's `is_error: true`
    # flag so the model knows to recover rather than treat the error
    # string as legitimate data (per Anthropic tool-use protocol).
    tool_executor:     Callable[[str, dict], tuple[str, bool]]
    spec_ref:          str
    max_iterations:    int = 6
    default_effort:    str = "medium"
    default_max_tokens: int = 2048


# ──────────────────────────────────────────────────────────────────────────────
# Result shape — caller-visible single-turn output
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class AgentTurnResult:
    """One full agent turn (may include multiple tool round-trips internally).

    Fields:
      final_text:       the agent's terminal text response (user-visible)
      tool_calls_log:   ordered tuple of {name, input, result_preview, iteration}
                        — for the Streamlit "tool calls this turn" expander
      n_iterations:     total LLM calls in this turn (1 if no tools used)
      total_cost_usd:   sum of cost across all iterations in this turn
      total_latency_ms: cumulative wall-clock
      new_messages:     delta to append to caller's history (caller persists)
      stop_reason:      last iteration's stop_reason from the model
    """
    final_text:       str
    tool_calls_log:   tuple[dict, ...]
    n_iterations:     int
    total_cost_usd:   float
    total_latency_ms: int
    new_messages:     list[dict]
    stop_reason:      str


# ──────────────────────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────────────────────
def chat_turn(
    persona:      AgentPersona,
    user_message: str,
    history:      Optional[list[dict]] = None,
    max_tokens:   Optional[int] = None,
    effort:       Optional[str] = None,
) -> AgentTurnResult:
    """One agent turn — handles tool-use round-trips internally.

    Args:
      persona:      AgentPersona — the agent identity (system prompt + tools
                    + routing). Defines who is answering.
      user_message: the new user message
      history:      prior Anthropic-format messages. None → fresh chat.
      max_tokens:   override persona.default_max_tokens
      effort:       override persona.default_effort

    Returns:
      AgentTurnResult — final text + tool log + cost + delta messages
      for caller to append to session state.

    Pattern 5 ban: `history` is verbatim what the caller passes. The
    caller is responsible for filtering cross-agent contamination
    (in per-page Streamlit, each agent has its own session_state key
    so this is automatic).
    """
    from engine.llm.call import call

    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": user_message})

    tool_calls_log: list[dict] = []
    total_cost = 0.0
    total_latency = 0
    last_stop_reason = ""

    final_max_tokens = max_tokens if max_tokens is not None else persona.default_max_tokens
    final_effort     = effort     if effort     is not None else persona.default_effort

    iteration_count = 0
    for iteration in range(persona.max_iterations):
        iteration_count = iteration + 1
        result = call(
            workload   = persona.workload,
            system     = persona.system_prompt,
            messages   = messages,
            tools      = persona.tools,
            agent_id   = persona.agent_id,
            max_tokens = final_max_tokens,
            effort     = final_effort,
            scope      = f"chat_turn_iter_{iteration}",
        )
        total_cost += result.cost_usd
        total_latency += result.latency_ms
        last_stop_reason = result.stop_reason

        # Build the assistant message we just produced. Must include BOTH
        # text blocks (if any) AND tool_use blocks — the Anthropic API
        # rejects messages with a dangling tool_use that isn't paired
        # with a subsequent tool_result.
        assistant_content: list[dict] = []
        if result.text:
            assistant_content.append({"type": "text", "text": result.text})
        for tc in result.tool_calls:
            assistant_content.append({
                "type":  "tool_use",
                "id":    tc.id,
                "name":  tc.name,
                "input": tc.input,
            })
        messages.append({"role": "assistant", "content": assistant_content})

        # No tool calls → terminal turn
        if not result.tool_calls:
            break

        # Execute each tool call, append results as a single user message.
        # Routed through the authority guard (least-privilege enforced at the executor,
        # blueprint spec id=78 Phase 3): an out-of-palette tool_use is blocked + audited,
        # never executed. Falls back to the bare executor if the guard is unavailable.
        tool_results: list[dict] = []
        try:
            from engine.agents.governance.authority import enforce_tool_call
        except ImportError:
            enforce_tool_call = None
        for tc in result.tool_calls:
            if enforce_tool_call is not None:
                tool_output, is_error = enforce_tool_call(persona, tc.name, tc.input)
            else:
                tool_output, is_error = persona.tool_executor(tc.name, tc.input)
            tool_calls_log.append({
                "name":           tc.name,
                "input":          tc.input,
                "result_preview": tool_output[:200],
                "is_error":       is_error,
                "iteration":      iteration,
            })
            result_block: dict = {
                "type":         "tool_result",
                "tool_use_id":  tc.id,
                "content":      tool_output,
            }
            if is_error:
                # Anthropic tool-use protocol: flag failed tool calls so
                # the model recovers / retries with different args instead
                # of consuming the error string as data.
                result_block["is_error"] = True
            tool_results.append(result_block)
        messages.append({"role": "user", "content": tool_results})
        # Loop continues — model sees tool results and either calls more
        # tools or returns final text.
    else:
        # Hit iteration cap without terminal text
        logger.warning(
            "chat_turn(%s): hit max iterations (%d) without terminal text; "
            "returning partial result",
            persona.agent_id, persona.max_iterations,
        )

    # Final text = the last assistant message's text content (if any)
    final_text = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        final_text = blk.get("text", "")
                        break
            elif isinstance(content, str):
                final_text = content
            break

    # new_messages = everything after the original history
    n_prior = len(history or [])
    new_messages = messages[n_prior:]

    return AgentTurnResult(
        final_text       = final_text,
        tool_calls_log   = tuple(tool_calls_log),
        n_iterations     = iteration_count,
        total_cost_usd   = total_cost,
        total_latency_ms = total_latency,
        new_messages     = new_messages,
        stop_reason      = last_stop_reason,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Streaming variant — yields events as the loop runs (for the SSE chat terminal)
# ──────────────────────────────────────────────────────────────────────────────
def chat_turn_events(
    persona:      AgentPersona,
    user_message: str,
    history:      Optional[list[dict]] = None,
    max_tokens:   Optional[int] = None,
    effort:       Optional[str] = None,
):
    """Generator twin of chat_turn: yields structured events as the tool loop runs, so a
    transport (SSE) can stream the agent's activity live (you watch CoS → delegate → specialist
    → result → synthesis). The loop mirrors chat_turn exactly; chat_turn itself is UNCHANGED
    (the eval harness + CoS delegation depend on its AgentTurnResult contract).

    Yields dicts, each with a "type":
      {"type":"start", agent_id, name}
      {"type":"iteration", "n":int}
      {"type":"assistant_text", "text":str}              # narration this iteration (may repeat)
      {"type":"tool_call", "name", "input", "iteration"}
      {"type":"tool_result", "name", "preview", "is_error"}
      {"type":"done", "final_text", "cost_usd", "latency_ms", "n_iterations",
                      "stop_reason", "new_messages"}      # caller appends new_messages to history

    The same Pattern-5 isolation holds: `history` is verbatim what the caller passes.
    """
    from engine.llm.call import call

    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": user_message})

    tool_calls_log: list[dict] = []
    total_cost = 0.0
    total_latency = 0
    last_stop_reason = ""
    final_max_tokens = max_tokens if max_tokens is not None else persona.default_max_tokens
    final_effort     = effort     if effort     is not None else persona.default_effort

    try:
        from engine.agents.governance.authority import enforce_tool_call
    except ImportError:
        enforce_tool_call = None

    yield {"type": "start", "agent_id": persona.agent_id, "name": persona.name}

    iteration_count = 0
    for iteration in range(persona.max_iterations):
        iteration_count = iteration + 1
        yield {"type": "iteration", "n": iteration_count}

        result = call(
            workload   = persona.workload,
            system     = persona.system_prompt,
            messages   = messages,
            tools      = persona.tools,
            agent_id   = persona.agent_id,
            max_tokens = final_max_tokens,
            effort     = final_effort,
            scope      = f"chat_events_iter_{iteration}",
        )
        total_cost += result.cost_usd
        total_latency += result.latency_ms
        last_stop_reason = result.stop_reason

        assistant_content: list[dict] = []
        if result.text:
            assistant_content.append({"type": "text", "text": result.text})
            yield {"type": "assistant_text", "text": result.text}
        for tc in result.tool_calls:
            assistant_content.append({
                "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input,
            })
        messages.append({"role": "assistant", "content": assistant_content})

        if not result.tool_calls:
            break

        tool_results: list[dict] = []
        for tc in result.tool_calls:
            yield {"type": "tool_call", "name": tc.name, "input": tc.input, "iteration": iteration}
            if enforce_tool_call is not None:
                tool_output, is_error = enforce_tool_call(persona, tc.name, tc.input)
            else:
                tool_output, is_error = persona.tool_executor(tc.name, tc.input)
            tool_calls_log.append({
                "name": tc.name, "input": tc.input,
                "result_preview": tool_output[:200], "is_error": is_error, "iteration": iteration,
            })
            yield {"type": "tool_result", "name": tc.name, "preview": tool_output[:280], "is_error": is_error}
            block: dict = {"type": "tool_result", "tool_use_id": tc.id, "content": tool_output}
            if is_error:
                block["is_error"] = True
            tool_results.append(block)
        messages.append({"role": "user", "content": tool_results})
    else:
        logger.warning(
            "chat_turn_events(%s): hit max iterations (%d) without terminal text",
            persona.agent_id, persona.max_iterations,
        )

    final_text = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        final_text = blk.get("text", "")
                        break
            elif isinstance(content, str):
                final_text = content
            break

    n_prior = len(history or [])
    new_messages = messages[n_prior:]
    yield {
        "type":         "done",
        "final_text":   final_text,
        "cost_usd":     round(total_cost, 6),
        "latency_ms":   int(total_latency),
        "n_iterations": iteration_count,
        "stop_reason":  last_stop_reason,
        "new_messages": new_messages,
    }
