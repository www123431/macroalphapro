"""
engine/quant_co_pilot/base.py — ReAct agent core for Decision Lineage Assistant.

Pre-registration: docs/spec_quant_co_pilot_decision_lineage_v1.md (id=53) §2.1, §2.3, §2.4

ReAct (Yao et al. 2022 ICLR) loop:
  Plan → Act (tool call) → Observe → repeat (max N_STEPS_MAX) → Final answer

Locked budget caps (spec §2.3):
  - N_STEPS_MAX = 8 iterations
  - COST_BUDGET_USD = $0.05 per query
  - LATENCY_BUDGET_MS = 30s per query
  - TEMPERATURE = 0.1 for stability

Citation validator (spec §2.4): regex extraction + authoritative source verify.
Failed citations annotated [UNVERIFIED] inline.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import re
import time
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)

# Locked per spec §2.3 — DO NOT change without amend_spec
N_STEPS_MAX_LOCKED:        int   = 8
COST_BUDGET_USD_LOCKED:    float = 0.05
LATENCY_BUDGET_MS_LOCKED:  int   = 30000
TEMPERATURE_LOCKED:        float = 0.1
LLM_MODEL_LOCKED:          str   = "gemini-2.5-flash"

# Citation patterns per spec §2.4
CITATION_PATTERNS: dict[str, str] = {
    "spec_id":        r"\bspec[_ ]?id[=: ]?(\d+)\b",
    "git_hash":       r"\b([0-9a-f]{7,40})\b",
    "memory_file":    r"\b((?:project|feedback|user|reference)_[a-z0-9_\-]+\.md)\b",
    "amendment_kind": r"\b(clarification|threshold_tweak|hypothesis_amend|endpoint_swap|superseded)\b",
    "capability_evidence": r"\b((?:factor_ensemble[_a-z0-9]+|[a-z_]+)_(?:descriptive|preliminary|robust|positive|partial|fail|negative)[_a-z0-9\-]*\.md)\b",
}

# Decision label whitelist (for verdict label citations)
VERDICT_LABELS_WHITELIST = (
    "DESCRIPTIVE_POSITIVE", "DESCRIPTIVE_NEGATIVE",
    "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION",
    "DESCRIPTIVE_INSUFFICIENT_NEGATIVE_DIRECTION",
    "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT",
    "DESCRIPTIVE_NEUTRAL",
    "PRELIMINARY_PASS", "PRELIMINARY_PARTIAL", "PRELIMINARY_FAIL",
    "PASS", "PARTIAL", "FAIL", "WITHDRAW",
)


@dataclasses.dataclass(frozen=True)
class TraceStep:
    """Single ReAct iteration: thought + action + observation."""
    step_idx:     int
    thought:      str
    action:       Optional[str]            # tool name; None on final-answer step
    action_input: Optional[dict]
    observation:  Optional[Any]            # tool result; None on final-answer step
    final_answer: Optional[str]            # set ONLY on terminal step
    elapsed_ms:   int
    cost_usd:     float


@dataclasses.dataclass(frozen=True)
class Citation:
    """Single citation extracted from final answer."""
    pattern:    str           # one of CITATION_PATTERNS keys
    raw_match:  str           # what the regex matched
    verified:   bool          # did authoritative source verify?
    verify_msg: str           # detail (e.g., "spec_id=99 not found in registry")


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Citation validation summary."""
    citations:        list[Citation]
    n_verified:       int
    n_unverified:     int
    annotated_answer: str     # answer with [UNVERIFIED: ...] inline annotations


@dataclasses.dataclass(frozen=True)
class TraceResult:
    """Full agent trace for one query."""
    query:        str
    final_answer: str
    citations:    list[Citation]
    annotated_answer: str
    steps:        list[TraceStep]
    cost_usd:     float
    latency_ms:   int
    abort_reason: Optional[str]
    completed_at: str          # ISO UTC


# ─────────────────────────────────────────────────────────────────────────────
# Citation validator (spec §2.4)
# ─────────────────────────────────────────────────────────────────────────────


def _verify_spec_id(raw: str) -> tuple[bool, str]:
    """Check spec_id exists in SpecRegistry."""
    try:
        from engine.memory import SessionFactory, SpecRegistry
        spec_id = int(raw)
        with SessionFactory() as s:
            row = s.query(SpecRegistry).filter(SpecRegistry.id == spec_id).first()
            if row is None:
                return False, f"spec_id={spec_id} not found in registry"
            return True, f"spec_id={spec_id} → {row.spec_path}"
    except Exception as exc:
        return False, f"verify error: {exc!s}"


def _verify_git_hash(raw: str) -> tuple[bool, str]:
    """Check git hash exists via git cat-file."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", raw],
            capture_output=True, text=True, timeout=5,
            cwd=str(_repo_root()),
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, f"git object: {result.stdout.strip()}"
        return False, f"git hash {raw} not found"
    except Exception as exc:
        return False, f"git cat-file error: {exc!s}"


def _verify_memory_file(raw: str) -> tuple[bool, str]:
    """Check memory file exists in memory/ dir."""
    from pathlib import Path
    memory_dir = Path.home() / ".claude" / "projects" / "c--Users-${USER}-Desktop-intern" / "memory"
    target = memory_dir / raw
    if target.exists():
        return True, f"memory file: {target.name}"
    return False, f"memory file {raw} not found in {memory_dir}"


def _verify_amendment_kind(raw: str) -> tuple[bool, str]:
    """Amendment kinds are a fixed set."""
    valid = {"clarification", "threshold_tweak", "hypothesis_amend", "endpoint_swap", "superseded"}
    if raw in valid:
        return True, f"valid amendment kind: {raw}"
    return False, f"unknown amendment kind: {raw}"


def _verify_capability_evidence(raw: str) -> tuple[bool, str]:
    """Check capability_evidence file exists in docs/capability_evidence/."""
    from pathlib import Path
    target = _repo_root() / "docs" / "capability_evidence" / raw
    if target.exists():
        return True, f"capability_evidence: {target.name}"
    return False, f"capability_evidence file {raw} not found"


def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent


_VERIFIERS: dict[str, Callable[[str], tuple[bool, str]]] = {
    "spec_id":             _verify_spec_id,
    "git_hash":            _verify_git_hash,
    "memory_file":         _verify_memory_file,
    "amendment_kind":      _verify_amendment_kind,
    "capability_evidence": _verify_capability_evidence,
}


def validate_citations(answer: str) -> ValidationResult:
    """Extract citation candidates from answer + verify each against authoritative source.

    Per spec §2.4: regex extraction + per-pattern verifier; failed → annotate
    answer with `[UNVERIFIED: <claim>]` inline.
    """
    citations: list[Citation] = []
    annotated = answer
    seen_keys: set[tuple[str, str]] = set()  # dedup (pattern, raw_match)

    for pattern_name, regex in CITATION_PATTERNS.items():
        verifier = _VERIFIERS.get(pattern_name)
        if verifier is None:
            continue
        # IGNORECASE: agent may write "Spec id=50" (capitalized) which still must match
        for match in re.finditer(regex, answer, re.IGNORECASE):
            raw = match.group(1) if match.groups() else match.group(0)
            key = (pattern_name, raw)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            verified, msg = verifier(raw)
            citations.append(Citation(
                pattern=pattern_name, raw_match=raw,
                verified=verified, verify_msg=msg,
            ))
            if not verified:
                # Annotate inline — replace first occurrence of raw with [UNVERIFIED: raw]
                annotated = annotated.replace(raw, f"[UNVERIFIED: {raw}]", 1)

    n_verified = sum(1 for c in citations if c.verified)
    return ValidationResult(
        citations=citations,
        n_verified=n_verified,
        n_unverified=len(citations) - n_verified,
        annotated_answer=annotated,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ReAct agent loop (spec §2.1)
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_ROLE_INTRO = """You are a Decision Lineage Assistant for a quant research project. Your job is to answer the user's query by orchestrating tool calls over the project's spec registry, git history, memory, verdict files, and capability evidence.

CRITICAL DISCIPLINE — DO NOT VIOLATE:
1. **You MUST call at least one tool before giving final_answer.** Never speculate factual claims.
2. **Never speculate factual claims** (counts, numbers, names, decisions, statuses). Always call a tool to get the real number.
3. **Cite sources inline** in final_answer:
   - spec_id refs: "spec id=50" (exact format)
   - git hashes: 7+ hex chars
   - memory filenames: e.g. "project_X.md"
   - capability_evidence filenames: e.g. "factor_ensemble_v1_descriptive_positive_2026-05-09.md"
4. If a query needs multiple sources, plan multi-step. Do NOT shortcut.
5. If a tool returns an error, try a different tool — do NOT give up by inventing an answer."""


def _build_react_prompt(
    query:              str,
    scratchpad:         list[TraceStep],
    tool_descriptions:  str,
    role_intro:         str = _DEFAULT_ROLE_INTRO,
) -> str:
    """Compose ReAct prompt: role + query + tool inventory + scratchpad + ask for next step.

    `role_intro` defaults to Tool 1's Decision Lineage Assistant framing; other
    agents reusing `run_react_agent` (e.g. Watchdog spec id=63) inject their
    own role description. The rest of the scaffolding (tool list, JSON-only
    output discipline, scratchpad) is universal across ReAct consumers.
    """
    history = ""
    for step in scratchpad:
        history += f"\nThought {step.step_idx}: {step.thought}"
        if step.action:
            history += f"\nAction {step.step_idx}: {step.action}"
            history += f"\nAction Input {step.step_idx}: {json.dumps(step.action_input, ensure_ascii=False)}"
            obs_str = json.dumps(step.observation, ensure_ascii=False, default=str)
            if len(obs_str) > 1500:
                obs_str = obs_str[:1500] + "...[truncated]"
            history += f"\nObservation {step.step_idx}: {obs_str}"

    n_prior_actions = sum(1 for s in scratchpad if s.action and not s.final_answer)
    tool_call_note = (
        f"You have called {n_prior_actions} tools so far."
        if n_prior_actions
        else "You have called NO tools yet — plan a tool call this step."
    )
    return f"""{role_intro}

You can use ONLY these tools (any other tool name = error):
{tool_descriptions}

{tool_call_note}

User query: {query}

Past trace:{history if history else " (no steps yet)"}

Respond as JSON only:
- If you need to call a tool: {{"thought": "...", "action": "<tool_name>", "action_input": {{...}}}}
- ONLY if you have already called a tool AND have enough info: {{"thought": "...", "final_answer": "<answer with citations inline>"}}
"""


def _call_llm(
    prompt:          str,
    response_schema: Optional[dict] = None,
    *,
    scope:           str = "",
    extra:           Optional[dict] = None,
    agent_id:        str = "tool1_decision_lineage",
) -> tuple[str, float, int]:
    """Call Gemini 2.5 Flash with structured output.

    Returns: (response_text, cost_usd, latency_ms)
    Raises on hard failure (no fallback — agent must abort with error).

    2026-05-10 (Sprint 2C-7): every successful call is recorded to
    `engine.llm_cost_ledger` under agent_id="tool1_decision_lineage". Implements
    spec id=53 §4.1 cost-persistence requirement (v1 had referenced fictional
    `LLMCallLog` ORM that was never built — this supersedes that).

    2026-05-12 (Watchdog Phase 2): `agent_id` parameter added so other agents
    that reuse `run_react_agent` (e.g. Ops Watchdog spec id=63) can record under
    their own ledger bucket. Default preserves Tool 1's locked behavior — Tool 1
    spec §4.1 contract unchanged.

    scope/extra are passed through to the ledger entry so caller can tag
    react step index, etc.
    """
    from engine.key_pool import get_pool
    pool = get_pool()
    model = pool.get_model(
        model_name=LLM_MODEL_LOCKED,
        response_schema=response_schema,
        temperature=TEMPERATURE_LOCKED,
    )
    t0 = time.time()
    resp = model.generate_content(prompt)
    pool.report_success(has_content=True)
    latency_ms = int((time.time() - t0) * 1000)

    raw_text = getattr(resp, "text", None) or str(resp)
    usage = getattr(resp, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(
        (getattr(usage, "candidates_token_count", 0) or 0)
        + (getattr(usage, "thoughts_token_count", 0) or 0)
    )
    # Gemini 2.5 Flash pricing: $0.30/M input + $2.50/M output
    cost = (in_tok * 0.30 + out_tok * 2.50) / 1_000_000.0

    # Persist to unified ledger (spec §4.1 compliance). Fail soft so a ledger
    # write hiccup doesn't crash the agent loop.
    try:
        from engine.llm_cost_ledger import record_call
        record_call(
            agent_id          = agent_id,
            provider          = "gemini",
            model             = LLM_MODEL_LOCKED,
            prompt_tokens     = in_tok,
            completion_tokens = out_tok,
            cost_usd          = cost,
            latency_ms        = latency_ms,
            scope             = scope,
            extra             = extra or {},
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "LLM cost ledger write failed (agent_id=%s): %s", agent_id, exc,
        )

    return raw_text, cost, latency_ms


# No response_schema — Gemini structured output forced empty {} on optional object types.
# Use natural JSON output instructed via prompt; we parse ourselves with robust extraction.
_REACT_RESPONSE_SCHEMA: Optional[dict] = None


def run_react_agent(
    query:                str,
    tool_dispatcher:      Callable[[str, dict], Any],
    tool_descriptions:    str,
    *,
    max_steps:            int   = N_STEPS_MAX_LOCKED,
    cost_budget_usd:      float = COST_BUDGET_USD_LOCKED,
    latency_budget_ms:    int   = LATENCY_BUDGET_MS_LOCKED,
    valid_tool_names:     Optional[set[str]] = None,
    agent_id:             str   = "tool1_decision_lineage",
    role_intro:           Optional[str] = None,
) -> TraceResult:
    """Run a ReAct agent loop with locked budget caps.

    Args:
        query: user natural language query
        tool_dispatcher(name, args) -> result: callable that runs a tool
        tool_descriptions: human-readable tool inventory string for prompt
        max_steps / cost_budget_usd / latency_budget_ms: per spec §2.3
        valid_tool_names: set of allowed tool names; agent calling other → fail loud
        agent_id: ledger bucket for cost recording. Default preserves Tool 1
                  spec §4.1 behavior; other agents reusing this primitive
                  (e.g. Watchdog spec id=63) pass their own agent_id so
                  per-agent cost accounting works (mode-13 budget rule etc.).

    Returns:
        TraceResult with full step-by-step trace + citation validation.
    """
    if valid_tool_names is None:
        from engine.quant_co_pilot.tools import TOOL_NAMES
        valid_tool_names = set(TOOL_NAMES)

    t_start = time.time()
    scratchpad: list[TraceStep] = []
    total_cost = 0.0
    abort_reason: Optional[str] = None
    final_answer = ""

    for step_idx in range(max_steps):
        # Budget check
        elapsed_ms = int((time.time() - t_start) * 1000)
        if elapsed_ms >= latency_budget_ms:
            abort_reason = f"latency budget exceeded ({elapsed_ms}ms ≥ {latency_budget_ms}ms cap)"
            break
        if total_cost >= cost_budget_usd:
            abort_reason = f"cost budget exceeded (${total_cost:.4f} ≥ ${cost_budget_usd:.4f} cap)"
            break

        # Plan
        if role_intro is None:
            prompt = _build_react_prompt(query, scratchpad, tool_descriptions)
        else:
            prompt = _build_react_prompt(query, scratchpad, tool_descriptions,
                                         role_intro=role_intro)
        try:
            raw_response, step_cost, step_latency = _call_llm(
                prompt,
                response_schema=_REACT_RESPONSE_SCHEMA,
                scope="react_step",
                extra={"step_idx": step_idx},
                agent_id=agent_id,
            )
        except Exception as exc:
            abort_reason = f"LLM call failed at step {step_idx}: {exc!s}"
            break

        total_cost += step_cost

        try:
            # Robust JSON extraction: strip markdown code fences if present
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                # ```json\n...\n``` or ```\n...\n```
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines)
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError) as exc:
            abort_reason = f"step {step_idx} response JSON parse failed: {exc!s}; raw[:300]={raw_response[:300]!r}"
            break

        thought = parsed.get("thought", "")
        n_prior_actions = sum(1 for s in scratchpad if s.action is not None and s.final_answer is None)

        # Final-answer terminal step — but enforce "at least 1 tool call before final_answer"
        if "final_answer" in parsed and parsed["final_answer"]:
            if n_prior_actions == 0:
                # Agent tried to short-circuit without calling any tool. Reject + force tool call.
                # Inject a synthetic observation telling agent it must call a tool first.
                obs = {
                    "error": (
                        "DISCIPLINE VIOLATION: you tried to give final_answer without calling any tool. "
                        "Per spec §2.1, you MUST call at least one tool before answering. "
                        "Try again — call a relevant tool from the inventory."
                    )
                }
                scratchpad.append(TraceStep(
                    step_idx=step_idx, thought=thought,
                    action="<rejected_premature_final_answer>",
                    action_input={"attempted_final_answer": parsed["final_answer"][:200]},
                    observation=obs, final_answer=None,
                    elapsed_ms=int((time.time() - t_start) * 1000),
                    cost_usd=step_cost,
                ))
                continue

            final_answer = parsed["final_answer"]
            scratchpad.append(TraceStep(
                step_idx=step_idx, thought=thought,
                action=None, action_input=None, observation=None,
                final_answer=final_answer,
                elapsed_ms=int((time.time() - t_start) * 1000),
                cost_usd=step_cost,
            ))
            break

        # Tool action step
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {}) or {}

        # Validate tool name
        if action not in valid_tool_names:
            obs = {"error": f"unknown tool '{action}'; valid tools: {sorted(valid_tool_names)}"}
            scratchpad.append(TraceStep(
                step_idx=step_idx, thought=thought,
                action=action, action_input=action_input, observation=obs,
                final_answer=None,
                elapsed_ms=int((time.time() - t_start) * 1000),
                cost_usd=step_cost,
            ))
            abort_reason = f"agent called unknown tool '{action}' at step {step_idx} (fail-loud per spec §2.2)"
            break

        # Dispatch
        try:
            obs = tool_dispatcher(action, action_input)
        except Exception as exc:
            obs = {"error": f"tool '{action}' raised: {exc!s}"}

        scratchpad.append(TraceStep(
            step_idx=step_idx, thought=thought,
            action=action, action_input=action_input, observation=obs,
            final_answer=None,
            elapsed_ms=int((time.time() - t_start) * 1000),
            cost_usd=step_cost,
        ))

    # If loop exhausted without final_answer
    if not final_answer and abort_reason is None:
        abort_reason = f"max_steps={max_steps} reached without final answer"

    # Citation validation
    validation = validate_citations(final_answer)

    return TraceResult(
        query=query,
        final_answer=final_answer,
        citations=validation.citations,
        annotated_answer=validation.annotated_answer,
        steps=scratchpad,
        cost_usd=total_cost,
        latency_ms=int((time.time() - t_start) * 1000),
        abort_reason=abort_reason,
        completed_at=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
