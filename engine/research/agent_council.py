"""engine/research/agent_council.py — Phase 4b: 3-agent critique
council (architect + behavioral_theorist + empirical_devils_advocate).

Per orchestration RFC §3.3:
  - Anthropic Claude Sonnet 4.6 direct (NOT wrapped in LangChain)
  - Pattern 1 fan-out (parallel; no autonomous debate)
  - Tools from engine.research.llm_tools.TOOLS (10 of them)
  - 1 source of truth: same tools the MCP server + REST shim expose

Two-stage flow:
  Stage A: architect_propose(seed_idea) → ProposalDict
  Stage B: critique_council(proposal) → CouncilVerdict
           (theorist + empirical_DA fan-out in parallel)

The single L4 entrypoint run_full_council(seed) chains A→B and
returns (proposal, verdict). Used by the Temporal outer-ring
workflow (Phase 4c+).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase 4b.5: persistent ledger of council runs — every architect_propose
# + critique_council + run_full_council writes one row here, so the
# Cockpit panel can render activity + drill into verdicts without
# re-running expensive LLM calls. JSONL = append-only, replayable.
COUNCIL_RUNS_LEDGER = REPO_ROOT / "data" / "research" / "council_runs.jsonl"

# Anthropic settings — Claude Sonnet 4.6 per RFC, deterministic
# enough for verdict but warm enough to surface non-obvious concerns.
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_MAX_TOKENS = 2000
ANTHROPIC_TEMPERATURE = 0.2
MAX_TOOL_CALLS_PER_AGENT = 6   # safety cap; council burns LLM tokens


# ── API key loading ───────────────────────────────────────────────────


def _load_anthropic_key() -> Optional[str]:
    """Find the Anthropic key from env or .streamlit/secrets.toml.

    Returns None if missing — caller should skip / fall back, not raise,
    so dev environments without a key still let smoke tests pass."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.is_file():
        try:
            import tomllib  # py>=3.11
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                logger.warning("no tomllib / tomli; skipping secrets.toml")
                return None
        try:
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            key = data.get("ANTHROPIC_API_KEY")
            if isinstance(key, str) and key.strip():
                return key.strip()
        except Exception:
            logger.exception("secrets.toml parse failed (non-fatal)")
    return None


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProposalDict:
    """Architect-produced candidate proposal. Mirrors the dict shape
    candidate_pipeline.run_candidate_pipeline expects, so the council
    output can flow straight into the inner ring."""
    title: str
    family: str
    parent_family: str
    proposed_role: str          # alpha_seeker / risk_premium_harvester / insurance / diversifier / regime_overlay
    economics_text: str         # 1-2 paragraphs of mechanism rationale
    required_data: list[str]
    motivation: str = ""        # 1-paragraph "why this might work"
    mechanism_id: Optional[str] = None  # if matching an existing library entry

    def to_dict(self) -> dict:
        return asdict(self)


VerdictLiteral = Literal["PASS", "WARN", "FAIL"]


@dataclass
class ToolCallLog:
    """One tool call inside an agent's loop. Recorded for audit +
    council trace UI (4e+)."""
    tool_name: str
    args: dict
    result_summary: str
    elapsed_ms: float


@dataclass
class AgentVerdict:
    """One agent's review of the proposal."""
    agent_name: str             # "architect" / "behavioral_theorist" / "empirical_devils_advocate"
    verdict: str                # PASS / WARN / FAIL
    confidence: float           # 0..1
    rationale: str              # narrative
    fatal_red_flags: list[str] = field(default_factory=list)
    material_concerns: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    elapsed_s: float = 0.0
    raw_response: str = ""      # truncated, for debug

    # Frontier 1 (2026-06-01): structured reflection-round fields.
    # round_1_*  captures the pre-reflection state so calibration tooling
    # (Frontier 3) can measure whether reflection improves outcomes.
    # reflection_action is None for round-1-only runs.
    round_1_verdict:    Optional[str]   = None
    round_1_confidence: Optional[float] = None
    reflection_action:  Optional[str]   = None  # confirmed/revised_up/revised_down/revised_lateral

    def to_dict(self) -> dict:
        d = asdict(self)
        # ToolCallLog is already serializable via asdict
        return d


@dataclass
class CouncilVerdict:
    """Aggregated council decision over one proposal."""
    proposal: ProposalDict
    verdicts: list[AgentVerdict]
    consensus: str              # APPROVE / NEEDS_REVISION / REJECT
    rationale: str              # meta-aggregator's synthesis
    elapsed_s: float = 0.0
    run_id: str = ""            # set by ledger writer

    # Frontier 1: round-1 consensus snapshot. If reflection was disabled
    # this equals (consensus, rationale). If reflection was enabled and
    # changed the consensus, this is the pre-reflection state — kept so
    # we can measure reflection ΔSharpe over many runs.
    round_1_consensus: Optional[str] = None
    round_1_rationale: Optional[str] = None
    reflection_enabled: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id":             self.run_id,
            "proposal":           self.proposal.to_dict(),
            "verdicts":           [v.to_dict() for v in self.verdicts],
            "consensus":          self.consensus,
            "rationale":          self.rationale,
            "elapsed_s":          self.elapsed_s,
            "round_1_consensus":  self.round_1_consensus,
            "round_1_rationale":  self.round_1_rationale,
            "reflection_enabled": self.reflection_enabled,
        }


# ── Ledger (Phase 4b.5) ───────────────────────────────────────────────


def _append_to_ledger(entry: dict) -> str:
    """Append one row to council_runs.jsonl. Returns the run_id.

    Best-effort: ledger failure must not crash the council itself.
    Each entry has its own UUID so the Cockpit panel can render +
    drill-down without colliding when concurrent runs land."""
    import datetime as _dt
    import uuid as _uuid
    run_id = _uuid.uuid4().hex[:12]
    try:
        COUNCIL_RUNS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id":     run_id,
            "ts":         _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **entry,
        }
        with COUNCIL_RUNS_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.exception("council ledger append failed (non-fatal)")
    return run_id


def read_council_runs(
    limit: int = 50,
    consensus: Optional[str] = None,
) -> list[dict]:
    """Read recent council runs. Newest first. Filterable by consensus.

    Used by:
      - GET /api/research/council/runs  (UI Cockpit panel)
      - L4 outer-ring workflow when learning from past iterations
      - Calibration tooling (council verdict vs eventual outcome
        correlation)"""
    if not COUNCIL_RUNS_LEDGER.is_file():
        return []
    out: list[dict] = []
    with COUNCIL_RUNS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if consensus and row.get("consensus") != consensus:
                continue
            out.append(row)
    out.reverse()  # newest first
    return out[:max(1, limit)]


def read_council_run_by_id(run_id: str) -> Optional[dict]:
    """Drill-down: load one full run by id (for the UI detail view)."""
    if not COUNCIL_RUNS_LEDGER.is_file():
        return None
    with COUNCIL_RUNS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("run_id") == run_id:
                return row
    return None


# ── Tool-use loop ─────────────────────────────────────────────────────


def run_agent_with_tools(
    *,
    agent_name: str,
    system_prompt: str,
    user_message: str,
    allowed_tools: list[str],
    max_tool_calls: int = MAX_TOOL_CALLS_PER_AGENT,
    api_key: Optional[str] = None,
    model: str = ANTHROPIC_MODEL,
) -> tuple[str, list[ToolCallLog]]:
    """Run one agent's tool-use loop against Anthropic.

    Returns (final_text, tool_call_logs). The final_text is whatever
    Claude says after it stops calling tools — the caller parses it
    (typically expects JSON for verdict shape).

    Raises RuntimeError if no API key is available. Tool calls are
    capped at max_tool_calls to prevent runaway loops.
    """
    import anthropic

    from engine.research.llm_tools import (
        TOOLS, dispatch, tool_specs_for_anthropic,
    )

    key = api_key or _load_anthropic_key()
    if not key:
        raise RuntimeError(
            "no ANTHROPIC_API_KEY found in env or .streamlit/secrets.toml"
        )

    # Filter the full tool registry down to what this agent is allowed
    all_specs = tool_specs_for_anthropic()
    tool_specs = [s for s in all_specs if s["name"] in allowed_tools]
    if not tool_specs:
        logger.warning("agent %s has no allowed_tools — running without tool use",
                       agent_name)

    client = anthropic.Anthropic(api_key=key)
    messages: list[dict] = [{"role": "user", "content": user_message}]
    tool_call_logs: list[ToolCallLog] = []

    # Phase 4f tracing: agent-level span groups all child anthropic_call
    # + tool.{name} spans for this agent's tool-use loop.
    from engine.research.trace_log import (
        add_attr as _trace_add_attr, span as _trace_span,
    )

    with _trace_span(
        f"agent.{agent_name}",
        kind_class="agent",
        agent=agent_name,
        n_allowed_tools=len(tool_specs),
        model=model,
    ):
        text_blocks: list = []
        for iter_idx in range(max_tool_calls + 1):
            try:
                with _trace_span(
                    f"anthropic.call.{agent_name}",
                    kind_class="llm_call",
                    agent=agent_name,
                    iter_idx=iter_idx,
                    n_messages=len(messages),
                ):
                    resp = client.messages.create(
                        model=model,
                        max_tokens=ANTHROPIC_MAX_TOKENS,
                        temperature=ANTHROPIC_TEMPERATURE,
                        system=system_prompt,
                        tools=tool_specs if tool_specs else None,
                        messages=messages,
                    )
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        _trace_add_attr(
                            in_tokens=getattr(usage, "input_tokens", None),
                            out_tokens=getattr(usage, "output_tokens", None),
                        )
            except Exception as exc:
                logger.exception("Anthropic call failed in agent %s iter %d",
                                  agent_name, iter_idx)
                raise

            tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
            text_blocks = [b for b in resp.content if b.type == "text"]

            if resp.stop_reason in ("end_turn", "stop_sequence") or not tool_use_blocks:
                final_text = "\n".join(b.text for b in text_blocks).strip()
                _trace_add_attr(
                    final_iter=iter_idx,
                    n_tool_calls=len(tool_call_logs),
                )
                return final_text, tool_call_logs

            assistant_content = []
            for b in resp.content:
                if b.type == "text":
                    assistant_content.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    assistant_content.append({
                        "type":  "tool_use",
                        "id":    b.id,
                        "name":  b.name,
                        "input": b.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for b in tool_use_blocks:
                tool_name = b.name
                args = b.input or {}
                t0 = time.perf_counter()
                try:
                    if tool_name not in TOOLS:
                        result = {"error": f"unknown tool: {tool_name}"}
                    else:
                        # dispatch() emits its own tool.{name} span
                        result = dispatch(tool_name, **args)
                    result_str = json.dumps(result, default=str)[:4000]
                    ok = True
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)[:300]})
                    ok = False
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                tool_call_logs.append(ToolCallLog(
                    tool_name=tool_name,
                    args=args,
                    result_summary=result_str[:400],
                    elapsed_ms=elapsed_ms,
                ))
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": b.id,
                    "content":     result_str,
                    "is_error":    not ok,
                })

            messages.append({"role": "user", "content": tool_results})

        # Cap hit
        final_text = "\n".join(b.text for b in text_blocks).strip()
        final_text += f"\n\n[NOTE: agent hit max_tool_calls={max_tool_calls}]"
        _trace_add_attr(hit_cap=True, n_tool_calls=len(tool_call_logs))
        return final_text, tool_call_logs


# ── Verdict parsing ───────────────────────────────────────────────────


def _parse_verdict_json(raw: str) -> dict:
    """Lenient JSON extraction: strip markdown fences, find first {...}.

    Personas are instructed to emit strict JSON, but LLMs sometimes
    wrap in ```json ... ``` or precede with prose. We tolerate both."""
    if not raw:
        return {}
    s = raw.strip()
    # Strip markdown code fence
    if s.startswith("```"):
        # remove the fence + optional language tag
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Find outermost JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    blob = s[start: end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return {}


def _normalize_verdict(parsed: dict, default: str = "WARN") -> VerdictLiteral:
    v = (parsed.get("verdict") or default).upper().strip()
    if v not in ("PASS", "WARN", "FAIL"):
        # Common LLM variations
        if v in ("APPROVE", "OK", "ACCEPT"):
            return "PASS"
        if v in ("REJECT", "BLOCK", "DENY"):
            return "FAIL"
        return "WARN"
    return v  # type: ignore[return-value]


# Persona definitions live in the companion module — split so prompts
# are easier to read + edit without scrolling through orchestration.


# ── Stage A: architect proposes ───────────────────────────────────────


def architect_propose(
    seed_idea: str,
    *,
    api_key: Optional[str] = None,
) -> ProposalDict:
    """Run the strategy_architect agent on a seed idea → ProposalDict.

    The architect MUST consult intuition_rules + graveyard before
    finalizing (its system prompt enforces this), so the returned
    proposal already passes a self-checked sanity bar.

    Raises RuntimeError if no API key OR if architect output cannot
    be parsed as a ProposalDict (LLM hallucinated structure)."""
    from engine.research.agent_council_personas import (
        ARCHITECT_SYSTEM_PROMPT, ARCHITECT_TOOLS,
    )
    user_msg = (
        "Propose a fully-specified candidate strategy from the "
        "following seed idea. Use your tools to verify the proposal "
        "does not duplicate a deployed sleeve or a dead family.\n\n"
        f"SEED IDEA:\n{seed_idea.strip()}\n"
    )
    t0 = time.perf_counter()
    final_text, _logs = run_agent_with_tools(
        agent_name="strategy_architect",
        system_prompt=ARCHITECT_SYSTEM_PROMPT,
        user_message=user_msg,
        allowed_tools=ARCHITECT_TOOLS,
        api_key=api_key,
    )
    parsed = _parse_verdict_json(final_text)
    if not parsed or "family" not in parsed:
        raise RuntimeError(
            f"architect output did not parse as ProposalDict. "
            f"Raw (first 400 chars):\n{final_text[:400]}"
        )
    # Coerce + default-fill
    return ProposalDict(
        title=str(parsed.get("title") or "unnamed_candidate"),
        family=str(parsed["family"]),
        parent_family=str(parsed.get("parent_family") or "unknown"),
        proposed_role=str(parsed.get("proposed_role") or "alpha_seeker"),
        economics_text=str(parsed.get("economics_text") or "")[:4000],
        required_data=list(parsed.get("required_data") or []),
        motivation=str(parsed.get("motivation") or "")[:1500],
        mechanism_id=parsed.get("mechanism_id"),
    )


# ── Stage B: critique fan-out ─────────────────────────────────────────


def _run_one_critic(
    *,
    agent_name: str,
    system_prompt: str,
    allowed_tools: list[str],
    proposal: ProposalDict,
    api_key: Optional[str] = None,
) -> AgentVerdict:
    """Synchronous critic execution — used by run_council via asyncio.to_thread."""
    from engine.research.agent_council_personas import format_proposal_for_review
    t0 = time.perf_counter()
    user_msg = format_proposal_for_review(proposal.to_dict())
    try:
        final_text, logs = run_agent_with_tools(
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_message=user_msg,
            allowed_tools=allowed_tools,
            api_key=api_key,
        )
    except Exception as exc:
        return AgentVerdict(
            agent_name=agent_name,
            verdict="WARN",
            confidence=0.0,
            rationale=f"critic crashed: {exc}",
            elapsed_s=time.perf_counter() - t0,
        )
    parsed = _parse_verdict_json(final_text)
    return AgentVerdict(
        agent_name=agent_name,
        verdict=_normalize_verdict(parsed),
        confidence=float(parsed.get("confidence") or 0.5),
        rationale=str(parsed.get("rationale") or "")[:2000],
        fatal_red_flags=list(parsed.get("fatal_red_flags") or []),
        material_concerns=list(parsed.get("material_concerns") or []),
        tool_calls=logs,
        elapsed_s=time.perf_counter() - t0,
        raw_response=final_text[:1500],
    )


# ── Frontier 1: reflection round ──────────────────────────────────────


# Reflection action taxonomy — used by aggregator + calibration tooling.
_REFLECTION_ACTIONS = {
    "confirmed", "revised_up", "revised_down", "revised_lateral",
}


def _run_one_reflector(
    *,
    agent_name: str,
    system_prompt: str,
    allowed_tools: list[str],
    proposal: ProposalDict,
    own_round_1: AgentVerdict,
    peer_round_1: AgentVerdict,
    api_key: Optional[str] = None,
) -> AgentVerdict:
    """Run ONE critic's reflection turn (round 2).

    The critic sees both verdicts from round 1 and produces a
    confirmed/revised verdict. Same persona + tool budget — only the
    user-message body changes. The returned AgentVerdict carries the
    round_1 snapshot so downstream can measure deltas.

    Reflection is STRUCTURED (one-shot, parallel across critics) — NOT
    autonomous debate. Pattern 6 doctrine.
    """
    from engine.research.agent_council_personas import (
        REFLECTION_USER_MESSAGE_TEMPLATE,
    )

    t0 = time.perf_counter()
    own_json = json.dumps({
        "verdict":           own_round_1.verdict,
        "confidence":        own_round_1.confidence,
        "fatal_red_flags":   own_round_1.fatal_red_flags,
        "material_concerns": own_round_1.material_concerns,
        "rationale":         own_round_1.rationale,
    }, indent=2, default=str)
    peer_json = json.dumps({
        "verdict":           peer_round_1.verdict,
        "confidence":        peer_round_1.confidence,
        "fatal_red_flags":   peer_round_1.fatal_red_flags,
        "material_concerns": peer_round_1.material_concerns,
        "rationale":         peer_round_1.rationale,
    }, indent=2, default=str)
    user_msg = REFLECTION_USER_MESSAGE_TEMPLATE.format(
        own_verdict_json=own_json,
        peer_name=peer_round_1.agent_name,
        peer_verdict_json=peer_json,
    )

    try:
        final_text, logs = run_agent_with_tools(
            agent_name=f"{agent_name}.reflection",
            system_prompt=system_prompt,
            user_message=user_msg,
            allowed_tools=allowed_tools,
            # Reflection should be cheap — half the round-1 cap.
            max_tool_calls=max(2, MAX_TOOL_CALLS_PER_AGENT // 2),
            api_key=api_key,
        )
    except Exception as exc:
        # Reflection failure must NOT lose the round-1 verdict — pass it
        # through with reflection_action="confirmed" + a note.
        own_round_1.reflection_action = "confirmed"
        own_round_1.rationale += f"\n\n[reflection skipped: {exc}]"
        return own_round_1

    parsed = _parse_verdict_json(final_text)
    new_verdict = _normalize_verdict(parsed, default=own_round_1.verdict)
    new_conf = float(parsed.get("confidence") or own_round_1.confidence)
    action = str(parsed.get("reflection_action") or "").lower().strip()
    if action not in _REFLECTION_ACTIONS:
        # Infer action from delta if model didn't emit one cleanly.
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        d = order.get(new_verdict, 1) - order.get(own_round_1.verdict, 1)
        if d > 0:
            action = "revised_up"
        elif d < 0:
            action = "revised_down"
        elif abs(new_conf - own_round_1.confidence) > 0.1:
            action = "revised_lateral"
        else:
            action = "confirmed"

    return AgentVerdict(
        agent_name=agent_name,
        verdict=new_verdict,
        confidence=new_conf,
        rationale=str(parsed.get("rationale") or own_round_1.rationale)[:2000],
        fatal_red_flags=list(parsed.get("fatal_red_flags")
                              or own_round_1.fatal_red_flags),
        material_concerns=list(parsed.get("material_concerns")
                                or own_round_1.material_concerns),
        # Reflection-round tool calls are appended so the trace shows
        # everything the critic touched across both rounds.
        tool_calls=own_round_1.tool_calls + logs,
        elapsed_s=own_round_1.elapsed_s + (time.perf_counter() - t0),
        raw_response=final_text[:1500],
        round_1_verdict=own_round_1.verdict,
        round_1_confidence=own_round_1.confidence,
        reflection_action=action,
    )


async def critique_council(
    proposal: ProposalDict,
    *,
    api_key: Optional[str] = None,
    enable_reflection: bool = False,
) -> CouncilVerdict:
    """Fan-out: theorist + DA run in parallel against the same proposal.

    The architect is NOT a critic in stage B — its job was stage A.
    Returns a CouncilVerdict with aggregated consensus + rationale.
    """
    from engine.research.agent_council_personas import (
        DA_SYSTEM_PROMPT, DA_TOOLS,
        THEORIST_SYSTEM_PROMPT, THEORIST_TOOLS,
    )
    t0 = time.perf_counter()

    theorist_task = asyncio.to_thread(
        _run_one_critic,
        agent_name="behavioral_theorist",
        system_prompt=THEORIST_SYSTEM_PROMPT,
        allowed_tools=THEORIST_TOOLS,
        proposal=proposal,
        api_key=api_key,
    )
    da_task = asyncio.to_thread(
        _run_one_critic,
        agent_name="empirical_devils_advocate",
        system_prompt=DA_SYSTEM_PROMPT,
        allowed_tools=DA_TOOLS,
        proposal=proposal,
        api_key=api_key,
    )
    round_1_verdicts: list[AgentVerdict] = list(
        await asyncio.gather(theorist_task, da_task)
    )
    round_1_consensus, round_1_rationale = aggregate_verdicts(round_1_verdicts)

    # Frontier 1 (2026-06-01): structured reflection round.
    # Each critic gets ONE chance to read the peer verdict and either
    # confirm or revise. Parallel, single-shot, bounded. Pattern 6.
    final_verdicts = round_1_verdicts
    if enable_reflection and len(round_1_verdicts) == 2:
        theorist_r1, da_r1 = round_1_verdicts
        reflect_theorist = asyncio.to_thread(
            _run_one_reflector,
            agent_name="behavioral_theorist",
            system_prompt=THEORIST_SYSTEM_PROMPT,
            allowed_tools=THEORIST_TOOLS,
            proposal=proposal,
            own_round_1=theorist_r1,
            peer_round_1=da_r1,
            api_key=api_key,
        )
        reflect_da = asyncio.to_thread(
            _run_one_reflector,
            agent_name="empirical_devils_advocate",
            system_prompt=DA_SYSTEM_PROMPT,
            allowed_tools=DA_TOOLS,
            proposal=proposal,
            own_round_1=da_r1,
            peer_round_1=theorist_r1,
            api_key=api_key,
        )
        final_verdicts = list(
            await asyncio.gather(reflect_theorist, reflect_da)
        )

    consensus, rationale = aggregate_verdicts(final_verdicts)

    council = CouncilVerdict(
        proposal=proposal,
        verdicts=final_verdicts,
        consensus=consensus,
        rationale=rationale,
        elapsed_s=time.perf_counter() - t0,
        round_1_consensus=round_1_consensus,
        round_1_rationale=round_1_rationale,
        reflection_enabled=enable_reflection,
    )
    # Persist to ledger so Cockpit / outcome-calibration can see this
    council.run_id = _append_to_ledger({
        "stage":     "critique_council",
        "consensus": consensus,
        "elapsed_s": round(council.elapsed_s, 2),
        "n_critics": len(council.verdicts),
        "n_tool_calls_total": sum(
            len(v.tool_calls) for v in council.verdicts
        ),
        "proposal":            proposal.to_dict(),
        "verdicts":            [v.to_dict() for v in council.verdicts],
        "rationale":           rationale,
        "round_1_consensus":   round_1_consensus,
        "round_1_rationale":   round_1_rationale,
        "reflection_enabled":  enable_reflection,
        "reflection_actions":  [
            v.reflection_action for v in council.verdicts
            if v.reflection_action is not None
        ] if enable_reflection else [],
    })
    return council


# ── Meta-aggregator ───────────────────────────────────────────────────


def aggregate_verdicts(
    verdicts: list[AgentVerdict],
) -> tuple[str, str]:
    """Combine individual verdicts into council consensus.

    Rules (codified, not LLM-judged — process rigor > LLM judgment):
      - ANY FAIL  → REJECT
      - ALL PASS  → APPROVE
      - any WARN  → NEEDS_REVISION
    Rationale is a synthesis of each critic's top concern.
    """
    if not verdicts:
        return "REJECT", "no critics responded"

    fails = [v for v in verdicts if v.verdict == "FAIL"]
    warns = [v for v in verdicts if v.verdict == "WARN"]
    passes = [v for v in verdicts if v.verdict == "PASS"]

    if fails:
        consensus = "REJECT"
    elif warns:
        consensus = "NEEDS_REVISION"
    else:
        consensus = "APPROVE"

    parts = []
    for v in verdicts:
        top = (v.fatal_red_flags[:1] or v.material_concerns[:1]
               or [v.rationale[:150]])[0]
        parts.append(f"[{v.agent_name} {v.verdict}] {top}")
    rationale = " || ".join(parts)

    if consensus == "REJECT":
        rationale = (
            f"REJECT due to FAIL verdict(s) from "
            f"{', '.join(v.agent_name for v in fails)}. " + rationale
        )
    elif consensus == "NEEDS_REVISION":
        rationale = (
            f"NEEDS_REVISION — {len(warns)} critic(s) raised material "
            f"concerns ({len(passes)} PASS). " + rationale
        )

    return consensus, rationale


# ── Stage A + B end-to-end ────────────────────────────────────────────


async def run_full_council(
    seed_idea: str,
    *,
    api_key: Optional[str] = None,
    enable_reflection: bool = False,
) -> tuple[ProposalDict, CouncilVerdict]:
    """Architect proposes → 2 critics fan out → consensus.

    The single entrypoint the Temporal outer-ring workflow (4c) calls
    per discovery loop iteration. Synchronous architect (blocking) +
    async critic fan-out — matches the data flow (must have proposal
    before critique).

    enable_reflection (Frontier 1, 2026-06-01):
      When True, runs an additional bounded reflection round after the
      initial critique. Each critic gets ONE shot to read the peer's
      verdict and either confirm or revise. ~2x LLM token cost; opt-in.
      Pattern 6 (DD workflow), NOT Pattern 5 (autonomous debate).
    """
    proposal = await asyncio.to_thread(architect_propose, seed_idea,
                                        api_key=api_key)
    council = await critique_council(
        proposal, api_key=api_key, enable_reflection=enable_reflection,
    )
    return proposal, council
