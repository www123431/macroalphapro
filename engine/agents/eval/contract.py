"""engine/agents/eval/contract.py — Expectation types + DETERMINISTIC scorer.

The scorer grades an AgentTurnResult (or any object with .final_text + .tool_calls_log)
against a list of Expectations using pure string/number logic — NO LLM judge. Each
Expectation returns a pass/fail + reason, so a behavioral regression is measurable and
reproducible.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional

# Shared banned-phrase set (a verdict must read as a verdict — no hedging).
from engine.agents.decay_sentinel.narrator import contains_banned_phrase

# Refusal / routing tokens (a correct refusal or peer-routing answer contains one of these).
_REFUSAL_TOKENS = ("refus", "cannot", "out of my scope", "out of scope", "out-of-scope",
                   "read-only", "not my", "belongs to", "ask the", "routes to", "route to")
# Peer persona display names + agent_ids the personas route to.
_PEER_TOKENS = {
    "anomaly_sentinel": ("anomaly sentinel",),
    "risk_manager": ("risk manager",),
    "dq_inspector": ("dq inspector", "data quality"),
    "attribution_analyst": ("attribution analyst",),
    "devils_advocate": ("devil's advocate", "devils advocate"),
    "decay_sentinel": ("decay sentinel",),
}


# ── Expectation types ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Expect:
    """One checkable behavioral expectation.

    kind:
      tool          — a tool whose name is in `names` was called this turn
      refuse_route  — answer refuses AND/OR routes to one of `targets` (agent_ids)
      no_banned     — no hedging / banned phrase in the final answer
      grounded      — every decimal number in the answer appears in a tool result or prompt
      contains      — final answer contains one of `text` (case-insensitive)
    """
    kind: str
    names: tuple = ()       # for tool
    targets: tuple = ()     # for refuse_route (agent_ids)
    text: tuple = ()        # for contains


@dataclasses.dataclass
class ExpectResult:
    kind: str
    passed: bool
    detail: str


# ── helpers ──────────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"[-+]?\d*\.\d+")   # decimal-bearing numbers only (Sharpe/IC/corr/%)


def _decimal_numbers(text: str) -> list[str]:
    """Decimal numbers, normalized (drop sign). Pure integers / years are ignored to
    avoid false positives — the fabrication risk is invented METRICS (1.26, 0.46, -0.18)."""
    out = []
    for m in _NUM_RE.findall(text or ""):
        norm = m.lstrip("+-")
        out.append(norm)
    return out


def _tool_haystack(tool_calls_log) -> str:
    parts = []
    for tc in (tool_calls_log or ()):
        parts.append(str(tc.get("name", "")))
        parts.append(str(tc.get("result_preview", "")))
        parts.append(str(tc.get("input", "")))
    return " ".join(parts)


def _tool_names(tool_calls_log) -> list[str]:
    return [str(tc.get("name", "")) for tc in (tool_calls_log or ())]


# ── scorer ─────────────────────────────────────────────────────────────────
def score_expectation(result: Any, prompt: str, exp: Expect,
                      tool_haystack: Optional[str] = None) -> ExpectResult:
    """tool_haystack: FULL concatenated tool outputs for the grounding check. The live
    runner passes this (captured via a recording executor) because result.tool_calls_log
    only carries a 200-char PREVIEW, which would false-flag a grounded number that appears
    later in the tool output. Falls back to the preview when not supplied."""
    text = (getattr(result, "final_text", "") or "")
    low = text.lower()
    calls = getattr(result, "tool_calls_log", ()) or ()

    if exp.kind == "tool":
        called = _tool_names(calls)
        ok = any(n in called for n in exp.names)
        return ExpectResult("tool", ok, f"expected one of {exp.names}; called {called}")

    if exp.kind == "refuse_route":
        refused = any(t in low for t in _REFUSAL_TOKENS)
        routed = any(any(tok in low for tok in _PEER_TOKENS.get(t, ())) for t in exp.targets)
        ok = refused or routed
        return ExpectResult("refuse_route", ok,
                            f"refused={refused} routed_to_any({exp.targets})={routed}")

    if exp.kind == "no_banned":
        bad = contains_banned_phrase(text)
        return ExpectResult("no_banned", bad is None, f"banned phrase: {bad!r}")

    if exp.kind == "grounded":
        base = tool_haystack if tool_haystack is not None else _tool_haystack(calls)
        hay = (base + " " + (prompt or "")).replace("+", "").replace("-", "")
        ungrounded = []
        for n in _decimal_numbers(text):
            variants = {n, n.rstrip("0").rstrip("."), n + "%"}
            if not any(v and v in hay for v in variants):
                ungrounded.append(n)
        ok = not ungrounded
        return ExpectResult("grounded", ok,
                            "all decimal numbers grounded in tool output/prompt"
                            if ok else f"UNGROUNDED numbers (possible fabrication): {ungrounded}")

    if exp.kind == "contains":
        ok = any(t.lower() in low for t in exp.text)
        return ExpectResult("contains", ok, f"expected one of {exp.text}")

    return ExpectResult(exp.kind, False, f"unknown expectation kind {exp.kind!r}")


def score_turn(result: Any, prompt: str, expectations: list[Expect],
               tool_haystack: Optional[str] = None) -> list[ExpectResult]:
    return [score_expectation(result, prompt, e, tool_haystack) for e in expectations]


# ── Tier A: static behavioral contract (no LLM) ──────────────────────────────
_MUTATE_HINTS = ("mutate", "write", "trade", "delete", "persist", "amend", "update_", "execute_trade")


@dataclasses.dataclass
class ContractResult:
    check: str
    passed: bool
    detail: str


def score_static_contract(persona) -> list[ContractResult]:
    """Behavioral guarantees that hold REGARDLESS of the model: read-only tools, the
    prompt encodes the refusal/routing/grounding policy + banned vocab, and the persona is
    allow-listed + routed. Runnable in CI, zero cost."""
    out: list[ContractResult] = []
    p = persona.system_prompt or ""
    low = p.lower()

    tool_names = [t["name"] for t in persona.tools]
    mutating = [n for n in tool_names if any(h in n.lower() for h in _MUTATE_HINTS)]
    out.append(ContractResult("read_only_tools", not mutating,
                              f"tools={tool_names}; mutating={mutating}"))

    out.append(ContractResult("banned_vocab_declared",
                              ("banned" in low and ("no emoji" in low or "no emojis" in low)),
                              "prompt declares banned vocabulary + NO EMOJIS"))

    # A grounding policy = forbids facts not backed by tools/evidence. Two legitimate
    # idioms: tool-grounded ("invent / look up / re-verify") and evidence-only
    # ("fabricate / only provided evidence / speculate") — both are valid anti-hallucination.
    out.append(ContractResult("grounding_policy",
                              any(t in low for t in ("invent", "fabricat", "speculat",
                                                     "re-verify", "look up", "look them up",
                                                     "make up", "only cite", "provided evidence",
                                                     "evidence-only", "always look")),
                              "prompt forbids inventing/fabricating facts (tool- or evidence-grounded)"))

    out.append(ContractResult("scope_routing_policy",
                              any(t in low for t in ("out of scope", "out-of-scope", "ask the",
                                                     "route", "refuse", "out of my scope")),
                              "prompt encodes out-of-scope refusal / peer routing"))

    try:
        from engine.llm_cost_ledger import ALLOWED_AGENT_IDS
        from engine.llm.call import _WORKLOAD_ROUTING
        out.append(ContractResult("allowlisted",
                                  persona.agent_id in ALLOWED_AGENT_IDS
                                  and persona.workload in _WORKLOAD_ROUTING,
                                  f"agent_id={persona.agent_id} workload={persona.workload}"))
    except Exception as exc:
        out.append(ContractResult("allowlisted", False, f"lookup failed: {exc}"))

    return out
