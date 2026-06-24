"""engine/agents/research_diagnostician/diagnostician.py — LLM-driven failure
diagnosis with tool-use + Reflexion refinement loop.

The Layer 3 (Reasoning) load-bearing build for the agentic AI architecture
per [[project-agentic-ai-real-architecture-2026-05-29]]. When the strict
gate returns non-GREEN, this module runs an Anthropic Claude tool-use loop
that:
  1. Receives the candidate name + initial context
  2. Calls 1+ deterministic tools from engine.agents.research_diagnostician.tools
     (each tool returns structured evidence — graveyard adjacency, deployed
     overlap, sample stress coverage, etc.)
  3. Synthesizes a causal diagnosis citing the tool evidence
  4. Self-critiques (Reflexion pattern, Shinn et al. 2023): "what's the
     strongest counter to my diagnosis?" — may call more tools, refines
  5. Stops on convergence (no new content for 2 rounds) OR max 3 critique
     rounds

Doctrine compliance
-------------------
- 0-LLM-in-DECISION: this module produces a NARRATIVE diagnosis. It does
  NOT override the gate verdict. The verdict is read-only input.
- Per pre-impl checklist A/B/C/D in the architecture memo:
  - C (Graceful degradation): falls back to deterministic synthesis if
    ANTHROPIC_API_KEY missing or any API failure
  - D (Human-in-the-loop): output is ADVISORY; never auto-triggers
    re-attempts, never modifies code or deployment
- Per Multi-Loop Reflexion discipline: every loop iteration has a specific
  question (diagnosis or critique target), a stopping criterion (convergence
  or max iters), and between-iteration diff visibility

Output schema
-------------
    {
      "candidate":         str,
      "verdict":           str,  # mirrors gate result, never modified
      "mode":              "llm" | "deterministic_fallback" | "deterministic_only",
      "initial_diagnosis": str,
      "refined_diagnosis": str,         # = initial if critique converged immediately
      "n_critique_rounds": int,
      "converged":         bool,
      "tools_called":      list[dict],  # tool name + args + result snippet
      "cost_usd":          float,       # 0.0 for deterministic
      "timestamp":         str,
    }
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path

from engine.agents.research_diagnostician.tools import (
    TOOL_SCHEMAS,
    execute_tool,
    fetch_gate_evidence,
    find_similar_candidates_t,
    check_deployed_overlap_t,
    sample_stress_coverage_t,
    subperiod_analysis_t,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGNOSTIC_LEDGER = REPO_ROOT / "data" / "research" / "diagnostic_reports.jsonl"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOOL_TURNS = 6              # Anthropic tool-use loop turns (per critique round)
MAX_CRITIQUE_ROUNDS = 3         # Reflexion outer loop
CONVERGENCE_REPEAT_THRESHOLD = 2 # consecutive no-change → stop


# ── System prompts ───────────────────────────────────────────────────────────

SYSTEM_DIAGNOSE = """You are the Research Diagnostician — an LLM-driven post-verdict analyst for a quantitative book. Your role: when a candidate strategy receives a non-GREEN verdict, use the provided tools to gather evidence and produce a CAUSAL DIAGNOSIS of why it failed (or why it's marginal).

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging.
NO EMOJIS in any response.
BANNED vocabulary: maybe, perhaps, could be, might be, probably, possibly, likely, I think, I feel, seems to, appears to.

# Doctrine
- The strict-gate VERDICT is final and not yours to change.
- Your job is the WHY layer — causal explanation, not new verdict.
- Cite specific numbers from the gate evidence and tool outputs.
- Identify ONE root cause (or two if equally weighted), not a list.

# Process
1. Start by calling `fetch_gate_evidence` for the candidate (always required).
2. Based on the evidence, call OTHER tools strategically:
   - High book correlation? → call `check_deployed_overlap`
   - Negative alpha-t? → call `find_similar_candidates` to see if prior REDs share theme
   - Standalone Sharpe weak? → call `sample_stress_coverage` to check if the sample is hostile
3. Synthesize a 2-4 sentence diagnosis stating the root cause + evidence.

# Output
End your final response with a single root-cause statement formatted exactly as:
ROOT CAUSE: <one sentence identifying the primary failure mode>

Do not list multiple equal causes; pick the dominant one. If truly co-dominant, name both connected by "AND".
"""

SYSTEM_CRITIQUE = """You are now critiquing your OWN previous diagnosis of a strategy candidate. Apply Reflexion-style self-critique:

1. What is the strongest counter-argument to your diagnosis?
2. Did you weight any evidence incorrectly?
3. Is there a different mechanism explanation you missed?
4. Did you miss any tool output that contradicts your conclusion?

You may call additional tools if needed.

# Output
If your previous diagnosis was complete and correct, respond ONLY with the literal text:
NO_CHANGE

If you have a refinement, produce the REFINED diagnosis with the same format:
- 2-4 sentences citing specific evidence
- End with: ROOT CAUSE: <one sentence>
- Banned vocabulary same as initial prompt
"""


# ── Anthropic key resolution ─────────────────────────────────────────────────

def _read_anthropic_key() -> str | None:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


# ── Tool-use loop (single critique round) ───────────────────────────────────

def _run_tool_use_loop(client, system: str, messages: list,
                       max_turns: int = MAX_TOOL_TURNS) -> tuple[str, list, list, float]:
    """Run an Anthropic tool-use loop until the LLM produces only text (no
    more tool calls) OR max_turns exhausted.

    Returns (final_text, updated_messages, tool_calls_log, cost_usd).
    """
    tool_calls_log = []
    cost_usd = 0.0

    for turn in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        # Track cost (approx)
        usage = response.usage
        cost_usd += (usage.input_tokens * 3.0 / 1_000_000
                     + usage.output_tokens * 15.0 / 1_000_000)

        # Parse response blocks
        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # Append assistant message (raw block list — Anthropic spec)
        messages.append({"role": "assistant", "content": response.content})

        if not tool_uses:
            return "\n\n".join(text_parts), messages, tool_calls_log, cost_usd

        # Execute each tool, append results
        tool_results = []
        for tu in tool_uses:
            result = execute_tool(tu.name, **tu.input)
            tool_calls_log.append({
                "name": tu.name,
                "input": dict(tu.input),
                "success": result.success,
                "payload_preview": (json.dumps(result.payload, default=str)[:200]
                                     if result.success else result.error),
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result.to_json(),
            })
        messages.append({"role": "user", "content": tool_results})

    # Max turns exhausted
    return f"[max_turns={max_turns} exhausted without final answer]", messages, tool_calls_log, cost_usd


# ── Deterministic fallback (no LLM) ──────────────────────────────────────────

def _diagnose_deterministic(candidate_name: str) -> dict:
    """Synthesize a template-based diagnosis from the 5 tools. No LLM."""
    ev = fetch_gate_evidence(candidate_name)
    if not ev.success:
        return {
            "candidate":         candidate_name,
            "verdict":           "UNKNOWN",
            "mode":              "deterministic_only",
            "initial_diagnosis": f"No gate_runs entry found for {candidate_name!r}.",
            "refined_diagnosis": f"No gate_runs entry found for {candidate_name!r}.",
            "n_critique_rounds": 0,
            "converged":         True,
            "tools_called":      [{"name": "fetch_gate_evidence", "success": False, "error": ev.error}],
            "cost_usd":          0.0,
            "timestamp":         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    gate = ev.payload
    verdict = gate.get("verdict", "UNKNOWN")
    sh = gate.get("standalone_sharpe")
    at = gate.get("alpha_t_ff5umd")
    at_p = gate.get("alpha_t_ff5umd_pead")
    cb = gate.get("corr_with_book")
    dsr = gate.get("deflated_sr")
    oos = gate.get("oos_sharpe")

    similar = find_similar_candidates_t(candidate_name)
    overlap = check_deployed_overlap_t(candidate_name)
    stress = sample_stress_coverage_t(candidate_name)

    # Synthesize root cause heuristically
    causes = []
    # PEAD-control deterioration (book cousin)
    if at is not None and at_p is not None and abs(at_p) > abs(at) + 1.0:
        if overlap.success and overlap.payload.get("n_overlapping_sleeves", 0) > 0:
            sleeve_names = list(overlap.payload.get("overlap_by_sleeve", {}).keys())
            causes.append(
                f"PEAD-cousin / mechanism redundancy: α-t worsens from {at} (FF5+UMD) "
                f"to {at_p} (with PEAD control), and structural overlap exists with "
                f"deployed sleeve(s) {sleeve_names}. The candidate captures information "
                f"already in the book."
            )
    # Wrong-direction significant alpha
    if at is not None and at < -3.0:
        causes.append(
            f"Wrong-direction significant alpha (α-t {at} vs FF5+UMD). The mechanism "
            f"is operating against its published direction — likely sample-era reversal "
            f"of the documented anomaly."
        )
    # Weak standalone with no book-redeeming feature
    if sh is not None and sh < 0.3 and (cb is None or abs(cb) > 0.3):
        causes.append(
            f"Weak standalone Sharpe ({sh}) without compensating diversification "
            f"benefit (book corr {cb}). Cost-adjusted realizable returns insufficient."
        )
    # Sample stress coverage — only flag if sample window is KNOWN.
    # Legacy gate_runs entries lack sample_start/end and would otherwise
    # all spuriously flag "0/9 stress periods" — that's noise, not signal.
    if stress.success:
        has_window = bool(stress.payload.get("sample_start")
                            and stress.payload.get("sample_end"))
        n_covered = len(stress.payload.get("stress_covered", []))
        if has_window and n_covered < 3:
            causes.append(
                f"Sample covers only {n_covered} of 9 canonical stress periods "
                f"(missed: {stress.payload.get('stress_missed', [])[:3]}...). "
                f"Strategy robustness across regimes undermeasured."
            )

    if causes:
        diagnosis = " ".join(causes[:2])  # top 2 causes
    else:
        diagnosis = (
            f"{candidate_name} received {verdict} with standalone Sharpe {sh}, "
            f"α-t {at}, deflated SR {dsr}, OOS {oos}. No single dominant failure mode "
            f"identified by deterministic heuristics."
        )

    diagnosis += f"\n\nROOT CAUSE: {causes[0].split(':')[0] if causes else 'multiple weak factors no single dominant'}"

    return {
        "candidate":         candidate_name,
        "verdict":           verdict,
        "mode":              "deterministic_only",
        "initial_diagnosis": diagnosis,
        "refined_diagnosis": diagnosis,
        "n_critique_rounds": 0,
        "converged":         True,
        "tools_called":      [
            {"name": "fetch_gate_evidence", "success": ev.success},
            {"name": "find_similar_candidates", "success": similar.success},
            {"name": "check_deployed_overlap", "success": overlap.success},
            {"name": "sample_stress_coverage", "success": stress.success},
        ],
        "cost_usd":          0.0,
        "timestamp":         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ── LLM diagnosis with Reflexion critique loop ───────────────────────────────

def _diagnose_llm(candidate_name: str, with_critique: bool = True) -> dict:
    """LLM-driven diagnosis with tool-use + optional Reflexion critique loop.
    Falls back to deterministic on API failure (per pre-impl checklist C)."""
    key = _read_anthropic_key()
    if not key:
        out = _diagnose_deterministic(candidate_name)
        out["mode"] = "deterministic_fallback_no_api_key"
        return out

    try:
        from anthropic import Anthropic
    except ImportError:
        out = _diagnose_deterministic(candidate_name)
        out["mode"] = "deterministic_fallback_no_sdk"
        return out

    try:
        client = Anthropic(api_key=key, timeout=120.0)

        # Initial diagnosis
        initial_messages = [
            {"role": "user", "content":
                f"Diagnose candidate {candidate_name!r}. Begin by calling "
                f"fetch_gate_evidence to see the verdict and metrics, then use "
                f"other tools as needed. Produce a causal diagnosis."}
        ]
        initial_text, messages, tool_calls, cost = _run_tool_use_loop(
            client, SYSTEM_DIAGNOSE, initial_messages, max_turns=MAX_TOOL_TURNS)

        refined_text = initial_text
        n_rounds = 0
        converged = True
        history = [initial_text]
        total_cost = cost

        if with_critique:
            no_change_count = 0
            for round_idx in range(MAX_CRITIQUE_ROUNDS):
                # Append critique prompt
                messages.append({
                    "role": "user",
                    "content": (
                        f"Critique your previous diagnosis. If complete, "
                        f"respond ONLY with 'NO_CHANGE'. If refinement needed, "
                        f"produce the REFINED diagnosis."
                    ),
                })
                new_text, messages, new_tool_calls, new_cost = _run_tool_use_loop(
                    client, SYSTEM_CRITIQUE, messages, max_turns=MAX_TOOL_TURNS)
                tool_calls.extend(new_tool_calls)
                total_cost += new_cost
                n_rounds = round_idx + 1

                stripped = new_text.strip()
                if "NO_CHANGE" in stripped or stripped == refined_text.strip():
                    no_change_count += 1
                    if no_change_count >= CONVERGENCE_REPEAT_THRESHOLD - 1:
                        converged = True
                        break
                else:
                    no_change_count = 0
                    refined_text = new_text
                    history.append(new_text)
            else:
                converged = False    # ran out of rounds without converging

        return {
            "candidate":         candidate_name,
            "verdict":           _extract_verdict_from_tools(tool_calls) or "UNKNOWN",
            "mode":              "llm",
            "initial_diagnosis": initial_text,
            "refined_diagnosis": refined_text,
            "n_critique_rounds": n_rounds,
            "converged":         converged,
            "tools_called":      tool_calls,
            "cost_usd":          round(total_cost, 4),
            "timestamp":         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    except Exception as exc:
        logger.warning("LLM diagnostician failed (%s); falling back deterministic", exc)
        out = _diagnose_deterministic(candidate_name)
        out["mode"] = f"deterministic_fallback_{type(exc).__name__.lower()}"
        return out


def _extract_verdict_from_tools(tool_calls: list[dict]) -> str | None:
    for tc in tool_calls:
        if tc.get("name") == "fetch_gate_evidence" and tc.get("success"):
            preview = tc.get("payload_preview", "")
            # crude regex parse
            import re
            m = re.search(r'"verdict"\s*:\s*"([^"]+)"', preview)
            if m:
                return m.group(1)
    return None


# ── Public entry ─────────────────────────────────────────────────────────────

def diagnose(candidate_name: str, use_llm: bool = True,
             with_critique: bool = True, log: bool = True) -> dict:
    """Top-level entry. Use deterministic mode for tests / cron when LLM cost
    matters; use_llm=True for interactive forensic work."""
    if use_llm:
        result = _diagnose_llm(candidate_name, with_critique=with_critique)
    else:
        result = _diagnose_deterministic(candidate_name)

    if log:
        DIAGNOSTIC_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with DIAGNOSTIC_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    return result


SYSTEM_DIAGNOSE_SLEEVE = """You are the Research Diagnostician — diagnosing a DEPLOYED SLEEVE that is degrading or has been flagged for review (NOT a new candidate that failed the gate).

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging. NO EMOJIS.
BANNED vocabulary: maybe, perhaps, could be, might be, probably, possibly, likely, I think, I feel, seems to, appears to.

# Doctrine
- The deterministic Decay Sentinel produces the verdict (HEALTHY/WATCH/ACTION). Your job is the causal WHY.
- Cite specific numbers from tool outputs.
- Identify the DOMINANT cause; do not list co-equal candidates unless the evidence genuinely supports it.

# Process
1. Start with `fetch_sleeve_health_history` to see the rolling Sharpe / decay ratio / signal IC trajectory.
2. Use `check_deployed_overlap` with the sleeve name to see book-level adjacencies (the sleeve IS a deployed entity; this tool can tell you which other sleeves are mechanistically adjacent).
3. Optionally use `find_similar_candidates` to see which prior RED CANDIDATES are mechanistically similar (might indicate this sleeve is now showing the failure mode of those REDs).
4. Synthesize 2-4 sentences citing specific evidence.

# Output
End with: ROOT CAUSE: <one sentence>
"""


def diagnose_sleeve(sleeve_name: str, use_llm: bool = True,
                    with_critique: bool = True, log: bool = True) -> dict:
    """Forensic diagnosis of a deployed sleeve's health. Manual entry point —
    NOT auto-triggered by the cron. Use when Decay Sentinel reports a status
    you want a causal explanation for.

    Differs from diagnose() (which handles new gate-tested CANDIDATES):
      - Input is a sleeve name, not a candidate name
      - Uses fetch_sleeve_health_history tool (not fetch_gate_evidence)
      - Prompt frames the task as forensic, not gate-rejection-cause
    """
    if not use_llm:
        # Deterministic stub: read sleeve history and summarize
        from engine.agents.research_diagnostician.tools import (
            fetch_sleeve_health_history_t, check_deployed_overlap_t,
        )
        hist = fetch_sleeve_health_history_t(sleeve_name, n_days=30)
        overlap = check_deployed_overlap_t(sleeve_name)
        out = {
            "sleeve":         sleeve_name,
            "mode":           "deterministic_only",
            "initial_diagnosis": (
                f"Deterministic forensic stub for {sleeve_name}: "
                f"{hist.payload.get('n_days_found', 0)} days of decay history examined; "
                f"{overlap.payload.get('n_overlapping_sleeves', 'n/a')} adjacent sleeve(s) detected."
            ),
            "refined_diagnosis": (
                f"Deterministic forensic stub for {sleeve_name}: use LLM mode for "
                f"causal narrative."
            ),
            "n_critique_rounds": 0,
            "converged":      True,
            "tools_called":   [
                {"name": "fetch_sleeve_health_history", "success": hist.success},
                {"name": "check_deployed_overlap",       "success": overlap.success},
            ],
            "cost_usd":       0.0,
            "timestamp":      datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        if log:
            DIAGNOSTIC_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with DIAGNOSTIC_LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
        return out

    # LLM path
    key = _read_anthropic_key()
    if not key:
        return diagnose_sleeve(sleeve_name, use_llm=False, with_critique=False, log=log)
    try:
        from anthropic import Anthropic
    except ImportError:
        return diagnose_sleeve(sleeve_name, use_llm=False, with_critique=False, log=log)

    try:
        client = Anthropic(api_key=key, timeout=120.0)
        initial_messages = [
            {"role": "user", "content":
                f"Diagnose the deployed sleeve {sleeve_name!r}. Begin by calling "
                f"fetch_sleeve_health_history to see its trajectory, then use other "
                f"tools as needed. Produce a causal diagnosis of its current health."}
        ]
        initial_text, messages, tool_calls, cost = _run_tool_use_loop(
            client, SYSTEM_DIAGNOSE_SLEEVE, initial_messages, max_turns=MAX_TOOL_TURNS)

        refined_text = initial_text
        n_rounds = 0
        converged = True
        total_cost = cost

        if with_critique:
            no_change_count = 0
            for round_idx in range(MAX_CRITIQUE_ROUNDS):
                messages.append({
                    "role": "user",
                    "content": "Critique your previous diagnosis. If complete, respond ONLY with 'NO_CHANGE'. If refinement needed, produce the REFINED diagnosis.",
                })
                new_text, messages, new_tool_calls, new_cost = _run_tool_use_loop(
                    client, SYSTEM_DIAGNOSE_SLEEVE, messages, max_turns=MAX_TOOL_TURNS)
                tool_calls.extend(new_tool_calls)
                total_cost += new_cost
                n_rounds = round_idx + 1
                stripped = new_text.strip()
                if "NO_CHANGE" in stripped or stripped == refined_text.strip():
                    no_change_count += 1
                    if no_change_count >= CONVERGENCE_REPEAT_THRESHOLD - 1:
                        break
                else:
                    no_change_count = 0
                    refined_text = new_text
            else:
                converged = False

        result = {
            "sleeve":            sleeve_name,
            "mode":              "llm",
            "initial_diagnosis": initial_text,
            "refined_diagnosis": refined_text,
            "n_critique_rounds": n_rounds,
            "converged":         converged,
            "tools_called":      tool_calls,
            "cost_usd":          round(total_cost, 4),
            "timestamp":         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        if log:
            DIAGNOSTIC_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with DIAGNOSTIC_LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        return result
    except Exception as exc:
        logger.warning("LLM sleeve diagnosis failed (%s); falling back", exc)
        return diagnose_sleeve(sleeve_name, use_llm=False, with_critique=False, log=log)


def read_diagnostic_ledger(limit: int = 50) -> list[dict]:
    if not DIAGNOSTIC_LEDGER.exists():
        return []
    rows = [json.loads(x) for x in DIAGNOSTIC_LEDGER.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    return rows[-limit:][::-1]
