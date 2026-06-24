"""engine/research/hypothesis_generator.py — Phase 2 ② LLM hypothesis generator.

Library-TRAVERSER (not creator). Selects a `currently_unexplored_in_our_book:
true` entry from the Mechanism Library, runs it through the 7 hygiene tools
(H1-H7), and proposes a pre-committed test design.

5 iron rules (system prompt + tool enforcement):
- R1 NO INVENT: LLM cannot propose a mechanism that's not already in library.
                 Inventions go to library_update_proposal (NOT a generator output).
- R2 EVIDENCE-FIRST: Every proposal field must trace to a tool call.
- R3 PARENT-FAMILY OVERRIDE: H2 hard_reject = abandon mechanism.
- R4 NO GRID HIDE: H5 reject_grid_hide = abandon proposal.
- R5 NO PROPOSAL = SUCCESS: empty H1 list → output 'no_proposal' (correct).

2-stage design:
  Stage 1: Generator (Claude tool-use loop with H1-H6)
  Stage 2: Adversarial critique via H7 (v1 deterministic; v2 = devils_advocate
             persona on DeepSeek for cross-vendor)
  Output: data/research/proposal_queue.jsonl (human review)

Doctrine:
- NEVER auto-runs gate. Output is a PROPOSAL.
- NEVER auto-flips audit_signature.
- NEVER mutates library YAMLs (writer's job, not generator's).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path

from engine.research.hygiene_tools import (
    TOOL_SCHEMAS as HYGIENE_SCHEMAS,
    execute_tool as run_hygiene,
    h1_list_unexplored_library_entries,
    h2_cousin_check_multilevel,
    h3_check_data_inventory,
    h4_verify_paper_in_library,
    h5_count_free_params,
    h6_post_pub_evidence_check,
    h7_kill_this_proposal,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
PROPOSAL_QUEUE = REPO_ROOT / "data" / "research" / "proposal_queue.jsonl"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOOL_TURNS = 8


@dataclasses.dataclass
class GeneratorProposal:
    mechanism_id:    str
    canonical_paper_id: str
    sample_start:    str         # YYYY-MM-DD
    sample_end:      str
    parameters:      list[str]   # pre-committed single-value list
    justification:   str
    hygiene_summary: dict        # H1-H6 result roll-up
    h7_critique:     dict        # H7 kill/survive
    execution_template: dict | None = None  # Layer 2: bridge to DSL runner
    proposed_ts:     str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Deterministic generator (no LLM) ────────────────────────────────────

def _deterministic_propose(*, include_pending: bool = False) -> dict:
    """Build a proposal for the FIRST H1 entry that survives H2/H3/H6.

    No LLM. Uses canonical_horizon as canonical_universe params; sample
    window from typical_sample. This is a v1 fallback / test entry point —
    not the production code path.

    Returns {proposal: GeneratorProposal | None, reason: str, ...}.
    """
    h1 = h1_list_unexplored_library_entries(include_pending=include_pending)
    candidates = h1.payload.get("entries") or []
    if not candidates:
        return {"proposal": None, "mode": "deterministic_only",
                "reason": "no unexplored visible library entries (R5: this is success)"}

    skipped = []
    for entry_summary in candidates:
        mid = entry_summary["id"]

        # H2 cousin check
        h2 = h2_cousin_check_multilevel(mid)
        if not h2.success or h2.payload["verdict"] == "hard_reject":
            skipped.append({"mid": mid, "stage": "H2",
                              "verdict": h2.payload.get("verdict"),
                              "reasons": h2.payload.get("hard_reject_reasons", [])})
            continue

        # Load full library entry for parameter/sample info
        import yaml
        try:
            full = yaml.safe_load(
                (LIBRARY_DIR / f"{mid}.yaml").read_text(encoding="utf-8"))
        except Exception as e:
            skipped.append({"mid": mid, "stage": "load_yaml", "error": str(e)})
            continue

        # H3 data check
        h3 = h3_check_data_inventory(full.get("required_data") or [])
        if not h3.payload["all_present"]:
            skipped.append({"mid": mid, "stage": "H3",
                              "missing": h3.payload["missing"]})
            continue

        # H4 canonical paper check
        cpid = full.get("canonical_paper_id")
        h4 = h4_verify_paper_in_library(cpid)
        if not h4.payload.get("verified"):
            skipped.append({"mid": mid, "stage": "H4",
                              "paper_id": cpid,
                              "reason": h4.payload.get("reason")})
            continue

        # H5 param check — use canonical_horizon as the param spec
        ch = full.get("canonical_horizon")
        params = [f"horizon={ch}"] if ch else []
        h5 = h5_count_free_params(params)
        if h5.payload["verdict"] != "ok":
            skipped.append({"mid": mid, "stage": "H5",
                              "rejected": h5.payload["rejected"]})
            continue

        # H6 post-pub evidence
        h6 = h6_post_pub_evidence_check(mid)
        if h6.payload.get("verdict") != "ok":
            skipped.append({"mid": mid, "stage": "H6",
                              "reason": h6.payload.get("verdict")})
            continue

        # Build proposal
        ts = full.get("typical_sample") or ""
        # Crude parse "1965-present" → 1965-01-01 / 2026-05-30
        sample_start = "1965-01-01"
        sample_end = datetime.date.today().isoformat()
        import re
        m = re.search(r"(\d{4})", ts)
        if m:
            sample_start = f"{m.group(1)}-01-01"

        proposal = GeneratorProposal(
            mechanism_id=    mid,
            canonical_paper_id= cpid,
            sample_start=    sample_start,
            sample_end=      sample_end,
            parameters=      params,
            execution_template= full.get("execution_template"),    # Layer 2 → 3 bridge
            justification=   (
                f"Selected from H1 unexplored list. H2 verdict "
                f"{h2.payload['verdict']}. H3 data inventory clean. "
                f"H4 paper {cpid} verified via crossref. "
                f"H5 parameters single-value. H6 has "
                f"{h6.payload['n_qualifying']} post-pub replications. "
                f"Sample window pre-committed from library typical_sample field. "
                f"This is a deterministic v1 proposal (LLM mode would add adaptive "
                f"sample design + cost model selection)."
            ),
            hygiene_summary={
                "H1_unexplored_count": len(candidates),
                "H2_verdict":          h2.payload["verdict"],
                "H3_all_data_present": h3.payload["all_present"],
                "H4_paper_verified":   h4.payload["verified"],
                "H5_verdict":          h5.payload["verdict"],
                "H6_n_qualifying":     h6.payload["n_qualifying"],
                "H6_verdict":          h6.payload["verdict"],
            },
            h7_critique={},
            proposed_ts=     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

        # Stage 2: H7 critique
        h7_input = {
            "mechanism_id":              mid,
            "canonical_paper_id":        cpid,
            "sample_start":              sample_start,
            "sample_end":                sample_end,
            "parameters":                params,
            "justification":             proposal.justification,
            "h2_cousin_check_result":    h2.payload,
            "h3_data_check_result":      h3.payload,
            "h4_paper_check_result":     h4.payload,
            "h5_param_check_result":     h5.payload,
            "h6_post_pub_check_result":  h6.payload,
        }
        h7 = h7_kill_this_proposal(h7_input)
        proposal.h7_critique = h7.payload

        if h7.payload["verdict"] == "kill":
            skipped.append({"mid": mid, "stage": "H7",
                              "kill_reasons": h7.payload["kill_reasons"]})
            continue

        return {
            "proposal":     proposal.to_dict(),
            "mode":         "deterministic_only",
            "n_candidates_considered": len(candidates),
            "skipped":      skipped,
        }

    return {
        "proposal": None,
        "mode":     "deterministic_only",
        "reason":   "all unexplored candidates failed hygiene gates",
        "skipped":  skipped,
    }


# ── LLM generator (Anthropic tool-use loop) ─────────────────────────────

SYSTEM_GENERATE = """You are the Hypothesis Generator. Your job is to select ONE library mechanism to propose for our strict-gate testing pipeline.

# IRON RULES (hard)
R1 NO INVENT — You may ONLY propose a mechanism whose `id` is returned by h1_list_unexplored_library_entries. Inventing a new mechanism is a separate workflow (library_update_proposal).
R2 EVIDENCE-FIRST — Every field in your proposal must trace to a tool call result. Never guess.
R3 PARENT-FAMILY OVERRIDE — If h2_cousin_check_multilevel returns hard_reject, abandon that mechanism entirely.
R4 NO GRID HIDE — Pre-commit ONE value per parameter (`lookback=12`, not `lookback ∈ [3,6,12]`).
R5 NO PROPOSAL = SUCCESS — If h1 returns 0 entries OR every entry hard-rejects through h2-h6, return {"proposal": null, "reason": "..."}. This is the CORRECT output, not a failure.

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging.
BANNED vocabulary: maybe, perhaps, probably, possibly, likely, I think, I feel, seems to, appears to.

# Process
1. Call h1_list_unexplored_library_entries (no args) → see candidate list
2. For your CHOSEN mechanism, call h2/h3/h4 (canonical_paper_id)/h5/h6 IN ORDER
3. Stop at FIRST hard_reject — try the next candidate
4. After all H1-H6 pass, build your final proposal JSON with these fields:
   - mechanism_id
   - canonical_paper_id
   - sample_start (YYYY-MM-DD)
   - sample_end (YYYY-MM-DD)
   - parameters: list of single-value strings
   - justification: 2-4 sentences citing specific tool results
5. Call h7_kill_this_proposal with the full proposal embedded
6. If H7 says kill, revise and re-validate ONCE, else return null proposal

# Output
End with the proposal JSON on a single line. If no proposal, output:
{"proposal": null, "reason": "..."}
"""


def _read_anthropic_key() -> str | None:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


def _llm_propose() -> dict:
    key = _read_anthropic_key()
    if not key:
        out = _deterministic_propose()
        out["mode"] = "deterministic_fallback (no API key)"
        return out
    try:
        from anthropic import Anthropic
    except ImportError:
        out = _deterministic_propose()
        out["mode"] = "deterministic_fallback (anthropic not installed)"
        return out

    client = Anthropic(api_key=key, timeout=120.0)
    messages = [{"role": "user", "content":
        "Begin by calling h1_list_unexplored_library_entries. Then traverse "
        "the 7-stage pipeline per the system prompt. Output the final proposal "
        "JSON or {\"proposal\": null, ...}."}]

    tool_calls = []
    cost = 0.0
    final_text = ""

    for turn in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3072,
            system=SYSTEM_GENERATE,
            tools=HYGIENE_SCHEMAS,
            messages=messages,
        )
        usage = response.usage
        cost += (usage.input_tokens * 3.0 / 1_000_000
                  + usage.output_tokens * 15.0 / 1_000_000)
        text_parts, tool_uses = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)
        messages.append({"role": "assistant", "content": response.content})

        if not tool_uses:
            final_text = "\n\n".join(text_parts)
            break

        tool_results = []
        for tu in tool_uses:
            result = run_hygiene(tu.name, **tu.input)
            tool_calls.append({
                "name":    tu.name,
                "input":   dict(tu.input),
                "success": result.success,
                "payload_preview": (json.dumps(result.payload, default=str)[:300]
                                     if result.success else result.error),
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result.to_json(),
            })
        messages.append({"role": "user", "content": tool_results})

    parsed = _extract_json(final_text)
    proposal = parsed.get("proposal") if parsed else None
    return {
        "proposal":   proposal,
        "mode":       "llm",
        "raw_text":   final_text,
        "tool_calls": tool_calls,
        "cost_usd":   round(cost, 4),
    }


# ── Public entry ────────────────────────────────────────────────────────

def generate_proposal(*, use_llm: bool = True, log: bool = True) -> dict:
    """Generate ONE hypothesis proposal (or 'no proposal' = R5 SUCCESS).

    Args:
      use_llm: if True, use Anthropic tool-use loop; else deterministic v1
      log:     if True AND proposal is non-null AND H7 verdict=survive,
                 append to data/research/proposal_queue.jsonl
    """
    result = _llm_propose() if use_llm else _deterministic_propose()

    proposal = result.get("proposal")
    if log and proposal:
        h7v = (proposal.get("h7_critique") or {}).get("verdict")
        if h7v != "kill":
            PROPOSAL_QUEUE.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts":        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "mode":      result["mode"],
                "proposal":  proposal,
            }
            with PROPOSAL_QUEUE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    return result


def read_proposal_queue(limit: int = 50) -> list[dict]:
    if not PROPOSAL_QUEUE.exists():
        return []
    rows = [json.loads(l) for l in PROPOSAL_QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--include-pending", action="store_true",
                         help="Use --no-llm with this to consider audit_signature=pending entries")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.no_llm and args.include_pending:
        res = _deterministic_propose(include_pending=True)
    else:
        res = generate_proposal(use_llm=not args.no_llm, log=not args.no_log)
    print(json.dumps(res, indent=2, default=str))
