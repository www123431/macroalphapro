"""engine/research/mutation_proposer.py — Phase 1 ① diagnostician → controlled mutation.

Closes the loop after a RED/YELLOW diagnostician verdict: agentic proposes a
SINGLE-PARAMETER controlled variant that:
- Touches exactly 1 of 4 WHITELISTED parameter types
- Pre-commits all other parameters from original spec
- Adds 1 to n_trials (so Deflated SR remains rigorous)
- NEVER flips sign / changes signal construction / opens grid search
- Cites a SPECIFIC diagnostic claim as justification

Doctrine (must respect):
- Max 2 mutations per original candidate — else p-hacking via mutation
- Sign flip BANNED (per strict-gate doctrine)
- Different signal construction = NEW CANDIDATE (separate pre-reg), not mutation
- All mutations PRE-COMMITTED before running gate
- Output is a PROPOSAL not an auto-execution. Human reviews; gate runs only on approval.

Whitelist of mutation types:
  sample_window  — restrict/extend test sample to address regime-specific failure
  cost_model     — adjust execution cost with justification (no looser than canonical)
  horizon        — change holding period (e.g. 30d → 60d)
  weighting      — change weighting scheme (EW ↔ VW; decile ↔ tercile)

Not in whitelist (= REJECT):
  signal_construction, signal_sign, universe definition, signal threshold
  (these are new candidates, not mutations)

Output format (MutationProposal dataclass):
  candidate_name, mutation_type, old_value, new_value, justification,
  cited_diagnosis_ts, n_trials_added=1, mutation_seq (1 or 2)

Persistence: data/research/mutation_proposals.jsonl (append-only).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_LEDGER = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
DIAGNOSTIC_LEDGER = REPO_ROOT / "data" / "research" / "diagnostic_reports.jsonl"
MUTATION_LEDGER = REPO_ROOT / "data" / "research" / "mutation_proposals.jsonl"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOOL_TURNS = 6
MAX_MUTATIONS_PER_CANDIDATE = 2


# ── Whitelist + validation rules ─────────────────────────────────────────────

WHITELIST_MUTATION_TYPES = frozenset([
    "sample_window",
    "cost_model",
    "horizon",
    "weighting",
])


@dataclasses.dataclass
class MutationProposal:
    candidate_name:    str
    mutation_type:     str       # one of WHITELIST_MUTATION_TYPES
    old_value:         str
    new_value:         str
    justification:     str       # must cite specific diagnostic claim
    cited_diagnosis_ts: str | None  # ts of the diagnostic_reports entry
    n_trials_added:    int = 1
    mutation_seq:      int = 1   # 1 or 2 — fails validation if >2
    proposed_ts:       str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ValidationResult:
    ok:      bool
    reasons: list[str]            # specific reject reasons; empty if ok

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Deterministic validator (the gate against LLM noise) ─────────────────────

def validate_mutation_proposal(proposal_json: dict | MutationProposal) -> ValidationResult:
    """Deterministic check that a proposal stays inside the controlled-mutation
    discipline. Used as a tool the LLM proposer can call, AND as a gate
    before any mutation is logged."""
    p = (proposal_json if isinstance(proposal_json, dict)
         else proposal_json.to_dict())
    reasons: list[str] = []

    # Required fields
    for k in ("candidate_name", "mutation_type", "old_value",
               "new_value", "justification"):
        if not p.get(k):
            reasons.append(f"missing required field: {k}")

    mt = p.get("mutation_type")
    if mt and mt not in WHITELIST_MUTATION_TYPES:
        reasons.append(
            f"mutation_type {mt!r} not in whitelist "
            f"{sorted(WHITELIST_MUTATION_TYPES)}"
        )

    # Reject signal-construction / sign-flip patterns by keyword scan
    ov = (str(p.get("old_value") or "") + " " + str(p.get("new_value") or "")).lower()
    if mt == "weighting" and any(t in ov for t in ("long-short flip",
                                                     "sign flip", "invert")):
        reasons.append("sign-flip mutation BANNED")
    if mt and any(t in ov for t in ("new signal", "different signal",
                                     "alternative construction",
                                     "different formula")):
        reasons.append("changing signal construction is a NEW CANDIDATE, not a mutation")

    # Mutation seq cap
    seq = p.get("mutation_seq", 1)
    try:
        seq = int(seq)
    except (TypeError, ValueError):
        seq = 1
    if seq > MAX_MUTATIONS_PER_CANDIDATE:
        reasons.append(
            f"mutation_seq {seq} > max {MAX_MUTATIONS_PER_CANDIDATE}: "
            f"further mutations = p-hacking"
        )

    # Existing mutation count for this candidate
    existing = _count_existing_mutations(p.get("candidate_name") or "")
    if existing >= MAX_MUTATIONS_PER_CANDIDATE:
        reasons.append(
            f"candidate already has {existing} mutation(s) on file; "
            f"max {MAX_MUTATIONS_PER_CANDIDATE} per original"
        )

    # n_trials must be exactly 1
    if p.get("n_trials_added", 1) != 1:
        reasons.append(
            f"n_trials_added must be exactly 1 per mutation; got "
            f"{p.get('n_trials_added')}"
        )

    # Justification must reference diagnostic
    if not p.get("cited_diagnosis_ts"):
        reasons.append(
            "cited_diagnosis_ts is required: a mutation must reference a "
            "specific diagnostic report"
        )

    # Justification not allowed to be vague
    just = str(p.get("justification") or "").strip()
    if just and len(just) < 30:
        reasons.append(
            f"justification too short ({len(just)} chars; need ≥30 + cite diagnosis)"
        )

    return ValidationResult(ok=not reasons, reasons=reasons)


def _count_existing_mutations(candidate_name: str) -> int:
    if not MUTATION_LEDGER.exists() or not candidate_name:
        return 0
    cnt = 0
    for line in MUTATION_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("candidate_name") == candidate_name:
            cnt += 1
    return cnt


# ── Ledger helpers ───────────────────────────────────────────────────────────

def _read_jsonl(path: Path, limit: int = 1000) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out[-limit:]


def _find_latest_diagnostic(candidate_name: str) -> dict | None:
    for row in reversed(_read_jsonl(DIAGNOSTIC_LEDGER, limit=5000)):
        if row.get("candidate") == candidate_name:
            return row
    return None


def _find_latest_gate_run(candidate_name: str) -> dict | None:
    for row in reversed(_read_jsonl(GATE_LEDGER, limit=5000)):
        if row.get("name") == candidate_name:
            return row
    return None


def _append_proposal(proposal: MutationProposal,
                      validation: ValidationResult) -> None:
    MUTATION_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = proposal.to_dict()
    entry["validation"] = validation.to_dict()
    entry["written_ts"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with MUTATION_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Deterministic proposer (pattern-match, no LLM) ───────────────────────────

_PATTERNS = [
    # (diagnostic-substring, mutation_type, suggested mutation template)
    ("missed 2008", "sample_window", "extend sample to include 2008 GFC"),
    ("missed 2018", "sample_window", "extend sample to include 2018 Vol-mageddon"),
    ("missed 2020", "sample_window", "extend sample to include 2020 COVID"),
    ("missed 2022", "sample_window", "extend sample to include 2022 rate-crash"),
    ("cost too aggressive", "cost_model", "tighten cost to academic canonical"),
    ("cost too loose", "cost_model", "tighten cost to academic canonical"),
    ("horizon", "horizon", "adjust holding period within canonical range"),
    ("equal-weight", "weighting", "switch from EW to VW per academic canonical"),
    ("decile", "weighting", "narrow from decile to tercile to reduce noise"),
]


def _propose_deterministic(candidate_name: str) -> dict:
    diag = _find_latest_diagnostic(candidate_name)
    gate = _find_latest_gate_run(candidate_name)
    if not diag:
        return {
            "candidate":   candidate_name,
            "mode":        "deterministic_only",
            "proposal":    None,
            "reason":      "no diagnostic ledger entry — run diagnose() first",
        }
    if not gate:
        return {
            "candidate":   candidate_name,
            "mode":        "deterministic_only",
            "proposal":    None,
            "reason":      "no gate_runs entry for this candidate",
        }
    if gate.get("verdict") == "GREEN":
        return {
            "candidate":   candidate_name,
            "mode":        "deterministic_only",
            "proposal":    None,
            "reason":      "GREEN verdict — no mutation needed",
        }

    diag_text = (diag.get("refined_diagnosis") or "").lower()
    matched_type = None
    matched_template = None
    for substring, mut_type, template in _PATTERNS:
        if substring in diag_text:
            matched_type = mut_type
            matched_template = template
            break

    if not matched_type:
        return {
            "candidate":   candidate_name,
            "mode":        "deterministic_only",
            "proposal":    None,
            "reason":      "no whitelisted-mutation pattern found in diagnosis text",
        }

    seq = _count_existing_mutations(candidate_name) + 1
    proposal = MutationProposal(
        candidate_name=    candidate_name,
        mutation_type=     matched_type,
        old_value=         "canonical original spec",
        new_value=         matched_template,
        justification=     (
            f"Deterministic pattern-match: diagnosis cites failure mode "
            f"consistent with {matched_type} mutation. Template: {matched_template}."
        ),
        cited_diagnosis_ts= diag.get("timestamp"),
        mutation_seq=      seq,
        proposed_ts=       datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    validation = validate_mutation_proposal(proposal)
    return {
        "candidate":   candidate_name,
        "mode":        "deterministic_only",
        "proposal":    proposal.to_dict(),
        "validation":  validation.to_dict(),
    }


# ── LLM proposer (with validator as a tool) ──────────────────────────────────

SYSTEM_PROPOSE = """You are the Mutation Proposer. After a RED/YELLOW gate verdict has been diagnosed, you propose ONE controlled variant test.

# Doctrine (HARD)
- Exactly 1 of 4 mutation types: sample_window / cost_model / horizon / weighting
- ALL OTHER parameters unchanged from original spec
- Mutation must cite a SPECIFIC diagnostic claim (quote the diagnosis)
- NEVER flip sign / change signal construction (= new candidate, separate workflow)
- NEVER open a grid (single old_value → single new_value)
- Max 2 mutations per original candidate

# Process
1. Call validate_mutation_proposal with your proposal JSON
2. If validation fails, revise specifically per the rejection reasons and re-validate
3. Return the FINAL validated proposal as JSON

If no whitelisted mutation type applies to the diagnosed failure mode,
return:
  {"mutation_proposal": null, "reason": "..."}

This is a SUCCESS not a failure — better to propose nothing than to
force a noisy mutation that p-hacks.
"""


_LLM_TOOL_SCHEMAS = [
    {
        "name": "validate_mutation_proposal",
        "description": "Deterministic validator: checks proposal against whitelist + sign-flip rules + n_trials + mutation_seq cap.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name":     {"type": "string"},
                "mutation_type":      {"type": "string",
                                        "enum": sorted(WHITELIST_MUTATION_TYPES)},
                "old_value":          {"type": "string"},
                "new_value":          {"type": "string"},
                "justification":      {"type": "string"},
                "cited_diagnosis_ts": {"type": "string"},
                "n_trials_added":     {"type": "integer"},
                "mutation_seq":       {"type": "integer"},
            },
            "required": ["candidate_name", "mutation_type", "old_value",
                          "new_value", "justification", "cited_diagnosis_ts"],
        }
    }
]


def _read_anthropic_key() -> str | None:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _propose_llm(candidate_name: str) -> dict:
    key = _read_anthropic_key()
    if not key:
        result = _propose_deterministic(candidate_name)
        result["mode"] = "deterministic_fallback (no API key)"
        return result
    try:
        from anthropic import Anthropic
    except ImportError:
        result = _propose_deterministic(candidate_name)
        result["mode"] = "deterministic_fallback (anthropic not installed)"
        return result

    diag = _find_latest_diagnostic(candidate_name)
    gate = _find_latest_gate_run(candidate_name)
    if not diag or not gate:
        return _propose_deterministic(candidate_name)

    client = Anthropic(api_key=key, timeout=120.0)
    initial = (
        f"Candidate {candidate_name!r} returned verdict {gate.get('verdict')!r} "
        f"with diagnostic:\n\n{diag.get('refined_diagnosis')}\n\n"
        f"Diagnosis timestamp: {diag.get('timestamp')}\n"
        f"Existing mutation count for this candidate: "
        f"{_count_existing_mutations(candidate_name)}\n\n"
        f"Propose at most ONE controlled mutation following doctrine. "
        f"Call validate_mutation_proposal first to confirm validity, "
        f"revise if rejected, then return the final JSON."
    )
    messages = [{"role": "user", "content": initial}]
    cost = 0.0
    tool_calls = []
    final_text = ""

    for turn in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROPOSE,
            tools=_LLM_TOOL_SCHEMAS,
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
            v = validate_mutation_proposal(dict(tu.input))
            tool_calls.append({"name": tu.name, "input": dict(tu.input),
                                "ok": v.ok, "reasons": v.reasons})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(v.to_dict()),
            })
        messages.append({"role": "user", "content": tool_results})

    # Parse the LLM's final answer
    proposal_dict = _extract_proposal_json(final_text)
    if proposal_dict is None:
        return {
            "candidate":   candidate_name,
            "mode":        "llm",
            "proposal":    None,
            "reason":      "LLM produced no parseable proposal (may have correctly returned null)",
            "cost_usd":    round(cost, 4),
            "tool_calls":  tool_calls,
        }
    proposal_dict.setdefault("proposed_ts",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
    proposal_dict.setdefault("n_trials_added", 1)
    proposal_dict.setdefault("mutation_seq",
        _count_existing_mutations(candidate_name) + 1)
    validation = validate_mutation_proposal(proposal_dict)
    return {
        "candidate":   candidate_name,
        "mode":        "llm",
        "proposal":    proposal_dict,
        "validation":  validation.to_dict(),
        "cost_usd":    round(cost, 4),
        "tool_calls":  tool_calls,
    }


def _extract_proposal_json(text: str) -> dict | None:
    """Best-effort: pull the first JSON object out of LLM text."""
    if not text:
        return None
    # Find first { and matching }
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


# ── Public entry ────────────────────────────────────────────────────────────

def propose_mutation(candidate_name: str, *, use_llm: bool = True,
                      log: bool = True) -> dict:
    """Public entry — proposes (at most) one controlled mutation for a
    candidate that received a RED/YELLOW gate verdict and a diagnostic.

    Returns a dict with:
      candidate, mode, proposal | None, validation, ... (LLM fields if used)

    Side effect: if a valid proposal is generated AND log=True, appends
    to data/research/mutation_proposals.jsonl.

    NEVER auto-runs the gate. Output is a PROPOSAL; a human approves and
    runs the variant separately."""
    if use_llm:
        result = _propose_llm(candidate_name)
    else:
        result = _propose_deterministic(candidate_name)

    proposal_dict = result.get("proposal")
    if log and proposal_dict and result.get("validation", {}).get("ok"):
        try:
            # Reconstruct dataclass from dict for clean append
            allowed = {f.name for f in dataclasses.fields(MutationProposal)}
            clean = {k: v for k, v in proposal_dict.items() if k in allowed}
            proposal_obj = MutationProposal(**clean)
            validation_obj = ValidationResult(
                ok=result["validation"]["ok"],
                reasons=result["validation"]["reasons"],
            )
            _append_proposal(proposal_obj, validation_obj)
        except Exception as exc:
            logger.warning("failed to append mutation proposal: %s", exc)

    return result


def read_mutation_ledger(limit: int = 50) -> list[dict]:
    return list(reversed(_read_jsonl(MUTATION_LEDGER, limit=limit)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_name")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = propose_mutation(args.candidate_name,
                            use_llm=not args.no_llm,
                            log=not args.no_log)
    print(json.dumps(res, indent=2, default=str))
