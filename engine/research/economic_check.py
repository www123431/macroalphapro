"""engine/research/economic_check.py — Phase 5 F: economic_plausibility_check.

The ONE place LLM contributes irreplaceable judgment: ECONOMIC INTUITION.

H1-H8 hygiene tools are all structural / statistical checks. They cannot
answer: "Does this mechanism make ECONOMIC sense in current markets?
Does the proposed test design make sense given the mechanism's theory?"

That requires the kind of reasoning LLMs are genuinely good at — synthesis
of finance literature, market microstructure understanding, regime
awareness, and skeptical evaluation.

Doctrine:
- ADVISORY ONLY, NEVER GATING. The LLM never overrides deterministic
  verdicts (run_gate / protocol_executor). It produces a structured
  reasoning narrative + plausibility_score that goes into proposal_queue
  for human review.
- Cross-vendor optional (v1 Claude only; v2 add DeepSeek for adversarial).
- Strict output schema enforced by JSON parse + field check.
- Cost capped at ~$0.10 per check.

Flexibility ↔ Rigor balance:
- FLEX: closes the "borrow ideas not copy literally" gap (detail A in roadmap)
- RIGOR: advisory not gating; deterministic verdicts unchanged; full audit
   trail to data/research/economic_plausibility_log.jsonl
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
PLAUSIBILITY_LOG = REPO_ROOT / "data" / "research" / "economic_plausibility_log.jsonl"

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclasses.dataclass
class PlausibilityCheck:
    mechanism_id:          str
    plausibility_score:    float          # 0.0 - 1.0
    economic_intuition:    str            # the LLM's economic reasoning
    concerns:              list[str]      # potential issues the LLM identified
    regime_assessment:     str            # current-regime applicability
    cousin_with_deployed:  str            # economic overlap with our deployed book
    fidelity_recommendation: str          # literal | adapted | inspired
    mode:                  str            # llm | deterministic_fallback
    cost_usd:              float
    ts:                    str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


SYSTEM_PROMPT = """You are an experienced quant factor researcher (10+ years at AQR / Two Sigma / Renaissance). You evaluate a candidate factor mechanism for ECONOMIC PLAUSIBILITY before it goes through statistical gating.

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging.
BANNED vocabulary: maybe, perhaps, probably, possibly, likely, seems, appears, I think, I feel.

# Doctrine
You are ADVISORY only. You do NOT decide GREEN/YELLOW/RED. The statistical gate does that. Your job is to surface ECONOMIC concerns the gate cannot see.

# What to evaluate
1. Does the proposed mechanism have a coherent economic theory? Cite the channel (limits-to-arbitrage / risk premium / behavioral / microstructure / etc).
2. Is the theory consistent with current-regime market conditions? (post-2018 quant crowding / 2022 rate cycle / electronification of execution)
3. Are there published falsifications / decay evidence we should weight?
4. Does the mechanism economically overlap with anything we already deploy? (Not the cousin name check — the deep economic mechanism check.)
5. What fidelity_level should this be implemented at?
   - literal:   exact paper specification; all data + universe + parameters per paper
   - adapted:   same mechanism, different data proxy or universe (e.g. paper uses HFT data, we use end-of-day)
   - inspired:  economic intuition borrowed; implementation substantially novel

# Output schema (STRICT JSON only)
{
  "plausibility_score": <float 0.0-1.0>,
  "economic_intuition": "<2-4 sentence economic theory + channel>",
  "concerns": ["<concern 1>", "<concern 2>", ...],
  "regime_assessment": "<1-2 sentences on current-regime applicability>",
  "cousin_with_deployed": "<1 sentence on economic overlap with deployed book OR 'distinct'>",
  "fidelity_recommendation": "<literal|adapted|inspired>"
}

# Scoring guide
plausibility_score:
  0.8-1.0: strong economic theory + regime-applicable + post-pub evidence persists
  0.5-0.8: defensible theory + some concerns
  0.2-0.5: weak theory OR substantial regime concerns OR strong cousin overlap
  0.0-0.2: theory contradicts current markets OR likely already arbitraged out

NEVER score 1.0 (no mechanism is certain).
NEVER score 0.0 (always SOME plausibility if the paper got published Tier 1).
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


def _build_user_prompt(mechanism: dict, deployed_summaries: list[str]) -> str:
    name = mechanism.get("id", "unknown")
    family = mechanism.get("family", "unknown")
    parent = mechanism.get("parent_family", "unknown")
    canonical = mechanism.get("canonical_paper_id", "unknown")
    economics = mechanism.get("mechanism_economics", "")
    break_conditions = mechanism.get("mechanism_break_conditions") or []
    required_data = mechanism.get("required_data") or []
    sample = mechanism.get("typical_sample", "")

    deployed_str = "\n".join(f"  - {d}" for d in deployed_summaries) or "  (no deployed sleeves provided)"

    return f"""Evaluate this candidate mechanism for economic plausibility:

Mechanism ID:      {name}
Family / Parent:   {family} / {parent}
Canonical paper:   {canonical}
Typical sample:    {sample}
Required data:     {', '.join(required_data)}

Mechanism economics (from library YAML):
{economics}

Documented break conditions:
{chr(10).join(f'  - {c}' for c in break_conditions) or '  (none documented)'}

Our currently DEPLOYED sleeves:
{deployed_str}

Now produce the strict JSON evaluation per the system prompt."""


def _deployed_summaries_default() -> list[str]:
    """List currently deployed mechanism summaries from library."""
    LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
    if not LIBRARY_DIR.exists():
        return []
    import yaml
    out = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if entry.get("status_in_our_book") == "DEPLOYED":
            mid = entry.get("id")
            fam = entry.get("family")
            econ_first = (entry.get("mechanism_economics") or "").split(".")[0]
            out.append(f"{mid} (family={fam}): {econ_first}.")
    return out


def _deterministic_fallback(mechanism: dict) -> PlausibilityCheck:
    """Used when no API key or anthropic missing. Returns a non-informative
    but schema-conforming result. Strong concerns to ensure human review."""
    mid = mechanism.get("id", "unknown")
    return PlausibilityCheck(
        mechanism_id=mid,
        plausibility_score=0.50,
        economic_intuition=(
            "Deterministic fallback: no LLM judgment available. Refer to "
            f"canonical paper {mechanism.get('canonical_paper_id', 'unknown')} "
            "and apply senior-researcher judgment manually."
        ),
        concerns=["LLM unavailable — no automated economic plausibility check performed"],
        regime_assessment="unknown",
        cousin_with_deployed="unknown",
        fidelity_recommendation="literal",
        mode="deterministic_fallback",
        cost_usd=0.0,
        ts=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


def check_economic_plausibility(
    mechanism: dict,
    *,
    deployed_summaries: list[str] | None = None,
    use_llm: bool = True,
    log: bool = True,
) -> PlausibilityCheck:
    """Run the economic plausibility check on a library mechanism.

    Args:
      mechanism:           loaded library YAML dict
      deployed_summaries:  list of brief descriptions of deployed sleeves;
                            if None, computed from library DEPLOYED entries
      use_llm:             if False, returns deterministic fallback
      log:                 if True, append to economic_plausibility_log.jsonl

    Returns: PlausibilityCheck (always — never raises).
    """
    if not use_llm:
        result = _deterministic_fallback(mechanism)
    else:
        key = _read_anthropic_key()
        if not key:
            result = _deterministic_fallback(mechanism)
        else:
            try:
                from anthropic import Anthropic
            except ImportError:
                result = _deterministic_fallback(mechanism)
            else:
                result = _run_llm_check(Anthropic, key, mechanism,
                                          deployed_summaries)

    if log:
        PLAUSIBILITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PLAUSIBILITY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    return result


def _run_llm_check(Anthropic_cls, key: str,
                    mechanism: dict,
                    deployed_summaries: list[str] | None) -> PlausibilityCheck:
    mid = mechanism.get("id", "unknown")
    if deployed_summaries is None:
        deployed_summaries = _deployed_summaries_default()

    client = Anthropic_cls(api_key=key, timeout=60.0)
    user_prompt = _build_user_prompt(mechanism, deployed_summaries)
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        logger.warning("economic_plausibility LLM call failed: %s", exc)
        return _deterministic_fallback(mechanism)

    usage = response.usage
    cost = (usage.input_tokens * 3.0 / 1_000_000
              + usage.output_tokens * 15.0 / 1_000_000)
    text_parts = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_parts)

    parsed = _parse_json(raw_text)
    if not parsed:
        logger.warning("economic_plausibility could not parse JSON: %s", raw_text[:200])
        fb = _deterministic_fallback(mechanism)
        fb = dataclasses.replace(fb, cost_usd=round(cost, 4))
        return fb

    return PlausibilityCheck(
        mechanism_id=        mid,
        plausibility_score=  float(parsed.get("plausibility_score", 0.5)),
        economic_intuition=  str(parsed.get("economic_intuition", "")),
        concerns=            list(parsed.get("concerns") or []),
        regime_assessment=   str(parsed.get("regime_assessment", "")),
        cousin_with_deployed= str(parsed.get("cousin_with_deployed", "")),
        fidelity_recommendation= str(parsed.get("fidelity_recommendation", "literal")),
        mode=                "llm",
        cost_usd=            round(cost, 4),
        ts=                  datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


def _parse_json(text: str) -> dict | None:
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


def read_plausibility_log(limit: int = 50) -> list[dict]:
    if not PLAUSIBILITY_LOG.exists():
        return []
    rows = [json.loads(l) for l in PLAUSIBILITY_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]
