"""engine.research_store.da_briefing.structured_output — JSON schema for tool_use.

When invoking the Anthropic SDK to run a DA briefing, the canonical way
to force structured output is via tool_use with a force_invoke tool that
matches the DAVerdict shape. The LLM has no path to return free-form
text — it must call the tool.

This module defines:

  DA_VERDICT_JSON_SCHEMA       — pure JSON schema (validates DAVerdict
                                 to_dict output)
  DA_VERDICT_TOOL_DEFINITION   — Anthropic SDK tool spec wrapping the
                                 schema; pass directly to messages.create
                                 tools= param

The tool spec is what the agents.devils_advocate (T4) module uses.
"""
from __future__ import annotations


# ─────────────────────── DAClaim JSON schema ──────────────────────────


_DA_CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stance", "chunk_id", "paper_id", "quote_text",
                 "section_ref", "argument"],
    "properties": {
        "stance": {
            "type": "string",
            "enum": ["refutes", "supports", "conditional", "insufficient"],
            "description":
                "Position of this evidence relative to the candidate "
                "hypothesis. 'refutes' = paper contradicts; 'supports' = "
                "corroborates; 'conditional' = applies only under stated "
                "conditions; 'insufficient' = relevant but inconclusive."
        },
        "chunk_id": {
            "type": "string",
            "minLength": 5,
            "description":
                "papers_chroma chunk_id (typically 'doi/xxxx::pNNNN'). MUST "
                "match a chunk you were shown in the retrieved context. "
                "Do NOT fabricate chunk_ids — verification will reject the "
                "verdict if any chunk_id does not resolve."
        },
        "paper_id": {
            "type": "string",
            "minLength": 5,
            "description":
                "papers_registry paper_id (UUID4) the chunk_id belongs to. "
                "Available in the retrieved context."
        },
        "quote_text": {
            "type": "string",
            "minLength": 20,
            "description":
                "VERBATIM substring from the chunk. Will be checked: must "
                "appear character-exact in the chunk's full text. Quote "
                "≥ 20 chars, ≤ 600 chars. NO paraphrase, NO ellipsis "
                "insertion."
        },
        "section_ref": {
            "type": "string",
            "description":
                "Where in the paper this quote is from, if known "
                "(e.g. 'p.1467, §3.2', 'Table 4', 'Abstract'). Empty "
                "string OK if unknown."
        },
        "argument": {
            "type": "string",
            "minLength": 20,
            "description":
                "1-2 sentence explanation of why this quote supports/"
                "refutes/conditions the candidate. Concrete, no hedging."
        },
    },
}


# ─────────────────────── DAVerdict JSON schema ────────────────────────


DA_VERDICT_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_name", "target_hypothesis_id",
                 "refutes", "supports", "conditional",
                 "overall_stance", "overall_rationale",
                 "n_chunks_retrieved", "papers_consulted"],
    "properties": {
        "candidate_name": {
            "type": "string",
            "minLength": 1,
            "description":
                "The candidate being critiqued (echo the input candidate_name)."
        },
        "target_hypothesis_id": {
            "type": "string",
            "minLength": 1,
            "description":
                "The Hypothesis.hypothesis_id this candidate proposes to "
                "test (echo the input)."
        },
        "refutes": {
            "type": "array",
            "items": _DA_CLAIM_SCHEMA,
            "description":
                "Claims with evidence that CONTRADICTS the candidate. If "
                "overall_stance=reject, this MUST have ≥ 1 entry."
        },
        "supports": {
            "type": "array",
            "items": _DA_CLAIM_SCHEMA,
            "description":
                "Claims with evidence that CORROBORATES the candidate. "
                "Devil's Advocate should still be skeptical — list these "
                "only when the paper text genuinely supports the candidate."
        },
        "conditional": {
            "type": "array",
            "items": _DA_CLAIM_SCHEMA,
            "description":
                "Claims that apply only under stated conditions (e.g. paper "
                "specifies US-only, but candidate is global). Include the "
                "condition in `argument`."
        },
        "overall_stance": {
            "type": "string",
            "enum": ["reject", "proceed_with_caveats", "needs_more_data",
                     "insufficient_evidence"],
            "description":
                "Bottom-line recommendation. 'reject' = ≥1 refute and "
                "evidence is strong. 'proceed_with_caveats' = mixed but "
                "testable. 'needs_more_data' = papers can't determine "
                "without additional data we don't have. "
                "'insufficient_evidence' = retrieved chunks don't speak "
                "to this candidate (use this ONLY when refutes + supports "
                "+ conditional are ALL empty)."
        },
        "overall_rationale": {
            "type": "string",
            "minLength": 50,
            "description":
                "Synthesis statement ≥ 50 chars. Why is this your overall "
                "stance? Refer to the strongest claims listed above."
        },
        "n_chunks_retrieved": {
            "type": "integer",
            "minimum": 0,
            "description":
                "How many chunks you were given as input context."
        },
        "papers_consulted": {
            "type": "array",
            "items": {"type": "string"},
            "description":
                "papers_registry paper_ids represented in the chunks you "
                "actually cited. Subset of the input paper_ids."
        },
    },
}


# ─────────────────────── Anthropic SDK tool definition ────────────────


# When passed to Anthropic SDK messages.create(tools=[...]), this forces
# the model to output a DAVerdict via tool_use. Combine with
# tool_choice={"type": "tool", "name": "submit_da_verdict"} to force-invoke.
DA_VERDICT_TOOL_DEFINITION: dict = {
    "name": "submit_da_verdict",
    "description":
        "Submit a Devil's Advocate verdict on a research candidate. The "
        "verdict must cite verbatim quotes from the retrieved paper "
        "chunks. Do NOT fabricate chunk_ids or quote text — they will be "
        "verified character-exact against papers_chroma. If the retrieved "
        "chunks don't speak to the candidate, set overall_stance="
        "'insufficient_evidence' with empty claim arrays.",
    "input_schema": DA_VERDICT_JSON_SCHEMA,
}
