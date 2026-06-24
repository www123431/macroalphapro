"""engine.agents.hypothesis_extractor.tool — submit_paper_hypotheses tool spec.

The LLM extractor is force-invoked on this tool. Output is an array of
hypothesis candidates, each with the testability fields the
Hypothesis schema requires.

`extract_hypotheses_from_chunks()` (in extractor.py) post-validates each
candidate's chunk_id resolution + verbatim substring against
papers_chroma. Candidates that fail post-validation are dropped from
the result and logged.
"""
from __future__ import annotations

from engine.research_store.red_lessons.mechanism_families import MechanismFamily


_MECHANISM_FAMILY_VALUES = [m.value for m in MechanismFamily]


_VERBATIM_QUOTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chunk_id", "quote_text", "section_ref", "relevance_note"],
    "properties": {
        "chunk_id": {
            "type": "string",
            "minLength": 5,
            "description":
                "papers_chroma chunk_id from the chunks shown to you. MUST "
                "be an exact match of one of the chunk_ids in the context. "
                "Fabricated chunk_ids will be rejected post-hoc."
        },
        "quote_text": {
            "type": "string",
            "minLength": 20,
            "description":
                "VERBATIM substring from the chunk text. Character-exact. "
                "No paraphrase, no ellipsis. ≥ 20 chars. Will be verified "
                "via `quote_text in chunk_text` post-hoc."
        },
        "section_ref": {
            "type": "string",
            "description":
                "Section / page / table ref if known (e.g. 'p.1467, §3.2', "
                "'Table 4'). Empty string OK."
        },
        "relevance_note": {
            "type": "string",
            "minLength": 10,
            "description":
                "1-line: why this quote substantiates the claim."
        },
    },
}


_HYPOTHESIS_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim", "mechanism_family", "mechanism_subtype",
        "predicted_direction", "predicted_magnitude",
        "required_data", "test_methodology",
        "source_chunk_ids", "verbatim_quotes",
    ],
    "properties": {
        "claim": {
            "type": "string",
            "minLength": 30,
            "maxLength": 800,
            "description":
                "1-3 sentence paraphrase of THE specific testable "
                "prediction this paper makes. Focus on what the paper "
                "claims will happen, not background. Skip claims that "
                "aren't directly testable."
        },
        "mechanism_family": {
            "type": "string",
            "enum": _MECHANISM_FAMILY_VALUES,
            "description":
                "Which mechanism family best fits this hypothesis. Pick "
                "from the controlled vocabulary. Use OTHER only when no "
                "category fits — explain in mechanism_subtype."
        },
        "mechanism_subtype": {
            "type": "string",
            "minLength": 3,
            "description":
                "Free-form refinement of the family. E.g. for CARRY: "
                "'cross_asset_carry' vs 'fx_carry'. For EARNINGS_DRIFT: "
                "'post_announcement_drift' vs 'guidance_drift'."
        },
        "predicted_direction": {
            "type": "string",
            "enum": ["positive", "negative", "zero"],
            "description":
                "Direction the paper predicts for the metric. 'positive' = "
                "alpha > 0 / Sharpe > 0. 'negative' = decay / refutation. "
                "'zero' = paper predicts null / no effect."
        },
        "predicted_magnitude": {
            "type": "string",
            "minLength": 10,
            "description":
                "Specific magnitude the paper predicts. E.g. 'Sharpe > 0.5 "
                "in US 1990-2020', 'alpha-t > 2.0', '26-58% decay post-"
                "publication'. Reference the paper's specific numbers if "
                "given."
        },
        "required_data": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 5},
            "description":
                "Data sources/universes the paper uses. E.g. "
                "['US cross-section monthly returns 1990-2020', "
                "'CRSP common stocks']. Each item is a specific "
                "requirement, not vague ('financial data' is too vague)."
        },
        "test_methodology": {
            "type": "string",
            "minLength": 20,
            "description":
                "Brief description of HOW the paper says to test this. "
                "E.g. 'Fama-MacBeth cross-sectional regression with "
                "industry controls; 12-1 momentum signal; monthly "
                "rebalance'."
        },
        "source_chunk_ids": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description":
                "papers_chroma chunk_ids that support this hypothesis. "
                "Must match exact chunk_ids from the input context. "
                "≥ 1 chunk_id required."
        },
        "verbatim_quotes": {
            "type": "array",
            "minItems": 2,
            "items": _VERBATIM_QUOTE_SCHEMA,
            "description":
                "≥ 2 verbatim quotes from the chunks supporting this "
                "claim. EACH quote must be a character-exact substring of "
                "the chunk_id it cites. Fabrication = rejection."
        },
    },
}


EXTRACTOR_TOOL_DEFINITION: dict = {
    "name": "submit_paper_hypotheses",
    "description":
        "Submit the testable hypotheses extracted from this paper batch. "
        "Each hypothesis is a SPECIFIC, TESTABLE prediction the paper "
        "makes, backed by ≥ 2 verbatim quotes from the provided chunks. "
        "If the batch contains no extractable testable claims (e.g. "
        "it's the References section or a methods footnote), submit "
        "an empty hypotheses array. Do NOT invent claims to fill the "
        "form.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses", "notes"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "items": _HYPOTHESIS_CANDIDATE_SCHEMA,
                "description":
                    "List of hypothesis candidates extracted from the "
                    "input chunks. Can be empty if the chunks don't "
                    "contain testable claims."
            },
            "notes": {
                "type": "string",
                "description":
                    "1-2 sentence summary of what kind of content was in "
                    "this chunk batch and why N hypotheses were "
                    "extracted (or 0)."
            },
        },
    },
}
