"""engine.agents.hypothesis_extractor.extractor — LLM-backed extraction.

Calls Anthropic SDK with tool_use force-invoke to extract
testable hypotheses from a batch of paper chunks.

Each candidate returned by the LLM is post-validated:
  - source_chunk_ids must all exist in the provided chunk batch
  - each verbatim_quote.quote_text must be a substring of the chunk_id
    it cites in the provided batch

Candidates failing post-validation are dropped from the result with a
log warning. The driver script can then re-attempt with adjusted
prompt or drop entirely.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Model selection — Sonnet 4.6 is the right cost/quality tier for batch
# extraction (Opus would burn budget; Haiku misses subtle claims).
EXTRACTOR_MODEL = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 4096


# ─────────────────────── auth ─────────────────────────────────────────


def _read_anthropic_key() -> str | None:
    """Resolve ANTHROPIC_API_KEY: env → .streamlit/secrets.toml → streamlit API."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    # Direct file read (works outside streamlit runtime).
    from pathlib import Path
    secrets_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / ".streamlit" / "secrets.toml"
    )
    if secrets_path.is_file():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib       # type: ignore
            with secrets_path.open("rb") as f:
                data = tomllib.load(f)
            v = data.get("ANTHROPIC_API_KEY")
            if v:
                return v
        except Exception:
            pass
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


# ─────────────────────── result dataclasses ───────────────────────────


@_dc.dataclass(frozen=True)
class HypothesisCandidate:
    """One hypothesis candidate extracted from the LLM. Post-validated.

    Maps cleanly to engine.research_store.hypothesis.Hypothesis fields.
    The driver builds the Hypothesis from this candidate + paper
    metadata.
    """
    claim:               str
    mechanism_family:    str   # MechanismFamily.value
    mechanism_subtype:   str
    predicted_direction: str   # "positive" | "negative" | "zero"
    predicted_magnitude: str
    required_data:       tuple[str, ...]
    test_methodology:    str
    source_chunk_ids:    tuple[str, ...]
    verbatim_quotes:     tuple[dict, ...]   # each: {chunk_id, quote_text, section_ref, relevance_note}


@_dc.dataclass(frozen=True)
class ExtractorResult:
    candidates:         tuple[HypothesisCandidate, ...]
    notes:              str
    n_dropped_post_validation: int
    drop_reasons:       tuple[str, ...]
    raw_response:       dict   # full tool input from LLM (for debugging)


# ─────────────────────── post-validation ──────────────────────────────


def _normalize_for_substring_match(s: str) -> str:
    """Collapse whitespace runs to single spaces; remove soft-hyphens;
    rejoin PDF line-break hyphenation ("consis- tent" → "consistent");
    normalize quote / dash glyphs; strip.

    LLMs normalize PDF-extracted whitespace AND undo line-break
    hyphenation when quoting. Strict `s in chunk` rejects valid quotes.
    This normalizer flattens both the LLM output and the chunk text
    to a common form before substring check.

    The persisted quote_text is still the LLM's version (not normalized);
    only the comparison itself uses the normalized form.
    """
    import re
    s = s.replace("­", "")     # soft hyphen U+00AD
    s = s.replace("‐", "-")    # hyphen U+2010
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    # PDF line-break hyphenation: word ends with "-" + whitespace + lowercase
    # continuation → rejoin. e.g. "consis- tent" → "consistent". Conservative:
    # only when continuation is lowercase (proper line-break behavior).
    s = re.sub(r"(\w)-\s+([a-z])", r"\1\2", s)
    # Collapse all remaining whitespace runs.
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _post_validate(
    candidates: list[dict],
    available_chunks: list[dict],
) -> tuple[list[HypothesisCandidate], list[str]]:
    """Filter LLM output candidates against the batch we actually showed.

    Drops candidates with:
      - source_chunk_id not in available batch
      - verbatim_quote.chunk_id not in available batch
      - verbatim_quote.quote_text not a substring of the cited chunk
        text (after whitespace normalization)
    """
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in available_chunks}
    chunk_norm_by_id = {cid: _normalize_for_substring_match(t)
                        for cid, t in chunk_text_by_id.items()}
    available_ids = set(chunk_text_by_id.keys())

    kept: list[HypothesisCandidate] = []
    drop_reasons: list[str] = []

    for i, cand in enumerate(candidates):
        bad_chunks = [cid for cid in (cand.get("source_chunk_ids") or [])
                      if cid not in available_ids]
        if bad_chunks:
            drop_reasons.append(
                f"cand[{i}]: source_chunk_ids {bad_chunks} not in batch — "
                f"hallucinated"
            )
            continue

        quotes_ok = True
        for j, q in enumerate(cand.get("verbatim_quotes") or []):
            qcid = q.get("chunk_id", "")
            qtext = q.get("quote_text", "")
            if qcid not in available_ids:
                drop_reasons.append(
                    f"cand[{i}].verbatim_quotes[{j}].chunk_id={qcid} not "
                    f"in batch — hallucinated"
                )
                quotes_ok = False
                break
            # Whitespace-normalized substring check (defends against PDF
            # whitespace LLMs naturally normalize, but still catches
            # paraphrase / fabrication).
            chunk_norm = chunk_norm_by_id[qcid]
            qtext_norm = _normalize_for_substring_match(qtext)
            if qtext_norm not in chunk_norm:
                drop_reasons.append(
                    f"cand[{i}].verbatim_quotes[{j}] for chunk {qcid} is "
                    f"NOT a verbatim substring — paraphrase suspected"
                )
                quotes_ok = False
                break
        if not quotes_ok:
            continue

        kept.append(HypothesisCandidate(
            claim               = cand["claim"],
            mechanism_family    = cand["mechanism_family"],
            mechanism_subtype   = cand["mechanism_subtype"],
            predicted_direction = cand["predicted_direction"],
            predicted_magnitude = cand["predicted_magnitude"],
            required_data       = tuple(cand.get("required_data") or ()),
            test_methodology    = cand["test_methodology"],
            source_chunk_ids    = tuple(cand.get("source_chunk_ids") or ()),
            verbatim_quotes     = tuple(cand.get("verbatim_quotes") or ()),
        ))

    return kept, drop_reasons


# ─────────────────────── live LLM call ────────────────────────────────


def extract_hypotheses_from_chunks(
    *,
    paper_metadata: dict,
    chunks:         list[dict],          # each: {chunk_id, section, text}
    model:          str = EXTRACTOR_MODEL,
    max_tokens:     int = MAX_OUTPUT_TOKENS,
    timeout_s:      int = 90,
    api_key:        str | None = None,
) -> ExtractorResult:
    """Call Anthropic SDK with tool_use force-invoke; return result.

    Raises:
      RuntimeError if no API key.
      anthropic.APIError on API failure (caller decides retry vs skip).
    """
    from anthropic import Anthropic
    from engine.agents.hypothesis_extractor.prompt import (
        SYSTEM_PROMPT, build_user_prompt,
    )
    from engine.agents.hypothesis_extractor.tool import (
        EXTRACTOR_TOOL_DEFINITION,
    )

    key = api_key or _read_anthropic_key()
    if not key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY in env or streamlit secrets; cannot "
            "run extractor"
        )

    client = Anthropic(api_key=key, timeout=timeout_s)
    user_prompt = build_user_prompt(
        paper_metadata=paper_metadata,
        chunks=chunks,
    )

    response = client.messages.create(
        model       = model,
        max_tokens  = max_tokens,
        system      = SYSTEM_PROMPT,
        tools       = [EXTRACTOR_TOOL_DEFINITION],
        tool_choice = {"type": "tool", "name": "submit_paper_hypotheses"},
        messages    = [{"role": "user", "content": user_prompt}],
    )

    # Find the tool_use block
    tool_use_block = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            tool_use_block = block
            break
    if tool_use_block is None:
        return ExtractorResult(
            candidates=(),
            notes="LLM did not call the tool (unexpected — tool_choice forces it)",
            n_dropped_post_validation=0,
            drop_reasons=(),
            raw_response={},
        )

    raw_input = dict(tool_use_block.input)
    candidates_raw = raw_input.get("hypotheses") or []
    notes = raw_input.get("notes", "")

    kept, drop_reasons = _post_validate(candidates_raw, chunks)
    if drop_reasons:
        logger.warning("hypothesis extraction post-validation dropped %d/%d: %s",
                       len(drop_reasons), len(candidates_raw), drop_reasons[:3])

    return ExtractorResult(
        candidates                = tuple(kept),
        notes                     = notes,
        n_dropped_post_validation = len(drop_reasons),
        drop_reasons              = tuple(drop_reasons),
        raw_response              = raw_input,
    )
