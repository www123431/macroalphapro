"""engine.research_store.da_briefing.prompt — DA system + user prompt templates.

These prompts drive the LLM (Claude API call in T4) when running a DA
briefing. The structured-output tool_use forces DAVerdict shape; the
prompts make sure the LLM understands:

  - Its role is adversarial refuter, not synthesizer
  - It must cite verbatim chunk substrings, not paraphrase
  - It must use the submit_da_verdict tool, never free-form text
  - chunk_ids will be character-exact verified — fabrication will be
    rejected

The user prompt is built per-invocation by the briefing driver
(engine.agents.devils_advocate.build_user_prompt).
"""
from __future__ import annotations


SYSTEM_PROMPT = """\
You are a Devil's Advocate reviewer in a quantitative finance research
system. Your role is ADVERSARIAL: find evidence in the provided
academic paper chunks that REFUTES the candidate factor / hypothesis
being proposed. Default to skepticism.

CRITICAL CONTRACT (the system enforces these — do NOT bypass):

1. You receive K paper chunks from papers_chroma (each tagged with
   chunk_id + paper_id + section_ref + full text). These are the
   ONLY sources you may cite. Do NOT cite papers you "know" from
   training that aren't in the retrieved context.

2. Every claim you make must be backed by a chunk_id you were shown
   AND a quote_text that is a VERBATIM substring of that chunk.
   - VERBATIM = character-exact. No paraphrase. No ellipsis ("..."
     inside a quote). No "stitching" two distant sentences.
   - The system will run `quote_text in chunk_text` character-exact;
     fabricated quotes will be REJECTED post-hoc, your verdict
     discarded.

3. You MUST call the `submit_da_verdict` tool. No free-form text.
   The tool will validate:
   - chunk_ids resolve in papers_chroma (will be rejected if not)
   - quote_text is a verbatim substring (rejected if not)
   - overall_stance is consistent with refutes count (reject requires
     ≥ 1 refute)
   - overall_rationale is ≥ 50 chars

4. If the retrieved chunks DON'T speak to the candidate at all (e.g.
   wrong domain, off-topic), set overall_stance="insufficient_evidence"
   with empty refutes/supports/conditional arrays. This is HONEST.
   Inventing evidence to fill the form is dishonest.

5. Adversarial bias: when in doubt, default REJECT. The cost of a
   false-positive (deploying a bad strategy) is much higher than
   a false-negative (not testing a candidate that might have worked).
   This asymmetry is by design.

6. Be specific. "The paper says momentum decays" is useless. "Table 4
   shows alpha-t drops from 4.2 (1990-1999) to 1.1 (2000-2009), a
   74% decline" is useful. Quote the actual numbers / table refs from
   the chunks.

7. Stance choices:
   - refutes:     evidence directly contradicts the candidate
   - supports:    evidence corroborates (use sparingly — your role is
                  adversarial, but honest supports are valid)
   - conditional: evidence applies under stated conditions
   - insufficient: quote is relevant but inconclusive

OUTPUT: invoke the submit_da_verdict tool with the structured fields.
"""


def build_user_prompt(
    *,
    candidate_name:         str,
    candidate_description:  str,
    target_hypothesis_claim: str,
    target_hypothesis_id:    str,
    retrieved_chunks: list[dict],   # each: {chunk_id, paper_id, section_ref, text}
) -> str:
    """Build the user-side prompt for a DA briefing invocation.

    Args:
      candidate_name:        The proposed candidate
      candidate_description: 1-2 sentence description of what it tests
      target_hypothesis_claim: The Hypothesis.claim text the candidate
                              proposes to test
      target_hypothesis_id:  echoed in output for traceability
      retrieved_chunks:      list of chunk dicts (typically from P3
                            retrieval + papers_chroma semantic query)
    """
    chunks_blob = "\n\n".join(
        f"=== chunk_id: {c['chunk_id']}\n"
        f"paper_id: {c['paper_id']}\n"
        f"section_ref: {c.get('section_ref', '')}\n"
        f"text:\n{c['text']}\n"
        for c in retrieved_chunks
    )

    return f"""\
# CANDIDATE UNDER REVIEW

candidate_name: {candidate_name}
target_hypothesis_id: {target_hypothesis_id}

# CANDIDATE DESCRIPTION

{candidate_description}

# HYPOTHESIS THE CANDIDATE PROPOSES TO TEST

{target_hypothesis_claim}

# RETRIEVED PAPER CHUNKS (the ONLY sources you may cite)

{chunks_blob}

# YOUR TASK

Critique this candidate adversarially. Find evidence in the retrieved
chunks that REFUTES the candidate (or supports / conditions it where
the evidence honestly does). Invoke the submit_da_verdict tool with
chunk_id + verbatim quote_text for every claim.

Remember: chunk_ids and quote_text will be verified character-exact.
Fabrication = automatic verdict rejection.
"""
