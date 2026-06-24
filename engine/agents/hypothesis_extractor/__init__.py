"""engine.agents.hypothesis_extractor — paper→hypothesis extraction agent.

T3 of the paper-driven research chain. Walks each ingested paper's
chunks in papers_chroma and extracts structured Hypothesis records via
Claude API tool_use force-invoke.

Output schema is `engine.research_store.hypothesis.Hypothesis`.
Cross-validation (verbatim quote substring, INGESTED paper) is
enforced at `save_hypothesis()` time.

Public API:

    from engine.agents.hypothesis_extractor import (
        extract_hypotheses_from_chunks,
        EXTRACTOR_TOOL_DEFINITION,
        SYSTEM_PROMPT,
    )
"""
from engine.agents.hypothesis_extractor.extractor import (
    extract_hypotheses_from_chunks,
    ExtractorResult,
    HypothesisCandidate,
)
from engine.agents.hypothesis_extractor.tool import (
    EXTRACTOR_TOOL_DEFINITION,
)
from engine.agents.hypothesis_extractor.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

__all__ = [
    "extract_hypotheses_from_chunks", "ExtractorResult", "HypothesisCandidate",
    "EXTRACTOR_TOOL_DEFINITION",
    "SYSTEM_PROMPT", "build_user_prompt",
]
