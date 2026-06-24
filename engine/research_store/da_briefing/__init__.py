"""engine.research_store.da_briefing — Devil's Advocate verdict schema + contract.

The DA briefing is the CRITIQUE step in the PAPER → HYPOTHESIS → TEST →
VERDICT chain (locked 2026-06-04, see
[[feedback-paper-driven-research-chain-locked-2026-06-04]]).

Given a candidate that proposes to test a specific Hypothesis, the DA
critiques it using verbatim quotes from the source paper(s) plus
related papers (via P3 retrieval). The DA must cite chunk_ids that
resolve in papers_chroma — no chunk_id = invalid verdict.

This package defines:

  - DAClaim:              one critique-evidence atom (stance, chunk_id,
                          verbatim quote, argument)
  - DAVerdict:            the full DA output (refutes / supports /
                          conditional + overall_stance)
  - JSON schema:          for Anthropic SDK tool_use / structured output
  - cross-validation:     verify chunk_ids resolve + quotes are verbatim

This is the CONTRACT layer. Actual LLM invocation lives in
`engine.agents.devils_advocate.run_da_briefing` (built in T4).

Public API:

    from engine.research_store.da_briefing import (
        DAClaim, DAVerdict, DAStance, OverallStance,
        DA_VERDICT_JSON_SCHEMA,
        validate_verdict_cross_store,
        load_verdicts, save_verdict, VERDICTS_PATH,
    )
"""
from engine.research_store.da_briefing.schema import (
    DA_VERDICT_SCHEMA_VERSION,
    DAClaim,
    DAStance,
    DAVerdict,
    OverallStance,
)
from engine.research_store.da_briefing.structured_output import (
    DA_VERDICT_JSON_SCHEMA,
    DA_VERDICT_TOOL_DEFINITION,
)
from engine.research_store.da_briefing.cross_validate import (
    validate_verdict_cross_store,
)
from engine.research_store.da_briefing.store import (
    VERDICTS_PATH,
    load_verdicts,
    save_verdict,
)

__all__ = [
    "DAClaim", "DAVerdict", "DAStance", "OverallStance",
    "DA_VERDICT_SCHEMA_VERSION",
    "DA_VERDICT_JSON_SCHEMA", "DA_VERDICT_TOOL_DEFINITION",
    "validate_verdict_cross_store",
    "VERDICTS_PATH", "load_verdicts", "save_verdict",
]
