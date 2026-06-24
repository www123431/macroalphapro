"""engine.agents.strengthener — Employee B (strengthener).

Phase 2.0 step 11 (2026-06-06): B takes A's synthesized hypotheses
(rows in hypotheses.jsonl with extraction_method=LLM_SYNTHESIS,
review_state=PROPOSED) and runs a second-pass review per candidate.

The output is a typed StrengthenerVerdict — one of:
  APPROVE_FOR_PIPELINE        → surfaces in /approvals as a candidate-
                                 pipeline approval row (step 13)
  REJECT                      → flips hypothesis review_state to REJECTED
  DOCTRINE_AMENDMENT_NEEDED   → surfaces in /approvals as a memory-
                                 amendment proposal (step 12)

B is a single Sonnet 4.6 call per hypothesis with strict JSON schema —
same Pattern 5-compliant shape as A's synthesis (single-agent + tool-
use schema + deterministic downstream gates). NOT multi-agent debate.
"""
from engine.agents.strengthener.review import (
    StrengthenerInput,
    StrengthenerVerdict,
    SleeveContextRef,
    DoctrineContextRef,
    VerdictType,
    run_strengthener_review,
)

__all__ = [
    "StrengthenerInput",
    "StrengthenerVerdict",
    "SleeveContextRef",
    "DoctrineContextRef",
    "VerdictType",
    "run_strengthener_review",
]
