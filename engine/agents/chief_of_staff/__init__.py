"""engine.agents.chief_of_staff — Phase 2.0 step 14 weekly orchestrator.

NOT an LLM-based agent. The chief_of_staff is a deterministic Python
sequencer that runs D → A → B in order, captures each substep's
structured result, emits one rolled-up chief_of_staff_session_run
event with parent_event_ids pointing at the substep emits.

Pattern 5 safety: there is NO multi-agent debate here. Each substep
already does its own typed single-LLM-call work in its own module
(book_monitor is pure rules / synthesis is Sonnet + JSON schema /
strengthener is Sonnet + JSON schema). The orchestrator only
SEQUENCES + DEDUPS + EMITS — nothing model-driven about its choices.

Step 14a (this commit): pure-Python sequencer + audit event.
Step 14b (later):       LLM-generated 5-bullet weekly memo at session end.
Step 14c (later):       /lab/today 'Run weekly session' UI button.
"""
from engine.agents.chief_of_staff.runner import (
    SessionResult,
    run_weekly_session,
)

__all__ = ["SessionResult", "run_weekly_session"]
