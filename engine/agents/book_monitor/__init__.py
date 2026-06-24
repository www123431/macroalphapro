"""engine.agents.book_monitor — Employee D (book monitor).

Phase 2.0 step 9 (2026-06-06): D watches the research event stream
and fires `doctrine_signal_detected` events when deterministic rules
see patterns worth doctrine attention.

D is RULES-BASED, NOT LLM-based. The reasoning behind each rule lives
in the rule's docstring + the spec memo
[[spec-research-session-orchestrator-2026-06-06]]; the rule itself is
a pure function over events.

Step 9b ships `family_red_cluster`; subsequent rules (sleeve decay,
gate rejection spike, etc.) follow the same pattern.
"""
from engine.agents.book_monitor.pattern_rules import (
    PatternHit,
    RULES,
    check_family_red_cluster,
)

__all__ = ["PatternHit", "RULES", "check_family_red_cluster"]
