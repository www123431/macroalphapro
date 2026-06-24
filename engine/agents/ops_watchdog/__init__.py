"""
engine/agents/ops_watchdog — Ops Watchdog Agent v1.0 (spec id=63 hash 9d050804).

Daily 06:10 SGT operations-layer agent. Detects 13 production error modes,
LLM-reasons over them via Tool 1 ReAct primitives, then hardcoded triage
classifies severity and dispatches notifications + auto-repair recipes.

ARCHITECTURE INVARIANTS (spec §6 forbidden modifications):
  - 0-LLM-in-alpha-decision-loop preserved (operations layer only)
  - MODE_SEVERITY_MAP hardcoded in triage.py (NOT LLM-decided)
  - AUTO_REPAIR_RECIPES hardcoded in auto_repair.py (NOT LLM-decided)
  - Production tables read-only from Watchdog perspective
  - LLM cost cap $0.20/run (8 ReAct steps × $0.02)

PHASE STATUS (2026-05-13):
  - Phase 1 ✅ — 11 detection rules in engine.auto_audit_rules.WATCHDOG_RULES
                  (mode 9 refactored per amendment 2 + unit-conversion follow-up)
                  Current spec hash: 645507ad (post-amendment 3 Phase 6 reconciliation)
  - Phase 2 ✅ — agent.py / tools.py / triage.py / prompt.py
  - Phase 3 ✅ — auto_repair.py (3 active recipes: modes 1/2/6; 3 deferred:
                  modes 4/10/12 per Option A root-cause-not-symptom-fix doctrine)
                  + Tier R guardrail rule_watchdog_auto_repair_no_raw_sql
  - Phase 4 ✅ — notifications.py (4 channels: dashboard / toast / email /
                  halt flag) + Tier R rule_watchdog_halt_flag_not_stuck
  - Phase 5 ✅ — pages/ops_watchdog.py (Streamlit widget, 4-state mock toggle)
                  + Tier R rule_watchdog_runs_daily + Windows Task Scheduler
                  "MacroAlphaPro_Watchdog" daily 06:10 SGT registered + dogfood
                  fire verified 2026-05-13 11:26 (LLM cost $0.0174, 7 ReAct
                  steps, 3 findings, auto-repair 1/2 succeeded)
  - Phase 6 pending — capability evidence + memory entry + final spec amendment
"""
from engine.agents.ops_watchdog.auto_repair import (
    AUTO_REPAIR_RECIPES_LOCKED,
    DEFERRED_MODES_LOCKED,
    MAX_RETRY_ATTEMPTS,
    RepairResult,
    execute_repair_for_finding,
    execute_repairs_for_findings,
)
from engine.agents.ops_watchdog.notifications import emit_notification
from engine.agents.ops_watchdog.triage import (
    MODE_SEVERITY_MAP_LOCKED,
    SEVERITY_LIGHT,
    SEVERITY_MEDIUM,
    SEVERITY_NONE,
    SEVERITY_SEVERE,
    RULE_TO_MODE_LOCKED,
    aggregate_severity,
    rule_name_to_mode,
)

__all__ = [
    "AUTO_REPAIR_RECIPES_LOCKED",
    "DEFERRED_MODES_LOCKED",
    "MAX_RETRY_ATTEMPTS",
    "MODE_SEVERITY_MAP_LOCKED",
    "RULE_TO_MODE_LOCKED",
    "RepairResult",
    "SEVERITY_LIGHT",
    "SEVERITY_MEDIUM",
    "SEVERITY_NONE",
    "SEVERITY_SEVERE",
    "aggregate_severity",
    "emit_notification",
    "execute_repair_for_finding",
    "execute_repairs_for_findings",
    "rule_name_to_mode",
]
