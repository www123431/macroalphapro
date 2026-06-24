"""engine.agents.workflow_executor — autonomous executor for codified
research workflows.

Phase 2 of the agentic build (2026-06-04). User feedback:
  "我感觉不到他的自主性 ... 明明那么多流程都规范的设计好了
   他可以自己调用啊"

Codified workflows already exist:
  - approve forward vector + open session + run pipeline + emit verdict
  - on factor_verdict_filed: increment family n_trials counter
  - diff yesterday's direction proposer top-3, alert if changed
  - on every DEPLOYED sleeve: re-test trailing OOS Sharpe

But none of them were autonomous. This module orchestrates them as
Workflow classes following 10 institutional rules (see commit notes
for the senior 施工建议).

Public surface:
  - Workflow                 base class with required attrs/methods
  - registry / register_workflow
  - run_one(workflow_id, **inputs) -> WorkflowResult
  - run_all_due() -> list[WorkflowResult]   (entry point for cron)

Subpackages (added in Phase 2.3):
  - handlers/                concrete Workflow subclasses
"""
from engine.agents.workflow_executor.base import (
    Workflow,
    WorkflowResult,
    ReversibilityLevel,
)
from engine.agents.workflow_executor.registry import (
    register_workflow,
    get_workflow,
    list_workflows,
    is_autorun_allowed,
)
from engine.agents.workflow_executor.runner import (
    run_one,
    run_all_due,
    is_paused,
    set_paused,
)

# Import handlers package as a SIDE EFFECT to register all workflows.
# This is the one place where the registry gets populated.
from engine.agents.workflow_executor import handlers  # noqa: F401

__all__ = [
    "Workflow",
    "WorkflowResult",
    "ReversibilityLevel",
    "register_workflow",
    "get_workflow",
    "list_workflows",
    "is_autorun_allowed",
    "run_one",
    "run_all_due",
    "is_paused",
    "set_paused",
]
