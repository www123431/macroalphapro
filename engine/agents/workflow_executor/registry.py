"""workflow_executor.registry — explicit registry of workflows.

Rule 5: workflow registry as CODE not config. No YAML, no DB —
register via decorator + import. Changing what's registered = a code
diff = git history + pre-commit hook + reviewer.

Rule 10: autorun whitelist defaults to ALL-FALSE. Even after Phase 2
ships, every workflow that wants to run autonomously needs explicit
enablement here. To enable one, edit AUTORUN_WHITELIST + commit.
"""
from __future__ import annotations

from typing import Callable, Optional
from engine.agents.workflow_executor.base import Workflow


# Registry: workflow_id -> Workflow class
_REGISTRY: dict[str, type[Workflow]] = {}


def register_workflow(cls: type[Workflow]) -> type[Workflow]:
    """Decorator. Adds the class to the registry under its workflow_id.

    Usage:
      @register_workflow
      class MyWorkflow(Workflow):
          workflow_id = "my_workflow"
          ...
    """
    if not getattr(cls, "workflow_id", ""):
        raise ValueError(f"workflow {cls.__name__} missing workflow_id")
    wid = cls.workflow_id
    if wid in _REGISTRY:
        raise ValueError(f"duplicate workflow_id: {wid}")
    _REGISTRY[wid] = cls
    return cls


def get_workflow(workflow_id: str) -> Optional[type[Workflow]]:
    return _REGISTRY.get(workflow_id)


def list_workflows() -> list[type[Workflow]]:
    return sorted(_REGISTRY.values(), key=lambda c: c.workflow_id)


# ── Autorun whitelist (rule 10) ─────────────────────────────────
#
# By default, NO workflow autoruns. Even after a workflow is
# registered, the runner enforces:
#   if workflow_id not in AUTORUN_WHITELIST: skip (or dry-run only)
#
# Enabling a workflow requires:
#   1. It has been observed in dry-run for ≥ 7 days
#   2. ≥ 90% of its dry-runs would have passed postcondition
#   3. A code review (this file is on a typed PR)
#
# Then add the workflow_id to AUTORUN_WHITELIST.

AUTORUN_WHITELIST: frozenset[str] = frozenset({
    # Empty by default. Phase 2.3 will add 3 LEVEL_0 workflows here
    # AFTER they pass dry-run observation.
})


def is_autorun_allowed(workflow_id: str) -> bool:
    return workflow_id in AUTORUN_WHITELIST
