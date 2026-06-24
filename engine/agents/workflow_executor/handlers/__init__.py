"""Concrete Workflow handlers.

Each module registers exactly one Workflow class via @register_workflow.
Importing this package triggers all registrations at once.

Discipline rules per handler module:
  - One Workflow class per file
  - reversibility level documented in class docstring
  - blast_radius_max realistic + tight (not "999 files")
  - precondition() never raises (all defensive)
  - run() idempotent + safe to retry
  - postcondition() validates structure of run output

Phase 2.3 lands 3 handlers, all LEVEL_0:
  - n_trials_family_counter
  - session_stale_audit
  - graveyard_collision_digest
"""

# Import side-effect: registers each handler
from engine.agents.workflow_executor.handlers import n_trials_family_counter  # noqa: F401
from engine.agents.workflow_executor.handlers import session_stale_audit       # noqa: F401
from engine.agents.workflow_executor.handlers import graveyard_collision_digest  # noqa: F401
