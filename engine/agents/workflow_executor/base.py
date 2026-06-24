"""workflow_executor.base — base class + types for codified workflows.

Every concrete workflow is a subclass with explicit attrs:

  workflow_id        unique slug ('decay_retest_sleeve')
  description        one-line human summary
  reversibility      ReversibilityLevel  (LEVEL_0 / LEVEL_1 / LEVEL_2 / LEVEL_3)
  blast_radius_max   {files_written:int, llm_tokens:int, wall_seconds:int}

and methods:

  idempotency_key(**inputs) -> str
      Stable string per logical run. If the runner sees the same key
      already-executed, it skips (rule 1).

  precondition(**inputs) -> tuple[bool, str]
      Run BEFORE the action. Returns (ok, reason). False = skip.

  run(**inputs) -> dict
      The actual work. Returns a dict that gets logged.

  postcondition(result) -> tuple[bool, str]
      Run AFTER the action. Returns (ok, reason). False = mark failed.

The runner's contract is to enforce ALL of the above. A workflow that
hides side effects, mutates non-LEVEL_0 state without declaring it, or
exceeds blast_radius_max breaks the contract and gets quarantined.

Reversibility levels (rule 3)
  LEVEL_0  append-only event/log         (reversing: ignore newest rows)
  LEVEL_1  cache file rewrite             (reversing: delete cache + regen)
  LEVEL_2  YAML/config / state mutation   (reversing: git revert)
  LEVEL_3  irreversible (orders, deletes, external API state changes)

Phase 2 autorun whitelist is restricted to LEVEL_0 only. LEVEL_1 needs
explicit per-workflow whitelist + manual user approval. LEVEL_2-3 are
NEVER autonomous — they must always go through a human approval flow.
"""
from __future__ import annotations

import dataclasses as _dc
import enum
from typing import Any, Optional


class ReversibilityLevel(str, enum.Enum):
    """How costly is undoing this workflow's effects.

    LEVEL_0: append-only — newest rows can be ignored / archived
    LEVEL_1: cache-rebuild — delete + regenerate
    LEVEL_2: config / state mutation — git revert needed
    LEVEL_3: irreversible (orders, deletes, external API state)
    """
    LEVEL_0 = "LEVEL_0"
    LEVEL_1 = "LEVEL_1"
    LEVEL_2 = "LEVEL_2"
    LEVEL_3 = "LEVEL_3"


@_dc.dataclass(frozen=True)
class WorkflowResult:
    """Structured output of one workflow run. Recorded to ledger so
    AgentHealth tile + future eval set can read."""
    workflow_id:        str
    idempotency_key:    str
    status:             str          # "ok" | "skipped" | "precondition_fail" | "postcondition_fail" | "error"
    reason:             str          # short explanation
    trigger:            str          # "cron" | "event:<event_type>" | "manual"
    started_ts:         str
    ended_ts:           str
    elapsed_s:          float
    inputs:             dict
    outputs:            dict
    decisions:          list[dict]   # if/branch decisions made
    dry_run:            bool
    reversibility:      str
    blast_radius_max:   dict
    blast_actual:       dict
    error:              Optional[str] = None


class Workflow:
    """Base class for a codified workflow. Subclasses fill in attrs +
    methods. The runner uses these to enforce the 10 institutional rules.
    """
    # Required class attrs — override in subclass
    workflow_id:      str = ""
    description:      str = ""
    reversibility:    ReversibilityLevel = ReversibilityLevel.LEVEL_0
    blast_radius_max: dict = _dc.field(
        default_factory=lambda: {"files_written": 5, "llm_tokens": 0, "wall_seconds": 60}
    )

    # ── Required overrides ─────────────────────────────────────

    def idempotency_key(self, **inputs) -> str:
        """Stable key per logical run. If runner sees the same key
        in the ledger already, skip — rule 1."""
        raise NotImplementedError

    def precondition(self, **inputs) -> tuple[bool, str]:
        """Returns (ok, reason). Run BEFORE side effects. Must check:
          - required data/files exist
          - dependencies healthy
          - inputs valid
        Returns (False, reason) → runner emits 'precondition_fail' + skip."""
        return (True, "no precondition check")

    def run(self, **inputs) -> dict:
        """Do the work. Returns a dict that goes into outputs + ledger.
        MUST NOT exceed self.blast_radius_max. MUST be idempotent."""
        raise NotImplementedError

    def postcondition(self, result: dict) -> tuple[bool, str]:
        """Returns (ok, reason). Run AFTER side effects. Must check:
          - outputs match expected schema
          - any side effects (files written, events emitted) within
            blast radius
        Returns (False, reason) → result marked 'postcondition_fail'."""
        return (True, "no postcondition check")

    # ── Optional overrides ─────────────────────────────────────

    def is_due(self, last_run_ts: Optional[str], inputs: dict) -> bool:
        """Cron-style: should this workflow run NOW? Override for
        cadence-aware logic (daily / weekly / on-event). Default: due
        if never run. Most cron-triggered workflows override this."""
        return last_run_ts is None
