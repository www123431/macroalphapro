"""
engine/agents/dq_inspector/agent.py — Phase 1 scaffold + entry-point.

Per spec id=70 (hash 31b5ad97). Scaffolds the agent shell + result
dataclass; Phase 2 gates.py + Phase 4 source_inspectors.py fill in
detection logic.

Three entry points (one per daily-cycle hook):
  - run_dq_check(today, phase='pre_batch')   — cheap freshness checks
                                                 BEFORE feed refresh
  - run_dq_check(today, phase='post_feed')   — coverage + anomaly
                                                 AFTER feed refresh
  - run_dq_check(today, phase='post_batch')  — row-count regression
                                                 AFTER paper_trade persist

Phase 1 returns zero-breach result so the daily script can be wired
now while Phase 2+ lands.

DOCTRINE invariants (per [[feedback-spec-lock-is-decision-contract]]):
  - Halt decision lives in gates.py (pure deterministic)
  - LLM narration runs AFTER decision (cannot flip halt flag)
  - Thresholds come from thresholds.py frozen dataclass
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import Literal

logger = logging.getLogger(__name__)


PhaseLiteral = Literal["pre_batch", "post_feed", "post_batch"]


@dataclasses.dataclass(frozen=True)
class DQInspectorRunResult:
    """One Data Quality Inspector check summary.

    Mirrors RiskManagerRunResult shape for downstream consumer parity
    (Risk Console UI / Watchdog read both via the same dashboard panel
    code).
    """
    started_at_iso:     str
    finished_at_iso:    str
    today_iso:          str
    phase:              str                     # "pre_batch" / "post_feed" / "post_batch"
    dry_run:            bool
    n_modes_evaluated:  int
    breaches:           tuple                    # tuple[Breach, ...] when Phase 2 lands
    halt:               bool
    severity:           str                     # "NONE" / "LIGHT" / "MEDIUM" / "SEVERE"
    narratives:         tuple[str, ...]
    llm_cost_usd:       float
    audit_alert_ids:    tuple[str, ...]


def run_dq_check(
    today:    datetime.date,
    phase:    PhaseLiteral = "pre_batch",
    dry_run:  bool         = False,
) -> DQInspectorRunResult:
    """Entry point — Phase 1 stub returns zero-breach result.

    Phase 2 (gates.py) fills in the 10 mode detectors per the three
    phase slots:
      pre_batch  → modes 1 / 2 / 3 / 4 (freshness)
      post_feed  → modes 5 / 6 / 7 / 9 (coverage + anomaly)
      post_batch → modes 8 / 10a / 10b (row-count regression)

    Args:
      today:    run date
      phase:    which hook is calling (determines mode subset)
      dry_run:  if True, do not persist alerts to DB

    Returns DQInspectorRunResult — Phase 1 always halt=False / NONE.
    """
    if phase not in ("pre_batch", "post_feed", "post_batch"):
        raise ValueError(
            f"phase must be 'pre_batch'/'post_feed'/'post_batch', got {phase!r}"
        )
    started = datetime.datetime.utcnow()
    logger.info(
        "dq_inspector: phase=%s today=%s dry_run=%s (Phase 1 stub)",
        phase, today, dry_run,
    )
    finished = datetime.datetime.utcnow()
    return DQInspectorRunResult(
        started_at_iso    = started.isoformat(),
        finished_at_iso   = finished.isoformat(),
        today_iso         = today.isoformat(),
        phase             = phase,
        dry_run           = dry_run,
        n_modes_evaluated = 0,
        breaches          = (),
        halt              = False,
        severity          = "NONE",
        narratives        = (),
        llm_cost_usd      = 0.0,
        audit_alert_ids   = (),
    )


class DQInspectorAgent:
    """Stateful wrapper holding spec lineage. Mirrors RiskManagerAgent."""

    def __init__(self) -> None:
        self.spec_id         = 70
        self.spec_hash_short = "31b5ad97"

    def check(
        self,
        today:    datetime.date,
        phase:    PhaseLiteral = "pre_batch",
        dry_run:  bool         = False,
    ) -> DQInspectorRunResult:
        return run_dq_check(today=today, phase=phase, dry_run=dry_run)
