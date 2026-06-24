"""
engine/agents/risk_manager/agent.py — Risk Manager Agent v1.0 entry point.

Phase 1 of spec id=69 (current hash lives in SpecRegistry — call
engine.agents.persona.tools.lookup_spec(69) for the canonical state).
Scaffolds the agent shell + result dataclass; the actual deterministic
gates land in Phase 2 (gates.py).

The agent exposes two surfaces:
  1. ``run_risk_manager_check(today, combined_book, signals, ...)``
     — pre-trade gate called by paper_trade_combined.run_paper_trade_day
     between step 2 (combine) and step 3 (persist). Returns
     RiskManagerRunResult with .halt flag honored by the orchestrator.
  2. ``RiskManagerAgent.advise_engineer_diff(...)`` (Phase 8)
     — Engineer-PR sign-off API returning a verdict + reasons.

DOCTRINE invariants enforced here (verified by Phase 9 tests):
  - DECISION layer (.halt / verdict) is computed by gates.py, NOT by LLM.
  - LLM narration runs AFTER decision, only adds prose; cannot flip flag.
  - All thresholds come from thresholds.py frozen dataclass; runtime
    overrides are rejected.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd
    from engine.portfolio.paper_trade_combined import StrategySignal

logger = logging.getLogger(__name__)


def _resolve_spec_hash_short(spec_id: int) -> str:
    """Read the current git-blob hash for ``spec_id`` from SpecRegistry.

    Resolved lazily so amendments propagate without a code edit. Returns
    "unknown" if the spec is not registered or the DB is unreachable.
    """
    try:
        from engine.preregistration import list_specs
        for row in list_specs():
            if int(row.get("id", -1)) == spec_id:
                h = row.get("current_hash") or ""
                return h[:8] if h else "unknown"
    except Exception as exc:
        logger.warning("agent.py: could not resolve spec %d hash: %s", spec_id, exc)
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass — populated by Phase 2 gates + Phase 7 narrator
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class RiskManagerRunResult:
    """One Risk Manager pre-trade or post-trade check summary.

    Phase 1 returns this with empty fields. Phase 2-8 fill in:
      breaches            : list of Breach dataclasses (gates.py)
      halt                : True if any HARD HALT mode breached
      severity            : "NONE" / "LIGHT" / "MEDIUM" / "SEVERE"
                            (absorbed from engine.circuit_breaker scheme)
      narratives          : list[str] one paragraph per breach (Phase 7)
      llm_cost_usd        : LLM cost spent on narration this run
      audit_alert_ids     : RiskManagerAlert table primary keys (Phase 4)
    """
    started_at_iso:     str
    finished_at_iso:    str
    today_iso:          str
    phase:              str                          # "pre_trade" / "post_trade"
    dry_run:            bool
    n_modes_evaluated:  int
    breaches:           tuple                         # tuple[Breach, ...] when Phase 2 lands
    halt:               bool
    severity:           str
    narratives:         tuple[str, ...]               # one per breach
    llm_cost_usd:       float
    audit_alert_ids:    tuple[int, ...]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 stub — populated in Phase 2 (gates.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_risk_manager_check(
    today:           datetime.date,
    combined_book:   "pd.Series",
    signals:         list["StrategySignal"],
    ticker_to_sleeve: dict[str, str],
    phase:           str       = "pre_trade",
    dry_run:         bool      = False,
) -> RiskManagerRunResult:
    """Entry point called by paper_trade_combined.run_paper_trade_day.

    Phase 1 scaffold: returns a zero-breach result so the orchestrator
    can be wired now and unit-tested while Phase 2 gates are implemented
    behind the same API.

    Args:
      today:            run date (orchestrator step 0)
      combined_book:    pd.Series ticker → book weight (post-leverage)
      signals:          list[StrategySignal] (orchestrator step 1 output)
      ticker_to_sleeve: ticker → originating sleeve_id mapping
      phase:            'pre_trade' or 'post_trade'
      dry_run:          if True, do not persist alerts to DB

    Returns:
      RiskManagerRunResult — Phase 1 always returns halt=False / severity=NONE
    """
    started = datetime.datetime.utcnow()
    logger.info(
        "risk_manager: phase=%s today=%s n_tickers=%d n_strategies=%d (Phase 1 stub)",
        phase, today, len(combined_book), len(signals),
    )
    finished = datetime.datetime.utcnow()
    return RiskManagerRunResult(
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


# ─────────────────────────────────────────────────────────────────────────────
# Class wrapper — Phase 8 adds advise_engineer_diff method
# ─────────────────────────────────────────────────────────────────────────────
class RiskManagerAgent:
    """Stateful wrapper around the functional API. Holds the registry
    handle so Phase 8 advise_engineer_diff can inspect proposed META
    against current sleeve registrations.

    Phase 1: just a thin wrapper around run_risk_manager_check. Phase 8
    adds .advise_engineer_diff(diff_text, affected_strategies, ...).
    """

    def __init__(self) -> None:
        # Spec hash is resolved dynamically from SpecRegistry so amendments
        # are picked up without a code edit. Literal hash strings here would
        # create a fixed-point bug (touching the file changes its own hash).
        self.spec_id = 69
        self.spec_hash_short = _resolve_spec_hash_short(self.spec_id)

    def check(
        self,
        today:            datetime.date,
        combined_book:    "pd.Series",
        signals:          list["StrategySignal"],
        ticker_to_sleeve: dict[str, str],
        phase:            str  = "pre_trade",
        dry_run:          bool = False,
    ) -> RiskManagerRunResult:
        return run_risk_manager_check(
            today           = today,
            combined_book   = combined_book,
            signals         = signals,
            ticker_to_sleeve= ticker_to_sleeve,
            phase           = phase,
            dry_run         = dry_run,
        )

    def advise_engineer_diff(self, *args, **kwargs):  # noqa: ARG002
        """Phase 8 placeholder. Returns NotImplementedError so callers
        explicitly know this surface isn't live yet."""
        raise NotImplementedError(
            "advise_engineer_diff lands in Phase 8 of spec id=69 build sequence. "
            "Tracked in todo list as Phase 1-11 sequencing."
        )
