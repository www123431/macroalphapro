"""engine/research/override_workflow.py — structured workflow for
overriding a graveyard BLOCK signal.

DOCTRINE (2026-05-31 senior re-design after user pushback):
  Evidence-rigor requirements ("must cite post-2020 paper with net
  Sharpe > 0.5") are UNREALISTIC for real institutional alpha — most
  proprietary edges are never published. Instead enforce PROCESS-rigor:
  any override that follows the 5-step structured workflow is granted.
  Accountability comes from outcome tracking in the ledger, not from
  trying to gate-keep evidence quality up front.

5-step override workflow (ALL required):

  1. Structured cousin analysis — explicit_struct rebuttal of EACH
     graveyard cousin
  2. Pre-committed falsification — quantified F1..Fn with abandon-on-fail
  3. Time-box exploration budget — max person-hours + checkpoint
  4. Devil's Advocate review (deferred to Phase 2 — placeholder for now)
  5. Outcome ledger entry (auto-written by this module)

After 10+ overrides cumulated in ledger, posterior success rate
empirically updates the prior for future overrides. Self-correcting.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERRIDE_LEDGER = REPO_ROOT / "data" / "research" / "override_ledger.jsonl"


# ── Schemas ────────────────────────────────────────────────────────────


class CousinRebuttal(BaseModel):
    """Per-cousin rebuttal. Each graveyard cousin must have one."""

    model_config = ConfigDict(frozen=True)

    cousin_id: str = Field(..., min_length=1)
    cousin_verdict: str = Field(..., description="RED / AMBER / similar")
    same_or_different: Literal["same", "different", "partial"]
    structural_diff_dimensions: list[str] = Field(
        default_factory=list,
        description="quantified dimensions where candidate differs from cousin "
                    "(e.g. retail_share_pct, settlement_cycle, "
                    "analyst_coverage_median)",
    )
    why_might_not_apply: str = Field(..., min_length=20)


class FalsificationCriterion(BaseModel):
    """One pre-committed quantified abandon-on-fail criterion."""

    model_config = ConfigDict(frozen=True)

    label: str           # e.g. "F1"
    metric: str          # e.g. "per_event_t_stat"
    operator: Literal[">=", "<=", ">", "<", "=="]
    threshold: float
    rationale: str = ""


class ExplorationBudget(BaseModel):
    """Time-box for the exploratory override work."""

    model_config = ConfigDict(frozen=True)

    max_person_hours: float = Field(..., gt=0, le=40)
    checkpoint_at_hours: float = Field(..., gt=0)
    abandon_if_budget_exhausted: bool = True

    @field_validator("checkpoint_at_hours")
    @classmethod
    def _check_within_budget(cls, v, info):
        # checkpoint_at_hours must be < max_person_hours
        max_h = info.data.get("max_person_hours")
        if max_h is not None and v >= max_h:
            raise ValueError(
                f"checkpoint_at_hours ({v}) must be < max_person_hours ({max_h})"
            )
        return v


class OverrideRequest(BaseModel):
    """Complete override package — all 5 process steps."""

    model_config = ConfigDict()

    candidate_id: str = Field(..., min_length=1)
    candidate_family: str
    candidate_title: str

    graveyard_signal: dict
    """Raw graveyard match output (recommendation, signals, cousin_count, ...)"""

    cousins_analysis: list[CousinRebuttal]
    """Step 1 — per-cousin rebuttal. Length must match cousin_count_in_family."""

    falsification: list[FalsificationCriterion]
    """Step 2 — pre-committed quantified abandon criteria. Must have ≥3."""

    exploration_budget: ExplorationBudget
    """Step 3 — time-box."""

    da_review_status: Literal["pending", "completed_pass", "completed_attack",
                              "deferred"] = "deferred"
    """Step 4 — Devil's Advocate review. 'deferred' = Phase 2 build."""

    override_author: str = Field(..., min_length=1)
    override_reason_summary: str = Field(..., min_length=30)
    cited_evidence: list[str] = Field(
        default_factory=list,
        description="Free-form: papers / heuristics / proprietary observations. "
                    "Quality not gated; quantity sufficient for cousin-count.",
    )

    @field_validator("falsification")
    @classmethod
    def _min_three_criteria(cls, v):
        if len(v) < 3:
            raise ValueError(
                f"override requires ≥3 falsification criteria; got {len(v)}"
            )
        return v


class OverrideOutcome(BaseModel):
    """Recorded after exploration; written to outcome ledger."""

    model_config = ConfigDict()

    candidate_id: str
    overall_verdict: Literal["REINFORCED_GRAVEYARD", "OVERTURNED_GRAVEYARD",
                             "ABANDONED_BUDGET", "INCONCLUSIVE"]
    falsification_results: dict[str, bool]  # label → passed
    actual_hours_spent: float
    lessons_learned: str
    timestamp: _dt.datetime = Field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))


# ── Validation + ledger writing ────────────────────────────────────────


def validate_override_against_graveyard(
    request: OverrideRequest,
) -> tuple[bool, list[str]]:
    """Verify override addresses EACH cousin in graveyard_signal.

    Returns (is_valid, list_of_errors). All errors must be empty for
    override to be granted.
    """
    errors = []
    expected_cousin_count = request.graveyard_signal.get(
        "cousin_count_in_family", 0,
    )
    if len(request.cousins_analysis) < expected_cousin_count:
        errors.append(
            f"cousins_analysis has {len(request.cousins_analysis)} "
            f"entries but graveyard reported {expected_cousin_count} "
            f"family cousins; each must have explicit rebuttal"
        )

    if request.exploration_budget.max_person_hours > 16:
        errors.append(
            f"exploration_budget.max_person_hours={request.exploration_budget.max_person_hours} "
            f"exceeds 16h cap (anti-sunk-cost); break into smaller exploration"
        )

    if not any(c.same_or_different == "different"
               for c in request.cousins_analysis):
        errors.append(
            "no cousin rebuttal marked 'different' — if EVERY cousin is "
            "'same' or 'partial', override has no positive case"
        )

    return (len(errors) == 0, errors)


def grant_override(
    request: OverrideRequest,
    *,
    ledger_path: Path = OVERRIDE_LEDGER,
) -> bool:
    """Validate + persist override request. Returns True on grant."""
    is_valid, errors = validate_override_against_graveyard(request)
    if not is_valid:
        logger.error("override DENIED:")
        for e in errors:
            logger.error("  %s", e)
        return False

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": "override_granted",
        "candidate_id": request.candidate_id,
        "override_author": request.override_author,
        "graveyard_signal": request.graveyard_signal,
        "cousins_count": len(request.cousins_analysis),
        "falsification_count": len(request.falsification),
        "exploration_budget_hours": request.exploration_budget.max_person_hours,
        "da_review_status": request.da_review_status,
        "request": request.model_dump(),
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    logger.info(
        "override GRANTED for %s; logged to %s",
        request.candidate_id, ledger_path.name,
    )
    return True


def record_outcome(
    outcome: OverrideOutcome,
    *,
    ledger_path: Path = OVERRIDE_LEDGER,
) -> None:
    """Append exploration outcome to ledger. Builds the empirical
    posterior for future override gates."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": "override_outcome",
        **outcome.model_dump(),
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    logger.info(
        "outcome recorded for %s: %s",
        outcome.candidate_id, outcome.overall_verdict,
    )


def empirical_override_success_rate(
    *, ledger_path: Path = OVERRIDE_LEDGER,
) -> dict:
    """Compute the running empirical success rate from past overrides.

    Used by future override gates to update prior. After 10+ overrides,
    surfaces "your historical override success rate is X%; require Y%
    higher process bar" warnings.
    """
    if not ledger_path.exists():
        return {"n_overrides": 0, "n_outcomes": 0,
                "success_rate": None, "note": "no ledger yet"}
    outcomes = []
    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") == "override_outcome":
                outcomes.append(rec.get("overall_verdict"))
    if not outcomes:
        return {"n_overrides": 0, "n_outcomes": 0,
                "success_rate": None, "note": "no outcomes yet"}
    n_success = sum(1 for v in outcomes if v == "OVERTURNED_GRAVEYARD")
    rate = n_success / len(outcomes)
    return {
        "n_outcomes": len(outcomes),
        "n_overturned": n_success,
        "n_reinforced": sum(1 for v in outcomes if v == "REINFORCED_GRAVEYARD"),
        "n_abandoned": sum(1 for v in outcomes if v == "ABANDONED_BUDGET"),
        "n_inconclusive": sum(1 for v in outcomes if v == "INCONCLUSIVE"),
        "success_rate": rate,
    }
