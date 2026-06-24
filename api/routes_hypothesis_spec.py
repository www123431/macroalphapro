"""api/routes_hypothesis_spec.py — read + extract hypothesis specs.

GET  /api/hypothesis_spec/{hypothesis_id}
     → latest structured spec for this hypothesis, or 404

POST /api/hypothesis_spec/extract {hypothesis_id, force?}
     → extracts (or re-extracts) the spec for a hypothesis via LLM,
       persists it, returns the new spec

Backs the spec preview card on /research/enhance PICK panel.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/hypothesis_spec", tags=["hypothesis_spec"])


class SignalLegOut(BaseModel):
    signal_type:      str
    sign:             str
    lookback_periods: list[int]
    quantile:         float
    role:             str
    note:             str


class UniverseOut(BaseModel):
    asset_class:        str
    subset:             str
    custom_tickers:     Optional[list[str]] = None
    min_history_months: int


class ConstructionOut(BaseModel):
    weighting:        str
    rebalance:        str
    skip_first_day:   bool
    holding_period_n: int


class RiskOut(BaseModel):
    vol_target_annual:   Optional[float] = None
    max_leverage:        Optional[float] = None
    turnover_cap_annual: Optional[float] = None
    max_position:        Optional[float] = None
    drawdown_stop:       Optional[float] = None


class OutcomeOut(BaseModel):
    predicted_direction: str
    predicted_sharpe_lo: Optional[float] = None
    predicted_sharpe_hi: Optional[float] = None
    rationale:           str


class ExtractionOut(BaseModel):
    method:        str
    confidence:    float
    extracted_ts:  str
    extractor_v:   str


class SpecOut(BaseModel):
    spec_id:              str
    spec_version:         int
    source_hypothesis_id: str
    version:              int
    spec_hash:            str
    family:               str
    claim_text:           str
    legs:                 list[SignalLegOut]
    universe:             UniverseOut
    construction:         ConstructionOut
    risk:                 RiskOut
    outcome:              OutcomeOut
    extraction:           ExtractionOut
    created_ts:           str
    git_sha:              str


def _spec_to_out(spec) -> SpecOut:
    from engine.hypothesis_spec.hash import spec_hash
    d = spec.to_dict()
    return SpecOut(
        spec_id              = spec.spec_id,
        spec_version         = spec.spec_version,
        source_hypothesis_id = spec.source_hypothesis_id,
        version              = spec.version,
        spec_hash            = spec_hash(spec),
        family               = spec.family.value,
        claim_text           = spec.claim_text,
        legs=[
            SignalLegOut(
                signal_type      = L.signal_type.value,
                sign             = L.sign.value,
                lookback_periods = list(L.lookback_periods),
                quantile         = L.quantile,
                role             = L.role,
                note             = L.note,
            ) for L in spec.legs
        ],
        universe = UniverseOut(
            asset_class        = spec.universe.asset_class.value,
            subset             = spec.universe.subset.value,
            custom_tickers     = list(spec.universe.custom_tickers)
                                  if spec.universe.custom_tickers else None,
            min_history_months = spec.universe.min_history_months,
        ),
        construction = ConstructionOut(
            weighting        = spec.construction.weighting.value,
            rebalance        = spec.construction.rebalance.value,
            skip_first_day   = spec.construction.skip_first_day,
            holding_period_n = spec.construction.holding_period_n,
        ),
        risk = RiskOut(
            vol_target_annual   = spec.risk.vol_target_annual,
            max_leverage        = spec.risk.max_leverage,
            turnover_cap_annual = spec.risk.turnover_cap_annual,
            max_position        = spec.risk.max_position,
            drawdown_stop       = spec.risk.drawdown_stop,
        ),
        outcome = OutcomeOut(
            predicted_direction = spec.outcome.predicted_direction.value,
            predicted_sharpe_lo = spec.outcome.predicted_sharpe_lo,
            predicted_sharpe_hi = spec.outcome.predicted_sharpe_hi,
            rationale           = spec.outcome.rationale,
        ),
        extraction = ExtractionOut(
            method       = spec.extraction.method,
            confidence   = spec.extraction.confidence,
            extracted_ts = spec.extraction.extracted_ts,
            extractor_v  = spec.extraction.extractor_v,
        ),
        created_ts = spec.created_ts,
        git_sha    = spec.git_sha,
    )


class ExtractRequest(BaseModel):
    hypothesis_id: str
    force:         bool = False


# IMPORTANT (2026-06-05 405 fix): /extract MUST be registered BEFORE the
# dynamic /{hypothesis_id} route. FastAPI/Starlette match routes by
# registration order on path; the catch-all /{hypothesis_id} would
# otherwise grab the literal /extract path with hypothesis_id="extract",
# then 405 the POST because that route is GET-only. Static routes
# always come first.
@router.post("/extract", response_model=SpecOut)
def extract(req: ExtractRequest) -> SpecOut:
    try:
        from engine.hypothesis_spec.store import latest_for, append
        from engine.hypothesis_spec.extractor import extract_spec
        from engine.research_store.hypothesis import load_hypotheses
        from engine.research_store.manifest import current_git_sha
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"import_failed:{exc}")

    existing = latest_for(req.hypothesis_id)
    if existing is not None and not req.force:
        return _spec_to_out(existing)

    # Find the source hypothesis
    hyps_raw = load_hypotheses()
    latest_by_id: dict = {}
    for h in hyps_raw:
        prior = latest_by_id.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            latest_by_id[h.hypothesis_id] = h
    h = latest_by_id.get(req.hypothesis_id)
    if h is None:
        raise HTTPException(status_code=404,
                            detail=f"hypothesis_not_found:{req.hypothesis_id}")

    spec = extract_spec(
        source_hypothesis_id = h.hypothesis_id,
        claim_text           = h.claim,
        mechanism_family     = h.mechanism_family.value,
        mechanism_subtype    = h.mechanism_subtype,
        git_sha              = current_git_sha() or "",
    )
    if spec is None:
        raise HTTPException(status_code=502,
                            detail="extractor_failed")
    append(spec)
    return _spec_to_out(spec)


@router.get("/{hypothesis_id}", response_model=SpecOut)
def get_spec(hypothesis_id: str) -> SpecOut:
    try:
        from engine.hypothesis_spec.store import latest_for
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"store_unavailable:{exc}")
    spec = latest_for(hypothesis_id)
    if spec is None:
        raise HTTPException(status_code=404,
                            detail=f"no_spec_for:{hypothesis_id}")
    return _spec_to_out(spec)
