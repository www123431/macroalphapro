"""api/routes_composer.py — Composer endpoints.

The "build series from spec" surface for the UI. Wraps engine.composer
with web-friendly response shapes.

  POST /api/composer/build  {hypothesis_id, force?}
       Look up the latest spec for hypothesis_id, compose if not
       cached, return {spec_hash, path, n_obs, components_used, ...}

  GET  /api/composer/coverage
       Component coverage summary (counts per role + keys per role).
       Used by SpecPreviewCard to show "5/5 roles covered" badge.

  GET  /api/composer/coverage/spec/{hypothesis_id}
       Per-spec coverage: is_spec_covered + gaps for the latest
       extracted spec of this hypothesis.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/composer", tags=["composer"])


class BuildRequest(BaseModel):
    hypothesis_id: str
    force:         bool = False


class BuildResult(BaseModel):
    ok:               bool
    spec_hash:        str
    path:             Optional[str] = None
    n_obs:            int = 0
    elapsed_s:        float = 0.0
    from_cache:       bool = False
    components_used:  list[dict] = []
    data_vintages:    dict = {}
    error:            Optional[str] = None
    coverage_gaps:    Optional[list[dict]] = None


class CoverageSummary(BaseModel):
    counts:        dict[str, int]
    keys_by_role:  dict[str, list[str]]
    total:         int


class CoverageGap(BaseModel):
    role:         str
    expected_key: str
    reason:       str


class SpecCoverage(BaseModel):
    hypothesis_id: str
    spec_hash:     Optional[str] = None
    covered:       bool
    gaps:          list[CoverageGap]


@router.post("/build", response_model=BuildResult)
def build(req: BuildRequest) -> BuildResult:
    try:
        from engine.composer import compose, ComponentNotFound, is_spec_covered
        from engine.hypothesis_spec.store import latest_for
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"composer_unavailable:{exc}")

    spec = latest_for(req.hypothesis_id)
    if spec is None:
        raise HTTPException(status_code=404,
                            detail=f"no_spec:{req.hypothesis_id}")

    # Pre-check coverage so the response carries gaps even on error path
    covered, gaps = is_spec_covered(spec)
    gap_dicts = [{"role": g.role.value, "expected_key": g.expected_key,
                  "reason": g.reason} for g in gaps]
    if not covered:
        return BuildResult(
            ok=False,
            spec_hash="",
            error=f"coverage_gaps:{[g['expected_key'] for g in gap_dicts]}",
            coverage_gaps=gap_dicts,
        )

    try:
        result = compose(spec, force=req.force)
        return BuildResult(**{
            k: v for k, v in result.items() if k in BuildResult.model_fields
        })
    except ComponentNotFound as exc:
        return BuildResult(
            ok=False,
            spec_hash="",
            error=str(exc),
            coverage_gaps=gap_dicts,
        )
    except Exception as exc:
        logger.exception("composer.compose failed")
        return BuildResult(
            ok=False,
            spec_hash="",
            error=str(exc)[:300],
        )


@router.get("/coverage", response_model=CoverageSummary)
def get_coverage() -> CoverageSummary:
    from engine.composer import coverage_summary
    return CoverageSummary(**coverage_summary())


@router.get("/coverage/spec/{hypothesis_id}", response_model=SpecCoverage)
def get_spec_coverage(hypothesis_id: str) -> SpecCoverage:
    try:
        from engine.composer import is_spec_covered
        from engine.hypothesis_spec.store import latest_for
        from engine.hypothesis_spec.hash import spec_hash
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"composer_unavailable:{exc}")
    spec = latest_for(hypothesis_id)
    if spec is None:
        return SpecCoverage(hypothesis_id=hypothesis_id, covered=False,
                            gaps=[CoverageGap(role="ANY", expected_key="(no spec)",
                                              reason="extract spec first")])
    covered, gaps = is_spec_covered(spec)
    return SpecCoverage(
        hypothesis_id = hypothesis_id,
        spec_hash     = spec_hash(spec),
        covered       = covered,
        gaps          = [CoverageGap(role=g.role.value,
                                      expected_key=g.expected_key,
                                      reason=g.reason) for g in gaps],
    )
