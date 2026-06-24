"""api/routes_series_factory.py — build returns series on demand.

POST /api/series_factory/build {family, hypothesis_id, params?}
  -> {ok, path, n_obs, family, hypothesis_id, from_cache, elapsed_s, error?}

GET  /api/series_factory/families
  -> {families: [...]}

The frontend RUN panel calls POST when:
  - User picked a hypothesis with no cached parquet
  - That hypothesis's family IS in the builder registry

If family is not registered, returns ok=False/error='unknown_family'.
Frontend then falls back to the Claude handoff path.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/series_factory", tags=["series"])


class BuildRequest(BaseModel):
    family:        str
    hypothesis_id: str
    subtype:       Optional[str] = None     # F6.1 (2026-06-05): if omitted, looked up from forward_vector
    params:        Optional[dict] = None
    force:         bool = False


class BuildResult(BaseModel):
    ok:               bool
    path:             Optional[str] = None
    family:           str
    subtype:          Optional[str] = None  # F6.1: what the caller asked for
    resolved_subtype: Optional[str] = None  # F6.4: which registered key actually ran
    via_alias:        bool = False          # F6.4: True if alias map mediated
    hypothesis_id:    str
    n_obs:            int = 0
    from_cache:       bool = False
    elapsed_s:        float = 0.0
    error:            Optional[str] = None
    ts:               Optional[str] = None


def _resolve_subtype_from_fv(hypothesis_id: str) -> Optional[str]:
    """F6.1 helper: look up mechanism_subtype from the forward_vector for
    this hypothesis_id. The frontend SHOULD pass subtype explicitly, but
    if it doesn't, we fall back to the typed source rather than 500."""
    try:
        from engine.research_store.forward_vectors import generate_forward_vectors
        for v in generate_forward_vectors():
            if v.source_hypothesis_id == hypothesis_id:
                return v.mechanism_subtype
    except Exception:
        logger.exception("series_factory: subtype lookup failed for %s", hypothesis_id)
    return None


@router.post("/build", response_model=BuildResult)
def build_series(req: BuildRequest) -> BuildResult:
    try:
        from engine.series_factory import build
    except Exception as exc:
        logger.exception("series_factory import failed")
        raise HTTPException(status_code=500, detail=f"import_failed:{exc}")

    # F6.1 (2026-06-05): resolve subtype. Frontend may pass it explicitly
    # (preferred); otherwise we look it up from the forward_vector store
    # for this hypothesis_id. Pre-F6.1 the API didn't pass subtype at all,
    # causing engine.series_factory.build to TypeError -> 500 HTML page ->
    # frontend JSON.parse("Internal S...") fail. Honest fallback: if
    # neither route resolves a subtype, return ok=False/error rather than
    # 500 so the frontend can show "subtype unknown — use Claude handoff".
    subtype = (req.subtype or "").strip()
    if not subtype:
        subtype = _resolve_subtype_from_fv(req.hypothesis_id) or ""
    if not subtype:
        return BuildResult(
            ok=False,
            family=req.family,
            subtype=None,
            hypothesis_id=req.hypothesis_id,
            error=("subtype_required: API caller did not pass `subtype` and "
                    "no forward_vector with this hypothesis_id has one. "
                    "Frontend should pass forward_vector.mechanism_subtype."),
        )

    result = build(
        family        = req.family,
        subtype       = subtype,
        hypothesis_id = req.hypothesis_id,
        params        = req.params,
        force         = req.force,
    )
    return BuildResult(**result)


@router.get("/families")
def list_registered_families() -> dict:
    try:
        from engine.series_factory import list_families
        return {"families": list_families()}
    except Exception as exc:
        return {"families": [], "error": str(exc)[:200]}
