"""api/routes_daily_memo.py — N1 "Book 当日简报" endpoint.

GET  /api/agents/state_of_book?force=0    return today's memo (cached
                                           if exists; generates on first
                                           call of the day)
POST /api/agents/state_of_book/regenerate  force regenerate now (ignores
                                           cache; costs one Claude call)

Frontend StateOfBookTile reads from GET and shows a "regenerate"
button that hits the POST.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents/state_of_book", tags=["agents"])


class StateOfBookMemo(BaseModel):
    date_key:     str
    generated_ts: str
    markdown:     str
    n_citations:  int
    model:        Optional[str] = None
    elapsed_s:    float
    from_cache:   bool
    error:        Optional[str] = None


@router.get("", response_model=StateOfBookMemo)
def get_memo(force: bool = Query(False, description="Bypass cache + regenerate")) -> StateOfBookMemo:
    """Return today's memo. Generates on first call of the day; cached
    thereafter. Pass ?force=true to regenerate explicitly (costs one
    Claude call)."""
    try:
        from engine.agents.daily_memo import generate
        memo = generate(force=force)
        return StateOfBookMemo(**memo)
    except Exception as exc:
        logger.exception("state_of_book GET failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])


@router.post("/regenerate", response_model=StateOfBookMemo)
def regenerate_memo() -> StateOfBookMemo:
    """Force regeneration. Equivalent to GET ?force=true but with a
    dedicated POST shape so frontend can wire it as a button."""
    try:
        from engine.agents.daily_memo import generate
        memo = generate(force=True)
        return StateOfBookMemo(**memo)
    except Exception as exc:
        logger.exception("state_of_book regenerate failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])
