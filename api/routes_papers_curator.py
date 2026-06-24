"""api/routes_papers_curator.py — Employee A daily digest surface.

Phase 1.6 (2026-06-05). Read-only endpoint that joins the three
papers_curator stores (cache + judgments + summaries) into the rows
the UI needs to render `/research/papers/incoming`.

GET /api/papers_curator/incoming?days=14
    → list of crawled candidates with judgment + summary attached,
      ordered by (recommended_action priority, judgment confidence,
      published_ts).

POST /api/papers_curator/skip {source, source_id}
    → mark a candidate as user-skipped; surfaces with status="user_skipped"
      and drops to bottom of daily digest.

Phase 2.0 step 5b (2026-06-06):

POST /api/papers_curator/synthesis/run {dry_run, extra_tags}
    → invoke the cross-source synthesis pipeline (Sonnet 4.6 LLM call).
      Returns the structured result from run_synthesis_pipeline().
      dry_run=true skips persistence — UI uses this for preview.

NO LLM calls except the synthesis endpoint, which is an explicit
user-triggered action (NOT automatic on page load).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/papers_curator", tags=["papers_curator"])

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SKIPS_PATH = _REPO_ROOT / "data" / "papers_curator" / "user_skips.jsonl"


# ──────────────────────────────────────────────────────────────────────
# Response shape
# ──────────────────────────────────────────────────────────────────────
class IncomingRow(BaseModel):
    """One candidate row joined across cache + judgment + summary."""
    # cache fields
    source:        str
    source_id:     str
    title:         str
    authors:       list[str]
    abstract:      str
    abs_url:       str
    pdf_url:       str
    published_ts:  str
    categories:    list[str]
    fetched_ts:    str

    # judgment fields (None until judge runs)
    judged:               bool = False
    is_tradable_factor:   bool = False
    filter_confidence:    float = 0.0
    filter_reason:        str = ""
    filter_category:      str = ""

    # summary fields (None until summarize runs)
    summarized:           bool = False
    thesis:               str = ""
    mechanism:            str = ""
    testable_hypothesis:  str = ""
    why_matters_for_us:   str = ""
    risk_flags:           list[str] = []
    recommended_action:   str = ""        # INGEST | READ_AND_DISCARD | SKIP | ""

    # status
    user_skipped:         bool = False    # user clicked skip in the UI


class IncomingDigest(BaseModel):
    days_requested:    int
    n_total:           int
    n_today:           int
    counts:            dict[str, int]    # by recommended_action + judged states
    rows:              list[IncomingRow]


# ──────────────────────────────────────────────────────────────────────
# User skip log (append-only, dedup on (source, source_id))
# ──────────────────────────────────────────────────────────────────────
def _load_skipped_keys() -> set[tuple[str, str]]:
    if not _SKIPS_PATH.is_file():
        return set()
    out: set[tuple[str, str]] = set()
    for line in _SKIPS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.add((str(d.get("source", "")), str(d.get("source_id", ""))))
        except Exception:
            pass
    return out


class SkipRequest(BaseModel):
    source:    str
    source_id: str
    reason:    str = ""    # optional free-text


@router.post("/skip")
def skip_candidate(req: SkipRequest):
    """Mark a candidate as user-skipped. Idempotent (re-skipping is
    a no-op; the log keeps the first entry)."""
    skipped = _load_skipped_keys()
    if (req.source, req.source_id) in skipped:
        return {"status": "already_skipped"}
    _SKIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SKIPS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source":     req.source,
            "source_id":  req.source_id,
            "reason":     req.reason,
            "skipped_ts": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, ensure_ascii=False) + "\n")
    return {"status": "skipped"}


# ──────────────────────────────────────────────────────────────────────
# Digest
# ──────────────────────────────────────────────────────────────────────
_ACTION_ORDER = {"INGEST": 0, "READ_AND_DISCARD": 1, "SKIP": 2, "": 3}


@router.get("/incoming", response_model=IncomingDigest)
def incoming_digest(days: int = Query(14, ge=1, le=60)):
    """Join cache + judgments + summaries for candidates fetched in the
    last `days` days. Ordering: recommended_action priority (INGEST >
    READ > SKIP > unrated), then confidence DESC, then published DESC.
    User-skipped candidates pushed to the very bottom (still visible
    so user can change their mind).
    """
    try:
        from engine.agents.papers_curator import (
            load_cache, latest_by_paper, latest_summary_by_paper,
        )
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"papers_curator import failed: {exc}")

    cache = load_cache()
    judgments = latest_by_paper()
    summaries = latest_summary_by_paper()
    skipped = _load_skipped_keys()

    # Filter by fetched_ts window
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    today_iso = _dt.datetime.utcnow().strftime("%Y-%m-%d")

    rows: list[IncomingRow] = []
    n_today = 0
    counts = {"INGEST": 0, "READ_AND_DISCARD": 0, "SKIP": 0,
              "unrated": 0, "user_skipped": 0}

    for c in cache:
        if c.fetched_ts < cutoff_iso:
            continue
        key = (c.source, c.source_id)
        j = judgments.get(key)
        s = summaries.get(key)
        is_skipped = key in skipped

        row = IncomingRow(
            source       = c.source,
            source_id    = c.source_id,
            title        = c.title,
            authors      = list(c.authors),
            abstract     = c.abstract,
            abs_url      = c.abs_url,
            pdf_url      = c.pdf_url,
            published_ts = c.published_ts,
            categories   = list(c.categories),
            fetched_ts   = c.fetched_ts,
            judged             = j is not None,
            is_tradable_factor = bool(j and j.is_tradable_factor),
            filter_confidence  = float(j.confidence) if j else 0.0,
            filter_reason      = j.one_line_reason if j else "",
            filter_category    = j.category_guess if j else "",
            summarized            = s is not None,
            thesis                = s.thesis if s else "",
            mechanism             = s.mechanism if s else "",
            testable_hypothesis   = s.testable_hypothesis if s else "",
            why_matters_for_us    = s.why_matters_for_us if s else "",
            risk_flags            = list(s.risk_flags) if s else [],
            recommended_action    = s.recommended_action if s else "",
            user_skipped          = is_skipped,
        )
        rows.append(row)
        if c.fetched_ts.startswith(today_iso):
            n_today += 1
        if is_skipped:
            counts["user_skipped"] += 1
        elif s and s.recommended_action in counts:
            counts[s.recommended_action] += 1
        else:
            counts["unrated"] += 1

    # Sort: user_skipped to bottom; otherwise (action_order, -confidence, -published)
    def _sort_key(r: IncomingRow):
        return (
            1 if r.user_skipped else 0,
            _ACTION_ORDER.get(r.recommended_action, 3),
            -r.filter_confidence,
            # reverse-alphabetical iso8601 = newest first
            "" if not r.published_ts else r.published_ts,
        )
    rows.sort(key=lambda r: (
        _sort_key(r)[0],
        _sort_key(r)[1],
        _sort_key(r)[2],
        # Negate by inverting comparison via tuple of (-int(ts)) won't
        # work on string; flip order with sort(reverse) for published.
        # Use a synthetic descending key via inversion:
        # newer published_ts → earlier in sort. Reverse the string.
    ))
    # Secondary stable pass for newest-first within ties on the above
    rows.sort(key=lambda r: r.published_ts, reverse=True)
    # Then primary pass (stable) for the real priority
    rows.sort(key=lambda r: (
        1 if r.user_skipped else 0,
        _ACTION_ORDER.get(r.recommended_action, 3),
        -r.filter_confidence,
    ))

    return IncomingDigest(
        days_requested = days,
        n_total        = len(rows),
        n_today        = n_today,
        counts         = counts,
        rows           = rows,
    )


# ──────────────────────────────────────────────────────────────────────
# Phase 2.0 step 5b: cross-source synthesis trigger
# ──────────────────────────────────────────────────────────────────────
class SynthesisRunRequest(BaseModel):
    """Trigger a synthesis run. dry_run=true skips persistence and only
    returns the candidates for UI preview (use this for the "what would
    A propose right now?" button); dry_run=false persists to
    hypotheses.jsonl as extraction_method=LLM_SYNTHESIS rows."""
    dry_run:        bool = True       # safe default — explicit opt-in to persist
    summaries_days: int  = 14
    events_days:    int  = 30
    extra_tags:     list[str] = []


class SynthesisRunResponse(BaseModel):
    """Mirror of run_synthesis_pipeline() result shape (typed for the UI)."""
    run_ts:                  str
    dry_run:                 bool
    snapshot:                dict
    candidates:              list[dict]    # rich SynthesizedCandidate dicts
    n_candidates:            int
    written_hypothesis_ids:  list[str]
    n_written:               int
    errors:                  list[str]
    event_id:                Optional[str] = None    # step 4c audit-event id


@router.post("/synthesis/run", response_model=SynthesisRunResponse)
def trigger_synthesis(req: SynthesisRunRequest):
    """Run Employee A's cross-source synthesis pipeline.

    Cost: ≤ $0.10/call (Sonnet 4.6 single shot). User-initiated only —
    NEVER called automatically by polling / page-load.

    The endpoint always returns 200 with a structured result. Errors
    surface in `response.errors[]` rather than as HTTP failures, so the
    UI can render partial results (snapshot succeeded but LLM failed,
    etc). 500 only for unrecoverable import errors.
    """
    try:
        from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"synthesis_runner import failed: {exc}")

    result = run_synthesis_pipeline(
        dry_run        = req.dry_run,
        summaries_days = req.summaries_days,
        events_days    = req.events_days,
        extra_tags     = tuple(req.extra_tags),
    )
    return SynthesisRunResponse(**result)
