"""api/routes_paper_chain.py — read-only REST endpoints for the PAPER →
HYPOTHESIS → TEST → VERDICT chain.

Powers the upcoming UI:
  /research/papers          — papers list w/ status, shelf, hypothesis count
  /research/papers/[id]     — paper detail + per-hypothesis test status
  /research/forward         — forward vectors (ranked by priority)
  /research/legacy          — pretrain_grounded lessons (segregated)

All endpoints respect the chain doctrine — `include_legacy=False` by
default so legacy 47 lessons are NOT mixed into the new surface unless
explicitly requested.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel


router = APIRouter(prefix="/api/paper_chain", tags=["paper_chain"])


# ─────────────────────── response models ──────────────────────────────


class PaperSummary(BaseModel):
    paper_id:        str
    title:           str
    year:            int
    authors:         list[str]
    venue:           str
    doi:             str
    fulltext_status: str
    n_chunks:        int
    shelves:         list[str]
    n_hypotheses:    int
    n_tested:        int          # of n_hypotheses, how many have a lesson


class PaperDetail(BaseModel):
    paper_id:        str
    title:           str
    year:            int
    authors:         list[str]
    venue:           str
    doi:             str
    fulltext_status: str
    n_chunks:        int
    shelves:         list[str]
    shelf_notes:     dict[str, str]
    pdf_source_url:  str
    referenced_by_lessons:    list[str]
    referenced_by_sleeves:    list[str]
    referenced_by_factors:    list[str]


class HypothesisSummary(BaseModel):
    hypothesis_id:        str
    source_paper_id:      str
    claim:                str
    mechanism_family:     str
    mechanism_subtype:    str
    predicted_direction:  str
    predicted_magnitude:  str
    required_data:        list[str]
    n_verbatim_quotes:    int
    review_state:         str
    is_tested:            bool
    tested_by_lessons:    list[str]


class ForwardVectorSummary(BaseModel):
    forward_vector_id:    str
    source_paper_id:      str
    paper_title:          str
    source_hypothesis_id: str
    claim:                str
    mechanism_family:     str
    mechanism_subtype:    str
    predicted_direction:  str
    predicted_magnitude:  str
    required_data:        list[str]
    priority:             str
    priority_signals:     dict
    # PM-approval overlay (2026-06-04). Keyed by source_hypothesis_id;
    # see engine.research_store.forward_vectors.review.
    pm_status:            str           # extracted | reviewed | approved | rejected
    pm_reviewed_ts:       Optional[str] = None
    pm_reviewed_by:       Optional[str] = None
    pm_note:              Optional[str] = None
    # P0-D — data-availability hint. Server scans data/cache/* and
    # data/series/* for token matches against required_data; this
    # field is "have" | "partial" | "missing" | "unknown" so the user
    # can filter for hypotheses we can actually test today.
    data_status:          str = "unknown"
    data_have:            list[str] = []    # which required_data terms matched
    data_missing:         list[str] = []    # which didn't

    # Stage C Tier B (2026-06-07): orthogonal_to_anchors carried
    # from the source Hypothesis (Phase E + Tier A populated this).
    # Empty list for pre-Phase-E hypotheses + paper-rooted extractions
    # (where the paper itself IS the anchor by construction).
    # Each entry: {anchor_paper_id: str (8-char short),
    #               why_orthogonal: str}.
    orthogonal_to_anchors: list[dict] = []


class ForwardVectorReviewRequest(BaseModel):
    source_hypothesis_id: str
    status:               str           # extracted | reviewed | approved | rejected
    reviewed_by:          Optional[str] = "user"
    note:                 Optional[str] = ""


class ComposerGap(BaseModel):
    role:         str   # SIGNAL / UNIVERSE / WEIGHTING / REBALANCE / RISK_FILTER
    expected_key: str   # the spec enum value the registry was looked up by
    reason:       str   # "missing" | "unknown_extracted"


class ForwardVectorRanked(BaseModel):
    """F1 (2026-06-05): the enriched forward-vector view powering the
    candidate pipeline UI. Joins five sources at view-time:
      - forward_vectors (generate_forward_vectors)
      - hypothesis_specs (latest_for source_hypothesis_id)
      - composer coverage (is_spec_covered)
      - graveyard_collision (check_collision)
      - direction_proposer additive score
    No new persistent state — the join is recomputed per request so
    a new RED in the graveyard or a freshly-built composer component
    affects ranking on the very next API call."""
    forward_vector_id:    str
    source_paper_id:      str
    paper_title:          str
    source_hypothesis_id: str
    claim:                str

    # From hypothesis_spec
    claim_type:           str       # FACTOR_HYPOTHESIS / METHODOLOGY / etc.
    spec_hash:            Optional[str] = None
    family:               str       # FamilyV2 if FACTOR_HYPOTHESIS else OTHER
    signal_type:          str       # primary leg signal_type
    asset_class:          str
    subset:               str
    weighting:            str
    rebalance:            str

    # Operational layer
    priority:             str       # high / medium / low
    pm_status:            str       # extracted / reviewed / approved / rejected
    data_status:          str       # have / partial / missing / unknown
    data_have:            list[str] = []
    data_missing:         list[str] = []

    # Composer readiness (the load-bearing field)
    composer_status:      str       # ready / missing_components / no_spec / not_factor
    composer_gaps:        list[ComposerGap] = []

    # Graveyard collision
    graveyard_verdict:    str       # CLEAN / WARN / RISK
    graveyard_n_matches:  int       # how many RED matches across all dims
    graveyard_top_match:  Optional[str] = None   # subject_id of top match

    # Direction proposer score (additive_v2_T3.5; range ~0..1)
    direction_score:      float
    direction_rank:       Optional[int] = None   # filled if user asks ordered=true


class LessonSummary(BaseModel):
    lesson_id:            str
    candidate_name:       str
    version:              int
    verdict:              str
    mechanism_family:     str
    mechanism_subtype:    str
    failure_modes:        list[str]
    grounding_method:     str
    tested_hypothesis_ids: list[str]
    n_verbatim_quotes:    int
    created_ts:           str
    summary:              str


# ─────────────────────── helper loaders (lazy) ────────────────────────


def _load_state():
    """Lazy import all stores once per request."""
    from engine.research_store.hypothesis import (
        latest_per_paper, load_hypotheses,
    )
    from engine.research_store.papers import (
        latest_per_doi, load_registry,
    )
    from engine.research_store.red_lessons import load_lessons
    from engine.research_store.red_lessons.store import latest_per_candidate

    lessons = list(latest_per_candidate(load_lessons()).values())
    registry = list(latest_per_doi(load_registry()).values())
    hyps = load_hypotheses()
    hyps_by_paper = latest_per_paper(hyps)
    hyps_by_id = {h.hypothesis_id: h for h in
                  (max([x for x in hyps if x.hypothesis_id == h.hypothesis_id],
                       key=lambda x: x.version) for h in hyps)}
    return {
        "lessons":       lessons,
        "registry":      registry,
        "hyps_by_paper": hyps_by_paper,
        "hyps_by_id":    hyps_by_id,
    }


def _tested_set(lessons: list, include_legacy: bool) -> set[str]:
    from engine.research_store.red_lessons.retrieval import (
        tested_hypothesis_ids,
    )
    return tested_hypothesis_ids(lessons=lessons, include_legacy=include_legacy)


# ─────────────────────── PAPERS endpoints ─────────────────────────────


@router.get("/papers", response_model=list[PaperSummary])
def list_papers(
    fulltext_status: Optional[str] = Query(None,
        description="ingested | metadata_only | paywalled | unattempted"),
    shelf: Optional[str] = Query(None,
        description="filter to papers carrying this shelf"),
    include_legacy: bool = False,
):
    """All papers in registry with summary stats."""
    st = _load_state()
    tested = _tested_set(st["lessons"], include_legacy)

    out: list[PaperSummary] = []
    for entry in st["registry"]:
        if fulltext_status and entry.fulltext_status.value != fulltext_status:
            continue
        if shelf and shelf not in {s.value for s in entry.shelves}:
            continue
        paper_hyps = st["hyps_by_paper"].get(entry.paper_id, [])
        n_tested = sum(1 for h in paper_hyps if h.hypothesis_id in tested)
        out.append(PaperSummary(
            paper_id        = entry.paper_id,
            title           = entry.title,
            year            = entry.year,
            authors         = list(entry.authors),
            venue           = entry.venue,
            doi             = entry.doi,
            fulltext_status = entry.fulltext_status.value,
            n_chunks        = entry.n_chunks,
            shelves         = [s.value for s in entry.shelves],
            n_hypotheses    = len(paper_hyps),
            n_tested        = n_tested,
        ))
    return out


@router.get("/papers/{paper_id}", response_model=PaperDetail)
def get_paper(paper_id: str):
    st = _load_state()
    entry = next((e for e in st["registry"] if e.paper_id == paper_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"paper {paper_id} not found")
    return PaperDetail(
        paper_id        = entry.paper_id,
        title           = entry.title,
        year            = entry.year,
        authors         = list(entry.authors),
        venue           = entry.venue,
        doi             = entry.doi,
        fulltext_status = entry.fulltext_status.value,
        n_chunks        = entry.n_chunks,
        shelves         = [s.value for s in entry.shelves],
        shelf_notes     = dict(entry.shelf_notes),
        pdf_source_url  = entry.pdf_source_url,
        referenced_by_lessons    = list(entry.referenced_by_lessons),
        referenced_by_sleeves    = list(entry.referenced_by_sleeves),
        referenced_by_factors    = list(entry.referenced_by_factors),
    )


@router.get("/papers/{paper_id}/hypotheses",
            response_model=list[HypothesisSummary])
def list_paper_hypotheses(paper_id: str, include_legacy: bool = False):
    """List hypotheses extracted from this paper + per-hypothesis test status."""
    st = _load_state()
    paper_hyps = st["hyps_by_paper"].get(paper_id, [])
    if not paper_hyps:
        # paper exists but no hypotheses → empty list OK
        if not any(e.paper_id == paper_id for e in st["registry"]):
            raise HTTPException(status_code=404,
                                detail=f"paper {paper_id} not found")
    tested = _tested_set(st["lessons"], include_legacy)

    out: list[HypothesisSummary] = []
    for h in paper_hyps:
        tested_by = [
            L.lesson_id for L in st["lessons"]
            if h.hypothesis_id in L.tested_hypothesis_ids
            and (include_legacy or L.grounding_method.value != "pretrain_grounded")
        ]
        out.append(HypothesisSummary(
            hypothesis_id       = h.hypothesis_id,
            source_paper_id     = h.source_paper_id,
            claim               = h.claim,
            mechanism_family    = h.mechanism_family.value,
            mechanism_subtype   = h.mechanism_subtype,
            predicted_direction = h.predicted_direction.value,
            predicted_magnitude = h.predicted_magnitude,
            required_data       = list(h.required_data),
            n_verbatim_quotes   = len(h.verbatim_quotes),
            review_state        = h.review_state.value,
            is_tested           = h.hypothesis_id in tested,
            tested_by_lessons   = tested_by,
        ))
    return out


# ─────────────────────── FORWARD VECTORS ──────────────────────────────


# P0-D — data-availability hint. Built once per request and reused
# across vectors. Cache filenames stripped to lowercase tokens so
# "_carry_3leg_monthly.parquet" matches required_data text like
# "currency carry, 3-leg, monthly" via token overlap.

import time as _time_mod
import re as _re_mod

_DATA_INV_CACHE: dict = {"tokens": None, "stems": None, "ts": 0.0}


def _build_data_inventory() -> tuple[set[str], list[str]]:
    """Return (token_set, stem_list). Token set is the union of every
    lowercase word-fragment found in cache + series filenames. Stem
    list is the full filename (lower, no extension) — used for the
    finer match step.

    Cached for 60s — file list rarely changes mid-session and re-
    scanning on every list_forward_vectors call would waste IO."""
    now = _time_mod.time()
    if _DATA_INV_CACHE["tokens"] is not None and now - _DATA_INV_CACHE["ts"] < 60.0:
        return _DATA_INV_CACHE["tokens"], _DATA_INV_CACHE["stems"]

    import pathlib as _path_mod
    root = _path_mod.Path(__file__).resolve().parent.parent
    sources = [root / "data" / "cache", root / "data" / "series"]
    stems: list[str] = []
    tokens: set[str] = set()
    for src in sources:
        if not src.is_dir(): continue
        for p in src.iterdir():
            if not p.is_file(): continue
            name = p.stem.lower().lstrip("_")
            if not name: continue
            stems.append(name)
            for tok in _re_mod.split(r"[_\-\s\.]+", name):
                tok = tok.strip()
                if len(tok) >= 3 and not tok.isdigit():
                    tokens.add(tok)
    _DATA_INV_CACHE["tokens"] = tokens
    _DATA_INV_CACHE["stems"]  = stems
    _DATA_INV_CACHE["ts"]     = now
    return tokens, stems


def _data_coverage_for(required_data: list[str]) -> tuple[str, list[str], list[str]]:
    """For a hypothesis's required_data array, classify each term as
    matched / unmatched against the inventory. Returns (status, have,
    missing) where status is:
        "have"     — all required_data items matched
        "partial"  — some matched, some didn't
        "missing"  — nothing matched
        "unknown"  — required_data was empty
    """
    if not required_data:
        return "unknown", [], []
    tokens, stems = _build_data_inventory()
    have: list[str] = []
    missing: list[str] = []
    for term in required_data:
        if not term or not term.strip():
            continue
        term_lc = term.lower()
        # Pull every word fragment from the term and check overlap.
        term_tokens = {t.strip() for t in _re_mod.split(r"[_\-\s\.,;()\[\]/]+", term_lc)
                       if len(t.strip()) >= 3 and not t.isdigit()}
        # Match logic: term is "have" if at least 2 of its substantive
        # tokens appear in the inventory token set, OR if a full stem
        # contains 2+ of the term tokens. Tweakable; deliberately
        # forgiving since required_data text is LLM-generated.
        n_token_hits = len(term_tokens & tokens)
        n_stem_hits  = sum(1 for s in stems if sum(1 for tt in term_tokens if tt in s) >= 2)
        if n_token_hits >= 2 or n_stem_hits >= 1:
            have.append(term)
        else:
            missing.append(term)
    if not have:    return "missing", have, missing
    if not missing: return "have",    have, missing
    return "partial", have, missing


@router.get("/forward-vectors", response_model=list[ForwardVectorSummary])
def list_forward_vectors(
    priority: Optional[str] = Query(None,
        description="high | medium | low"),
    mechanism_family: Optional[str] = Query(None),
    pm_status: Optional[str] = Query(None,
        description="extracted | reviewed | approved | rejected. "
                    "Pass 'open' as a synonym for 'approved,reviewed,extracted' "
                    "(everything not rejected)."),
    data_status: Optional[str] = Query(None,
        description="have | partial | missing | unknown — filter by "
                    "whether required_data is covered by data/cache+series. "
                    "Comma-separated multi-select supported."),
    top: int = Query(100, ge=1, le=500),
):
    """Forward vectors — ranked by priority (high first), with PM review overlay."""
    from engine.research_store.forward_vectors import generate_forward_vectors
    from engine.research_store.forward_vectors.review import load_latest_reviews

    vecs = generate_forward_vectors()
    reviews = load_latest_reviews()

    if priority:
        vecs = [v for v in vecs if v.priority.value == priority]
    if mechanism_family:
        vecs = [v for v in vecs if v.mechanism_family.value == mechanism_family]

    if pm_status:
        wanted = set()
        if pm_status == "open":
            wanted = {"approved", "reviewed", "extracted"}
        else:
            wanted = {s.strip() for s in pm_status.split(",") if s.strip()}
        def _match(v) -> bool:
            r = reviews.get(v.source_hypothesis_id)
            actual = r.status.value if r else "extracted"
            return actual in wanted
        vecs = [v for v in vecs if _match(v)]

    # Compute data coverage up-front so the filter can run before the
    # `top` truncation (otherwise data=have filter would return < top
    # because the first N happened to be missing).
    coverage: dict[str, tuple[str, list[str], list[str]]] = {}
    for v in vecs:
        coverage[v.source_hypothesis_id] = _data_coverage_for(list(v.required_data))

    if data_status:
        wanted_d = {s.strip() for s in data_status.split(",") if s.strip()}
        vecs = [v for v in vecs
                if coverage[v.source_hypothesis_id][0] in wanted_d]

    vecs = vecs[: top]

    # Stage C Tier B: bulk-load Hypothesis store ONCE for the JOIN
    # against orthogonal_to_anchors. Avoid per-row store reads.
    orth_by_hid: dict[str, list[dict]] = {}
    try:
        from engine.research_store.hypothesis.store import load_hypotheses
        hyps = load_hypotheses()
        # Latest version per hypothesis_id
        by_hid: dict = {}
        for h in hyps:
            prior = by_hid.get(h.hypothesis_id)
            if prior is None or h.version > prior.version:
                by_hid[h.hypothesis_id] = h
        for hid, h in by_hid.items():
            if h.orthogonal_to_anchors:
                orth_by_hid[hid] = [
                    dict(o) for o in h.orthogonal_to_anchors
                ]
    except Exception:
        # Read failure → orth_by_hid stays empty; UI sees [] per row
        # (degraded but not broken)
        pass

    out: list[ForwardVectorSummary] = []
    for v in vecs:
        r = reviews.get(v.source_hypothesis_id)
        ds, have, missing = coverage[v.source_hypothesis_id]
        out.append(ForwardVectorSummary(
            forward_vector_id    = v.forward_vector_id,
            source_paper_id      = v.source_paper_id,
            paper_title          = v.paper_title,
            source_hypothesis_id = v.source_hypothesis_id,
            claim                = v.claim,
            mechanism_family     = v.mechanism_family.value,
            mechanism_subtype    = v.mechanism_subtype,
            predicted_direction  = v.predicted_direction,
            predicted_magnitude  = v.predicted_magnitude,
            required_data        = list(v.required_data),
            priority             = v.priority.value,
            priority_signals     = dict(v.priority_signals),
            pm_status            = r.status.value if r else "extracted",
            pm_reviewed_ts       = r.reviewed_ts  if r else None,
            pm_reviewed_by       = r.reviewed_by  if r else None,
            pm_note              = r.note         if r else None,
            data_status          = ds,
            data_have            = have,
            data_missing         = missing,
            orthogonal_to_anchors = orth_by_hid.get(
                                       v.source_hypothesis_id, []),
        ))
    return out


@router.get("/forward-vectors/ranked", response_model=list[ForwardVectorRanked])
def list_forward_vectors_ranked(
    top:           int = Query(100, ge=1, le=500),
    family:        Optional[str] = Query(None, description="filter by mechanism_family"),
    composer:      Optional[str] = Query(None,
        description="filter by composer_status: ready / missing_components / no_spec / not_factor "
                    "— comma-separated for multi-select"),
    include_legacy_non_factor: bool = Query(False,
        description="include non-FACTOR_HYPOTHESIS rows (METHODOLOGY / DECAY_STUDY etc.). "
                    "Default False = hide rows that can't ever become a candidate."),
):
    """F1 (2026-06-05): enriched forward-vectors for candidate pipeline UI.

    Joins fv + hypothesis_spec + composer coverage + graveyard collision
    + direction_proposer additive score at view-time. Sorted by
    (composer_status priority, direction_score desc) so READY candidates
    surface first.

    The composer_status field tells the UI whether a candidate can be
    composed-and-tested today:
      ready               compose + run pipeline
      missing_components  UI shows which signal/universe/etc. are gaps
      no_spec             hypothesis exists but extractor returned None
      not_factor          claim_type != FACTOR_HYPOTHESIS (hidden by default)
    """
    from engine.research_store.forward_vectors import generate_forward_vectors
    from engine.research_store.forward_vectors.review import load_latest_reviews
    from engine.hypothesis_spec.store import latest_for
    from engine.hypothesis_spec.hash import spec_hash
    from engine.hypothesis_spec.enums import ClaimType
    from engine.composer.contract import is_spec_covered
    from engine.agents.graveyard_collision import check_collision
    from engine.agents.direction_proposer import _compose_total_additive
    from engine.agents.direction_proposer import (
        _score_priority, _score_data, _score_orthogonality,
        _score_graveyard, _score_saturation, _deployed_families,
        _data_status_for,
    )

    vecs = generate_forward_vectors()
    reviews = load_latest_reviews()
    deployed = _deployed_families()

    # Family count for saturation (same logic as direction_proposer)
    family_count: dict[str, int] = {}
    for v in vecs:
        fv_fam = (v.mechanism_family.value if hasattr(v.mechanism_family, "value")
                  else str(v.mechanism_family)).lower()
        family_count[fv_fam] = family_count.get(fv_fam, 0) + 1

    rows: list[ForwardVectorRanked] = []
    for v in vecs:
        spec = latest_for(v.source_hypothesis_id)
        if spec is None:
            ct_value, spec_h = "<no spec>", None
            composer_status = "no_spec"
            gaps_out: list[ComposerGap] = []
            fam, st = "OTHER", "UNKNOWN"
            ac, sub = "UNKNOWN", "UNKNOWN"
            w, rb = "UNKNOWN", "UNKNOWN"
        else:
            ct_value = spec.claim_type.value
            spec_h = spec_hash(spec)
            fam = spec.family.value
            st = spec.legs[0].signal_type.value if spec.legs else "UNKNOWN"
            ac = spec.universe.asset_class.value
            sub = spec.universe.subset.value
            w = spec.construction.weighting.value
            rb = spec.construction.rebalance.value

            if spec.claim_type != ClaimType.FACTOR_HYPOTHESIS:
                composer_status = "not_factor"
                gaps_out = []
            else:
                covered, gaps = is_spec_covered(spec)
                composer_status = "ready" if covered else "missing_components"
                gaps_out = [
                    ComposerGap(
                        role=g.role.value,
                        expected_key=g.expected_key,
                        reason=("unknown_extracted" if g.expected_key == "UNKNOWN"
                                or g.expected_key.endswith("__UNKNOWN")
                                else "missing"),
                    )
                    for g in gaps
                ]

        # Hide non-factor by default (METHODOLOGY etc. can't be candidates)
        if not include_legacy_non_factor and composer_status == "not_factor":
            continue

        # Family filter
        v_fam = v.mechanism_family.value if hasattr(v.mechanism_family, "value") else str(v.mechanism_family)
        if family and v_fam.lower() != family.lower():
            continue

        # PM review overlay + data status
        r = reviews.get(v.source_hypothesis_id)
        pm_status = r.status.value if r else "extracted"
        if pm_status == "rejected":
            continue
        ds, have, missing = _data_coverage_for(list(v.required_data or []))

        # Graveyard collision
        try:
            gc = check_collision(
                candidate_name=v.source_hypothesis_id,
                family=fam,
                mechanism_subtype=v.mechanism_subtype,
                claim_text=v.claim,
            )
            gc_verdict = gc.get("verdict", "CLEAN")
            gc_matches = gc.get("matches", []) or []
            gc_top = (gc_matches[0].get("red_candidate") if gc_matches else None)
        except Exception:
            gc_verdict, gc_matches, gc_top = "CLEAN", [], None

        # Direction additive score (T3.5)
        sp = _score_priority(v.priority.value if hasattr(v.priority, "value") else str(v.priority))
        sa = _score_data(ds)
        so = _score_orthogonality(v_fam, deployed)
        sg = _score_graveyard(gc_verdict)
        sf = _score_saturation(v_fam, family_count)
        score = _compose_total_additive(sp, sa, so, sg, sf)

        rows.append(ForwardVectorRanked(
            forward_vector_id    = v.forward_vector_id,
            source_paper_id      = v.source_paper_id,
            paper_title          = v.paper_title,
            source_hypothesis_id = v.source_hypothesis_id,
            claim                = v.claim,
            claim_type           = ct_value,
            spec_hash            = spec_h,
            family               = fam,
            signal_type          = st,
            asset_class          = ac,
            subset               = sub,
            weighting            = w,
            rebalance            = rb,
            priority             = v.priority.value if hasattr(v.priority, "value") else str(v.priority),
            pm_status            = pm_status,
            data_status          = ds,
            data_have            = have,
            data_missing         = missing,
            composer_status      = composer_status,
            composer_gaps        = gaps_out,
            graveyard_verdict    = gc_verdict,
            graveyard_n_matches  = len(gc_matches),
            graveyard_top_match  = gc_top,
            direction_score      = round(score, 3),
        ))

    if composer:
        wanted = {c.strip() for c in composer.split(",") if c.strip()}
        rows = [r for r in rows if r.composer_status in wanted]

    # Sort: ready first, then by direction_score desc within group.
    _status_order = {"ready": 0, "missing_components": 1, "no_spec": 2, "not_factor": 3}
    rows.sort(key=lambda r: (_status_order.get(r.composer_status, 99),
                              -r.direction_score))
    for i, r in enumerate(rows[:top], start=1):
        r.direction_rank = i
    return rows[:top]


@router.post("/forward-vectors/review", response_model=ForwardVectorSummary)
def review_forward_vector(req: ForwardVectorReviewRequest):
    """Record a PM review decision against a hypothesis. Append-only.

    The status overlay is keyed by source_hypothesis_id (stable across
    forward-vector regenerations). Latest entry wins. History is
    preserved in data/research_store/forward_vector_reviews.jsonl for
    audit.
    """
    from engine.research_store.forward_vectors import generate_forward_vectors
    from engine.research_store.forward_vectors.review import (
        record_review, load_latest_reviews, PMReviewStatus,
    )
    from fastapi import HTTPException

    try:
        record_review(
            source_hypothesis_id = req.source_hypothesis_id,
            status               = req.status,
            reviewed_by          = req.reviewed_by or "user",
            note                 = req.note or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Re-locate the vector to return its updated shape; if the
    # hypothesis was never in the forward queue (e.g. already tested),
    # 404 so the UI knows the state didn't take.
    vecs = generate_forward_vectors()
    reviews = load_latest_reviews()
    for v in vecs:
        if v.source_hypothesis_id == req.source_hypothesis_id:
            r = reviews.get(v.source_hypothesis_id)
            return ForwardVectorSummary(
                forward_vector_id    = v.forward_vector_id,
                source_paper_id      = v.source_paper_id,
                paper_title          = v.paper_title,
                source_hypothesis_id = v.source_hypothesis_id,
                claim                = v.claim,
                mechanism_family     = v.mechanism_family.value,
                mechanism_subtype    = v.mechanism_subtype,
                predicted_direction  = v.predicted_direction,
                predicted_magnitude  = v.predicted_magnitude,
                required_data        = list(v.required_data),
                priority             = v.priority.value,
                priority_signals     = dict(v.priority_signals),
                pm_status            = r.status.value if r else "extracted",
                pm_reviewed_ts       = r.reviewed_ts  if r else None,
                pm_reviewed_by       = r.reviewed_by  if r else None,
                pm_note              = r.note         if r else None,
            )
    raise HTTPException(
        status_code=404,
        detail=f"hypothesis_id {req.source_hypothesis_id!r} not in current forward queue "
               "(may already be tested or never extracted)",
    )


# ─────────────────────── LESSONS ──────────────────────────────────────


@router.get("/lessons", response_model=list[LessonSummary])
def list_lessons(
    grounding_method: Optional[str] = Query(None,
        description="paper_grounded | stat_only_grounded | pretrain_grounded"),
    include_legacy: bool = Query(False,
        description="if False (default), pretrain_grounded lessons excluded"),
    candidate_name: Optional[str] = Query(None),
    mechanism_family: Optional[str] = Query(None),
    verdict: Optional[str] = Query(None,
        description="red | yellow | green — filter by verdict (substring match)"),
    limit: int = Query(200, ge=1, le=1000),
):
    """List lessons. Default: only paper_grounded + stat_only_grounded.
    Pass include_legacy=true to include pretrain_grounded (the 47
    pre-2026-06-04 legacy records) — needed for graveyard check use cases."""
    st = _load_state()
    out: list[LessonSummary] = []
    verdict_lc = verdict.lower() if verdict else None
    for L in st["lessons"]:
        gm = L.grounding_method.value
        if not include_legacy and gm == "pretrain_grounded":
            continue
        if grounding_method and gm != grounding_method:
            continue
        if candidate_name and L.candidate_name != candidate_name:
            continue
        if mechanism_family and L.mechanism_family.value != mechanism_family:
            continue
        if verdict_lc and verdict_lc not in L.verdict.lower():
            continue
        out.append(LessonSummary(
            lesson_id           = L.lesson_id,
            candidate_name      = L.candidate_name,
            version             = L.version,
            verdict             = L.verdict,
            mechanism_family    = L.mechanism_family.value,
            mechanism_subtype   = L.mechanism_subtype,
            failure_modes       = [m.value for m in L.failure_modes],
            grounding_method    = gm,
            tested_hypothesis_ids = list(L.tested_hypothesis_ids),
            n_verbatim_quotes   = len(L.verbatim_quotes),
            created_ts          = L.created_ts,
            summary             = L.summary,
        ))
    # Newest first — most recent verdict on the family is the most
    # relevant graveyard signal.
    out.sort(key=lambda x: x.created_ts, reverse=True)
    return out[:limit]


@router.get("/lessons/{lesson_id}")
def get_lesson(lesson_id: str):
    """Full lesson dict (for detail page). All fields including
    verbatim_quotes + failure_evidence + stat_evidence."""
    st = _load_state()
    L = next((x for x in st["lessons"] if x.lesson_id == lesson_id), None)
    if L is None:
        raise HTTPException(status_code=404,
                            detail=f"lesson {lesson_id} not found")
    return L.to_dict()


# ─────────────────────── CHUNKS (full-text reader) ───────────────────


class ChunkOut(BaseModel):
    chunk_id:        str
    text:            str
    section:         str
    paragraph_idx:   int
    quoted_by:       list[dict]   # [{hypothesis_id, quote_text, lesson_ids}]


@router.get("/papers/{paper_id}/chunks", response_model=list[ChunkOut])
def list_paper_chunks(paper_id: str):
    """Return all chunks for the paper, sorted by paragraph_idx, with
    annotations of which hypothesis quotes hit which chunk. Powers the
    paper-reader UI (highlights quotes inline in chunk text)."""
    from engine.research_store.red_lessons.papers_chroma import get_collection

    st = _load_state()
    paper = next((e for e in st["registry"] if e.paper_id == paper_id), None)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {paper_id} not found")

    coll = get_collection()
    res = coll.get(where={"doi": paper.doi})
    ids       = res.get("ids") or []
    docs      = res.get("documents") or []
    metas     = res.get("metadatas") or []

    # Build per-chunk annotation: which hypothesis_quotes hit this chunk
    paper_hyps = st["hyps_by_paper"].get(paper_id, [])
    # build map: chunk_id -> [(hypothesis_id, quote_text, [lesson_ids citing this hyp])]
    annotations: dict[str, list[dict]] = {}
    lessons = st["lessons"]
    for h in paper_hyps:
        # lessons that cite this hyp
        lesson_ids_for_h = [
            L.lesson_id for L in lessons
            if h.hypothesis_id in L.tested_hypothesis_ids
            and L.grounding_method.value != "pretrain_grounded"
        ]
        for q in h.verbatim_quotes:
            annotations.setdefault(q.chunk_id, []).append({
                "hypothesis_id": h.hypothesis_id,
                "quote_text":    q.quote_text,
                "section_ref":   q.section_ref,
                "lesson_ids":    lesson_ids_for_h,
            })

    out: list[ChunkOut] = []
    for cid, doc, meta in zip(ids, docs, metas):
        out.append(ChunkOut(
            chunk_id      = cid,
            text          = doc,
            section       = meta.get("section", ""),
            paragraph_idx = int(meta.get("paragraph_idx", 0)),
            quoted_by     = annotations.get(cid, []),
        ))
    out.sort(key=lambda c: c.paragraph_idx)
    return out


# ─────────────────────── SEMANTIC SEARCH ─────────────────────────────


class SearchHit(BaseModel):
    chunk_id:        str
    text:            str
    paper_id:        str
    paper_title:     str
    section:         str
    distance:        Optional[float]


@router.get("/search", response_model=list[SearchHit])
def semantic_search(
    q:   str = Query(..., min_length=2, description="natural-language query"),
    top: int = Query(10, ge=1, le=50),
):
    """Semantic search over papers_chroma full-text. Powers the global
    search bar. Returns the top-K most similar chunks across all papers."""
    from engine.research_store.red_lessons.retrieval import query_papers_semantic
    hits = query_papers_semantic(q, top_k=top)

    # Resolve paper_id + title from chunk metadata
    st = _load_state()
    by_doi = {e.doi.lower(): e for e in st["registry"] if e.doi}

    out: list[SearchHit] = []
    for h in hits:
        meta = h.get("metadata") or {}
        doi  = (meta.get("doi") or "").lower()
        paper = by_doi.get(doi)
        out.append(SearchHit(
            chunk_id    = h.get("chunk_id", ""),
            text        = (h.get("text") or "")[:600],   # truncate for header dropdown
            paper_id    = paper.paper_id if paper else "",
            paper_title = paper.title if paper else "(unknown)",
            section     = meta.get("section", ""),
            distance    = h.get("distance"),
        ))
    return out


# ─────────────────────── PAPER INGEST (R2.4) ──────────────────────────


class PaperPreviewResponse(BaseModel):
    """Heuristic metadata extracted from a PDF / URL before commit."""
    preview_id:       str          # opaque id /ingest uses to pick up cached text
    title_guess:      str
    authors_guess:    list[str]
    year_guess:       Optional[int]
    doi_guess:        str
    venue_guess:      str
    abstract_guess:   str
    n_pages:          int
    text_chars:       int
    text_preview:     str          # first ~600 chars for sanity
    extraction_note:  str          # human-readable status note


class PaperIngestReasonIn(BaseModel):
    """Phase 1.7 step 3 — user-supplied reason at ingest time.

    free_text:       ≤ 200 chars, trimmed; empty triggers null on save.
    source:          "user" or "agent" (see IngestionReasonSource enum).
    intent_category: one of IntentCategory values (2026-06-06 — user now
                     picks at ingest via dropdown rather than waiting for
                     LLM extraction). None / unknown silently downgrades
                     to OTHER on the persisted entry.
    """
    free_text:       str
    source:          str               # "user" | "agent"
    intent_category: Optional[str] = None


class PaperIngestRequest(BaseModel):
    """Confirmed metadata + shelves to write into the registry."""
    title:           str
    year:            int
    authors:         list[str]
    venue:           str
    doi:             str
    abstract:        str
    shelves:         list[str]      # at least one required (default "other")
    shelf_notes:     dict[str, str] = {}
    pdf_source_url:  str = ""
    note:            str = ""
    # Used to find the pre-fetched PDF text in the cache (see /preview).
    preview_id:      str
    # Phase 1.7 step 3 (2026-06-06): optional — None when user leaves
    # the textarea blank. See PaperIngestReasonIn.
    ingestion_reason: Optional[PaperIngestReasonIn] = None


class PaperIngestResponse(BaseModel):
    paper_id:      str
    title:         str
    n_chunks:      int
    n_hypotheses:  int
    n_specs:       int = 0        # F11: typed HypothesisSpecs auto-extracted
    registry_path: str
    next_url:      str            # /research/papers/<id>


# Simple in-memory cache of recent PDF previews so /ingest can pick up
# the text without re-uploading. Keyed by preview_id (uuid4). TTL 10
# minutes (the user must click ingest within that window). Not durable
# across restarts — by design (don't keep raw PDFs in memory longer
# than needed).
import time as _time
import uuid as _uuid
import re as _re
_PDF_CACHE: dict[str, tuple[float, dict]] = {}
_PDF_CACHE_TTL_SEC = 600


def _cache_put(payload: dict) -> str:
    """Evict expired entries; store payload; return preview_id."""
    now = _time.time()
    # Evict
    expired = [k for k, (ts, _) in _PDF_CACHE.items() if now - ts > _PDF_CACHE_TTL_SEC]
    for k in expired:
        _PDF_CACHE.pop(k, None)
    pid = _uuid.uuid4().hex[:16]
    _PDF_CACHE[pid] = (now, payload)
    return pid


def _cache_get(pid: str) -> Optional[dict]:
    item = _PDF_CACHE.get(pid)
    if not item: return None
    ts, payload = item
    if _time.time() - ts > _PDF_CACHE_TTL_SEC:
        _PDF_CACHE.pop(pid, None)
        return None
    return payload


# ─── Heuristic metadata guessers (no LLM, ~ms) ───

def _guess_title(text: str) -> str:
    """First non-blank line, trimmed to 200 chars. Skips short lines that
    look like running headers."""
    for line in text.splitlines()[:30]:
        s = line.strip()
        if len(s) >= 20 and not s.lower().startswith(("page ", "doi:", "arxiv", "http")):
            return s[:200]
    return ""

_DOI_RE = _re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", _re.I)

def _guess_doi(text: str) -> str:
    m = _DOI_RE.search(text[:8000])
    return m.group(0).rstrip(".,;)") if m else ""

_YEAR_RE = _re.compile(r"\b((?:19|20)\d{2})\b")

def _guess_year(text: str) -> Optional[int]:
    """Most plausible year in the first 3000 chars (mode of mentions)."""
    from collections import Counter as _C
    years = [int(y) for y in _YEAR_RE.findall(text[:3000])]
    if not years: return None
    return _C(years).most_common(1)[0][0]


def _guess_authors(text: str, max_authors: int = 6) -> list[str]:
    """Best-effort: lines between title and abstract, with names sized
    2-4 words, no all-caps, no punctuation soup."""
    lines = [l.strip() for l in text.splitlines()[:80] if l.strip()]
    out: list[str] = []
    for line in lines[:20]:
        if "@" in line or "http" in line.lower(): continue
        # commas often separate authors on one line
        for cand in [c.strip() for c in line.split(",")]:
            words = cand.split()
            if 2 <= len(words) <= 4 and all(w[0:1].isupper() for w in words if w):
                out.append(cand[:80])
                if len(out) >= max_authors: return out
    return out


def _guess_abstract(text: str) -> str:
    """Find the abstract heading or use the first long paragraph."""
    lc = text.lower()
    idx = lc.find("abstract")
    if idx >= 0:
        chunk = text[idx + 8: idx + 8 + 1800].strip()
        # Cut at the next heading-like line (short uppercase or section number)
        cut = _re.search(r"\n\n[A-Z][a-z]*\.?\s*[Ii]ntroduction|\n\n\d+\.\s+[A-Z]", chunk)
        if cut: chunk = chunk[: cut.start()]
        return chunk.strip()[:1500]
    # Fallback — first paragraph of length 300+
    for para in text.split("\n\n"):
        p = para.strip()
        if len(p) >= 300: return p[:1500]
    return ""


@router.post("/papers/preview", response_model=PaperPreviewResponse)
async def papers_preview(
    file:           Optional[UploadFile] = File(None),
    pdf_source_url: Optional[str]        = Form(None),
):
    """Upload a PDF (multipart) OR pass `pdf_source_url`. Returns
    heuristic-extracted metadata + a preview_id that /ingest uses to
    pick up the cached text without re-uploading.

    No LLM call here — title / authors / doi / year guessed from
    regex + position. Fast (~100ms for a 30-page PDF).
    """
    from engine.research_store.red_lessons.paper_acquisition import extract_pdf_text
    import urllib.request

    pdf_bytes: bytes
    src_kind: str
    src_url: str
    if file is not None:
        pdf_bytes = await file.read()
        src_kind  = "upload"
        src_url   = file.filename or ""
    elif pdf_source_url:
        try:
            req = urllib.request.Request(pdf_source_url, headers={"User-Agent": "MacroAlphaPro/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                pdf_bytes = r.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to fetch URL: {e}")
        src_kind = "url"
        src_url  = pdf_source_url
    else:
        raise HTTPException(status_code=400, detail="provide either 'file' or 'pdf_source_url'")

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty PDF")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="not a PDF (missing magic bytes)")

    text = extract_pdf_text(pdf_bytes)
    if not text:
        raise HTTPException(status_code=422,
            detail="pymupdf failed to extract text (scanned-image PDF?)")

    n_pages = text.count("\f") + 1   # pymupdf inserts form-feed per page
    payload = {
        "src_kind":   src_kind,
        "src_url":    src_url,
        "text":       text,
        "pdf_bytes_len": len(pdf_bytes),
    }
    preview_id = _cache_put(payload)

    return PaperPreviewResponse(
        preview_id      = preview_id,
        title_guess     = _guess_title(text),
        authors_guess   = _guess_authors(text),
        year_guess      = _guess_year(text),
        doi_guess       = _guess_doi(text),
        venue_guess     = "",
        abstract_guess  = _guess_abstract(text),
        n_pages         = n_pages,
        text_chars      = len(text),
        text_preview    = text[:600].strip(),
        extraction_note = f"preview_id valid for {_PDF_CACHE_TTL_SEC // 60} minutes",
    )


class PaperPreviewFromTextRequest(BaseModel):
    """Body for /papers/preview-from-text — manual escape hatch for
    scanned-image PDFs that pymupdf can't extract from. User copy-pastes
    the text from a PDF reader / browser / wherever and we treat it the
    same as if pymupdf had extracted it.
    """
    text:           str
    src_url:        str = ""
    src_label:      str = ""   # free-text source label ("colleague's blog post", etc.)


@router.post("/papers/preview-from-text", response_model=PaperPreviewResponse)
def papers_preview_from_text(req: PaperPreviewFromTextRequest):
    """Manual-paste path for PDFs pymupdf can't extract (scanned images,
    DRM-locked, image-only). User pastes the text directly; we cache it
    under a preview_id and the rest of the ingest chain (chunker,
    hypothesis extractor, spec extractor) runs unchanged.

    Minimum useful text: 200 chars (otherwise the chunker produces
    nothing and the hypothesis extractor has no signal). No PDF magic-
    byte check since there's no PDF.
    """
    text = (req.text or "").strip()
    if len(text) < 200:
        raise HTTPException(status_code=400,
            detail=f"text too short ({len(text)} chars); paste at least 200 chars")
    payload = {
        "src_kind":      "paste_text",
        "src_url":       req.src_url or req.src_label or "manual paste",
        "text":          text,
        "pdf_bytes_len": 0,
    }
    preview_id = _cache_put(payload)
    # Same metadata guessing as the PDF path — works on any text body
    n_pages = max(1, len(text) // 3000)   # rough heuristic; no \f markers in paste
    return PaperPreviewResponse(
        preview_id      = preview_id,
        title_guess     = _guess_title(text),
        authors_guess   = _guess_authors(text),
        year_guess      = _guess_year(text),
        doi_guess       = _guess_doi(text),
        venue_guess     = "",
        abstract_guess  = _guess_abstract(text),
        n_pages         = n_pages,
        text_chars      = len(text),
        text_preview    = text[:600].strip(),
        extraction_note = (f"manual paste — preview_id valid for "
                            f"{_PDF_CACHE_TTL_SEC // 60} min"),
    )


@router.post("/papers/ingest", response_model=PaperIngestResponse)
def papers_ingest(req: PaperIngestRequest):
    """Commit a paper into the registry + extract hypotheses.

    Runs SYNCHRONOUSLY because the hypothesis extractor uses Sonnet 4.6
    and we want loud failures (no orphan partial ingests). Typical
    runtime: 20-60s depending on paper length.
    """
    from engine.research_store.papers.schema import (
        PaperRegistryEntry, FulltextStatus, Shelf,
    )
    from engine.research_store.papers.store import save_entry, find_by_doi, load_registry
    from engine.research_store.red_lessons.papers_chroma import chunk_paper

    cached = _cache_get(req.preview_id)
    if not cached:
        raise HTTPException(status_code=404,
            detail=f"preview_id expired or unknown — re-upload the PDF")

    if not req.shelves:
        raise HTTPException(status_code=400, detail="at least one shelf is required")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="title is required")
    # 2026-06-06: abstract is now required at ingest time. The hypothesis
    # extractor (Sonnet 4.6) reads the abstract to produce candidate
    # claims; an empty abstract makes the entire T7 chain a no-op. Old
    # entries with empty abstracts remain valid at-rest (validation
    # runs at save not load) but new ingests must provide one.
    if not req.abstract.strip():
        raise HTTPException(status_code=400,
            detail="abstract is required — the hypothesis extractor needs it")

    # ─── Dedupe by DOI ───
    existing = find_by_doi(req.doi, load_registry()) if req.doi else None
    if existing:
        raise HTTPException(status_code=409,
            detail=f"DOI {req.doi!r} already in registry as paper_id={existing.paper_id}")

    # F11.1 (2026-06-05): arXiv / SSRN / preprint papers often have no
    # DOI. chunk_paper hard-requires a doi (used as chunk id prefix).
    # Synthesize one so non-DOI papers can ingest:
    #   arXiv URL  → "arxiv:<id>"        (stable, dedupe-friendly)
    #   else       → "paper:<8 hex>"     (random; uniqueness only)
    # The synthesized doi is ONLY used as a chunk-id prefix; req.doi
    # stays empty in the registry entry so the UI shows "no DOI".
    chunk_doi = req.doi
    if not chunk_doi:
        import re as _re
        src_url = cached.get("src_url", "") or req.pdf_source_url or ""
        m = _re.search(r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5})", src_url, _re.I)
        if m:
            chunk_doi = f"arxiv:{m.group(1)}"
        else:
            chunk_doi = f"paper:{_uuid.uuid4().hex[:8]}"

    # ─── Chunk the full text ───
    chunks = chunk_paper(
        cached["text"],
        doi      = chunk_doi,
        title    = req.title,
        year     = req.year,
        authors  = tuple(req.authors),
        venue    = req.venue,
        source_kind = cached["src_kind"],
    )

    # ─── Create registry entry ───
    paper_id = _uuid.uuid4().hex
    now_iso  = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())

    # 2026-06-06: derive shelf from user-picked intent on manual
    # ingest. /incoming → "Ingest now" already passes ?shelf= via the
    # JS classifier (filter.category_guess + recommended_action), so
    # this only kicks in when no upstream classifier ran. Mapping:
    #   methodology_borrow / address_decay → doctrine_method
    #   improve_existing_sleeve             → green_critique
    #   challenge_doctrine                  → red_critique
    #   expand_breadth                      → other (tentative, verdict
    #                                                refines)
    #   fact_check / curiosity / author_trust / other → other
    def _shelf_from_intent(intent_val: str) -> "Optional[str]":
        if intent_val in ("methodology_borrow", "address_decay"):
            return "doctrine_method"
        if intent_val == "improve_existing_sleeve":
            return "green_critique"
        if intent_val == "challenge_doctrine":
            return "red_critique"
        return None    # expand_breadth / other / etc — leave default

    # Phase 1.7 step 3 (2026-06-06): construct IngestionReason if the
    # caller supplied one. Save if EITHER free_text non-empty OR a
    # specific intent_category was picked (2026-06-06 simplification:
    # the intent dropdown alone is signal worth preserving).
    from engine.research_store.papers.schema import (
        IngestionReason as _IngestionReason,
        IngestionReasonSource as _IRSource,
        IntentCategory as _IntentCategory,
    )
    from engine.research_store.papers.shelves import Shelf as _Shelf
    ingestion_reason_obj = None
    if req.ingestion_reason:
        free_text = req.ingestion_reason.free_text.strip()
        intent_raw = (req.ingestion_reason.intent_category or "").strip()
        intent_cat: Optional[_IntentCategory] = None
        if intent_raw:
            try:
                intent_cat = _IntentCategory(intent_raw)
            except ValueError:
                intent_cat = _IntentCategory.OTHER
        # Persist iff there's something worth persisting
        if free_text or (intent_cat and intent_cat != _IntentCategory.OTHER):
            try:
                src = _IRSource(req.ingestion_reason.source)
            except ValueError:
                src = _IRSource.USER
            ingestion_reason_obj = _IngestionReason(
                free_text       = free_text[:200],
                intent_category = intent_cat,
                source          = src,
                user_ts         = now_iso,
            )

    # 2026-06-06 manual-ingest shelf auto-classify. Only kicks in when
    # the caller didn't already pre-classify (req.shelves is the default
    # ["other"]) AND the user picked an intent that maps to a non-OTHER
    # shelf. /incoming "Ingest now" already mapped category_guess +
    # recommended_action client-side, so this NEVER overrides an
    # already-classified shelf.
    effective_shelves = list(req.shelves)
    effective_shelf_notes = dict(req.shelf_notes or {})
    if effective_shelves == ["other"] and ingestion_reason_obj is not None:
        ic = ingestion_reason_obj.intent_category
        derived = _shelf_from_intent(ic.value) if ic else None
        if derived:
            effective_shelves = [derived]
            # Preserve any user-written rationale; otherwise stamp it
            # with the intent label so the shelf entry has provenance.
            existing_note = effective_shelf_notes.get("other") \
                            or effective_shelf_notes.get(derived) or ""
            effective_shelf_notes = {derived: existing_note or f"intent={ic.value}"}

    entry = PaperRegistryEntry(
        paper_id              = paper_id,
        version               = 1,
        parent_paper_id       = None,
        schema_version        = 2,    # Phase 1.7 — see schema module
        doi                   = req.doi or "",
        title                 = req.title.strip(),
        year                  = req.year,
        authors               = tuple(req.authors),
        venue                 = req.venue,
        abstract              = req.abstract or "",
        fulltext_status       = FulltextStatus.INGESTED if chunks else FulltextStatus.METADATA_ONLY,
        pdf_source_kind       = cached["src_kind"],
        pdf_source_url        = req.pdf_source_url or cached.get("src_url", ""),
        n_chunks              = len(chunks),
        ingested_ts           = now_iso,
        referenced_by_lessons = (),
        referenced_by_factors = (),
        referenced_by_sleeves = (),
        referenced_by_doctrines = (),
        shelves               = tuple(
            Shelf(s) if isinstance(s, str) else s for s in effective_shelves
        ),
        shelf_notes           = {
            (Shelf(k) if isinstance(k, str) else k): v
            for k, v in (effective_shelf_notes or {}).items()
        },
        created_ts            = now_iso,
        updated_ts            = now_iso,
        created_by            = "ui:/research/papers/new",
        tags                  = ("ui_ingested",),
        note                  = req.note,
        ingestion_reason      = ingestion_reason_obj,
    )
    save_entry(entry)

    # ─── Index chunks in ChromaDB if available ───
    n_chunks_indexed = 0
    try:
        from engine.research_store.red_lessons.papers_chroma import ingest_chunks
        n_chunks_indexed = ingest_chunks(chunks)
    except Exception as exc:
        # Non-fatal — registry succeeded; chroma indexing can be done later.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "chroma indexing failed for paper_id=%s — %s", paper_id, exc)

    # ─── Run hypothesis extractor on the chunks ───
    # F11.2 (2026-06-05): correct call signature. Prior version passed
    # paper_id/paper_title kwargs that don't exist + treated return as a
    # list of Hypothesis (it's actually ExtractorResult with
    # .candidates: tuple[HypothesisCandidate, ...]). Mirrors the working
    # pattern in scripts/extract_paper_hypotheses.py:163-215.
    n_hypotheses = 0
    extracted_hyps: list = []
    try:
        from engine.agents.hypothesis_extractor.extractor import (
            extract_hypotheses_from_chunks,
        )
        from engine.research_store.hypothesis.store import save_hypothesis
        from engine.research_store.hypothesis.schema import (
            Hypothesis, VerbatimQuote, MechanismFamily,
            HypothesisDirection, ExtractionMethod, HypothesisReviewState,
        )
        import datetime as _dt
        import dataclasses as _dc

        paper_metadata = {
            "title":    req.title,
            "authors":  list(req.authors),
            "year":     req.year,
            "venue":    req.venue,
            "doi":      chunk_doi,
            "paper_id": paper_id,
        }
        # extractor expects list[dict]; chunks from chunk_paper are
        # PaperChunk dataclasses, convert to dicts
        chunk_dicts = [_dc.asdict(c) for c in chunks]

        result = extract_hypotheses_from_chunks(
            paper_metadata = paper_metadata,
            chunks         = chunk_dicts,
        )

        def _cand_to_hyp(cand):
            now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            quotes = tuple(
                VerbatimQuote(
                    chunk_id       = q["chunk_id"],
                    quote_text     = q["quote_text"],
                    section_ref    = q.get("section_ref", ""),
                    relevance_note = q.get("relevance_note", ""),
                )
                for q in cand.verbatim_quotes
            )
            return Hypothesis(
                hypothesis_id        = Hypothesis.new_id(),
                source_paper_id      = paper_id,
                version              = 1,
                parent_hypothesis_id = None,
                source_chunk_ids     = cand.source_chunk_ids,
                verbatim_quotes      = quotes,
                claim                = cand.claim,
                mechanism_family     = MechanismFamily(cand.mechanism_family),
                mechanism_subtype    = cand.mechanism_subtype,
                predicted_direction  = HypothesisDirection(cand.predicted_direction),
                predicted_magnitude  = cand.predicted_magnitude,
                required_data        = cand.required_data,
                test_methodology     = cand.test_methodology,
                extraction_method    = ExtractionMethod.LLM_EXTRACT,
                review_state         = HypothesisReviewState.PROPOSED,
                created_ts           = now_iso,
                updated_ts           = now_iso,
                created_by           = "ui:/research/papers/new",
                tags                 = ("ui_ingested",),
            )

        for cand in result.candidates:
            try:
                hyp = _cand_to_hyp(cand)
                save_hypothesis(hyp, validate_strict=False, skip_cross_checks=True)
                n_hypotheses += 1
                extracted_hyps.append(hyp)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "save_hypothesis failed for paper_id=%s — %s", paper_id, exc)
    except Exception as exc:
        # Non-fatal — registry + chunks succeeded.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "hypothesis extraction failed for paper_id=%s — %s", paper_id, exc)

    # F11 (2026-06-05): close the hypothesis → typed-spec wiring gap.
    # Pre-F11, ingest stopped at the NL hypothesis layer; converting
    # claims into typed HypothesisSpec (B.2) required Claude to run
    # scripts/backfill_hypothesis_specs.py manually. Every new paper's
    # hypotheses landed STALE until the next manual backfill — which is
    # why F1 inventory showed 0/88 composer-ready.
    #
    # With F11, every successfully-saved hypothesis immediately gets a
    # typed spec via engine.hypothesis_spec.extract_spec (one Anthropic
    # call, ~$0.005, ~3s each). Failures are non-fatal: the hypothesis
    # is preserved so a future re-run can spec it.
    #
    # Cost note: a paper with 5 hypotheses adds ~$0.025 + ~15s to
    # ingest. Acceptable for self-serve UX.
    n_specs = 0
    try:
        from engine.hypothesis_spec.extractor import extract_spec
        from engine.hypothesis_spec.store import append as append_spec
        from engine.research_store.manifest import current_git_sha
        git_sha = current_git_sha() or ""
        for h in extracted_hyps:
            try:
                spec = extract_spec(
                    source_hypothesis_id = h.hypothesis_id,
                    claim_text           = h.claim,
                    mechanism_family     = h.mechanism_family.value,
                    mechanism_subtype    = h.mechanism_subtype,
                    git_sha              = git_sha,
                )
                if spec is not None:
                    append_spec(spec)
                    n_specs += 1
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "F11 extract_spec failed for hyp=%s — %s",
                    h.hypothesis_id, exc)
    except Exception as exc:
        # Non-fatal — hypotheses already saved; spec extraction can be
        # re-run via the backfill script if it failed broadly.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "F11 spec extraction batch failed for paper_id=%s — %s",
            paper_id, exc)

    # Drop the cached PDF text — no longer needed.
    _PDF_CACHE.pop(req.preview_id, None)

    return PaperIngestResponse(
        paper_id      = paper_id,
        title         = req.title,
        n_chunks      = len(chunks),
        n_hypotheses  = n_hypotheses,
        n_specs       = n_specs,   # F11: typed specs auto-extracted
        registry_path = "data/research_store/papers_registry.jsonl",
        # 2026-06-06: use query-param URL so the post-ingest "Open paper
        # page" button works for newly-ingested IDs. The static export
        # only pre-renders [paper_id] slugs that existed at build time;
        # /view?id= sidesteps that because the route shape is static and
        # the variable part lives in the query string read client-side.
        next_url      = f"/research/papers/view?id={paper_id}",
    )


# ─────────────────────── DASHBOARD overview ───────────────────────────


@router.get("/chain-flow")
def chain_flow():
    """Sankey-shaped flow data for visualizing the
    PAPER → HYPOTHESIS → TEST → VERDICT chain.

    Stages:
      1. Papers grouped by primary shelf (doctrine_method, green_*,
         yellow_*, red_*, other) — colored by relevance.
      2. Hypotheses extracted from each paper (joined via
         source_paper_id), bucketed by mechanism family — shows
         how each shelf feeds which mechanism areas.
      3. Tested vs. untested status — the gate where extraction
         meets actual testing effort.
      4. Verdict outcome (RED / GREEN / MARGINAL / pending) — only
         the tested branch flows here; untested stays as a leaf.

    Returns echarts-compatible nodes + links arrays.
    """
    st = _load_state()
    tested_ids = _tested_set(st["lessons"], include_legacy=True)

    # ---- helpers ----
    _SHELF_PRIORITY = [
        "doctrine_method",
        "green_motivation",
        "green_critique",
        "yellow_motivation",
        "dormant_revisit",
        "red_critique",
        "red_motivation",
        "other",
    ]
    def primary_shelf(p) -> str:
        shelves = p.shelves or ["other"]
        for s in _SHELF_PRIORITY:
            if s in shelves:
                return s
        return "other"

    # ---- stage 1: papers by shelf ----
    paper_shelf: dict[str, str] = {p.paper_id: primary_shelf(p) for p in st["registry"]}
    shelf_counts: Counter = Counter(paper_shelf.values())

    # ---- stage 2: hypotheses by family, joined back to shelf ----
    # hyps_by_paper is dict[paper_id, list[Hypothesis]]
    family_counts: Counter = Counter()
    shelf_to_family: Counter = Counter()  # (shelf, family) -> n
    paper_hyp_count: dict[str, int] = {}
    for pid, hyps in st["hyps_by_paper"].items():
        paper_hyp_count[pid] = len(hyps)
        sh = paper_shelf.get(pid, "other")
        for h in hyps:
            fam = h.mechanism_family.value
            family_counts[fam] += 1
            shelf_to_family[(sh, fam)] += 1

    # ---- stage 3: tested / untested per family ----
    fam_tested: Counter = Counter()
    fam_untested: Counter = Counter()
    for pid, hyps in st["hyps_by_paper"].items():
        for h in hyps:
            fam = h.mechanism_family.value
            if h.hypothesis_id in tested_ids:
                fam_tested[fam] += 1
            else:
                fam_untested[fam] += 1

    # ---- stage 4: verdict for tested hypotheses ----
    # Map hypothesis_id -> verdict label via lessons
    hyp_verdict: dict[str, str] = {}
    for L in st["lessons"]:
        verdict_label = _bucket_verdict(L.verdict)
        for hid in (L.tested_hypothesis_ids or []):
            # Keep first seen (newest lesson wins via store sort)
            hyp_verdict.setdefault(hid, verdict_label)

    fam_to_verdict: Counter = Counter()  # (family, verdict) -> n
    for pid, hyps in st["hyps_by_paper"].items():
        for h in hyps:
            if h.hypothesis_id in tested_ids:
                v = hyp_verdict.get(h.hypothesis_id, "PENDING")
                fam_to_verdict[(h.mechanism_family.value, v)] += 1

    # ---- assemble echarts payload ----
    # Node name format: "{stage}:{label}" so duplicates across stages
    # don't collide. Display name strips the stage prefix.
    nodes: list[dict] = []
    seen_names: set[str] = set()

    def add_node(name: str, depth: int, category: str | None = None):
        if name in seen_names: return
        seen_names.add(name)
        node = {"name": name, "depth": depth}
        if category: node["category"] = category
        nodes.append(node)

    # Stage 1 root
    add_node("PAPERS", 0, "papers")

    # Stage 1.5: papers split by shelf
    for shelf in _SHELF_PRIORITY:
        if shelf_counts.get(shelf, 0) > 0:
            add_node(f"shelf:{shelf}", 1, "shelf")

    # Stage 2: families
    for fam, _ in family_counts.most_common():
        add_node(f"family:{fam}", 2, "family")

    # Stage 3: tested / untested
    add_node("UNTESTED", 3, "status")
    add_node("TESTED",   3, "status")

    # Stage 4: verdicts
    for v in ("RED", "GREEN", "MARGINAL", "PENDING"):
        if any(k == v for _, k in fam_to_verdict):
            add_node(f"verdict:{v}", 4, "verdict")

    links: list[dict] = []
    # Stage 0 -> 1: PAPERS -> shelves
    for shelf, n in shelf_counts.items():
        links.append({"source": "PAPERS", "target": f"shelf:{shelf}", "value": n})

    # Stage 1 -> 2: shelves -> families (via hypotheses)
    for (shelf, fam), n in shelf_to_family.items():
        if shelf_counts.get(shelf, 0) == 0: continue
        links.append({"source": f"shelf:{shelf}", "target": f"family:{fam}", "value": n})

    # Stage 2 -> 3: families -> tested / untested
    for fam, n in fam_untested.items():
        if n > 0:
            links.append({"source": f"family:{fam}", "target": "UNTESTED", "value": n})
    for fam, n in fam_tested.items():
        if n > 0:
            links.append({"source": f"family:{fam}", "target": "TESTED", "value": n})

    # Stage 3 -> 4: tested -> verdicts (aggregated across families)
    verdict_totals: Counter = Counter()
    for (fam, v), n in fam_to_verdict.items():
        verdict_totals[v] += n
    for v, n in verdict_totals.items():
        if n > 0:
            links.append({"source": "TESTED", "target": f"verdict:{v}", "value": n})

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "n_papers":         len(st["registry"]),
            "n_papers_with_hyps": len(st["hyps_by_paper"]),
            "n_hyps":           sum(family_counts.values()),
            "n_tested":         sum(fam_tested.values()),
            "n_untested":       sum(fam_untested.values()),
            "verdicts":         dict(verdict_totals),
        },
    }


def _bucket_verdict(verdict_str: str) -> str:
    """Reduce verbose verdicts to RED / GREEN / MARGINAL / PENDING buckets."""
    v = (verdict_str or "").upper()
    if "GREEN" in v:    return "GREEN"
    if "RED" in v:      return "RED"
    if "MARGINAL" in v or "YELLOW" in v: return "MARGINAL"
    return "PENDING"


@router.get("/index_stats")
def index_stats():
    """R4.3 — health stats from the SQLite read index.

    Fast (~ms): SELECT COUNT(*) per table. Returns the same shape
    as /overview's row counts, plus index freshness so the UI can
    show "index rebuilt N seconds ago"."""
    from engine.research_store._index import stats as _idx_stats
    return _idx_stats()


@router.get("/overview")
def dashboard_overview():
    """Single-call dashboard summary stats."""
    st = _load_state()
    tested = _tested_set(st["lessons"], include_legacy=False)

    # Registry stats
    fulltext_counts = Counter(e.fulltext_status.value for e in st["registry"])

    # Hypothesis stats
    n_hyps = sum(len(v) for v in st["hyps_by_paper"].values())
    family_counts = Counter()
    for hyps in st["hyps_by_paper"].values():
        for h in hyps:
            family_counts[h.mechanism_family.value] += 1

    # Lesson stats
    grounding_counts = Counter(L.grounding_method.value for L in st["lessons"])

    return {
        "papers": {
            "total":            len(st["registry"]),
            "by_status":        dict(fulltext_counts),
            "with_hypotheses":  len(st["hyps_by_paper"]),
        },
        "hypotheses": {
            "total":            n_hyps,
            "tested":           len(tested),
            "untested":         n_hyps - len(tested),
            "by_family":        dict(family_counts),
        },
        "lessons": {
            "total":            len(st["lessons"]),
            "by_grounding":     dict(grounding_counts),
        },
    }
