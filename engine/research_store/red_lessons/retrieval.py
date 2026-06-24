"""engine.research_store.red_lessons.retrieval — 3-layer retrieval API.

When a new candidate X is proposed, we want structured access to:

  Layer 1 — same mechanism family ∩ same failure mode      → hard STOP
  Layer 2 — same mechanism family ≠ failure mode           → reconsider
  Layer 3 — cross-family ∩ same failure mode               → structural
  Dormant — dormant_revisits whose conditions now apply    → reactivation

Plus paper-side queries:

  - filter by Shelf
  - filter by mechanism family / failure mode
  - semantic search over ingested full-text via ChromaDB

Honest scope:

  - We do NOT have a structured `data_window` field on lessons. Layer 1
    queries don't filter by data window; caller must read the
    `stat_evidence.n_months` / candidate_name hint and decide.
  - ChromaDB semantic search is OPTIONAL — papers_chroma may be empty
    or unreachable; the API returns empty results without raising.
  - `X.likely_failure_modes` defaults to "all failure modes ever observed
    in X.family" when the caller doesn't specify — captures the "this
    family's known weak points" heuristic.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
from typing import Any

from engine.research_store.red_lessons.failure_modes import FailureMode
from engine.research_store.red_lessons.mechanism_families import MechanismFamily
from engine.research_store.red_lessons.schema import REDLesson
from engine.research_store.red_lessons.store import (
    latest_per_candidate, load_lessons,
)
from engine.research_store.papers import (
    PaperRegistryEntry, Shelf, load_registry, latest_per_doi,
)

logger = logging.getLogger(__name__)


# ─────────────────────── candidate input ──────────────────────────────


@_dc.dataclass(frozen=True)
class Candidate:
    """A proposal under evaluation.

    name:                 short identifier (used for logging only)
    mechanism_family:     MechanismFamily (required)
    mechanism_subtype:    free-form refinement (optional)
    likely_failure_modes: caller-asserted suspected failure modes; if empty,
                          retrieval infers from "all failure modes observed
                          in this family historically"
    description:          free text — used for semantic paper search
    """
    name:                 str
    mechanism_family:     MechanismFamily
    mechanism_subtype:    str = ""
    likely_failure_modes: tuple[FailureMode, ...] = ()
    description:          str = ""


@_dc.dataclass(frozen=True)
class LessonHit:
    """A retrieval match. `relevance_reason` explains WHY this matched."""
    lesson:           REDLesson
    layer:            str               # "L1" | "L2" | "L3" | "dormant"
    relevance_reason: str
    score:            float             # 0..1, higher = more relevant


@_dc.dataclass(frozen=True)
class PaperHit:
    """A paper-registry match."""
    entry:            PaperRegistryEntry
    shelves_matched:  tuple[Shelf, ...]
    relevance_reason: str


@_dc.dataclass(frozen=True)
class RetrievalBriefing:
    """Structured 3-layer briefing for a candidate.

    Always-non-null; layers with no matches have empty tuples.
    """
    candidate:           Candidate
    layer1_hard_stop:    tuple[LessonHit, ...]
    layer2_reconsider:   tuple[LessonHit, ...]
    layer3_structural:   tuple[LessonHit, ...]
    dormant_reactivate:  tuple[LessonHit, ...]
    related_papers:      tuple[PaperHit, ...]
    note:                str = ""

    def total_hits(self) -> int:
        return (len(self.layer1_hard_stop) + len(self.layer2_reconsider)
                + len(self.layer3_structural) + len(self.dormant_reactivate))


# ─────────────────────── helper: infer likely failure modes ────────────


def infer_failure_modes_from_family(
    family: MechanismFamily,
    lessons: list[REDLesson],
) -> tuple[FailureMode, ...]:
    """Union of all failure modes seen in this family's historical lessons."""
    seen: set[FailureMode] = set()
    for L in lessons:
        if L.mechanism_family == family:
            for fm in L.failure_modes:
                seen.add(fm)
    # Stable order
    return tuple(sorted(seen, key=lambda f: f.value))


# ─────────────────────── strength score helper ────────────────────────


def _lesson_strength_score(lesson: REDLesson) -> float:
    """0..1 score: stat-evidence strength + recency proxy.

    Higher = more reason to trust this lesson's verdict.
    """
    score = 0.0
    se = lesson.stat_evidence or {}
    dsr = se.get("deflated_sr") or se.get("net_deflated_sr")
    if isinstance(dsr, (int, float)):
        # very LOW DSR = strong RED evidence (clear fail)
        score += min(0.4, max(0.0, (0.9 - dsr) * 0.5))
    n_obs = se.get("n_months") or se.get("n_obs")
    if isinstance(n_obs, (int, float)):
        # > 60 months = adequate power
        score += min(0.3, n_obs / 200.0)
    # Paper anchor present is positive evidence
    if lesson.paper_motivation is not None:
        score += 0.15
    # Stronger if review_state advanced
    rs = (lesson.review_state.value if lesson.review_state else "")
    if rs in ("human_reviewed", "locked"):
        score += 0.15
    return min(1.0, score)


# ─────────────────────── the 3 layers ─────────────────────────────────


def query_layer1_hard_stop(
    candidate: Candidate,
    lessons:   list[REDLesson],
) -> list[LessonHit]:
    """L1: same family AND any failure-mode overlap.

    These are "the same pattern has already been tried and failed via the
    same failure mode" — caller should treat as hard STOP unless evidence
    is presented that X genuinely avoids the failure mode.
    """
    if not candidate.likely_failure_modes:
        likely = infer_failure_modes_from_family(candidate.mechanism_family, lessons)
    else:
        likely = candidate.likely_failure_modes
    likely_set = set(likely)
    hits: list[LessonHit] = []
    for L in lessons:
        if L.mechanism_family != candidate.mechanism_family:
            continue
        overlap = likely_set & set(L.failure_modes)
        if not overlap:
            continue
        score = _lesson_strength_score(L)
        reason = (
            f"Same mechanism_family ({candidate.mechanism_family.value}); "
            f"failure_modes overlap: {sorted(m.value for m in overlap)}"
        )
        hits.append(LessonHit(L, "L1", reason, score))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def query_layer2_reconsider(
    candidate: Candidate,
    lessons:   list[REDLesson],
) -> list[LessonHit]:
    """L2: same family but DIFFERENT failure modes.

    The family has known REDs but X's likely failure modes don't overlap.
    Worth reconsidering whether X really avoids the family's other failures.
    """
    if not candidate.likely_failure_modes:
        likely = infer_failure_modes_from_family(candidate.mechanism_family, lessons)
    else:
        likely = candidate.likely_failure_modes
    likely_set = set(likely)
    hits: list[LessonHit] = []
    for L in lessons:
        if L.mechanism_family != candidate.mechanism_family:
            continue
        if likely_set & set(L.failure_modes):
            continue   # overlap → goes to L1, not L2
        if not L.failure_modes:
            continue
        score = _lesson_strength_score(L)
        reason = (
            f"Same mechanism_family ({candidate.mechanism_family.value}); "
            f"DIFFERENT failure_modes: {[m.value for m in L.failure_modes]} — "
            f"does X actually avoid these?"
        )
        hits.append(LessonHit(L, "L2", reason, score))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def query_layer3_structural(
    candidate: Candidate,
    lessons:   list[REDLesson],
) -> list[LessonHit]:
    """L3: cross-family BUT same failure mode.

    The failure pattern is structural — recurs across families. X's
    candidate having the same failure mode means it's at structural risk.
    """
    if not candidate.likely_failure_modes:
        # For L3, must have a target failure mode to compare; if caller
        # didn't supply one, infer from candidate's own family
        likely = infer_failure_modes_from_family(candidate.mechanism_family, lessons)
    else:
        likely = candidate.likely_failure_modes
    likely_set = set(likely)
    hits: list[LessonHit] = []
    for L in lessons:
        if L.mechanism_family == candidate.mechanism_family:
            continue   # cross-family ONLY
        overlap = likely_set & set(L.failure_modes)
        if not overlap:
            continue
        score = _lesson_strength_score(L)
        reason = (
            f"Cross-family ({L.mechanism_family.value}) but same "
            f"failure mode(s): {sorted(m.value for m in overlap)} — "
            f"structural risk pattern."
        )
        hits.append(LessonHit(L, "L3", reason, score))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def query_dormant_reactivations(
    candidate: Candidate,
    lessons:   list[REDLesson],
    available_data_signals: tuple[str, ...] = (),
) -> list[LessonHit]:
    """Dormant: lessons with dormant_revisits whose conditions now
    apply.

    `available_data_signals` is a free-form list of strings that the caller
    knows about — e.g. ("OptionMetrics extends pre-1990", "CN A-share PIT
    Wikipedia"). We match by substring on the condition_label / check.
    """
    sigs_lower = [s.lower() for s in available_data_signals]
    hits: list[LessonHit] = []
    for L in lessons:
        for dr in L.dormant_revisits:
            hay = (dr.condition_label + " " + dr.condition_check).lower()
            if any(s in hay for s in sigs_lower):
                score = _lesson_strength_score(L) * 0.7
                reason = (
                    f"Dormant revisit unlocked by signal '"
                    f"{dr.condition_label}': {dr.reactivation_note[:120]}"
                )
                hits.append(LessonHit(L, "dormant", reason, score))
                break  # one trigger per lesson is enough
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# ─────────────────────── hypothesis-side queries (T4) ────────────────


def query_lessons_for_hypothesis(
    hypothesis_id: str,
    lessons: list[REDLesson] | None = None,
    *,
    include_legacy: bool = False,
) -> list[REDLesson]:
    """Return lessons whose tested_hypothesis_ids contains hypothesis_id.

    Default: exclude legacy lessons (pretrain_grounded can't have
    structurally tested a hypothesis — those references are never real).
    """
    from engine.research_store.red_lessons.schema import GroundingMethod

    if lessons is None:
        lessons = list(latest_per_candidate(load_lessons()).values())
    out = []
    for L in lessons:
        if not include_legacy and L.grounding_method == GroundingMethod.pretrain_grounded:
            continue
        if hypothesis_id in L.tested_hypothesis_ids:
            out.append(L)
    return out


def tested_hypothesis_ids(
    lessons: list[REDLesson] | None = None,
    *,
    include_legacy: bool = False,
) -> set[str]:
    """Union of hypothesis_ids ever tested by any non-legacy lesson.

    Used by P5 forward-vector discovery: untested hypotheses (in the
    hypotheses store but NOT in this set) are candidates for new tests.
    """
    from engine.research_store.red_lessons.schema import GroundingMethod

    if lessons is None:
        lessons = list(latest_per_candidate(load_lessons()).values())
    out: set[str] = set()
    for L in lessons:
        if not include_legacy and L.grounding_method == GroundingMethod.pretrain_grounded:
            continue
        out.update(L.tested_hypothesis_ids)
    return out


@_dc.dataclass(frozen=True)
class PaperHypothesisStatus:
    """For a given paper, status of each Hypothesis derived from it."""
    paper_id:          str
    paper_title:       str
    hypothesis_id:     str
    hypothesis_claim:  str
    tested_by_lessons: tuple[str, ...]      # lesson_ids that cite this hypothesis_id
    is_untested:       bool


def query_paper_hypothesis_status(
    paper_id: str,
    *,
    lessons: list[REDLesson] | None = None,
    include_legacy: bool = False,
) -> list[PaperHypothesisStatus]:
    """Given a paper, return its hypotheses + which lessons tested each.

    Lazy-imports hypothesis store to avoid cycles.
    """
    from engine.research_store.hypothesis import (
        latest_per_paper, load_hypotheses,
    )
    from engine.research_store.papers import (
        find_by_doi, load_registry, latest_per_doi,
    )

    if lessons is None:
        lessons = list(latest_per_candidate(load_lessons()).values())

    # Find paper title
    reg = list(latest_per_doi(load_registry()).values())
    paper = next((e for e in reg if e.paper_id == paper_id), None)
    paper_title = paper.title if paper else "(unknown)"

    # Find hypotheses for this paper
    hyps = load_hypotheses()
    by_paper = latest_per_paper(hyps)
    paper_hyps = by_paper.get(paper_id, [])

    # For each hypothesis, find lessons that tested it
    out: list[PaperHypothesisStatus] = []
    for h in paper_hyps:
        tested_by = query_lessons_for_hypothesis(
            h.hypothesis_id, lessons=lessons, include_legacy=include_legacy,
        )
        out.append(PaperHypothesisStatus(
            paper_id          = paper_id,
            paper_title       = paper_title,
            hypothesis_id     = h.hypothesis_id,
            hypothesis_claim  = h.claim,
            tested_by_lessons = tuple(L.lesson_id for L in tested_by),
            is_untested       = len(tested_by) == 0,
        ))
    return out


# ─────────────────────── paper-side queries ───────────────────────────


def query_papers_by_shelf(
    shelves: tuple[Shelf, ...],
    registry: list[PaperRegistryEntry],
) -> list[PaperHit]:
    """Return papers carrying ANY of the requested shelves (multi-label OR)."""
    want = set(shelves)
    hits: list[PaperHit] = []
    for e in registry:
        matched = want & set(e.shelves)
        if matched:
            reason = f"shelves matched: {sorted(s.value for s in matched)}"
            hits.append(PaperHit(e, tuple(sorted(matched, key=lambda s: s.value)),
                                 reason))
    return hits


def query_papers_by_factor(
    factor_name: str,
    registry: list[PaperRegistryEntry],
) -> list[PaperHit]:
    """Find papers referenced_by_factor == factor_name (or sleeve)."""
    hits: list[PaperHit] = []
    for e in registry:
        match = (
            factor_name in e.referenced_by_factors
            or factor_name in e.referenced_by_sleeves
        )
        if match:
            reason = f"factor/sleeve name found in registry's reverse links"
            hits.append(PaperHit(e, e.shelves, reason))
    return hits


def query_papers_semantic(
    query_text: str,
    top_k: int = 5,
    where_shelves: tuple[Shelf, ...] | None = None,
) -> list[dict[str, Any]]:
    """ChromaDB semantic search over ingested paper full-text.

    Returns raw hit dicts (not PaperHits) — caller can re-correlate with
    registry by DOI if desired.

    On empty collection / missing chromadb / errors: returns [].
    """
    try:
        from engine.research_store.red_lessons.papers_chroma import get_collection
        coll = get_collection()
        # Build optional where filter (chromadb expects dict)
        # Per-chunk metadata stores `candidate_names` etc but NOT shelves
        # (shelf is paper-level, not chunk-level). For shelf filtering, do
        # post-hoc by looking up DOI in registry.
        try:
            count = coll.count()
        except Exception:
            count = 0
        if count == 0:
            return []
        n = min(top_k, count)
        result = coll.query(query_texts=[query_text], n_results=n)
        # Unpack chromadb result format
        ids        = (result.get("ids") or [[]])[0]
        documents  = (result.get("documents") or [[]])[0]
        metadatas  = (result.get("metadatas") or [[]])[0]
        distances  = (result.get("distances") or [[]])[0]
        out = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            out.append({
                "chunk_id":  doc_id,
                "text":      documents[i] if i < len(documents) else "",
                "metadata":  meta,
                "distance":  distances[i] if i < len(distances) else None,
            })
        # Optional post-hoc shelf filter
        if where_shelves:
            registry = list(latest_per_doi(load_registry()).values())
            by_doi = {e.doi.lower(): e for e in registry if e.doi}
            allowed_shelves = set(where_shelves)
            filtered = []
            for hit in out:
                doi = (hit["metadata"].get("doi") or "").lower()
                entry = by_doi.get(doi)
                if entry and (set(entry.shelves) & allowed_shelves):
                    filtered.append(hit)
            return filtered
        return out
    except Exception as e:
        logger.info("semantic query failed (returning []): %s", e)
        return []


# ─────────────────────── top-level briefing ───────────────────────────


def get_briefing(
    candidate: Candidate,
    *,
    available_data_signals: tuple[str, ...] = (),
    lessons: list[REDLesson] | None = None,
    registry: list[PaperRegistryEntry] | None = None,
    top_papers: int = 10,
    include_legacy: bool = False,
) -> RetrievalBriefing:
    """Top-level: produce the full 3-layer + dormant + papers briefing.

    Args:
      include_legacy: by default (False), lessons with
                      grounding_method=pretrain_grounded are EXCLUDED
                      from retrieval. Per 2026-06-04 chain doctrine,
                      legacy 47 lessons are not authoritative; they're
                      in a separate /research/legacy surface.
                      Pass True to opt-in (audit / debugging only).

    Pass `lessons` / `registry` explicitly to avoid re-loading (useful for
    tests and batch evaluation). Otherwise loads from disk.
    """
    # Import lazily to avoid cycle with red_lessons package init
    from engine.research_store.red_lessons.schema import GroundingMethod

    if lessons is None:
        lessons = list(latest_per_candidate(load_lessons()).values())
    if not include_legacy:
        lessons = [L for L in lessons
                   if L.grounding_method != GroundingMethod.pretrain_grounded]
    if registry is None:
        registry = list(latest_per_doi(load_registry()).values())

    l1 = query_layer1_hard_stop(candidate, lessons)
    l2 = query_layer2_reconsider(candidate, lessons)
    l3 = query_layer3_structural(candidate, lessons)
    dr = query_dormant_reactivations(candidate, lessons, available_data_signals)

    # Papers: pull motivation / critique shelves matching the family
    # (heuristic — use red_motivation + green_motivation for the family of
    # interest). The actual shelf-by-family resolution requires walking the
    # registry's referenced_by_* fields; here we just surface top papers
    # by motivation shelf for any-family.
    related_shelves = (
        Shelf.RED_MOTIVATION,
        Shelf.GREEN_MOTIVATION,
        Shelf.DOCTRINE_METHOD,
    )
    paper_hits = query_papers_by_shelf(related_shelves, registry)
    # Sort: prefer entries with more shelves matched, then more lesson refs
    paper_hits.sort(
        key=lambda h: (len(h.shelves_matched), len(h.entry.referenced_by_lessons)),
        reverse=True,
    )
    paper_hits = paper_hits[:top_papers]

    note = ""
    if not candidate.likely_failure_modes:
        inferred = infer_failure_modes_from_family(candidate.mechanism_family, lessons)
        note = (
            f"Caller did not provide likely_failure_modes; inferred from "
            f"family historical lessons: {[m.value for m in inferred]}"
        )

    return RetrievalBriefing(
        candidate           = candidate,
        layer1_hard_stop    = tuple(l1),
        layer2_reconsider   = tuple(l2),
        layer3_structural   = tuple(l3),
        dormant_reactivate  = tuple(dr),
        related_papers      = tuple(paper_hits),
        note                = note,
    )
