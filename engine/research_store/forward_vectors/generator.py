"""engine.research_store.forward_vectors.generator — produce ForwardVectorV2 from untested hypotheses.

DUAL-TRACK as of Phase 2.1b (2026-06-06):

  Track 2 — paper-stated (LLM_EXTRACT)            [UNCHANGED]
    Hypothesis was extracted from a single paper's claim. Source paper
    is required + drives the priority via paper_shelves.

  Track 1 — brainstorm (LLM_SYNTHESIS)           [NEW]
    Hypothesis was synthesized cross-source by Employee A. No single
    source paper. Gated on:
      (a) emit.forward_vector_created event exists for this
          hypothesis_id (means principal approved B's verdict in
          /approvals)
      (b) B verdict still resolvable from verdicts.jsonl
    Priority comes from B's confidence (+ addresses_decay_in bump).

  Track 3 — human-authored (HUMAN_AUTHORED)      [NEW, escape hatch]
    Principal wrote it directly. No gate. Priority = MEDIUM default.
    Useful for paper-replication-on-demand workflows that didn't fit
    Tracks 1 or 2.

Both tracks coexist; the principal sees them in the SAME
/research/forward queue, distinguished by `priority_signals.track`
(paper_stated / brainstorm / human_authored) and
`priority_signals.extraction_method`. UI surfaces the badge via
those fields (separate commit).

Filtering (applies to all 3 tracks):
  - hypothesis must NOT be in `tested_hypothesis_ids(lessons)` already
  - Track 2 silently drops hypotheses whose source_paper_id resolves
    to a missing paper (data integrity guard)

Priority rules (deterministic — no LLM):
  Track 2:
    HIGH    paper carries shelf DOCTRINE_METHOD or GREEN_MOTIVATION
    MEDIUM  paper carries any motivation shelf
    LOW     paper on OTHER / DORMANT shelves only
  Track 1:
    HIGH    B confidence ≥ 0.75  OR addresses_decay_in non-empty
    MEDIUM  B confidence ≥ 0.55
    LOW     otherwise
  Track 3:
    MEDIUM  (manual — principal chose to write it, that's already
             a non-trivial signal)
"""
from __future__ import annotations

import datetime
import logging

from engine.research_store.forward_vectors.schema import (
    ForwardVectorStatus, ForwardVectorV2, Priority,
)
from engine.research_store.hypothesis.schema import ExtractionMethod

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Priority helpers — one per track
# ────────────────────────────────────────────────────────────────────
def _compute_priority(paper_shelves: set[str]) -> tuple[Priority, dict]:
    """Track 2 (paper_stated) — UNCHANGED priority rule."""
    signals = {"paper_shelves": sorted(paper_shelves)}
    if "doctrine_method" in paper_shelves or "green_motivation" in paper_shelves:
        return Priority.HIGH, signals
    motivation_shelves = {"yellow_motivation", "red_motivation",
                          "green_critique", "red_critique"}
    if paper_shelves & motivation_shelves:
        return Priority.MEDIUM, signals
    return Priority.LOW, signals


def _priority_from_b_verdict(verdict: dict) -> tuple[Priority, dict]:
    """Track 1 (brainstorm) — priority from B's confidence + whether the
    hypothesis explicitly addresses a known decay."""
    try:
        conf = float(verdict.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    addresses_decay = bool(verdict.get("addresses_decay_in") or
                            verdict.get("addresses_decay"))
    signals = {
        "b_confidence":      conf,
        "addresses_decay":   addresses_decay,
        "b_verdict_type":    verdict.get("verdict_type"),
    }
    if conf >= 0.75 or addresses_decay:
        return Priority.HIGH, signals
    if conf >= 0.55:
        return Priority.MEDIUM, signals
    return Priority.LOW, signals


# ────────────────────────────────────────────────────────────────────
# Cross-store reads — Track 1 dependencies
# ────────────────────────────────────────────────────────────────────
def _load_fv_created_set() -> set[str]:
    """Read events.jsonl, return the set of hypothesis_ids for which
    a forward_vector_created event has been emitted. These are the
    LLM_SYNTHESIS candidates the principal already approved in
    /approvals (step P2.1a wires the emit on resolve)."""
    try:
        from engine.research_store.store import filter_events
        evs = filter_events(event_type="forward_vector_created")
        return {ev.subject_id for ev in evs if ev.subject_id}
    except Exception as exc:
        logger.warning("forward_vectors: fv_created event read failed: %s", exc)
        return set()


def _load_b_verdicts_by_hid() -> dict[str, dict]:
    """Read verdicts.jsonl, return latest verdict per hypothesis_id.
    Used to compute Track 1 priority from B confidence."""
    try:
        from engine.agents.strengthener.approval_view import (
            _load_verdicts, _DEFAULT_VERDICTS_PATH,
        )
        rows = _load_verdicts(_DEFAULT_VERDICTS_PATH)
        latest: dict[str, dict] = {}
        for v in rows:
            hid = v.get("hypothesis_id")
            if not hid:
                continue
            prior = latest.get(hid)
            if prior is None or v.get("review_ts", "") > prior.get("review_ts", ""):
                latest[hid] = v
        return latest
    except Exception as exc:
        logger.warning("forward_vectors: verdicts read failed: %s", exc)
        return {}


def _synthesis_title_from(hyp) -> str:
    """Track 1 has no single source paper. Build a synthetic title
    that surfaces N papers cited in the synthesis. UI uses it for the
    row's headline."""
    n = len(getattr(hyp, "synthesizes_paper_ids", ()) or ())
    if n == 0:
        return "[Synthesis] (event-driven, no papers cited)"
    return f"[Synthesis] across {n} paper{'s' if n != 1 else ''}"


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def generate_forward_vectors() -> list[ForwardVectorV2]:
    """Generate fresh ForwardVectorV2 records from current state.

    Dual-track (Phase 2.1b):
      LLM_EXTRACT      → paper_stated lane    (paper required)
      LLM_SYNTHESIS    → brainstorm lane      (fv_created + B verdict gate)
      HUMAN_AUTHORED   → escape hatch         (always accepted)

    Does NOT persist — returns a list. The CLI driver decides whether
    to save.
    """
    from engine.research_store.hypothesis import load_hypotheses
    from engine.research_store.papers import load_registry, latest_per_doi
    from engine.research_store.red_lessons.retrieval import tested_hypothesis_ids

    hyps = load_hypotheses()
    latest_hyps_by_id = {}
    for h in hyps:
        prior = latest_hyps_by_id.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            latest_hyps_by_id[h.hypothesis_id] = h
    logger.info("loaded %d hypotheses (latest per id)", len(latest_hyps_by_id))

    tested_set = tested_hypothesis_ids()
    logger.info("tested set: %d hypothesis_ids", len(tested_set))

    # Track 2 deps (existing)
    reg = list(latest_per_doi(load_registry()).values())
    paper_by_id = {e.paper_id: e for e in reg}

    # Track 1 deps (new)
    fv_created_set = _load_fv_created_set()
    b_verdicts     = _load_b_verdicts_by_hid()
    logger.info("fv_created set: %d hyp_ids, B verdicts: %d hyp_ids",
                 len(fv_created_set), len(b_verdicts))

    out: list[ForwardVectorV2] = []
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for hyp_id, hyp in latest_hyps_by_id.items():
        if hyp_id in tested_set:
            continue   # already tested → done

        em             = hyp.extraction_method
        track_label    = ""
        priority       = Priority.LOW
        signals: dict  = {}
        paper_title    = ""
        source_paper_id = ""

        if em == ExtractionMethod.LLM_EXTRACT:
            # ── Track 2: paper-stated, unchanged ──
            paper = paper_by_id.get(hyp.source_paper_id)
            if paper is None:
                logger.warning(
                    "hypothesis %s (LLM_EXTRACT) references missing paper "
                    "%s; skipping", hyp_id, hyp.source_paper_id,
                )
                continue
            paper_shelves = {s.value for s in paper.shelves}
            priority, signals = _compute_priority(paper_shelves)
            paper_title       = paper.title
            source_paper_id   = hyp.source_paper_id
            track_label       = "paper_stated"

        elif em == ExtractionMethod.LLM_SYNTHESIS:
            # ── Track 1: brainstorm, new ──
            if hyp_id not in fv_created_set:
                continue   # B + principal double-gate not cleared yet
            verdict = b_verdicts.get(hyp_id)
            if verdict is None:
                # fv_created event exists but verdict can't be found —
                # log + skip defensively (shouldn't happen in normal flow)
                logger.warning(
                    "hypothesis %s has forward_vector_created event but "
                    "no resolvable B verdict; skipping", hyp_id,
                )
                continue
            priority, signals = _priority_from_b_verdict(verdict)
            paper_title       = _synthesis_title_from(hyp)
            source_paper_id   = ""   # synthesis has no single source
            track_label       = "brainstorm"

        elif em == ExtractionMethod.HUMAN_AUTHORED:
            # ── Track 3: manual escape hatch ──
            priority = Priority.MEDIUM
            signals  = {"manually_authored": True}
            # source_paper_id may or may not be set; if set, surface
            # the paper title for context
            if hyp.source_paper_id:
                paper = paper_by_id.get(hyp.source_paper_id)
                if paper is not None:
                    paper_title     = paper.title
                    source_paper_id = hyp.source_paper_id
            track_label = "human_authored"

        else:
            # Defensive — future extraction_method we don't know yet
            logger.warning("hypothesis %s has unknown extraction_method %s; "
                            "skipping", hyp_id, em)
            continue

        # Tag the track + method onto priority_signals so the UI can
        # render the right badge (and audit consumers can query by
        # track without re-deriving from extraction_method)
        signals["track"]             = track_label
        signals["extraction_method"] = em.value

        fv = ForwardVectorV2(
            forward_vector_id    = ForwardVectorV2.new_id(),
            version              = 1,
            parent_id            = None,
            source_paper_id      = source_paper_id,
            paper_title          = paper_title,
            source_hypothesis_id = hyp_id,
            claim                = hyp.claim,
            mechanism_family     = hyp.mechanism_family,
            mechanism_subtype    = hyp.mechanism_subtype,
            predicted_direction  = hyp.predicted_direction.value,
            predicted_magnitude  = hyp.predicted_magnitude,
            required_data        = hyp.required_data,
            test_methodology     = hyp.test_methodology,
            priority             = priority,
            priority_signals     = signals,
            status               = ForwardVectorStatus.PROPOSED,
            created_ts           = now_iso,
            created_by           = "engine.forward_vectors.generator",
            tags                 = ("t5_auto_generated", f"track:{track_label}"),
        )
        out.append(fv)

    # Sort by priority then by (track, paper_id|hyp_id) for stable
    # ordering. Brainstorm rows have no source_paper_id so they sort
    # by hypothesis_id within their priority band.
    priority_rank = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    out.sort(key=lambda fv: (
        priority_rank[fv.priority],
        fv.priority_signals.get("track", ""),
        fv.source_paper_id or fv.source_hypothesis_id,
    ))
    return out
