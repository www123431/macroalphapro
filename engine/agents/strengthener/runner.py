"""engine.agents.strengthener.runner — Phase 2.0 step 11b.

Loops PROPOSED + LLM_SYNTHESIS hypothesis rows from hypotheses.jsonl,
builds a StrengthenerInput per hypothesis (deployed sleeves + family
verdicts + doctrine snippets [stub]), calls run_strengthener_review
per row, persists verdicts to data/strengthener/verdicts.jsonl.

The runner does NOT yet:
  - transition Hypothesis review_state on REJECT (deferred; the
    Hypothesis schema's PROPOSED → REJECTED transition is a
    different concern from B's audit-trail output)
  - create /approvals rows for APPROVE_FOR_PIPELINE or
    DOCTRINE_AMENDMENT_NEEDED (that's step 12)

What it DOES do:
  - skip hypotheses B has already reviewed (idempotent — re-running
    the runner won't double-review the same hypothesis)
  - fail-safe: an exception on one hypothesis doesn't kill the batch
  - return structured result the CLI / API can render

Same fail-safe + structured-result + dry-run contract as
synthesis_runner.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

from engine.agents.strengthener.review import (
    DoctrineContextRef,
    FamilyVerdictRef,
    HypothesisRef,
    SleeveContextRef,
    StrengthenerInput,
    StrengthenerVerdict,
    run_strengthener_review,
)
from engine.research_store.hypothesis import Hypothesis
from engine.research_store.hypothesis.schema import (
    ExtractionMethod,
    HypothesisReviewState,
)

logger = logging.getLogger(__name__)


_REPO_ROOT       = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_VERDICTS_PATH = _REPO_ROOT / "data" / "strengthener" / "verdicts.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _cutoff_iso(days: int) -> str:
    return (_dt.datetime.utcnow() - _dt.timedelta(days=days)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Hypothesis selection
# ────────────────────────────────────────────────────────────────────
def _select_for_review(
    *,
    hypotheses_path: Optional[Path],
    already_reviewed: set[str],
) -> list[Hypothesis]:
    """Load hypotheses.jsonl, keep only PROPOSED rows produced by A's
    synthesis (extraction_method == LLM_SYNTHESIS). Drop anything
    already in `already_reviewed` (idempotency)."""
    from engine.research_store.hypothesis.store import load_hypotheses

    hyps = load_hypotheses(path=hypotheses_path)
    out: list[Hypothesis] = []
    for h in hyps:
        if h.extraction_method != ExtractionMethod.LLM_SYNTHESIS:
            continue
        if h.review_state != HypothesisReviewState.PROPOSED:
            continue
        if h.hypothesis_id in already_reviewed:
            continue
        out.append(h)
    return out


def _load_already_reviewed(verdicts_path: Path) -> set[str]:
    """Read verdicts.jsonl and return the set of hypothesis_ids
    already reviewed."""
    if not verdicts_path.is_file():
        return set()
    out: set[str] = set()
    with verdicts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                hid = d.get("hypothesis_id")
                if hid:
                    out.add(hid)
            except Exception:
                continue
    return out


# ────────────────────────────────────────────────────────────────────
# Context builders
# ────────────────────────────────────────────────────────────────────
def _hypothesis_ref(h: Hypothesis) -> HypothesisRef:
    return HypothesisRef(
        hypothesis_id        = h.hypothesis_id,
        claim                = h.claim,
        mechanism_family     = h.mechanism_family.value,
        mechanism_subtype    = h.mechanism_subtype,
        predicted_direction  = h.predicted_direction.value,
        predicted_magnitude  = h.predicted_magnitude,
        required_data        = tuple(h.required_data),
        test_methodology     = h.test_methodology,
        extraction_method    = h.extraction_method.value,
        synthesizes_paper_ids= tuple(h.synthesizes_paper_ids),
        synthesizes_event_ids= tuple(h.synthesizes_event_ids),
        addresses_decay_in   = h.addresses_decay_in,
        created_ts           = h.created_ts,
        # Phase 2.2c
        citation_quality     = h.citation_quality,
    )


def _load_deployed_sleeves() -> tuple[SleeveContextRef, ...]:
    """Reuse synthesis_context's sleeve loader; project to B's dataclass."""
    from engine.agents.papers_curator.synthesis_context import (
        _load_deployed_sleeves as _gather_sleeves,
    )
    raw = _gather_sleeves()
    return tuple(
        SleeveContextRef(
            sleeve_id           = s.sleeve_id,
            family              = s.family,
            ann_sharpe_live     = s.ann_sharpe_live,
            months_since_deploy = s.months_since_deploy,
            last_decay_alert    = s.last_decay_alert,
        )
        for s in raw
    )


def _load_family_verdicts(
    family: str,
    *,
    days: int = 60,
    max_rows: int = 15,
) -> tuple[FamilyVerdictRef, ...]:
    """Fetch recent factor_verdict_filed events in `family` for B's
    context. 60-day window so B sees a wider lens than A's 30-day
    synthesis window (B's question is 'how have we been doing in
    this family' — wants a bigger sample)."""
    try:
        from engine.research_store.store import filter_events
    except Exception:
        return ()
    since = _cutoff_iso(days)
    events = filter_events(
        event_type = "factor_verdict_filed",
        family     = family,
        since      = since,
    )
    out: list[FamilyVerdictRef] = []
    for ev in events[:max_rows]:
        out.append(FamilyVerdictRef(
            event_id   = ev.event_id,
            subject_id = ev.subject_id,
            verdict    = ev.verdict.value,
            ts         = ev.ts,
            summary    = (ev.summary or "")[:200],
        ))
    return tuple(out)


def _load_doctrine_snippets(*, family: str,
                              claim_hint: str = "",
                              top_k: int = 3) -> tuple[DoctrineContextRef, ...]:
    """Tier-2 (2026-06-07): real doctrine retrieval for B's review.

    Builds a topic_hint from family + claim and asks doctrine_index
    for top-K relevant memory entries. B's review prompt surfaces
    these so the model can flag DOCTRINE_AMENDMENT_NEEDED when a
    candidate conflicts with locked doctrine, or weight by doctrine
    fit when proposing APPROVE_FOR_PIPELINE.

    Empty family AND empty claim → () (no anchor). Chroma failure
    → () (B degrades to no-doctrine reasoning, same as pre-tier-2)."""
    hint_parts = []
    if family:
        hint_parts.append(f"family: {family}")
    if claim_hint:
        hint_parts.append(claim_hint[:300])
    topic = ". ".join(hint_parts)
    if not topic.strip():
        return ()

    try:
        from engine.agents.papers_curator.doctrine_index import query_doctrine
        raw_hits = query_doctrine(topic, top_k=top_k)
    except Exception as exc:
        logger.warning("strengthener: query_doctrine raised: %s", exc)
        return ()

    out: list[DoctrineContextRef] = []
    for h in raw_hits:
        out.append(DoctrineContextRef(
            memory_file_id = h.name,
            headline       = (h.description[:160] if h.description
                               else h.name),
            snippet        = h.snippet[:400],
            relevance_note = f"distance={h.distance:.3f}",
        ))
    return tuple(out)


def build_input_for(h: Hypothesis) -> StrengthenerInput:
    """Assemble the full StrengthenerInput per hypothesis. Pulled out
    of run_strengthener_pipeline so tests can build inputs with
    synthetic Hypothesis rows."""
    hr = _hypothesis_ref(h)
    sleeves = _load_deployed_sleeves()
    family_verdicts = _load_family_verdicts(h.mechanism_family.value)
    # Tier-2 Q3: pass the candidate's claim into doctrine retrieval so
    # the topic anchor is "family + claim" not just "family". chroma
    # returns memory entries semantically relevant to the SPECIFIC
    # candidate B is reviewing, not just the family bucket.
    doctrine = _load_doctrine_snippets(
        family     = h.mechanism_family.value,
        claim_hint = h.claim,
    )
    return StrengthenerInput(
        hypothesis        = hr,
        deployed_sleeves  = sleeves,
        doctrine_snippets = doctrine,
        family_verdicts   = family_verdicts,
        snapshot_ts       = _utc_iso(),
    )


# ────────────────────────────────────────────────────────────────────
# Verdict persistence
# ────────────────────────────────────────────────────────────────────
def _append_verdict(verdict: StrengthenerVerdict, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dc.asdict(verdict)
    # Serialize the enum
    payload["verdict_type"] = verdict.verdict_type.value
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────────
# Top-level entry
# ────────────────────────────────────────────────────────────────────
def run_strengthener_pipeline(
    *,
    dry_run:           bool = False,
    max_hypotheses:    int = 10,
    hypotheses_path:   Optional[Path] = None,
    verdicts_path:     Optional[Path] = None,
) -> dict:
    """Loop PROPOSED + LLM_SYNTHESIS hypotheses, call B per row, persist
    verdicts. Returns structured result for CLI / API rendering.

    Args:
      dry_run:          run B's LLM call but skip verdict persistence.
      max_hypotheses:   cap per-run (cost gate; ~$0.05 × N per run).
      hypotheses_path:  override hypotheses.jsonl path (tests).
      verdicts_path:    override verdicts.jsonl path (tests).

    Returns:
      {
        "run_ts":           iso,
        "dry_run":          bool,
        "n_candidates":     int,   # PROPOSED + LLM_SYNTHESIS in store
        "n_reviewed":       int,   # called B
        "n_persisted":      int,   # verdicts written (0 on dry_run)
        "verdicts":         [verdict-as-dict, ...],
        "errors":           [str, ...],
      }
    """
    verdicts_path = verdicts_path or _DEFAULT_VERDICTS_PATH
    run_ts = _utc_iso()
    result: dict = {
        "run_ts":         run_ts,
        "dry_run":        dry_run,
        "n_candidates":   0,
        "n_reviewed":     0,
        "n_persisted":    0,
        "verdicts":       [],
        "errors":         [],
    }

    # ── Load + idempotency filter ──────────────────────────────
    try:
        already = _load_already_reviewed(verdicts_path)
        to_review = _select_for_review(
            hypotheses_path  = hypotheses_path,
            already_reviewed = already,
        )
        result["n_candidates"] = len(to_review)
    except Exception as exc:
        logger.exception("strengthener_runner: load failed")
        result["errors"].append(f"load: {exc}")
        return result

    if not to_review:
        return result

    to_review = to_review[:max_hypotheses]

    # ── Loop, gather context, review, persist ──────────────────
    for h in to_review:
        try:
            si = build_input_for(h)
        except Exception as exc:
            logger.exception("strengthener_runner: build_input failed for %s", h.hypothesis_id)
            result["errors"].append(f"context:{h.hypothesis_id}: {exc}")
            continue

        try:
            v = run_strengthener_review(si)
        except Exception as exc:
            logger.exception("strengthener_runner: review raised for %s", h.hypothesis_id)
            result["errors"].append(f"review:{h.hypothesis_id}: {exc}")
            continue

        if v is None:
            result["errors"].append(f"review:{h.hypothesis_id}: returned None")
            continue

        result["n_reviewed"] += 1
        verdict_dict = _dc.asdict(v)
        verdict_dict["verdict_type"] = v.verdict_type.value
        result["verdicts"].append(verdict_dict)

        if dry_run:
            continue

        try:
            _append_verdict(v, path=verdicts_path)
            result["n_persisted"] += 1
        except Exception as exc:
            logger.exception(
                "strengthener_runner: persist failed for %s", h.hypothesis_id,
            )
            result["errors"].append(f"persist:{h.hypothesis_id}: {exc}")

    return result
