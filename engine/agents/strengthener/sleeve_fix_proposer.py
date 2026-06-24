"""engine.agents.strengthener.sleeve_fix_proposer — Stage B P2 piece 1.

Routes D's (book_monitor) `doctrine_signal_detected` events into
B-shaped fix candidate Hypotheses that flow through the existing
strengthener review pipeline.

Per [[project-four-employee-agentic-roadmap-2026-06-05]]:
  > "D is NOT independent. D's alerts are SIGNAL not ACTION; they
  > MUST route into B's inbox where B generates actionable fix
  > candidates. A bare D alert without B coupling = noise the user
  > can't act on."

Design:
  - DETERMINISTIC template-based proposer (NOT LLM in this piece).
    LLM-driven proposal generation can come in a later piece (P3
    active B worker) once the deterministic baseline proves the
    plumbing.
  - Pattern-name-specific templates produce a Hypothesis with:
      extraction_method      = LLM_SYNTHESIS
      synthesizes_event_ids  = (doctrine_signal_event_id,)
      addresses_decay_in     = sleeve_id (if signal points at a sleeve)
      mechanism_family       = signal.family or OTHER
      claim / test_methodology / predicted_direction / etc. from
        pattern-specific template
  - Idempotency: skip signals already linked via an existing
    Hypothesis with synthesizes_event_ids containing that signal's
    event_id.

Cost: $0 per signal (no LLM call). Output flows into the same
hypothesis store B reviews on its next run.

The three pattern templates (matching engine.agents.book_monitor.
pattern_rules) — each produces a single Hypothesis per signal:

  family_red_cluster   → "Pause new candidates in family X. Investigate
                          spec saturation or feature overfit. Test:
                          run F8 OVERFIT classifier on the cluster."
  sleeve_sharpe_decay  → "Sleeve X EWMA Sharpe below decay floor.
                          Propose decay re-test + replacement seek.
                          Test: rerun decay_sentinel + scan
                          replacement candidates."
  gate_rejection_spike → "Strict gate reject rate above threshold.
                          Propose upstream filter audit. Test: review
                          last 20 RED verdicts for common failure
                          mode."

Any unknown pattern_name falls through to a generic template — better
to surface "we saw an unexpected signal" than to silently drop it.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import uuid
from pathlib import Path
from typing import Optional

from engine.research_store.hypothesis.schema import (
    ExtractionMethod, HypothesisDirection, HypothesisReviewState,
)
from engine.research_store.red_lessons.mechanism_families import (
    MechanismFamily,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Pattern templates
# ────────────────────────────────────────────────────────────────────
_PATTERN_TEMPLATES = {
    "family_red_cluster": {
        "claim_tmpl":          ("Family {family} shows {n_red} RED "
                                  "verdicts in the recent window — "
                                  "investigate spec saturation / "
                                  "overfit before proposing new "
                                  "{family} candidates."),
        "test_methodology":    ("Run engine.research_store.red_lessons."
                                  "F8 OVERFIT_INDUCED classifier on the "
                                  "RED cluster. If ≥ 50% classify as "
                                  "F8, pause family for 30d minimum."),
        "predicted_magnitude": "marginal — fix is procedural not factor",
        "mechanism_subtype":   "family_pause_proposal",
        "required_data":       ("factor_verdict_filed events for family",
                                  "F8 OVERFIT_INDUCED classifier output"),
    },
    "sleeve_sharpe_decay": {
        "claim_tmpl":          ("Deployed sleeve {sleeve_id} EWMA Sharpe "
                                  "below decay floor — propose decay "
                                  "re-test + replacement candidate "
                                  "scan."),
        "test_methodology":    ("Rerun engine.validation.decay_sentinel "
                                  "on sleeve {sleeve_id} with extended "
                                  "lookback. If confirmed, scan "
                                  "papers_curator substrate for "
                                  "replacement mechanism."),
        "predicted_magnitude": "high — affects deployed capital",
        "mechanism_subtype":   "sleeve_decay_response",
        "required_data":       ("sleeve PnL daily history",
                                  "decay_sentinel re-test output"),
    },
    "gate_rejection_spike": {
        "claim_tmpl":          ("Strict-gate reject rate above threshold "
                                  "({n_red}/N RED in window) — propose "
                                  "upstream filter audit."),
        "test_methodology":    ("Review last 20 RED verdicts for common "
                                  "failure mode. If clustered, tighten "
                                  "the responsible upstream filter "
                                  "(citation_verifier, "
                                  "synthesis_context, etc.)."),
        "predicted_magnitude": "marginal — affects proposal hit rate",
        "mechanism_subtype":   "filter_tighten_proposal",
        "required_data":       ("recent factor_verdict_filed RED events",
                                  "upstream filter audit log"),
    },
}
_GENERIC_TEMPLATE = {
    "claim_tmpl":          ("Unrecognized doctrine signal "
                              "{pattern_name} for {subject_id} — "
                              "investigate manually."),
    "test_methodology":    ("Open the event_id in events.jsonl, walk "
                              "the parent_event_ids chain to root "
                              "cause."),
    "predicted_magnitude": "unknown",
    "mechanism_subtype":   "unknown_signal_response",
    "required_data":       ("source doctrine_signal_detected event",),
}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _looks_like_sleeve_id(subject_id: str) -> bool:
    """Heuristic: sleeve IDs are lowercase_with_underscores and
    DON'T start with 'auto_' (which is the spec_hash prefix for
    candidate factors). Sleeve IDs come from library_yaml entries
    like 'cross_asset_carry', 'time_series_momentum'."""
    if not subject_id:
        return False
    if subject_id.startswith("auto_"):
        return False
    return "_" in subject_id and subject_id == subject_id.lower()


def _resolve_family(signal_family: Optional[str]) -> MechanismFamily:
    """Map signal.family (str) → MechanismFamily enum. Unknown / None
    → OTHER so the Hypothesis still validates."""
    if not signal_family:
        return MechanismFamily.OTHER
    try:
        return MechanismFamily(signal_family.upper())
    except (ValueError, AttributeError):
        return MechanismFamily.OTHER


# ────────────────────────────────────────────────────────────────────
# Build one fix-proposal Hypothesis from one doctrine_signal event
# ────────────────────────────────────────────────────────────────────
def build_fix_hypothesis_from_signal(signal_ev) -> "Hypothesis":
    """Build a fix-proposal Hypothesis from a doctrine_signal_detected
    event. Returns a non-persisted Hypothesis dataclass — caller
    decides whether to write it.

    `signal_ev` is a ResearchEvent (or SimpleNamespace duck-type) with
    fields: event_id, subject_id, family, ts, summary, metrics.
    """
    from engine.research_store.hypothesis.schema import Hypothesis

    metrics    = signal_ev.metrics or {}
    pattern    = str(metrics.get("pattern_name") or "")
    family_str = signal_ev.family or metrics.get("family") or ""
    n_red      = metrics.get("n_red") or metrics.get("count") or "N"
    subject_id = signal_ev.subject_id or ""
    sleeve_id  = (subject_id if _looks_like_sleeve_id(subject_id)
                   else "")

    template = _PATTERN_TEMPLATES.get(pattern) or _GENERIC_TEMPLATE
    claim = template["claim_tmpl"].format(
        family       = family_str or "(unknown family)",
        n_red        = n_red,
        sleeve_id    = sleeve_id or "(unknown sleeve)",
        subject_id   = subject_id or "(unknown subject)",
        pattern_name = pattern or "(unknown pattern)",
    )

    now = _utc_iso()
    return Hypothesis(
        hypothesis_id        = str(uuid.uuid4()),
        source_paper_id      = "",
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = (),
        verbatim_quotes      = (),
        claim                = claim,
        mechanism_family     = _resolve_family(family_str),
        mechanism_subtype    = template["mechanism_subtype"],
        predicted_direction  = HypothesisDirection.ZERO,   # fixes don't
                                                              # predict a
                                                              # direction
        predicted_magnitude  = template["predicted_magnitude"],
        required_data        = template.get("required_data") or (
                                  "source doctrine_signal_detected event",
                                ),
        test_methodology     = template["test_methodology"],
        extraction_method    = ExtractionMethod.LLM_SYNTHESIS,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = now,
        updated_ts           = now,
        created_by           = "engine.agents.strengthener.sleeve_fix_proposer",
        tags                 = ("source:doctrine_signal",
                                  f"pattern:{pattern}",
                                  f"signal_event:{signal_ev.event_id}"),
        synthesizes_paper_ids = (),
        synthesizes_event_ids = (signal_ev.event_id,) if signal_ev.event_id else (),
        addresses_decay_in    = sleeve_id or None,
    )


# ────────────────────────────────────────────────────────────────────
# Idempotency — which signal events have we already turned into
# fix-proposals?
# ────────────────────────────────────────────────────────────────────
def _already_proposed_event_ids(
    *, hypotheses_path: Optional[Path] = None,
) -> set[str]:
    """Set of signal event_ids that already have an associated
    fix-proposal hypothesis (matched via synthesizes_event_ids)."""
    try:
        from engine.research_store.hypothesis.store import load_hypotheses
        hyps = load_hypotheses(path=hypotheses_path)
    except Exception as exc:
        logger.warning("sleeve_fix_proposer: load_hypotheses failed: %s",
                        exc)
        return set()
    seen: set[str] = set()
    for h in hyps:
        for eid in (h.synthesizes_event_ids or ()):
            if eid:
                seen.add(str(eid))
    return seen


# ────────────────────────────────────────────────────────────────────
# Main entry — read signals, propose, persist
# ────────────────────────────────────────────────────────────────────
def propose_sleeve_fixes(
    *,
    since:           Optional[str] = None,
    days:            int = 30,
    max_signals:     int = 10,
    dry_run:         bool = False,
    hypotheses_path: Optional[Path] = None,
) -> dict:
    """Scan recent doctrine_signal_detected events; for each not yet
    linked to a fix-proposal, build + persist a fix Hypothesis.

    Returns:
      {
        run_ts:            iso,
        dry_run:           bool,
        n_signals_seen:    int,
        n_already_done:    int,   # skipped (idempotent)
        n_proposed:        int,   # new fix hypotheses built
        n_persisted:       int,   # written to disk (0 on dry_run)
        proposed_ids:      list[str],   # hypothesis_ids
        errors:            list[str],
      }
    """
    from engine.research_store.store import filter_events
    from engine.research_store.hypothesis.store import save_hypothesis

    run_ts = _utc_iso()
    result = {
        "run_ts":         run_ts,
        "dry_run":        dry_run,
        "n_signals_seen": 0,
        "n_already_done": 0,
        "n_proposed":     0,
        "n_persisted":    0,
        "proposed_ids":   [],
        "errors":         [],
    }

    if since is None:
        since = (_dt.datetime.utcnow()
                  - _dt.timedelta(days=days)
                  ).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        signals = filter_events(
            event_type = "doctrine_signal_detected",
            since      = since,
            limit      = max_signals * 4,  # over-read; we filter below
        )
    except Exception as exc:
        logger.exception("sleeve_fix_proposer: filter_events failed")
        result["errors"].append(f"load: {exc}")
        return result

    result["n_signals_seen"] = len(signals)
    if not signals:
        return result

    already = _already_proposed_event_ids(
        hypotheses_path=hypotheses_path,
    )

    proposed_count = 0
    for sig in signals:
        if proposed_count >= max_signals:
            break
        if sig.event_id in already:
            result["n_already_done"] += 1
            continue
        try:
            h = build_fix_hypothesis_from_signal(sig)
        except Exception as exc:
            logger.exception("sleeve_fix_proposer: build failed for %s",
                              sig.event_id)
            result["errors"].append(
                f"build:{sig.event_id}: {exc}")
            continue

        result["proposed_ids"].append(h.hypothesis_id)
        proposed_count += 1
        result["n_proposed"] += 1

        if dry_run:
            continue
        try:
            # skip_cross_checks=True because synthesis-derived Hypotheses
            # don't have papers_registry / papers_chroma resolution
            # to cross-validate
            save_hypothesis(h, path=hypotheses_path,
                              skip_cross_checks=True)
            result["n_persisted"] += 1
        except Exception as exc:
            logger.exception("sleeve_fix_proposer: persist failed for "
                              "%s", h.hypothesis_id)
            result["errors"].append(
                f"persist:{h.hypothesis_id}: {exc}")

    return result
