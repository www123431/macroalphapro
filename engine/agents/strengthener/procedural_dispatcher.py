"""engine.agents.strengthener.procedural_dispatcher — Stage B close-loop.

Auto-executes procedural Hypotheses through deterministic dispatchers
+ persists result to procedural_dispatch_log.jsonl. Closes the link
the user previously had to walk by hand.

Why NOT factor_verdict_filed: that event type requires (a) a
registered subject (procedural hashes are not registered factors)
and (b) artifacts.evidence_doc pointing at a real capability_evidence
markdown that exists on disk. Procedural dispatch is governance
data, not strict-gate factor research — mixing them dilutes the
event-store semantic. We write to a dedicated audit log instead.

Motivation — found 2026-06-07 during manual close-loop of pending
hid 47893a71:
  - That hypothesis took 5 minutes to test by hand
  - 80% of the friction was: (a) finding the right module path
    (`engine.research_store.red_lessons.backfill_heuristics`) and
    (b) finding the right input fields (`metrics.deflated_sr`)
  - Risk profile is LOW: no DB queries, no look-ahead, no statistical
    estimation — just function call routing
  - Therefore: procedural hypotheses are the SAFE first target for
    auto-test. Factor backtests stay human-written (look-ahead risk).

Per [[project-a-plus-b-substrate-first-roadmap-2026-06-05]] capital
line: this emits factor_verdict_filed events, NOT paper_trade or
allocation changes. Human still owns capital decisions; only the
PROCEDURAL test step is automated.

Pattern-5-compliant design:
  - Single Sonnet call for SPEC EXTRACTION (not code generation)
  - LLM picks ONE of a controlled `dispatch_kind` enum + args; can
    NOT add new dispatchers
  - Deterministic Python dispatcher then executes
  - If LLM picks 'unrecognized' or returns invalid spec, dispatcher
    short-circuits to verdict='MARGINAL' with note "needs human"

Trigger criteria:
  - hypothesis.predicted_direction == ZERO
  - hypothesis.mechanism_subtype matches procedural regex
  - addresses_decay_in is set OR synthesizes_event_ids non-empty
    (so the dispatcher has context to operate on)
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import re
import uuid
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# Controlled dispatch_kind enum — single source of truth.
# Adding a kind = adding a registered dispatcher in DISPATCHERS dict.
# LLM can ONLY pick from these names; cannot invent new ones.
DISPATCH_KINDS = (
    "failure_mode_classify",   # run F8/F4/F3 classifier on a RED cluster
    "decay_resentinel_rerun",  # re-run decay_sentinel on a sleeve_id
    "red_cluster_recount",     # honestly recount a cluster with dedup
    "unrecognized",            # LLM signals "no clean dispatch fits"
)

# Procedural mechanism_subtype patterns. Any hypothesis where
# subtype matches one of these is eligible for the dispatcher.
_PROCEDURAL_SUBTYPE_RE = re.compile(
    r"(proposal|pause|audit|fix|response|tighten|recount|classify)",
    re.IGNORECASE,
)


# ────────────────────────────────────────────────────────────────────
# Eligibility check
# ────────────────────────────────────────────────────────────────────
def is_procedural_hypothesis(h) -> bool:
    """True if a Hypothesis is safe + worth auto-dispatching:
      - predicted_direction == ZERO (not predicting returns)
      - mechanism_subtype matches procedural regex
      - has at least one provenance link (addresses_decay_in OR
        synthesizes_event_ids non-empty) so dispatcher has context
    """
    try:
        if h.predicted_direction.value != "zero":
            return False
    except AttributeError:
        return False
    if not _PROCEDURAL_SUBTYPE_RE.search(h.mechanism_subtype or ""):
        return False
    if not h.addresses_decay_in and not h.synthesizes_event_ids:
        return False
    return True


# ────────────────────────────────────────────────────────────────────
# Spec extraction (LLM)
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class DispatchSpec:
    """LLM-extracted dispatch spec. Args dict shape depends on
    dispatch_kind; dispatcher functions validate their own args."""
    dispatch_kind: str
    args:          dict
    rationale:     str   # 1-sentence why this kind fits


_SPEC_SYSTEM_PROMPT = """\
You are translating ONE procedural research hypothesis into a
structured dispatch spec. Your role is NOT to write code or generate
strategy; only to MAP the hypothesis's test_methodology to one of
the CONTROLLED dispatch_kind values + extract the args needed.

The dispatchers are deterministic Python functions. If the
hypothesis's test_methodology does not cleanly map to any registered
kind, return dispatch_kind='unrecognized' (the human will write the
test by hand). Do not force-fit.

Available dispatch_kinds:

  failure_mode_classify
    Runs the F8/F4/F3 failure-mode classifier on a set of RED
    factor_verdict_filed events. Args:
      red_event_ids: list[str]  — event IDs to classify (REQUIRED;
                                    pull from synthesizes_event_ids
                                    or addresses_decay_in)
      threshold_pct: float      — if ≥ this fraction classify as
                                    F8, emit MARGINAL verdict
                                    recommending pause. Default 0.5.

  decay_resentinel_rerun
    Re-runs engine.validation.decay_sentinel on the deployed sleeve
    pointed to by addresses_decay_in. Args:
      sleeve_id: str  — REQUIRED, the addresses_decay_in value

  red_cluster_recount
    Honestly recount a family_red_cluster with dedup-by-subject so
    the recount reflects unique failure cases. Args:
      family:        str        — REQUIRED, the family name
      window_days:   int        — default 30
      red_event_ids: list[str]  — optional; if absent the dispatcher
                                    re-pulls all RED in family

  unrecognized
    Use when the hypothesis test_methodology doesn't fit any of the
    above cleanly. The dispatcher emits MARGINAL with note
    "auto-dispatch unrecognized; needs human test".

Output: invoke the emit_dispatch_spec tool EXACTLY ONCE with the
dispatch_kind + args.
"""


_SPEC_TOOL = {
    "name": "emit_dispatch_spec",
    "description": ("Emit ONE dispatch spec for this procedural "
                    "hypothesis. Pick dispatch_kind from the "
                    "controlled enum; extract args from the "
                    "hypothesis text."),
    "input_schema": {
        "type": "object",
        "properties": {
            "dispatch_kind": {
                "type": "string",
                "enum": list(DISPATCH_KINDS),
            },
            "args": {
                "type": "object",
                # args shape varies by kind; dispatchers self-validate
            },
            "rationale": {"type": "string"},
        },
        "required": ["dispatch_kind", "args", "rationale"],
        "additionalProperties": False,
    },
}


def _format_spec_user(h) -> str:
    parts = [
        f"HYPOTHESIS_ID:       {h.hypothesis_id}",
        f"MECHANISM_FAMILY:    {h.mechanism_family.value}",
        f"MECHANISM_SUBTYPE:   {h.mechanism_subtype}",
        f"ADDRESSES_DECAY_IN:  {h.addresses_decay_in or '(none)'}",
        f"SYNTHESIZES_EVENT_IDS: {list(h.synthesizes_event_ids or ())}",
        f"PREDICTED_DIRECTION: {h.predicted_direction.value}",
        f"PREDICTED_MAGNITUDE: {h.predicted_magnitude}",
        "",
        "CLAIM:",
        h.claim.strip(),
        "",
        "TEST_METHODOLOGY:",
        h.test_methodology.strip(),
        "",
        "REQUIRED_DATA:",
    ]
    for rd in (h.required_data or ()):
        parts.append(f"  - {rd}")
    return "\n".join(parts)


def extract_dispatch_spec(h) -> Optional[DispatchSpec]:
    """Single LLM call. Returns None on hard failure / tool not
    called / invalid kind (dispatcher treats None same as
    'unrecognized')."""
    try:
        result = llm_call(
            workload   = "strengthener_spec_extract",
            system     = _SPEC_SYSTEM_PROMPT,
            user       = _format_spec_user(h),
            agent_id   = "strengthener_spec_extract",
            tools      = [_SPEC_TOOL],
            max_tokens = 1024,
            scope      = "stage_b_procedural_dispatcher",
        )
    except Exception as exc:
        logger.warning("spec_extract: llm_call failed for %s: %s",
                        h.hypothesis_id, exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_dispatch_spec":
            payload = tc.input
            break
    if payload is None:
        return None
    kind = str(payload.get("dispatch_kind") or "")
    if kind not in DISPATCH_KINDS:
        logger.warning("spec_extract: %s emitted unknown dispatch_kind=%r",
                        h.hypothesis_id, kind)
        return None
    return DispatchSpec(
        dispatch_kind = kind,
        args          = dict(payload.get("args") or {}),
        rationale     = str(payload.get("rationale") or ""),
    )


# ────────────────────────────────────────────────────────────────────
# Dispatchers (deterministic Python — NO LLM)
# ────────────────────────────────────────────────────────────────────
def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _dispatch_failure_mode_classify(args: dict) -> dict:
    """Run F8/F4/F3 classifier on a set of RED factor_verdict_filed
    event IDs. Threshold logic: if ≥threshold_pct classify as F8,
    emit MARGINAL ("recommend pause"); else GREEN ("no over-mining
    pattern detected").

    One-hop walk: the LLM commonly passes doctrine_signal_detected
    event_ids (because that's what's in synthesizes_event_ids). If
    so, walk the signal's parent_event_ids to get the underlying
    factor_verdict_filed events. Caught 2026-06-07 on first live
    close-loop run.
    """
    from engine.research_store.store import filter_events, by_event_id
    from engine.research_store.red_lessons.backfill_heuristics import (
        classify_failure_modes,
    )

    red_ids = list(args.get("red_event_ids") or [])
    threshold = float(args.get("threshold_pct") or 0.5)

    if not red_ids:
        return {
            "verdict":     "MARGINAL",
            "score":       0,
            "summary":     ("no red_event_ids supplied to classifier; "
                              "cannot make recommendation"),
            "metrics":     {"n_events": 0, "n_f8": 0,
                              "threshold_pct": threshold},
        }

    # One-hop walk: if any supplied id resolves to a doctrine_signal,
    # substitute with the signal's parent_event_ids (which point at
    # the underlying factor_verdict_filed REDs).
    resolved_ids: list[str] = []
    walked_from_signal = False
    for eid in red_ids:
        try:
            ev = by_event_id(eid)
        except Exception:
            ev = None
        if ev is not None and getattr(ev.event_type, "value", "") \
                == "doctrine_signal_detected":
            for pid in (ev.parent_event_ids or ()):
                if pid:
                    resolved_ids.append(pid)
            walked_from_signal = True
        else:
            resolved_ids.append(eid)

    # Build a lookup-by-id for the candidates (single store read)
    all_red = {e.event_id: e for e in filter_events(
        event_type="factor_verdict_filed", limit=10000)}

    classified_f8 = 0
    n_total = 0
    f8_evidence: list[str] = []
    for eid in resolved_ids:
        ev = all_red.get(eid)
        if ev is None:
            continue
        n_total += 1
        modes, evidence = classify_failure_modes(ev.metrics or {})
        kinds = {m.value for m in modes}
        if "F8_OVERFIT_INDUCED" in kinds:
            classified_f8 += 1
            f8_evidence.append(
                evidence.get("F8_OVERFIT_INDUCED", "")[:80])

    if n_total == 0:
        return {
            "verdict": "MARGINAL",
            "score":   0,
            "summary": ("none of the supplied red_event_ids resolved "
                          "in events.jsonl"),
            "metrics": {"n_events": 0, "n_f8": 0,
                          "threshold_pct": threshold},
        }

    f8_pct = classified_f8 / n_total
    if f8_pct >= threshold:
        verdict = "MARGINAL"   # MARGINAL = "actionable signal" per
                                # _DOCTRINE_SIGNAL_SEVERITY_TO_VERDICT
        summary = (f"{classified_f8}/{n_total} ({f8_pct:.0%}) classify "
                    f"as F8_OVERFIT_INDUCED — recommend family pause")
    else:
        verdict = "GREEN"
        summary = (f"{classified_f8}/{n_total} ({f8_pct:.0%}) F8 — "
                    f"below {threshold:.0%} threshold; no pause "
                    "recommended")

    return {
        "verdict": verdict,
        "score":   max(0, min(7, int(round(7 * (1.0 - f8_pct))))),
        "summary": summary,
        "metrics": {
            "n_events":              n_total,
            "n_f8":                  classified_f8,
            "f8_pct":                f8_pct,
            "threshold_pct":         threshold,
            "f8_evidence_samples":   f8_evidence[:3],
            "walked_from_signal":    walked_from_signal,
            "resolved_event_ids":    resolved_ids[:10],
        },
    }


def _dispatch_decay_resentinel_rerun(args: dict) -> dict:
    """Re-invoke decay_sentinel on the named sleeve. Returns its
    verdict mapped to factor_verdict_filed shape. NOT YET WIRED to
    actual engine.validation.decay_sentinel (would need that module's
    sleeve_id API surface); for now returns MARGINAL stub indicating
    'human must invoke decay_sentinel by hand'."""
    sleeve_id = str(args.get("sleeve_id") or "")
    if not sleeve_id:
        return {
            "verdict": "MARGINAL",
            "score":   0,
            "summary": "decay_resentinel_rerun: no sleeve_id provided",
            "metrics": {"dispatched": False, "reason": "missing_arg"},
        }
    # NOTE: Full integration deferred — would need to validate that
    # engine.validation.decay_sentinel exposes a sleeve_id-keyed entry
    # point + that the result schema maps to factor_verdict_filed.
    # For now this stub is intentional: surfaces in /approvals as
    # "system suggests rerunning decay_sentinel on sleeve X" and the
    # human kicks the actual rerun.
    return {
        "verdict": "MARGINAL",
        "score":   3,
        "summary": (f"recommend human rerun engine.validation."
                     f"decay_sentinel on sleeve {sleeve_id}"),
        "metrics": {
            "dispatched": False,
            "reason":     "auto_decay_rerun_not_yet_wired",
            "sleeve_id":  sleeve_id,
        },
    }


def _dispatch_red_cluster_recount(args: dict) -> dict:
    """Run check_family_red_cluster (now with dedup) on the named
    family + window and report the recount."""
    from engine.research_store.store import filter_events
    from engine.agents.book_monitor.pattern_rules import (
        check_family_red_cluster,
    )

    family = str(args.get("family") or "")
    window_days = int(args.get("window_days") or 30)
    if not family:
        return {
            "verdict": "MARGINAL",
            "score":   0,
            "summary": "red_cluster_recount: no family provided",
            "metrics": {},
        }

    cutoff = (_dt.datetime.utcnow()
              - _dt.timedelta(days=window_days)
              ).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = filter_events(event_type="factor_verdict_filed",
                            verdict="RED", since=cutoff, limit=500)
    family_events = [e for e in events if (e.family or "") == family]
    hits = check_family_red_cluster(family_events,
                                       window_days=window_days)
    if not hits:
        return {
            "verdict": "GREEN",
            "score":   6,
            "summary": (f"{family}: 0 cluster signal after dedup "
                          f"({len(family_events)} raw RED events in "
                          f"{window_days}d but dedup → below threshold)"),
            "metrics": {
                "family":           family,
                "window_days":      window_days,
                "raw_red_count":    len(family_events),
                "deduped_cluster":  0,
            },
        }
    hit = hits[0]
    return {
        "verdict": "MARGINAL",
        "score":   max(0, min(7, 7 - hit.metrics["red_count"])),
        "summary": (f"{family}: {hit.metrics['red_count']} unique "
                     f"REDs after dedup — cluster signal IS real "
                     f"({hit.severity})"),
        "metrics": {
            "family":          family,
            "window_days":     window_days,
            "raw_red_count":   len(family_events),
            "deduped_cluster": hit.metrics["red_count"],
            "unique_subjects": hit.metrics["red_subject_ids"],
        },
    }


def _dispatch_unrecognized(args: dict) -> dict:
    return {
        "verdict": "MARGINAL",
        "score":   0,
        "summary": ("auto-dispatch: spec extractor returned "
                     "'unrecognized' — needs human test"),
        "metrics": {"dispatched": False, "reason": "llm_unrecognized"},
    }


DISPATCHERS = {
    "failure_mode_classify":   _dispatch_failure_mode_classify,
    "decay_resentinel_rerun":  _dispatch_decay_resentinel_rerun,
    "red_cluster_recount":     _dispatch_red_cluster_recount,
    "unrecognized":            _dispatch_unrecognized,
}


# ────────────────────────────────────────────────────────────────────
# Main entry — extract spec, dispatch, emit
# ────────────────────────────────────────────────────────────────────
def auto_dispatch_procedural(
    h,
    *,
    dry_run: bool = False,
) -> dict:
    """Take ONE procedural Hypothesis, extract dispatch spec via
    Sonnet, run the deterministic dispatcher, emit factor_verdict_filed.

    Returns:
      {
        hypothesis_id:    str,
        eligible:         bool,
        spec:             {dispatch_kind, args, rationale} | None,
        dispatch_result:  {verdict, score, summary, metrics} | None,
        emitted_event_id: str | None,
        errors:           list[str],
      }
    """
    result = {
        "hypothesis_id":    getattr(h, "hypothesis_id", ""),
        "eligible":         False,
        "spec":             None,
        "dispatch_result":  None,
        "emitted_event_id": None,
        "errors":           [],
    }

    if not is_procedural_hypothesis(h):
        return result
    result["eligible"] = True

    spec = extract_dispatch_spec(h)
    if spec is None:
        spec = DispatchSpec(
            dispatch_kind="unrecognized", args={},
            rationale="spec extractor returned None",
        )
    result["spec"] = _dc.asdict(spec)

    dispatcher = DISPATCHERS.get(spec.dispatch_kind,
                                    _dispatch_unrecognized)
    try:
        dr = dispatcher(spec.args)
    except Exception as exc:
        logger.exception("auto_dispatch: dispatcher %s raised for %s",
                          spec.dispatch_kind, h.hypothesis_id)
        result["errors"].append(f"dispatcher:{spec.dispatch_kind}: {exc}")
        dr = {
            "verdict": "MARGINAL",
            "score":   0,
            "summary": (f"dispatcher raised: {exc}"),
            "metrics": {"dispatcher_error": str(exc)[:200]},
        }
    result["dispatch_result"] = dr

    if dry_run:
        return result

    try:
        dispatch_event_id = _append_dispatch_log({
            "dispatch_event_id":  str(uuid.uuid4()),
            "ts":                 _utc_iso(),
            "hypothesis_id":      h.hypothesis_id,
            "mechanism_family":   h.mechanism_family.value,
            "mechanism_subtype":  h.mechanism_subtype,
            "addresses_decay_in": h.addresses_decay_in,
            "synthesizes_event_ids": list(h.synthesizes_event_ids or ()),
            "spec": {
                "dispatch_kind": spec.dispatch_kind,
                "args":          spec.args,
                "rationale":     spec.rationale,
            },
            "dispatch_result":    dr,
            "actor":              "engine.agents.strengthener.procedural_dispatcher",
        })
        result["emitted_event_id"] = dispatch_event_id
    except Exception as exc:
        logger.exception("auto_dispatch: log append failed for %s",
                          h.hypothesis_id)
        result["errors"].append(f"log_append: {exc}")

    return result


# ────────────────────────────────────────────────────────────────────
# Dispatch log — append-only audit trail
# ────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DISPATCH_LOG_PATH = (_REPO_ROOT / "data" / "strengthener"
                       / "procedural_dispatch_log.jsonl")


def _append_dispatch_log(record: dict) -> str:
    """Append one dispatch record to procedural_dispatch_log.jsonl.
    Returns the dispatch_event_id for callers + audit JOINs."""
    DISPATCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DISPATCH_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["dispatch_event_id"]
