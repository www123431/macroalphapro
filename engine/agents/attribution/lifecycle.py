"""engine.agents.attribution.lifecycle — Layer 4 piece 3b.

Read-time JOIN over events / hypotheses / verdicts / resolutions /
cache to produce per-candidate lifecycle records + rollup aggregates.

Per project_anti_rut_doctrine_2026-06-07.md: this is the institutional
'self-evolving system' answer — NOT weight-level learning, but
SYSTEM-level attribution that feeds workflow optimization in piece 3c
(watchlist + doctrine retrieval reweighting based on conversion data).

Aggregates this surface produces:
  aggregate_by_author(days)              "which watchlist authors yield
                                          candidates that survive?"
  aggregate_by_doctrine_snippet(days)    "which memory entries correlate
                                          with GREEN outcomes?"
  aggregate_by_source(days)              "arxiv vs SS watchlist
                                          conversion?"
  calibration_a_confidence(days)         "A says moderate_GREEN —
                                          actual GREEN rate?"

All four read from current state — NO denormalization. When piece 3c
ships and consumes these aggregates, it gets fresh truth.

Empty data path: all four return () gracefully when there are no
qualifying records yet (early system has N=0; that's a real state
not an error).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVENTS_PATH      = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
_VERDICTS_PATH    = _REPO_ROOT / "data" / "strengthener" / "verdicts.jsonl"
_RESOLUTIONS_PATH = _REPO_ROOT / "data" / "strengthener" / "resolutions.jsonl"


# ────────────────────────────────────────────────────────────────────
# Output shapes
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class CandidateLifecycle:
    """Full lineage walk for one candidate. Built by JOIN across stores.

    Fields are Optional where downstream data isn't there yet (early
    candidates that haven't reached B / principal / strict gate yet).
    """
    # Identity + creation
    hypothesis_id:               str
    extraction_method:           str
    mechanism_family:            str
    created_ts:                  str
    claim:                       str

    # Generation inputs (what A actually saw)
    cited_paper_ids:             tuple[str, ...]
    cited_paper_sources:         dict      # paper_id → 'arxiv' / 'semantic_scholar' / None
    cited_watchlist_authors:     tuple[str, ...]  # authors on watchlist among cited authors
    doctrine_snippet_ids:        tuple[str, ...]
    citation_quality:            "dict | None"
    a_expected_outcome_prior:    str       # A's predicted tier

    # B review
    b_verdict_type:              "str | None"
    b_confidence:                "float | None"

    # Principal decision
    principal_decision:          "str | None"   # approved / rejected / deferred
    principal_decision_ts:       str

    # Strict gate (auto_<hash> subject lookup)
    strict_gate_subject_id:      str
    strict_gate_verdict:         "str | None"   # GREEN / MARGINAL / RED / SKIPPED
    strict_gate_score:           "int | None"

    # Roll-up state — most downstream-progressed thing we observed
    final_state:                 str   # see enum below


# Lifecycle final_state semantic ladder (most-progressed first):
FINAL_STATE_GREEN                    = "GREEN"
FINAL_STATE_MARGINAL                 = "MARGINAL"
FINAL_STATE_RED                      = "RED"
FINAL_STATE_SKIPPED_PRE_COMPUTE       = "SKIPPED_PRE_COMPUTE"
FINAL_STATE_PRINCIPAL_APPROVED        = "PRINCIPAL_APPROVED_NO_TEST"
FINAL_STATE_PRINCIPAL_REJECTED        = "PRINCIPAL_REJECTED"
FINAL_STATE_PRINCIPAL_DEFERRED        = "PRINCIPAL_DEFERRED"
FINAL_STATE_B_AMENDMENT               = "B_AMENDMENT"
FINAL_STATE_B_REJECTED                = "B_REJECTED"
FINAL_STATE_B_APPROVED_NO_DECISION    = "B_APPROVED_NO_DECISION"
FINAL_STATE_PRE_B_REVIEW              = "PRE_B_REVIEW"


# Aggregates
@_dc.dataclass(frozen=True)
class AuthorAggregate:
    author_name:                str
    n_candidates_cited:         int       # candidates that cited at least one paper authored by them
    n_b_approved:               int       # of those, B_APPROVED_NO_DECISION or onward
    n_principal_approved:       int       # of those, PRINCIPAL_APPROVED_NO_TEST or onward
    n_strict_gate_run:          int       # of those, reached GREEN/MARGINAL/RED
    n_green:                    int
    n_red:                      int
    conversion_rate_to_green:   float     # n_green / max(1, n_candidates_cited)


@_dc.dataclass(frozen=True)
class DoctrineSnippetAggregate:
    memory_file_id:             str
    n_synthesis_runs_seen:      int       # times A retrieved this snippet
    n_candidates_in_those_runs: int       # candidates produced by runs that saw this snippet
    n_green:                    int       # of those, ended GREEN
    n_red:                      int
    conversion_rate_to_green:   float


@_dc.dataclass(frozen=True)
class SourceAggregate:
    source:                     str       # 'arxiv' / 'semantic_scholar' / 'unknown'
    n_candidates_cited:         int
    n_b_approved:               int
    n_principal_approved:       int
    n_strict_gate_run:          int
    n_green:                    int
    n_red:                      int
    conversion_rate_to_green:   float


@_dc.dataclass(frozen=True)
class ConfidenceCalibrationBucket:
    """One row of the calibration table — does A's predicted outcome
    tier match observed reality?"""
    a_predicted_tier:           str       # what A wrote in expected_outcome_prior
    n_candidates:               int       # candidates with this prediction
    n_reached_strict_gate:      int       # of those, reached strict gate
    n_actual_green:             int       # of those, actual GREEN
    actual_green_rate:          float     # n_actual_green / max(1, n_reached_strict_gate)


# ────────────────────────────────────────────────────────────────────
# Lazy index builders — read each store once, cache by hash
# ────────────────────────────────────────────────────────────────────
def _iter_jsonl(p: Path) -> Iterable[dict]:
    if not p.is_file():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _utc_iso_cutoff(days: int) -> str:
    return (_dt.datetime.utcnow()
            - _dt.timedelta(days=days)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_events(since: Optional[str] = None) -> list[dict]:
    out: list[dict] = []
    for ev in _iter_jsonl(_EVENTS_PATH):
        if since is not None and ev.get("ts", "") < since:
            continue
        out.append(ev)
    return out


def _load_verdicts() -> dict[str, dict]:
    """Latest verdict per hypothesis_id."""
    latest: dict[str, dict] = {}
    for v in _iter_jsonl(_VERDICTS_PATH):
        hid = v.get("hypothesis_id")
        if not hid:
            continue
        prev = latest.get(hid)
        if prev is None or v.get("review_ts", "") > prev.get("review_ts", ""):
            latest[hid] = v
    return latest


def _load_resolutions() -> dict[str, dict]:
    """Latest resolution per hypothesis_id."""
    latest: dict[str, dict] = {}
    for r in _iter_jsonl(_RESOLUTIONS_PATH):
        hid = r.get("hypothesis_id")
        if not hid:
            continue
        prev = latest.get(hid)
        if prev is None or r.get("resolved_ts", "") > prev.get("resolved_ts", ""):
            latest[hid] = r
    return latest


def _load_hypotheses() -> dict[str, "Hypothesis"]:
    """All Hypothesis records, latest version per id."""
    try:
        from engine.research_store.hypothesis.store import load_hypotheses
        hyps = load_hypotheses()
    except Exception as exc:
        logger.warning("attribution: load_hypotheses failed: %s", exc)
        return {}
    latest: dict[str, object] = {}
    for h in hyps:
        prior = latest.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            latest[h.hypothesis_id] = h
    return latest


# ────────────────────────────────────────────────────────────────────
# Single-candidate lifecycle JOIN
# ────────────────────────────────────────────────────────────────────
def _strict_gate_outcome_for(
    hypothesis_id: str,
    events: list[dict],
) -> tuple[str, str | None, int | None]:
    """Walk events for the candidate's auto_<hash> subject and report
    the strict-gate outcome. Returns (subject_id, verdict, score).

    The link is via candidate_pipeline_started (or
    candidate_skipped_pre_compute) which carry source_hypothesis_id
    in metrics. That event's subject_id is the auto_<hash> factor
    subject used by all downstream strict-gate events."""
    subject_id = ""
    # 1. Find the auto_<hash> subject for this hypothesis
    for ev in events:
        et = ev.get("event_type") or ""
        if et not in ("candidate_pipeline_started",
                       "candidate_skipped_pre_compute"):
            continue
        metrics = ev.get("metrics") or {}
        if metrics.get("source_hypothesis_id") == hypothesis_id:
            subject_id = str(ev.get("subject_id") or "")
            # candidate_skipped is terminal
            if et == "candidate_skipped_pre_compute":
                return subject_id, "SKIPPED", 0
            break

    if not subject_id:
        return "", None, None

    # 2. Find latest factor_verdict_filed for this subject
    latest_verdict: tuple[str, str, int] | None = None
    for ev in events:
        if ev.get("event_type") != "factor_verdict_filed":
            continue
        if ev.get("subject_id") != subject_id:
            continue
        verdict = str(ev.get("verdict") or "")
        score = int((ev.get("metrics") or {}).get("score", 0) or 0)
        ts = str(ev.get("ts") or "")
        if latest_verdict is None or ts > latest_verdict[0]:
            latest_verdict = (ts, verdict, score)

    if latest_verdict is None:
        return subject_id, None, None
    return subject_id, latest_verdict[1], latest_verdict[2]


def _compute_final_state(
    hyp,
    verdict: Optional[dict],
    resolution: Optional[dict],
    strict_gate_verdict: Optional[str],
) -> str:
    """Pick the most-progressed observable state for this candidate."""
    if strict_gate_verdict == "GREEN":
        return FINAL_STATE_GREEN
    if strict_gate_verdict == "MARGINAL":
        return FINAL_STATE_MARGINAL
    if strict_gate_verdict == "RED":
        return FINAL_STATE_RED
    if strict_gate_verdict == "SKIPPED":
        return FINAL_STATE_SKIPPED_PRE_COMPUTE
    # No strict gate yet — what's the latest gate?
    if resolution is not None:
        dec = resolution.get("decision")
        if dec == "approved":
            return FINAL_STATE_PRINCIPAL_APPROVED
        if dec == "rejected":
            return FINAL_STATE_PRINCIPAL_REJECTED
        if dec == "deferred":
            return FINAL_STATE_PRINCIPAL_DEFERRED
    if verdict is not None:
        vt = verdict.get("verdict_type")
        if vt == "APPROVE_FOR_PIPELINE":
            return FINAL_STATE_B_APPROVED_NO_DECISION
        if vt == "REJECT":
            return FINAL_STATE_B_REJECTED
        if vt == "DOCTRINE_AMENDMENT_NEEDED":
            return FINAL_STATE_B_AMENDMENT
    return FINAL_STATE_PRE_B_REVIEW


def get_candidate_lifecycle(hypothesis_id: str
                              ) -> Optional[CandidateLifecycle]:
    """Build the full lineage walk for one candidate. Returns None if
    no Hypothesis row exists for the given id."""
    from engine.agents.attribution.helpers import (
        get_paper_source, paper_from_watchlist_authors,
    )

    hyps = _load_hypotheses()
    h = hyps.get(hypothesis_id)
    if h is None:
        return None

    verdicts = _load_verdicts()
    resolutions = _load_resolutions()
    events = _load_events()
    verdict = verdicts.get(hypothesis_id)
    resolution = resolutions.get(hypothesis_id)

    # Find the synthesis event that produced this hypothesis (if any)
    # so we can recover doctrine_snippet_ids and A's expected outcome
    doctrine_snippet_ids: tuple[str, ...] = ()
    a_predicted = ""
    for ev in events:
        if ev.get("event_type") != "papers_curator_synthesis_run":
            continue
        m = ev.get("metrics") or {}
        # Match by candidate claim or by hypothesis_id in written_ids
        # — candidates_summary doesn't carry hypothesis_id, so we fall
        # back to matching by claim substring
        for c in (m.get("candidates_summary") or []):
            if c.get("claim", "")[:60] in (h.claim or "")[:60]:
                doctrine_snippet_ids = tuple(m.get("doctrine_snippet_ids") or ())
                a_predicted = str(c.get("expected_outcome_prior") or "")
                break
        if doctrine_snippet_ids or a_predicted:
            break

    # Cited paper sources + watchlist authors
    cited = tuple(h.synthesizes_paper_ids or ())
    source_map = {pid: get_paper_source(pid) for pid in cited}
    wl_authors_set: set[str] = set()
    for pid in cited:
        for a in paper_from_watchlist_authors(pid):
            wl_authors_set.add(a)

    # Strict gate outcome
    subj_id, sg_verdict, sg_score = _strict_gate_outcome_for(
        hypothesis_id, events,
    )

    final_state = _compute_final_state(h, verdict, resolution, sg_verdict)

    return CandidateLifecycle(
        hypothesis_id           = h.hypothesis_id,
        extraction_method       = h.extraction_method.value,
        mechanism_family        = h.mechanism_family.value,
        created_ts              = h.created_ts,
        claim                   = h.claim,
        cited_paper_ids         = cited,
        cited_paper_sources     = source_map,
        cited_watchlist_authors = tuple(sorted(wl_authors_set)),
        doctrine_snippet_ids    = doctrine_snippet_ids,
        citation_quality        = h.citation_quality,
        a_expected_outcome_prior= a_predicted,
        b_verdict_type          = (verdict or {}).get("verdict_type"),
        b_confidence            = (verdict or {}).get("confidence"),
        principal_decision      = (resolution or {}).get("decision"),
        principal_decision_ts   = str((resolution or {}).get("resolved_ts") or ""),
        strict_gate_subject_id  = subj_id,
        strict_gate_verdict     = sg_verdict,
        strict_gate_score       = sg_score,
        final_state             = final_state,
    )


def _is_at_or_past(state: str, threshold: str) -> bool:
    """Is `state` at or past `threshold` on the lifecycle ladder?
    Strict-gate states (GREEN/MARGINAL/RED/SKIPPED) all count as
    'reached strict gate'."""
    if threshold == "b_approved":
        return state not in {FINAL_STATE_PRE_B_REVIEW,
                              FINAL_STATE_B_REJECTED}
    if threshold == "principal_approved":
        return state in {
            FINAL_STATE_PRINCIPAL_APPROVED,
            FINAL_STATE_SKIPPED_PRE_COMPUTE,
            FINAL_STATE_GREEN, FINAL_STATE_MARGINAL, FINAL_STATE_RED,
        }
    if threshold == "strict_gate":
        return state in {FINAL_STATE_SKIPPED_PRE_COMPUTE,
                          FINAL_STATE_GREEN, FINAL_STATE_MARGINAL,
                          FINAL_STATE_RED}
    return False


# ────────────────────────────────────────────────────────────────────
# Iterate all candidates' lifecycles within a window
# ────────────────────────────────────────────────────────────────────
def _all_lifecycles_in_window(days: int) -> list[CandidateLifecycle]:
    """Compute lifecycles for every Hypothesis created in the last
    `days` days. Returns sorted by created_ts desc."""
    cutoff = _utc_iso_cutoff(days)
    hyps = _load_hypotheses()
    out: list[CandidateLifecycle] = []
    for hid, h in hyps.items():
        if h.created_ts < cutoff:
            continue
        lc = get_candidate_lifecycle(hid)
        if lc is not None:
            out.append(lc)
    out.sort(key=lambda lc: lc.created_ts, reverse=True)
    return out


# ────────────────────────────────────────────────────────────────────
# Aggregates
# ────────────────────────────────────────────────────────────────────
def aggregate_by_author(days: int = 180
                          ) -> tuple[AuthorAggregate, ...]:
    """Per watchlist author: candidates that cited their work + how
    many survived each downstream gate."""
    rows = _all_lifecycles_in_window(days)

    by_author: dict[str, dict] = defaultdict(lambda: {
        "n_candidates_cited":   0,
        "n_b_approved":         0,
        "n_principal_approved": 0,
        "n_strict_gate_run":    0,
        "n_green":              0,
        "n_red":                0,
    })

    for lc in rows:
        for author in lc.cited_watchlist_authors:
            d = by_author[author]
            d["n_candidates_cited"] += 1
            if _is_at_or_past(lc.final_state, "b_approved"):
                d["n_b_approved"] += 1
            if _is_at_or_past(lc.final_state, "principal_approved"):
                d["n_principal_approved"] += 1
            if _is_at_or_past(lc.final_state, "strict_gate"):
                d["n_strict_gate_run"] += 1
            if lc.final_state == FINAL_STATE_GREEN:
                d["n_green"] += 1
            elif lc.final_state == FINAL_STATE_RED:
                d["n_red"] += 1

    out: list[AuthorAggregate] = []
    for name, d in by_author.items():
        n_cited = d["n_candidates_cited"]
        out.append(AuthorAggregate(
            author_name              = name,
            n_candidates_cited       = n_cited,
            n_b_approved             = d["n_b_approved"],
            n_principal_approved     = d["n_principal_approved"],
            n_strict_gate_run        = d["n_strict_gate_run"],
            n_green                  = d["n_green"],
            n_red                    = d["n_red"],
            conversion_rate_to_green = (d["n_green"] / n_cited
                                          if n_cited else 0.0),
        ))
    out.sort(key=lambda a: a.n_candidates_cited, reverse=True)
    return tuple(out)


def aggregate_by_source(days: int = 180
                          ) -> tuple[SourceAggregate, ...]:
    """Per source ('arxiv' / 'semantic_scholar' / 'unknown'): same
    gate-by-gate counts as aggregate_by_author."""
    rows = _all_lifecycles_in_window(days)

    by_source: dict[str, dict] = defaultdict(lambda: {
        "n_candidates_cited":   0,
        "n_b_approved":         0,
        "n_principal_approved": 0,
        "n_strict_gate_run":    0,
        "n_green":              0,
        "n_red":                0,
    })

    for lc in rows:
        # A single candidate can cite multiple sources; count each
        # source it touched once per candidate
        sources_in_this = set(
            src or "unknown"
            for src in lc.cited_paper_sources.values()
        )
        for src in sources_in_this:
            d = by_source[src]
            d["n_candidates_cited"] += 1
            if _is_at_or_past(lc.final_state, "b_approved"):
                d["n_b_approved"] += 1
            if _is_at_or_past(lc.final_state, "principal_approved"):
                d["n_principal_approved"] += 1
            if _is_at_or_past(lc.final_state, "strict_gate"):
                d["n_strict_gate_run"] += 1
            if lc.final_state == FINAL_STATE_GREEN:
                d["n_green"] += 1
            elif lc.final_state == FINAL_STATE_RED:
                d["n_red"] += 1

    out: list[SourceAggregate] = []
    for src, d in by_source.items():
        n_cited = d["n_candidates_cited"]
        out.append(SourceAggregate(
            source                   = src,
            n_candidates_cited       = n_cited,
            n_b_approved             = d["n_b_approved"],
            n_principal_approved     = d["n_principal_approved"],
            n_strict_gate_run        = d["n_strict_gate_run"],
            n_green                  = d["n_green"],
            n_red                    = d["n_red"],
            conversion_rate_to_green = (d["n_green"] / n_cited
                                          if n_cited else 0.0),
        ))
    out.sort(key=lambda a: a.n_candidates_cited, reverse=True)
    return tuple(out)


def aggregate_by_doctrine_snippet(days: int = 180
                                     ) -> tuple[DoctrineSnippetAggregate, ...]:
    """Per doctrine memory_file_id: how often A retrieved it + GREEN
    rate of candidates A produced in those runs."""
    rows = _all_lifecycles_in_window(days)

    # First aggregate: for each candidate, which doctrine snippets were
    # the synthesis run that produced it?
    snippet_counts: Counter = Counter()
    snippet_candidate_outcomes: dict[str, list[str]] = defaultdict(list)
    for lc in rows:
        for sid in lc.doctrine_snippet_ids:
            snippet_counts[sid] += 1
            snippet_candidate_outcomes[sid].append(lc.final_state)

    out: list[DoctrineSnippetAggregate] = []
    for sid, n_runs in snippet_counts.items():
        outcomes = snippet_candidate_outcomes[sid]
        n_cands = len(outcomes)
        n_green = sum(1 for s in outcomes if s == FINAL_STATE_GREEN)
        n_red   = sum(1 for s in outcomes if s == FINAL_STATE_RED)
        out.append(DoctrineSnippetAggregate(
            memory_file_id            = sid,
            n_synthesis_runs_seen     = n_runs,
            n_candidates_in_those_runs= n_cands,
            n_green                   = n_green,
            n_red                     = n_red,
            conversion_rate_to_green  = (n_green / n_cands
                                           if n_cands else 0.0),
        ))
    out.sort(key=lambda a: a.n_candidates_in_those_runs, reverse=True)
    return tuple(out)


def calibration_a_confidence(days: int = 180
                                ) -> tuple[ConfidenceCalibrationBucket, ...]:
    """Per A-predicted tier: how often was it actually GREEN?

    Empty data → () (calibration requires at least one candidate that
    REACHED strict gate)."""
    rows = _all_lifecycles_in_window(days)

    by_tier: dict[str, dict] = defaultdict(lambda: {
        "n_candidates":          0,
        "n_reached_strict_gate": 0,
        "n_actual_green":        0,
    })

    for lc in rows:
        tier = lc.a_expected_outcome_prior or "(unknown)"
        d = by_tier[tier]
        d["n_candidates"] += 1
        if _is_at_or_past(lc.final_state, "strict_gate"):
            d["n_reached_strict_gate"] += 1
        if lc.final_state == FINAL_STATE_GREEN:
            d["n_actual_green"] += 1

    out: list[ConfidenceCalibrationBucket] = []
    for tier, d in by_tier.items():
        n_reached = d["n_reached_strict_gate"]
        out.append(ConfidenceCalibrationBucket(
            a_predicted_tier      = tier,
            n_candidates          = d["n_candidates"],
            n_reached_strict_gate = n_reached,
            n_actual_green        = d["n_actual_green"],
            actual_green_rate     = (d["n_actual_green"] / n_reached
                                      if n_reached else 0.0),
        ))
    out.sort(key=lambda b: b.n_candidates, reverse=True)
    return tuple(out)
